#!/usr/bin/env python3
"""Train the official Clink GRULM on the Phase 2 corpus (CPU, data-parallel, resumable).

What is official (vendor/tools/train-neural.py, unmodified, imported through nlib):
  model class GRULM (embedding -> 2-layer GRU -> projection -> tied output, log-softmax over the
  NEXT token), the tokenizer (clean_word + BPE1) and therefore the training targets, the context
  limit (`--window` 12 model tokens), Adam, batch-mean NLL. The exported network is that class.
What differs from the official training loop, and why (documented in LOG-phase2.md):
  * The official loop materialises every (context, next) pair as Python lists and runs the GRU over
    the whole context for each single target (one output per sequence). On this CPU that is ~1,000
    targets/s; here every position of a <= 12-token window is scored in one pass (~6x faster), and
    4 processes x 2 threads train data-parallel (gradients all-reduced every step).
    Context lengths still never exceed the window: a sentence's first window starts at <bos>
    (targets 1..12, exactly the official contexts); later windows start every `--stride` 6 tokens
    and only score targets that have 7..12 tokens of context (the official trainer uses exactly 12
    there). No left padding, so no unmasked <eos> padding in front of contexts (the official loop
    pads with <eos>; Core ML inference never sees padding).
  * Seeded, checkpointed every few minutes (model, optimizer, schedule, data position) and
    resumable after a crash; the official trainer only saves at the end.
  * Per-source sampling weights (conversational emphasis) and Phase 1's artifact demotion as a loss
    weight (0.05) on the first token of a demoted word in the source that causes it.
  * Warm-up + cosine learning-rate decay and gradient clipping (1.0).
  * Dev loss (exact official contexts, <= 12 tokens) per held-out set, for model selection.

Usage:
  train.py --run NAME [--dim 128 --hidden 256 --layers 2] [--mix wiki=1,wikitalk=3,...]
           [--epochs 2 | --total-steps N] [--world 4 --threads 2 --batch 64] [--lr 2e-3]
  Re-running the same command resumes from work/neural/runs/NAME/last.pt.
"""
import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import EVAL_SETS, NEURAL, WORK, Tokenizer, dump, official_grulm  # noqa: E402

ENC = NEURAL / "enc"
RUNS = NEURAL / "runs"
DEMOTE_WEIGHT = 0.05


# ----------------------------------------------------------------------------- data
def windows_for(starts, W, S):
    """Window table for a source: absolute start, input length, first scored position."""
    L = np.diff(starts)                      # sentence lengths incl. bos/eos
    first_n = np.minimum(W, L - 1)
    abs0 = starts[:-1]
    # extra windows k >= 1 at s = k*S, scoring targets t in [W+1+(k-1)S, W+kS] (context W-S+1..W)
    K = np.maximum(0, -(-(L - 1 - W) // S))
    sent = np.repeat(np.arange(len(L)), K)
    k = np.arange(K.sum()) - np.repeat(np.cumsum(K) - K, K) + 1
    s = k * S
    n_in = np.minimum(W, L[sent] - 1 - s)
    starts_all = np.concatenate([abs0, abs0[sent] + s])
    n_all = np.concatenate([first_n, n_in]).astype(np.int16)
    j0 = np.concatenate([np.zeros(len(L), np.int16), np.full(len(s), W - S, np.int16)])
    keep = n_all > j0
    return starts_all[keep].astype(np.int64), n_all[keep], j0[keep]


class Data:
    def __init__(self, mix, W, S):
        self.W, self.S = W, S
        toks, dems, tabs = [], [], []
        off = 0
        self.info = {}
        for src, m in mix.items():
            z = np.load(ENC / f"{src}.npz")
            t, st, d = z["tokens"], z["starts"], z["demote"]
            a, n, j0 = windows_for(st, W, S)
            tabs.append((src, m, a + off, n, j0))
            toks.append(t)
            dems.append(d)
            off += len(t)
            self.info[src] = {"weight": m, "windows": int(len(a)), "targets": int((n - j0).sum())}
        toks.append(np.zeros(W + 1, np.uint16))   # tail guard for gathers past the end
        dems.append(np.zeros(W + 1, np.uint8))
        self.tokens = np.concatenate(toks).astype(np.int64)
        self.demote = np.concatenate(dems)
        self.tabs = tabs

    def epoch_order(self, seed, epoch):
        """Deterministic window order for an epoch of the mixture (floor(m) full passes plus a random
        fraction m - floor(m) of each source), shuffled."""
        rng = np.random.default_rng([seed, epoch])
        parts = []
        for src, m, a, n, j0 in self.tabs:
            idx = np.arange(len(a))
            full = int(math.floor(m))
            for _ in range(full):
                parts.append((a, n, j0, idx))
            frac = m - full
            if frac > 1e-9:
                pick = rng.choice(len(a), size=int(round(frac * len(a))), replace=False)
                parts.append((a, n, j0, np.sort(pick)))
        A = np.concatenate([p[0][p[3]] for p in parts])
        N = np.concatenate([p[1][p[3]] for p in parts])
        J = np.concatenate([p[2][p[3]] for p in parts])
        perm = rng.permutation(len(A))
        return A[perm], N[perm], J[perm]

    def batch(self, A, N, J):
        W = self.W
        idx = A[:, None] + np.arange(W)[None, :]
        inp = self.tokens[idx]
        tgt = self.tokens[idx + 1]
        jj = np.arange(W)[None, :]
        mask = (jj >= J[:, None]) & (jj < N[:, None])
        w = np.where(self.demote[idx + 1] == 1, DEMOTE_WEIGHT, 1.0).astype(np.float32)
        return inp, tgt, mask, w


# ----------------------------------------------------------------------------- dev
def dev_sets(tok, n_per_set, seed=12345):
    """Fixed dev samples (work/eval/<set>.dev.txt), encoded like training sentences."""
    import random
    out = {}
    for name in EVAL_SETS:
        lines = [l.rstrip("\n").split("\t", 1)[1] for l in open(WORK / "eval" / f"{name}.dev.txt", encoding="utf-8")]
        rng = random.Random(seed)
        lines = rng.sample(lines, min(n_per_set, len(lines)))
        seqs = []
        for text in lines:
            words = tok.clean(text)
            if words:
                seqs.append(tok.encode_words(words))
        out[name] = seqs
    return out


def dev_loss(model, seqs, W, eos, batch=512):
    """Mean NLL per target with EXACT official contexts: ids[max(0, t-W):t] for target ids[t]."""
    import torch
    import torch.nn.functional as F
    prefix_in, prefix_tgt, deep_ctx, deep_tgt = [], [], [], []
    for ids in seqs:
        n = min(W, len(ids) - 1)
        prefix_in.append(ids[:n] + [eos] * (W - n))
        prefix_tgt.append(ids[1:n + 1] + [-100] * (W - n))
        for t in range(W + 1, len(ids)):
            deep_ctx.append(ids[t - W:t])
            deep_tgt.append(ids[t])
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for s in range(0, len(prefix_in), batch):
            x = torch.tensor(prefix_in[s:s + batch])
            y = torch.tensor(prefix_tgt[s:s + batch])
            out, _ = model.gru(model.emb(x))
            logits = model.proj(out) @ model.emb.weight.t() + model.bias
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum")
            total += float(nll)
            count += int((y != -100).sum())
        for s in range(0, len(deep_ctx), batch):
            x = torch.tensor(deep_ctx[s:s + batch])
            y = torch.tensor(deep_tgt[s:s + batch])
            lp = model(x)                       # the official forward: last position
            total += float(F.nll_loss(lp, y, reduction="sum"))
            count += len(y)
    model.train()
    return total / max(count, 1), count


# ----------------------------------------------------------------------------- training
def lr_at(step, total, base, warmup, floor):
    if step < warmup:
        return base * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def worker(rank, cfg):
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    torch.set_num_threads(cfg["threads"])
    torch.set_num_interop_threads(1)
    world = cfg["world"]
    if world > 1:
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{cfg['port']}", rank=rank, world_size=world)
    run = Path(cfg["run_dir"])
    tok = Tokenizer(cfg["vocab"])
    W, S = cfg["window"], cfg["stride"]
    data = Data(cfg["mix"], W, S)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"] + rank)
    GRULM = official_grulm(tok.V, tok.eos, cfg["dim"], cfg["hidden"], cfg["layers"])
    model = GRULM()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    params = [p for p in model.parameters()]

    # epoch geometry (identical on every rank)
    per_epoch = []
    for e in range(cfg["epochs_ceil"]):
        A, _, _ = data.epoch_order(cfg["seed"], e)
        per_epoch.append(len(A) // (world * cfg["batch"]))
    total_steps = cfg["total_steps"] or int(sum(per_epoch[:-1]) + per_epoch[-1] * (cfg["epochs"] - (cfg["epochs_ceil"] - 1)))
    state = {"step": 0, "epoch": 0, "epoch_step": 0, "seconds": 0.0, "targets": 0}
    ckpt = run / "last.pt"
    if ckpt.exists():
        z = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(z["model"])
        opt.load_state_dict(z["opt"])
        state = z["state"]
        if rank == 0:
            print(f"[resume] {ckpt} at step {state['step']} (epoch {state['epoch']}, {state['epoch_step']})", flush=True)
    if world > 1:   # identical start on every rank
        for p in params:
            dist.broadcast(p.data, 0)
    dev = dev_sets(tok, cfg["dev_n"]) if rank == 0 else None
    if rank == 0:
        info = dict(cfg, data=data.info, steps_per_epoch=per_epoch, total_steps=total_steps,
                    parameters=sum(p.numel() for p in params), V=tok.V)
        dump(run / "config.json", info)
        print(json.dumps({k: info[k] for k in ("parameters", "V", "total_steps", "steps_per_epoch", "data")}), flush=True)
    log = open(run / "progress.jsonl", "a") if rank == 0 else None
    t_last_ckpt = time.time()
    t_start = time.time() - state["seconds"]
    acc_loss, acc_n, acc_t = 0.0, 0, time.time()
    flat_numel = sum(p.numel() for p in params)

    def save(tag=None):
        tmp = run / "last.pt.tmp"
        state["seconds"] = time.time() - t_start
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "config": cfg}, tmp)
        os.replace(tmp, ckpt)
        if tag:
            torch.save({"model": model.state_dict(), "state": dict(state), "config": cfg}, run / f"{tag}.pt")

    def evaluate():
        res = {}
        for name, seqs in dev.items():
            nll, n = dev_loss(model, seqs, W, tok.eos)
            res[name] = round(nll, 5)
        res["mean"] = round(sum(res[k] for k in EVAL_SETS) / len(EVAL_SETS), 5)
        return res

    done = False
    paused = False
    t_session = time.time()
    while not done:
        e = state["epoch"]
        A, N, J = data.epoch_order(cfg["seed"], e)
        B = cfg["batch"]
        n_steps = len(A) // (world * B)          # same on every rank
        A, N, J = A[rank::world], N[rank::world], J[rank::world]
        while state["epoch_step"] < n_steps:
            i = state["epoch_step"] * B
            inp, tgt, mask, w = data.batch(A[i:i + B], N[i:i + B], J[i:i + B])
            if cfg["time_schedule"]:   # pilots: cosine over the wall-clock budget (equal-time comparisons)
                frac = (time.time() - t_start) / (cfg["time_limit"] * 3600)
                lr = lr_at(int(frac * 10**6), 10**6, cfg["lr"], 0, cfg["lr_floor"]) * min(1.0, (state["step"] + 1) / cfg["warmup"])
            else:
                lr = lr_at(state["step"], total_steps, cfg["lr"], cfg["warmup"], cfg["lr_floor"])
            for g in opt.param_groups:
                g["lr"] = lr
            x = torch.from_numpy(inp)
            m = torch.from_numpy(mask)
            out, _ = model.gru(model.emb(x))
            sel = out[m]
            logits = model.proj(sel) @ model.emb.weight.t() + model.bias
            nll = F.cross_entropy(logits, torch.from_numpy(tgt)[m], reduction="none")
            wt = torch.from_numpy(w)[m]
            loss = (nll * wt).sum() / m.sum()
            opt.zero_grad(set_to_none=False)
            loss.backward()
            if world > 1:
                flat = torch.cat([p.grad.reshape(-1) for p in params])
                dist.all_reduce(flat)
                flat /= world
                o = 0
                for p in params:
                    k = p.numel()
                    p.grad.copy_(flat[o:o + k].view_as(p))
                    o += k
            torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            opt.step()
            state["step"] += 1
            state["epoch_step"] += 1
            ntg = int(m.sum())
            state["targets"] += ntg * world
            acc_loss += float((nll.detach()).sum())
            acc_n += ntg
            last_step = state["step"] >= total_steps
            if cfg["time_limit"] and time.time() - t_start > cfg["time_limit"] * 3600:
                last_step = True
            if rank == 0 and (state["step"] % cfg["log_every"] == 0 or last_step):
                dt = time.time() - acc_t
                rec = {"step": state["step"], "epoch": e, "lr": round(lr, 6), "train_nll": round(acc_loss / max(acc_n, 1), 4),
                       "targets": state["targets"], "targets_per_s": round(acc_n * world / dt), "hours": round((time.time() - t_start) / 3600, 3)}
                if state["step"] % cfg["dev_every"] == 0 or last_step:
                    rec["dev"] = evaluate()
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
                acc_loss, acc_n, acc_t = 0.0, 0, time.time()
            # checkpoint / stop decisions must be identical across ranks -> broadcast from rank 0
            want = 1 if (rank == 0 and time.time() - t_last_ckpt > cfg["ckpt_minutes"] * 60) else 0
            pause = 1 if (rank == 0 and cfg["session_limit"] and not last_step
                          and time.time() - t_session > cfg["session_limit"] * 3600) else 0
            if world > 1:
                tw = torch.tensor([want, int(last_step), pause])
                dist.broadcast(tw, 0)
                want, last_step, pause = int(tw[0]), bool(tw[1]), int(tw[2])
            if (want or pause) and rank == 0:
                save()
                t_last_ckpt = time.time()
            if last_step:
                done = True
                break
            if pause:
                paused = True
                break
        if paused:
            break
        if not done:
            state["epoch"] += 1
            state["epoch_step"] = 0
            if rank == 0:
                save(f"epoch{e + 1}")
                t_last_ckpt = time.time()
            if state["epoch"] >= cfg["epochs_ceil"] and not cfg["total_steps"]:
                done = True
    if paused:
        if rank == 0:
            print(f"[pause] session limit reached at step {state['step']}; rerun the same command to resume", flush=True)
            log.close()
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    if rank == 0:
        save("final")
        final = {"step": state["step"], "targets": state["targets"], "hours": round((time.time() - t_start) / 3600, 3), "dev": evaluate()}
        dump(run / "final.json", final)
        print("[final]", json.dumps(final), flush=True)
        log.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--vocab", default=str(NEURAL / "ro.bpevocab"))
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--mix", default="wiki=1,wikitalk=3,tatoeba=20,cv=5,wikinews=3,wikiquote=3,wikivoyage=3,wikibooks=3")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=0, help="override: schedule length and stop")
    ap.add_argument("--time-limit", type=float, default=0, help="hours; stop (and save) when exceeded")
    ap.add_argument("--time-schedule", action="store_true", help="cosine LR over --time-limit (pilots)")
    ap.add_argument("--session-limit", type=float, default=0,
                    help="hours for THIS invocation; then checkpoint and exit (rerun to resume)")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64, help="windows per process per step")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lr-floor", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--dev-n", type=int, default=1500)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--dev-every", type=int, default=2000)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    args = ap.parse_args()
    mix = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
    run = RUNS / args.run
    run.mkdir(parents=True, exist_ok=True)
    cfg = {"run": args.run, "run_dir": str(run), "vocab": args.vocab, "dim": args.dim, "hidden": args.hidden,
           "layers": args.layers, "window": args.window, "stride": args.stride, "mix": mix,
           "epochs": args.epochs, "epochs_ceil": max(1, math.ceil(args.epochs)), "total_steps": args.total_steps,
           "time_limit": args.time_limit, "time_schedule": args.time_schedule, "world": args.world, "threads": args.threads, "batch": args.batch,
           "lr": args.lr, "warmup": args.warmup, "lr_floor": args.lr_floor, "clip": args.clip, "seed": args.seed,
           "dev_n": args.dev_n, "log_every": args.log_every, "dev_every": args.dev_every,
           "ckpt_minutes": args.ckpt_minutes, "session_limit": args.session_limit, "port": free_port()}
    if args.world > 1:
        import torch.multiprocessing as tmp
        tmp.spawn(worker, args=(cfg,), nprocs=args.world, join=True)
    else:
        worker(0, cfg)


if __name__ == "__main__":
    main()
