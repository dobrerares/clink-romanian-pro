#!/usr/bin/env python3
"""Next-word evaluation, Phase 2: Phase 1 .cngm vs neural model vs their combination.

Protocol = Phase 1 (tools/evaluate.py `next`), reused literally: the same held-out files
(work/eval/<set>.<split>.txt), the same sentence sampler (`sample(..., lo=2, hi=40)`, 2,000 sentences
per seed, bootstrap resample when the set is smaller), the same seeds, the same pair extraction
(official build-next-word tokeniser, pairs never cross . ! ? … ¦), the same metrics:
  top3         is the next word among the three suggestions
  top3_after1  the next word's first letter is known; is it among three suggestions with that letter
and the same intervals (mean ± t(0.975, seeds-1)·sd/√seeds over seeds; differences paired).

Systems
  (a) cngm     Phase 1 Lexicons/ro.cngm exactly as Phase 1 scored it (followers of the previous word,
               empty slots filled with the pack's three most frequent words / by first letter).
  (b) neural   the PyTorch model (same weights as neural/ro.mlpackage). Context = the words of the
               sentence segment before the next word, cleaned by the official clean_word (so
               hyphenated words and numbers are dropped, as in training), [<bos>] + BPE, last 12
               tokens. Whole-word decoding: exact best-first search over BPE continuations (a word ends
               with a "</w>" token; <eos>/<unk>/<bos> excluded) keeping every word whose path
               probability can still reach the top 5, at most 16 partial words per step, at most 6
               tokens per word; continuation tokens extend the GRU state of the context. Probabilities
               of identical words reached by different token paths are summed. "after 1 letter":
               the first token must start with that letter.
  (c) combo    candidates = top-20 .cngm followers ∪ the neural model's words ∪ the pack's top
               unigrams; score = λ·P_cngm(w|prev) + (1-λ)·P_neural(w|context), P_neural of a candidate
               = its canonical BPE path probability (teacher-forced; 0 for words the neural tokenizer
               cannot produce, e.g. hyphenated ones). λ is chosen on DEV.

Usage: evaluate_next.py CHECKPOINT --split dev|test [--seeds 1,2,3,4,5] [--n-next 2000]
                        [--lambdas 0,0.1,...] [--out build/neural-eval-test.json]
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import EVAL_SETS, ROOT, dump  # noqa: E402
from export import load_model  # noqa: E402
import evaluate as P1  # noqa: E402  (Phase 1 tools/evaluate.py)
from ngrams import SENT_END, SPLIT_RE, word  # noqa: E402
from corpus import QUOTES  # noqa: E402

K_SEARCH, B_SEARCH, D_SEARCH = 5, 16, 6
N_FOLLOW = 20
NEG = -1e30


# ----------------------------------------------------------------------------------- pairs
def sentence_pairs(text, tok):
    """Phase 1 next_run() iteration; plus the neural context words for each pair."""
    out = []
    for seg in SENT_END.split(text):
        toks = [word(t) for t in SPLIT_RE.findall(seg)]
        for i, (a, b) in enumerate(zip(toks, toks[1:])):
            if a is None or b is None:
                continue
            ctx = []
            for t in toks[:i + 1]:
                if t is None:
                    continue
                c = tok.off.clean_word(t.translate(QUOTES), "ro")
                if c:
                    ctx.append(c)
            out.append((a, b, ctx))
    return out


# ----------------------------------------------------------------------------------- neural
class Neural:
    def __init__(self, model, tok, window):
        import torch
        self.torch = torch
        self.m, self.tok, self.W = model, tok, window
        V = tok.V
        special = np.zeros(V, bool)
        for t in (tok.unk, tok.bos, tok.eos):
            special[t] = True
        WE = tok.off.WORD_END
        end = np.array([v.endswith(WE) and v != WE for v in tok.vocab]) & ~special
        part = np.array([not v.endswith(WE) for v in tok.vocab]) & ~special
        self.end = torch.from_numpy(end)
        self.part = torch.from_numpy(part)
        self.first = [v[0] if not special[i] else "" for i, v in enumerate(tok.vocab)]
        self.letter_mask = {}
        self.strs = [v[:-len(WE)] if v.endswith(WE) else v for v in tok.vocab]
        self.E = model.emb.weight.detach()

    def lmask(self, L):
        m = self.letter_mask.get(L)
        if m is None:
            m = self.torch.tensor([f == L for f in self.first])
            self.letter_mask[L] = m
        return m

    def logits(self, top):
        return self.m.proj(top) @ self.E.t() + self.m.bias

    def contexts(self, ctx_ids):
        """Exact (unpadded) forward of each context; returns states [layers, N, H] and lp0 [N, V]."""
        torch = self.torch
        N = len(ctx_ids)
        L = self.m.gru.num_layers
        H = self.m.gru.hidden_size
        h = torch.zeros(L, N, H)
        lp = torch.zeros(N, self.tok.V)
        by_len = defaultdict(list)
        for i, c in enumerate(ctx_ids):
            by_len[len(c)].append(i)
        for n, idx in by_len.items():
            x = torch.tensor([ctx_ids[i] for i in idx])
            _, hh = self.m.gru(self.m.emb(x))
            h[:, idx] = hh
            lp[idx] = torch.log_softmax(self.logits(hh[-1]), -1)
        return h, lp

    def step(self, h, tokens):
        torch = self.torch
        x = torch.tensor(tokens)[:, None]
        _, hh = self.m.gru(self.m.emb(x), h)
        return hh, torch.log_softmax(self.logits(hh[-1]), -1)

    def search(self, h, lp0, letters):
        """Best-first whole-word search. letters[i]: required first letter or None.
        Returns list of dicts word -> log P (path probabilities summed)."""
        torch = self.torch
        N = lp0.shape[0]
        res = [dict() for _ in range(N)]

        def add(i, w, s):
            d = res[i]
            if w in d:
                d[w] = float(np.logaddexp(d[w], s))
            else:
                d[w] = s

        def thr(i):
            d = res[i]
            if len(d) < K_SEARCH:
                return NEG
            return sorted(d.values(), reverse=True)[K_SEARCH - 1]

        endm = self.end.unsqueeze(0).expand(N, -1).clone()
        partm = self.part.unsqueeze(0).expand(N, -1).clone()
        for i, L in enumerate(letters):
            if L is not None:
                lm = self.lmask(L)
                endm[i] &= lm
                partm[i] &= lm
        ev, ei = lp0.masked_fill(~endm, NEG).topk(K_SEARCH, -1)
        for i in range(N):
            for v, t in zip(ev[i].tolist(), ei[i].tolist()):
                if v > NEG / 2:
                    add(i, self.strs[t], v)
        T = [thr(i) for i in range(N)]
        pv, pi = lp0.masked_fill(~partm, NEG).topk(B_SEARCH, -1)
        hyps = []   # (context index, prefix string, score, state index into Hs, last token)
        for i in range(N):
            for v, t in zip(pv[i].tolist(), pi[i].tolist()):
                if v > T[i] and v > NEG / 2:
                    hyps.append((i, self.strs[t], v, i, t))
        states = h
        for depth in range(D_SEARCH - 1):
            if not hyps:
                break
            hs = states[:, [hp[3] for hp in hyps]]
            newh, lp = self.step(hs, [hp[4] for hp in hyps])
            ev, ei = lp.masked_fill(~self.end, NEG).topk(K_SEARCH, -1)
            pv, pi = lp.masked_fill(~self.part, NEG).topk(B_SEARCH, -1)
            ev, ei, pv, pi = ev.tolist(), ei.tolist(), pv.tolist(), pi.tolist()
            for m, (i, pre, s, _, _) in enumerate(hyps):
                for v, t in zip(ev[m], ei[m]):
                    add(i, pre + self.strs[t], s + v)
            T = [thr(i) for i in range(N)]
            nxt = defaultdict(list)
            for m, (i, pre, s, _, _) in enumerate(hyps):
                for v, t in zip(pv[m], pi[m]):
                    if s + v > T[i]:
                        nxt[i].append((s + v, pre + self.strs[t], m, t))
            hyps = []
            for i, lst in nxt.items():
                lst.sort(reverse=True)
                for s, pre, m, t in lst[:B_SEARCH]:
                    hyps.append((i, pre, s, m, t))
            states = newh
        return res

    def score_words(self, h, lp0, items):
        """items: list of (context index, token ids). Exact canonical-path log P for each."""
        torch = self.torch
        out = [0.0] * len(items)
        groups = defaultdict(list)
        for j, (i, ids) in enumerate(items):
            if len(ids) == 1:
                out[j] = float(lp0[i, ids[0]])
            else:
                groups[len(ids)].append(j)
        for k, js in groups.items():
            for s in range(0, len(js), 512):
                chunk = js[s:s + 512]
                ci = [items[j][0] for j in chunk]
                X = torch.tensor([items[j][1] for j in chunk])
                o, _ = self.m.gru(self.m.emb(X[:, :-1]), h[:, ci])
                lp = torch.log_softmax(self.logits(o), -1)                  # [M, k-1, V]
                g = lp.gather(2, X[:, 1:, None]).squeeze(-1).sum(-1)       # [M]
                first = lp0[ci, X[:, 0]]
                for jj, v in zip(chunk, (g + first).tolist()):
                    out[jj] = v
        return out


# ----------------------------------------------------------------------------------- per pair
def evaluate_pairs(pairs, pack, nn, lambdas, batch=768):
    """pairs: list of (a, b, ctx_words). Returns list of per-pair result dicts (0/1 hits)."""
    torch = nn.torch
    tok = nn.tok
    results = []
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        ctx_ids = [tok.encode_words(c, bos=True, eos=False)[-nn.W:] for _, _, c in chunk]
        with torch.no_grad():
            h, lp0 = nn.contexts(ctx_ids)
            free = nn.search(h, lp0, [None] * len(chunk))
            lett = nn.search(h, lp0, [b[0] for _, b, _ in chunk])
            # candidates for the combination and the words whose neural probability is needed
            cands = []
            need = []
            for i, (a, b, _) in enumerate(chunk):
                fol = pack.cngm.followers.get(a, []) if pack.cngm else []
                L = b[0]
                c_free = {}
                for w, l10 in fol[:N_FOLLOW]:
                    c_free[w] = 10 ** l10
                c_let = {}
                for w, l10 in [x for x in fol if x[0][0] == L][:N_FOLLOW]:
                    c_let[w] = 10 ** l10
                for w in pack.uni_top:
                    c_free.setdefault(w, 0.0)
                for w in pack.by_initial.get(L, []):
                    c_let.setdefault(w, 0.0)
                for w in free[i]:
                    c_free.setdefault(w, 0.0)
                for w in lett[i]:
                    c_let.setdefault(w, 0.0)
                cands.append((c_free, c_let))
                for w in set(c_free) | set(c_let):
                    if w in free[i] or w in lett[i]:
                        continue
                    cw = tok.off.clean_word(w, "ro")
                    if cw is None or cw != w:
                        continue
                    ids = tok.word_ids(w)
                    if tok.unk in ids:
                        continue
                    need.append((i, w, ids))
            scored = nn.score_words(h, lp0, [(i, ids) for i, _, ids in need]) if need else []
        known = [dict() for _ in chunk]
        for (i, w, _), v in zip(need, scored):
            known[i][w] = v
        for i, (a, b, _) in enumerate(chunk):
            r = {}
            # (a) Phase 1 .cngm, Phase 1 code path
            fol = pack.cngm.followers.get(a, []) if pack.cngm else []
            top = [w for w, _ in fol[:3]]
            fb = (top + [u for u in pack.uni_top if u not in top])[:3]
            r["cngm_top3"] = int(b in fb)
            c1 = [w for w, _ in fol if w[0] == b[0]][:3]
            c1 += [u for u in pack.by_initial.get(b[0], []) if u not in c1]
            r["cngm_top3_after1"] = int(b in c1[:3])
            # (b) neural alone
            nf = sorted(free[i].items(), key=lambda kv: -kv[1])[:3]
            r["nn_top3"] = int(b in [w for w, _ in nf])
            nl = [w for w, _ in sorted(lett[i].items(), key=lambda kv: -kv[1])[:3]]
            nl += [u for u in pack.by_initial.get(b[0], []) if u not in nl]
            r["nn_top3_after1"] = int(b in nl[:3])
            r["nn_top1"] = int(bool(nf) and nf[0][0] == b)
            # (c) interpolation
            def pnn(w):
                if w in free[i]:
                    return math.exp(free[i][w])
                if w in lett[i]:
                    return math.exp(lett[i][w])
                v = known[i].get(w)
                return math.exp(v) if v is not None else 0.0
            c_free, c_let = cands[i]
            pf = {w: (png, pnn(w)) for w, png in c_free.items()}
            pl = {w: (png, pnn(w)) for w, png in c_let.items()}
            for lam in lambdas:
                key = f"{lam:g}"
                rank = sorted(pf, key=lambda w: -(lam * pf[w][0] + (1 - lam) * pf[w][1] + 1e-9 * pf[w][1]))[:3]
                r[f"combo{key}_top3"] = int(b in rank)
                rank1 = sorted(pl, key=lambda w: -(lam * pl[w][0] + (1 - lam) * pl[w][1] + 1e-9 * pl[w][1]))[:3]
                rank1 += [u for u in pack.by_initial.get(b[0], []) if u not in rank1]
                r[f"combo{key}_top3_after1"] = int(b in rank1[:3])
            r["b_neural_producible"] = int(tok.off.clean_word(b, "ro") == b)
            results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--n-next", type=int, default=2000)
    ap.add_argument("--sets", default=",".join(EVAL_SETS))
    ap.add_argument("--pack", default=str(ROOT / "Lexicons" / "ro"))
    ap.add_argument("--lambdas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--int8sim", action="store_true", help="evaluate with the int8 weight quantization simulated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    t0 = time.time()
    model, tok, cfg, z = load_model(args.checkpoint)
    if args.int8sim:
        from export import int8_simulated
        model = int8_simulated(model)
    nn = Neural(model, tok, cfg["window"])
    pack = P1.Pack("pro", args.pack)
    lambdas = [float(x) for x in args.lambdas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    out = {"split": args.split, "seeds": seeds, "n_next": args.n_next, "checkpoint": str(args.checkpoint),
           "int8sim": bool(args.int8sim),
           "checkpoint_state": z.get("state"), "pack": args.pack, "lambdas": lambdas,
           "search": {"K": K_SEARCH, "B": B_SEARCH, "max_tokens": D_SEARCH, "followers_for_combo": N_FOLLOW}, "sets": {}}
    for name in args.sets.split(","):
        src = P1.load_set(name, args.split)
        samples = {seed: P1.sample(src, args.n_next, seed, lo=2, hi=40) for seed in seeds}
        uniq = sorted({s for v in samples.values() for s in v})
        pairs, owner = [], []
        for k, text in enumerate(uniq):
            for p in sentence_pairs(text, tok):
                pairs.append(p)
                owner.append(k)
        t1 = time.time()
        res = evaluate_pairs(pairs, pack, nn, lambdas)
        per_sent = defaultdict(list)
        for k, r in zip(owner, res):
            per_sent[k].append(r)
        index = {t: k for k, t in enumerate(uniq)}
        metrics = list(res[0].keys())
        per_seed = defaultdict(list)
        npairs = []
        for seed in seeds:
            tot = defaultdict(int)
            n = 0
            for text in samples[seed]:
                for r in per_sent[index[text]]:
                    n += 1
                    for m in metrics:
                        tot[m] += r[m]
            npairs.append(n)
            for m in metrics:
                per_seed[m].append(tot[m] / max(n, 1))
        summary = {m: [round(x, 5) for x in P1.ci(v)] for m, v in per_seed.items()}
        diffs = {}
        for m in metrics:
            if m.startswith(("nn_", "combo")) and m.endswith(("top3", "top3_after1")):
                base = "cngm_top3_after1" if m.endswith("after1") else "cngm_top3"
                d = [x - y for x, y in zip(per_seed[m], per_seed[base])]
                diffs[f"{m}_minus_cngm"] = [round(x, 5) for x in P1.ci(d)]
        out["sets"][name] = {"unique_sentences": len(uniq), "unique_pairs": len(pairs), "pairs_per_seed": npairs,
                             "metrics": summary, "paired_diffs": diffs, "per_seed": {m: v for m, v in per_seed.items()},
                             "seconds": round(time.time() - t1, 1)}
        show = {k: summary[k] for k in ("cngm_top3", "nn_top3", "cngm_top3_after1", "nn_top3_after1")}
        print(name, len(uniq), "sentences", len(pairs), "pairs", f"{time.time() - t1:.0f}s", json.dumps(show), flush=True)
    # best lambda on this split (mean over sets of top3 and of top3_after1)
    best = {}
    for kind in ("top3", "top3_after1"):
        scores = {}
        for lam in lambdas:
            key = f"combo{lam:g}_{kind}"
            scores[f"{lam:g}"] = round(sum(out["sets"][s]["metrics"][key][0] for s in out["sets"]) / len(out["sets"]), 5)
        best[kind] = scores
    out["lambda_scan_mean_over_sets"] = best
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(best), flush=True)
    if args.out:
        dump(args.out, out)


if __name__ == "__main__":
    main()
