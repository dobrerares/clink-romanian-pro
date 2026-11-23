#!/usr/bin/env python3
"""Tune CNGM source weights and pair pruning on the DEV split (never on test).

Usage: tune_ngrams.py WORDLIST CLEX [--n 3000]

For every configuration builds work/tune/<cfg>.cngm from the saved pair counts and reports the
dev next-word top-3 (+unigram fallback) and top-3-after-1-letter accuracy per dev set, with size.
Writes build/tune-ngrams.json.
"""
import argparse
import itertools
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK  # noqa: E402
from evaluate import Pack, load_set, next_run, sample  # noqa: E402
from ngrams import build  # noqa: E402

WEIGHTS = {
    "balanced": "wiki=0.35,wikitalk=0.2,tatoeba=0.25,cv=0.1,wikinews=0.04,wikiquote=0.03,wikivoyage=0.02,wikibooks=0.01",
    "formal": "wiki=0.55,wikitalk=0.15,tatoeba=0.15,cv=0.05,wikinews=0.04,wikiquote=0.03,wikivoyage=0.02,wikibooks=0.01",
    "chatty": "wiki=0.2,wikitalk=0.25,tatoeba=0.4,cv=0.1,wikinews=0.02,wikiquote=0.01,wikivoyage=0.01,wikibooks=0.01",
    "pooled": "POOLED",
}
SETS = ("tatoeba", "cv", "wikitalk", "wiki")


def pooled_weights():
    """Weights proportional to each source's pair total = plain count pooling."""
    out = {}
    for src in ("wiki", "wikitalk", "tatoeba", "cv", "wikinews", "wikiquote", "wikivoyage", "wikibooks"):
        meta = json.loads((WORK / "pairs" / f"{src}.json").read_text())
        out[src] = meta["pair_tokens"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wordlist")
    ap.add_argument("clex")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--weights", default=",".join(WEIGHTS))
    ap.add_argument("--top-k", default="3,8,16,32,64")
    ap.add_argument("--min-count", default="1,2,3,5")
    args = ap.parse_args()
    tune = WORK / "tune"
    tune.mkdir(parents=True, exist_ok=True)
    dev = {s: sample(load_set(s, "dev"), args.n, 1, lo=2, hi=40) for s in SETS}
    results = []
    for wname, k, m in itertools.product(args.weights.split(","), [int(x) for x in args.top_k.split(",")],
                                         [int(x) for x in args.min_count.split(",")]):
        spec = WEIGHTS[wname]
        weights = pooled_weights() if spec == "POOLED" else {a: float(b) for a, b in (kv.split("=") for kv in spec.split(","))}
        cfg = f"{wname}-k{k}-m{m}"
        prefix = tune / cfg
        info = build(args.wordlist, f"{prefix}.cngm", weights, k, m)
        shutil.copyfile(args.clex, f"{prefix}.clex")
        pack = Pack(cfg, prefix)
        row = {"cfg": cfg, "weights": wname, "top_k": k, "min_count": m, "pairs": info["pairs"], "bytes": info["bytes"]}
        for s, sents in dev.items():
            st = next_run(sents, [pack])[cfg]
            row[f"{s}.top3"] = round(st["top3_fallback"] / st["n"], 4)
            row[f"{s}.after1"] = round(st["top3_after_1_letter"] / st["n"], 4)
        row["mean.top3"] = round(sum(row[f"{s}.top3"] for s in SETS) / len(SETS), 4)
        row["mean.after1"] = round(sum(row[f"{s}.after1"] for s in SETS) / len(SETS), 4)
        results.append(row)
        print(json.dumps(row), flush=True)
        Path(f"{prefix}.clex").unlink()
        Path(f"{prefix}.cngm").unlink()
    (ROOT / "build").mkdir(exist_ok=True)
    (ROOT / "build" / "tune-ngrams.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
