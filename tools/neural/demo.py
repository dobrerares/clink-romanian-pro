#!/usr/bin/env python3
"""Top-3 next-word suggestions for a few chat-style prefixes: (a) .cngm, (b) neural, (c) combination.

Usage: demo.py CHECKPOINT [--lambda 0.6] ["prefix one" "prefix two" ...]
Prints a markdown table (illustration only; the numbers are in EVAL-phase2.md).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import ROOT  # noqa: E402
from export import load_model  # noqa: E402
from evaluate_next import Neural, evaluate_pairs  # noqa: E402
import evaluate as P1  # noqa: E402

PREFIXES = ["nu știu", "mulțumesc", "ce faci", "unde ești", "te sun", "ne vedem", "la ce oră", "mi-e dor de",
            "vreau să", "am ajuns", "poți să", "sunt în", "o zi", "îmi pare rău", "nu mai", "hai să",
            "la mulți", "ce mai", "nu-mi", "aștept"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5)
    ap.add_argument("prefixes", nargs="*")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    model, tok, cfg, _ = load_model(args.checkpoint)
    nn = Neural(model, tok, cfg["window"])
    pack = P1.Pack("pro", ROOT / "Lexicons" / "ro")
    print(f"| prefix | (a) .cngm | (b) neural | (c) λ={args.lam:g} |")
    print("|---|---|---|---|")
    for pre in args.prefixes or PREFIXES:
        words = pre.lower().split()
        ctx = [w for w in (tok.off.clean_word(x, "ro") for x in words) if w]
        a = words[-1]
        ids = tok.encode_words(ctx, bos=True, eos=False)[-nn.W:]
        with torch.no_grad():
            h, lp0 = nn.contexts([ids])
            free = nn.search(h, lp0, [None])[0]
        fol = pack.cngm.followers.get(a, [])
        top_a = ([w for w, _ in fol[:3]] + [u for u in pack.uni_top if u not in [w for w, _ in fol[:3]]])[:3]
        top_b = [w for w, _ in sorted(free.items(), key=lambda kv: -kv[1])[:3]]
        # combination: reuse the evaluator's ranking with a dummy target, read back the ranking
        cand = {}
        import math
        for w, l10 in fol[:20]:
            cand[w] = [10 ** l10, 0.0]
        for w, lp in free.items():
            cand.setdefault(w, [0.0, 0.0])[1] = math.exp(lp)
        need = [(w, tok.word_ids(w)) for w in cand if w not in free and tok.off.clean_word(w, "ro") == w
                and tok.unk not in tok.word_ids(w)]
        with torch.no_grad():
            sc = nn.score_words(h, lp0, [(0, ids_) for _, ids_ in need]) if need else []
        for (w, _), v in zip(need, sc):
            cand[w][1] = math.exp(v)
        lam = args.lam
        top_c = sorted(cand, key=lambda w: -(lam * cand[w][0] + (1 - lam) * cand[w][1]))[:3]
        print(f"| {pre} | {', '.join(top_a)} | {', '.join(top_b)} | {', '.join(top_c)} |")


if __name__ == "__main__":
    main()
