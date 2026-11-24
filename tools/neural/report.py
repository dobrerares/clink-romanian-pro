#!/usr/bin/env python3
"""Markdown tables for EVAL-phase2.md from tools/neural/evaluate_next.py JSON files.

Usage: report.py TEST.json [--dev DEV.json] [--lambda 0.6] [--int8 INT8DEV.json --fp32 FP32DEV.json]
Prints markdown to stdout.
"""
import argparse
import json

NAMES = {"tatoeba": "Tatoeba test", "wiki": "Wikipedia held-out articles", "wikitalk": "Wikipedia held-out discussions",
         "cv": "Common Voice test"}
ORDER = ("wikitalk", "tatoeba", "cv", "wiki")


def pm(v, scale=100):
    m, h = v
    return f"{m * scale:.1f} ± {h * scale:.1f}"


def dpm(v, scale=100):
    m, h = v
    s = f"{m * scale:+.1f} ± {h * scale:.1f}".replace("+-", "−").replace("-", "−")
    if abs(m) <= h:
        s += " (n.s.)"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("test")
    ap.add_argument("--dev")
    ap.add_argument("--lambda", dest="lam", default=None)
    ap.add_argument("--int8")
    ap.add_argument("--fp32")
    args = ap.parse_args()
    T = json.load(open(args.test))
    lam = args.lam or T.get("chosen_lambda")
    lk = f"{float(lam):g}"
    print(f"### Next-word prediction, {T['split']} split ({len(T['seeds'])} seeds × {T['n_next']} sentences per set)\n")
    print(f"(a) Phase 1 `.cngm`; (b) neural model alone; (c) interpolation λ = {lk} (chosen on dev). "
          "± = 95% CI over seeds; Δ are paired differences against (a).\n")
    print("| Set | metric | (a) Phase 1 .cngm | (b) neural | (c) combination | Δ (b) − (a) | Δ (c) − (a) |")
    print("|---|---|---|---|---|---|---|")
    for s in ORDER:
        if s not in T["sets"]:
            continue
        S = T["sets"][s]
        M, D = S["metrics"], S["paired_diffs"]
        for metric, label in (("top3", "top-3"), ("top3_after1", "top-3 after 1 letter")):
            a = M[f"cngm_{metric}"]
            b = M[f"nn_{metric}"]
            c = M[f"combo{lk}_{metric}"]
            print(f"| {NAMES[s]} | {label} | {pm(a)} | {pm(b)} | {pm(c)} | {dpm(D[f'nn_{metric}_minus_cngm'])} | "
                  f"{dpm(D[f'combo{lk}_{metric}_minus_cngm'])} |")
    print()
    print("| Set | evaluated pairs per seed | unique sentences | next word producible by the neural tokenizer (not hyphenated etc.) | neural top-1 |")
    print("|---|---|---|---|---|")
    for s in ORDER:
        if s in T["sets"]:
            S = T["sets"][s]
            n = S["pairs_per_seed"]
            print(f"| {NAMES[s]} | {min(n):,}-{max(n):,} | {S['unique_sentences']:,} | {pm(S['metrics']['b_neural_producible'])} | "
                  f"{pm(S['metrics']['nn_top1'])} |")
    print()
    if args.dev:
        Dv = json.load(open(args.dev))
        lams = [f"{x:g}" for x in Dv["lambdas"]]
        print(f"### λ scan on the dev split ({len(Dv['seeds'])} seeds × {Dv['n_next']} sentences per set; mean over the four sets)\n")
        print("| λ (weight of .cngm) | " + " | ".join(lams) + " |")
        print("|---|" + "---|" * len(lams))
        for kind, label in (("top3", "top-3"), ("top3_after1", "top-3 after 1 letter")):
            row = Dv["lambda_scan_mean_over_sets"][kind]
            print(f"| {label} | " + " | ".join(f"{row[x] * 100:.2f}" for x in lams) + " |")
        print()
        print("Per dev set, top-3: (a) / (b) / (c at the chosen λ)\n")
        print("| Set | (a) .cngm | (b) neural | (c) combination |")
        print("|---|---|---|---|")
        for s in ORDER:
            if s in Dv["sets"]:
                M = Dv["sets"][s]["metrics"]
                print(f"| {NAMES[s].replace('test', 'dev').replace('held-out', 'dev')} | {pm(M['cngm_top3'])} | {pm(M['nn_top3'])} | {pm(M[f'combo{lk}_top3'])} |")
        print()
    if args.int8 and args.fp32:
        A, B = json.load(open(args.fp32)), json.load(open(args.int8))
        print(f"### int8 weight quantization ({A['split']} split; simulated in PyTorch exactly as coremltools quantizes)\n")
        print("| Set | metric | fp32 (evaluated model) | int8 weights | Δ |")
        print("|---|---|---|---|---|")
        for s in ORDER:
            if s in A["sets"] and s in B["sets"]:
                for metric in ("nn_top3", f"combo{lk}_top3"):
                    x = A["sets"][s]["per_seed"][metric]
                    y = B["sets"][s]["per_seed"][metric]
                    d = [p - q for p, q in zip(y, x)]
                    import statistics, math
                    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(len(d) - 1, 2.0)
                    h = t * statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0
                    label = NAMES[s].replace("test", A["split"]).replace("held-out", A["split"])
                    mname = {"nn_top3": "(b) neural top-3", f"combo{lk}_top3": f"(c) λ={lk} top-3"}[metric]
                    print(f"| {label} | {mname} | {100 * sum(x) / len(x):.2f} | {100 * sum(y) / len(y):.2f} | "
                          f"{dpm([sum(d) / len(d), h])} |")
        print()


if __name__ == "__main__":
    main()
