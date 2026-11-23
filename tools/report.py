#!/usr/bin/env python3
"""Render build/eval-test.json (tools/evaluate.py) as Markdown tables -> EVAL.md body.

Usage: report.py build/eval-test.json > EVAL-tables.md
"""
import json
import sys

PACKS = ("official", "published", "pro")
LABEL = {"official": "Official ro", "published": "Published (dobrerares)", "pro": "Romanian Pro"}
SETNAME = {"tatoeba": "Tatoeba test", "wiki": "Wikipedia held-out articles", "wikitalk": "Wikipedia held-out discussions",
           "cv": "Common Voice test", "english": "English Tatoeba test"}


def pc(v, digits=1):
    m, h = v
    return f"{100 * m:.{digits}f} ± {100 * h:.{digits}f}"


def diff(v, digits=1):
    m, h = v
    sign = "+" if m >= 0 else "−"
    sig = "" if abs(m) > h else " (n.s.)"
    return f"{sign}{abs(100 * m):.{digits}f} ± {100 * h:.{digits}f}{sig}"


def dget(row, key):
    return diff(row[key]) if key in row else "–"


def main():
    global PACKS
    r = json.load(open(sys.argv[1], encoding="utf-8"))
    PACKS = tuple(k for k in PACKS if k in r["packs"])
    out = []
    p = r["packs"]
    out.append("### Pack contents\n")
    out.append("| | " + " | ".join(LABEL[k] for k in PACKS) + " |")
    out.append("|---|" + "---|" * len(PACKS))
    out.append("| words | " + " | ".join(f"{p[k]['words']:,}" for k in PACKS) + " |")
    out.append("| next-word pairs | " + " | ".join(f"{p[k]['pairs']:,}" for k in PACKS) + " |")
    out.append("| ro.clex bytes | " + " | ".join(f"{p[k]['bytes'].get('clex', 0):,}" for k in PACKS) + " |")
    out.append("| ro.cngm bytes | " + " | ".join(f"{p[k]['bytes'].get('cngm', 0):,}" for k in PACKS) + " |")
    out.append("")
    if "coverage" in r:
        out.append("### Coverage: share of held-out word tokens in the dictionary\n")
        out.append("| Held-out set | tokens | " + " | ".join(LABEL[k] for k in PACKS) + " |")
        out.append("|---|---|" + "---|" * len(PACKS))
        for s, row in r["coverage"].items():
            out.append(f"| {SETNAME[s]} | {row['tokens']:,} | " + " | ".join(f"{100 * row[k]:.2f}%" for k in PACKS) + " |")
        out.append("")
    if "typo" in r:
        for noise in ("gaussian", "realistic"):
            for cfg in ("combined", "ro"):
                keys = [k for k in r["typo"] if k.startswith(f"{noise}/{cfg}/")]
                if not keys:
                    continue
                title = "Combined mode (pack + official en.clex)" if cfg == "combined" else "Romanian pack alone"
                out.append(f"### Typo correction proxy — {noise} noise — {title}\n")
                out.append("Word accuracy (% of words that end up as intended) ± 95% CI over seeds; "
                           "Δ = Romanian Pro minus baseline (paired).\n")
                out.append("| Set | typing | " + " | ".join(LABEL[k] for k in PACKS) + " | Δ vs official | Δ vs published |")
                out.append("|---|---|" + "---|" * (len(PACKS) + 2))
                for k in keys:
                    _, _, s, mode = k.split("/")
                    row = r["typo"][k]
                    mode_l = {"ro_plain": "no diacritics", "ro_diac": "with diacritics", "en": "English"}[mode]
                    out.append(f"| {SETNAME[s]} | {mode_l} | " + " | ".join(pc(row[x]["word_acc"]) for x in PACKS)
                               + f" | {dget(row, 'pro_minus_official_word_acc')} | {dget(row, 'pro_minus_published_word_acc')} |")
                out.append("")
                out.append("Wrong changes (% of all words the corrector changed into a wrong word) / clean words broken "
                           "(% of correctly typed words changed) / accent restoration (% of words typed without their "
                           "diacritics that were restored):\n")
                out.append("| Set | typing | " + " | ".join(LABEL[x] for x in PACKS) + " | Δ wrong changes vs official | vs published |")
                out.append("|---|---|" + "---|" * (len(PACKS) + 2))
                for k in keys:
                    _, _, s, mode = k.split("/")
                    row = r["typo"][k]
                    mode_l = {"ro_plain": "no diacritics", "ro_diac": "with diacritics", "en": "English"}[mode]
                    cells = []
                    for x in PACKS:
                        c = f"{pc(row[x]['wrong_change'])} / {pc(row[x]['clean_broken'])}"
                        if "accent_fixed" in row[x]:
                            c += f" / {pc(row[x]['accent_fixed'])}"
                        cells.append(c)
                    out.append(f"| {SETNAME[s]} | {mode_l} | " + " | ".join(cells)
                               + f" | {dget(row, 'pro_minus_official_wrong_change')} | {dget(row, 'pro_minus_published_wrong_change')} |")
                out.append("")
    if "next" in r:
        out.append("### Next-word prediction (each pack's own .cngm decoded offline)\n")
        out.append("Top-3 accuracy with the pack's three most frequent words filling empty slots; "
                   "\"after 1 letter\": followers starting with the typed first letter, then the commonest "
                   "dictionary words with that letter. ± 95% CI over seeds.\n")
        out.append("| Set | metric | " + " | ".join(LABEL[k] for k in PACKS) + " | Δ vs official | Δ vs published |")
        out.append("|---|---|" + "---|" * (len(PACKS) + 2))
        for s, row in r["next"].items():
            flag = " ⚠" if s == "tatoeba" else ""
            for metric, lab in (("top3_fallback", "top-3"), ("top3_after_1_letter", "top-3 after 1 letter")):
                out.append(f"| {SETNAME[s]}{flag} | {lab} | " + " | ".join(pc(row[k][metric]) for k in PACKS)
                           + f" | {dget(row, f'pro_minus_official_{metric}')} | {dget(row, f'pro_minus_published_{metric}')} |")
        out.append("")
        out.append("⚠ Both baselines' next-word models were built from *all* Tatoeba sentences, including this "
                   "test split; Romanian Pro never saw them. The Tatoeba next-word rows favour the baselines.\n")
        out.append("Share of evaluated pairs whose previous word has any follower in the model:\n")
        out.append("| Set | " + " | ".join(LABEL[k] for k in PACKS) + " |")
        out.append("|---|" + "---|" * len(PACKS))
        for s, row in r["next"].items():
            out.append(f"| {SETNAME[s]} | " + " | ".join(pc(row[k]["has_followers"]) for k in PACKS) + " |")
        out.append("")
    if "diacless" in r:
        out.append("### Words typed without diacritics (no other typo), dictionary proxy\n")
        out.append("| intended | typed | " + " | ".join(LABEL[k] for k in PACKS) + " |")
        out.append("|---|---|" + "---|" * len(PACKS))
        for row in r["diacless"]:
            cells = []
            for k in PACKS:
                v = row[k]
                cells.append(f"**{v}**" if v.split(" ")[0] == row["intended"] else v)
            out.append(f"| {row['intended']} | {row['typed']} | " + " | ".join(cells) + " |")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
