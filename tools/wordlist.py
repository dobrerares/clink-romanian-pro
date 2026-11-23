#!/usr/bin/env python3
"""Merge per-source word counts into the Romanian Pro word list (source/ro.txt).

Usage: wordlist.py [--size N] [--curve] [--out source/ro.txt]

1. Weighted merge. Each source's counts become relative frequencies; the merged score of a word
   is sum_s WEIGHT[s] * count_s(word) / tokens_s. Weights (documented in README/LOG):
     freqwords (OpenSubtitles 2018, spoken/informal) 0.45   wiki (Wikipedia articles, formal) 0.27
     wikitalk (Wikipedia discussions, informal written) 0.10   tatoeba (conversational) 0.08
     cv (Common Voice, short read sentences) 0.03   wikiother (Wikinews/-quote/-voyage/-books) 0.07
2. Evidence filter (junk / single-occurrence noise / OCR errors), see keep_reason().
3. Diacritic cleaning: the rules and constants of autocorrect-engine/tools/clean_ro_pack.py
   (ASCII spellings dominated by their diacritic siblings are removed and their score moved to the
   siblings; valid ASCII words are re-weighted by their share in diacritic-written text; sa, ca,
   fata, pana, ma are always kept). Evidence: all training sentences written with diacritics.
4. Rank by score and cut at --size (chosen from the coverage curve, see --curve).
Writes source/ro.txt ("word<TAB>count", count-descending, counts = score scaled to 1e10) and
build/wordlist-report.json, build/wordlist-removed.tsv, build/wordlist-changed.tsv.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK, fold, hunspell_valid, tokens  # noqa: E402
from packio import Clex  # noqa: E402

COUNTS = WORK / "counts"
BUILD = ROOT / "build"
UPSTREAM_EN = Path("/home/rdobre/projects/clink/upstream/clink-language-packs/Lexicons/en.clex")

GROUPS = {  # source file -> (group, weight)
    "freqwords": ("subs", 0.45),
    "wiki": ("wiki", 0.27),
    "wikitalk": ("talk", 0.10),
    "tatoeba": ("conv", 0.08),
    "cv": ("conv", 0.03),
    "wikinews": ("other", 0.07 * 0.35),
    "wikiquote": ("other", 0.07 * 0.25),
    "wikivoyage": ("other", 0.07 * 0.20),
    "wikibooks": ("other", 0.07 * 0.20),
}
# clean_ro_pack.py constants (unchanged)
KEEP = {"sa", "ca", "fata", "pana", "ma"}
MIN_SHARE = 0.01
MIN_EVIDENCE = 40
MIN_EVIDENCE_SHARE = 20
DOMINATE = 1.0

# Leftover wiki/markup/web vocabulary (never a typed Romanian word in this sense).
MARKUP = {"thumb", "px", "nbsp", "jpg", "jpeg", "png", "svg", "gif", "pdf", "upright", "colspan", "rowspan",
          "bgcolor", "wikitable", "reflist", "infobox", "isbn", "doi", "issn", "html", "htm", "http", "https",
          "www", "php", "asp", "aspx", "utc", "eest", "cest", "ndash", "mdash", "quot", "amp", "frac", "sqrt",
          "displaystyle", "mathrm", "mathbf", "textstyle", "cdot", "alpha", "beta", "langle", "rangle", "lt",
          "gt", "rr", "xx", "redirect", "redirecționează", "defaultsort", "sortkey", "ifeq", "ifexist",
          "pmid", "arxiv", "bibcode", "jstor", "oclc", "lccn", "wayback", "archiveurl", "accessdate", "dts"}
# English closed-class words: not loanwords. Dropped unless Hunspell ro_RO accepts them (a, are, do, in...).
EN_FUNCTION = {"the", "and", "of", "to", "you", "is", "was", "were", "it", "its", "that", "this", "these",
               "those", "with", "for", "on", "my", "your", "me", "we", "they", "he", "she", "him", "her", "his",
               "our", "their", "them", "what", "who", "how", "why", "when", "where", "which", "be", "been",
               "being", "have", "has", "had", "does", "did", "not", "an", "or", "at", "by", "from", "but",
               "if", "than", "then", "there", "here", "will", "would", "can", "could", "should", "shall",
               "may", "might", "must", "just", "all", "any", "some", "no", "yes", "up", "out", "about",
               "into", "over", "after", "before", "because", "so", "too", "very", "i", "am", "us", "off",
               "of", "s", "t", "don", "doesn", "didn", "isn", "aren", "wasn", "won", "can't", "let", "get",
               "got", "go", "going", "know", "see", "come", "want", "like", "one", "two", "three"}
ROMAN = re.compile(r"m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})(-lea|-a)?")
SINGLE_OK = {"a", "o", "e", "i", "î"}
TRIPLE = re.compile(r"([a-zăâîșț])\1\1")


def load_counts():
    per = {}
    totals = {}
    for src in GROUPS:
        path = COUNTS / f"{src}.tsv"
        if not path.exists():
            print(f"warning: {path} missing; weight redistributed", file=sys.stderr)
            continue
        d = {}
        tot = 0
        with open(path, encoding="utf-8") as fh:
            next(fh)
            for line in fh:
                w, c, docs, mid, cap, diac = line.rstrip("\n").split("\t")
                c = int(c)
                d[w] = (c, int(docs), int(mid), int(cap), int(diac))
                tot += c
        per[src] = d
        totals[src] = tot
    wsum = sum(GROUPS[s][1] for s in per)
    weights = {s: GROUPS[s][1] / wsum for s in per}
    return per, totals, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=150000)
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "source" / "ro.txt"))
    args = ap.parse_args()
    BUILD.mkdir(exist_ok=True)
    per, totals, weights = load_counts()
    score = defaultdict(float)
    raw = Counter()
    groups = defaultdict(set)
    docs = Counter()
    mid = Counter()
    cap = Counter()
    diac_ev = Counter()
    for src, d in per.items():
        g = GROUPS[src][0]
        wgt = weights[src] / totals[src]
        for w, (c, dc, m, cp, dv) in d.items():
            score[w] += wgt * c
            raw[w] += c
            groups[w].add(g)
            docs[w] += dc
            mid[w] += m
            cap[w] += cp
            diac_ev[w] += dv
    print(f"{len(score):,} word types in {len(per)} sources; weights {json.dumps({k: round(v, 4) for k, v in weights.items()})}",
          flush=True)

    # Hunspell validity (build-time only) for every word with at least 3 occurrences.
    cand = [w for w in score if raw[w] >= 3]
    hv_raw = hunspell_valid(cand + [w.capitalize() for w in cand])
    hv = {w for w in cand if w in hv_raw or w.capitalize() in hv_raw}
    print(f"{len(cand):,} candidates with >= 3 occurrences; Hunspell accepts {len(hv):,}", flush=True)
    en_words = set(Clex(UPSTREAM_EN).words)

    reasons = Counter()

    def keep_reason(w):
        if raw[w] < 3:
            return None, "fewer than 3 occurrences"
        if len(w) > 30:
            return None, "longer than 30 letters"
        if w in MARKUP:
            return None, "markup/web token"
        if len(w) == 1 and w not in SINGLE_OK:
            return None, "single letter"
        if w.startswith("-") or w.endswith("-") or "--" in w:
            return None, "stray hyphen"
        if TRIPLE.search(w) and "iii" not in w:
            return None, "letter repeated 3+ times (elongation/noise)"
        valid = w in hv
        if not valid and ROMAN.fullmatch(w):
            return None, "Roman numeral"
        if not valid and w in EN_FUNCTION:
            return None, "English function word"
        g = groups[w]
        subs = per.get("freqwords", {}).get(w, (0,))[0]
        text_docs = docs[w]
        if valid:
            # Hunspell-valid: 3 occurrences anywhere, but a Wikipedia-only word must occur in 2+ articles.
            if g <= {"wiki", "other"} and text_docs < 2:
                return None, "valid but only in one Wikimedia article"
            return "hunspell-valid", None
        # Not in Hunspell: names, loanwords, slang, inflections Hunspell lacks - or noise.
        if len(g) >= 2 and raw[w] >= 5 and (text_docs >= 2 or subs >= 3):
            return "2+ source groups", None
        if subs >= 50:
            return "frequent in subtitles", None
        if text_docs >= 10:
            return "10+ documents", None
        return None, "not in Hunspell and too little independent evidence"

    kept = {}
    dropped = []
    why_kept = Counter()
    for w in score:
        k, r = keep_reason(w)
        if k:
            kept[w] = score[w]
            why_kept[k] += 1
        else:
            reasons[r] += 1
            if raw[w] >= 50:
                dropped.append((w, raw[w], r))
    print(f"evidence filter kept {len(kept):,}; reasons kept {dict(why_kept)}; dropped {dict(reasons)}", flush=True)

    # ---- diacritic cleaning (clean_ro_pack.py rules) ----
    counts = dict(kept)
    orig = dict(kept)
    classes = defaultdict(list)
    for w in counts:
        classes[fold(w)].append(w)
    ascii_with = [f for f, ws in classes.items() if f in counts and len(ws) > 1 and len(f) >= 2]
    removed, changed = [], defaultdict(lambda: [0.0, []])
    stats = Counter()

    def move(f, amount, sibs):
        to = [s for s in sibs if s in hv] or sibs
        tot = sum(orig[s] for s in to)
        for s in to:
            add = amount * orig[s] / tot
            counts[s] += add
            changed[s][0] += add
            changed[s][1].append(f)
        return to

    for f in sorted(ascii_with):
        sibs = [w for w in classes[f] if w != f]
        cf = diac_ev[f]
        cd = sum(diac_ev[s] for s in sibs)
        nev = cf + cd
        share = cf / nev if nev else None
        total = counts[f]
        ok = f in hv
        dominates = sum(orig[s] for s in sibs) >= DOMINATE * orig[f]
        if not ok and not dominates:
            stats["kept_invalid_siblings_rarer"] += 1
        elif not ok:
            to = move(f, total, sibs)
            del counts[f]
            removed.append((f, raw[f], "hunspell rejects; diacritic siblings dominate", to, cf, cd))
            stats["removed_invalid"] += 1
        elif f not in KEEP and dominates and nev >= MIN_EVIDENCE and share < MIN_SHARE:
            to = move(f, total, sibs)
            del counts[f]
            removed.append((f, raw[f], f"valid but {100 * share:.2f}% of {nev} diacritic-text uses", to, cf, cd))
            stats["removed_dominated"] += 1
        elif nev >= MIN_EVIDENCE_SHARE:
            sh = (cf + 0.5) / (nev + 1.0)
            cls_total = total + sum(counts[s] for s in sibs)
            new = cls_total * sh
            if new < total:
                move(f, total - new, sibs)
                counts[f] = new
                changed[f][0] -= total - new
                changed[f][1].append(f"kept: {100 * sh:.1f}% of {nev} diacritic-text uses")
                stats["kept_reduced"] += 1
            else:
                stats["kept_unchanged"] += 1
        else:
            stats["kept_little_evidence"] += 1

    # Extension (not in clean_ro_pack.py): spellings with *some* diacritics that Hunspell rejects
    # (făra, știinta) and pre-1993 î-inside-word spellings (cînd, sînt). Removed only when a
    # Hunspell-valid sibling is at least twice as frequent; the score moves to that sibling.
    def move_to(w, sibs, why):
        tot = sum(orig.get(s, counts[s]) for s in sibs)
        for s in sibs:
            add = counts[w] * orig.get(s, counts[s]) / tot
            counts[s] += add
            changed[s][0] += add
            changed[s][1].append(w)
        removed.append((w, raw[w], why, sibs, diac_ev[w], 0))
        del counts[w]

    for cls, ws in list(classes.items()):
        present = [w for w in ws if w in counts]
        if len(present) < 2:
            continue
        for w in present:
            if fold(w) == w or w in hv or w not in counts:
                continue
            sibs = [s for s in present if s != w and s in hv and s in counts]
            if sibs and sum(counts[s] for s in sibs) >= 2 * counts[w]:
                move_to(w, sibs, "partial diacritics; Hunspell rejects; valid sibling dominates")
                stats["removed_partial_diacritics"] += 1
    for w in list(counts):
        if w in hv or len(w) < 3 or "î" not in w[1:-1]:
            continue
        alt = ("sunt" + w[4:]) if w.startswith("sînt") else w[0] + w[1:-1].replace("î", "â") + w[-1]
        if alt != w and alt in counts and alt in hv and counts[alt] >= 2 * counts[w]:
            move_to(w, [alt], "pre-1993 orthography (î inside the word)")
            stats["removed_old_orthography"] += 1
    print(f"diacritic cleaning: {dict(stats)}", flush=True)

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    # superset vocabulary for next-word pair counting (tools/ngrams.py count), so pair counts
    # can be reused for any final dictionary size up to this many words
    (BUILD / "ranked-superset.txt").write_text("".join(f"{w}\n" for w, _ in ranked[:600000]), encoding="utf-8")
    if args.curve:
        curve(ranked)
    final = ranked[:args.size]
    total = sum(c for _, c in final)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{w}\t{max(1, round(c / total * 1e10))}\n" for w, c in final), encoding="utf-8")
    with open(BUILD / "wordlist-removed.tsv", "w", encoding="utf-8") as fh:
        fh.write("# ASCII spellings removed by the diacritic rules. word\traw count\treason\tmoved to\tdiacritic-text uses ascii/diacritic\n")
        for f, c, why, to, cf, cd in sorted(removed, key=lambda r: -r[1]):
            fh.write(f"{f}\t{c}\t{why}\t{','.join(to)}\t{cf}/{cd}\n")
        fh.write("# Frequent words (raw >= 50) dropped by the evidence filter. word\traw count\treason\n")
        for w, c, r in sorted(dropped, key=lambda r: -r[1]):
            fh.write(f"{w}\t{c}\t{r}\n")
    with open(BUILD / "wordlist-changed.tsv", "w", encoding="utf-8") as fh:
        fh.write("# word\tscore before\tscore after\twhy\n")
        for w, (delta, why) in sorted(changed.items(), key=lambda kv: -abs(kv[1][0])):
            if w in counts:
                fh.write(f"{w}\t{orig[w]:.3e}\t{counts[w]:.3e}\t{'; '.join(why[:6])}\n")
    proper = sum(1 for w, _ in final if mid[w] >= 5 and cap[w] / mid[w] >= 0.9)
    english = sum(1 for w, _ in final if w in en_words and w not in hv)
    report = {"sources": {s: {"tokens": totals[s], "types": len(per[s]), "weight": round(weights[s], 4)} for s in per},
              "candidates_3plus": len(cand), "hunspell_valid": len(hv), "evidence_kept": len(kept),
              "evidence_kept_by": dict(why_kept), "evidence_dropped_by": dict(reasons),
              "diacritic_cleaning": dict(stats), "after_cleaning": len(counts), "size": len(final),
              "proper_nouns_in_list (>=90% capitalised mid-sentence)": proper,
              "english_spelled_not_hunspell_in_list": english,
              "keep": sorted(KEEP), "min_share": MIN_SHARE, "min_evidence": MIN_EVIDENCE,
              "min_evidence_share": MIN_EVIDENCE_SHARE, "dominate": DOMINATE}
    (BUILD / "wordlist-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    loans = [w for w, _ in final if w in en_words and w not in hv][:300]
    (BUILD / "wordlist-english-loans.txt").write_text("\n".join(loans) + "\n", encoding="utf-8")


def curve(ranked):
    """Token coverage of the dev sets by the top-N words (the size decision)."""
    sizes = [20000, 30000, 40000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, 250000, 300000, 400000, 500000]
    rank = {w: i for i, (w, _) in enumerate(ranked)}
    res = {}
    for src in ("tatoeba", "cv", "wikitalk", "wiki"):
        path = WORK / "eval" / f"{src}.dev.txt"
        if not path.exists():
            continue
        toks = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                toks += [t.lower() for t in tokens(line.split("\t", 1)[1])]
        n = len(toks)
        rk = [rank.get(t, 10 ** 9) for t in toks]
        res[src] = {"tokens": n, **{str(s): round(sum(r < s for r in rk) / n, 5) for s in sizes},
                    "all": round(sum(r < 10 ** 9 for r in rk) / n, 5)}
    (BUILD / "coverage-curve.json").write_text(json.dumps(res, indent=1) + "\n")
    print("coverage curve (share of dev tokens in top-N):")
    print("N\t" + "\t".join(res))
    for s in [str(x) for x in sizes] + ["all"]:
        print(s + "\t" + "\t".join(f"{res[src][s]:.4f}" for src in res))


if __name__ == "__main__":
    main()
