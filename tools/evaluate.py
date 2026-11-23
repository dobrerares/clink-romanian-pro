#!/usr/bin/env python3
"""Offline comparison of Clink Romanian packs on held-out text.

Usage: evaluate.py [--split test|dev] [--seeds 1,2,3,4,5] [--n 1000]
                   [--pack name=path/prefix ...] [--tasks coverage,typo,next,diacless] [--out build/eval-test]

Every pack is read from its binary files (.clex/.cngm) with tools/packio.py, i.e. the dequantised
frequencies and next-word lists the keyboard itself receives.

PROXY WARNING: Clink's autocorrect and prediction engine is closed source and cannot run here.
"typo" uses a simple dictionary corrector (DictCorrector) over each pack's word list; "next" ranks
each pack's CNGM followers. Numbers compare the *data* in the packs under one fixed, transparent
decoder; they are not measurements of Clink on a device.

Tasks
  coverage : share of held-out word tokens (letters, internal hyphens allowed) in each dictionary.
  typo     : every word of sampled held-out sentences is typed with synthetic touch noise
             (tools/typing_noise.py: gaussian and realistic modes, sigma 0.22, 3% slips), Romanian
             either with diacritics (long-press) or without (ro_plain: sa, si, fata, dupa, stiu...).
             DictCorrector keeps a typed word that is in the dictionary; otherwise it restores
             diacritics (most frequent word with the same letters without marks), else picks the most
             frequent word one edit away in diacritic-folded space, else keeps the typing.
             Two configurations: the pack alone (ro) and Combined mode (pack + official en.clex).
             English Tatoeba test sentences measure what the Romanian pack does to English typing.
  next     : top-3 next-word accuracy: for each adjacent word pair in held-out sentences (official
             build-next-word tokeniser, pairs not crossing sentence ends), is the next word among the
             first three CNGM followers of the previous word? "+fallback" fills empty slots with the
             pack's three most frequent words; "after 1 letter" filters followers by the next word's
             first letter, then the most frequent dictionary words with that letter.
  diacless : deterministic table of what each pack does with common words typed without diacritics.

95% confidence intervals: each seed draws its own sentence sample and noise; the interval is
mean +- t(0.975, seeds-1) * sd / sqrt(seeds) over seeds; differences to the baselines are paired
(same sentences and same typed strings for every pack).
"""
import argparse
import bz2
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, WORK, dedup_key, fold, norm, split_tatoeba, tokens  # noqa: E402
from ngrams import SENT_END, SPLIT_RE, word  # noqa: E402
from packio import Clex, Cngm  # noqa: E402
from typing_noise import TapNoise, strip_diacritics, type_word  # noqa: E402

UP = Path("/home/rdobre/projects/clink/upstream/clink-language-packs/Lexicons")
PUB = Path("/home/rdobre/projects/clink/clink-romanian-language/Lexicons")
DEFAULT_PACKS = {"official": UP / "ro", "published": PUB / "ro", "pro": ROOT / "Lexicons" / "ro"}
EN_CLEX = UP / "en.clex"
SETS = ("tatoeba", "wiki", "wikitalk", "cv")
CORE = re.compile(r"^([\"'„“(«]*)(.*?)([.,!?;:\"'”»)]*)$")
LETTERS = re.compile(r"[A-Za-zĂÂÎȘȚăâîșț]+")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci(values):
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, T975.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))


# ------------------------------------------------------------------ data
def load_set(name, split):
    path = WORK / "eval" / f"{name}.{split}.txt"
    return [line.rstrip("\n").split("\t", 1)[1] for line in open(path, encoding="utf-8")]


def load_english(split):
    seen, out = set(), []
    with bz2.open(DATA / "tatoeba" / "eng_sentences.tsv.bz2", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or split_tatoeba(parts[0]) != split:
                continue
            k = dedup_key(parts[2])
            if k not in seen:
                seen.add(k)
                out.append(norm(parts[2]))
    return out


def sample(sentences, n, seed, lo=3, hi=25):
    pool = [s for s in sentences if lo <= len(s.split()) <= hi]
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


# ------------------------------------------------------------------ packs
class Pack:
    def __init__(self, name, prefix):
        self.name = name
        self.clex = Clex(f"{prefix}.clex")
        cngm = Path(f"{prefix}.cngm")
        self.cngm = Cngm(cngm, self.clex) if cngm.exists() else None
        self.freq = self.clex.freq()
        self.words = set(self.clex.words)
        ranked = sorted(self.freq.items(), key=lambda kv: -kv[1])
        self.uni_top = [w for w, _ in ranked[:3]]
        self.by_initial = defaultdict(list)
        for w, _ in ranked:
            if len(self.by_initial[w[0]]) < 3:
                self.by_initial[w[0]].append(w)
        self.files = {s: Path(f"{prefix}.{s}").stat().st_size for s in ("clex", "cngm") if Path(f"{prefix}.{s}").exists()}


class DictCorrector:
    """Dictionary-corrector PROXY (not Clink's engine). See module docstring."""

    def __init__(self, ro_freq, en_freq=None):
        freq = dict(ro_freq)
        if en_freq:
            for w, p in en_freq.items():
                freq[w] = max(freq.get(w, 0.0), p)
        self.freq = freq
        self.by_fold = defaultdict(list)
        self.dels = defaultdict(list)
        for w in freq:
            f = fold(w)
            self.by_fold[f].append(w)
            self.dels[f].append(w)
            for i in range(len(f)):
                self.dels[f[:i] + f[i + 1:]].append(w)

    def best(self, cands):
        return max(cands, key=lambda w: (self.freq[w], w))

    def fix(self, low):
        if low in self.freq or len(low) < 2:
            return low
        f = fold(low)
        same = self.by_fold.get(f)
        if same:
            return self.best(same)
        cands = set()
        for v in [f] + [f[:i] + f[i + 1:] for i in range(len(f))]:
            cands.update(self.dels.get(v, ()))
        cands = [w for w in cands if abs(len(w) - len(f)) <= 1]
        return self.best(cands) if cands else low


# ------------------------------------------------------------------ tasks
def coverage(packs, sets):
    res = {}
    for name, sents in sets.items():
        toks = [t.lower() for s in sents for t in tokens(s)]
        toks = [t for t in toks if re.fullmatch(r"[a-zăâîșț]+(?:-[a-zăâîșț]+)*", t)]
        res[name] = {"tokens": len(toks)}
        for p in packs:
            res[name][p.name] = round(sum(t in p.words for t in toks) / len(toks), 5)
    return res


def typo_run(sents, lang, mode, noise_mode, seed, correctors, sigma=0.22, err=0.03):
    """correctors: {pack: DictCorrector}. Same typed strings for every pack (paired)."""
    rng = random.Random(seed * 7919 + 17)
    noise = TapNoise(noise_mode, seed)
    stats = {p: defaultdict(int) for p in correctors}
    for text in sents:
        for raw in text.split():
            m = CORE.match(raw)
            core = m.group(2)
            if not core or not LETTERS.fullmatch(core):
                continue
            want = core.lower()
            diac = mode == "ro_diac" or lang == "en"
            typed = type_word(core, rng, sigma, diac, err, noise).lower()
            if mode == "ro_plain":
                typed = strip_diacritics(typed)
            kind = "clean" if typed == want else ("accent" if fold(want) == typed else "typo")
            for p, corr in correctors.items():
                out = corr.fix(typed)
                st = stats[p]
                st["words"] += 1
                st["ok"] += out == want
                st[kind] += 1
                st[kind + "_ok"] += out == want
                st["changed"] += out != typed
                st["wrong_change"] += out != typed and out != want
    return stats


def summarize_typo(st):
    w = st["words"] or 1
    r = {"word_acc": st["ok"] / w, "wrong_change": st["wrong_change"] / w}
    if st["clean"]:
        r["clean_broken"] = 1 - st["clean_ok"] / st["clean"]
    if st["typo"]:
        r["typo_fixed"] = st["typo_ok"] / st["typo"]
    if st["accent"]:
        r["accent_fixed"] = st["accent_ok"] / st["accent"]
    return r


def next_run(sents, packs):
    stats = {p.name: defaultdict(int) for p in packs}
    for text in sents:
        for seg in SENT_END.split(text):
            toks = [word(t) for t in SPLIT_RE.findall(seg)]
            for a, b in zip(toks, toks[1:]):
                if a is None or b is None:
                    continue
                for p in packs:
                    st = stats[p.name]
                    st["n"] += 1
                    fol = p.cngm.followers.get(a, []) if p.cngm else []
                    top = [w for w, _ in fol[:3]]
                    st["top3"] += b in top
                    st["has_followers"] += bool(fol)
                    fb = (top + [u for u in p.uni_top if u not in top])[:3]
                    st["top3_fallback"] += b in fb
                    c1 = [w for w, _ in fol if w[0] == b[0]][:3]
                    c1 += [u for u in p.by_initial.get(b[0], []) if u not in c1]
                    st["top3_after_1_letter"] += b in c1[:3]
                    st["next_in_dictionary"] += b in p.words
    return stats


DIACLESS = ["să", "și", "după", "știu", "fată", "fața", "până", "mă", "că", "mulțumesc", "așa", "ți", "îți",
            "ăsta", "asta", "țară", "țara", "mâine", "astăzi", "încă", "înainte", "făcut", "trebuie", "vrea",
            "acasă", "dacă", "fără", "pentru", "câteva", "nimeni", "bună", "aici", "acolo", "ușă", "școală",
            "frumoasă", "mașină", "întâlnire", "săptămână", "lună", "tău", "său", "ție", "mie", "pe", "tine",
            "îmi", "spune", "spună", "vină", "vina", "casă", "casa", "masă", "masa", "fată", "fata", "pana", "peste"]


def diacless(packs, en):
    rows = [{"intended": w, "typed": fold(w)} for w in dict.fromkeys(DIACLESS)]
    for p in packs:
        corr_ro = DictCorrector(p.freq)
        corr_co = DictCorrector(p.freq, en)
        for row in rows:
            a, b = corr_ro.fix(row["typed"]), corr_co.fix(row["typed"])
            row[p.name] = a if a == b else f"{a} (combined: {b})"
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-next", type=int, default=2000)
    ap.add_argument("--pack", action="append", default=[])
    ap.add_argument("--tasks", default="coverage,typo,next,diacless")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--noise", default="gaussian,realistic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    specs = dict(kv.split("=", 1) for kv in args.pack) if args.pack else {k: str(v) for k, v in DEFAULT_PACKS.items()}
    packs = [Pack(n, p) for n, p in specs.items()]
    en = Clex(EN_CLEX).freq()
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = set(args.tasks.split(","))
    sets = {s: load_set(s, args.split) for s in args.sets.split(",")}
    out = {"split": args.split, "seeds": seeds, "packs": {p.name: {"prefix": specs[p.name], "words": len(p.words),
           "pairs": p.cngm.pairs if p.cngm else 0, "bytes": p.files} for p in packs},
           "sentences_available": {k: len(v) for k, v in sets.items()}}
    if "coverage" in tasks:
        out["coverage"] = coverage(packs, sets)
        print(json.dumps(out["coverage"]), flush=True)
    if "typo" in tasks:
        corr = {"ro": {p.name: DictCorrector(p.freq) for p in packs},
                "combined": {p.name: DictCorrector(p.freq, en) for p in packs}}
        english = load_english(args.split)
        res = {}
        for noise_mode in args.noise.split(","):
            for cfg, cs in corr.items():
                for setname in list(sets) + ["english"]:
                    modes = ["en"] if setname == "english" else ["ro_plain", "ro_diac"]
                    if setname == "english" and cfg == "ro":
                        continue
                    for mode in modes:
                        per_seed = defaultdict(list)
                        for seed in seeds:
                            src = english if setname == "english" else sets[setname]
                            sents = sample(src, args.n, seed)
                            st = typo_run(sents, "en" if setname == "english" else "ro", mode, noise_mode, seed, cs)
                            for p in cs:
                                for k, v in summarize_typo(st[p]).items():
                                    per_seed[(p, k)].append(v)
                            per_seed[("_", "words")].append(st[packs[0].name]["words"])
                        key = f"{noise_mode}/{cfg}/{setname}/{mode}"
                        res[key] = {}
                        for (p, k), vals in per_seed.items():
                            m, h = ci(vals)
                            res[key].setdefault(p, {})[k] = [round(m, 5), round(h, 5)]
                        for base in ("official", "published"):
                            if base in cs and "pro" in cs:
                                d = [a - b for a, b in zip(per_seed[("pro", "word_acc")], per_seed[(base, "word_acc")])]
                                m, h = ci(d)
                                res[key][f"pro_minus_{base}_word_acc"] = [round(m, 5), round(h, 5)]
                        print(key, json.dumps(res[key]), flush=True)
        out["typo"] = res
    if "next" in tasks:
        res = {}
        for setname, src in sets.items():
            per_seed = defaultdict(list)
            for seed in seeds:
                st = next_run(sample(src, args.n_next, seed, lo=2, hi=40), packs)
                for p in packs:
                    n = st[p.name]["n"] or 1
                    for k in ("top3", "top3_fallback", "top3_after_1_letter", "has_followers", "next_in_dictionary"):
                        per_seed[(p.name, k)].append(st[p.name][k] / n)
                per_seed[("_", "pairs")].append(st[packs[0].name]["n"])
            res[setname] = {}
            for (p, k), vals in per_seed.items():
                m, h = ci(vals)
                res[setname].setdefault(p, {})[k] = [round(m, 5), round(h, 5)]
            for base in ("official", "published"):
                if base in specs and "pro" in specs:
                    for k in ("top3_fallback", "top3_after_1_letter"):
                        d = [a - b for a, b in zip(per_seed[("pro", k)], per_seed[(base, k)])]
                        m, h = ci(d)
                        res[setname][f"pro_minus_{base}_{k}"] = [round(m, 5), round(h, 5)]
            print("next", setname, json.dumps(res[setname]), flush=True)
        out["next"] = res
    if "diacless" in tasks:
        out["diacless"] = diacless(packs, en)
    if args.out:
        Path(args.out + ".json").parent.mkdir(parents=True, exist_ok=True)
        Path(args.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return out


if __name__ == "__main__":
    main()
