#!/usr/bin/env python3
"""Next-word (CNGM v1) model for Romanian Pro.

Subcommands
  count  VOCAB SOURCE...    count adjacent word pairs in work/train/<SOURCE>.txt over the ranked
                            superset vocabulary VOCAB (one word per line, first column); saves
                            work/pairs/<SOURCE>.npz. Pairs never span: sentence ends (. ! ? …), gap
                            marks left by removed markup (¦), or any token that is not a vocabulary
                            word (numbers, foreign script, rare words).
  build  WORDLIST OUT [--weights s=w,...] [--top-k K] [--min-count M]
                            assemble a CNGM for WORDLIST (the list given to build-pack.py) from the
                            saved counts: per-source counts are normalised by the source's pair total
                            and mixed with the weights; P(next|prev) is the mixed count over all of
                            prev's followers (before pruning, as in the official builder); then each
                            previous word keeps its K most likely followers seen at least M times
                            (raw, all sources); with --min-top T a word's three most likely
                            followers only need T occurrences.
  official WORDLIST CORPUS OUT
                            exactly the algorithm of upstream tools/build-next-word.py (used to prove
                            that this serialiser is byte-identical to the official one).

The file layout and quantisation are those of tools/build-next-word.py: "CNGM", u32 1, u32 n,
n prev ids, n next ids, n bytes round((log10 p + 6) * 42); ids are positions in the bytewise-sorted
word list; groups by prev id ascending, most likely follower first, ties by next id.
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import re
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK  # noqa: E402

PAIRS = WORK / "pairs"
SPLIT_RE = re.compile(r'[^\s.,!?;:"“”‘’()\[\]{}]+')  # official tokeniser
SENT_END = re.compile(r"[.!?…¦]+")
SHIFT = 20  # pair key = prev << SHIFT | next  (vocabulary < 2**20)


def word(value):
    """Official normalisation (build-next-word.py)."""
    value = unicodedata.normalize("NFC", value.strip().lower())
    return value if value and not any(c.isspace() for c in value) and any(c.isalpha() for c in value) else None


def load_wordlist(path):
    words = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            cand = word(raw.rsplit(maxsplit=1)[0])
            if cand:
                words.add(cand)
    return sorted(words, key=lambda v: v.encode("utf-8"))


# ---------------------------------------------------------------- official reproduction
def official(wordlist, corpus, out):
    """Count pairs exactly like upstream build-next-word.py (out-of-list tokens are dropped
    *before* pairing), then serialise through assemble() - the same code path as `build`."""
    ordered = load_wordlist(wordlist)
    ids = {v: i for i, v in enumerate(ordered)}
    pairs = collections.Counter()
    for raw in Path(corpus).read_text(encoding="utf-8").splitlines():
        sentence = raw.rsplit("\t", 1)[-1]
        toks = [word(t) for t in SPLIT_RE.findall(sentence)]
        toks = [t for t in toks if t in ids]
        pairs.update(zip(toks, toks[1:]))
    keys = np.array([(ids[a] << SHIFT) | ids[b] for a, b in pairs], dtype=np.int64)
    cnts = np.array(list(pairs.values()), dtype=np.int64)
    blob, info = assemble(len(ordered), [(keys, cnts, 1.0)], top_k=1 << 30, min_count=1)
    Path(out).write_bytes(blob)
    print(f"official-equivalent: {info['pairs']:,} pairs -> {out}")


# ---------------------------------------------------------------- counting
_VOCAB = None


def _init(vocab_path):
    global _VOCAB
    words = [l.split("\t", 1)[0] for l in Path(vocab_path).read_text(encoding="utf-8").splitlines() if l]
    _VOCAB = {w: i for i, w in enumerate(words)}


def _count_chunk(lines):
    keys = []
    for line in lines:
        text = line.split("\t", 1)[-1]
        for seg in SENT_END.split(text):
            prev = None
            for t in SPLIT_RE.findall(seg):
                i = _VOCAB.get(word(t)) if any(c.isalpha() for c in t) else None
                if i is not None and prev is not None:
                    keys.append((prev << SHIFT) | i)
                prev = i
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    u, c = np.unique(np.array(keys, dtype=np.int64), return_counts=True)
    return u, c.astype(np.int64)


def _merge(parts):
    keys = np.concatenate([p[0] for p in parts])
    cnts = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=cnts).astype(np.int64)


def _chunks(path, size=20000):
    buf = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line.rstrip("\n"))
            if len(buf) >= size:
                yield buf
                buf = []
    if buf:
        yield buf


def count(vocab, sources, workers=7):
    PAIRS.mkdir(parents=True, exist_ok=True)
    vocab_words = [l.split("\t", 1)[0] for l in Path(vocab).read_text(encoding="utf-8").splitlines() if l]
    assert len(vocab_words) < 2 ** SHIFT
    for src in sources:
        parts, acc = [], None
        with mp.Pool(workers, initializer=_init, initargs=(vocab,)) as pool:
            for part in pool.imap_unordered(_count_chunk, _chunks(WORK / "train" / f"{src}.txt")):
                parts.append(part)
                if len(parts) >= 40:
                    acc = _merge(parts + ([acc] if acc is not None else []))
                    parts = []
        acc = _merge(parts + ([acc] if acc is not None else []))
        np.savez(PAIRS / f"{src}.npz", keys=acc[0], counts=acc[1])
        meta = {"source": src, "vocab": str(vocab), "vocab_size": len(vocab_words),
                "distinct_pairs": int(len(acc[0])), "pair_tokens": int(acc[1].sum())}
        (PAIRS / f"{src}.json").write_text(json.dumps(meta) + "\n")
        (PAIRS / "vocab.txt").write_text("\n".join(vocab_words) + "\n", encoding="utf-8")
        print(json.dumps(meta), flush=True)


# ---------------------------------------------------------------- building
def assemble(n_words, sources, top_k, min_count, min_top=None):
    """sources: [(keys, counts, weight, multiplier-or-None)] with keys = prev << SHIFT | next in final
    word ids. One source: P(next|prev) = count / total(prev), exactly the official computation.
    Several: each source's counts are divided by its pair total and multiplied by its weight (and by
    the optional per-pair multiplier used for artifact demotion)."""
    if len(sources) == 1:
        keys, cnts = sources[0][0], sources[0][1]
        u, inv = np.unique(keys, return_inverse=True)
        w = np.bincount(inv, weights=cnts.astype(np.float64))
        r = w.copy()
    else:
        wsum = sum(x[2] for x in sources)
        keys = np.concatenate([x[0] for x in sources])
        u, inv = np.unique(keys, return_inverse=True)
        parts = []
        for x in sources:
            c, wg = x[1], x[2]
            mult = x[3] if len(x) > 3 and x[3] is not None else 1.0
            parts.append(c * mult * (wg / wsum / c.sum()))
        w = np.bincount(inv, weights=np.concatenate(parts))
        r = np.bincount(inv, weights=np.concatenate([x[1].astype(np.float64) for x in sources]))
    prev = u >> SHIFT
    nxt = u & ((1 << SHIFT) - 1)
    totals = np.bincount(prev, weights=w, minlength=n_words)
    # order: prev ascending, most likely first, ties by next id (official sort key)
    order = np.lexsort((nxt, -w, prev))
    prev, nxt, w, r = prev[order], nxt[order], w[order], r[order]
    starts = np.r_[0, np.flatnonzero(np.diff(prev)) + 1]
    rank = np.arange(len(prev)) - np.repeat(starts, np.diff(np.r_[starts, len(prev)]))
    keep = (rank < top_k) & (r >= min_count)
    if min_top is not None:
        # the three most likely followers (the ones a next-word bar shows) need less evidence,
        # so small conversational sources are not pruned away by a raw-count threshold
        keep |= (rank < min(3, top_k)) & (r >= min_top)
    prev, nxt, w = prev[keep], nxt[keep], w[keep]
    tot = totals[prev]
    # quantise with Python floats/math.log10, as the official tool does
    qb = bytes(max(0, min(255, round((math.log10(a / b) + 6) * 42))) for a, b in zip(w.tolist(), tot.tolist()))
    blob = bytearray(b"CNGM" + struct.pack("<II", 1, len(prev)))
    blob += prev.astype("<u4").tobytes()
    blob += nxt.astype("<u4").tobytes()
    blob += qb
    info = {"pairs": int(len(prev)), "bytes": len(blob), "previous_words": int(len(np.unique(prev))),
            "candidate_pairs": int(len(u)), "top_k": top_k, "min_count": min_count, "min_top": min_top}
    return bytes(blob), info


# Corpus artifacts demoted as *followers* (weight x DEMOTE_FACTOR) in the source that produces them;
# they stay in the dictionary and keep their evidence from every other source.
#  tatoeba : Tatoeba's stock characters (Tom, Mary, John) follow "că", "pe", "lui" in thousands of sentences.
#  wikitalk: Wikipedia self-reference in discussion pages ("pe Wikipedia", "la pagina", "acest articol").
DEMOTE = {
    "tatoeba": ["tom", "mary", "john"],
    "wikitalk": ["wikipedia", "wiki", "articol", "articolul", "articolului", "articole", "articolele", "articolelor",
                 "pagina", "paginii", "pagini", "paginile", "paginilor", "discuție", "discuția", "discuții",
                 "discuțiile", "utilizator", "utilizatorul", "utilizatorului", "utilizatori", "utilizatorii",
                 "format", "formatul", "categoria", "categorie", "categorii", "ștergere", "ștergerea", "șters",
                 "ștearsă", "redirect", "redirecționare", "politicile", "politica", "bloca", "blocat", "blocare",
                 "administrator", "administratori", "administratorii", "editare", "editări", "editările",
                 "modificare", "modificarea", "modificările", "sursă", "sursa", "surse", "sursele", "referințe",
                 "referințele", "vot", "votul", "votat", "voturi", "wikipediei", "proiect", "proiectul", "cafenea"],
}
DEMOTE_FACTOR = 0.05


def build(wordlist, out, weights, top_k, min_count, report=None, min_top=None, demote=False):
    vocab_words = (PAIRS / "vocab.txt").read_text(encoding="utf-8").split("\n")
    ordered = load_wordlist(wordlist)
    ids = {v: i for i, v in enumerate(ordered)}
    remap = np.full(len(vocab_words), -1, dtype=np.int64)
    for i, w in enumerate(vocab_words):
        j = ids.get(w)
        if j is not None:
            remap[i] = j
    sources = []
    for src, wgt in weights.items():
        if wgt <= 0:
            continue
        z = np.load(PAIRS / f"{src}.npz")
        k, c = z["keys"], z["counts"]
        p, q = remap[k >> SHIFT], remap[k & ((1 << SHIFT) - 1)]
        ok = (p >= 0) & (q >= 0)
        mult = None
        if demote and src in DEMOTE:
            bad = np.array([ids[x] for x in DEMOTE[src] if x in ids], dtype=np.int64)
            mult = np.where(np.isin(q[ok], bad), DEMOTE_FACTOR, 1.0)
        sources.append((((p[ok] << SHIFT) | q[ok]), c[ok], wgt, mult))
    blob, info = assemble(len(ordered), sources, top_k, min_count, min_top)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(blob)
    info["weights"] = weights
    info["demote"] = DEMOTE if demote else None
    info["demote_factor"] = DEMOTE_FACTOR if demote else None
    info["wordlist_words"] = len(ordered)
    if report:
        Path(report).write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))
    return info


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("count")
    a.add_argument("vocab")
    a.add_argument("sources", nargs="+")
    b = sub.add_parser("build")
    b.add_argument("wordlist")
    b.add_argument("out")
    b.add_argument("--weights", default="wiki=0.35,wikitalk=0.2,tatoeba=0.25,cv=0.1,wikinews=0.04,wikiquote=0.03,wikivoyage=0.02,wikibooks=0.01")
    b.add_argument("--top-k", type=int, default=32)
    b.add_argument("--min-count", type=int, default=2)
    b.add_argument("--min-top", type=int, default=None, help="lower raw-count threshold for each word's top-3 followers")
    b.add_argument("--demote", action="store_true", help="demote corpus-artifact followers (DEMOTE)")
    b.add_argument("--report", default=None)
    o = sub.add_parser("official")
    o.add_argument("wordlist")
    o.add_argument("corpus")
    o.add_argument("out")
    args = ap.parse_args()
    if args.cmd == "count":
        count(args.vocab, args.sources)
    elif args.cmd == "official":
        official(args.wordlist, args.corpus, args.out)
    else:
        weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
        build(args.wordlist, args.out, weights, args.top_k, args.min_count, args.report, args.min_top, args.demote)


if __name__ == "__main__":
    main()
