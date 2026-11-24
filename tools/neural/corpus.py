#!/usr/bin/env python3
"""Phase 2 neural corpus: filter the Phase 1 TRAIN text for the neural model.

Input : work/train/<source>.txt  ("<doc id>\t<sentence>"), i.e. the Phase 1 training split after
        tools/heldout.py removed every near-duplicate of any dev/test sentence of any source.
        NFC + comma-below normalisation was applied by the Phase 1 extractors; it is re-applied here
        (common.norm, idempotent).
Output: work/neural/corpus/<source>.tsv  ("<doc id>\t<sentence>") + work/neural/corpus/stats.json

Filters (a sentence is kept or dropped as a whole; the counts per reason go to stats.json):
  quotes      Romanian/typographic quotes „ “ ” « » are mapped to ASCII '"' first, because the
              official clean_word() only strips ASCII quotes: "„Apa" would otherwise be dropped as
              a non-word by the trainer.
  gap         contains the Phase 1 gap mark ¦ (inline markup was removed there).
  junk        a token that the official clean_word() rejects and that is not a hyphenated word
              (numbers, URLs, symbols, words glued by punctuation). The trainer would silently drop
              it and join the neighbours ("este de 35.405 locuitori" -> "este de locuitori").
              Hyphenated words (s-a, într-o) are also rejected by clean_word - as in every official
              model, whose vocabulary has no '-' - but they are ordinary Romanian, so such sentences
              are kept and the trainer drops just those words (consistent with the official model).
  foreign     a kept word with a letter outside a-z ă â î ș ț or an apostrophe (foreign names and
              scripts: keeps the BPE alphabet to the letters a Romanian keyboard types).
  short/long  fewer than 2 or more than 50 kept words.
  diacless    a word that Phase 1 removed as a diacritic-less or pre-1993 spelling (si, dupa, stiu,
              sînt, asa, maine ...; build/wordlist-removed.tsv): text written without diacritics would
              teach the model to predict those spellings.
  nodiac      8+ kept words and not a single ă â î ș ț (almost always diacritic-less typing).
  english     2+ English function words making up >= 20% of the words.
  template    "shape" cap: digits -> #, capitalised non-initial words -> P; at most 3 sentences per
              shape (Wikipedia/Wikimedia/talk only). Removes templated stubs such as
              "P este o comună din departamentul P, P." that Phase 1 saw as `o -> comună`.
Tatoeba stock names and Wikipedia self-reference are NOT filtered here: they are demoted at training
time (loss weight 0.05 when such a word is the predicted next word), the neural analogue of Phase 1's
follower demotion (tools/ngrams.py DEMOTE).

Usage: corpus.py [--sources wiki,wikitalk,...] [--workers 7]
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import NEURAL, ROOT, SOURCES, WORK, LANG, dump, official  # noqa: E402
from common import norm  # noqa: E402

OUT = NEURAL / "corpus"
QUOTES = str.maketrans({"„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‚": "'", "‘": "'",
                        "​": None, "‌": None, "‍": None, "﻿": None, "⁠": None})
NODIAC_EXEMPT = {"tatoeba", "cv"}  # curated: short correct sentences without ă/â/î/ș/ț are common
ALLOWED = set("abcdefghijklmnopqrstuvwxyzăâîșț'")
DIAC = set("ăâîșț")
HYPHENATED = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)+")
TEMPLATE_CAP = 3
NO_CAP = {"tatoeba", "cv"}
MIN_WORDS, MAX_WORDS = 2, 50
DIACLESS_REASONS = ("hunspell rejects; diacritic siblings dominate", "partial diacritics",
                    "pre-1993 orthography", "valid but")
# English function words: Phase 1's list (build/wordlist-removed.tsv, reason "English function
# word") plus the commonest ones that never made the candidate list. Romanian homographs (a, are,
# in, an, de, pe, la, nu ...) are excluded.
EN_EXTRA = {"the", "of", "to", "for", "and", "i", "i'm", "don't", "it's", "i'll", "you're", "can't",
            "please", "thanks", "thank", "will", "just", "or", "but"}

_OFF = None
_DIACLESS = None
_ENGLISH = None


def load_lists():
    diacless, english = set(), set(EN_EXTRA)
    for line in (ROOT / "build" / "wordlist-removed.tsv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 3:
            continue
        if f[2].startswith(DIACLESS_REASONS):
            diacless.add(f[0])
        elif f[2] == "English function word":
            english.add(f[0])
    return diacless, english


def _init():
    global _OFF, _DIACLESS, _ENGLISH
    _OFF = official()
    _DIACLESS, _ENGLISH = load_lists()


def shape_hash(text):
    out = []
    for i, tok in enumerate(text.split()):
        core = tok.strip(_OFF._STRIP)
        if not core:
            continue
        if any(c.isdigit() for c in core):
            out.append("#")
        elif i > 0 and core[0].isupper():
            out.append("P")
        else:
            out.append(core.lower())
    return int.from_bytes(hashlib.blake2b(" ".join(out).encode(), digest_size=8).digest(), "little")


def judge(text, nodiac=True):
    """-> (reason or None, kept-word count, shape hash)"""
    if "¦" in text:
        return "gap", 0, 0
    words = []
    raw = text.lower().split()
    for tok in raw:
        w = _OFF.clean_word(tok, LANG)
        if w is not None:
            words.append(w)
            continue
        core = _OFF.canonicalize(LANG, tok.strip().lower()).strip(_OFF._STRIP)
        if not core or HYPHENATED.fullmatch(core):
            # hyphenated: dropped by the trainer, but still evidence of diacritic-less typing (sa-mi)
            if core in _DIACLESS:
                return "diacless", 0, 0
            continue
        return "junk", 0, 0
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return ("short" if len(words) < MIN_WORDS else "long"), 0, 0
    if any(c not in ALLOWED for w in words for c in w):
        return "foreign", 0, 0
    if any(w in _DIACLESS for w in words):
        return "diacless", 0, 0
    if nodiac and len(words) >= 8 and not any(c in DIAC for w in words for c in w):
        return "nodiac", 0, 0
    n_en = sum(w in _ENGLISH for w in words)
    if n_en >= 2 and n_en >= 0.2 * len(words):
        return "english", 0, 0
    return None, len(words), shape_hash(text)


def _chunk(job):
    lines, nodiac = job
    out = []
    for line in lines:
        sid, _, text = line.partition("\t")
        text = " ".join(norm(text).translate(QUOTES).split())
        reason, n, h = judge(text, nodiac)
        out.append((sid, text, reason, n, h))
    return out


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


def build(src, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"in": 0, "kept": 0, "kept_words": 0}
    shapes = {}
    with mp.Pool(workers, initializer=_init) as pool, open(OUT / f"{src}.tsv", "w", encoding="utf-8") as out:
        nodiac = src not in NODIAC_EXEMPT
        jobs = ((c, nodiac) for c in _chunks(WORK / "train" / f"{src}.txt"))
        for res in pool.imap(_chunk, jobs):  # ordered
            for sid, text, reason, n, h in res:
                stats["in"] += 1
                if reason is None and src not in NO_CAP:
                    c = shapes.get(h, 0)
                    shapes[h] = c + 1
                    if c >= TEMPLATE_CAP:
                        reason = "template"
                if reason is not None:
                    stats[reason] = stats.get(reason, 0) + 1
                    continue
                stats["kept"] += 1
                stats["kept_words"] += n
                out.write(f"{sid}\t{text}\n")
    print(src, json.dumps(stats), flush=True)
    return stats


def assemble():
    """work/neural/ro_sentences.tsv: every kept sentence once, "<source>:<doc id>\t<sentence>",
    sources in SOURCES order. This is the trainer's sentence file (official format: the text is the
    last tab-separated column); per-source weights are applied at training time (train.py)."""
    n = 0
    with open(NEURAL / "ro_sentences.tsv", "w", encoding="utf-8") as out:
        for src in SOURCES:
            with open(OUT / f"{src}.tsv", encoding="utf-8") as fh:
                for line in fh:
                    out.write(f"{src}:{line}")
                    n += 1
    print(f"assembled {n} sentences -> {NEURAL / 'ro_sentences.tsv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    if args.assemble_only:
        assemble()
        return
    stats_path = OUT / "stats.json"
    allstats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for src in args.sources.split(","):
        allstats[src] = build(src, args.workers)
    allstats["_filters"] = {"template_cap": TEMPLATE_CAP, "min_words": MIN_WORDS, "max_words": MAX_WORDS,
                            "no_template_cap": sorted(NO_CAP), "nodiac_exempt": sorted(NODIAC_EXEMPT)}
    dump(stats_path, allstats)
    assemble()


if __name__ == "__main__":
    main()
