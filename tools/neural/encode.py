#!/usr/bin/env python3
"""Encode the neural corpus with the official tokenizer -> work/neural/enc/<source>.npz

Each sentence becomes exactly what the official trainer's encode() produces:
  [<bos>] + bpe_encode_word(w) for every clean_word(w) + [<eos>]
Arrays: tokens (uint16, all sentences concatenated), starts (int64, sentence offsets, plus the end),
demote (uint8, 1 where the token is the FIRST token of a demoted word in this source: the neural
analogue of Phase 1's follower demotion in tools/ngrams.py DEMOTE - tatoeba: tom/mary/john; wikitalk:
Wikipedia self-reference words).

Usage: encode.py BPEVOCAB [--sources ...] [--workers 7]
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import NEURAL, SOURCES, Tokenizer, dump  # noqa: E402
from ngrams import DEMOTE  # noqa: E402  (Phase 1 lists)

ENC = NEURAL / "enc"
_TOK = None
_DEM = None


def _init(vocab, demote):
    global _TOK, _DEM
    _TOK = Tokenizer(vocab)
    _DEM = set(demote)


def _chunk(lines):
    toks, lens, dem = [], [], []
    for line in lines:
        words = _TOK.clean(line.split("\t", 1)[-1])
        ids = [_TOK.bos]
        flags = [0]
        for w in words:
            wi = _TOK.word_ids(w)
            ids.extend(wi)
            flags.append(1 if w in _DEM else 0)
            flags.extend([0] * (len(wi) - 1))
        ids.append(_TOK.eos)
        flags.append(0)
        toks.extend(ids)
        dem.extend(flags)
        lens.append(len(ids))
    return np.array(toks, np.uint16), np.array(lens, np.int64), np.array(dem, np.uint8)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vocab")
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    ENC.mkdir(parents=True, exist_ok=True)
    report = {}
    for src in args.sources.split(","):
        parts = []
        with mp.Pool(args.workers, initializer=_init, initargs=(args.vocab, DEMOTE.get(src, []))) as pool:
            for part in pool.imap(_chunk, _chunks(NEURAL / "corpus" / f"{src}.tsv")):
                parts.append(part)
        tokens = np.concatenate([p[0] for p in parts])
        lens = np.concatenate([p[1] for p in parts])
        demote = np.concatenate([p[2] for p in parts])
        starts = np.r_[0, np.cumsum(lens)].astype(np.int64)
        np.savez(ENC / f"{src}.npz", tokens=tokens, starts=starts, demote=demote)
        report[src] = {"sentences": int(len(lens)), "tokens": int(len(tokens)),
                       "targets": int(len(tokens) - len(lens)), "demoted_targets": int(demote.sum()),
                       "max_len": int(lens.max())}
        print(src, json.dumps(report[src]), flush=True)
    dump(ENC / "report.json", report)


if __name__ == "__main__":
    main()
