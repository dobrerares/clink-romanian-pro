#!/usr/bin/env python3
"""Make leak-free training and evaluation text from work/sent/<source>.<split>.tsv.

1. Held-out keys = near-duplicate keys (common.dedup_key) of every dev and test sentence of
   every source.
2. work/train/<source>.txt: "<doc id>\t<sentence>" training lines, with exact near-duplicates within
   the source collapsed and every sentence whose key is a held-out key removed (so nothing in
   any dev/test set - Tatoeba, Wikipedia, talk pages, Common Voice - is ever trained on, even if
   it occurs in another source).
3. work/eval/<source>.<split>.txt: dev/test sentences, deduplicated, without gap marks (removed
   inline markup), test sentences whose key also occurs in dev are dropped.

Usage: heldout.py SOURCE [SOURCE ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SENT, WORK, key_hash  # noqa: E402

TRAIN = WORK / "train"
EVAL = WORK / "eval"


def rows(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            sid, _, text = line.rstrip("\n").partition("\t")
            if text:
                yield sid, text


def main():
    sources = sys.argv[1:]
    TRAIN.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    held = set()
    report = {}
    for src in sources:
        dev_keys = set()
        for split in ("dev", "test"):
            n_in = n_out = 0
            seen = set()
            with open(EVAL / f"{src}.{split}.txt", "w", encoding="utf-8") as out:
                for sid, text in rows(SENT / f"{src}.{split}.tsv"):
                    n_in += 1
                    k = key_hash(text)
                    held.add(k)
                    if "¦" in text or k in seen or (split == "test" and k in dev_keys):
                        continue
                    seen.add(k)
                    if split == "dev":
                        dev_keys.add(k)
                    out.write(f"{sid}\t{text}\n")
                    n_out += 1
            report[f"{src}.{split}"] = {"in": n_in, "kept": n_out}
    (WORK / "heldout-keys.txt").write_text(f"{len(held)} held-out sentence keys from {sources}\n")
    for src in sources:
        seen = set()
        n_in = n_dup = n_held = 0
        with open(TRAIN / f"{src}.txt", "w", encoding="utf-8") as out:
            for sid, text in rows(SENT / f"{src}.train.tsv"):
                n_in += 1
                k = key_hash(text)
                if k in held:
                    n_held += 1
                    continue
                if k in seen:
                    n_dup += 1
                    continue
                seen.add(k)
                out.write(f"{sid}\t{text}\n")
        report[f"{src}.train"] = {"in": n_in, "dropped_duplicate": n_dup, "dropped_heldout_match": n_held,
                                  "kept": n_in - n_dup - n_held}
        print(src, json.dumps(report[f"{src}.train"]), flush=True)
    (WORK / "heldout-report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
