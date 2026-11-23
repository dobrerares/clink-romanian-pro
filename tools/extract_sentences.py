#!/usr/bin/env python3
"""Normalise and split the sentence corpora that are not wiki dumps.

  tatoeba : data/tatoeba/ron_sentences.tsv.bz2 (id, lang, text). Near-duplicates collapse to one
            representative (spelling with diacritics first, then the lowest id) whose id decides the
            split - the same rule and buckets as autocorrect-engine/tools/corpus.py, so the Tatoeba
            test split is the one used by earlier Bilingual Guard evaluations.
  cv      : data/commonvoice/sentence-collector.txt (one sentence per line, CC0). Split 80/10/10
            on the hash of the near-duplicate key; duplicates collapse.

Writes work/sent/<source>.<split>.tsv ("<id>\t<sentence>") and work/sent/<source>.stats.json.
"""
import bz2
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DIAC, SENT, dedup_key, norm, split_sentence, split_tatoeba  # noqa: E402


def write(source, rows):
    SENT.mkdir(parents=True, exist_ok=True)
    outs = {sp: open(SENT / f"{source}.{sp}.tsv", "w", encoding="utf-8") for sp in ("train", "dev", "test")}
    stats = Counter()
    for sid, split, text in rows:
        outs[split].write(f"{sid}\t{text}\n")
        stats[split] += 1
    for fh in outs.values():
        fh.close()
    return stats


def tatoeba():
    best = {}
    raw = 0
    with bz2.open(DATA / "tatoeba" / "ron_sentences.tsv.bz2", "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            raw += 1
            sid, text = parts[0], norm(parts[2]).strip()
            k = dedup_key(text)
            rank = (0 if DIAC & set(text.lower()) else 1, int(sid) if sid.isdigit() else 0)
            if k not in best or rank < best[k][0]:
                best[k] = (rank, sid, text)
    rows = sorted(((sid, split_tatoeba(sid), text) for _, sid, text in best.values()), key=lambda r: int(r[0]))
    stats = write("tatoeba", rows)
    stats["raw"] = raw
    stats["near_duplicates_dropped"] = raw - len(rows)
    return stats


def commonvoice():
    seen = set()
    rows = []
    raw = 0
    for i, line in enumerate((DATA / "commonvoice" / "sentence-collector.txt").read_text(encoding="utf-8").splitlines()):
        text = norm(line).strip()
        if not text:
            continue
        raw += 1
        k = dedup_key(text)
        if k in seen:
            continue
        seen.add(k)
        rows.append((f"cv{i}", split_sentence("cv", text), text))
    stats = write("cv", rows)
    stats["raw"] = raw
    stats["near_duplicates_dropped"] = raw - len(rows)
    return stats


def main():
    for name, fn in (("tatoeba", tatoeba), ("cv", commonvoice)):
        stats = fn()
        (SENT / f"{name}.stats.json").write_text(json.dumps(stats, indent=1) + "\n")
        print(name, json.dumps(stats))


if __name__ == "__main__":
    main()
