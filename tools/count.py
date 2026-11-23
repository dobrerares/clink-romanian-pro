#!/usr/bin/env python3
"""Count word types per source from leak-free training text.

Usage: count.py SOURCE [SOURCE ...]     (text sources: work/train/<source>.txt)
       count.py freqwords                (FrequencyWords ro_full.txt: counts only)

Writes work/counts/<source>.tsv: word, count, documents, mid-sentence occurrences,
mid-sentence capitalised occurrences (proper-noun evidence; the CLEX builder lower-cases
everything, so case only informs filtering), occurrences in sentences written with diacritics
(evidence for the clean_ro_pack rules), sorted by count.
A word is lower-case NFC Romanian/English letters with optional internal hyphens
(common.WORD_OK); tokens with digits, apostrophes or other scripts are not words.
"""
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, DIAC, WORD_OK, WORK, norm, tokens  # noqa: E402

COUNTS = WORK / "counts"


def count_chunk(lines):
    cnt, docs, mid, cap, diac = Counter(), Counter(), Counter(), Counter(), Counter()
    last_doc = None
    doc_words = set()
    for line in lines:
        sid, _, text = line.partition("\t")
        if sid != last_doc:
            docs.update(doc_words)
            doc_words = set()
            last_doc = sid
        has_diac = not DIAC.isdisjoint(text.lower())
        for i, tok in enumerate(tokens(text)):
            low = tok.lower()
            if not WORD_OK.fullmatch(low):
                continue
            cnt[low] += 1
            if has_diac:
                diac[low] += 1
            doc_words.add(low)
            if i > 0:
                mid[low] += 1
                if tok[0].isupper():
                    cap[low] += 1
    docs.update(doc_words)
    return cnt, docs, mid, cap, diac


def chunks(path, size=20000):
    buf = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            buf.append(line.rstrip("\n"))
            if len(buf) >= size:
                yield buf
                buf = []
    if buf:
        yield buf


def write(source, cnt, docs, mid, cap, diac):
    COUNTS.mkdir(parents=True, exist_ok=True)
    with open(COUNTS / f"{source}.tsv", "w", encoding="utf-8") as out:
        out.write("#word\tcount\tdocs\tmid\tcap_mid\tin_diacritic_sentences\n")
        for w, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
            out.write(f"{w}\t{c}\t{docs.get(w, 0)}\t{mid.get(w, 0)}\t{cap.get(w, 0)}\t{diac.get(w, 0)}\n")
    stats = {"source": source, "types": len(cnt), "tokens": sum(cnt.values())}
    print(json.dumps(stats), flush=True)
    return stats


def text_source(source, workers=7):
    tot = [Counter(), Counter(), Counter(), Counter(), Counter()]
    with mp.Pool(workers) as pool:
        for part in pool.imap_unordered(count_chunk, chunks(WORK / "train" / f"{source}.txt")):
            for t, p in zip(tot, part):
                t.update(p)
    return write(source, *tot)


def freqwords():
    cnt = Counter()
    rejected = 0
    for raw in (DATA / "frequencywords" / "ro_full.txt").read_text(encoding="utf-8").splitlines():
        parts = raw.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        w = norm(parts[0].strip()).lower()
        if WORD_OK.fullmatch(w):
            cnt[w] += int(parts[1])  # cedilla and comma spellings are one word: add them up
        else:
            rejected += 1
    stats = write("freqwords", cnt, {}, {}, {}, {})
    stats["rejected_entries"] = rejected
    return stats


def main():
    for src in sys.argv[1:]:
        stats = freqwords() if src == "freqwords" else text_source(src)
        (COUNTS / f"{src}.stats.json").write_text(json.dumps(stats) + "\n")


if __name__ == "__main__":
    main()
