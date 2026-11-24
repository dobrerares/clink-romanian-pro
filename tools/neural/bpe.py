#!/usr/bin/env python3
"""Learn the BPE vocabulary exactly as the official trainer does, but incrementally.

The official train_bpe() (vendor/tools/train-neural.py) recounts every adjacent pair of every word
for each of the 12,000 merges, which takes many hours on a multi-million-sentence corpus. This file
implements the SAME algorithm with incremental pair counts:
  * input   = word frequencies from official iter_sentences() over the sentence file
              (clean_word, then each word -> chars + "</w>");
  * a merge = the pair with the largest (count, pair) - ties broken by the larger pair tuple, as in
              max(pair_counts, key=lambda p: (pair_counts[p], p));
  * applying a merge rewrites each word left to right without overlaps, exactly like the official
    loop; the vocabulary list (single chars, then multi-char tokens, sorted; then <unk> <bos> <eos>)
    and the BPE1 file are built with the official rules.
`--check N M` runs both implementations on the first N sentences with M merges and asserts that
merges and vocabulary are identical.

Usage: bpe.py SENTENCES.tsv OUT.bpevocab [--merges 12000]
       bpe.py SENTENCES.tsv --check 20000 1500
"""
import argparse
import heapq
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import LANG, official, write_bpevocab  # noqa: E402


class _Max:
    """Heap entry ordered so that heapq pops the largest (count, pair) first."""
    __slots__ = ("c", "p")

    def __init__(self, c, p):
        self.c, self.p = c, p

    def __lt__(self, other):
        return (self.c, self.p) > (other.c, other.p)


def word_freqs_from(path, max_lines=0):
    off = official()
    wf = defaultdict(int)
    n = 0
    with open(path, encoding="utf-8") as f:   # same reading as official iter_sentences()
        for line in f:
            if max_lines and n >= max_lines:
                break
            text = line.split("\t", 1)[-1].strip().lower()
            if text:
                toks = [t for t in (off.clean_word(x, LANG) for x in text.split()) if t]
                if toks:
                    for w in toks:
                        wf[w] += 1
                    n += 1
    return wf, n


def train_bpe_fast(word_counts, num_merges, log=True):
    """word_counts: {word: count}. Returns (vocab_list, merges) identical to official train_bpe."""
    off = official()
    WORD_END = off.WORD_END
    words = [list(w) + [WORD_END] for w in word_counts]
    freqs = list(word_counts.values())
    vocab_set = set()
    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for wi, w in enumerate(words):
        vocab_set.update(w)
        f = freqs[wi]
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += f
            where[p].add(wi)
    heap = [_Max(c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    t0 = time.time()
    if log:
        print(f"[bpe] {len(words)} unique words, {len(vocab_set)} initial tokens", flush=True)
    for step in range(num_merges):
        best = None
        while heap:
            e = heapq.heappop(heap)
            if e.c > 0 and pair_counts.get(e.p) == e.c:
                best = e.p
                break
        if best is None:
            break
        a, b = best
        merged = a + b
        merges.append(best)
        vocab_set.add(merged)
        changed = set()
        for wi in list(where.pop(best)):
            w = words[wi]
            if len(w) < 2:
                continue
            f = freqs[wi]
            # does this word still contain the pair?
            if not any(w[i] == a and w[i + 1] == b for i in range(len(w) - 1)):
                continue
            for i in range(len(w) - 1):
                p = (w[i], w[i + 1])
                pair_counts[p] -= f
                changed.add(p)
            new = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new.append(merged)
                    i += 2
                else:
                    new.append(w[i])
                    i += 1
            words[wi] = new
            for i in range(len(new) - 1):
                p = (new[i], new[i + 1])
                pair_counts[p] += f
                where[p].add(wi)
                changed.add(p)
        for p in changed:
            c = pair_counts[p]
            if c > 0:
                heapq.heappush(heap, _Max(c, p))
            else:
                del pair_counts[p]
        if log and (step + 1) % 2000 == 0:
            print(f"[bpe] merge {step + 1}/{num_merges}: '{a}' + '{b}' -> '{merged}' "
                  f"({time.time() - t0:.0f}s)", flush=True)
    # vocabulary list: the official construction
    chars = sorted(c for c in vocab_set if len(c) == 1)
    multi = sorted(c for c in vocab_set if len(c) > 1)
    seen = set()
    vocab_list = []
    for t in chars + multi:
        if t not in seen:
            seen.add(t)
            vocab_list.append(t)
    vocab_list += [off.UNK, off.BOS, off.EOS]
    return vocab_list, merges


def check(path, n_lines, n_merges):
    off = official()
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(sentences) >= n_lines:
                break
            text = line.split("\t", 1)[-1].strip().lower()
            toks = [t for t in (off.clean_word(x, LANG) for x in text.split()) if t]
            if toks:
                sentences.append(toks)
    t = time.time()
    v_off, m_off = off.train_bpe(sentences, n_merges)
    t_off = time.time() - t
    wc = defaultdict(int)
    for s in sentences:
        for w in s:
            wc[w] += 1
    t = time.time()
    v_new, m_new = train_bpe_fast(wc, n_merges, log=False)
    t_new = time.time() - t
    assert m_off == m_new, next(i for i, (x, y) in enumerate(zip(m_off, m_new)) if x != y)
    assert v_off == v_new
    print(f"identical: {len(m_new)} merges, {len(v_new)} tokens from {len(sentences)} sentences; "
          f"official {t_off:.1f}s, incremental {t_new:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sentences")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--merges", type=int, default=12000)
    ap.add_argument("--check", nargs=2, type=int, metavar=("LINES", "MERGES"))
    args = ap.parse_args()
    if args.check:
        check(args.sentences, *args.check)
        return
    wc, n = word_freqs_from(args.sentences)
    print(f"[bpe] {n} sentences, {sum(wc.values())} words", flush=True)
    vocab, merges = train_bpe_fast(wc, args.merges)
    write_bpevocab(args.out, vocab, merges)
    print(f"[bpe] wrote {args.out}: {len(vocab)} tokens, {len(merges)} merges")


if __name__ == "__main__":
    main()
