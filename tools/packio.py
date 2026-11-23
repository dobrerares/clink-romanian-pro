#!/usr/bin/env python3
"""Offline readers for Clink CLEX v1 dictionaries and CNGM v1 next-word models.

Formats as written by the official clink-language-packs builders (copied to vendor/):
  build-pack.py   -> CLEX: "CLEX", u32 version=1, u32 count, u32 n_alpha; n_alpha u32 code points;
                     (n_alpha+1)*n_alpha letter-transition bytes; (count+1) u32 byte offsets;
                     count log-prob bytes  round((log10 p + 9) * 28); count length bytes; UTF-8 word blob.
                     Words are sorted bytewise (UTF-8). Optional trailing "CBOS" section
                     (tools/augment-corpus-coverage.py): u32 version, u32 count, count u32 sentence-start counts.
  build-next-word.py -> CNGM: "CNGM", u32 version=1, u32 pairs; pairs u32 previous ids; pairs u32 next ids;
                     pairs probability bytes round((log10 P(next|prev) + 6) * 42).  Pairs are grouped by
                     previous id ascending, most frequent follower first. Ids index the CLEX word order.

Usage: packio.py PACK.clex [PACK.cngm] [word ...]
"""
import struct
import sys
from pathlib import Path


class Clex:
    def __init__(self, path):
        data = Path(path).read_bytes()
        if data[:4] != b"CLEX":
            raise ValueError(f"{path}: not a CLEX file")
        self.version, self.count, self.n_alpha = struct.unpack_from("<III", data, 4)
        pos = 16
        self.alphabet = [chr(c) for c in struct.unpack_from("<" + "I" * self.n_alpha, data, pos)]
        pos += 4 * self.n_alpha + (self.n_alpha + 1) * self.n_alpha
        offsets = struct.unpack_from("<" + "I" * (self.count + 1), data, pos)
        pos += 4 * (self.count + 1)
        self.logp_bytes = data[pos:pos + self.count]
        pos += self.count
        self.lengths = data[pos:pos + self.count]
        pos += self.count
        blob_len = offsets[-1]
        blob = data[pos:pos + blob_len]
        self.words = [blob[offsets[i]:offsets[i + 1]].decode("utf-8") for i in range(self.count)]
        pos += blob_len
        self.bos = None
        if data[pos:pos + 4] == b"CBOS":
            _v, n = struct.unpack_from("<II", data, pos + 4)
            self.bos = list(struct.unpack_from("<" + "I" * n, data, pos + 12))
        self.index = {w: i for i, w in enumerate(self.words)}
        self.size = len(data)

    def log10p(self, i):
        return self.logp_bytes[i] / 28.0 - 9.0

    def freq(self):
        """word -> dequantised probability (what the keyboard sees)."""
        return {w: 10 ** self.log10p(i) for i, w in enumerate(self.words)}


class Cngm:
    def __init__(self, path, clex):
        data = Path(path).read_bytes()
        if data[:4] != b"CNGM":
            raise ValueError(f"{path}: not a CNGM file")
        self.version, n = struct.unpack_from("<II", data, 4)
        prev = struct.unpack_from("<" + "I" * n, data, 12)
        nxt = struct.unpack_from("<" + "I" * n, data, 12 + 4 * n)
        prob = data[12 + 8 * n:12 + 9 * n]
        self.pairs = n
        self.size = len(data)
        self.followers = {}
        words = clex.words
        for p, q, b in zip(prev, nxt, prob):
            self.followers.setdefault(words[p], []).append((words[q], b / 42.0 - 6.0))

    def top(self, prev, k=3):
        return [w for w, _ in self.followers.get(prev, [])[:k]]


def main():
    clex = Clex(sys.argv[1])
    print(f"{sys.argv[1]}: {clex.count:,} words, {clex.size:,} bytes, alphabet={''.join(clex.alphabet)!r}, "
          f"CBOS={'yes' if clex.bos else 'no'}")
    rest = sys.argv[2:]
    ng = None
    if rest and rest[0].endswith(".cngm"):
        ng = Cngm(rest[0], clex)
        print(f"{rest[0]}: {ng.pairs:,} pairs, {ng.size:,} bytes, {len(ng.followers):,} previous words")
        rest = rest[1:]
    for w in rest:
        i = clex.index.get(w)
        line = f"{w}\t{'yes' if i is not None else 'no'}"
        if i is not None:
            line += f"\tlog10p={clex.log10p(i):.2f}"
        if ng:
            line += f"\tnext={ng.top(w, 5)}"
        print(line)


if __name__ == "__main__":
    main()
