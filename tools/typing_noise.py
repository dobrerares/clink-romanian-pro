"""Synthetic touch-typing noise, copied unchanged in behaviour from
autocorrect-engine/tools/eval.py (TapNoise, nearest, type_word) so this repo is self-contained.

Each intended letter is a tap drawn from a 2D Gaussian around its key on an iPhone-proportioned
QWERTY (key pitch 1, row pitch RY); the key under the tap is what gets typed. Small rates of a
missed letter, a doubled letter and two swapped letters are added (err). Romanian diacritics are
typed by long-press: the tap lands on the base key and, when typing with diacritics, the letter
keeps its mark. "realistic" adds key-specific bias, row-dependent spread and heavy tails
(a stress model, not a measured iOS profile).
"""
import hashlib
import random

ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
RY = 1.35
KW = 0.86
KH = 0.78 * RY
CEN = {}
for _r in range(3):
    for _i, _ch in enumerate(ROWS[_r]):
        CEN[_ch] = (_i + [0.0, 0.5, 1.5][_r], _r * RY)
BASEC = {"ă": "a", "â": "a", "î": "i", "ș": "s", "ț": "t"}


class TapNoise:
    def __init__(self, mode="gaussian", seed=1):
        if mode not in ("gaussian", "realistic"):
            raise ValueError("noise mode must be gaussian or realistic")
        self.mode = mode
        self.seed = seed
        self._noise_rng = random.Random(seed + 0xC11A)

    def bias(self, key):
        if self.mode == "gaussian":
            return (0.0, 0.0)
        digest = hashlib.blake2b(f"{self.seed}:{key}".encode(), digest_size=8).digest()
        local = random.Random(int.from_bytes(digest, "big"))
        return (local.uniform(-0.12, 0.12), local.uniform(-0.08, 0.08))

    def row_sigma(self, row, sigma):
        if self.mode == "gaussian":
            return sigma, sigma * 0.9 * RY
        factor = (0.9, 1.0, 1.2)[max(0, min(2, row))]
        return sigma * factor, sigma * 0.9 * RY * factor

    def offset(self, rng, key, row, sigma):
        sx, sy = self.row_sigma(row, sigma)
        sample_rng = rng if self.mode == "gaussian" else self._noise_rng
        if self.mode == "realistic" and sample_rng.random() < 0.08:
            sx *= 2.2
            sy *= 2.2
        bx, by = self.bias(key)
        return sample_rng.gauss(bx, sx), sample_rng.gauss(by, sy)


def nearest(x, y):
    r = min(2, max(0, int(round(y / RY))))
    row = ROWS[r]
    i = min(len(row) - 1, max(0, int(round(x - [0.0, 0.5, 1.5][r]))))
    return row[i]


def type_word(word, rng, sigma, diacritics, err, noise=None):
    """-> typed string for an intended word."""
    letters = list(word)
    noise = noise or TapNoise("gaussian")
    if len(letters) >= 3 and rng.random() < err:
        kind = rng.random()
        i = rng.randrange(len(letters))
        if kind < 0.45:
            del letters[i]
        elif kind < 0.75 and i + 1 < len(letters):
            letters[i], letters[i + 1] = letters[i + 1], letters[i]
        else:
            letters.insert(i, letters[i])
    typed = ""
    for ch in letters:
        low = ch.lower()
        base = BASEC.get(low, low)
        if base not in CEN:
            typed += ch
            continue
        cx, cy = CEN[base]
        dx, dy = noise.offset(rng, base, min(2, max(0, int(round(cy / RY)))), sigma)
        key = nearest(cx + dx, cy + dy)
        out = key
        if key == base and low != base and diacritics:
            out = low
        if ch.isupper():
            out = out.upper()
        typed += out
    return typed


def strip_diacritics(text):
    return "".join(BASEC.get(c.lower(), c.lower()).upper() if c.isupper() else BASEC.get(c, c) for c in text)
