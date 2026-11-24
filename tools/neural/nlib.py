"""Shared helpers for the Phase 2 neural model (tools/neural/*).

The official Clink trainer is vendored UNMODIFIED as vendor/tools/train-neural.py
(anti-ltd/clink-language-packs, SHA-256 19ae4eb4...ce8a). Everything that defines what the
keyboard will see - word cleaning (clean_word), BPE learning semantics (train_bpe), word
encoding (bpe_encode_word), the special tokens, the GRULM model class and the Core ML
conversion settings - is taken from that file, not re-implemented:

  * functions/constants are imported from the module itself (official());
  * the GRULM class is defined inside the official main(), so its source text is extracted with
    `ast` and executed verbatim (official_grulm()).
"""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
NEURAL = WORK / "neural"
OFFICIAL_PATH = ROOT / "vendor" / "tools" / "train-neural.py"
OFFICIAL_SHA256 = "19ae4eb48f3fc1f9b7966a5bcd22b1ea157e6e52d16a240c9bbc3d769d53ce8a"
LANG = "ro"
SOURCES = ("wiki", "wikitalk", "tatoeba", "cv", "wikinews", "wikiquote", "wikivoyage", "wikibooks")
EVAL_SETS = ("tatoeba", "wiki", "wikitalk", "cv")

sys.path.insert(0, str(ROOT / "tools"))

_OFFICIAL = None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def official():
    """The unmodified official trainer as a module (its torch/coremltools imports live in main())."""
    global _OFFICIAL
    if _OFFICIAL is None:
        digest = sha256(OFFICIAL_PATH)
        if digest != OFFICIAL_SHA256:
            raise SystemExit(f"{OFFICIAL_PATH} was modified (sha256 {digest})")
        spec = importlib.util.spec_from_file_location("clink_train_neural", OFFICIAL_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _OFFICIAL = mod
    return _OFFICIAL


def official_grulm(V, eos, dim, hidden, layers):
    """Return the official GRULM class, executed from the trainer's own source text."""
    import torch
    import torch.nn as nn
    from types import SimpleNamespace

    tree = ast.parse(OFFICIAL_PATH.read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    cls = next(n for n in ast.walk(main) if isinstance(n, ast.ClassDef) and n.name == "GRULM")
    src = ast.get_source_segment(OFFICIAL_PATH.read_text(encoding="utf-8"), cls)
    import textwrap
    ns = {"nn": nn, "torch": torch, "V": V, "eos": eos,
          "args": SimpleNamespace(dim=dim, hidden=hidden, layers=layers)}
    exec(compile(textwrap.dedent(src), str(OFFICIAL_PATH), "exec"), ns)
    return ns["GRULM"]


def read_bpevocab(path):
    """Parse an official BPE1 file -> (vocab list, merges list of (a, b))."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    assert lines[0] == "BPE1", path
    n = int(lines[1])
    vocab = lines[2:2 + n]
    m = int(lines[2 + n])
    merges = [tuple(l.split(" ")) for l in lines[3 + n:3 + n + m]]
    assert all(len(x) == 2 for x in merges)
    return vocab, merges


def write_bpevocab(path, vocab, merges):
    """Byte-for-byte the official writer (train-neural.py, 'BPE vocab: format marker ...')."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("BPE1\n")
        f.write(f"{len(vocab)}\n")
        for t in vocab:
            f.write(t + "\n")
        f.write(f"{len(merges)}\n")
        for a, b in merges:
            f.write(f"{a} {b}\n")


class Tokenizer:
    """Official word cleaning + official BPE encoding, with a per-word cache."""

    def __init__(self, bpevocab):
        self.off = official()
        self.vocab, merges = read_bpevocab(bpevocab)
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.unk = self.stoi[self.off.UNK]
        self.bos = self.stoi[self.off.BOS]
        self.eos = self.stoi[self.off.EOS]
        self.V = len(self.vocab)
        self._cache = {}

    def clean(self, text):
        """Exactly iter_sentences(): text.lower().split() -> clean_word -> drop None."""
        return [t for t in (self.off.clean_word(x, LANG) for x in text.strip().lower().split()) if t]

    def word_ids(self, word):
        ids = self._cache.get(word)
        if ids is None:
            ids = [self.stoi.get(t, self.unk) for t in self.off.bpe_encode_word(word, self.merges)]
            self._cache[word] = ids
        return ids

    def encode_words(self, words, bos=True, eos=True):
        ids = [self.bos] if bos else []
        for w in words:
            ids.extend(self.word_ids(w))
        if eos:
            ids.append(self.eos)
        return ids

    def decode_word(self, ids):
        s = "".join(self.vocab[i] for i in ids)
        return s[:-len(self.off.WORD_END)] if s.endswith(self.off.WORD_END) else s


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
