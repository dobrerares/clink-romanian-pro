"""Shared helpers for the Romanian Pro build and evaluation.

Normalisation (applied to every source before anything is counted):
  * Unicode NFC;
  * cedilla ş ţ Ş Ţ -> comma-below ș ț Ș Ț (the Romanian standard; same word either way);
  * look-alike hyphens/apostrophes folded to ASCII '-' and "'".

Splits (deterministic, never trained on dev/test):
  * Wikipedia / Wikimedia projects: by document (page id) hash, see split_doc().
  * Tatoeba: by sentence id, exactly as autocorrect-engine/tools/corpus.py
    (bucket 0-79 train, 80-89 dev, 90-99 test) with its near-duplicate rule.
  * Common Voice: by hash of the near-duplicate key of the sentence.
Near-duplicates across sources are removed from training by tools/heldout.py (any training
sentence whose dedup_key() equals that of a dev/test sentence of any source is skipped).
"""
import hashlib
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
WORK = ROOT / "work"
SENT = WORK / "sent"

CEDILLA = str.maketrans({"ş": "ș", "ţ": "ț", "Ş": "Ș", "Ţ": "Ț",
                         "‐": "-", "‑": "-", "­": "",
                         "’": "'", "ʼ": "'"})


def norm(text):
    return unicodedata.normalize("NFC", text).translate(CEDILLA)


FOLD = str.maketrans("ăâîșțĂÂÎȘȚ", "aaistAAIST")


def fold(word):
    return word.translate(FOLD)


RO_LOWER = "abcdefghijklmnopqrstuvwxyzăâîșț"
DIAC = set("ăâîșț")
# A dictionary word: Romanian/English letters, optionally joined by internal hyphens
# (într-o, s-a, dă-mi, nord-vest, e-mail). Digits, apostrophes and other scripts are rejected.
WORD_OK = re.compile(r"[a-zăâîșț]+(?:-[a-zăâîșț]+)*")
# Tokeniser for counting: maximal runs of letters (any script) with internal hyphens/apostrophes,
# so that 'nr.5', 'abc123' and 'http' fragments are split off and then rejected by WORD_OK.
TOKEN = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")


def tokens(text):
    return TOKEN.findall(text)


def dedup_key(text):
    """Near-duplicate key: lower-case, diacritics folded, punctuation and spacing dropped
    (same rule as autocorrect-engine/tools/corpus.py)."""
    return " ".join(re.findall(r"[a-z0-9']+", fold(text.lower())))


def key_hash(text):
    return int.from_bytes(hashlib.blake2b(dedup_key(text).encode(), digest_size=8).digest(), "little")


def bucket(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % 100


def split_tatoeba(sid):
    b = bucket(sid)
    return "train" if b < 80 else "dev" if b < 90 else "test"


def split_doc(source, doc_id):
    """Wikimedia documents: 94% train, 3% dev, 3% test, by page id."""
    b = bucket(f"{source}:{doc_id}")
    return "train" if b < 94 else "dev" if b < 97 else "test"


def split_sentence(source, text):
    """Sentence-level sources without stable ids (Common Voice): 80/10/10 on the dedup key."""
    b = bucket(f"{source}:{dedup_key(text)}")
    return "train" if b < 80 else "dev" if b < 90 else "test"


def hunspell_valid(words, dictionary=DATA / "hunspell" / "ro_RO"):
    """The words Hunspell ro_RO accepts. BUILD-TIME CHECK ONLY: nothing from Hunspell is shipped."""
    import subprocess
    words = sorted(set(words))
    if not words:
        return set()
    res = subprocess.run(["hunspell", "-d", str(dictionary), "-G"], input="\n".join(words) + "\n",
                         capture_output=True, text=True, check=True)
    return set(res.stdout.split())
