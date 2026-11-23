#!/usr/bin/env python3
"""Extract clean Romanian sentences from a MediaWiki XML dump (.xml.bz2).

Usage: extract_wiki.py DUMP.xml.bz2 SOURCE --ns 0 [--ns 1 ...] [--workers 7] [--limit N]

Writes work/sent/<SOURCE>.<split>.tsv: "<page id>\t<sentence>" (split by page-id hash,
common.split_doc) and work/sent/<SOURCE>.stats.json. Redirects and disambiguation pages are
skipped. Markup removed: comments, <ref>, templates {{..}}, tables {|..|}, files/categories/
interwiki/user links, external links (label kept), bold/italic quotes, HTML tags, magic words,
talk-page signatures and timestamps. Each sentence is normalised (NFC, comma-below ș/ț).
Only sentences that look like Romanian prose are kept (sentence_ok()).
"""
import argparse
import bz2
import html
import json
import multiprocessing as mp
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SENT, norm, split_doc, tokens  # noqa: E402

DROP_NS = ("fișier", "fişier", "file", "imagine", "image", "media", "categorie", "category",
           "utilizator", "user", "discuție utilizator", "discuţie utilizator", "user talk",
           "special", "format", "template", "modul", "module", "s", "w", "wikt", "q", "n", "b", "v",
           "commons", "d", "wikidata", "meta", "m", "mw", "species", "c", "voy", "foundation", "phab")
INTERWIKI = re.compile(r"^:?[a-z]{2,3}(?:-[a-z]{2,8})*:", re.I)
BLOCK_TAGS = ("ref", "math", "chem", "code", "pre", "gallery", "timeline", "syntaxhighlight", "source",
              "score", "graph", "hiero", "imagemap", "references", "templatedata", "mapframe",
              "maplink", "poem", "nowiki", "small", "sup", "sub", "table", "ce", "inputbox", "categorytree")
RE_COMMENT = re.compile(r"<!--.*?(?:-->|$)", re.S)
RE_REF_SELF = re.compile(r"<ref[^>]*/\s*>", re.I)
RE_BLOCK = re.compile(r"<(" + "|".join(BLOCK_TAGS) + r")\b[^>]*>.*?</\1\s*>", re.S | re.I)
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_PARAM = re.compile(r"\{\{\{[^{}]*\}\}\}")
RE_TABLE = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.S)
RE_LINK = re.compile(r"\[\[([^\[\]]*)\]\]")
RE_EXT = re.compile(r"\[(?:https?:|ftp:)?//[^\s\]]+\s*([^\]]*)\]")
RE_URL = re.compile(r"(?:https?|ftp)://\S+|www\.\S+")
RE_QUOTES = re.compile(r"'{2,}")
RE_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
RE_MAGIC = re.compile(r"__[A-ZĂÂÎȘȚ_]+__")
# talk-page signature timestamp: 12:34, 5 mai 2020 (EEST)
RE_SIG = re.compile(r"(?:\(\s*(?:discuție|discuţie|talk|contribuții|mesaje?)\s*\)\s*)?"
                    r"(?:\d{1,2}:\d{2},? \d{1,2} [a-zăâîșțA-ZĂÂÎȘȚ]+\.? \d{4}|\d{1,2} [a-zăâîșțA-ZĂÂÎȘȚ]+\.? \d{4},? \d{1,2}:\d{2})"
                    r"\s*\((?:EET|EEST|UTC|CET|CEST|ora României)\)", re.I)
RE_TALKBITS = re.compile(r"\(\s*(?:discuție|discuţie|talk|contribuții)\s*\)", re.I)
# Removed inline material (templates, <ref>, <small>...) leaves a gap mark: word pairs never
# span it (tools/ngrams.py), and dev/test sentences containing it are discarded.
GAP = " ¦ "
RE_GAPS = re.compile(r"(?:\s*¦\s*)+")
RE_SPACE = re.compile(r"[ \t   ]+")
# sentence boundary candidates
RE_BOUND = re.compile(r"([.!?…]+)([\"”»)\]]*)\s+(?=[\"„«(\[]?[A-ZĂÂÎȘȚ0-9])")
ABBR = {"d", "dl", "dna", "dnei", "dlui", "dr", "prof", "nr", "art", "alin", "lit", "pct", "etc", "sec", "sf",
        "st", "str", "bd", "bul", "jud", "mun", "com", "p", "pp", "vol", "ed", "cap", "fig", "tab", "ex", "cca",
        "aprox", "mil", "mld", "km", "kg", "cm", "mm", "h", "min", "ian", "feb", "febr", "mar", "apr", "aug",
        "sept", "oct", "nov", "dec", "gen", "lt", "col", "mr", "mrs", "jr", "sr", "dvs", "hr", "chr", "op",
        "cf", "vs", "ș.a", "î.hr", "d.hr", "a.c", "ș.u", "prof.univ", "conf", "ing", "pr", "ep", "sfânta",
        "ss", "nr.crt", "resp", "id", "ibid", "loc", "cit", "trad", "coord", "red", "sg", "pl", "masc", "fem"}
RO_MARKERS = {"și", "de", "în", "la", "cu", "pe", "nu", "este", "sunt", "o", "un", "care", "că", "să", "din",
              "pentru", "mai", "se", "ce", "am", "ai", "fost", "sau", "dar", "era", "lui", "ei", "el", "ea", "eu",
              "tu", "noi", "voi", "le", "îi", "mă", "te", "ne", "vă", "al", "ale", "cel", "cea", "acest",
              "această", "foarte", "doar", "fi", "fără", "după", "până", "prin", "despre", "către", "între",
              "iar", "dacă", "când", "unde", "cum", "bine", "da", "mersi", "salut", "bună", "unei", "unui",
              "acum", "aici", "asta", "ăsta", "deci", "chiar", "poate", "trebuie", "vreau", "cred", "mulțumesc"}
EN_MARKERS = {"the", "and", "of", "to", "is", "was", "for", "that", "with", "this", "are", "you", "it",
              "be", "on", "by", "from", "have", "not", "which", "an", "or", "at", "his", "they"}
DISAMBIG = re.compile(r"\{\{\s*(?:dezambiguizare|dez|disambig|hndis|nume de familie)\b", re.I)


def strip_templates(text):
    for _ in range(12):
        new = RE_TEMPLATE.sub(GAP, RE_PARAM.sub(GAP, text))
        if new == text:
            break
        text = new
    return text


def link_text(m):
    inner = m.group(1)
    target, _, label = inner.partition("|")
    t = target.strip()
    low = t.lower().lstrip(":").replace("_", " ")
    if ":" in low:
        prefix = low.split(":", 1)[0].strip()
        if prefix in DROP_NS or INTERWIKI.match(t):
            return " "
    if label:
        return label.rsplit("|", 1)[-1]
    return t.split("#", 1)[0] if not t.startswith("#") else t[1:]


def clean_wikitext(text):
    text = RE_COMMENT.sub(" ", text)
    text = RE_REF_SELF.sub(" ", text)
    text = RE_BLOCK.sub(GAP, text)
    text = strip_templates(text)
    for _ in range(6):
        new = RE_TABLE.sub(" ", text)
        if new == text:
            break
        text = new
    for _ in range(6):
        new = RE_LINK.sub(link_text, text)
        if new == text:
            break
        text = new
    text = RE_EXT.sub(r"\1", text)
    text = RE_URL.sub(" ", text)
    text = RE_QUOTES.sub("", text)
    text = RE_TAG.sub(" ", text)
    text = RE_MAGIC.sub(" ", text)
    text = html.unescape(text)
    text = RE_SIG.sub(" ", text)
    text = RE_TALKBITS.sub(" ", text)
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s or s[0] in "|!{}=" or s.startswith("[[") or s.startswith("----"):
            continue
        s = RE_GAPS.sub(GAP, s.lstrip("*#:; ")).strip(" ¦")
        if s:
            out.append(RE_SPACE.sub(" ", s))
    return out


def split_sentences(par):
    out, start = [], 0
    for m in RE_BOUND.finditer(par):
        end = m.end(2)
        if m.group(1) == ".":
            before = par[start:m.start(1)].rsplit(" ", 1)[-1].lstrip("(\"„«[").lower()
            if len(before) == 1 or before in ABBR:
                continue
        out.append(par[start:end].strip(" ¦"))
        start = m.end()
    tail = par[start:].strip(" ¦")
    if tail:
        out.append(tail)
    return out


BAD_CHARS = set("{}|[]<>=\\_^~@$")
_RO_VOCAB = None


def ro_vocab():
    """Romanian reference vocabulary for the language check: the 30,000 most frequent
    FrequencyWords (OpenSubtitles) words, minus English closed-class markers."""
    global _RO_VOCAB
    if _RO_VOCAB is None:
        from common import WORK
        vocab = set()
        with open(WORK / "counts" / "freqwords.tsv", encoding="utf-8") as fh:
            next(fh)
            for i, line in enumerate(fh):
                if i >= 30000:
                    break
                vocab.add(line.split("\t", 1)[0])
        _RO_VOCAB = (vocab | RO_MARKERS) - EN_MARKERS
    return _RO_VOCAB


def sentence_ok(s, require_end=True):
    if len(s) < 8 or len(s) > 1000:
        return False
    if any(c in BAD_CHARS for c in s) or "://" in s:
        return False
    if require_end and s[-1] not in ".!?…\"”»)":
        return False
    nonspace = [c for c in s if not c.isspace()]
    letters = sum(c.isalpha() for c in nonspace)
    if letters < 0.65 * len(nonspace):
        return False
    toks = [t.lower() for t in tokens(s)]
    if len(toks) < 3:
        return False
    if any(not ("a" <= c <= "z" or c in "ăâîșț-'") for t in toks for c in t):
        # other scripts or foreign accented letters: allow a little (names), not a lot
        foreign = sum(1 for t in toks if any(not ("a" <= c <= "z" or c in "ăâîșț-'") for c in t))
        if foreign > max(1, len(toks) // 6):
            return False
    # Romanian prose: at least 40% of the words are common Romanian words (or carry Romanian
    # diacritics), and English closed-class words stay rare.
    vocab = ro_vocab()
    ro = sum(1 for t in toks if t in vocab or not DIAC_SET.isdisjoint(t))
    en = sum(t in EN_MARKERS for t in toks)
    if ro < 0.4 * len(toks) or en >= max(2, 0.2 * len(toks)):
        return False
    return True


DIAC_SET = set("ăâîșț")


def process(job):
    source, ns, pid, title, text, talk = job
    if DISAMBIG.search(text[:4000]) and not talk:
        return pid, None, [], "disambig"
    split = split_doc(source, pid)
    text = norm(html.unescape(text)) if "&" in text else norm(text)
    sents = []
    for par in clean_wikitext(text):
        for s in split_sentences(par):
            if sentence_ok(s, require_end=not talk):
                sents.append(s)
    return pid, split, sents, None


RE_PAGE_FIELD = re.compile(r"<(title|ns|id)>(.*?)</\1>")


def pages(path, namespaces, limit):
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        n = 0
        title = ns = pid = None
        redirect = False
        in_text = False
        buf = []
        for line in fh:
            if in_text:
                end = line.find("</text>")
                if end >= 0:
                    buf.append(line[:end])
                    in_text = False
                    if ns in namespaces and not redirect:
                        yield ns, pid, title, "".join(buf)
                        n += 1
                        if limit and n >= limit:
                            return
                    buf = []
                else:
                    buf.append(line)
                continue
            s = line.strip()
            if s == "<page>":
                title = ns = pid = None
                redirect = False
            elif s.startswith("<title>") or s.startswith("<ns>") or (s.startswith("<id>") and pid is None):
                m = RE_PAGE_FIELD.match(s)
                if m:
                    k, v = m.groups()
                    if k == "title":
                        title = html.unescape(v)
                    elif k == "ns":
                        ns = int(v)
                    else:
                        pid = int(v)
            elif s.startswith("<redirect"):
                redirect = True
            elif s.startswith("<text"):
                gt = line.find(">")
                if line[gt - 1] == "/":  # <text ... />
                    continue
                rest = line[gt + 1:]
                end = rest.find("</text>")
                if end >= 0:
                    if ns in namespaces and not redirect:
                        yield ns, pid, title, rest[:end]
                        n += 1
                        if limit and n >= limit:
                            return
                else:
                    in_text = True
                    buf = [rest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("source")
    ap.add_argument("--ns", type=int, action="append", required=True)
    ap.add_argument("--talk", action="store_true", help="talk/discussion pages: sentence end not required")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    SENT.mkdir(parents=True, exist_ok=True)
    outs = {sp: open(SENT / f"{args.source}.{sp}.tsv", "w", encoding="utf-8") for sp in ("train", "dev", "test")}
    stats = Counter()
    from xml.sax.saxutils import unescape

    def jobs():
        for ns, pid, title, text in pages(args.dump, set(args.ns), args.limit):
            yield args.source, ns, pid, title, unescape(text), args.talk

    with mp.Pool(args.workers) as pool:
        for pid, split, sents, why in pool.imap_unordered(process, jobs(), chunksize=64):
            stats["pages"] += 1
            if why:
                stats["skipped_" + why] += 1
                continue
            stats[f"pages_{split}"] += 1
            stats[f"sentences_{split}"] += len(sents)
            fh = outs[split]
            for s in sents:
                fh.write(f"{pid}\t{s}\n")
            if stats["pages"] % 50000 == 0:
                print(json.dumps(stats), flush=True)
    for fh in outs.values():
        fh.close()
    stats_out = dict(stats, dump=str(args.dump), namespaces=args.ns)
    (SENT / f"{args.source}.stats.json").write_text(json.dumps(stats_out, indent=1) + "\n")
    print(json.dumps(stats_out))


if __name__ == "__main__":
    main()
