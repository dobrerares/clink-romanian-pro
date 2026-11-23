#!/bin/sh
# Romanian Pro: full rebuild from the raw downloads recorded in data/MANIFEST.json.
# Run from the repository root. Downloads (tools/fetch.py) are listed in LOG.md.
set -eu
PY="python3 -I"
D=data

# 1. text extraction + splits (work/sent)
$PY tools/extract_wiki.py $D/wikipedia/rowiki-20261101-pages-articles.xml.bz2 wiki --ns 0
$PY tools/extract_wiki.py $D/wikipedia/rowiki-20261101-pages-meta-current.xml.bz2 wikitalk --talk \
    --ns 1 --ns 3 --ns 4 --ns 5 --ns 7 --ns 11 --ns 13 --ns 15 --ns 101 --ns 103
for w in wikinews wikiquote wikivoyage wikibooks; do
    $PY tools/extract_wiki.py $D/$w/ro$w-20261101-pages-articles.xml.bz2 $w --ns 0
done
$PY tools/extract_sentences.py

# 2. leak-free training text and held-out sets (work/train, work/eval)
$PY tools/heldout.py wiki wikitalk tatoeba cv wikinews wikiquote wikivoyage wikibooks

# 3. word counts and the word list (source/ro.txt)
$PY tools/count.py freqwords wiki wikitalk tatoeba cv wikinews wikiquote wikivoyage wikibooks
$PY tools/wordlist.py --curve --size "${SIZE:-150000}"

# 4. dictionary with the official builder
$PY vendor/build-pack.py ro source/ro.txt

# 5. next-word model: pair counts over the ranked superset, then the tuned configuration
$PY tools/ngrams.py count build/ranked-superset.txt wiki wikitalk tatoeba cv wikinews wikiquote wikivoyage wikibooks
$PY tools/ngrams.py build source/ro.txt Lexicons/ro.cngm --weights "${NGRAM_WEIGHTS}" \
    --top-k "${TOP_K}" --min-count "${MIN_COUNT}" --report build/cngm-report.json

# 6. validate with the official validator
$PY vendor/tools/validate-pack.py ro
