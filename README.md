# Romanian Pro — Clink language pack (Phase 1, local build)

A larger, register-balanced Romanian dictionary and next-word model for the Clink iOS keyboard,
aimed at people who type Romanian and English together (Combined mode with Bilingual Guard).

**Status: local build only. Nothing has been published.**

## Assets (`Lexicons/`)

| File | What | Licence |
|---|---|---|
| `ro.clex` | dictionary, built by the official `build-pack.py` from `source/ro.txt` | CC BY-SA 4.0 |
| `ro.cngm` | next-word model (CNGM v1, official format) | CC BY-SA 4.0 |
| `ro.emoji.json` | official Clink Romanian emoji metadata, copied unchanged | Clink Community Assets License 1.0 |

No neural model (`.mlmodelc` + `.bpevocab`): training needs macOS/Core ML tools. The official
pack's neural model is trained with its own vocabulary and is not mixed in here.

## How it is built

See `tools/build_all.sh` (full pipeline), `SOURCES.md` (sources, licences, attribution) and
`LOG.md` (decisions, numbers, problems). In short:

1. **Normalise** every source: NFC, cedilla ş/ţ → comma-below ș/ț, markup stripped.
   The official builder lower-cases every word, so proper nouns are stored in lower case like in
   every official pack (case is only used to recognise names during filtering).
2. **Word list** (`tools/wordlist.py`): weighted mix of relative frequencies from
   OpenSubtitles/FrequencyWords (spoken), Wikipedia articles (formal), Wikipedia discussions
   (informal written), Tatoeba (conversational), Common Voice and small Wikimedia projects;
   junk/noise filter (Hunspell ro_RO used only as a build-time yes/no check); the diacritic rules
   of `clean_ro_pack.py` (drop dominated ASCII spellings such as dupa/stiu/multumesc, keep sa,
   ca, fata, pana, ma); size from a words-vs-coverage curve.
3. **Next-word model** (`tools/ngrams.py`): word pairs from all training sentences, mixed per
   source, pruned per previous word; weights and pruning tuned on dev sets only. Serialised by the
   same code that reproduces the official `build-next-word.py` output byte for byte.
4. **Validate** with the official `vendor/tools/validate-pack.py`.

Held-out data (Tatoeba test split, held-out Wikipedia articles and discussion pages, Common Voice
test sentences) is never trained on: any training sentence whose near-duplicate key matches a
dev/test sentence of any source is removed.

## Evaluation

See `EVAL.md`.
