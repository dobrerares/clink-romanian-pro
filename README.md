# Romanian Pro — Clink language pack

A larger, register-balanced Romanian dictionary and next-word model for the Clink iOS keyboard,
aimed at people who type Romanian and English together (Combined mode with Bilingual Guard).

**Status:** Phase 1 (dictionary + next-word model) released. Not yet checked on a device; a neural
next-word model (Phase 2) is in progress.

## Install in Clink

1. In Clink, add this repository as a language repository: `https://github.com/dobrerares/clink-romanian-pro`.
2. If you use `dobrerares/clink-romanian-language`, remove it: both ship the same codes
   (`ro`, plus the identical fallbacks `ro_ro`, `mo` and `rup`, because Clink may hide a custom pack
   whose code collides with its official `ro`).
3. Turn on Romanian (and English for Combined mode). Pairs well with the Bilingual Guard plugin
   from `dobrerares/clink-typing-plugins`.

## Assets (`Lexicons/`)

| File | Size | What | Licence |
|---|---|---|---|
| `ro.clex` | 2,150,727 B | 150,000-word dictionary, built by the official `build-pack.py` from `source/ro.txt` | CC BY-SA 4.0 |
| `ro.cngm` | 2,738,604 B | next-word model, 304,288 pairs for 100,156 previous words (CNGM v1, official format) | CC BY-SA 4.0 |
| `ro.emoji.json` | 325,321 B | official Clink Romanian emoji metadata, copied unchanged | Clink Community Assets License 1.0 |

`python3 vendor/tools/validate-pack.py ro` → `ro: looks ready for release.`
SHA-256: `ro.clex` 4d08a8b1…4c4121, `ro.cngm` 9cc0af4b…28ca9be, `ro.emoji.json` 03b8b9cf…8878cb6
(identical to the official file), `source/ro.txt` 18b9a8ad…44c69.
See `Lexicons/LICENSE` for attribution. No neural model (`.mlmodelc` + `.bpevocab`): training
needs macOS/Core ML tools; the official pack's neural model belongs with its own vocabulary and is
not mixed in.

## Results in one paragraph

On held-out text (never trained on), compared with the official `ro` pack and the published
dobrerares pack, using an offline dictionary-corrector **proxy** (Clink's engine is closed):
dictionary coverage 99.4% / 95.6% / 94.3% / 99.1% on Tatoeba / Wikipedia / Wikipedia discussions /
Common Voice (official 96.3 / 81.0 / 87.0 / 93.8; published 94.6 / 88.6 / 90.0 / 95.9); typo-proxy
word accuracy higher than both baselines on every Romanian set, noise model and typing mode
(e.g. 80.2% vs 69.6% / 79.2% typing Tatoeba sentences without diacritics in Combined mode) with fewer
wrong changes and no change for English; next-word top-3 24.7-26.4% vs 12.8-21.3% on Wikipedia,
discussions and Common Voice. On Tatoeba both baselines trained on the test sentences; against a
decontaminated rebuild of the published model Romanian Pro is level (26.2% vs 25.6%). Full tables
with 95% confidence intervals: **`EVAL.md`**.

## How it is built

`tools/build_all.sh` runs the whole pipeline; `SOURCES.md` lists every source, licence and
attribution; `LOG.md` records each decision with its numbers.

1. **Normalise** every source: NFC, cedilla ş/ţ → comma-below ș/ț, wiki markup stripped, Romanian
   sentence split. The official builder lower-cases every word, so proper nouns are stored in lower
   case like in every official pack (case is only used while filtering).
2. **Split** held-out data first: Tatoeba by sentence id (same split as autocorrect-engine),
   Wikimedia by page id (94/3/3), Common Voice by sentence hash; any training sentence that is a
   near-duplicate of a dev/test sentence of any source is removed.
3. **Word list** (`tools/wordlist.py`): mixed relative frequencies — FrequencyWords/OpenSubtitles
   0.45 (spoken), Wikipedia articles 0.27 (formal), Wikipedia discussions 0.10 (informal written),
   Tatoeba 0.08, Common Voice 0.03, other Wikimedia projects 0.07; evidence filter against noise
   (Hunspell ro_RO as a build-time yes/no check only); the `clean_ro_pack.py` diacritic rules (drop
   dominated ASCII spellings such as dupa/stiu/multumesc/asa/maine, keep sa/ca/fata/pana/ma),
   extended to partially-diacritised and pre-1993 î-spellings and to hyphenated clitic forms;
   English loanwords enter only through their frequency in Romanian text (ok, weekend, laptop,
   e-mail, site, like, deadline, update, meeting…). **150,000 words**, chosen on dev from the
   coverage curve and a typo-proxy size sweep (conversational accuracy peaks at 100-150k).
4. **Next-word model** (`tools/ngrams.py`): word pairs (never across sentence ends, removed markup,
   numbers or out-of-dictionary tokens), mixed per source (Wikipedia .37, discussions .22,
   Tatoeba .28, Common Voice .03, other .10), up to 256 followers per word with ≥ 50 occurrences
   (≥ 5 for a word's top three), corpus artifacts demoted (Tatoeba's Tom/Mary, Wikipedia
   self-reference). All settings tuned on dev. Serialised by the code path that reproduces the
   official `build-next-word.py` output byte for byte.
5. **Validate** with the official `vendor/tools/validate-pack.py`.

## Known limitations / Phase 2 candidates

- Templated Wikipedia stubs still bias a few predictions (`o → comună, suprafață`).
- `sa ca fata pana ma` are kept as typed (standing rule), so plain typing never restores
  să/că/fată/până/mă from the dictionary alone; context (Bilingual Guard / next-word) must do it.
- No `CBOS` sentence-start section (semantics in the app undocumented) and no neural model.
- Proper nouns are ~27% of the list (lower-cased); OpenSubtitles contributes English names.
