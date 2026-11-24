# Romanian Pro — Clink language pack

A larger, register-balanced Romanian dictionary and next-word model for the Clink iOS keyboard,
aimed at people who type Romanian and English together (Combined mode with Bilingual Guard).

**Status:** Phase 1 (dictionary + next-word model) released. Phase 2 (branch `neural`): a neural
next-word model, trained and evaluated (it beats Phase 1 on every held-out set, see
**`EVAL-phase2.md`**); it ships after one macOS compile step (`.github/workflows/compile-neural.yml`
→ `tools/install-neural.sh`). Not yet checked on a device.

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
| `ro.mlmodelc/` + `ro.bpevocab` | ≈ 2.4 MB + 241,855 B | neural next-word model (official Clink architecture, int8) and its BPE vocabulary; compiled on macOS from `neural/ro.mlpackage` and installed by `tools/install-neural.sh` (not in `Lexicons/` until then) | CC BY-SA 4.0 |

`python3 vendor/tools/validate-pack.py ro` → `ro: looks ready for release.`
SHA-256: `ro.clex` 4d08a8b1…4c4121, `ro.cngm` 9cc0af4b…28ca9be, `ro.emoji.json` 03b8b9cf…8878cb6
(identical to the official file), `source/ro.txt` 18b9a8ad…44c69.
See `Lexicons/LICENSE` for attribution. The neural model is our own (trained here with the official
trainer's model, tokenizer and Core ML conversion); the official pack's neural model belongs with its
own vocabulary and is not mixed in. `ro_ro`, `mo` and `rup` get identical copies of every file.

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

Phase 2, same next-word protocol and held-out sets: the neural model raises top-3 from 26.4% to
35.3% on Wikipedia discussions, 26.2% → 31.8% on Tatoeba, 26.2% → 32.8% on Common Voice and
24.7% → 27.0% on Wikipedia articles; combined with the Phase 1 next-word model (interpolation chosen
on dev) 35.8 / 32.8 / 33.5 / 29.2%, and 53.2-62.0% after the first letter (Phase 1: 47.5-53.5%). Every
difference is beyond its 95% interval; int8 quantization costs nothing measurable.
**`EVAL-phase2.md`**.

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
6. **Neural model** (Phase 2, `tools/neural/build_neural.sh`, log in `LOG-phase2.md`): the same
   training split, filtered for the neural model (2.72M sentences, 47.0M words; diacritic-less text,
   numbers/markup fragments, foreign scripts and templated stubs removed), weighted toward
   conversation (discussions ×3, Tatoeba ×20 per epoch, Wikipedia ×0.5), Phase 1's artifact demotion as
   a loss weight; official `GRULM` (2.28M parameters, 12,036-token BPE learned with the official
   algorithm) trained on CPU for 145M targets; exported with the official Core ML conversion
   (`neural/ro.mlpackage`, int8). Compiled on a GitHub macOS runner and checked against PyTorch
   reference outputs (`neural/fixtures.json`) before installation.

## Known limitations / Phase 2 candidates

- Templated Wikipedia stubs still bias a few predictions (`o → comună, suprafață`).
- `sa ca fata pana ma` are kept as typed (standing rule), so plain typing never restores
  să/că/fată/până/mă from the dictionary alone; context (Bilingual Guard / next-word) must do it.
- No `CBOS` sentence-start section (semantics in the app undocumented).
- Neural model: like the official one it ignores hyphenated words (s-a, într-o, nu-mi) in its
  context and cannot suggest them (the `.cngm` can); how Clink combines it with the `.cngm` is not
  public, so `EVAL-phase2.md` reports both alone and an interpolation. The release manifest has no
  `neural` metadata block because the 348 MB training-sentence file is not committed
  (`source/ro_neural-corpus.json` describes it), as for the official `ro` pack.
- Proper nouns are ~27% of the list (lower-cased); OpenSubtitles contributes English names.
