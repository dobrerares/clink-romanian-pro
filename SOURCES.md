# Sources

Every input was checked against the licence statement on the source's own page (2026-11-23).
Raw downloads live in `data/` (git-ignored); `data/MANIFEST.json` records URL, byte count,
SHA-256 (and the publisher's SHA-1 where one is published) and the licence of every file.
Only derived data ships: lower-cased word counts (`source/ro.txt` → `Lexicons/ro.clex`) and
word-pair probabilities (`Lexicons/ro.cngm`). No sentence text is shipped.

Phase 2 (branch `neural`) adds a neural next-token model: network weights and a 12,036-token
subword vocabulary learned from the *training split* of the same licensed sentence text
(Wikipedia articles and discussions, Wikiquote/Wikivoyage/Wikibooks, Wikinews, Tatoeba, Common Voice;
not FrequencyWords, which has no sentences). It ships as `Lexicons/<code>.mlmodelc` (compiled from
`neural/ro.mlpackage`) + `Lexicons/<code>.bpevocab`, under CC BY-SA 4.0 like the other derived files.
No sentence text is shipped; `source/ro_neural-corpus.json` records the exact training-file hash,
counts and filters (`LOG-phase2.md`). The official Clink trainer `tools/train-neural.py` is copied
unchanged into `vendor/tools/` (Clink Community Assets License 1.0).

Because FrequencyWords and the Wikimedia text are CC BY-SA, `source/ro.txt`, `Lexicons/ro.clex`
and `Lexicons/ro.cngm` are released under **CC BY-SA 4.0**. `Lexicons/ro.emoji.json` and the
copied builder tools in `vendor/` keep their own licence (Clink Community Assets License 1.0).

## Accepted

| Source | URL | Licence (as stated by the source) | Used for | Attribution |
|---|---|---|---|---|
| Romanian Wikipedia, dump 2026-11-01 (articles, ns 0) | https://dumps.wikimedia.org/rowiki/20261101/ (pages-meta-current via the official mirror https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/rowiki/20261101/, SHA-1 verified) | CC BY-SA 4.0 and GFDL (https://dumps.wikimedia.org/legal.html; Wikimedia Terms of Use §7) | Formal-register word counts; word pairs | "Contains text from Romanian Wikipedia (ro.wikipedia.org), © Wikipedia contributors, CC BY-SA 4.0. Changes: text extracted from wikitext, normalised, reduced to word and word-pair statistics." |
| Romanian Wikipedia discussion pages (talk namespaces + project namespace 4), same dump | as above | CC BY-SA 4.0 and GFDL | Informal written register: word counts; word pairs | "Contains text from Romanian Wikipedia discussion pages, © Wikipedia contributors, CC BY-SA 4.0 (changes as above)." |
| Romanian Wikiquote, Wikivoyage, Wikibooks, dumps 2026-11-01 (ns 0) | https://dumps.wikimedia.org/rowikiquote/20261101/ etc. | CC BY-SA 4.0 and GFDL | Word counts; word pairs (small) | "Contains text from Romanian Wikiquote, Wikivoyage and Wikibooks, © their contributors, CC BY-SA 4.0." |
| Romanian Wikinews, dump 2026-11-01 (ns 0) | https://dumps.wikimedia.org/rowikinews/20261101/ | CC BY 2.5 ("all Wikinews textual content is licensed under the Creative Commons Attribution 2.5 License", dumps legal page) | Word counts; word pairs (small) | "Contains text from Romanian Wikinews, © Wikinews contributors, CC BY 2.5." |
| FrequencyWords `content/2018/ro/ro_full.txt` (Hermit Dave; from OpenSubtitles 2018 via OPUS) | https://github.com/hermitdave/FrequencyWords | README: "CC-by-sa-4.0 for content", MIT for code | Spoken/informal word frequencies (no text, counts only) | "Word frequencies from FrequencyWords by Hermit Dave (https://github.com/hermitdave/FrequencyWords), CC BY-SA 4.0, derived from OpenSubtitles 2018 (opus.nlpl.eu; P. Lison and J. Tiedemann, LREC 2016)." |
| Tatoeba Romanian sentences (export `ron_sentences.tsv.bz2`) | https://tatoeba.org/en/downloads | CC BY 2.0 FR ("These files are released under CC BY 2.0 FR"; Romanian is not in the CC0 subset) | Conversational word counts; word pairs (train split only); held-out test set | "Contains sentences from Tatoeba (https://tatoeba.org), CC BY 2.0 FR, by the Tatoeba contributors." |
| Mozilla Common Voice Romanian sentences `server/data/ro/sentence-collector.txt` | https://github.com/common-voice/common-voice/tree/main/server/data/ro | CC0 (README: sentence text "released under a CC0 public domain Creative Commons license"; repo code MPL-2.0) | Short read-speech sentences: word counts, word pairs (train split); held-out test set | Not required (CC0). Courtesy: "Sentences from Mozilla Common Voice (Ardila et al., LREC 2020)." Caveat: the sentences were contributed through the Sentence Collector under a public-domain declaration; many look Europarl-derived. We rely on Mozilla's CC0 statement for this file. |
| Clink official `Lexicons/ro.emoji.json` (anti-ltd/clink-language-packs, SHA-256 03b8b9cf…cb6) | /home/rdobre/projects/clink/upstream/clink-language-packs | Clink Community Assets License 1.0 (© 2026 Anti Limited and contributors) | Copied unchanged as `Lexicons/ro.emoji.json` | "ro.emoji.json © Anti Limited and contributors, from https://github.com/anti-ltd/clink-language-packs, Clink Community Assets License 1.0, unchanged." |
| Clink builder tools (`build-pack.py`, `tools/build-next-word.py`, `tools/validate-pack.py`, `tools/train-neural.py`, `requirements-train.txt`) | same repository | Clink Community Assets License 1.0 | Copied unchanged into `vendor/`; used to build/validate; the neural model class, tokenizer and Core ML conversion are taken from `train-neural.py` | as above |
| Neural model training (Phase 2) | — | — | The text sources above (training split only, Wikimedia + Tatoeba + Common Voice; not FrequencyWords) | Attribution as for each source; the trained weights are CC BY-SA 4.0 |

## Build-time only (never shipped)

| Source | Licence | Use |
|---|---|---|
| Hunspell ro_RO (rospell / LibreOffice dictionaries `ro`), copy from autocorrect-engine/data/hunspell | GPL 2.0 / LGPL 2.1 / MPL 1.1 tri-licence ("Copyright (C) 2005-2013 Rospell Team") | Yes/no validity checks during filtering and diacritic cleaning. No Hunspell word or affix data is copied into the pack. |
| Tatoeba English sentences (`eng_sentences.tsv.bz2`) | CC BY 2.0 FR | Evaluation only: English test-split sentences for the Combined-mode check. |
| Clink official `en.clex` | Clink Community Assets License 1.0 / CC BY-SA 4.0 (FrequencyWords) | Evaluation only (Combined mode) and to recognise English-spelled tokens while filtering. Nothing copied. |

## Excluded

| Source | Why |
|---|---|
| MOROCO (Moldavian and Romanian Dialectal Corpus) | CC BY-NC-SA 4.0: non-commercial. |
| OSCAR (2301 and earlier) | CC0 covers only packaging/metadata/annotations; "We do not own any of the text"; access suspended; described as research-only. |
| CC-100 | No licence for the text ("No claims of intellectual property are made on the work of preparation of the corpus"); Common Crawl web text with third-party copyright. |
| Leipzig Corpora Collection | Terms page could not be read (bot-check page); licence not verifiable → excluded by rule. |
| Reddit / social media | Platform terms restrict redistribution; not attempted. |
| Romanian Wikisource | Licence is fine (CC BY-SA / public domain) and it was downloaded, but it is mostly pre-1993 orthography (sînt, cînd, î inside words) and archaic literary language, which would add obsolete spellings to a keyboard dictionary. Not used. |
| OPUS OpenSubtitles raw text | No explicit licence for the text; only the CC BY-SA FrequencyWords counts are used. |
