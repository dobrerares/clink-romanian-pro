# Romanian Pro — Phase 1 log

Running log, newest entries at the bottom. Times are UTC.

## 2026-11-23

**20:25 Setup.** Repo existed (git init, no commits) with BRIEF-phase1.md and a .gitignore
(`data/`, `__pycache__/`). `git config user.*` needs interactive approval in this sandbox, so
commits set the identity per command (`git -c user.name=Daedalus -c user.email=daedalus@localhost`).

**20:26 Format study.** Read upstream clink-language-packs README/PROMPT/VERIFY, `build-pack.py`,
`tools/build-next-word.py`, `tools/validate-pack.py`, `tools/build-release-manifest.py`, LICENSE.md.
- CLEX v1 (build-pack.py): words are lower-cased + NFC by the builder, so case is *not* stored;
  proper nouns go in lower case like every official pack (Clink capitalises at runtime).
  Per-word data = one log-probability byte `round((log10 p + 9) * 28)` + a 48-letter transition table.
- CNGM v1 (build-next-word.py): pairs (prev id, next id, byte `round((log10 P(next|prev) + 6) * 42)`),
  grouped by prev id, most frequent first. The official tool has no pruning and no weights, and it
  drops out-of-dictionary tokens *before* pairing, so it creates pairs across removed words.
- No size limit is documented. Reference sizes: official en.clex 837 KB / en.cngm 904 KB (100k pairs);
  official ro.clex 404 KB (29,920 words) / ro.cngm 360 KB (40,000 pairs);
  published dobrerares ro.clex 583 KB (41,630 words) / ro.cngm 785 KB (87,210 pairs).
- Upstream tools and `ro.emoji.json` are under the *Clink Community Assets License 1.0* (Clink-only,
  no sale, share-alike under the same licence). They are copied unchanged into `vendor/` and
  `Lexicons/ro.emoji.json` with that licence; the word/pair data we derive ships under CC BY-SA 4.0.
- Official `ro.*` provenance (upstream Lexicons LICENSE): unigrams FrequencyWords/OpenSubtitles 2018,
  bigrams **all** Tatoeba sentences. The published dobrerares pack's `ro.cngm` is also built from all
  Tatoeba sentences (`source/ro.sentences.txt`). **Both baselines have therefore seen the Tatoeba
  test sentences in their next-word model**; Tatoeba next-word numbers favour them. Wikipedia and
  Common Voice held-out sets are clean for all three packs.
- Official pack also ships `ro.mlmodelc` + `ro.bpevocab` (Core ML neural ranker). It cannot run on
  Linux, so the comparison below covers dictionary + next-word model only.

**20:27-20:31 Downloads** (all recorded in data/MANIFEST.json with SHA-256; Wikimedia files also
checked against the publisher's SHA-1 from dumpstatus.json):
- rowiki 20261101 pages-articles (788 MB) from dumps.wikimedia.org.
- rowiki 20261101 pages-meta-current (1.03 GB): dumps.wikimedia.org returned 404 for it (also the
  `latest/` alias); fetched from the official mirror ftp.acc.umu.se, SHA-1 matches Wikimedia's.
- rowikiquote / rowikinews / rowikivoyage / rowikibooks / rowikisource 20261101 pages-articles
  (their meta-current files also 404 on the main host).
- FrequencyWords `content/2018/ro/ro_full.txt` (1,228,691 types, 332.6M tokens) + LICENSE.
- Common Voice `server/data/ro/sentence-collector.txt` (13,691 lines, CC0) from GitHub raw.
- Local copies (recorded with their original URL): Tatoeba ron export, ro_50k, Hunspell ro_RO.

**20:30 Licences verified** from each source's own page (summary in SOURCES.md):
accepted Wikimedia text (CC BY-SA 4.0/GFDL; Wikinews CC BY 2.5), FrequencyWords content
(CC BY-SA 4.0), Tatoeba (CC BY 2.0 FR; Romanian not in the CC0 subset), Common Voice
sentence-collector text (CC0). Excluded: MOROCO (CC BY-NC-SA, non-commercial), OSCAR (content not
licensed, research-only), CC-100 (no licence for the text), Leipzig (terms page behind a bot check,
licence not verifiable), Reddit/social media (not attempted: platform terms). Hunspell ro_RO
(GPL/LGPL/MPL) is used only as a build-time validity check and never shipped.

**20:33 Splits.** Tatoeba: 33,594 train / 4,152 dev / 4,169 test after 160 near-duplicates; the
dropped-id set is identical to autocorrect-engine's `dedup-ro.txt`, so the test split is the same
one used in earlier Bilingual Guard evals. Common Voice: 10,815 / 1,411 / 1,388 (77 duplicates).
Wikimedia: by page-id hash, 94 / 3 / 3.

**20:35-21:15 Wikitext extraction** (`tools/extract_wiki.py`): stream parse, strip markup
(templates, tables, refs, files/categories/interwiki/user links, external links, quotes, tags,
magic words, talk signatures), NFC + comma-below, Romanian sentence split with abbreviation list.
Removed *inline* markup leaves a gap mark `¦`: word pairs never span it and dev/test sentences
containing it are discarded. First full run stopped and redone after a filter audit on 1,500
articles showed real prose being dropped (`%` treated as markup, sentences > 600 chars, and
Romanian sentences without diacritics failing a function-word language check — e.g.
"Majoritatea evreilor erau negustori."). The language check is now "≥ 40% of tokens are among the
30k most frequent FrequencyWords words or carry ro diacritics, and few English function words";
85% of article tokens survive (the rest: headings/list fragments without final punctuation,
bibliography lines, foreign quotations). Talk-page run also redone: signatures in the
"21 ianuarie 2010 13:23 (EET)" format and `Discuție_utilizator:` (underscore) links were leaking
user names and month names. Articles were re-extracted with the final code from pages-articles
(same 20261101 revisions as meta-current ns 0) so every source uses one extractor version.
- Byte-identity check: `tools/ngrams.py official` (my serialiser, the code path that writes the
  shipped `.cngm`) produces exactly the bytes of upstream `build-next-word.py` on the published
  pack's inputs (sha256 5007202c…c690, 87,210 pairs) — which is also byte-identical to the
  published `ro.cngm`, confirming that file was trained on all Tatoeba sentences.
- Decided not to write the optional CLEX `CBOS` sentence-start section: upstream says it affects
  sentence casing/swipe in the app, and its semantics are not documented well enough to add it
  blind. Phase 2 candidate.
- First commit 6916a30 (tooling + docs).

**21:40 Held-out filter + counts.** `tools/heldout.py` over all 8 text sources. Training sentences
whose near-duplicate key equals any dev/test key are dropped: wiki 243,941, talk 2,755,849 (subst'd
welcome/warning templates repeated over ~1M user-talk pages), Tatoeba 49, others < 500. Kept:
wiki 4,773,682 sentences / 100.3M tokens; talk 1,172,424 / 20.5M; Tatoeba 33,545 / 223k;
Common Voice 10,814 / 78k; Wikinews/-quote/-voyage/-books 11-13k sentences each (~0.86M tokens).
FrequencyWords ro_full: 1,075,681 types / 331.8M tokens after normalisation (cedilla and comma
spellings summed; 153k entries with digits/apostrophes/other scripts rejected).

**21:45-22:05 Word list** (`tools/wordlist.py`). Weights (share of the mixed relative frequency):
FrequencyWords 0.45, Wikipedia 0.27, discussions 0.10, Tatoeba 0.08, Common Voice 0.03, other
Wikimedia 0.07. Evidence filter: ≥ 3 occurrences; Hunspell-valid words pass (Wikipedia-only ones
need 2+ articles); others need 2+ source groups (≥ 5 occurrences), or ≥ 50 in subtitles, or 10+
documents. Diacritic cleaning = clean_ro_pack.py rules/constants (KEEP sa ca fata pana ma), with
evidence from *all* training sentences written with diacritics, plus two extensions: partially
diacritised Hunspell-rejected spellings (făra, îti) and pre-1993 î-spellings (sînt→sunt, cînd→când)
are merged into a Hunspell-valid sibling that is ≥ 2× as frequent.
Bugs found by inspecting dev OOVs and fixed: (1) my markup stoplist contained `doi` (meant the
DOI identifier; it is "two") — the stoplist now only drops Hunspell-invalid words; (2) the
Roman-numeral rule's optional `-a` suffix matched the clitics mi-a, l-a, i-a, m-a; now only `-lea`
and multi-letter x/v numerals with `-a`, plus an exemption list (mix, cd, cm, mm, ml, mi, vi...);
(3) Hunspell rejects almost every hyphenated clitic form, so `să-ti` survived next to `să-ți`:
hyphenated words now count as valid when every part is a Hunspell word or a clitic fragment
(44,575 forms), which lets the diacritic rules clean clitic spellings too; (4) `like` and `no`
were on the English function-word stoplist; removed (dă-i like; "no, bine").
Result before the size cut: 425,572 words after cleaning (20,036 invalid ASCII spellings removed,
1,678 valid-but-dominated, 3,899 partial-diacritic, 1,157 old-orthography; 6,087 valid ASCII words
re-weighted). Largest removals: si→și, in→în, esti, daca, stiu, stii, multumesc, imi, cand, poti,
facut, rau→rău/râu, dupa, viata, sînt→sunt.
Dev coverage curve (share of dev tokens in top-N; `build/coverage-curve.json`):

| N | Tatoeba | Common Voice | discussions | Wikipedia |
|---|---|---|---|---|
| 50k | 98.37 | 98.26 | 91.03 | 90.69 |
| 100k | 99.18 | 99.23 | 92.89 | 93.91 |
| 150k | 99.41 | 99.42 | 93.86 | 95.17 |
| 200k | 99.55 | 99.49 | 94.52 | 95.92 |
| 300k | 99.66 | 99.68 | 95.57 | 96.63 |
| all (425k) | 99.74 | 99.73 | 96.01 | 96.92 |

Conversational text saturates at ~100-150k (+0.2pp per extra 50k words); formal/discussion text
keeps gaining ~0.7pp per 50k, almost all names. Discussion pages also contain many deliberately
*removed* ASCII spellings (si, in, daca, dupa, stiu, sînt) — counted as "not covered" here by design.

**22:05-22:35 Size decision** (`tools/size_sweep.py`, dev only, typo proxy, gaussian noise,
3 seeds × 600 sentences; `build/size-sweep.json`). Combined-mode word accuracy (%):

| words | clex KB | Tatoeba plain/diac | CV plain/diac | talk plain/diac | Wiki plain/diac | English |
|---|---|---|---|---|---|---|
| 50k | 705 | 80.67 / 93.81 | 80.46 / 93.90 | 76.48 / 89.05 | 79.38 / 90.36 | 93.81 |
| 100k | 1426 | 80.82 / 94.18 | 80.55 / 94.34 | 77.48 / 89.67 | 80.70 / 91.69 | 93.80 |
| 125k | 1787 | 80.76 / 94.20 | 80.29 / 94.26 | 77.82 / 89.90 | 80.98 / 92.01 | 93.75 |
| **150k** | 2150 | 80.72 / 94.21 | 80.17 / 94.21 | 78.16 / 90.10 | 81.11 / 92.16 | 93.64 |
| 200k | 2876 | 80.45 / 93.98 | 79.82 / 94.04 | 78.58 / 90.37 | 81.26 / 92.32 | 93.52 |
| 300k | 4353 | 80.23 / 93.83 | 79.61 / 93.86 | 79.38 / 90.91 | 81.36 / 92.51 | 93.43 |

Conversational accuracy peaks at 100-150k and falls beyond (typos landing on rare valid words);
formal/discussion accuracy keeps rising (names); English in Combined mode slowly falls. Wrong-change
rates fall monotonically with size (fewer correctly typed rare words "corrected"). **Chosen: 150,000
words** — within 0.1-0.4pp of the conversational peak, +0.4-0.7pp on formal text vs 100k, 2.15 MB.
`Lexicons/ro.clex` built by the official `vendor/build-pack.py` from `source/ro.txt`.

**22:40 Next-word weights** (`tools/tune_ngrams.py`, dev, 3000 sentences per set, top-K 16,
min count 3; `build/tune-ngrams-weights.json`). Mean dev top-3 (+fallback) / after-1-letter:
balanced (wiki .35, talk .2, Tatoeba .25, CV .1, other .1) 26.60 / 49.29; formal (wiki .55) 26.23 /
49.12; chatty (Tatoeba .4) 26.40 / 49.09; plain pooled counts 24.92 / 47.60. → balanced. Unpruned
at K16/m3 it is 813k pairs / 7.3 MB, so pruning is tuned next.
