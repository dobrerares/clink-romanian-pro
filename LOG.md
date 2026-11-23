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
