You are an engineering worker for Daedalus. Task: Phase 1 of "Romanian Pro", a much stronger Romanian language pack for the Clink iOS keyboard (user types mixed Romanian + English in Combined mode, alongside the Bilingual Guard 1.1 plugin). Work in the local git repo /home/rdobre/projects/clink/romanian-pro (create it; git user Daedalus <daedalus@localhost>). Keep a running log in LOG.md there.

HARD LIMITS
- Local only: no git push, no gh, no new GitHub repos, no releases, nothing external except downloading openly licensed data.
- Do not modify ~/projects/clink/clink-romanian-language, clink-ro-pack, clink-typing-plugins, upstream/*, or anything under ~/.hermes. Copy what you need.
- No sudo, no reboot, no paid services. Downloads go under romanian-pro/data/ (gitignored) with sha256 + URL + licence recorded in data/MANIFEST.json.
- Licences: every source must allow redistribution of derived word counts / n-grams in a public pack. Record each in SOURCES.md (name, URL, licence, what is derived, attribution text). If a licence is unclear, non-commercial (NC), or "research only", EXCLUDE it and note why. ShareAlike (CC BY-SA) is acceptable; the pack then ships under CC BY-SA 4.0.

CANDIDATE SOURCES (verify each licence yourself from the source's own page before use)
- Romanian Wikipedia pages-articles dump (CC BY-SA 4.0 / GFDL): main frequency + n-gram source (formal register).
- FrequencyWords ro 2018 (OpenSubtitles counts; CC BY-SA 4.0 content, MIT code): spoken/informal register. Already at ~/projects/clink/autocorrect-engine/data/ro_50k.txt (also try the ro_full list).
- Tatoeba Romanian (CC BY 2.0 FR): conversational sentences; ~/projects/clink/autocorrect-engine/data/ron_sentences.tsv.bz2.
- Mozilla Common Voice Romanian sentence text (CC0): short conversational sentences, if downloadable without an account.
- Romanian Wikisource / Wikinews / Wikiquote / Wikipedia talk pages (CC BY-SA): extra and more informal text.
- Leipzig Corpora Collection, CC-100, OSCAR, MOROCO, Reddit, social media: only if the licence clearly allows redistribution of derived counts; default EXCLUDE.
- Hunspell ro_RO (rospell; GPL/LGPL/MPL): build-time validity checks ONLY, never shipped.
- English loanwords/tech terms: select English-spelled tokens that are actually frequent in the Romanian sources above (e.g. "laptop", "weekend", "update", "deadline", "meeting", "ok", "mail"), not a whole English list.

BUILD
1. Normalise: NFC; cedilla ş/ţ/Ş/Ţ -> comma ș/ț/Ș/Ț; strip markup; keep case info for proper nouns as the official builder expects.
2. Word list with real frequencies: merge sources with documented weights (Wikipedia formal, OpenSubtitles/Tatoeba/Common Voice informal). Apply the diacritic-cleaning rules from ~/projects/clink/autocorrect-engine/tools/clean_ro_pack.py (drop dominated ASCII misspellings like dupa/stiu/multumesc, keep valid sa/ca/fata/pana/ma). Drop junk (URLs, numbers, wiki tokens, single-occurrence noise). Target size: decide from a words-vs-coverage curve; expect ~100k-200k words.
3. ro.cngm next-word model from all sentence sources; tune pair pruning on dev.
4. Copy ro.emoji.json from the official pack (~/projects/clink/upstream/clink-language-packs/Lexicons/ro.emoji.json) with its attribution.
5. Build with the official builder (~/projects/clink/upstream/clink-language-packs/build-pack.py and tools/build-next-word.py; copy them in) and validate with tools/validate-pack.py. Read README/PROMPT/VERIFY there for format and any size guidance.

EVAL (held-out, never trained on; split by document/sentence hash with near-duplicate removal, like ~/projects/clink/autocorrect-engine/tools/corpus.py)
- Compare three packs: official ro (upstream), current published dobrerares/clink-romanian-language (local copy ~/projects/clink/clink-romanian-language, cleaned 41,630 words), and Romanian Pro.
- Typo correction: synthetic touch noise (Gaussian and --noise realistic modes from autocorrect-engine/tools/eval.py), including diacritic-less typing (sa, si, fata, dupa, stiu), using a dictionary-corrector proxy that loads each pack's word list and frequencies. State clearly it is a proxy for Clink's closed engine.
- Next-word top-3 accuracy using each pack's .cngm decoded offline (write a reader from the builder's format).
- Held-out sets: Tatoeba test split, held-out Wikipedia articles, held-out informal text if available. Report 95% CIs over several seeds, plus wrong-change rates.
- Coverage: share of held-out tokens in each dictionary.

FINISH with: sources accepted/excluded with licences, pack file sizes, eval table vs both baselines with CIs, local commit hashes, blockers. If something blocks you, write it to LOG.md and stop with an exact description of what's needed.
