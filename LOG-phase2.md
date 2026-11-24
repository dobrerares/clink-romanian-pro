# Romanian Pro — Phase 2 log (neural next-word model)

Running log, newest entries at the bottom. Times are UTC (local clock is CET = UTC+1).
Branch `neural`; commits as Daedalus <daedalus@localhost>; nothing is pushed.

## 2026-11-23

**22:27 Start.** Read BRIEF-phase2.md, README.md, EVAL.md, LOG.md, SOURCES.md, and the official
contract: upstream `tools/train-neural.py`, README.md, PROMPT.md, VERIFY.md,
`tools/build-release-manifest.py`, `tools/validate-pack.py`; prior work in
`~/projects/clink/romanian-neural-experiment` (REPORT.md, export_coreml.py, ci/verify_macos.py,
ci/coreml-workflow.yml, proper-model/REPORT.md + DEPLOYMENT.md). Created branch `neural` from `main`
(c7c1c35).

**22:30 What the shipped official model really is** (decoded from the upstream
`Lexicons/ro.mlmodelc/metadata.json` + `model.mil`):
- dim 128, hidden 256, 2 GRU layers, 12,050 outputs (47 single characters + 12,000 merges + 3
  specials), int8 weights (14 `constexpr_affine_dequantize`), iOS 17 ML Program, input `tokens`
  Int32 [1, 1…12] (RangeDim), coremltools 9.0 / torch 2.8.0.
- **Output `logProbs` is Float16 [1, 12050]**, not Float32 as the trainer's docstring says: the
  trainer passes `ct.TensorType(name="logProbs")` without a dtype and the iOS 17 target defaults to
  fp16 compute. The Clink runtime therefore demonstrably consumes the Float16 output. Decision:
  export with the official conversion call unchanged (→ Float16 output, same as every shipped
  official model) rather than forcing Float32, which no shipped model uses. The Mac check compares
  in float32 after reading the output.
- The official ro vocabulary contains no `-` token: the official `clean_word()` drops every
  hyphenated word (s-a, într-o, dintr-un …) and the runtime has been working with that. Kept as is
  (changing the tokenizer could silently disagree with the closed runtime). Consequence: the neural
  model alone can never predict a hyphenated word; the `.cngm` can.
- Release manifest: `neural` metadata (sentenceCount, vocabularySize, bpeMerges) is added only when
  `source/<code>_sentences.tsv` exists; the official `ro` pack has no such file, so the field is
  optional and a neural model ships without it.

**22:35 Environment.** System `python3` is 3.14 (no coremltools wheel newer than 0.8 for 3.14), so
`.venv` was created from `/usr/bin/python3.12` (the first 3.14 venv was moved to `work/trash/`).
Installed torch 2.7.0+cpu, coremltools 9.0, numpy 2.2.6 (the versions the prior experiment used
successfully for Linux conversion). Upstream pins torch 2.4.0 / coremltools 8.0
(`vendor/requirements-train.txt`, copied); the shipped ro model was converted with coremltools 9.0.

**22:40 Official trainer vendored unmodified**: `vendor/tools/train-neural.py`, SHA-256
19ae4eb48f3fc1f9b7966a5bcd22b1ea157e6e52d16a240c9bbc3d769d53ce8a (identical to the upstream checkout
and to the copy pinned by the prior experiment at upstream commit 2cc15342). `tools/neural/nlib.py`
imports it as a module (refusing to run if the hash changes) and executes its `GRULM` class from the
file's own source text, so model, cleaning, BPE semantics and conversion are the official code.

**22:45 Throughput on this VM** (QEMU, 8 vCPUs with AVX2; behaves like ~4 physical cores). Default
model (dim 128, hidden 256, 2 layers, V≈12k: 2.28M parameters), Adam step, batch of 12-token windows:
- official loop shape (one output per context, full 12-step GRU per target): ~1,000 targets/s at 8
  threads, i.e. ~36M targets in 10 h;
- scoring every window position: 5,700-7,200 targets/s single process (8 threads); 2 threads
  4,300/s, 4 threads 5,100/s (poor thread scaling).
Decision: keep the official model/tokenizer/objective but replace the training *loop* (see
`tools/neural/train.py` docstring): windows of ≤ 12 tokens with every position scored, 4 processes ×
2 threads data-parallel (gloo all-reduce) - measured 6,650 targets/s end to end with the real data
pipeline (single process 8 threads: 2,200/s; 2×4: 4,250/s). Context length never exceeds the
official window: first window of a sentence = official contexts exactly; later windows start every
6 tokens and only score targets with 7-12 tokens of context.

**22:50 Corpus** (`tools/neural/corpus.py` → `work/neural/corpus/`, stats in `stats.json`). Input:
Phase 1 `work/train/*.txt` (TRAIN split, already stripped of every near-duplicate of any dev/test
sentence of any source). Typographic quotes „“”«» → ASCII (the official `clean_word` strips only
ASCII quotes, so „Apa would be dropped as a non-word) and zero-width characters removed. Whole
sentences dropped for: gap mark ¦; a token the official cleaner rejects that is not a hyphenated
word (numbers, URLs, glued punctuation: the trainer would silently join the neighbours);
letters outside a-z ă â î ș ț '; < 2 or > 50 words; a Phase 1 removed diacritic-less / pre-1993
spelling (si, dupa, stiu, sînt, asa, maine …); 8+ words without any diacritic (not applied to the
curated Tatoeba/Common Voice, where it hit correct sentences such as "Ea a primit un telefon mobil
de ziua ei."); ≥ 2 English function words making ≥ 20% of the sentence; and a *template cap*: at
most 3 sentences per "shape" (digits → #, capitalised non-initial words → P) in the Wikimedia and
talk sources, which removes templated stubs (the Phase 1 `o → comună, suprafață` artifact).

| source | train sentences in | kept | kept words | main drops |
|---|---|---|---|---|
| Wikipedia articles | 4,773,682 | 1,971,363 | 35,970,733 | junk 1,690,073 (mostly numbers), gap 786,417, foreign 114,125, template 52,028 |
| Wikipedia discussions | 1,172,424 | 678,331 | 10,296,572 | junk 204,428, diacless 163,786, template 57,853 |
| Tatoeba (train) | 33,545 | 32,479 | 209,634 | short 532, junk 358, diacless 122 |
| Common Voice (train) | 10,814 | 10,697 | 76,478 | diacless 103 |
| Wikinews/-quote/-voyage/-books | 47,472 | 30,877 | 483,936 | junk, diacless |

Total **2,723,747 sentences, 47.0M words** (`work/neural/ro_sentences.tsv`, 348 MB, official
`id<TAB>sentence` format).

**22:55 BPE** (`tools/neural/bpe.py`). The official `train_bpe` recounts all pairs for each merge
(90 s for 1,500 merges on 20k sentences; hours for 12,000 merges on 650k word types). Re-implemented
incrementally with the same tie-break; verified **identical merges and vocabulary** to the official
function on two samples (20,000 talk sentences / 1,500 merges; 32,479 Tatoeba sentences / 2,500
merges). Full run on the sentence file, 12,000 merges, 3 min: **12,036 tokens** (32 letters + `</w>`
+ 12,000 merges + 3 specials) → `work/neural/ro.bpevocab` (official BPE1 writer).

**23:00 Encoding** (`tools/neural/encode.py`, official `clean_word` + `bpe_encode_word`):
training targets per source - Wikipedia 50.9M, discussions 13.0M, Tatoeba 0.29M, Common Voice 0.10M,
other Wikimedia 0.67M (64.9M total). Phase 1 demotion carried over as a **loss weight 0.05** on the
first token of a demoted word in the source that causes it (Tatoeba tom/mary/john: 5,177 targets;
talk-page Wikipedia self-reference: 364,205 targets).

**23:05 Pilots** (equal wall-clock, 4 in parallel, 1 process × 2 threads each, 0.85 h, cosine LR over
the time budget, lr 2e-3, 64 windows/step):
- A: 128/256, mix M1 = wiki 1, talk 3, Tatoeba 20, CV 5, other 3 (per-epoch repetition factors)
- B: 192/384 (3.95M parameters), mix M1
- C: 128/256, stride 12 (deep windows score all 12 positions: contexts 1-12), mix M1
- D: 128/256, mix M0 = wiki 0.5, talk 3, Tatoeba 30, CV 8, other 3 (more conversational)

**23:05-23:20 Export, evaluation, Mac job, installer (written and tested while the pilots run).**
- `tools/neural/export.py`: the official export block verbatim (trace on CPU, `ct.convert` with
  `tokens` Int32 [1, RangeDim(1, 12)], `logProbs` without dtype, compute units ALL, iOS 17;
  `linear_quantize_weights` linear_symmetric int8), plus descriptive metadata only; no `xcrun`.
  Tested on a smoke checkpoint: output **FLOAT16 [1, 12036]**, 14 `constexpr_affine_dequantize` ops,
  2.38 MB package, and the converted program's op histogram is **identical, op type by op type and
  count by count, to the shipped official `ro.mlmodelc`** (2 while_loops, 14 linear, 14 cast …).
- int8 simulation in PyTorch (coremltools' `quantize_weight_by_dtype` formula: per output row,
  ±127, scale = 2·max|w|/254 on the fp16 weights): reproduces the converter's quantized embedding
  exactly (max |Δ| = 0.0). Used for the fixtures and to measure the accuracy cost of int8.
- `neural/fixtures.json`: 49 held-out **dev** contexts (12 per set + edge cases so every length 1-12
  occurs), top-20 ids + log-probs from the fp32 model and from the int8 simulation.
- `tools/neural/verify_compiled.py` (macOS only; refuses elsewhere): `xcrun coremlcompiler compile`,
  interface check from the compiled `metadata.json`, CPU_ONLY predictions vs the int8 reference
  (max/mean |Δ| at the reference top-20 ids, top-1 agreement, top-5 overlap), vocabulary hash = the
  fixtures' hash. Found while dry-running it: the official graph computes `log(softmax(x))` in fp16, so
  tokens with p < ~6e-8 legitimately come out as −inf; the check therefore requires no NaN/+inf and
  finite top-20 values instead of "all finite". Linux dry run with fp16 logits/softmax/log: max |Δ|
  0.005 (tolerance 0.35), top-1/top-5 agreement 1.0.
- `.github/workflows/compile-neural.yml`: workflow_dispatch only, `macos-latest`, `contents: read`,
  checkout without persisted credentials, Python 3.12 + coremltools 9.0, runs the verifier, uploads
  `out/` (ro.mlmodelc, ro.bpevocab, verification.json, compile.log) as artifact `ro-mlmodelc`,
  retention 3 days. No secrets, no caches, no paid features.
- `tools/install-neural.sh ARTIFACT_DIR`: refuses unless `verification.json` says passed and the
  vocabulary hash matches `neural/fixtures.json`; copies to `Lexicons/{ro,ro_ro,mo,rup}.mlmodelc` +
  `.bpevocab`; runs `vendor/tools/validate-pack.py` per code. Tested on a scratch copy of the repo with
  a mock artifact (all four "looks ready for release"; refusal path exits 1); the repo's manifest
  builder then lists every `<code>.mlmodelc/*` file and `<code>.bpevocab` as assets (no `neural`
  block, since there is no `source/<code>_sentences.tsv`). The real `Lexicons/` was not touched.
- `tools/neural/evaluate_next.py`: Phase 1's sampler, pair extraction, (a) code path and intervals
  imported from `tools/evaluate.py`; (b) exact best-first whole-word search; (c) interpolation.
- Web search for public documentation of how Clink combines the neural model with the `.cngm`: none
  found. The combination in EVAL-phase2.md is therefore an explicit assumption.

**23:55 Pilot results** (equal wall-clock 0.85 h each, LR annealed; dev = 1,500 sentences per dev
set, NLL per BPE target with exact official ≤ 12-token contexts; `work/neural/runs/pilot*/final.json`):

| pilot | targets seen | mean | Tatoeba | Wikipedia | talk | CV |
|---|---|---|---|---|---|---|
| A 128/256, M1, stride 6 | 6.5M | 5.591 | 5.297 | 6.321 | 5.582 | 5.166 |
| B 192/384, M1, stride 6 | 4.1M | 5.677 | 5.395 | 6.424 | 5.650 | 5.239 |
| C 128/256, M1, stride 12 | 8.0M | 5.555 | 5.235 | 6.290 | 5.608 | 5.087 |
| D 128/256, M0, stride 6 | 7.0M | **5.506** | **5.094** | 6.407 | **5.484** | **5.039** |

Whole-word next-word top-3 on dev (`evaluate_next.py`, 2 seeds × 800 sentences per set, paired;
`work/neural/scratch/pilot*-dev.json`), (a) Phase 1 `.cngm` for reference:

| | Tatoeba | Wikipedia | talk | CV | mean | (c) λ 0.5, mean | (c) after 1 letter |
|---|---|---|---|---|---|---|---|
| (a) .cngm | 25.8 | 25.1 | 25.9 | 26.4 | 25.8 | — | 51.3 (a) |
| A | 24.0 | 23.3 | 30.0 | 27.5 | 26.2 | 29.2 | 54.9 |
| C | 24.5 | 23.1 | 29.6 | 28.0 | 26.3 | 28.9 | 54.9 |
| D | 25.5 | 22.4 | 30.8 | 28.8 | **26.9** | **29.4** | **55.1** |

Decisions: **size stays at the official default** (dim 128, hidden 256, 2 layers, window 12): the
1.7× larger model is behind at equal time, and the default matches the shipped models' on-device
budget (2.4 MB). **Mixture M0 (conversational)**: best on Tatoeba, talk and CV, −0.09 nats on
Wikipedia. **Stride 6**: stride 12 wins slightly on dev NLL but not on top-3 (mixed per set), and
stride 6 keeps training contexts closer to the inference contexts. The final run sees ~20× the
pilot's data, so the two smallest sources are repeated a little less per epoch than in D
(Tatoeba 30 → 20, Common Voice 8 → 6) to limit memorisation (≈ 40 passes over Tatoeba in total
instead of ≈ 55).

**00:05 Final run** `work/neural/runs/final` (`--run final --world 4 --threads 2 --batch 64 --dim 128
--hidden 256 --layers 2 --window 12 --stride 6 --mix wiki=0.5,wikitalk=3,tatoeba=20,cv=6,
wikinews=3,wikiquote=3,wikivoyage=3,wikibooks=3 --total-steps 80000 --lr 2e-3 --warmup 1000
--lr-floor 0.05 --clip 1.0 --seed 20261124`). A 3-minute probe measured 3.97 steps/s (7,200
targets/s, 1,812 targets per step of 256 windows; 40,166 steps per mixture epoch of 72.8M targets),
so 80,000 steps ≈ 2 epochs ≈ 145M targets ≈ 5.6-5.8 h. The harness stops background jobs after 2 h,
so `train.py` got `--session-limit`: each invocation checkpoints and exits after 1.85 h and the same
command resumes (tested with 2 ranks: pause at step 82, resume from step 82). A first launch was
stopped during setup (no step trained; moved to `work/trash/`).

**00:10 Decision: `source/<code>_sentences.tsv`.** What it must contain: the trainer's input, one
training sentence per line (`id<TAB>sentence`), i.e. `work/neural/ro_sentences.tsv` -
2,723,747 sentences, 348 MB (the manifest builder only counts its lines, for `neural.sentenceCount`).
That is 7× the repository's 50 MB rule and 3.5× GitHub's hard 100 MB per-file limit, would be needed
four times (one per code, or symlinks), and would add ~350 MB (≈ 90 MB compressed) of sentence text
to the public history for a single number in the manifest. Not committed. Consequences: the release
manifest has no `neural` block (sentenceCount, vocabularySize, bpeMerges, modelVersion) for these
packs - exactly like the official `ro` pack, which ships its neural model without
`source/ro_sentences.tsv`; the assets themselves (`<code>.mlmodelc/*`, `<code>.bpevocab`) are
listed and hashed as usual. What is committed instead: `source/ro_neural-corpus.json` (receipt:
SHA-256 and line count of the exact training file, per-source counts, filters, BPE and training
settings, rebuild commands; the file is rebuilt bit-for-bit by `tools/neural/corpus.py` from the
Phase 1 data). Alternatives if the `neural` block turns out to matter on device: (1) let the repo's
own `tools/build-release-manifest.py` take the count from the receipt when the TSV is absent (a small
change to release tooling, so not done without Daedalus's approval); (2) publish the compressed corpus
as a separate release download outside `Lexicons/` (Clink ignores it). Never a shortened TSV: its
count would misdescribe the model.

## 2026-11-24

**00:05-05:45 Final training** in four resumable sessions (pauses at steps 26,405 / 52,573 / 78,132,
each resumed from `last.pt` with the same command; ~7,200 targets/s throughout). **80,000 steps,
144,981,296 targets, 5.69 h** (20,471 s) of training. Dev NLL per BPE target (exact official
contexts) fell monotonically on all four sets, including the most repeated one (Tatoeba
5.14 → 4.29: no sign of over-fitting after ≈ 40 passes):

| step | Tatoeba | Wikipedia | discussions | Common Voice | mean |
|---|---|---|---|---|---|
| 5,000 | 5.138 | 6.350 | 5.519 | 5.036 | 5.511 |
| 20,000 | 4.621 | 5.739 | 5.042 | 4.583 | 4.996 |
| 40,000 (epoch 1) | 4.411 | 5.528 | 4.896 | 4.401 | 4.809 |
| 60,000 | 4.325 | 5.418 | 4.805 | 4.326 | 4.718 |
| **80,000 (final)** | **4.290** | **5.384** | **4.782** | **4.301** | **4.689** |

Total training compute for Phase 2: pilots 0.85 h (4 in parallel) + probes ≈ 0.1 h + final 5.69 h
≈ 6.6 h wall-clock, within the ~10 h budget.

**05:47 Export** (`tools/neural/export.py work/neural/runs/final/final.pt --out neural`):
`neural/ro.mlpackage` 2,377,274 bytes (weight.bin 2,338,824 B, sha256 5221ca12…85ed5),
`logProbs` FLOAT16 [1, 12036], 14 int8 constexpr ops, int8 simulation = converter's quantized
embedding exactly; `neural/ro.bpevocab` sha256 ea1bc387…d5056 (identical to the vocabulary the model
was trained with); `neural/fixtures.json` 49 dev contexts (sha256 c0ac7e4f…1cbee), fp32 vs int8
top-1 agreement 48/49. Linux dry run of the macOS check with fp16 logits/softmax/log on the real model:
max |Δ| 0.014, mean 0.0016, top-1/top-5 1.0; up to 11,571 of 12,036 log-probs become −inf in fp16
(handled). `source/ro_neural-corpus.json` written (training file sha256 5f2a38eb…fead0,
2,723,747 lines, 1.992 mixture epochs).

**05:50-06:03 Evaluation** (`build/neural-eval-dev.json`, `-dev-int8sim.json`, `-test.json`;
EVAL-phase2.md). Dev (5 seeds × 2,000 sentences per set): λ scan over 0…1 → **λ = 0.3** best for
mean top-3 (33.00) and after 1 letter (59.76); fixed before the test run, which computed only
λ ∈ {0, 0.3, 1}. int8 simulation on dev: neural top-3 within 0.04 points, combination within 0.15.
Test, top-3 / after 1 letter, (a) Phase 1 `.cngm` → (b) neural → (c) λ 0.3:
- discussions 26.4 → 35.3 → 35.8 / 53.1 → 59.5 → 60.5
- Tatoeba 26.2 → 31.8 → 32.8 / 53.2 → 60.4 → 62.0
- Common Voice 26.2 → 32.8 → 33.5 / 53.5 → 60.3 → 61.7
- Wikipedia 24.7 → 27.0 → 29.2 / 47.5 → 49.9 → 53.2
All 16 paired differences vs (a) positive, intervals ±0.1-0.5. The (a) column equals EVAL.md's
Romanian Pro column to the decimal (protocol identical). 1.2-2.4% of next words are hyphenated
(unreachable for the neural model alone).

**Verdict: ship** (ship gate met by (b) alone and by (c)), conditional on the macOS compile job
passing its numeric check and an on-device look in Clink.

**06:05 Files for the commit.** `neural/ro.mlpackage/` (Manifest.json, Data/com.apple.CoreML/
model.mlmodel + weights/weight.bin), `neural/ro.bpevocab`, `neural/fixtures.json`,
`neural/export-receipt.json`, `source/ro_neural-corpus.json`, `build/neural-eval-{dev,dev-int8sim,
test}.json`, EVAL-phase2.md, README.md, LOG-phase2.md, `tools/neural/*` updates. Not in `Lexicons/`:
no `.mlmodelc` exists yet. `main` untouched.

**06:10 Commits on `neural`** (local only, author Daedalus <daedalus@localhost>; `main` still at
c7c1c35): 08a8094 Phase 2 tooling; ade142d model, vocabulary, fixtures, evaluation, docs; followed
by this log update. Nothing was pushed, tagged or released; no GitHub call was made; the other
repositories were only read.

### What Daedalus has to do (nothing here was pushed, tagged or released)

1. Push the branch: `git push origin neural`.
2. Run the compile job on the branch: `gh workflow run compile-neural.yml --ref neural`, then
   `gh run watch` (≈ 5 min on `macos-latest`, free for public repositories). It must end green;
   `verification.json` in the artifact shows the numbers.
3. Download the artifact: `gh run download <run-id> -n ro-mlmodelc -D /tmp/ro-mlmodelc`.
4. Install: `tools/install-neural.sh /tmp/ro-mlmodelc` (refuses unless verification passed and the
   vocabulary matches; copies `ro.mlmodelc` + `ro.bpevocab` to `Lexicons/{ro,ro_ro,mo,rup}.*` and
   runs `vendor/tools/validate-pack.py` for each code).
5. Commit `Lexicons/` on `neural`, merge `neural` into `main` (fast-forward), then release as for
   Phase 1 (`./update.sh` tags `vYYYY.MM.DD.HHMMSS` and pushes; the release workflow builds the
   manifest with every `<code>.mlmodelc/*` file and `<code>.bpevocab` as assets).
6. On a device: update the pack in Clink and check that predictions change (e.g. after "hai să",
   "mi-e dor de"). If the neural model does not seem to load, the missing manifest `neural` block is
   the first suspect (see the 00:10 entry for the two ways to add it).
