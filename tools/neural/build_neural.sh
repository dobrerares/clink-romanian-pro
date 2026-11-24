#!/bin/sh
# Romanian Pro, Phase 2: neural next-word model, full rebuild on Linux (CPU) from the Phase 1 data.
# Run from the repository root after tools/build_all.sh steps 1-2 (work/train, work/eval exist).
# Python: .venv (python3.12 -m venv .venv; .venv/bin/pip install torch==2.7.0 --index-url
# https://download.pytorch.org/whl/cpu; .venv/bin/pip install coremltools==9.0 numpy==2.2.6).
# Every decision and number is in LOG-phase2.md; results in EVAL-phase2.md.
set -eu
PY=".venv/bin/python -I"
RUN=final

# 1. corpus: filter the Phase 1 TRAIN split for the neural model -> work/neural/ro_sentences.tsv
$PY tools/neural/corpus.py
# 2. BPE vocabulary, 12,000 merges (same algorithm and output as the official train_bpe)
$PY tools/neural/bpe.py work/neural/ro_sentences.tsv work/neural/ro.bpevocab --merges 12000
# 3. encode with the official clean_word + bpe_encode_word
$PY tools/neural/encode.py work/neural/ro.bpevocab
# 4. train the official GRULM (4 processes x 2 threads; resumable: rerun the same command)
$PY tools/neural/train.py --run $RUN @TRAIN_ARGS@
# 5. Core ML export (official conversion call), vocabulary, fixtures for the macOS check
$PY tools/neural/export.py work/neural/runs/$RUN/final.pt --out neural
$PY tools/neural/receipt.py work/neural/runs/$RUN
# 6. evaluation: lambda on dev, then test (EVAL-phase2.md)
$PY tools/neural/evaluate_next.py work/neural/runs/$RUN/final.pt --split dev --out build/neural-eval-dev.json
$PY tools/neural/evaluate_next.py work/neural/runs/$RUN/final.pt --split test --out build/neural-eval-test.json
# 7. on a Mac / GitHub macOS runner: .github/workflows/compile-neural.yml (tools/neural/verify_compiled.py),
#    then tools/install-neural.sh <artifact dir>
