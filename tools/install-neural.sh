#!/usr/bin/env bash
# Install the compiled neural model into the pack for every shipped code.
#
# Usage: tools/install-neural.sh ARTIFACT_DIR
#   ARTIFACT_DIR = the downloaded "ro-mlmodelc" artifact of .github/workflows/compile-neural.yml
#                  (contains ro.mlmodelc/, ro.bpevocab, verification.json, compile.log), e.g.
#                  gh run download <run-id> -n ro-mlmodelc -D /tmp/ro-mlmodelc
#
# Refuses to install unless the artifact's verification passed and its vocabulary is byte-identical
# to neural/ro.bpevocab (the vocabulary of the same training run as neural/ro.mlpackage).
# Copies ro.mlmodelc -> Lexicons/<code>.mlmodelc and neural/ro.bpevocab -> Lexicons/<code>.bpevocab
# for ro, ro_ro, mo and rup (identical copies, like the other assets), then runs the official
# validator for each code. Does not commit, tag or publish.
set -euo pipefail

src="${1:?usage: tools/install-neural.sh ARTIFACT_DIR}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
codes=(ro ro_ro mo rup)
vocab="$repo/neural/ro.bpevocab"

test -d "$src/ro.mlmodelc" || { echo "missing $src/ro.mlmodelc" >&2; exit 1; }
test -f "$src/ro.mlmodelc/coremldata.bin" || { echo "$src/ro.mlmodelc is not a compiled Core ML model" >&2; exit 1; }
test -f "$src/verification.json" || { echo "missing $src/verification.json" >&2; exit 1; }
test -f "$vocab" || { echo "missing $vocab" >&2; exit 1; }

python3 - "$src/verification.json" "$vocab" "$repo/neural/fixtures.json" <<'PY'
import hashlib, json, sys
report = json.load(open(sys.argv[1]))
if report.get("passed") is not True:
    sys.exit(f"verification did not pass: {report.get('checks')}")
fixtures = json.load(open(sys.argv[3]))
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if digest != fixtures["vocab_sha256"]:
    sys.exit("neural/ro.bpevocab does not match neural/fixtures.json (different training run)")
lines = open(sys.argv[2], encoding="utf-8").read().split("\n")
if lines[0] != "BPE1":
    sys.exit("neural/ro.bpevocab is not BPE1")
print(f"verification passed: {report['numeric']['vs_int8sim']}; vocabulary {int(lines[1])} tokens")
PY

if [ -f "$src/ro.bpevocab" ]; then
  cmp -s "$src/ro.bpevocab" "$vocab" || { echo "artifact ro.bpevocab differs from neural/ro.bpevocab" >&2; exit 1; }
fi

cd "$repo"
for code in "${codes[@]}"; do
  rm -rf "Lexicons/$code.mlmodelc"
  cp -R "$src/ro.mlmodelc" "Lexicons/$code.mlmodelc"
  cp "$vocab" "Lexicons/$code.bpevocab"
  # Finder/AppleDouble sidecars are never model data (the manifest builder skips them anyway)
  find "Lexicons/$code.mlmodelc" \( -name '._*' -o -name '.DS_Store' \) -delete
done

for code in "${codes[@]}"; do
  python3 vendor/tools/validate-pack.py "$code"
done
du -sh Lexicons/*.mlmodelc
echo "Installed. Review with: git status Lexicons/  (then commit and release as usual)"
