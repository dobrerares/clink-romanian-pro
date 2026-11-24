#!/usr/bin/env python3
"""macOS only: compile neural/ro.mlpackage with xcrun coremlcompiler and check it numerically.

Run by .github/workflows/compile-neural.yml (or by hand on any Mac with Xcode command-line tools):
  python3 tools/neural/verify_compiled.py --out out
Writes out/ro.mlmodelc (the compiled model), out/ro.bpevocab (the matching vocabulary, copied),
out/compile.log and out/verification.json; exits non-zero if any check fails.

Checks
  * the vocabulary is BPE1, its token count equals the model's output width and its SHA-256 equals
    the one recorded in neural/fixtures.json (model and vocabulary from the same training run);
  * the compiled model's interface: input "tokens" Int32 [1, 1..window], output "logProbs" [1, V];
  * for every fixture context (held-out dev text, every length 1..window), CPU_ONLY prediction:
    no NaN (very unlikely tokens may be -inf: the official graph takes log(softmax) in fp16),
    log-softmax normalised (|logsumexp| small), and close to the PyTorch reference with the
    same int8 weight quantization (neural/fixtures.json "int8sim": top-20 ids and log-probs):
    max/mean absolute difference at those ids, top-1 agreement, top-5 overlap. The fp32 PyTorch
    comparison is reported for information.
Nothing is downloaded and no credentials are used.
"""
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=str(ROOT / "neural" / "ro.mlpackage"))
    ap.add_argument("--vocab", default=str(ROOT / "neural" / "ro.bpevocab"))
    ap.add_argument("--fixtures", default=str(ROOT / "neural" / "fixtures.json"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    args = ap.parse_args()
    if platform.system() != "Darwin":
        sys.exit("macOS (xcrun coremlcompiler + Core ML runtime) is required; nothing was compiled.")
    import coremltools as ct

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"platform": platform.platform(), "machine": platform.machine(), "coremltools": ct.__version__,
              "checks": {}, "passed": False}
    fx = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    lines = Path(args.vocab).read_text(encoding="utf-8").split("\n")
    V = int(lines[1])
    report["checks"]["vocab_format"] = lines[0] == "BPE1" and int(lines[2 + V]) >= 0
    report["checks"]["vocab_sha256_matches_fixtures"] = sha256(args.vocab) == fx["vocab_sha256"]
    report["checks"]["vocab_size_matches_fixtures"] = V == fx["V"]

    target = out / "ro.mlmodelc"
    if target.exists():
        shutil.rmtree(target)
    p = subprocess.run(["xcrun", "coremlcompiler", "compile", args.package, str(out)], capture_output=True, text=True)
    (out / "compile.log").write_text(p.stdout + p.stderr)
    report["compile_exit_code"] = p.returncode
    if p.returncode != 0 or not target.is_dir():
        report["checks"]["compiled"] = False
        (out / "verification.json").write_text(json.dumps(report, indent=1) + "\n")
        sys.exit("coremlcompiler failed; see compile.log")
    report["checks"]["compiled"] = True
    shutil.copy2(args.vocab, out / "ro.bpevocab")

    meta = json.loads((target / "metadata.json").read_text())[0]
    inp = {x["name"]: x for x in meta["inputSchema"]}
    outp = {x["name"]: x for x in meta["outputSchema"]}
    report["interface"] = {"input": inp.get("tokens"), "output": outp.get("logProbs")}
    try:
        report["checks"]["interface"] = (
            inp["tokens"]["dataType"] == "Int32"
            and json.loads(inp["tokens"]["shapeRange"]) == [[1, 1], [1, fx["window"]]]
            and json.loads(outp["logProbs"]["shape"]) == [1, V])
    except (KeyError, ValueError, TypeError):
        report["checks"]["interface"] = False

    model = ct.models.CompiledMLModel(str(target), compute_units=ct.ComputeUnit.CPU_ONLY)
    diffs, diffs32, top1, top5, lse, lengths = [], [], [], [], [], set()
    finite = True
    for c in fx["contexts"]:
        x = np.asarray([c["tokens"]], dtype=np.int32)
        lengths.add(x.shape[1])
        y = np.asarray(model.predict({"tokens": x})["logProbs"], dtype=np.float32).reshape(-1)
        ref = c["int8sim"]
        ids = np.asarray(ref["ids"])
        # The official graph computes log(softmax(x)) in fp16, so tokens with p < ~6e-8 legitimately
        # come out as -inf (the shipped official models do the same). Require: no NaN, +inf or wrong
        # shape, and finite values for the reference top-20 tokens.
        if y.shape[0] != V or np.isnan(y).any() or np.isposinf(y).any() or not np.isfinite(y[ids]).all():
            finite = False
            continue
        report.setdefault("neg_inf_entries_max", 0)
        report["neg_inf_entries_max"] = max(report["neg_inf_entries_max"], int(np.isneginf(y).sum()))
        m = y.max()
        lse.append(float(m + np.log(np.exp(y - m).sum())))
        diffs.append(np.abs(y[ids] - np.asarray(ref["logProbs"], dtype=np.float32)))
        diffs32.append(np.abs(y[np.asarray(c["fp32"]["ids"])] - np.asarray(c["fp32"]["logProbs"], dtype=np.float32)))
        order = np.argsort(-y)
        top1.append(int(order[0] == ids[0]))
        top5.append(len(set(order[:5].tolist()) & set(ids[:5].tolist())) / 5)
    tol = fx["tolerances"]["vs_int8sim"]
    d = np.concatenate(diffs) if diffs else np.array([np.inf])
    d32 = np.concatenate(diffs32) if diffs32 else np.array([np.inf])
    report["numeric"] = {
        "fixtures": len(fx["contexts"]), "context_lengths": sorted(lengths),
        "vs_int8sim": {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                       "top1_agreement": float(np.mean(top1)) if top1 else 0.0,
                       "top5_overlap": float(np.mean(top5)) if top5 else 0.0},
        "vs_fp32_pytorch": {"max_abs": float(d32.max()), "mean_abs": float(d32.mean())},
        "max_abs_logsumexp": float(np.max(np.abs(lse))) if lse else None,
        "tolerances": tol,
    }
    n = report["numeric"]["vs_int8sim"]
    report["checks"]["shape_no_nan_top20_finite"] = finite
    report["checks"]["normalised"] = bool(lse) and max(abs(v) for v in lse) < 0.05
    report["checks"]["all_lengths"] = sorted(lengths) == list(range(1, fx["window"] + 1))
    report["checks"]["numeric"] = (n["max_abs"] <= tol["max_abs"] and n["mean_abs"] <= tol["mean_abs"]
                                   and n["top1_agreement"] >= tol["min_top1_agreement"]
                                   and n["top5_overlap"] >= tol["min_top5_overlap"])
    report["compiled_files"] = {str(f.relative_to(target)): {"bytes": f.stat().st_size, "sha256": sha256(f)}
                                for f in sorted(target.rglob("*")) if f.is_file()}
    report["passed"] = all(report["checks"].values())
    (out / "verification.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("checks", "numeric", "passed")}, indent=1))
    if not report["passed"]:
        sys.exit("verification failed")


if __name__ == "__main__":
    main()
