#!/usr/bin/env python3
"""Export a trained checkpoint to Core ML exactly as the official trainer does, minus `xcrun`.

  neural/ro.mlpackage    int8-weight ML Program (official conversion call, unchanged: TorchScript
                         trace, input "tokens" Int32 [1, RangeDim(1..window)], output "logProbs"
                         (fp16, like every shipped official model), iOS 17, compute units ALL,
                         linear_symmetric int8 weight quantization)
  neural/ro.bpevocab     the matching BPE1 vocabulary (official writer) from the same training run
  neural/fixtures.json   reference outputs for the macOS compile check: for ~50 held-out (dev)
                         contexts, the top-20 log-probabilities of (a) the PyTorch fp32 model used
                         for evaluation and (b) the same model with coremltools' int8 weight
                         quantization simulated in PyTorch (what the compiled model should match up
                         to fp16 arithmetic)
  neural/export-receipt.json  hashes, sizes, parameters, IO contract, checks
The official trainer then runs `xcrun coremlcompiler compile`; that step needs macOS and is done by
.github/workflows/compile-neural.yml.

Usage: export.py CHECKPOINT [--out neural]
"""
import argparse
import copy
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import EVAL_SETS, OFFICIAL_SHA256, ROOT, WORK, Tokenizer, dump, official_grulm, sha256  # noqa: E402

TOPK = 20


def load_model(ckpt_path):
    import torch
    z = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = z["config"]
    tok = Tokenizer(cfg["vocab"])
    GRULM = official_grulm(tok.V, tok.eos, cfg["dim"], cfg["hidden"], cfg["layers"])
    model = GRULM()
    model.load_state_dict(z["model"])
    model.eval()
    return model, tok, cfg, z


def int8_simulated(model):
    """Copy of the model whose weight matrices are replaced by coremltools' linear_symmetric int8
    round trip (coremltools/optimize/_utils.py quantize_weight_by_dtype: per output row, range
    [-127, 127], scale = 2*max|w| / 254, computed on the fp16 weights as the converter does)."""
    import torch
    q = copy.deepcopy(model)
    with torch.no_grad():
        for name, p in q.named_parameters():
            if p.dim() != 2 or p.numel() < 2048:      # official config: weight_threshold 2048
                continue
            w = p.detach().numpy().astype(np.float16)
            max_abs = np.max(np.abs(w), axis=1, keepdims=True)
            scale = ((max_abs - (-max_abs)) / np.float16(254)).astype(np.float16)
            scale = np.where(scale == 0, np.float16(1), scale)
            data = np.clip(np.round(w / scale), -127, 127)
            p.copy_(torch.from_numpy((data.astype(np.float32) * scale.astype(np.float32))))
    return q


def fixture_contexts(tok, window, n_per_set=12, seed=7):
    """Held-out (dev split) contexts: a sentence prefix of random length, last `window` tokens,
    plus edge cases so that every context length 1..window occurs."""
    rng = random.Random(seed)
    out = []
    for name in EVAL_SETS:
        lines = [l.rstrip("\n").split("\t", 1)[1] for l in open(WORK / "eval" / f"{name}.dev.txt", encoding="utf-8")]
        rng.shuffle(lines)
        k = 0
        for text in lines:
            words = tok.clean(text)
            if len(words) < 2:
                continue
            cut = rng.randint(1, len(words) - 1)
            ids = tok.encode_words(words[:cut], bos=True, eos=False)[-window:]
            out.append({"set": f"{name}.dev", "prefix": " ".join(words[:cut]), "tokens": ids})
            k += 1
            if k >= n_per_set:
                break
    lengths = {len(c["tokens"]) for c in out}
    words = "nu știu ce să zic despre asta dar cred că mâine o să fie bine pentru noi toți".split()
    full = tok.encode_words(words, bos=True, eos=False)
    for T in range(1, window + 1):
        if T not in lengths:
            out.append({"set": "edge", "prefix": " ".join(words), "tokens": full[:T]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default=str(ROOT / "neural"))
    args = ap.parse_args()
    import torch
    import coremltools as ct

    torch.manual_seed(0)
    t0 = time.time()
    model, tok, cfg, z = load_model(args.checkpoint)
    window = cfg["window"]
    eos = tok.eos
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- the official export block (train-neural.py, "Export to Core ML (int8)") -------------
    model.eval()
    model.to("cpu")
    example = torch.full((1, window), eos, dtype=torch.long)   # CPU
    traced = torch.jit.trace(model, example)
    seq = ct.RangeDim(lower_bound=1, upper_bound=window, default=window)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="tokens", shape=(1, seq), dtype=np.int32)],
        outputs=[ct.TensorType(name="logProbs")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )
    from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights
    qcfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    mlmodel = linear_quantize_weights(mlmodel, config=qcfg)
    # ---------------------------------------------------------------------------------------------
    # descriptive metadata only (no effect on the network)
    mlmodel.short_description = "Romanian Pro next-token model (official Clink GRULM, BPE1 vocabulary ro.bpevocab)"
    mlmodel.author = "Romanian Pro (github.com/dobrerares/clink-romanian-pro), trained with the official Clink train-neural.py model"
    mlmodel.license = "CC BY-SA 4.0 (trained on Wikimedia text CC BY-SA 4.0, Wikinews CC BY 2.5, Tatoeba CC BY 2.0 FR, Common Voice CC0)"
    md = mlmodel.user_defined_metadata
    md["clink.vocab.sha256"] = sha256(cfg["vocab"])
    md["clink.contextWindow"] = str(window)
    md["clink.trainer.sha256"] = OFFICIAL_SHA256
    md["clink.architecture"] = f"GRULM dim={cfg['dim']} hidden={cfg['hidden']} layers={cfg['layers']} V={tok.V}"
    pkg = out / "ro.mlpackage"
    if pkg.exists():
        shutil.rmtree(pkg)
    mlmodel.save(str(pkg))
    shutil.copy2(cfg["vocab"], out / "ro.bpevocab")

    # ---- structural checks on the saved package -------------------------------------------------
    spec = ct.models.MLModel(str(pkg), skip_model_load=True).get_spec()
    inp, res = spec.description.input[0], spec.description.output[0]
    FT = ct.proto.FeatureTypes_pb2.ArrayFeatureType
    assert inp.name == "tokens" and res.name == "logProbs"
    assert inp.type.multiArrayType.dataType == FT.INT32
    ranges = [(r.lowerBound, r.upperBound) for r in inp.type.multiArrayType.shapeRange.sizeRanges]
    assert ranges == [(1, 1), (1, window)], ranges
    out_dtype = FT.ArrayDataType.Name(res.type.multiArrayType.dataType)
    ops = []

    def visit(block):
        for op in block.operations:
            ops.append(op.type)
            for child in op.blocks:
                visit(child)
    for fn in spec.mlProgram.functions.values():
        for block in fn.block_specializations.values():
            visit(block)
    n_quant = sum(t.startswith("constexpr_") for t in ops)
    assert n_quant > 0, "no quantized weights"

    # ---- check the int8 simulation against the converter's quantized embedding ----------------
    qmodel = int8_simulated(model)
    sim_check = None
    try:
        prog = mlmodel._mil_program
        for op in prog.functions["main"].operations:
            if op.op_type == "constexpr_affine_dequantize" and tuple(op.quantized_data.val.shape) == tuple(model.emb.weight.shape):
                deq = op.quantized_data.val.astype(np.float32) * op.scale.val.astype(np.float32)[:, None]
                sim = qmodel.emb.weight.detach().numpy()
                sim_check = {"tensor": "emb.weight", "max_abs_diff": float(np.abs(deq - sim).max())}
                break
    except Exception as exc:  # informative only
        sim_check = {"error": repr(exc)}

    # ---- fixtures --------------------------------------------------------------------------------
    fixtures = []
    agree = 0
    with torch.no_grad():
        for c in fixture_contexts(tok, window):
            x = torch.tensor([c["tokens"]])
            a = model(x)[0]
            b = qmodel(x)[0]
            ta, tb = torch.topk(a, TOPK), torch.topk(b, TOPK)
            agree += int(ta.indices[0] == tb.indices[0])
            ref = traced(x)[0]
            assert torch.allclose(ref, a, atol=1e-5), "trace mismatch"
            fixtures.append(dict(c, fp32={"ids": ta.indices.tolist(), "logProbs": [round(v, 5) for v in ta.values.tolist()]},
                                 int8sim={"ids": tb.indices.tolist(), "logProbs": [round(v, 5) for v in tb.values.tolist()]}))
    tol = {"vs_int8sim": {"max_abs": 0.35, "mean_abs": 0.06, "min_top1_agreement": 0.9, "min_top5_overlap": 0.8},
           "note": "Compiled model (int8 weights, fp16 arithmetic, CPU_ONLY) vs the PyTorch int8 simulation, "
                   "compared at the reference top-20 token ids; fp32 values are informative."}
    dump(out / "fixtures.json", {"vocab_sha256": sha256(out / "ro.bpevocab"), "window": window, "V": tok.V,
                                 "topk": TOPK, "tolerances": tol, "contexts": fixtures})

    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "export-receipt.json":
            files[str(p.relative_to(out))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    pkg_bytes = sum(v["bytes"] for k, v in files.items() if k.startswith("ro.mlpackage/"))
    receipt = {
        "checkpoint": str(Path(args.checkpoint).relative_to(ROOT)) if str(args.checkpoint).startswith(str(ROOT)) else args.checkpoint,
        "checkpoint_sha256": sha256(args.checkpoint), "checkpoint_state": z.get("state"),
        "architecture": {"dim": cfg["dim"], "hidden": cfg["hidden"], "layers": cfg["layers"], "window": window, "V": tok.V,
                         "parameters": int(sum(p.numel() for p in model.parameters()))},
        "io": {"input": f"tokens Int32 [1, T], T in 1..{window} (RangeDim)", "output": f"logProbs {out_dtype} [1, {tok.V}]"},
        "quantization": "linear_symmetric int8 weights (official OpLinearQuantizerConfig)", "quantized_constexpr_ops": n_quant,
        "op_histogram": {t: ops.count(t) for t in sorted(set(ops))},
        "int8_simulation_check": sim_check,
        "fixtures": len(fixtures), "fixture_top1_fp32_vs_int8sim": agree / len(fixtures),
        "mlpackage_bytes": pkg_bytes, "coremltools": ct.__version__, "torch": torch.__version__,
        "trainer_sha256": OFFICIAL_SHA256, "compiled": False,
        "compile_note": "xcrun coremlcompiler is macOS-only; run .github/workflows/compile-neural.yml",
        "seconds": round(time.time() - t0, 1), "files": files,
    }
    dump(out / "export-receipt.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("files", "op_histogram")}, indent=1))


if __name__ == "__main__":
    main()
