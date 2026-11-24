#!/usr/bin/env python3
"""Write source/ro_neural-corpus.json: what the neural model was trained on, without the text.

The release manifest builder records `neural.sentenceCount` only when source/<code>_sentences.tsv
exists. The real training file (work/neural/ro_sentences.tsv) is 2.7M sentences / 348 MB, above
GitHub's 100 MB file limit and the repository's 50 MB rule, so it is not committed; this receipt
records its exact hash, line count, per-source composition, filters, BPE and training settings so the
file can be rebuilt bit-for-bit from the Phase 1 data (tools/neural/corpus.py) and checked.

Usage: receipt.py RUN_DIR [--out source/ro_neural-corpus.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nlib import NEURAL, OFFICIAL_SHA256, ROOT, dump, sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=str(ROOT / "source" / "ro_neural-corpus.json"))
    args = ap.parse_args()
    run = Path(args.run_dir)
    sent = NEURAL / "ro_sentences.tsv"
    with open(sent, "rb") as fh:
        lines = sum(1 for _ in fh)
    stats = json.loads((NEURAL / "corpus" / "stats.json").read_text())
    enc = json.loads((NEURAL / "enc" / "report.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    final = json.loads((run / "final.json").read_text()) if (run / "final.json").exists() else None
    sources = {}
    for src, st in stats.items():
        if src.startswith("_"):
            continue
        sources[src] = {"phase1_train_sentences": st["in"], "kept_sentences": st["kept"], "kept_words": st["kept_words"],
                        "dropped": {k: v for k, v in st.items() if k not in ("in", "kept", "kept_words")},
                        "bpe_tokens": enc[src]["tokens"], "training_targets": enc[src]["targets"],
                        "demoted_targets": enc[src]["demoted_targets"], "mixture_weight": cfg["mix"].get(src)}
    receipt = {
        "what": "Training corpus of the Romanian Pro neural next-token model (neural/ro.mlpackage, ro.bpevocab). "
                "The sentence text itself is not committed (348 MB); see LOG-phase2.md.",
        "sentence_file": {"path": "work/neural/ro_sentences.tsv", "format": "<source>:<doc id>\\t<sentence> (official trainer format)",
                          "sentences": lines, "bytes": sent.stat().st_size, "sha256": sha256(sent)},
        "built_from": "Phase 1 work/train/<source>.txt (training split; every near-duplicate of any dev/test sentence removed)",
        "sources": sources,
        "filters": stats.get("_filters"),
        "bpe": {"merges": 12000, "tokens": cfg.get("V") or None, "vocab_sha256": sha256(ROOT / "neural" / "ro.bpevocab")
                if (ROOT / "neural" / "ro.bpevocab").exists() else None,
                "learner": "tools/neural/bpe.py (incremental; identical merges to the official train_bpe, verified)"},
        "training": {k: cfg[k] for k in ("dim", "hidden", "layers", "window", "stride", "mix", "epochs", "total_steps",
                                          "world", "threads", "batch", "lr", "warmup", "lr_floor", "clip", "seed") if k in cfg},
        "training_result": final,
        "official_trainer_sha256": OFFICIAL_SHA256,
        "reproduce": [
            "python3 tools/neural/corpus.py",
            "python3 tools/neural/bpe.py work/neural/ro_sentences.tsv work/neural/ro.bpevocab --merges 12000",
            "python3 tools/neural/encode.py work/neural/ro.bpevocab",
            "python3 tools/neural/train.py --run <name> <settings above>",
            "python3 tools/neural/export.py work/neural/runs/<name>/final.pt",
        ],
    }
    dump(args.out, receipt)
    print(f"wrote {args.out}: {lines} sentences")


if __name__ == "__main__":
    main()
