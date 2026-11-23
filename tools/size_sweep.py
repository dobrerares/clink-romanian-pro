#!/usr/bin/env python3
"""Dictionary-size sweep on the DEV split: builds a CLEX (official build-pack.py) for each size
from the ranked list and runs the typo-correction proxy (Combined mode and pack alone).

Usage: size_sweep.py RANKED_WORDLIST --sizes 75000,100000,... [--seeds 1,2,3] [--n 600]
Writes build/size-sweep.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, WORK  # noqa: E402
from evaluate import EN_CLEX, DictCorrector, Pack, ci, load_english, load_set, sample, summarize_typo, typo_run  # noqa: E402
from packio import Clex  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranked")
    ap.add_argument("--sizes", default="75000,100000,125000,150000,200000,250000,300000")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    sweep = WORK / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    lines = Path(args.ranked).read_text(encoding="utf-8").splitlines()
    en = Clex(EN_CLEX).freq()
    seeds = [int(s) for s in args.seeds.split(",")]
    sets = {s: load_set(s, "dev") for s in ("tatoeba", "cv", "wikitalk", "wiki")}
    sets["english"] = load_english("dev")
    results = []
    for size in [int(x) for x in args.sizes.split(",")]:
        src = sweep / f"ro-{size}.txt"
        src.write_text("\n".join(lines[:size]) + "\n", encoding="utf-8")
        code = f"sweep{size}"
        subprocess.run([sys.executable, "-I", str(ROOT / "vendor" / "build-pack.py"), code, str(src)], cwd=ROOT,
                       check=True, capture_output=True)
        shutil.move(ROOT / "Lexicons" / f"{code}.clex", sweep / f"{code}.clex")
        pack = Pack(code, sweep / code)
        row = {"size": size, "clex_bytes": pack.files["clex"]}
        for cfg in ("combined", "ro"):
            corr = {code: DictCorrector(pack.freq, en if cfg == "combined" else None)}
            for setname, sents_all in sets.items():
                if setname == "english" and cfg == "ro":
                    continue
                for mode in (["en"] if setname == "english" else ["ro_plain", "ro_diac"]):
                    vals = defaultdict(list)
                    for seed in seeds:
                        st = typo_run(sample(sents_all, args.n, seed), "en" if setname == "english" else "ro",
                                      mode, "gaussian", seed, corr)
                        for k, v in summarize_typo(st[code]).items():
                            vals[k].append(v)
                    row[f"{cfg}/{setname}/{mode}"] = {k: round(ci(v)[0], 4) for k, v in vals.items()}
        results.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "build" / "size-sweep.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
