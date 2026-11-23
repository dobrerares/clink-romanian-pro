#!/usr/bin/env python3
"""Download one openly licensed source file into data/<subdir>/ and record it in data/MANIFEST.json.

Usage: fetch.py SUBDIR URL --licence "CC BY-SA 4.0" [--name FILE] [--sha1 HEX] [--note TEXT]
       fetch.py SUBDIR --copy LOCALPATH --url ORIGINAL_URL --licence ...   (record a local copy)

Records: url, local path, bytes, sha256 (and sha1 when the publisher lists one), licence,
retrieval time. Never executes anything it downloads.
"""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "MANIFEST.json"


def digest(path):
    h256, h1 = hashlib.sha256(), hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h256.update(block)
            h1.update(block)
    return h256.hexdigest(), h1.hexdigest()


def record(entry):
    import fcntl
    with open(DATA / ".manifest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
        manifest["files"] = [f for f in manifest["files"] if f["path"] != entry["path"]] + [entry]
        manifest["files"].sort(key=lambda f: f["path"])
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subdir")
    ap.add_argument("url", nargs="?")
    ap.add_argument("--copy", default=None)
    ap.add_argument("--url", dest="orig_url", default=None)
    ap.add_argument("--licence", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--sha1", default=None, help="publisher-listed SHA-1 to verify")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    url = args.url or args.orig_url
    if not url:
        raise SystemExit("need a URL")
    dest_dir = DATA / args.subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or (Path(args.copy).name if args.copy else url.rstrip("/").rsplit("/", 1)[-1])
    dest = dest_dir / name
    if args.copy:
        shutil.copyfile(args.copy, dest)
        how = f"copied from local {args.copy}"
    else:
        tmp = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "romanian-pro-data-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, 1 << 20)
        tmp.replace(dest)
        how = "downloaded"
    s256, s1 = digest(dest)
    if args.sha1 and args.sha1 != s1:
        raise SystemExit(f"SHA-1 mismatch for {dest}: {s1} != {args.sha1}")
    entry = {"path": str(dest.relative_to(DATA)), "url": url, "bytes": dest.stat().st_size, "sha256": s256,
             "licence": args.licence, "how": how,
             "retrieved": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if args.sha1:
        entry["publisher_sha1_verified"] = args.sha1
    if args.note:
        entry["note"] = args.note
    record(entry)
    print(json.dumps(entry, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
