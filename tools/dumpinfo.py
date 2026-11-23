#!/usr/bin/env python3
"""Print file names, sizes and SHA-1s of a Wikimedia dump run. Usage: dumpinfo.py WIKI DATE"""
import json
import sys
import urllib.request

for wiki in sys.argv[1:-1]:
    date = sys.argv[-1]
    url = f"https://dumps.wikimedia.org/{wiki}/{date}/dumpstatus.json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "romanian-pro-data-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001
        print(wiki, "ERROR", e)
        continue
    for job in ("articlesdumprecombine", "articlesdump", "metacurrentdumprecombine", "metacurrentdump"):
        j = d["jobs"].get(job, {})
        for f, v in j.get("files", {}).items():
            print(wiki, job, j.get("status"), f, v.get("size"), v.get("sha1"))
