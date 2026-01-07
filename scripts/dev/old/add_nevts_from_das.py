#!/usr/bin/env python3
"""
Add 'nevts' to each entry in a dataset JSON using DAS file metadata.

Input JSON format (top-level object mapping dataset -> metadata dict):
{
  "/.../NANOAODSIM": {
    "das_name": "/.../NANOAODSIM",
    "run": "Run3",
    "year": "2022",
    "era": "Run3Summer22",
    "sample": "...",
    "physics_group": "TW",
    "xsec": 4.66797,
    "datatype": "mc"
  },
  ...
}

Usage:
  python3 add_nevts_from_das.py --in samples.json --out samples_with_nevts.json

Notes:
  - Requires `dasgoclient` in PATH (CMS env).
  - We sum per-file `nevents` (no ROOT I/O).
"""

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Tuple, Set

def run(cmd: List[str]) -> str:
    try:
        out = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return out.stdout
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[dasgoclient error]\n{e.stderr}\n")
        raise

def parse_das_json(stdout: str) -> List[Any]:
    # dasgoclient may return a single JSON doc, a list, or JSONL
    try:
        blob = json.loads(stdout)
        return blob if isinstance(blob, list) else [blob]
    except json.JSONDecodeError:
        docs = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return docs

def extract_nevents_anywhere(obj: Any) -> Iterable[int]:
    if isinstance(obj, dict):
        if "nevents" in obj and isinstance(obj["nevents"], int):
            yield obj["nevents"]
        for v in obj.values():
            yield from extract_nevents_anywhere(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from extract_nevents_anywhere(it)

def sum_file_nevents(dataset: str) -> int:
    # Ask for file-level nevents; JSON for robust parsing
    query = f'file dataset={dataset} | grep file.nevents'
    stdout = run(["dasgoclient", "-json", "-query", query])
    docs = parse_das_json(stdout)

    seen: Set[int] = set()
    total = 0
    for doc in docs:
        for nev in extract_nevents_anywhere(doc):
            # Dedup rare repeats; sum integers >=0
            if isinstance(nev, int) and nev >= 0:
                # Can't dedup by file name here since we only requested nevents;
                # but JSON per-file entries are unique in practice. We just sum.
                total += nev
    return total

def main():
    ap = argparse.ArgumentParser(description="Add 'nevts' to dataset JSON via DAS file.nevents sum.")
    ap.add_argument("--in", dest="in_json", required=True, help="Input JSON file (mapping dataset -> metadata dict)")
    ap.add_argument("--out", dest="out_json", required=True, help="Output JSON file")
    args = ap.parse_args()

    if shutil.which("dasgoclient") is None:
        sys.stderr.write("ERROR: dasgoclient not found in PATH. Source your CMS env.\n")
        sys.exit(2)

    with open(args.in_json) as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    updated: Dict[str, Dict[str, Any]] = {}
    for key, meta in data.items():
        das_name = meta.get("das_name") or key
        if not isinstance(das_name, str):
            sys.stderr.write(f"WARNING: Skipping '{key}' (no valid 'das_name').\n")
            updated[key] = meta
            continue
        try:
            total = sum_file_nevents(das_name)
            meta = dict(meta)  # shallow copy
            meta["nevts"] = total
            updated[key] = meta
            print(f"{total:12d}  {das_name}")
        except Exception:
            sys.stderr.write(f"ERROR: Failed to query '{das_name}'. Leaving entry unchanged.\n")
            updated[key] = meta

    with open(args.out_json, "w") as f:
        json.dump(updated, f, indent=2, sort_keys=False)
    print(f"\nWrote updated JSON to {args.out_json}")

if __name__ == "__main__":
    main()
