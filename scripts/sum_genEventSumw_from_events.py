#!/usr/bin/env python3
"""
sum_genEventSumw_from_events.py  (Py3.8+ compatible, with progress counter)

Reads a JSON (datasets -> files), opens each ROOT file, and computes genEventSumw
by summing Events.genWeight (float64) in chunks.

Usage:
  python sum_genEventSumw_from_events.py input.json [--workers 8] [--out out.json] [--step "200 MB"]

Notes:
- Uses uproot.iterate to stream genWeight in chunks, summing in float64.
- Shows per-file progress counter for each dataset.
- Gracefully skips files without Events or genWeight.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict
import threading

import numpy as np
import uproot


print_lock = threading.Lock()  # to avoid mixed stdout from parallel threads


def _find_branch_case_insensitive(tree, target: str) -> Optional[str]:
    """Return the exact branch name matching target (case-insensitive), else None."""
    t = target.lower()
    try:
        for name in tree.keys():
            if str(name).lower() == t:
                return str(name)
    except Exception:
        pass
    return None


def sum_genweight_in_file(file_url: str, idx: int, total_files: int,
                          step_size: str = "200 MB", timeout: int = 120) -> float:
    """
    Open a ROOT file (local/xrootd) and sum Events.genWeight in float64.
    Prints a progress line like "[3/57] Processing: filename.root"
    """
    with print_lock:
        short = file_url.split("/")[-1]
        print(f"[{idx}/{total_files}] Processing: {short}", flush=True)

    total = 0.0
    try:
        with uproot.open(file_url, timeout=timeout) as fin:
            if "Events" not in fin:
                with print_lock:
                    print(f"[warn] No 'Events' tree in file: {file_url}", file=sys.stderr)
                return 0.0
            events = fin["Events"]
            branch = _find_branch_case_insensitive(events, "genWeight")
            if branch is None:
                with print_lock:
                    print(f"[warn] No 'genWeight' branch in Events for: {file_url}", file=sys.stderr)
                return 0.0

        expr = [branch]
        for arrays in uproot.iterate(
            file_url + ":Events",
            expr,
            library="np",
            step_size=step_size,
            allow_missing=False,
        ):
            vals = arrays.get(branch)
            if vals is not None:
                total += float(np.sum(vals, dtype=np.float64))
        return total

    except Exception as e:
        with print_lock:
            print(f"[warn] Failed to read file: {file_url} (error: {e})", file=sys.stderr)
        return 0.0


def process_dataset(dataset_key: str, files_map: Dict[str, str],
                    workers: int, step_size: str) -> float:
    """Sum genWeight over all files in a dataset, with progress counter."""
    file_urls = list(files_map.keys())
    total_files = len(file_urls)
    total_sum = 0.0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {
            ex.submit(sum_genweight_in_file, url, i + 1, total_files, step_size): url
            for i, url in enumerate(file_urls)
        }
        for fut in as_completed(futures):
            try:
                total_sum += fut.result()
            except Exception as e:
                with print_lock:
                    print(f"[warn] Unexpected failure: {e}", file=sys.stderr)

    return total_sum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_in", help="Input JSON file (datasets -> files)")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    ap.add_argument("--step", type=str, default="200 MB", help="Chunk step size for uproot.iterate (default: '200 MB')")
    ap.add_argument("--out", type=str, default="", help="Optional output JSON path")
    args = ap.parse_args()

    with open(args.json_in, "r") as f:
        data = json.load(f)

    results: Dict[str, float] = {}
    grand_total = 0.0

    for dataset_key, payload in data.items():
        files_map = payload.get("files", {})
        if not files_map:
            print(f"[warn] No files listed for dataset: {dataset_key}", file=sys.stderr)
            results[dataset_key] = 0.0
            continue

        print(f"\n=== Processing dataset: {dataset_key} ===")
        ds_sum = process_dataset(dataset_key, files_map, args.workers, args.step)
        results[dataset_key] = ds_sum
        grand_total += ds_sum

        print(f"{dataset_key}\n  genEventSumw_total (from Events.genWeight) = {ds_sum}")

    if args.out:
        try:
            with open(args.out, "w") as fout:
                json.dump(
                    {
                        "per_dataset_genEventSumw": results,
                        "grand_total_genEventSumw": grand_total,
                        "method": "sum_over_Events.genWeight",
                        "step_size": args.step,
                    },
                    fout,
                    indent=2,
                )
            print(f"\n[info] Wrote results to: {args.out}")
        except Exception as e:
            print(f"[warn] Failed to write '{args.out}': {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
