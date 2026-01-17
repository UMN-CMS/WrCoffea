#!/usr/bin/env python3
"""
recalc_genEventSumw_drop_nevts.py

Input JSON format (top-level key is DAS dataset; value is metadata dict).
For each dataset entry:
  - Query DAS for all files in that dataset
  - Open each file via FNAL redirector root://cmsxrootd.fnal.gov//
  - Read Runs.genEventSumw and sum across files
  - Write the recomputed total back to "genEventSumw"
  - Delete "nevts" (if present)
  - Write output JSON with a consistent key order (nevts removed)

Requirements:
  - dasgoclient in PATH (cmsenv)
  - python packages: uproot, numpy

Examples:
  python3 recalc_genEventSumw_drop_nevts.py -i dy.json -o dy_out.json
  python3 recalc_genEventSumw_drop_nevts.py -i dy.json -o /tmp/out.json --max-datasets 2 --max-files 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import uproot


FNAL_REDIRECTOR = "root://cmsxrootd.fnal.gov//"

KEY_ORDER = [
    "das_name",
    "run",
    "year",
    "era",
    "dataset",
    "physics_group",
    "xsec",
    "total_uncertainty",
    "genEventSumw",
    "datatype",
]


def die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr)


def order_payload_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce KEY_ORDER first, then append any extra keys (should be none after dropping nevts,
    but this avoids losing info if your schema evolves).
    """
    ordered: Dict[str, Any] = {}
    for k in KEY_ORDER:
        if k in payload:
            ordered[k] = payload[k]
    for k, v in payload.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def das_query_lines(query: str) -> List[str]:
    if shutil.which("dasgoclient") is None:
        die("dasgoclient not found in PATH. Run inside a CMSSW environment (cmsenv).")

    cmd = ["dasgoclient", "--query", query]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        warn(f"dasgoclient failed for query: {query}\n{e.output.strip()}")
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def das_files_for_dataset(dataset: str) -> List[str]:
    return das_query_lines(f"file dataset={dataset}")


def open_with_retries(
    url: str,
    timeout: int,
    retries: int,
    sleep_s: float,
    verbose: bool = False,
) -> Optional[uproot.ReadOnlyDirectory]:
    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            if verbose:
                info(f"Opening (attempt {attempt}/{retries}): {url}")
            f = uproot.open(url, timeout=timeout)
            return f
        except Exception as e:
            last_err = str(e)
            if verbose:
                warn(f"Open failed: {url}\n  {last_err}")
            if attempt < retries:
                time.sleep(sleep_s)

    if last_err:
        warn(f"Giving up on: {url}\n  Last error: {last_err}")
    return None


def file_genEventSumw(
    url: str,
    timeout: int,
    retries: int,
    sleep_s: float,
    verbose: bool = False,
) -> float:
    f = open_with_retries(url, timeout=timeout, retries=retries, sleep_s=sleep_s, verbose=verbose)
    if f is None:
        return 0.0

    try:
        if "Runs" not in f:
            warn(f"No 'Runs' tree in file: {url}")
            return 0.0
        runs = f["Runs"]
        if "genEventSumw" not in runs.keys():
            warn(f"No 'genEventSumw' branch in Runs for file: {url}")
            return 0.0
        arr = runs["genEventSumw"].array(library="np")
        return float(np.sum(arr))
    except Exception as e:
        warn(f"Failed reading genEventSumw from: {url}\n  {e}")
        return 0.0
    finally:
        try:
            f.close()
        except Exception:
            pass


def dataset_genEventSumw(
    dataset: str,
    timeout: int,
    retries: int,
    sleep_s: float,
    max_files: int,
    verbose: bool,
) -> float:
    files = das_files_for_dataset(dataset)
    if not files:
        warn(f"No files found in DAS for dataset: {dataset}")
        return 0.0

    if max_files > 0:
        files = files[:max_files]

    total = 0.0
    for i, lf in enumerate(files, start=1):
        url = FNAL_REDIRECTOR + lf.lstrip("/")
        if verbose:
            info(f"[{i}/{len(files)}] {lf}")
        total += file_genEventSumw(url, timeout=timeout, retries=retries, sleep_s=sleep_s, verbose=verbose)
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input JSON")
    ap.add_argument("-o", "--output", required=True, help="Output JSON (nevts removed, genEventSumw recalculated)")
    ap.add_argument("--timeout", type=int, default=60, help="Per-file open timeout (seconds)")
    ap.add_argument("--retries", type=int, default=6, help="Retries per file")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between retries (seconds)")
    ap.add_argument("--max-datasets", type=int, default=0, help="Process only first N datasets (0=all)")
    ap.add_argument("--max-files", type=int, default=0, help="Process only first N files per dataset (0=all)")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if DAS file list is empty or if recomputed sum is 0 (usually indicates failures)")
    ap.add_argument("--sort-top-level", action="store_true",
                    help="Sort top-level dataset keys alphabetically in the output JSON")
    args = ap.parse_args()

    with open(args.input, "r") as f:
        data: Dict[str, Any] = json.load(f)

    items = list(data.items())
    if args.max_datasets and args.max_datasets > 0:
        items = items[: args.max_datasets]

    out: Dict[str, Any] = {}

    for idx, (key, payload) in enumerate(items, start=1):
        if not isinstance(payload, dict):
            die(f"Entry '{key}' is not an object/dict; found type={type(payload)}")

        ds = payload.get("das_name", key)
        if not isinstance(ds, str) or not ds.startswith("/"):
            warn(f"Skipping entry with non-dataset das_name: key={key}, das_name={ds}")
            # Still drop nevts if present and keep ordering
            new_payload = dict(payload)
            new_payload.pop("nevts", None)
            out[key] = order_payload_keys(new_payload)
            continue

        info(f"({idx}/{len(items)}) Recomputing genEventSumw for: {ds}")

        files = das_files_for_dataset(ds)
        if not files:
            warn(f"No files in DAS for dataset: {ds}")
            if args.strict:
                die(f"Strict mode: no files for {ds}")
            new_payload = dict(payload)
            new_payload["genEventSumw"] = 0.0
            new_payload.pop("nevts", None)
            out[key] = order_payload_keys(new_payload)
            continue

        total = dataset_genEventSumw(
            ds,
            timeout=args.timeout,
            retries=args.retries,
            sleep_s=args.sleep,
            max_files=args.max_files,
            verbose=args.verbose,
        )

        if args.strict and total == 0.0:
            die(f"Strict mode: computed genEventSumw=0 for {ds}")

        new_payload = dict(payload)
        new_payload["genEventSumw"] = float(total)
        new_payload.pop("nevts", None)  # delete nevts
        out[key] = order_payload_keys(new_payload)

        info(f"  -> genEventSumw = {total:.6g}")

    # If we didn't process all datasets (max-datasets), copy the rest with nevts removed but without recompute
    if len(items) != len(data):
        processed_keys = set(k for k, _ in items)
        for key, payload in data.items():
            if key in processed_keys:
                continue
            if not isinstance(payload, dict):
                out[key] = payload
                continue
            new_payload = dict(payload)
            new_payload.pop("nevts", None)
            out[key] = order_payload_keys(new_payload)

    if args.sort_top_level:
        out = {k: out[k] for k in sorted(out.keys())}

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)  # no sort_keys
        f.write("\n")

    info(f"Done. wrote: {args.output}")


if __name__ == "__main__":
    main()
