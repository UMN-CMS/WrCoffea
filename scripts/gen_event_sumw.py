#!/usr/bin/env python3
"""
check_or_update_genEventSumw.py

Given a JSON like:
  {
    "<DAS_DATASET>": {
      "das_name": "<DAS_DATASET>",
      ...
      "genEventSumw": <number>,
      ...
    },
    ...
  }

This script:
  - (optionally) filters entries by payload["physics_group"] == --physics-group
  - queries DAS for the dataset's files
  - opens files via FNAL redirector (root://cmsxrootd.fnal.gov//)
  - computes sum(Runs.genEventSumw) across all files
  - compares with the JSON value (tolerance-based)
  - updates payload["genEventSumw"] if different (or missing)
  - writes output JSON WITHOUT reordering any keys (keeps existing key order)

Requirements:
  - dasgoclient in PATH (cmsenv)
  - python: uproot, numpy

Examples:
  python3 gen_event_sumw.py -i dy.json -o dy_checked.json
  python3 gen_event_sumw.py -i mc.json -o mc_checked.json --physics-group DYJets
  python3 gen_event_sumw.py -i mc.json -o mc_checked.json --dry-run --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import uproot


logger = logging.getLogger(__name__)

FNAL_REDIRECTOR = "root://cmsxrootd.fnal.gov//"


def das_query_lines(query: str) -> List[str]:
    if shutil.which("dasgoclient") is None:
        logger.critical("dasgoclient not found in PATH. Run inside a CMSSW environment (cmsenv).")
        sys.exit(1)

    cmd = ["dasgoclient", "--query", query]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.warning("dasgoclient failed for query: %s\n%s", query, e.output.strip())
        return []
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def das_files_for_dataset(dataset: str) -> List[str]:
    return das_query_lines(f"file dataset={dataset}")


def open_with_retries(
    url: str,
    timeout: int,
    retries: int,
    sleep_s: float,
) -> Optional[uproot.ReadOnlyDirectory]:
    last_err: Optional[str] = None
    for attempt in range(1, retries + 1):
        try:
            logger.debug("Opening (attempt %d/%d): %s", attempt, retries, url)
            return uproot.open(url, timeout=timeout)
        except Exception as e:
            last_err = str(e)
            logger.debug("Open failed: %s\n  %s", url, last_err)
            if attempt < retries:
                time.sleep(sleep_s)

    if last_err:
        logger.warning("Giving up on: %s\n  Last error: %s", url, last_err)
    return None


def file_genEventSumw(
    url: str,
    timeout: int,
    retries: int,
    sleep_s: float,
) -> float:
    f = open_with_retries(url, timeout=timeout, retries=retries, sleep_s=sleep_s)
    if f is None:
        return 0.0
    try:
        if "Runs" not in f:
            logger.warning("No 'Runs' tree in file: %s", url)
            return 0.0
        runs = f["Runs"]
        if "genEventSumw" not in runs.keys():
            logger.warning("No 'genEventSumw' branch in Runs for file: %s", url)
            return 0.0
        arr = runs["genEventSumw"].array(library="np")
        return float(np.sum(arr))
    except Exception as e:
        logger.warning("Failed reading genEventSumw from: %s\n  %s", url, e)
        return 0.0
    finally:
        try:
            f.close()
        except Exception:
            pass


def compute_dataset_genEventSumw(
    dataset: str,
    timeout: int,
    retries: int,
    sleep_s: float,
    max_files: int,
) -> Tuple[float, int]:
    """
    Returns (sumw, nfiles_used). If max_files>0, uses only first max_files files.
    """
    files = das_files_for_dataset(dataset)
    if not files:
        return (0.0, 0)

    if max_files > 0:
        files = files[:max_files]

    total = 0.0
    for i, lf in enumerate(files, start=1):
        url = FNAL_REDIRECTOR + lf.lstrip("/")
        logger.debug("[%d/%d] %s", i, len(files), lf)
        total += file_genEventSumw(url, timeout=timeout, retries=retries, sleep_s=sleep_s)

    return (total, len(files))


def nearly_equal(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate/update genEventSumw in config JSONs by summing from DAS files."
    )
    ap.add_argument("-i", "--input", required=True, help="Input JSON")
    ap.add_argument("-o", "--output", required=True, help="Output JSON (updated if needed)")
    ap.add_argument("--physics-group", default=None,
                    help="Only process entries whose payload['physics_group'] matches (e.g. DYJets, tt_tW)")
    ap.add_argument("--timeout", type=int, default=60, help="Per-file open timeout (seconds)")
    ap.add_argument("--retries", type=int, default=6, help="Retries per file")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between retries (seconds)")
    ap.add_argument("--max-files", type=int, default=0, help="Process only first N files per dataset (0=all)")
    ap.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for comparing sums")
    ap.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute tolerance for comparing sums")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes; still prints what would happen")
    ap.add_argument("--verbose", action="store_true", help="Show DEBUG-level messages (file-by-file progress)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if DAS has 0 files or computed sum is 0 for any processed entry")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    with open(args.input, "r") as f:
        data: Dict[str, Any] = json.load(f)

    updated = 0
    checked = 0
    skipped = 0

    # IMPORTANT: we preserve insertion order by:
    #  - not using sort_keys
    #  - updating payload dicts in place (only overwriting genEventSumw value)
    for top_key, payload in data.items():
        if not isinstance(payload, dict):
            skipped += 1
            continue

        if args.physics_group is not None:
            pg = payload.get("physics_group", None)
            if pg != args.physics_group:
                skipped += 1
                continue

        ds = payload.get("das_name", top_key)
        if not isinstance(ds, str) or not ds.startswith("/"):
            logger.warning("Skipping (no valid das_name): %s", top_key)
            skipped += 1
            continue

        checked += 1
        logger.info("Checking genEventSumw for: %s", ds)

        computed, nfiles = compute_dataset_genEventSumw(
            ds,
            timeout=args.timeout,
            retries=args.retries,
            sleep_s=args.sleep,
            max_files=args.max_files,
        )

        if nfiles == 0:
            logger.warning("DAS returned 0 files for: %s", ds)
            if args.strict:
                logger.critical("Strict mode: 0 files for %s", ds)
                sys.exit(1)
            continue

        if args.strict and computed == 0.0:
            logger.critical("Strict mode: computed genEventSumw=0 for %s (likely read failures)", ds)
            sys.exit(1)

        current = parse_float(payload.get("genEventSumw", None))
        if current is None:
            logger.warning("Missing/non-numeric genEventSumw in JSON for %s -> setting to computed=%.6g", ds, computed)
            if not args.dry_run:
                payload["genEventSumw"] = computed
            updated += 1
            continue

        if nearly_equal(current, computed, rel_tol=args.rel_tol, abs_tol=args.abs_tol):
            logger.info("  OK: JSON=%.6g  DAS=%.6g  (nfiles=%d)", current, computed, nfiles)
        else:
            logger.warning("  MISMATCH: JSON=%.6g  DAS=%.6g  (nfiles=%d) -> updating", current, computed, nfiles)
            if not args.dry_run:
                payload["genEventSumw"] = computed
            updated += 1

    logger.info("Done. checked=%d, updated=%d, skipped=%d", checked, updated, skipped)

    if args.dry_run:
        return

    with open(args.output, "w") as f:
        # No sort_keys => preserves existing key order for top-level and inner dicts
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
