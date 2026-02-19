#!/usr/bin/env python3
"""
gen_event_sumw.py

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
  - opens files via redirector fallback (FNAL -> CERN global -> INFN by default)
  - computes sum(Runs.genEventSumw) across all files
  - compares with the JSON value (tolerance-based)
  - updates payload["genEventSumw"] if different (or missing)
  - writes the same config JSON in place WITHOUT reordering any keys
  - fails fast on unreadable/corrupt files by default (opt-in lenient mode available)

Requirements:
  - dasgoclient in PATH (cmsenv)
  - python: uproot, numpy

Examples:
  python3 gen_event_sumw.py --config dy.json
  python3 gen_event_sumw.py --config mc.json --physics-group DYJets
  python3 gen_event_sumw.py --config mc.json --dry-run --verbose
  python3 gen_event_sumw.py --config mc.json --allow-file-failures --max-bad-files 3
  python3 gen_event_sumw.py --config mc.json --redirectors root://cmsxrootd.fnal.gov// root://cms-xrd-global.cern.ch// root://xrootd-cms.infn.it//
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import uproot


logger = logging.getLogger(__name__)

DEFAULT_REDIRECTORS = [
    "root://cmsxrootd.fnal.gov//",
    "root://cms-xrd-global.cern.ch//",
    "root://xrootd-cms.infn.it//",
]


@dataclass
class FileReadResult:
    sumw: float
    bad: bool
    zero_events: bool


@dataclass
class DatasetSumwResult:
    sumw: float
    nfiles_used: int
    bad_files: int
    zero_event_files: int


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
) -> FileReadResult:
    f = open_with_retries(url, timeout=timeout, retries=retries, sleep_s=sleep_s)
    if f is None:
        return FileReadResult(sumw=0.0, bad=True, zero_events=False)
    try:
        if "Runs" not in f:
            logger.warning("No 'Runs' tree in file: %s", url)
            return FileReadResult(sumw=0.0, bad=True, zero_events=False)
        runs = f["Runs"]
        if "genEventSumw" not in runs.keys():
            logger.warning("No 'genEventSumw' branch in Runs for file: %s", url)
            return FileReadResult(sumw=0.0, bad=True, zero_events=False)
        arr = runs["genEventSumw"].array(library="np")
        # Empty Runs.genEventSumw arrays are treated as valid zero-event files.
        if arr.size == 0:
            return FileReadResult(sumw=0.0, bad=False, zero_events=True)

        zero_events = False
        if "Events" in f:
            try:
                zero_events = f["Events"].num_entries == 0
            except Exception as e:
                logger.debug("Could not inspect Events tree for %s: %s", url, e)

        return FileReadResult(sumw=float(np.sum(arr)), bad=False, zero_events=zero_events)
    except Exception as e:
        logger.warning("Failed reading genEventSumw from: %s\n  %s", url, e)
        return FileReadResult(sumw=0.0, bad=True, zero_events=False)
    finally:
        try:
            f.close()
        except Exception:
            pass


def file_genEventSumw_with_fallback(
    lfn: str,
    redirectors: List[str],
    timeout: int,
    retries: int,
    sleep_s: float,
) -> FileReadResult:
    last_result: Optional[FileReadResult] = None
    clean_lfn = lfn.lstrip("/")

    for i, redirector in enumerate(redirectors, start=1):
        url = redirector + clean_lfn
        logger.debug("Trying redirector (%d/%d): %s", i, len(redirectors), redirector)
        result = file_genEventSumw(url, timeout=timeout, retries=retries, sleep_s=sleep_s)
        if not result.bad:
            return result
        last_result = result

    # At least one redirector is guaranteed by argument validation in main().
    return last_result if last_result is not None else FileReadResult(sumw=0.0, bad=True, zero_events=False)


def compute_dataset_genEventSumw(
    dataset: str,
    redirectors: List[str],
    timeout: int,
    retries: int,
    sleep_s: float,
    max_files: int,
) -> DatasetSumwResult:
    """
    Returns dataset-level sumw and file status counts.
    If max_files>0, uses only first max_files files.
    """
    files = das_files_for_dataset(dataset)
    if not files:
        return DatasetSumwResult(sumw=0.0, nfiles_used=0, bad_files=0, zero_event_files=0)

    if max_files > 0:
        files = files[:max_files]

    total = 0.0
    bad_files = 0
    zero_event_files = 0
    for i, lf in enumerate(files, start=1):
        logger.debug("[%d/%d] %s", i, len(files), lf)
        result = file_genEventSumw_with_fallback(
            lf,
            redirectors=redirectors,
            timeout=timeout,
            retries=retries,
            sleep_s=sleep_s,
        )
        total += result.sumw
        if result.bad:
            bad_files += 1
        if result.zero_events:
            zero_event_files += 1

    return DatasetSumwResult(
        sumw=total,
        nfiles_used=len(files),
        bad_files=bad_files,
        zero_event_files=zero_event_files,
    )


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
    ap.add_argument("--config", required=True, help="Config JSON to read/update in place")
    ap.add_argument("--physics-group", default=None,
                    help="Only process entries whose payload['physics_group'] matches (e.g. DYJets, tt_tW)")
    ap.add_argument("--dataset", default=None,
                    help="Only process the entry whose das_name (or top-level key) contains this substring")
    ap.add_argument(
        "--redirectors",
        nargs="+",
        default=list(DEFAULT_REDIRECTORS),
        help="Redirector fallback order (each redirector gets its own retry loop)",
    )
    ap.add_argument("--timeout", type=int, default=30, help="Per-file open timeout (seconds)")
    ap.add_argument("--retries", type=int, default=10, help="Retries per redirector, per file")
    ap.add_argument("--sleep", type=float, default=10.0, help="Seconds to wait between retries")
    ap.add_argument("--max-files", type=int, default=0, help="Process only first N files per dataset (0=all)")
    ap.add_argument(
        "--allow-file-failures",
        action="store_true",
        help="Continue when some files are unreadable/corrupt/malformed (default: fail fast)",
    )
    ap.add_argument(
        "--max-bad-files",
        type=int,
        default=0,
        help="Maximum bad files allowed per dataset when --allow-file-failures is set",
    )
    ap.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for comparing sums")
    ap.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute tolerance for comparing sums")
    ap.add_argument("--dry-run", action="store_true", help="Do not write changes; still prints what would happen")
    ap.add_argument("--verbose", action="store_true", help="Show DEBUG-level messages (file-by-file progress)")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if DAS has 0 files, any bad files, or computed sum is 0 for any processed entry")
    args = ap.parse_args()
    if args.max_bad_files < 0:
        ap.error("--max-bad-files must be >= 0")
    if args.retries < 1:
        ap.error("--retries must be >= 1")
    if args.timeout < 1:
        ap.error("--timeout must be >= 1")
    if args.sleep < 0:
        ap.error("--sleep must be >= 0")

    redirectors = [r.rstrip("/") + "//" for r in args.redirectors if r.strip()]
    if not redirectors:
        ap.error("--redirectors must contain at least one non-empty value")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger.info("Using redirectors (in order): %s", ", ".join(redirectors))

    with open(args.config, "r") as f:
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

        if args.dataset is not None and args.dataset not in ds:
            skipped += 1
            continue

        checked += 1
        logger.info("Checking genEventSumw for: %s", ds)

        result = compute_dataset_genEventSumw(
            ds,
            redirectors=redirectors,
            timeout=args.timeout,
            retries=args.retries,
            sleep_s=args.sleep,
            max_files=args.max_files,
        )
        computed = result.sumw
        nfiles = result.nfiles_used
        bad_files = result.bad_files
        zero_event_files = result.zero_event_files

        if nfiles == 0:
            logger.warning("DAS returned 0 files for: %s", ds)
            if args.strict:
                logger.critical("Strict mode: 0 files for %s", ds)
                sys.exit(1)
            continue

        if bad_files > 0:
            if args.strict:
                logger.critical("Strict mode: %d/%d files failed for %s", bad_files, nfiles, ds)
                sys.exit(1)
            if not args.allow_file_failures:
                logger.critical(
                    "%d/%d files failed for %s. Re-run with --allow-file-failures to proceed.",
                    bad_files, nfiles, ds,
                )
                sys.exit(1)
            if bad_files > args.max_bad_files:
                logger.critical(
                    "%d/%d files failed for %s, which exceeds --max-bad-files=%d",
                    bad_files, nfiles, ds, args.max_bad_files,
                )
                sys.exit(1)
            logger.warning(
                "Proceeding with %d/%d bad files for %s due to --allow-file-failures (max=%d).",
                bad_files, nfiles, ds, args.max_bad_files,
            )

        if zero_event_files > 0:
            logger.info(
                "  Note: %d/%d files have zero events (valid, contributes 0 to genEventSumw).",
                zero_event_files,
                nfiles,
            )

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

    with open(args.config, "w") as f:
        # No sort_keys => preserves existing key order for top-level and inner dicts
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
