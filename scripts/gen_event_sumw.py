#!/usr/bin/env python3
"""
gen_event_sumw.py

Given a JSON like:
  {
    "<DAS_DATASET>": {
      "das_name": "<DAS_DATASET>",
      ...
      ...
    },
    ...
  }

This script:
  - (optionally) filters entries by payload["physics_group"] == --physics-group
  - queries DAS for the dataset's files
  - opens files via redirector fallback (FNAL -> CERN global -> INFN by default)
  - computes sum(Runs.genEventSumw) across all files
  - reports it (the configs carry no genEventSumw: the skimmed filesets take
    it from the skims' Runs trees, and unskimmed runs sum genWeight themselves)
  - also collapses the ratio-valued theory-weight sums (LHEScaleSumw,
    LHEPdfSumw, PSSumw) as genEventSumw-weighted means over all files and
    writes them to the payload (disable with --no-lhe-sums).  Skims made
    before 2026-09 carry element-wise *sums* of these ratios, so this is the
    way to obtain correct values without re-skimming.
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
from dataclasses import dataclass, field
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


# Runs-tree branches NanoAOD stores as ratios to genEventSumw.
RATIO_RUNS_BRANCHES = ("LHEScaleSumw", "LHEPdfSumw", "PSSumw")


def weighted_lhe_sum(sumw_entries, rows) -> np.ndarray:
    """Return sum_i sumw_i * ratio_i (element-wise) over the Runs entries of one file.

    *sumw_entries* is the per-entry genEventSumw, *rows* the per-entry jagged
    ratio branch.  Dividing the total over all files by the total genEventSumw
    gives the dataset-level ratio (see :func:`finalize_lhe`).
    """
    sumw = np.asarray(sumw_entries, dtype=np.float64)
    rows = [np.asarray(r, dtype=np.float64) for r in (rows.tolist() if hasattr(rows, "tolist") else rows)]
    if len(rows) != len(sumw):
        raise ValueError(f"{len(rows)} ratio entries but {len(sumw)} genEventSumw entries")
    if not rows:
        return np.zeros(0, dtype=np.float64)
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError(f"ratio entries have different lengths {sorted(lengths)}")
    return (np.vstack(rows) * sumw[:, None]).sum(axis=0)


def finalize_lhe(weighted: Dict[str, np.ndarray], total_sumw: float) -> Dict[str, List[float]]:
    """Turn accumulated weighted sums into dataset-level ratios (lists for JSON)."""
    if total_sumw <= 0:
        return {}
    return {name: (np.asarray(v, dtype=np.float64) / total_sumw).tolist() for name, v in weighted.items()}


@dataclass
class FileReadResult:
    sumw: float
    bad: bool
    zero_events: bool
    # branch -> sum_i sumw_i * ratio_i for this file (see weighted_lhe_sum)
    lhe: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class DatasetSumwResult:
    sumw: float
    nfiles_used: int
    bad_files: int
    zero_event_files: int
    # branch -> weighted sum over every good file that carries the branch
    lhe: Dict[str, np.ndarray] = field(default_factory=dict)
    # branches present in some good files but not all (not written)
    lhe_incomplete: List[str] = field(default_factory=list)


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

        lhe: Dict[str, np.ndarray] = {}
        for bname in RATIO_RUNS_BRANCHES:
            if bname not in runs.keys():
                continue
            try:
                lhe[bname] = weighted_lhe_sum(arr, runs[bname].array(library="ak"))
            except Exception as e:
                logger.warning("Failed reading %s from %s: %s", bname, url, e)

        return FileReadResult(sumw=float(np.sum(arr)), bad=False, zero_events=zero_events, lhe=lhe)
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
            if i > 1:
                logger.info("  Fallback succeeded on redirector %d/%d (%s) for: %s", i, len(redirectors), redirector, lfn)
            return result
        if i < len(redirectors):
            logger.info("  Redirector %d/%d failed (%s), falling back to next for: %s", i, len(redirectors), redirector, lfn)
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
    exclude_lfns: Optional[List[str]] = None,
    workers: int = 1,
) -> DatasetSumwResult:
    """
    Returns dataset-level sumw and file status counts.
    If max_files>0, uses only first max_files files.
    With workers>1 the files are opened concurrently (the Runs-tree read is
    latency bound); the sums are accumulated in the calling thread in file
    order, so the result is identical to the sequential one.
    """
    files = das_files_for_dataset(dataset)
    if not files:
        return DatasetSumwResult(sumw=0.0, nfiles_used=0, bad_files=0, zero_event_files=0)

    if exclude_lfns:
        before = len(files)
        files = [f for f in files if not any(exc in f for exc in exclude_lfns)]
        n_excluded = before - len(files)
        if n_excluded:
            logger.info("Excluded %d/%d files matching --exclude-lfn", n_excluded, before)

    if max_files > 0:
        files = files[:max_files]

    total = 0.0
    bad_files = 0
    zero_event_files = 0
    good_files = 0
    lhe_sums: Dict[str, np.ndarray] = {}
    lhe_counts: Dict[str, int] = {}
    def _one(lf: str) -> FileReadResult:
        return file_genEventSumw_with_fallback(
            lf,
            redirectors=redirectors,
            timeout=timeout,
            retries=retries,
            sleep_s=sleep_s,
        )

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=workers)
        results_iter = pool.map(_one, files)
    else:
        pool = None
        results_iter = map(_one, files)
    for i, result in enumerate(results_iter, start=1):
        if i % 200 == 0 or i == len(files):
            logger.info("  [%d/%d files] %s", i, len(files), dataset.split("/")[1][:50])
        total += result.sumw
        if result.bad:
            bad_files += 1
        else:
            good_files += 1
            for bname, v in result.lhe.items():
                if bname in lhe_sums and lhe_sums[bname].shape != v.shape:
                    logger.warning("%s has inconsistent length across files of %s; dropping it", bname, dataset)
                    lhe_counts[bname] = -1
                    continue
                lhe_sums[bname] = lhe_sums.get(bname, 0.0) + v
                lhe_counts[bname] = lhe_counts.get(bname, 0) + 1
        if result.zero_events:
            zero_event_files += 1

    if pool is not None:
        pool.shutdown(wait=True)
    complete = {b: v for b, v in lhe_sums.items() if lhe_counts.get(b) == good_files}
    incomplete = sorted(b for b in lhe_sums if lhe_counts.get(b) != good_files)
    return DatasetSumwResult(
        sumw=total,
        nfiles_used=len(files),
        bad_files=bad_files,
        zero_event_files=zero_event_files,
        lhe=complete,
        lhe_incomplete=incomplete,
    )


def nearly_equal(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sum genEventSumw over a config's DAS files (reported only) and write the "
                    "theory-weight sums into the config."
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
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent file opens per dataset (default 1 = sequential; the result does not depend on it)")
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
    ap.add_argument(
        "--exclude-lfn",
        nargs="+",
        default=[],
        help="LFN(s) to exclude from the sumw calculation (substring match on filename)",
    )
    ap.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for comparing sums")
    ap.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute tolerance for comparing sums")
    ap.add_argument("--no-lhe-sums", action="store_true",
                    help="Do not compute/write the ratio-valued LHEScaleSumw/LHEPdfSumw/PSSumw entries")
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
    #  - updating payload dicts in place (only the theory-weight sums)
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
            exclude_lfns=args.exclude_lfn or None,
            workers=args.workers,
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

        if not args.no_lhe_sums:
            if result.lhe_incomplete:
                logger.warning("  %s present in only some files of %s; not written", result.lhe_incomplete, ds)
            for bname, ratios in finalize_lhe(result.lhe, computed).items():
                existing = payload.get(bname)
                same = (
                    isinstance(existing, list) and len(existing) == len(ratios)
                    and all(nearly_equal(float(a), float(b), rel_tol=args.rel_tol, abs_tol=args.abs_tol)
                            for a, b in zip(existing, ratios))
                )
                if same:
                    logger.info("  OK: %s (%d entries)", bname, len(ratios))
                    continue
                logger.warning("  %s: %s -> writing %d weighted-mean ratios (first: %.4f)",
                               bname, "MISMATCH" if existing is not None else "missing", len(ratios), ratios[0] if ratios else float("nan"))
                if not args.dry_run:
                    payload[bname] = ratios
                updated += 1

        logger.info("  DAS genEventSumw = %.6g (nfiles=%d); reported only, the configs do not carry it",
                    computed, nfiles)

    logger.info("Done. checked=%d, updated=%d, skipped=%d", checked, updated, skipped)

    if args.dry_run:
        return

    with open(args.config, "w") as f:
        # No sort_keys => preserves existing key order for top-level and inner dicts
        json.dump(data, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
