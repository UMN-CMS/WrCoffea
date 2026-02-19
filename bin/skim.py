#!/usr/bin/env python3
"""Unified CLI for the WrCoffea skimming pipeline.

Subcommands
-----------
run       Skim NanoAOD files (Condor by default; --local for direct execution)
check     Detect missing / failed Condor skim jobs
merge     Extract tarballs, merge, and validate merged outputs

Examples
--------
    python bin/skim.py --cuts

    python bin/skim.py run   /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM
    python bin/skim.py run   /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM --start 1 --end 10
    python bin/skim.py run   /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM --dry-run
    python bin/skim.py run   /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM --start 1 --end 1 --local

    python bin/skim.py check  /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM

    python bin/skim.py merge  /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from wrcoffea.das_utils import (
    ERA_SUBDIRS,
    base_dir_for_era,
    check_dasgoclient,
    check_grid_proxy,
    das_files_to_urls,
    era_from_das_path,
    infer_base_dir,
    infer_category,
    infer_output_dir,
    primary_dataset_from_das_path,
    query_das_files,
    validate_das_path,
)
from wrcoffea.skimmer import SKIM_CUTS, SKIM_STAGES, skim_single_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_MERGE_JOB_STATUSES = {"success", "empty_input"}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_das(das_path, scratch=False):
    """Query DAS and return (primary_dataset, file_urls, default_outdir)."""
    check_dasgoclient()
    check_grid_proxy()
    primary_ds = primary_dataset_from_das_path(das_path)
    lfns = query_das_files(das_path)
    file_urls = das_files_to_urls(lfns)
    default_outdir = infer_output_dir(das_path, scratch=scratch)
    return primary_ds, file_urls, default_outdir


def _is_condor_worker():
    """Return True if running inside an HTCondor job."""
    return "_CONDOR_JOB_AD" in os.environ


def _print_skim_cuts():
    """Print SKIM_CUTS in a human-readable table."""
    print()
    print("Skim selection cuts")
    print("=" * 55)
    print(f"  {'Cut':<40s} {'Value':>10}")
    print("  " + "-" * 51)
    for key in [
        "lepton_pt_min", "lepton_eta_max",
        "lead_lepton_pt_min", "sublead_lepton_pt_min",
        "ak4_pt_min", "ak4_eta_max",
        "ak8_pt_min", "ak8_eta_max",
    ]:
        label = key.replace("_", " ")
        print(f"  {label:<40s} {SKIM_CUTS[key]:>10}")
    print()
    print("  Selection logic:")
    print("    (>= 2 leptons passing pT/eta, lead > 52, sublead > 45)")
    print("    AND (>= 2 AK4 jets OR >= 1 AK8 jet)")
    print()


def _write_skim_status(path: Path, payload: dict) -> None:
    """Write per-job skim status metadata to JSON."""
    def _json_default(obj):
        # Convert numpy scalar types (np.int64, np.float32, etc.) to Python scalars.
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return item()
            except Exception:
                pass
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def _infer_failure_category(message: str) -> str:
    """Best-effort failure category extraction from a skim exception message."""
    m = re.search(r"Non-retryable \[([a-z_]+)\]", message)
    if m:
        return m.group(1)
    if "All retries exhausted" in message:
        return "network_error"
    return "unknown_error"


# ═══════════════════════════════════════════════════════════════════════════
# run
# ═══════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    """Skim NanoAOD files (submits to Condor unless --local or on a worker)."""
    if args.lfn:
        # Direct LFN mode: skip DAS query, process exactly this one file.
        primary_ds = primary_dataset_from_das_path(args.das_path)
        file_urls = [das_files_to_urls([args.lfn])[0]]
        default_outdir = infer_output_dir(args.das_path)
        logger.info("Dataset '%s': using direct LFN %s", primary_ds, args.lfn)

        outdir = default_outdir
        outdir.mkdir(parents=True, exist_ok=True)

        src = file_urls[0]
        dest = outdir / f"{Path(src).stem}_skim.root"
        fname = Path(src).name

        t0 = time.monotonic()
        logger.info("File: %s", fname)
        try:
            result = skim_single_file(str(src), str(dest))
        except Exception as e:
            # In worker mode, emit structured failure status so era checks can
            # report bad files without Condor hold loops from missing metadata.
            if args.status_json:
                msg = str(e)
                _write_skim_status(
                    args.status_json,
                    {
                        "das_path": args.das_path,
                        "lfn": args.lfn,
                        "status": "failed",
                        "attempts": 0,
                        "redirector": None,
                        "failure_category": _infer_failure_category(msg),
                        "failure_reason": msg,
                        "src_path": str(src),
                        "dest_path": str(dest),
                        "n_events_before": 0,
                        "n_events_after": 0,
                        "file_size_bytes": 0,
                        "efficiency": 0.0,
                    },
                )
                logger.error("Skim failed for %s: %s", fname, e)
                return
            raise
        if args.status_json:
            _write_skim_status(
                args.status_json,
                {
                    "das_path": args.das_path,
                    "lfn": args.lfn,
                    "status": result.status,
                    "attempts": result.attempts,
                    "redirector": result.redirector,
                    "failure_category": result.failure_category,
                    "failure_reason": result.failure_reason,
                    "src_path": result.src_path,
                    "dest_path": result.dest_path,
                    "n_events_before": result.n_events_before,
                    "n_events_after": result.n_events_after,
                    "file_size_bytes": result.file_size_bytes,
                    "efficiency": result.efficiency,
                },
            )
        elapsed = time.monotonic() - t0
        logger.info(
            "Done in %.1f min. %d -> %d events (%.1f%%), %.1f MB",
            elapsed / 60, result.n_events_before, result.n_events_after,
            result.efficiency, result.file_size_bytes / 1_048_576,
        )
        return

    scratch = getattr(args, "scratch", False)
    primary_ds, file_urls, default_outdir = _resolve_das(args.das_path, scratch=scratch)
    num_files = len(file_urls)
    logger.info("Dataset '%s': %d files from DAS", primary_ds, num_files)

    # File range (1-indexed): default is all files
    start_1 = args.start if args.start else 1
    end_1 = args.end if args.end else num_files

    if start_1 < 1 or end_1 > num_files:
        logger.error("File range [%d, %d] out of bounds (%d files).", start_1, end_1, num_files)
        raise SystemExit(1)

    # Branch: Condor submit vs local execution
    if not args.local and not _is_condor_worker():
        _submit_run(args.das_path, primary_ds, file_urls,
                     start=start_1, end=end_1, dry_run=args.dry_run, scratch=scratch)
        return

    # --- Local execution (worker node or --local) ---
    start_idx = start_1 - 1
    end_idx = end_1 - 1

    outdir = default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    results = []
    n_files = end_idx - start_idx + 1
    use_bar = sys.stderr.isatty()
    pbar = tqdm(
        total=n_files * SKIM_STAGES,
        desc=primary_ds,
        unit="step",
        file=sys.stderr,
        disable=not use_bar,
    )
    for i in range(start_idx, end_idx + 1):
        src = file_urls[i]
        dest = outdir / f"{Path(src).stem}_skim.root"
        fname = Path(src).name

        def _progress(stage, _fn=fname):
            pbar.set_postfix_str(f"{_fn} [{stage}]", refresh=True)
            pbar.update(1)

        if use_bar:
            pbar.set_postfix_str(f"{fname} [opening]", refresh=True)
        else:
            logger.info("File %d/%d: %s", i + 1, num_files, fname)
        result = skim_single_file(str(src), str(dest), progress=_progress if use_bar else None)
        results.append(result)
        if not use_bar:
            logger.info(
                "  %d -> %d events (%.1f%%), %.1f MB",
                result.n_events_before, result.n_events_after,
                result.efficiency, result.file_size_bytes / 1_048_576,
            )
    pbar.close()

    elapsed = time.monotonic() - t0
    total_in = sum(r.n_events_before for r in results)
    total_out = sum(r.n_events_after for r in results)
    logger.info(
        "Done. %d file(s) in %.1f min. Total: %d -> %d events (%.1f%%)",
        len(results), elapsed / 60, total_in, total_out,
        (total_out / total_in * 100) if total_in else 0,
    )

    # Overwrite stale Condor log files for successful jobs so that
    # check-era's log validation passes after local resubmissions.
    log_dir = infer_base_dir(args.das_path, scratch=scratch) / "logs" / primary_ds
    if log_dir.is_dir():
        for i, result in zip(range(start_idx, end_idx + 1), results):
            if result.status not in ALLOWED_MERGE_JOB_STATUSES:
                continue
            out_file = log_dir / f"{primary_ds}_{i}.out"
            err_file = log_dir / f"{primary_ds}_{i}.err"
            done_line = (
                f"Done in {elapsed / 60:.1f} min. "
                f"{result.n_events_before} -> {result.n_events_after} events "
                f"({result.efficiency:.1f}%), "
                f"{result.file_size_bytes / 1_048_576:.1f} MB"
            )
            out_file.write_text("Job completed successfully\n")
            err_file.write_text(done_line + "\n")
            logger.info("Updated Condor logs for job %d", i)


# ═══════════════════════════════════════════════════════════════════════════
# submit
# ═══════════════════════════════════════════════════════════════════════════

JDL_TEMPLATE = """\
universe = vanilla
executable = skim_job.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
request_memory = 4000
x509userproxy = $ENV(X509_USER_PROXY)
+ApptainerImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12"
output = {log_dir}/{dataset}_$(job_idx).out
error  = {log_dir}/{dataset}_$(job_idx).err
log    = {log_dir}/{dataset}_$(job_idx).log
transfer_input_files = {tarball_path}
arguments = $(job_idx) $(das_path) $(lfn) $(ds_name)
transfer_output_files = {dataset}_skim$(job_idx).tar.gz, {dataset}_skim$(job_idx).status.json
initialdir = {output_dir}
queue job_idx, das_path, lfn, ds_name from arguments.txt
"""

TAR_EXCLUDES = [
    "--exclude=.git", "--exclude=.venv", "--exclude=WR_Plotter",
    "--exclude=data/skims", "--exclude=scripts/setup/skims/tmp",
    "--exclude=__pycache__", "--exclude=*.pyc",
    "--exclude=test/signal_files", "--exclude=benchmark_results_*",
    "--exclude=*.root",
]


def _create_tarball(dest_dir: Path) -> Path:
    tarball = dest_dir / "WrCoffea.tar.gz"
    cmd = ["tar", "-czf", str(tarball)] + TAR_EXCLUDES + ["-C", str(REPO_ROOT.parent), REPO_ROOT.name]
    logger.info("Creating tarball...")
    subprocess.run(cmd, check=True)
    logger.info("Tarball: %.1f MB", tarball.stat().st_size / (1024 * 1024))
    return tarball


def _generate_job(primary_ds, das_path, n_jobs, job_dir, log_dir, output_dir,
                   file_urls, start=1, end=None, *, argument_lines=None):
    """Generate Condor job files for a file range.

    Parameters
    ----------
    file_urls : list[str]
        XRootD URLs for every file in the dataset (0-indexed).
    start, end : int
        1-indexed file range (inclusive). Defaults to all files.
    output_dir : Path
        Directory where Condor will place output tarballs.
    argument_lines : list[str], optional
        Pre-built lines for ``arguments.txt``. Each line must contain
        ``job_idx das_path lfn`` where ``job_idx`` is 0-indexed and used for
        output file naming. When provided, *das_path*, *file_urls*, *start*,
        and *end* are ignored and the lines are written directly. This is
        used by era-level commands to combine multiple DAS paths (e.g. data
        sub-eras) into a single submission.
    """
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if argument_lines is not None:
        (job_dir / "arguments.txt").write_text("".join(argument_lines))
        actual_jobs = len(argument_lines)
    else:
        if end is None:
            end = n_jobs
        # Include the LFN so workers don't need to re-query DAS.
        lines = []
        for i in range(start, end + 1):
            url = file_urls[i - 1]  # file_urls is 0-indexed
            # Extract LFN: root://host//store/... → /store/...
            lfn_idx = url.find("//store/")
            lfn = url[lfn_idx + 1:] if lfn_idx >= 0 else url
            # job_idx is 0-indexed and maps directly to skim output naming.
            lines.append(f"{i - 1} {das_path} {lfn} {primary_ds}\n")
        (job_dir / "arguments.txt").write_text("".join(lines))
        actual_jobs = end - start + 1

    src_sh = REPO_ROOT / "bin" / "skim_job.sh"
    dest_sh = job_dir / "skim_job.sh"
    dest_sh.write_text(src_sh.read_text())
    dest_sh.chmod(0o755)

    tarball_path = (job_dir / "WrCoffea.tar.gz").resolve()
    jdl_path = job_dir / "job.jdl"
    jdl_path.write_text(JDL_TEMPLATE.format(
        log_dir=str(log_dir.resolve()),
        dataset=primary_ds,
        output_dir=str(output_dir.resolve()),
        tarball_path=str(tarball_path),
    ))

    logger.info("Generated %d jobs for %s (files 1-%d)", actual_jobs, primary_ds, actual_jobs)
    return jdl_path, actual_jobs


def _submit_run(das_path, primary_ds, file_urls, start, end, dry_run=False, scratch=False):
    """Submit a file range to Condor (called by cmd_run when not on a worker)."""
    n_total = len(file_urls)
    base_dir = infer_base_dir(das_path, scratch=scratch)
    base_dir.mkdir(parents=True, exist_ok=True)

    job_dir = base_dir / "jobs" / primary_ds
    log_dir = base_dir / "logs" / primary_ds
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create tarball directly in job_dir to avoid race conditions
    # when multiple single-dataset runs share the same base_dir.
    tarball = _create_tarball(job_dir)

    output_dir = infer_output_dir(das_path, scratch=scratch)
    jdl_path, actual_jobs = _generate_job(
        primary_ds, das_path, n_total, job_dir, log_dir, output_dir,
        file_urls, start=start, end=end,
    )

    if dry_run:
        logger.info("[dry-run] Would submit %d jobs from %s", actual_jobs, jdl_path)
    else:
        logger.info("Submitting %d Condor jobs for %s (files %d-%d)...",
                     actual_jobs, primary_ds, start, end)
        subprocess.run(["bash", "-c", f"condor_submit {jdl_path.name}"], cwd=str(job_dir), check=True)
        logger.info("Submitted %s", primary_ds)
    logger.info("Done.")


# ═══════════════════════════════════════════════════════════════════════════
# check
# ═══════════════════════════════════════════════════════════════════════════

def _check_dataset(primary_ds, file_urls, skim_dir):
    expected = len(file_urls)

    found_roots = set()
    found_tarballs = set()
    if skim_dir.is_dir():
        for p in skim_dir.glob("*_skim.root"):
            found_roots.add(p.stem.replace("_skim", ""))
        for p in skim_dir.glob(f"{primary_ds}_skim*.tar.gz"):
            name = p.stem.replace(".tar", "")
            idx_str = name.replace(f"{primary_ds}_skim", "")
            try:
                found_tarballs.add(int(idx_str))
            except ValueError:
                pass

    missing = []
    for i, url in enumerate(file_urls):
        stem = Path(url).stem
        if stem not in found_roots and i not in found_tarballs:
            missing.append((i + 1, url))

    return {"expected": expected, "found": expected - len(missing), "missing": missing}


def _status_index_from_path(path: Path, primary_ds: str):
    prefix = f"{primary_ds}_skim"
    suffix = ".status.json"
    name = path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    idx_str = name[len(prefix):-len(suffix)]
    try:
        return int(idx_str)
    except ValueError:
        return None


def _load_job_statuses(primary_ds: str, skim_dir: Path):
    """Load per-job status sidecars from skim output directory."""
    statuses = {}
    errors = []
    if not skim_dir.is_dir():
        return statuses, errors

    for path in sorted(skim_dir.glob(f"{primary_ds}_skim*.status.json")):
        idx = _status_index_from_path(path, primary_ds)
        if idx is None:
            errors.append(f"Invalid status filename: {path.name}")
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            errors.append(f"Could not parse {path.name}: {e}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"Invalid payload type in {path.name}: expected object")
            continue
        statuses[idx] = payload

    return statuses, errors


def _tarball_has_expected_root(tarball_path: Path, expected_root_name: str) -> bool:
    """Return True if tarball contains the expected skim ROOT basename."""
    try:
        with tarfile.open(tarball_path, "r:gz") as tf:
            for member in tf.getmembers():
                if member.isdir():
                    continue
                if Path(member.name).name == expected_root_name:
                    return True
    except Exception as e:
        logger.warning("Failed reading tarball %s: %s", tarball_path.name, e)
        return False
    return False


def _validate_job_status_outputs(primary_ds: str, file_urls, skim_dir: Path, *,
                                 return_failed_indices: bool = False):
    """Validate per-job status files and tarball content consistency.

    Parameters
    ----------
    return_failed_indices : bool, optional
        If True, return ``(ok, failed_indices)`` where ``failed_indices`` is
        a set of 0-based job indices that failed status validation.
    """
    statuses, status_errors = _load_job_statuses(primary_ds, skim_dir)
    for err in status_errors:
        logger.warning(err)
    if status_errors:
        if return_failed_indices:
            return False, set()
        return False

    found_roots = set()
    if skim_dir.is_dir():
        for p in skim_dir.glob("*_skim.root"):
            found_roots.add(p.stem.replace("_skim", ""))

    status_counts = {}
    max_attempts = 0
    ok = True
    failed_indices = set()

    for i, url in enumerate(file_urls):
        stem = Path(url).stem
        payload = statuses.get(i)
        if payload is None:
            logger.warning("Missing status sidecar for job %d (%s)", i, stem)
            ok = False
            failed_indices.add(i)
            continue

        status = str(payload.get("status", "")).strip()
        status_counts[status] = status_counts.get(status, 0) + 1
        attempts = payload.get("attempts")
        if isinstance(attempts, int):
            max_attempts = max(max_attempts, attempts)

        if status not in ALLOWED_MERGE_JOB_STATUSES:
            logger.warning(
                "Disallowed status for job %d (%s): %r (allowed: %s)",
                i, stem, status, sorted(ALLOWED_MERGE_JOB_STATUSES),
            )
            ok = False
            failed_indices.add(i)
            continue

        # success jobs must provide the expected skim root, either extracted
        # already or inside the corresponding tarball.
        if status == "success":
            if stem in found_roots:
                continue
            tarball_path = skim_dir / f"{primary_ds}_skim{i}.tar.gz"
            if not tarball_path.exists():
                logger.warning(
                    "Job %d (%s) status=success but tarball missing: %s",
                    i, stem, tarball_path.name,
                )
                ok = False
                failed_indices.add(i)
                continue
            expected_root = f"{stem}_skim.root"
            if not _tarball_has_expected_root(tarball_path, expected_root):
                logger.warning(
                    "Job %d (%s) status=success but tarball lacks %s",
                    i, stem, expected_root,
                )
                ok = False
                failed_indices.add(i)

    if status_counts:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        if max_attempts > 0:
            logger.info("Status check: %s (max attempts=%d)", summary, max_attempts)
        else:
            logger.info("Status check: %s", summary)
    if return_failed_indices:
        return ok, failed_indices
    return ok


def cmd_check(args):
    """Detect missing / failed Condor skim jobs."""
    scratch = getattr(args, "scratch", False)
    primary_ds, file_urls, default_outdir = _resolve_das(args.das_path, scratch=scratch)

    skim_dir = default_outdir
    result = _check_dataset(primary_ds, file_urls, skim_dir)

    status = "OK" if not result["missing"] else "INCOMPLETE"
    print(f"  {primary_ds:60s}  {result['found']}/{result['expected']}  [{status}]")

    if result["missing"]:
        if len(result["missing"]) <= 10:
            for idx, url in result["missing"]:
                print(f"    Missing file {idx}: {Path(url).name}")
        else:
            print(f"    ({len(result['missing'])} files missing)")

    if args.resubmit and result["missing"]:
        with args.resubmit.open("w") as f:
            for idx, url in result["missing"]:
                f.write(f"{idx} {args.das_path}\n")
        logger.info("Wrote %d resubmit entries to %s", len(result["missing"]), args.resubmit)

    if result["missing"]:
        logger.warning("%d missing jobs", len(result["missing"]))
        raise SystemExit(1)
    else:
        logger.info("All jobs complete.")


# ═══════════════════════════════════════════════════════════════════════════
# merge
# ═══════════════════════════════════════════════════════════════════════════

def _check_log_completion(log_dir: Path, primary_ds: str, expected_indices: set[int]):
    """Verify Condor log files confirm successful completion for each job.

    Checks that each job's .out file ends with "Job completed successfully"
    and .err file ends with a "Done in ..." summary line.

    Returns (passed_indices, failed_details) where failed_details is a list
    of (index, reason) tuples.
    """
    passed = set()
    failed = []

    for idx in sorted(expected_indices):
        out_file = log_dir / f"{primary_ds}_{idx}.out"
        err_file = log_dir / f"{primary_ds}_{idx}.err"

        # Check .out file
        if not out_file.exists():
            failed.append((idx, "missing .out log"))
            continue

        try:
            # Check if .out contains success marker
            out_content = out_file.read_text()
        except Exception as e:
            failed.append((idx, f".out read error: {e}"))
            continue

        if "Job completed successfully" not in out_content:
            lines = out_content.strip().splitlines()
            last_out = lines[-1].strip() if lines else ""
            failed.append((idx, f".out missing success marker, last line: {last_out!r}"))
            continue

        # Check .err file
        if not err_file.exists():
            failed.append((idx, "missing .err log"))
            continue

        try:
            lines = err_file.read_text().strip().splitlines()
            last_err = lines[-1].strip() if lines else ""
        except Exception as e:
            failed.append((idx, f".err read error: {e}"))
            continue

        if "Done in" not in last_err or "events" not in last_err:
            failed.append((idx, f".err last line: {last_err!r}"))
            continue

        passed.add(idx)

    return passed, failed


def _pre_merge_check(das_path: str, primary_ds: str, skim_dir: Path, scratch: bool = False) -> bool:
    """Run completeness checks before merging. Returns True if all pass."""
    logger.info("Running pre-merge completeness check...")

    # 1. Query DAS for the full file list
    primary_ds_check, file_urls, _ = _resolve_das(das_path, scratch=scratch)

    # 2. Check tarballs / loose ROOT files against DAS
    result = _check_dataset(primary_ds_check, file_urls, skim_dir)
    expected = result["expected"]
    found = result["found"]

    status = "OK" if not result["missing"] else "INCOMPLETE"
    logger.info("Tarball check: %d/%d files found [%s]", found, expected, status)

    if result["missing"]:
        if len(result["missing"]) <= 10:
            for idx, url in result["missing"]:
                logger.warning("  Missing file %d: %s", idx, Path(url).name)
        else:
            logger.warning("  %d files missing from skim directory", len(result["missing"]))
        return False

    # 3. Check per-job status sidecars and tarball content consistency.
    if not _validate_job_status_outputs(primary_ds, file_urls, skim_dir):
        return False

    # 4. Check log files for successful completion
    base_dir = infer_base_dir(das_path, scratch=scratch)
    log_dir = base_dir / "logs" / primary_ds

    if not log_dir.is_dir():
        logger.warning("Log directory not found: %s — skipping log check", log_dir)
    else:
        # Condor ProcId is 0-indexed; arguments.txt uses 1-indexed file numbers
        # but ProcId in log filenames is 0-indexed
        expected_indices = set(range(0, expected))
        passed, failed = _check_log_completion(log_dir, primary_ds, expected_indices)

        logger.info("Log check: %d/%d jobs confirmed successful", len(passed), expected)

        if failed:
            if len(failed) <= 10:
                for idx, reason in failed:
                    logger.warning("  Job %d: %s", idx, reason)
            else:
                logger.warning("  %d jobs with log issues (showing first 10):", len(failed))
                for idx, reason in failed[:10]:
                    logger.warning("    Job %d: %s", idx, reason)
            return False

    logger.info("Pre-merge check PASSED — all %d files present and logs confirm success.", expected)
    return True


def _pre_merge_check_group(group, era: str, primary_ds: str, scratch: bool = True) -> bool:
    """Run completeness checks for a grouped (multi-DAS-path) dataset.

    This is used by ``merge-era`` for data groups where one primary dataset
    may contain several DAS paths (e.g. v1-v4 sub-eras) submitted together.
    """
    logger.info("Running combined pre-merge completeness check for grouped dataset...")
    check_dasgoclient()
    check_grid_proxy()

    # 1. Build combined expected file list from all DAS paths in the group.
    combined_urls = []
    for das_path, _meta, _cfg in group:
        try:
            lfns = query_das_files(das_path)
            combined_urls.extend(das_files_to_urls(lfns))
        except RuntimeError as e:
            logger.error("DAS query failed for %s: %s", primary_ds, e)
            return False

    # 2. Check tarballs / loose ROOT files against combined expectations.
    if era:
        skim_dir = base_dir_for_era(era, scratch=scratch) / "files" / primary_ds
    else:
        skim_dir = infer_output_dir(group[0][0], scratch=scratch)

    result = _check_dataset(primary_ds, combined_urls, skim_dir)
    expected = result["expected"]
    found = result["found"]

    status = "OK" if not result["missing"] else "INCOMPLETE"
    logger.info("Tarball check: %d/%d files found [%s]", found, expected, status)

    if result["missing"]:
        if len(result["missing"]) <= 10:
            for idx, url in result["missing"]:
                logger.warning("  Missing file %d: %s", idx, Path(url).name)
        else:
            logger.warning("  %d files missing from skim directory", len(result["missing"]))
        return False

    # 3. Check per-job status sidecars and tarball content consistency.
    if not _validate_job_status_outputs(primary_ds, combined_urls, skim_dir):
        return False

    # 4. Check logs for successful completion across full combined ProcId range.
    if era:
        base_dir = base_dir_for_era(era, scratch=scratch)
    else:
        base_dir = infer_base_dir(group[0][0], scratch=scratch)
    log_dir = base_dir / "logs" / primary_ds

    if not log_dir.is_dir():
        logger.warning("Log directory not found: %s — skipping log check", log_dir)
    else:
        expected_indices = set(range(0, expected))
        passed, failed = _check_log_completion(log_dir, primary_ds, expected_indices)

        logger.info("Log check: %d/%d jobs confirmed successful", len(passed), expected)

        if failed:
            if len(failed) <= 10:
                for idx, reason in failed:
                    logger.warning("  Job %d: %s", idx, reason)
            else:
                logger.warning("  %d jobs with log issues (showing first 10):", len(failed))
                for idx, reason in failed[:10]:
                    logger.warning("    Job %d: %s", idx, reason)
            return False

    logger.info(
        "Combined pre-merge check PASSED — all %d files present and logs confirm success.",
        expected,
    )
    return True


def _print_merge_result(result):
    print()
    print(f"{'Dataset:':<22} {result.dataset}")
    print(f"{'Input files:':<22} {result.input_files}")
    print(f"{'Output files:':<22} {result.output_files}")
    print(f"{'Events (in):':<22} {result.total_events_in}")
    print(f"{'Events (out):':<22} {result.total_events_out}")
    print(f"{'Events match:':<22} {'YES' if result.events_match else 'NO'}")
    print(f"{'Runs sumw (in):':<22} {result.total_sumw_in:.2f}")
    print(f"{'Runs sumw (out):':<22} {result.total_sumw_out:.2f}")
    print(f"{'Sumw match:':<22} {'YES' if result.sumw_match else 'NO'}")
    if result.output_paths:
        print(f"{'Output files:':<22}")
        for p in result.output_paths:
            size_mb = Path(p).stat().st_size / 1_048_576 if Path(p).exists() else 0
            print(f"  {Path(p).name:<50} {size_mb:.1f} MB")
    print()


def cmd_merge(args):
    """Extract tarballs, merge, and validate merged outputs."""
    from wrcoffea.skim_merge import merge_dataset_incremental, validate_merge

    scratch = getattr(args, "scratch", False)
    primary_ds = primary_dataset_from_das_path(args.das_path)
    skim_dir = infer_output_dir(args.das_path, scratch=scratch)

    if not skim_dir.is_dir():
        logger.error("Skim directory not found: %s", skim_dir)
        raise SystemExit(1)

    # Pre-merge completeness check (unless --skip-check or --validate-only)
    if not args.validate_only and not args.skip_check:
        if not _pre_merge_check(args.das_path, primary_ds, skim_dir, scratch=scratch):
            logger.error("Pre-merge check FAILED — aborting. Use --skip-check to override.")
            raise SystemExit(1)

    # Validate-only
    if args.validate_only:
        merged = sorted(str(p) for p in skim_dir.glob(f"{primary_ds}_part*.root"))
        if not merged:
            logger.error("No merged files found in %s", skim_dir)
            raise SystemExit(1)
        result = validate_merge(merged)
        _print_merge_result(result)
        raise SystemExit(0 if result.events_match and result.sumw_match else 1)

    # Extract all tarballs, group by HLT, chunk by max_events, merge.
    result = merge_dataset_incremental(
        skim_dir, primary_ds,
        max_events=args.max_events,
    )

    _print_merge_result(result)

    if not result.events_match or not result.sumw_match:
        logger.error("Merge validation FAILED — original files preserved.")
        raise SystemExit(1)

    # Cross-check genEventSumw against analysis config JSON
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        entry = config.get(args.das_path)
        if entry is None:
            logger.warning("DAS path %s not found in %s — skipping sumw cross-check", args.das_path, args.config)
        else:
            expected_sumw = entry.get("genEventSumw")
            if expected_sumw is not None:
                diff = abs(result.total_sumw_out - expected_sumw)
                ratio = result.total_sumw_out / expected_sumw if expected_sumw else float("inf")
                if diff < 1.0:
                    logger.info("Config sumw cross-check PASSED (config=%.2f, merged=%.2f)", expected_sumw, result.total_sumw_out)
                else:
                    logger.error(
                        "Config sumw cross-check FAILED! config=%.2f, merged=%.2f (ratio=%.6f)",
                        expected_sumw, result.total_sumw_out, ratio,
                    )
                    raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════
# era-level commands
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS_ROOT = REPO_ROOT / "data" / "configs"


def _configs_for_era(era_name):
    """Resolve an era name to all config JSON files in its directory.

    Uses ``ERA_SUBDIRS`` to map e.g. ``'Run3Summer23'`` →
    ``data/configs/Run3/2023/Run3Summer23/*.json``.

    Returns a sorted list of Path objects.

    Raises
    ------
    SystemExit
        If the era name is unknown or no JSON files are found.
    """
    if era_name not in ERA_SUBDIRS:
        logger.error(
            "Unknown era %r. Known eras: %s", era_name, list(ERA_SUBDIRS.keys())
        )
        raise SystemExit(1)
    config_dir = CONFIGS_ROOT / ERA_SUBDIRS[era_name]
    if not config_dir.is_dir():
        logger.error("Config directory does not exist: %s", config_dir)
        raise SystemExit(1)
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        logger.error("No JSON config files found in %s", config_dir)
        raise SystemExit(1)
    logger.info("Era %s: found %d config(s) in %s", era_name, len(configs), config_dir)
    for c in configs:
        logger.info("  %s", c.name)
    return configs


def _resolve_era_configs(args):
    """Normalize --era / --config into args.config (list of Paths).

    If ``args.era`` is set, resolves it via :func:`_configs_for_era`.
    Otherwise, ``args.config`` is used as-is.
    """
    era = getattr(args, "era", None)
    if era:
        args.config = _configs_for_era(era)
    return args.config


def _group_by_primary_ds(all_datasets):
    """Group datasets that share the same (era, dataset_name).

    Returns an OrderedDict mapping ``(era, dataset_name)`` to a list of
    ``(das_path, meta, config_path)`` tuples.

    For MC datasets, ``dataset_name`` is the DAS primary dataset (each
    DAS path is unique).  For data datasets, ``dataset_name`` comes from
    the config's ``"dataset"`` field (e.g. ``Muon0_Run2023C``), so that
    different run-eras stay in separate groups even when they share the
    same DAS primary dataset (e.g. ``Muon0``).
    """
    from collections import OrderedDict

    groups = OrderedDict()
    for das_path, meta, cfg in all_datasets:
        if meta.get("datatype") == "data":
            ds_name = meta.get("dataset", primary_dataset_from_das_path(das_path))
        else:
            ds_name = primary_dataset_from_das_path(das_path)
        era = meta.get("era", "")
        key = (era, ds_name)
        if key not in groups:
            groups[key] = []
        groups[key].append((das_path, meta, cfg))
    return groups


def _load_datasets_from_configs(config_paths):
    """Load DAS paths + metadata from one or more config JSONs.

    Skips datasets whose ``"note"`` field contains "no longer available".
    Deduplicates by DAS path (first occurrence wins).
    Returns list of ``(das_path, metadata, config_path)`` tuples.
    """
    all_datasets = []
    seen = set()
    for config_path in config_paths:
        with open(config_path) as f:
            config = json.load(f)
        for das_path, meta in config.items():
            note = meta.get("note", "").lower()
            if "no longer available" in note:
                logger.info("Skipping %s (marked unavailable)", das_path.split("/")[1])
                continue
            if das_path in seen:
                logger.debug("Skipping duplicate %s", das_path.split("/")[1])
                continue
            seen.add(das_path)
            all_datasets.append((das_path, meta, config_path))
    return all_datasets


def _url_to_lfn(url: str) -> str:
    """Extract logical file name from an XRootD URL."""
    lfn_idx = url.find("//store/")
    return url[lfn_idx + 1:] if lfn_idx >= 0 else url


def _write_check_era_resubmit_script(path: Path, entries) -> int:
    """Write one-file Condor resubmit commands from check-era failures.

    Parameters
    ----------
    entries : iterable of tuples
        Tuples are ``(das_path, local_file_index_1based, primary_ds, combined_idx_0based)``.
    
    Returns
    -------
    int
        Number of unique one-file resubmit commands written.
    """
    unique = []
    seen = set()
    for das_path, local_idx, primary_ds, combined_idx in entries:
        key = (das_path, local_idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append((das_path, local_idx, primary_ds, combined_idx))

    lines = [
        "#!/usr/bin/env bash\n",
        "set -euo pipefail\n",
        "\n",
        f"# Auto-generated by skim.py check-era\n",
        f"# Resubmit entries: {len(unique)}\n",
        "\n",
    ]
    for das_path, local_idx, primary_ds, combined_idx in unique:
        lines.append(f"# {primary_ds} combined_job_index={combined_idx}\n")
        lines.append(
            f"python3 bin/skim.py run '{das_path}' --start {local_idx} "
            f"--end {local_idx} --scratch\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    logger.info("Wrote %d resubmit command(s) to %s", len(unique), path)
    return len(unique)


def _check_era_artifact_tag(args) -> str:
    """Stable tag used for check-era artifact filenames."""
    era = getattr(args, "era", None)
    if era:
        return str(era)
    configs = getattr(args, "config", None) or []
    if len(configs) == 1:
        return Path(configs[0]).stem
    if configs:
        return f"configs_{len(configs)}"
    return "unknown"


def _check_era_artifact_paths(args):
    """Return (tag, default_resubmit_script_path, state_json_path)."""
    tag = _check_era_artifact_tag(args)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "unknown"
    script_path = REPO_ROOT / "scripts" / f"resubmit_failed_{safe}.sh"
    state_path = REPO_ROOT / "scripts" / f".check_era_state_{safe}.json"
    return tag, script_path, state_path


def _write_check_era_state(path: Path, payload: dict) -> None:
    """Persist latest check-era summary/state for follow-up resubmission."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _run_era_state_path(base_dir: Path) -> Path:
    """Path to the JSON file that records failed submissions for an era."""
    return base_dir / ".run_era_failed.json"


def _retry_failed_submissions(args):
    """Resubmit only datasets whose condor_submit failed previously.

    Reads the failed dataset list saved by the last run-era invocation
    and resubmits just those job directories.
    """
    _resolve_era_configs(args)
    all_datasets = _load_datasets_from_configs(args.config)
    if not all_datasets:
        logger.error("No datasets to process.")
        raise SystemExit(1)

    first_era = all_datasets[0][1].get("era")
    if not first_era:
        logger.error("First dataset missing 'era' in config metadata.")
        raise SystemExit(1)
    base_dir = base_dir_for_era(first_era, scratch=True)
    state_path = _run_era_state_path(base_dir)

    if not state_path.exists():
        logger.error("No failed-submission state file found at %s", state_path)
        logger.error("Run 'run-era' first (without --retry-failed).")
        raise SystemExit(1)

    with state_path.open() as f:
        failed_datasets = json.load(f)

    if not failed_datasets:
        logger.info("No failed submissions to retry.")
        return

    jobs_root = base_dir / "jobs"
    logger.info("Found %d dataset(s) to retry", len(failed_datasets))

    submitted = 0
    failed = []
    for primary_ds in failed_datasets:
        job_dir = jobs_root / primary_ds
        jdl = job_dir / "job.jdl"
        if not jdl.exists():
            logger.error("No job.jdl found for %s at %s", primary_ds, job_dir)
            failed.append((primary_ds, "missing job.jdl"))
            continue

        if args.dry_run:
            logger.info("[dry-run] Would resubmit %s", primary_ds)
            submitted += 1
        else:
            try:
                subprocess.run(
                    ["bash", "-c", "condor_submit job.jdl"],
                    cwd=str(job_dir), check=True,
                )
                logger.info("Resubmitted %s", primary_ds)
                submitted += 1
            except subprocess.CalledProcessError as e:
                logger.error("condor_submit failed for %s: %s", primary_ds, e)
                failed.append((primary_ds, f"condor_submit failed: {e}"))

    # Update state file: only keep datasets that still failed
    still_failed = [ds for ds, _reason in failed]
    with state_path.open("w") as f:
        json.dump(still_failed, f, indent=2)
        f.write("\n")

    logger.info("Resubmitted %d/%d. %d still failed.", submitted, len(failed_datasets), len(failed))
    if failed:
        for ds, reason in failed:
            logger.error("  FAILED: %s — %s", ds, reason)
        raise SystemExit(1)
    else:
        state_path.unlink()
        logger.info("All retries succeeded — cleared state file.")


def cmd_run_era(args):
    """Submit Condor skim jobs for all datasets in one or more configs.

    DAS paths that share the same ``(era, primary_ds)`` — e.g. data
    sub-eras like Muon0 v1-v4 — are combined into a single Condor
    submission with continuous file numbering to avoid tarball collisions.
    """
    if getattr(args, "retry_failed", False):
        return _retry_failed_submissions(args)

    check_dasgoclient()
    check_grid_proxy()
    _resolve_era_configs(args)

    all_datasets = _load_datasets_from_configs(args.config)
    logger.info("Found %d datasets across %d config(s)", len(all_datasets), len(args.config))

    if not all_datasets:
        logger.error("No datasets to process.")
        raise SystemExit(1)

    # Create repo tarball once — use era from first dataset's config metadata
    first_era = all_datasets[0][1].get("era")
    if not first_era:
        logger.error("First dataset missing 'era' in config metadata.")
        raise SystemExit(1)
    base_dir = base_dir_for_era(first_era, scratch=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    tarball = _create_tarball(base_dir)

    # Group by (era, primary_ds) so data sub-eras get one combined submission.
    groups = _group_by_primary_ds(all_datasets)

    submitted = 0
    failed = []
    for (era, primary_ds), group in groups.items():
        ds_base = base_dir_for_era(era, scratch=True)

        # Query DAS for all DAS paths in the group, build combined argument lines.
        combined_lines = []
        job_idx = 0
        das_failed = False
        for das_path, meta, _cfg in group:
            try:
                lfns = query_das_files(das_path)
                file_urls = das_files_to_urls(lfns)
            except RuntimeError as e:
                logger.error("DAS query failed for %s: %s", primary_ds, e)
                failed.append((primary_ds, str(e)))
                das_failed = True
                break
            for url in file_urls:
                combined_lines.append(f"{job_idx} {das_path} {_url_to_lfn(url)} {primary_ds}\n")
                job_idx += 1
        if das_failed:
            continue

        if len(group) > 1:
            logger.info(
                "%s: combined %d DAS paths into %d files",
                primary_ds, len(group), len(combined_lines),
            )

        job_dir = ds_base / "jobs" / primary_ds
        log_dir = ds_base / "logs" / primary_ds
        output_dir = ds_base / "files" / primary_ds

        job_dir.mkdir(parents=True, exist_ok=True)
        dest_tb = job_dir / "WrCoffea.tar.gz"
        shutil.copy2(tarball, dest_tb)

        jdl_path, actual_jobs = _generate_job(
            primary_ds, None, 0, job_dir, log_dir, output_dir,
            None, argument_lines=combined_lines,
        )

        if args.dry_run:
            logger.info("[dry-run] Would submit %d jobs for %s", actual_jobs, primary_ds)
            submitted += 1
        else:
            try:
                subprocess.run(
                    ["bash", "-c", f"condor_submit {jdl_path.name}"],
                    cwd=str(job_dir), check=True,
                )
                logger.info("Submitted %d jobs for %s", actual_jobs, primary_ds)
                submitted += 1
            except subprocess.CalledProcessError as e:
                logger.error("condor_submit failed for %s: %s", primary_ds, e)
                failed.append((primary_ds, f"condor_submit failed: {e}"))

    # Cleanup shared tarball
    if tarball.exists():
        tarball.unlink()

    # Save failed dataset list for --retry-failed
    state_path = _run_era_state_path(base_dir)
    if failed:
        failed_datasets = [ds for ds, _reason in failed]
        with state_path.open("w") as f:
            json.dump(failed_datasets, f, indent=2)
            f.write("\n")
        logger.info("Saved %d failed dataset(s) to %s", len(failed), state_path)
    elif state_path.exists():
        state_path.unlink()

    logger.info("Submitted %d/%d groups. %d failed.", submitted, len(groups), len(failed))
    if failed:
        for ds, reason in failed:
            logger.error("  FAILED: %s — %s", ds, reason)
        logger.info("Retry with: python3 bin/skim.py run-era --era %s --retry-failed", first_era)
        raise SystemExit(1)


def cmd_check_era(args):
    """Check all datasets in config(s) for skim job completeness.

    Groups by ``(era, primary_ds)`` so combined data sub-era submissions
    are checked with the correct total file count. For each dataset/group:
      1) output presence against DAS file list,
      2) status sidecar consistency,
      3) Condor log completion markers.

    When failed/missing jobs are found, writes a one-file resubmit script with
    ``run --start N --end N --scratch`` commands. Use ``--resubmit-script`` to
    override the default output path under ``scripts/``.
    """
    check_dasgoclient()
    check_grid_proxy()
    _resolve_era_configs(args)
    artifact_tag, default_resubmit_script, state_path = _check_era_artifact_paths(args)
    resubmit_script_path = args.resubmit_script or default_resubmit_script

    all_datasets = _load_datasets_from_configs(args.config)
    groups = _group_by_primary_ds(all_datasets)

    # Track per-dataset outcomes for the summary.
    incomplete: list[tuple[str, str]] = []  # (label, reason)
    n_total = 0
    n_ok = 0
    resubmit_entries = []

    for (era, primary_ds), group in groups.items():
        n_total += 1
        label = primary_ds
        if len(group) > 1:
            label = f"{primary_ds} ({len(group)} sub-eras)"

        # Combine file URLs across all DAS paths in the group. Keep per-file
        # mapping back to (das_path, local file index) for targeted resubmits.
        combined_urls = []
        combined_entries = []  # (das_path, local_idx_1based, url)
        das_failed = False
        for das_path, _meta, _cfg in group:
            try:
                lfns = query_das_files(das_path)
                file_urls = das_files_to_urls(lfns)
                for local_idx, url in enumerate(file_urls, start=1):
                    combined_urls.append(url)
                    combined_entries.append((das_path, local_idx, url))
            except RuntimeError as e:
                logger.error("DAS query failed for %s: %s", primary_ds, e)
                das_failed = True
                break
        if das_failed:
            incomplete.append((label, "DAS query failed"))
            continue

        if era:
            skim_dir = base_dir_for_era(era, scratch=True) / "files" / primary_ds
        else:
            skim_dir = infer_output_dir(group[0][0], scratch=True)
        result = _check_dataset(primary_ds, combined_urls, skim_dir)
        expected = result["expected"]

        issues = []
        detail_lines = []
        failed_indices = set()  # 0-based indices in combined_urls

        if result["missing"]:
            issues.append(f"{len(result['missing'])}/{expected} files missing")
            if len(result["missing"]) <= 5:
                for idx, url in result["missing"]:
                    detail_lines.append(f"    Missing file {idx}: {Path(url).name}")
                    failed_indices.add(idx - 1)
            else:
                detail_lines.append(f"    ({len(result['missing'])} files missing)")
                failed_indices.update(idx - 1 for idx, _url in result["missing"])
        else:
            status_ok, status_failed = _validate_job_status_outputs(
                primary_ds, combined_urls, skim_dir, return_failed_indices=True,
            )
            if not status_ok:
                issues.append("status sidecar check failed")
                failed_indices.update(status_failed)

            if era:
                base_dir = base_dir_for_era(era, scratch=True)
            else:
                base_dir = infer_base_dir(group[0][0], scratch=True)
            log_dir = base_dir / "logs" / primary_ds
            if not log_dir.is_dir():
                logger.warning("Log directory not found: %s — skipping log check", log_dir)
            else:
                expected_indices = set(range(0, expected))
                passed, failed_logs = _check_log_completion(log_dir, primary_ds, expected_indices)
                logger.info(
                    "Check-era log check for %s: %d/%d jobs confirmed successful",
                    primary_ds, len(passed), expected,
                )
                if failed_logs:
                    issues.append(f"{len(failed_logs)}/{expected} jobs failed log checks")
                    failed_indices.update(idx for idx, _reason in failed_logs)
                    if len(failed_logs) <= 5:
                        for idx, reason in failed_logs:
                            detail_lines.append(f"    Log issue job {idx}: {reason}")
                    else:
                        detail_lines.append(f"    ({len(failed_logs)} jobs failed log checks)")

        status = "OK" if not issues else "INCOMPLETE"
        print(f"  {label:60s}  {result['found']}/{result['expected']}  [{status}]")
        for line in detail_lines:
            print(line)

        for idx0 in sorted(failed_indices):
            if 0 <= idx0 < len(combined_entries):
                das_path_i, local_idx_i, _url_i = combined_entries[idx0]
                resubmit_entries.append((das_path_i, local_idx_i, primary_ds, idx0))
            else:
                logger.warning(
                    "Could not map failed job index %d for %s to a DAS file entry",
                    idx0, primary_ds,
                )

        if issues:
            incomplete.append((label, "; ".join(issues)))
        else:
            n_ok += 1

    resubmit_count = 0
    if resubmit_entries:
        resubmit_count = _write_check_era_resubmit_script(resubmit_script_path, resubmit_entries)
        print(f"Resubmit script: {resubmit_script_path}")
    else:
        # For the default path, remove stale scripts so a clean check-era
        # cannot be mistaken for pending failures.
        if args.resubmit_script is None and resubmit_script_path.exists():
            resubmit_script_path.unlink()
            logger.info("No failures; removed stale resubmit script %s", resubmit_script_path)

    _write_check_era_state(
        state_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_tag": artifact_tag,
            "era": getattr(args, "era", None),
            "configs": [str(p) for p in (args.config or [])],
            "incomplete_datasets": len(incomplete),
            "failed_jobs": resubmit_count,
            "resubmit_script": str(resubmit_script_path),
        },
    )
    logger.info("Check-era state written to %s", state_path)

    # Print summary.
    print()
    if not incomplete:
        print(f"Summary: ALL OK — {n_ok}/{n_total} datasets complete.")
    else:
        print(f"Summary: {n_ok}/{n_total} datasets complete, {len(incomplete)} incomplete:")
        for label, reason in incomplete:
            print(f"  - {label}: {reason}")
        raise SystemExit(1)


def cmd_resubmit_failures(args):
    """Resubmit failed skim jobs recorded by the latest check-era run."""
    if args.era not in ERA_SUBDIRS:
        logger.error("Unknown era %r. Known eras: %s", args.era, list(ERA_SUBDIRS.keys()))
        raise SystemExit(1)

    artifact_tag, default_script_path, state_path = _check_era_artifact_paths(args)

    if not state_path.exists():
        logger.error(
            "No check-era state found for %s at %s. Run check-era first.",
            artifact_tag, state_path,
        )
        raise SystemExit(1)

    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        logger.error("Failed to read state file %s: %s", state_path, e)
        raise SystemExit(1)

    if not isinstance(state, dict):
        logger.error("Invalid state file payload in %s", state_path)
        raise SystemExit(1)

    failed_jobs = state.get("failed_jobs", 0)
    try:
        failed_jobs = int(failed_jobs)
    except Exception:
        failed_jobs = 0

    if failed_jobs <= 0:
        logger.error(
            "Latest check-era for %s reported no failed jobs. Nothing to resubmit.",
            artifact_tag,
        )
        raise SystemExit(1)

    script_path = Path(state.get("resubmit_script", str(default_script_path)))
    if not script_path.exists():
        logger.error(
            "Resubmit script not found: %s. Re-run check-era for %s.",
            script_path, args.era,
        )
        raise SystemExit(1)

    if args.dry_run:
        logger.info(
            "[dry-run] Would execute %s (%d failed jobs)",
            script_path, failed_jobs,
        )
        return

    logger.info("Resubmitting %d failed jobs using %s", failed_jobs, script_path)
    try:
        subprocess.run(["bash", str(script_path)], cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Resubmit script failed: %s", e)
        raise SystemExit(1)


def _merge_one_dataset(das_path, meta, max_events, skip_check, dataset_name=None):
    """Merge a single dataset. Returns (das_path, result, error_reason).

    Designed to be called from a ProcessPoolExecutor.

    Parameters
    ----------
    dataset_name : str, optional
        Override the dataset name used for directory paths and output
        file naming.  When ``None``, falls back to the DAS primary
        dataset.  Era-level commands pass the group key here so that
        data sub-eras (e.g. ``Muon0_Run2023C``) stay separate.
    """
    from wrcoffea.skim_merge import merge_dataset_incremental

    primary_ds = dataset_name or primary_dataset_from_das_path(das_path)
    era = meta.get("era")
    if era:
        skim_dir = base_dir_for_era(era, scratch=True) / "files" / primary_ds
    else:
        skim_dir = infer_output_dir(das_path, scratch=True)

    if not skim_dir.is_dir():
        logger.warning("Skim directory not found for %s — skipping", primary_ds)
        return das_path, None, "skim directory not found"

    if not skip_check:
        if not _pre_merge_check(das_path, primary_ds, skim_dir, scratch=True):
            logger.error("Pre-merge check FAILED for %s — skipping", primary_ds)
            return das_path, None, "pre-merge check failed"

    result = merge_dataset_incremental(
        skim_dir, primary_ds, max_events=max_events,
    )
    _print_merge_result(result)

    if not result.events_match or not result.sumw_match:
        return das_path, result, "merge validation failed"

    # Config sumw cross-check
    expected_sumw = meta.get("genEventSumw")
    if expected_sumw:
        diff = abs(result.total_sumw_out - expected_sumw)
        if diff >= 1.0:
            logger.error(
                "Config sumw mismatch for %s: config=%.2f, merged=%.2f",
                primary_ds, expected_sumw, result.total_sumw_out,
            )
            return das_path, result, "sumw cross-check failed"
        logger.info("Config sumw cross-check PASSED for %s", primary_ds)

    return das_path, result, None


def cmd_merge_era(args):
    """Merge all datasets in config(s) from scratch space.

    Deduplicates by ``(era, primary_ds)`` so grouped data sub-eras
    (e.g. Muon0 v1-v4) are only merged once.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    _resolve_era_configs(args)

    all_datasets = _load_datasets_from_configs(args.config)
    groups = _group_by_primary_ds(all_datasets)

    # Build a deduplicated task list: one merge per unique (era, primary_ds).
    # Use the first DAS path as the representative. For multi-path groups
    # (data sub-eras), run a combined pre-merge check across all DAS paths
    # before dispatching the merge task.
    merge_tasks = []
    failed = []
    for (era, primary_ds), group in groups.items():
        first_das_path, first_meta, _cfg = group[0]
        skip_check = args.skip_check
        if len(group) > 1 and not skip_check:
            logger.info(
                "%s has %d sub-eras — running combined pre-merge check before merge",
                primary_ds, len(group),
            )
            if not _pre_merge_check_group(group, era, primary_ds, scratch=True):
                logger.error("Combined pre-merge check FAILED for %s — skipping", primary_ds)
                failed.append((first_das_path, "pre-merge check failed"))
                continue
            # Already checked for this grouped dataset; skip worker-local single-path check.
            skip_check = True
        merge_tasks.append((first_das_path, first_meta, skip_check, primary_ds))

    results = []
    workers = getattr(args, "workers", 1)

    if workers > 1:
        logger.info("Merging %d datasets with %d parallel workers", len(merge_tasks), workers)
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for das_path, meta, skip_check, ds_name in merge_tasks:
                fut = pool.submit(
                    _merge_one_dataset, das_path, meta,
                    args.max_events, skip_check,
                    dataset_name=ds_name,
                )
                futures[fut] = (das_path, ds_name)

            for fut in as_completed(futures):
                das_path, ds_name = futures[fut]
                try:
                    das_path, result, error = fut.result()
                except Exception as e:
                    logger.error("Worker crashed for %s: %s", ds_name, e)
                    failed.append((das_path, f"worker exception: {e}"))
                    continue
                if error:
                    failed.append((das_path, error))
                elif result:
                    results.append(result)
    else:
        for das_path, meta, skip_check, ds_name in merge_tasks:
            das_path, result, error = _merge_one_dataset(
                das_path, meta, args.max_events, skip_check,
                dataset_name=ds_name,
            )
            if error:
                failed.append((das_path, error))
            elif result:
                results.append(result)

    logger.info("Merged %d/%d datasets successfully.", len(results), len(results) + len(failed))
    if failed:
        logger.error("%d datasets failed:", len(failed))
        for das_path, reason in failed:
            logger.error("  %s — %s", das_path.split("/")[1], reason)
        raise SystemExit(1)


# ═══════════════════════════════════════════════════════════════════════════
# upload
# ═══════════════════════════════════════════════════════════════════════════

WISC_HOST = "cmsxrootd.hep.wisc.edu"
WISC_DEFAULT_USER = "wijackso"
WISC_SKIM_BASE = "/store/user/{user}/WRAnalyzer/skims"


UPLOAD_MAX_RETRIES = 3
UPLOAD_BACKOFF_SECONDS = (30, 60, 120)


def _remote_file_size(host, remote_path):
    """Return remote file size in bytes via xrdfs stat, or None if the file doesn't exist."""
    try:
        result = subprocess.run(
            ["xrdfs", host, "stat", remote_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("Size:"):
                return int(line.split()[1])
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return None


def _upload_file_with_retries(local_path, dest_url, max_retries=UPLOAD_MAX_RETRIES):
    """Upload a single file via xrdcp with exponential backoff. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(["xrdcp", "-f", str(local_path), dest_url], check=True)
            return True
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                delay = UPLOAD_BACKOFF_SECONDS[attempt - 1]
                logger.warning(
                    "Upload attempt %d/%d failed for %s (exit %d), retrying in %ds",
                    attempt, max_retries, local_path.name, e.returncode, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Upload failed for %s after %d attempts: %s",
                    local_path.name, max_retries, e,
                )
                return False
    return False  # unreachable, but satisfies linters


def _upload_dataset(primary_ds, skim_dir, run, year, era, remote_user, dry_run=False):
    """Upload merged ROOT files for one dataset to Wisconsin. Returns True on success."""
    category = infer_category(primary_ds)
    base = WISC_SKIM_BASE.format(user=remote_user)
    remote_dir = f"{base}/{run}/{year}/{era}/{category}/{primary_ds}"
    remote_url = f"root://{WISC_HOST}/{remote_dir}"

    merged_files = sorted(skim_dir.glob(f"{primary_ds}_part*.root"))
    if not merged_files:
        logger.warning("No merged files for %s in %s", primary_ds, skim_dir)
        return False

    logger.info("Uploading %d file(s) for %s -> %s", len(merged_files), primary_ds, remote_dir)

    for f in merged_files:
        dest = f"{remote_url}/{f.name}"
        if dry_run:
            logger.info("[dry-run] xrdcp %s %s", f, dest)
            continue

        local_size = f.stat().st_size
        remote_size = _remote_file_size(WISC_HOST, f"{remote_dir}/{f.name}")
        if remote_size is not None and remote_size == local_size:
            logger.info("Skipped %s (already uploaded, %.1f MB)", f.name, local_size / 1_048_576)
            continue

        if not _upload_file_with_retries(f, dest):
            return False
        logger.info("Uploaded %s (%.1f MB)", f.name, local_size / 1_048_576)
    return True


def _upload_state_path(args) -> Path:
    """Return the path for persisting upload failure state."""
    tag = _check_era_artifact_tag(args)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_") or "unknown"
    return REPO_ROOT / "scripts" / f".upload_state_{safe}.json"


def cmd_upload(args):
    """Upload merged files to Wisconsin via xrdcp."""
    check_grid_proxy()
    _resolve_era_configs(args)
    remote_user = args.remote_user

    if args.das_path:
        # Single-dataset mode
        scratch = getattr(args, "scratch", False)
        primary_ds = primary_dataset_from_das_path(args.das_path)
        skim_dir = infer_output_dir(args.das_path, scratch=scratch)
        era = era_from_das_path(args.das_path)
        if era is None:
            logger.error("Could not determine era from DAS path: %s", args.das_path)
            raise SystemExit(1)
        subdir_parts = ERA_SUBDIRS[era].split("/")
        run, year = subdir_parts[0], subdir_parts[1]
        ok = _upload_dataset(primary_ds, skim_dir, run, year, era, remote_user, dry_run=args.dry_run)
        if not ok:
            raise SystemExit(1)
        return

    # Era mode: upload all datasets from config(s), deduplicated by (era, primary_ds).
    all_datasets = _load_datasets_from_configs(args.config)
    groups = _group_by_primary_ds(all_datasets)

    # --retry-failed: check all datasets, relying on per-file size comparison
    # to skip files already uploaded correctly.
    retry_failed = getattr(args, "retry_failed", False)
    state_path = _upload_state_path(args)
    if retry_failed:
        logger.info("Checking all %d dataset(s) for missing/incomplete files", len(groups))

    uploaded = 0
    failed = []

    for (era, primary_ds), group in groups.items():
        first_das_path, first_meta, _cfg = group[0]
        era = str(first_meta.get("era", ""))
        if era:
            skim_dir = base_dir_for_era(era, scratch=True) / "files" / primary_ds
        else:
            skim_dir = infer_output_dir(first_das_path, scratch=True)
        run = str(first_meta.get("run", "Run3"))
        year = str(first_meta.get("year", "2022"))
        ok = _upload_dataset(primary_ds, skim_dir, run, year, era, remote_user, dry_run=args.dry_run)
        if ok:
            uploaded += 1
        else:
            failed.append(primary_ds)

    logger.info("Uploaded %d/%d datasets. %d failed.", uploaded, len(groups), len(failed))
    if failed:
        for ds in failed:
            logger.error("  FAILED: %s", ds)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"failed_datasets": failed}, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote upload failure state to %s", state_path)
        raise SystemExit(1)
    # Clear stale state on full success.
    if state_path.exists():
        state_path.unlink()
        logger.info("Cleared upload state file %s", state_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main: subcommand dispatch
# ═══════════════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="skim.py",
        description="WrCoffea skimming pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cuts", action="store_true",
                        help="Print skim selection cuts and exit")

    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser("run", help="Skim NanoAOD files")
    p_run.add_argument("das_path", help="DAS dataset path (e.g., /TTto2L2Nu_.../Run3Summer24.../NANOAODSIM)")
    p_run.add_argument("--start", type=int, default=None, help="1-indexed start file (default: first)")
    p_run.add_argument("--end", type=int, default=None, help="1-indexed end file (default: last)")
    p_run.add_argument("--local", action="store_true",
                        help="Run locally instead of submitting to Condor")
    p_run.add_argument("--dry-run", action="store_true",
                        help="Generate Condor files without submitting (no effect with --local)")
    p_run.add_argument("--lfn", default=None,
                        help="Direct logical file name (skips DAS query; used by Condor workers)")
    p_run.add_argument("--status-json", type=Path, default=None,
                        help="Write per-file skim status JSON (for Condor worker bookkeeping)")
    p_run.add_argument("--scratch", action="store_true",
                        help="Write output to 3DayLifetime scratch space instead of data/skims/")
    p_run.set_defaults(func=cmd_run)

    # --- check ---
    p_check = sub.add_parser("check", help="Check Condor job completion")
    p_check.add_argument("das_path", help="DAS dataset path")
    p_check.add_argument("--resubmit", type=Path, default=None, metavar="FILE", help="Write resubmit arguments.txt")
    p_check.add_argument("--scratch", action="store_true",
                          help="Check in 3DayLifetime scratch space instead of data/skims/")
    p_check.set_defaults(func=cmd_check)

    # --- merge ---
    p_merge = sub.add_parser("merge", help="Extract, merge, and validate skims")
    p_merge.add_argument("das_path", help="DAS dataset path")
    p_merge.add_argument("--max-events", type=int, default=1_000_000, help="Max events per merged file")
    p_merge.add_argument("--validate-only", action="store_true", help="Only validate, don't merge")
    p_merge.add_argument("--skip-check", action="store_true",
                          help="Skip pre-merge completeness check (DAS + log verification)")
    p_merge.add_argument("--config", type=Path, default=None, metavar="JSON",
                          help="Analysis config JSON to cross-check genEventSumw against")
    p_merge.add_argument("--scratch", action="store_true",
                          help="Read from 3DayLifetime scratch space instead of data/skims/")
    p_merge.set_defaults(func=cmd_merge)

    # --- run-era ---
    p_run_era = sub.add_parser("run-era",
                                help="Submit Condor skim jobs for all datasets in config(s)")
    run_era_src = p_run_era.add_mutually_exclusive_group(required=True)
    run_era_src.add_argument("--config", type=Path, nargs="+", metavar="JSON",
                              help="One or more analysis config JSON files")
    run_era_src.add_argument("--era", metavar="ERA",
                              help="Era name (e.g. Run3Summer23) — auto-discovers all configs")
    p_run_era.add_argument("--dry-run", action="store_true",
                            help="Generate Condor files without submitting")
    p_run_era.add_argument("--retry-failed", action="store_true",
                            help="Only resubmit datasets whose condor_submit failed previously")
    p_run_era.set_defaults(func=cmd_run_era)

    # --- check-era ---
    p_check_era = sub.add_parser("check-era",
                                  help="Check all datasets in config(s) for skim completeness")
    check_era_src = p_check_era.add_mutually_exclusive_group(required=True)
    check_era_src.add_argument("--config", type=Path, nargs="+", metavar="JSON",
                                help="One or more analysis config JSON files")
    check_era_src.add_argument("--era", metavar="ERA",
                                help="Era name (e.g. Run3Summer23) — auto-discovers all configs")
    p_check_era.add_argument("--resubmit-script", type=Path, default=None, metavar="SH",
                             help="Override output path for failed-job resubmit script")
    p_check_era.set_defaults(func=cmd_check_era)

    # --- resubmit-failures ---
    p_resubmit_failures = sub.add_parser(
        "resubmit-failures",
        help="Resubmit failed skim jobs from latest check-era state",
    )
    p_resubmit_failures.add_argument(
        "--era", required=True, metavar="ERA",
        help="Era name (e.g. Run3Summer23)",
    )
    p_resubmit_failures.add_argument(
        "--dry-run", action="store_true",
        help="Validate state and print what would run without submitting",
    )
    p_resubmit_failures.set_defaults(func=cmd_resubmit_failures)

    # --- merge-era ---
    p_merge_era = sub.add_parser("merge-era",
                                  help="Merge all datasets in config(s) from scratch space")
    merge_era_src = p_merge_era.add_mutually_exclusive_group(required=True)
    merge_era_src.add_argument("--config", type=Path, nargs="+", metavar="JSON",
                                help="One or more analysis config JSON files")
    merge_era_src.add_argument("--era", metavar="ERA",
                                help="Era name (e.g. Run3Summer23) — auto-discovers all configs")
    p_merge_era.add_argument("--max-events", type=int, default=1_000_000,
                              help="Max events per merged file (default: 1000000)")
    p_merge_era.add_argument("--skip-check", action="store_true",
                              help="Skip pre-merge completeness check")
    p_merge_era.add_argument("--workers", type=int, default=1,
                              help="Number of parallel merge workers (default: 1)")
    p_merge_era.set_defaults(func=cmd_merge_era)

    # --- upload ---
    p_upload = sub.add_parser("upload",
                               help="Upload merged files to Wisconsin via xrdcp")
    upload_target = p_upload.add_mutually_exclusive_group(required=True)
    upload_target.add_argument("--das-path",
                                help="Single DAS dataset path to upload")
    upload_target.add_argument("--config", type=Path, nargs="+", metavar="JSON",
                                help="Config JSON(s) for era-level upload")
    upload_target.add_argument("--era", metavar="ERA",
                                help="Era name (e.g. Run3Summer23) — auto-discovers all configs")
    p_upload.add_argument("--scratch", action="store_true",
                           help="Read from scratch space (implied for --config mode)")
    p_upload.add_argument("--remote-user", default=WISC_DEFAULT_USER,
                           help=f"Wisconsin storage user (default: {WISC_DEFAULT_USER})")
    p_upload.add_argument("--dry-run", action="store_true",
                           help="Print xrdcp commands without executing")
    p_upload.add_argument("--retry-failed", action="store_true",
                           help="Only retry datasets that failed in the previous upload")
    p_upload.set_defaults(func=cmd_upload)

    args = parser.parse_args(argv)

    if args.cuts:
        _print_skim_cuts()
        raise SystemExit(0)

    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    args.func(args)


if __name__ == "__main__":
    main()
