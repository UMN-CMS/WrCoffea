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
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

from wrcoffea.das_utils import (
    check_dasgoclient,
    check_grid_proxy,
    das_files_to_urls,
    infer_output_dir,
    query_das_files,
    validate_das_path,
    primary_dataset_from_das_path,
)
from wrcoffea.skimmer import SKIM_CUTS, SKIM_STAGES, skim_single_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_das(das_path):
    """Query DAS and return (primary_dataset, file_urls, default_outdir)."""
    check_dasgoclient()
    check_grid_proxy()
    primary_ds = primary_dataset_from_das_path(das_path)
    lfns = query_das_files(das_path)
    file_urls = das_files_to_urls(lfns)
    default_outdir = infer_output_dir(das_path)
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


# ═══════════════════════════════════════════════════════════════════════════
# run
# ═══════════════════════════════════════════════════════════════════════════

def cmd_run(args):
    """Skim NanoAOD files (submits to Condor unless --local or on a worker)."""
    primary_ds, file_urls, default_outdir = _resolve_das(args.das_path)
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
                     start=start_1, end=end_1, dry_run=args.dry_run)
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
+ApptainerImage = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:latest"
output = {log_dir}/{dataset}_$(ProcId).out
error  = {log_dir}/{dataset}_$(ProcId).err
log    = {log_dir}/{dataset}_$(ProcId).log
transfer_input_files = {tarball_path}
transfer_output_files = {dataset}_skim$(ProcId).tar.gz
initialdir = {output_dir}
queue arguments from arguments.txt
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


def _generate_job(primary_ds, das_path, n_jobs, job_dir, log_dir, output_dir, start=1, end=None):
    """Generate Condor job files for a file range.

    Parameters
    ----------
    start, end : int
        1-indexed file range (inclusive). Defaults to all files.
    output_dir : Path
        Directory where Condor will place output tarballs.
    """
    if end is None:
        end = n_jobs
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    (job_dir / "arguments.txt").write_text(
        "".join(f"{i} {das_path}\n" for i in range(start, end + 1))
    )

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

    actual_jobs = end - start + 1
    logger.info("Generated %d jobs for %s (files %d-%d)", actual_jobs, primary_ds, start, end)
    return jdl_path, actual_jobs


def _submit_run(das_path, primary_ds, file_urls, start, end, dry_run=False):
    """Submit a file range to Condor (called by cmd_run when not on a worker)."""
    n_total = len(file_urls)
    base_dir = infer_output_dir(das_path).parent  # data/skims/
    base_dir.mkdir(parents=True, exist_ok=True)
    tarball = _create_tarball(base_dir)

    job_dir = base_dir / "jobs" / primary_ds
    log_dir = base_dir / "logs" / primary_ds

    job_dir.mkdir(parents=True, exist_ok=True)
    dest_tb = job_dir / "WrCoffea.tar.gz"
    if not dest_tb.exists() or dest_tb.stat().st_size != tarball.stat().st_size:
        shutil.copy2(tarball, dest_tb)

    output_dir = infer_output_dir(das_path)
    jdl_path, actual_jobs = _generate_job(
        primary_ds, das_path, n_total, job_dir, log_dir, output_dir,
        start=start, end=end,
    )

    if dry_run:
        logger.info("[dry-run] Would submit %d jobs from %s", actual_jobs, jdl_path)
    else:
        logger.info("Submitting %d Condor jobs for %s (files %d-%d)...",
                     actual_jobs, primary_ds, start, end)
        subprocess.run(["bash", "-c", f"condor_submit {jdl_path.name}"], cwd=str(job_dir), check=True)
        logger.info("Submitted %s", primary_ds)

    if tarball.exists():
        tarball.unlink()
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


def cmd_check(args):
    """Detect missing / failed Condor skim jobs."""
    primary_ds, file_urls, default_outdir = _resolve_das(args.das_path)

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

    # Merge only needs the DAS path for directory derivation — no DAS query
    primary_ds = primary_dataset_from_das_path(args.das_path)
    skim_dir = infer_output_dir(args.das_path)

    if not skim_dir.is_dir():
        logger.error("Skim directory not found: %s", skim_dir)
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

    # Incremental: extract tarballs one-by-one, merge in ~max_events batches,
    # delete skim files as we go to keep disk usage bounded.
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
        import json
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
    p_run.set_defaults(func=cmd_run)

    # --- check ---
    p_check = sub.add_parser("check", help="Check Condor job completion")
    p_check.add_argument("das_path", help="DAS dataset path")
    p_check.add_argument("--resubmit", type=Path, default=None, metavar="FILE", help="Write resubmit arguments.txt")
    p_check.set_defaults(func=cmd_check)

    # --- merge ---
    p_merge = sub.add_parser("merge", help="Extract, merge, and validate skims")
    p_merge.add_argument("das_path", help="DAS dataset path")
    p_merge.add_argument("--max-events", type=int, default=1_000_000, help="Max events per merged file")
    p_merge.add_argument("--validate-only", action="store_true", help="Only validate, don't merge")
    p_merge.add_argument("--config", type=Path, default=None, metavar="JSON",
                          help="Analysis config JSON to cross-check genEventSumw against")
    p_merge.set_defaults(func=cmd_merge)

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
