import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="htcondor.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Missing cross-reference", module="coffea.*")
import argparse
import time
import json
import logging
from pathlib import Path

from wrcoffea.era_utils import get_era_details
from wrcoffea.cli_utils import (
    build_fileset_path,
    list_eras,
    list_samples,
    load_and_select_fileset,
    load_masses_from_csv,
    normalize_mass_point,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_arguments(args, sig_points):
    if args.sample == "Signal" and not args.mass:
        logging.error("For 'Signal', you must provide a --mass argument (e.g. --mass WR2000_N1900).")
        raise ValueError("Missing mass argument for Signal sample.")
    if args.sample == "Signal" and args.mass not in sig_points:
        logging.error(f"The provided signal point {args.mass} is not valid. Choose from {sig_points}.")
        raise ValueError("Invalid mass argument for Signal sample.")
    if args.sample != "Signal" and args.mass:
        logging.error("The --mass option is only valid for 'Signal' samples.")
        raise ValueError("Mass argument provided for non-signal sample.")
    if args.reweight and args.sample != "DYJets":
        logging.error("Reweighting can only be applied to DY")
        raise ValueError("Invalid sample for reweighting.")
    if args.dy is not None and args.sample != "DYJets":
        raise ValueError(
            f"Trying to specify a DY sample for a non-DY background"
        )
    if args.max_workers is not None and args.max_workers < 1:
        raise ValueError("--max-workers must be a positive integer")
    if args.threads_per_worker is not None and args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be a positive integer")

def run_analysis(args, filtered_fileset, run_on_condor):

    # Heavy imports (coffea/dask/analyzer + transitive deps like pandas/scipy)
    # are intentionally delayed so `--preflight-only` is fast.
    from dask.distributed import Client, LocalCluster, WorkerPlugin
    from coffea.nanoevents import NanoAODSchema
    from coffea.processor import Runner, DaskExecutor

    from wrcoffea.analyzer import WrAnalysis

    NanoAODSchema.warn_missing_crossrefs = False
    NanoAODSchema.error_missing_event_ids = False

    if run_on_condor:
        from lpcjobqueue import LPCCondorCluster

        class _CondorWorkerSetup(WorkerPlugin):
            """Runs on every worker (including late arrivals) to fix paths."""
            def setup(self, worker):
                import sys, os
                for p in (".", "wrcoffea"):
                    if os.path.isdir(p) and p not in sys.path:
                        sys.path.insert(0, p)
                # Condor transfers "data/lumis" and "data/jsonpog" as flat
                # dirs ("lumis/", "jsonpog/") but config references them
                # under "data/...".  Recreate the expected structure.
                os.makedirs("data", exist_ok=True)
                for subdir in ("lumis", "jsonpog"):
                    src = os.path.abspath(subdir)
                    dst = os.path.join("data", subdir)
                    if os.path.isdir(src) and not os.path.exists(dst):
                        os.symlink(src, dst)

        repo_root = Path(__file__).resolve().parent.parent
        log_dir = f"/uscmst1b_scratch/lpc1/3DayLifetime/{os.environ['USER']}/dask-logs"

        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files=[
                str(repo_root / "wrcoffea"),
                str(repo_root / "bin"),
                str(repo_root / "data" / "lumis"),
                str(repo_root / "data" / "jsonpog"),
            ],
            log_directory=log_dir,
        )

        NWORKERS = args.max_workers or 100
        cluster.scale(NWORKERS)

        client = Client(cluster)
        client.register_worker_plugin(_CondorWorkerSetup())
        logging.info("Waiting for Condor workers (requested %d)...", NWORKERS)
        client.wait_for_workers(1, timeout="180s")
        logging.info("Started with %d/%d workers; remaining will join dynamically.", len(client.scheduler_info()["workers"]), NWORKERS)

    else:
        n_workers = args.max_workers or 6
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=(args.threads_per_worker or 1))
        client = Client(cluster)

    run = Runner(
        executor = DaskExecutor(client=client, compression=None, retries=10),
        chunksize = 250_000,
        maxchunks = None, # Change to 1 for testing, None for all
        skipbadfiles=False,
        xrootdtimeout = 60 if run_on_condor else 10,
        align_clusters = False,
        savemetrics=True,
        schema=NanoAODSchema,
    )

    try:
        logging.info("***PREPROCESSING***")
        preproc = run.preprocess(fileset=filtered_fileset, treename="Events")
        logging.info("Preprocessing completed")

        logging.info("***PROCESSING***")
        histograms, metrics = run(
            preproc,
            treename="Events",
            processor_instance=WrAnalysis(mass_point=args.mass, enabled_systs=args.systs, region=args.region),
        )
        logging.info("Processing completed")
        return histograms
    finally:
        try:
            client.close()
        finally:
            cluster.close()

if __name__ == "__main__":
    ERA_CHOICES = list_eras()
    SAMPLE_CHOICES = list_samples()
    DY_CHOICES = ["LO_inclusive", "NLO_mll_binned", "LO_HT"]

    parser = argparse.ArgumentParser(description="Processing script for WR analysis.")
    parser.add_argument("era", nargs="?", default=None, type=str, choices=ERA_CHOICES, help="Campaign to analyze.")
    parser.add_argument("sample", nargs="?", default=None, type=str, choices=SAMPLE_CHOICES, help="Sample to analyze (e.g., Signal, DYJets).")
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument("--dy", type=str, default=None, choices=DY_CHOICES, help="Specific DY sample to analyze (LO, NLO, etc)")
    optional.add_argument("--mass", type=str, default=None, help="Signal mass point to analyze.")
    optional.add_argument("--dir", type=str, default=None, help="Create a new output directory.")
    optional.add_argument("--name", type=str, default=None, help="Append the filenames of the output ROOT files.")
    optional.add_argument("--debug", action='store_true', help="Debug mode (don't compute histograms)")
    optional.add_argument("--reweight", type=str, default=None, help="Path to json file of DY reweights")
    optional.add_argument("--unskimmed", action='store_true', help="Run on unskimmed files.")
    optional.add_argument("--condor", action='store_true', help="Run on condor.")
    optional.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of Dask workers (local default: 6, condor default: 100).",
    )
    optional.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="Threads per Dask worker for local runs (LocalCluster threads_per_worker).",
    )
    optional.add_argument(
        "--systs",
        nargs="*",
        default=[],
        choices=["lumi"],
        help="Enable systematic histogram variations. Currently supported: lumi.",
    )
    optional.add_argument(
        "--region",
        type=str,
        default="both",
        choices=["resolved", "boosted", "both"],
        help="Analysis region to run: resolved, boosted, or both (default: both).",
    )
    optional.add_argument("--list-eras", action="store_true", help="Print available eras and exit.")
    optional.add_argument("--list-samples", action="store_true", help="Print available samples and exit.")
    optional.add_argument("--list-masses", action="store_true", help="Print available signal mass points for the given era (or all eras if none provided) and exit.")
    optional.add_argument("--preflight-only", action="store_true", help="Validate fileset path/schema and selection, then exit without processing.")
    args = parser.parse_args()

    # Listing helpers should work without positional args.
    if args.list_eras:
        print("\n".join(ERA_CHOICES))
        raise SystemExit(0)

    if args.list_samples:
        print("\n".join(SAMPLE_CHOICES))
        raise SystemExit(0)

    if args.list_masses:
        eras = [args.era] if args.era else ERA_CHOICES
        for e in eras:
            csv_path = Path(f"data/signal_points/{e}_mass_points.csv")
            masses = load_masses_from_csv(csv_path)
            if args.era is None:
                print(f"[{e}]")
            print("\n".join(masses))
            if args.era is None:
                print()
        raise SystemExit(0)

    if not args.era or not args.sample:
        parser.error("Missing required arguments: era and sample. Use --list-eras/--list-samples for discovery.")

    # Normalize any legacy naming to canonical WR/N format before validation.
    args.mass = normalize_mass_point(args.mass)

    signal_points = Path(f"data/signal_points/{args.era}_mass_points.csv")
    MASS_CHOICES = load_masses_from_csv(signal_points)

    print()
    logging.info(f"Analyzing {args.era} - {args.sample} events")
    
    validate_arguments(args, MASS_CHOICES)
    run, year, era = get_era_details(args.era)

    filepath = build_fileset_path(era=era, sample=args.sample, unskimmed=args.unskimmed, dy=args.dy)

    logging.info(f"Reading files from {filepath}")

    if not filepath.exists():
        raise FileNotFoundError(
            f"Fileset JSON not found: {filepath}. "
            "Create filesets first (see docs/filesets.md), or check --unskimmed and era/sample names."
        )

    filtered_fileset = load_and_select_fileset(
        filepath=filepath,
        desired_process=args.sample,
        mass=args.mass,
    )

    logging.info(
        "Selected %d dataset(s) after filtering.",
        len(filtered_fileset),
    )

    if args.preflight_only:
        logging.info("Preflight-only requested; exiting before processing.")
        raise SystemExit(0)

    t0 = time.monotonic()
    try:
        hists_dict = run_analysis(args, filtered_fileset, args.condor)
    except Exception as e:
        logging.error("Run failed: %s", e)
        raise

    if not args.debug:
        from wrcoffea.save_hists import save_histograms

        save_histograms(hists_dict, args)
    exec_time = time.monotonic() - t0
    logging.info(f"Execution took {exec_time/60:.2f} minutes")

