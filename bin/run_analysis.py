import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
# Each coffea-image shell on an LPC login node has its own PID namespace, so this process
# gets the same PID in all of them, and the OpenMP runtime loaded with numpy registers
# itself under that PID in the node's shared /dev/shm. A second run started on the same
# node (or one after a killed run) then aborts at start with "OMP: Error #15 ... found
# unknown library already initialized" (2026-10-14). One process here only ever loads one
# OpenMP runtime, the case in which the runtime allows this check to be switched off.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="htcondor.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Missing cross-reference", module="coffea.*")
import argparse
import time
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

from wrcoffea.era_utils import get_era_details
from wrcoffea.analysis_config import CONFIG_OVERLAY_ENV, DY_VARIANTS
from wrcoffea.cli_utils import (
    COMPOSITE_SAMPLES,
    SPLITTABLE_GROUPS,
    build_fileset_path,
    build_sample_to_group_map,
    default_dy_variant,
    list_eras,
    list_samples,
    load_and_select_fileset,
    load_composite_fileset,
    load_masses_from_csv,
    mass_choices_for,
    normalize_mass_point,
    select_signal_points,
    split_sample_groups,
)
from wrcoffea.regions import STUDIES, check_unblind_allowed, resolve_studies
from wrcoffea.systematics import FAMILIES, FIT_OBSERVABLES, validate_requested, resolve_syst_observables, normalize_omitted
from wrcoffea.xrootd_fallback import (
    DEFAULT_RETRIES_PER_REDIRECTOR,
    DEFAULT_RETRY_SLEEP_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    REDIRECTORS,
    extract_lfn_from_url,
    extract_root_url_from_error,
    resolve_url_with_redirectors,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Corrections of the LO HT-binned DY sample (config.yaml dy_zpt_reweight, dy_reshape).  Since
# 2026-10-12 the default is none: DY at its NNLO-normalized cross section, not reweighted.
DY_REWEIGHT_CHOICES = ("none", "zpt", "zpt+reshape")


def resolve_dy_reweight(args) -> str:
    """The DY corrections to apply: ``--dy-reweight`` (default ``none``) or a legacy flag.

    The legacy flags switched corrections off while both were on by default, and keep
    the output they gave then: ``--no-dy-zpt-reweight`` = neither (``none``),
    ``--no-dy-reshape`` = the Z-pT reweight alone (``zpt``).
    """
    mode = getattr(args, "dy_reweight", None)
    if getattr(args, "no_dy_zpt_reweight", False):
        legacy = ("--no-dy-zpt-reweight", "none")  # it switched the reshape off too
    elif getattr(args, "no_dy_reshape", False):
        legacy = ("--no-dy-reshape", "zpt")
    else:
        return mode or "none"
    if mode is not None:
        raise ValueError(f"{legacy[0]} cannot be combined with --dy-reweight.")
    logging.warning("%s is deprecated: it means --dy-reweight %s (the default is none since "
                    "2026-10-12).", *legacy)
    return legacy[1]


def _format_dask_log_entries(entries):
    """Normalize scheduler/worker log payloads into printable lines."""
    if entries is None:
        return []
    if isinstance(entries, str):
        return entries.splitlines()
    if isinstance(entries, list):
        lines = []
        for entry in entries:
            if isinstance(entry, tuple) and len(entry) == 2:
                lines.append(f"[{entry[0]}] {entry[1]}")
            else:
                lines.append(str(entry))
        return lines
    return [str(entries)]


def _dump_dask_diagnostics(client, *, label, out_dir=Path("logs"), max_entries=300):
    """Write scheduler/worker diagnostics to a local file for post-mortem debugging."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in label)
    out_file = Path(out_dir) / f"{safe_label}_failure_{stamp}.log"
    lines = [f"UTC timestamp: {datetime.now(timezone.utc).isoformat()}"]

    try:
        info = client.scheduler_info()
        workers = info.get("workers", {})
        lines.append(f"Scheduler workers visible at failure: {len(workers)}")
        if workers:
            lines.append("Workers:")
            lines.extend(f"  - {w}" for w in workers.keys())
    except Exception as e:
        lines.append(f"Failed to query scheduler_info: {e!r}")

    try:
        lines.append("")
        lines.append("=== Scheduler Logs ===")
        lines.extend(_format_dask_log_entries(client.get_scheduler_logs(n=max_entries)))
    except Exception as e:
        lines.append(f"Failed to fetch scheduler logs: {e!r}")

    try:
        lines.append("")
        lines.append("=== Worker Logs ===")
        worker_logs = client.get_worker_logs(n=max_entries)
        if isinstance(worker_logs, dict):
            for worker, entries in worker_logs.items():
                lines.append(f"-- {worker} --")
                lines.extend(_format_dask_log_entries(entries))
                lines.append("")
        else:
            lines.extend(_format_dask_log_entries(worker_logs))
    except Exception as e:
        lines.append(f"Failed to fetch worker logs: {e!r}")

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logging.error("Saved Dask failure diagnostics to %s", out_file)
        return out_file
    except Exception:
        logging.exception("Failed to write Dask diagnostics file.")
        return None


def _format_job_states(states):
    if not states:
        return "job states unknown"
    text = ", ".join(f"{states[k]} {k}" for k in ("idle", "running", "held", "ended"))
    if states.get("hold_reason"):
        text += f" (hold reason: {states['hold_reason']})"
    return text


def _wait_for_workers_or_raise(client, *, requested_workers, min_workers, timeout_s, label,
                               job_states=None, poll_s=60, report_s=300):
    """Wait for the first workers; raise a diagnostic-rich error when none can come.

    ``job_states()`` returns the Condor states of the submitted worker jobs ({"idle", "running",
    "held", "ended"} counts and a "hold_reason"). Jobs idle in the queue mean a busy pool:
    keep waiting until ``timeout_s`` and say so every ``report_s``. Jobs that end or are held
    before any worker connected mean workers that cannot start: stop at once instead of
    waiting out the timeout (2026-10-14: a busy pool kept every worker away for over 30
    minutes, and a broken environment made every worker die at start).
    """
    timeout_s = int(timeout_s)
    logging.info(
        "Waiting for Condor workers (requested %d, need %d, timeout=%ss)...",
        requested_workers,
        min_workers,
        timeout_s,
    )
    start = last_report = time.monotonic()
    states = None
    failed = None
    while True:
        remaining = timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            break
        try:
            client.wait_for_workers(min_workers, timeout=f"{max(1, min(poll_s, int(remaining)))}s")
            return
        except TimeoutError:
            pass
        if job_states is not None:
            try:
                states = job_states()
            except Exception as exc:  # the schedd query is advisory only
                logging.debug("Could not query the Condor job states: %r", exc)
        if states and states["ended"] + states["held"] >= min(3, requested_workers):
            failed = (f"{states['ended']} worker job(s) ended and {states['held']} were held before any "
                      "worker connected: the workers cannot start")
            break
        if time.monotonic() - last_report >= report_s:
            last_report = time.monotonic()
            logging.info("Still waiting for Condor workers after %.0f s: %s.",
                         last_report - start, _format_job_states(states))
    diag_file = _dump_dask_diagnostics(client, label=label)
    diag_msg = f" Diagnostics: {diag_file}" if diag_file is not None else ""
    if failed:
        raise RuntimeError(f"{failed} ({_format_job_states(states)}). The worker logs are in "
                           f"{CONDOR_LOG_DIR_TEMPLATE.format(user=os.environ.get('USER', '$USER'))}."
                           f"{diag_msg}")
    raise RuntimeError(
        f"Timed out waiting for at least {min_workers} Condor worker(s) "
        f"after {timeout_s}s (requested {requested_workers}; {_format_job_states(states)}). "
        f"Idle jobs mean the pool had no free slot: rerun later or raise --worker-wait-timeout.{diag_msg}"
    )


def validate_arguments(args, sig_points):
    """Check CLI argument combinations are valid before running."""
    is_composite = args.sample in COMPOSITE_SAMPLES

    # Data signal regions are blind unless --unblind, which only Run 2 allows.
    if getattr(args, "unblind", False):
        check_unblind_allowed(args.era)

    # The Condor workers read config.yaml as shipped: an overlay would reach only this process.
    if (getattr(args, "condor", False) or getattr(args, "scheduler", None)) and os.environ.get(CONFIG_OVERLAY_ENV):
        raise ValueError(f"{CONFIG_OVERLAY_ENV} is set, but the Condor workers never read it (they read "
                         "config.yaml as shipped): run locally, or unset it.")

    picks_points = getattr(args, "signal_points", None) or getattr(args, "mwr_range", None)
    if is_composite:
        if args.mass:
            raise ValueError("--mass is not valid for composite modes.")
        if picks_points and "Signal" not in COMPOSITE_SAMPLES[args.sample]:
            raise ValueError("--signal-points / --mwr-range need a composite mode that includes Signal "
                             f"({', '.join(k for k, v in COMPOSITE_SAMPLES.items() if 'Signal' in v)}).")
        if args.dy is not None and "DYJets" not in COMPOSITE_SAMPLES[args.sample]:
            raise ValueError("--dy is only valid when DYJets is included in the composite mode.")
    else:
        if args.sample == "Signal" and not args.mass:
            logging.error("For 'Signal', you must provide a --mass argument (e.g. --mass WR2000_N1900).")
            raise ValueError("Missing mass argument for Signal sample.")
        # With --fileset the fileset decides which points exist (e.g. the UL18
        # pre-legacy scan, RunIISummer20UL18_signal_scan_fileset.json).
        if args.sample == "Signal" and args.mass not in sig_points and not getattr(args, "fileset", None):
            logging.error(f"The provided signal point {args.mass} is not valid. Choose from {sig_points}.")
            raise ValueError("Invalid mass argument for Signal sample.")
        if args.sample != "Signal" and args.mass:
            logging.error("The --mass option is only valid for 'Signal' samples.")
            raise ValueError("Mass argument provided for non-signal sample.")
        if args.dy is not None and args.sample != "DYJets":
            raise ValueError(
                f"Trying to specify a DY sample for a non-DY background"
            )
        if picks_points:
            raise ValueError("--signal-points / --mwr-range select the points of a composite mode "
                             "(signal, mc, all); use --mass for a single Signal point.")

    if args.dy is not None and args.dy not in DY_VARIANTS.get(args.era, []):
        raise ValueError(
            f"--dy {args.dy} is not available for {args.era}. "
            f"Valid choices: {DY_VARIANTS.get(args.era, [])}"
        )
    # nlo_ptll is not a DY prediction: it runs, but do not compare it with data or the other
    # DY samples.  Run 3: the four DYto2L-2Jets PTLL-100/200/400/600 samples are open-ended
    # (generator pT(ll) > 100, > 200, > 400, > 600 GeV; LHE_Vpt of the 2024 skims checked
    # 2026-10-12) and the fileset adds them unstitched, so pT(ll) > 200 GeV is counted twice,
    # > 400 three times, > 600 four times, and pT(ll) < 100 GeV is missing.  UL: the
    # LHEFilterPtZ-0To50 sample is not a 0-50 GeV slice (UL17 header 1504 pb against the
    # inclusive sample's 5917 pb below 50 GeV), so the six slices do not rebuild the spectrum
    # either.  Usable only after stitching on LHE_Vpt (Run 3) and a check of every UL slice.
    if args.dy == "nlo_ptll":
        logging.warning("--dy nlo_ptll: the pT(ll)-binned DY samples are added unstitched (Run 3: "
                        "open-ended slices, pT(ll) > 200 GeV counted 2-4 times, nothing below 100 GeV; "
                        "UL: the 0To50 slice is not a slice). Not a DY prediction; see run_analysis.py.")

    # Systematics: every requested family must be producible for this era,
    # and every requested observable must be a booked histogram.
    validate_requested(args.era, getattr(args, "systs", None) or [],
                       getattr(args, "omit_systs", None) or [])
    dy_reweight = getattr(args, "dy_reweight", None) or "none"
    if dy_reweight != "none" and args.dy not in (None, default_dy_variant(args.era)):
        raise ValueError(f"--dy-reweight corrects the {default_dy_variant(args.era)} DY sample only "
                         f"(config.yaml dy_reweight_datasets); --dy {args.dy} has nothing to reweight.")
    if "dyzpt" in (getattr(args, "systs", None) or []) and dy_reweight == "none":
        raise ValueError("--systs dyzpt varies the DY Z-pT reweighting, which is off by default: "
                         "add --dy-reweight zpt or --dy-reweight zpt+reshape.")
    if "dyreshape" in (getattr(args, "systs", None) or []) and dy_reweight != "zpt+reshape":
        raise ValueError("--systs dyreshape varies the DY reshape, which is off unless "
                         "--dy-reweight zpt+reshape.")
    from wrcoffea.histograms import all_histogram_names
    resolve_syst_observables(getattr(args, "syst_observables", None), all_histogram_names())

    if args.max_workers is not None and args.max_workers < 1:
        raise ValueError("--max-workers must be a positive integer")
    if args.threads_per_worker is not None and args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be a positive integer")
    if args.worker_wait_timeout is not None and args.worker_wait_timeout < 1:
        raise ValueError("--worker-wait-timeout must be a positive integer")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be a positive integer")
    if args.xrd_fallback_timeout < 1:
        raise ValueError("--xrd-fallback-timeout must be a positive integer")
    if args.xrd_fallback_retries_per_redirector < 1:
        raise ValueError("--xrd-fallback-retries-per-redirector must be a positive integer")
    if args.xrd_fallback_sleep < 0:
        raise ValueError("--xrd-fallback-sleep must be >= 0")


def normalize_by_sumw(hists_dict):
    """Normalize histograms by the on-the-fly accumulated sum of genWeights.

    When ``compute_sumw=True``, the processor fills histograms with
    ``genWeight * xsec * lumi * 1000`` (no ``/sumw``).  This function
    divides each dataset's histograms by the accumulated ``_sumw`` to
    complete the normalization.

    Recursively normalizes nested structures (e.g., cutflow histograms),
    but skips unweighted cutflow histograms (keys containing "unweighted")
    which hold raw event counts and must not be rescaled.  The MC
    event-weight distributions (``event_weights``) are never produced in
    this mode and are refused if present: they count events vs log10|w|,
    and weights lacking 1/sumw would put them at the wrong axis values.
    """
    import hist as hist_mod

    for dataset, data in hists_dict.items():
        sumw = data.pop("_sumw", None)
        if sumw is None or sumw == 0.0:
            continue
        logging.info("Normalizing %s by computed sumw = %.6g", dataset, sumw)

        def _normalize_recursive(obj, key=""):
            """Recursively normalize Hist objects in nested dicts."""
            if "unweighted" in key:
                return
            if key == "event_weights":
                raise RuntimeError(
                    f"Dataset '{dataset}' carries event-weight distributions in compute_sumw "
                    "(--unskimmed) mode; their log10|w| axis lacks the 1/sumw factor and cannot "
                    "be normalized by rescaling. The analyzer must not book them in this mode."
                )
            if key == "weight" and hasattr(obj, "value") and not isinstance(obj, hist_mod.Hist):
                # --save-events per-event weight column (event_lists/<region>/weight);
                # the event masses alongside it are left untouched.
                obj.value[...] /= sumw
                return
            if isinstance(obj, hist_mod.Hist):
                view = obj.view(flow=True)
                if hasattr(view, "value"):
                    view.value /= sumw
                    view.variance /= sumw * sumw
                else:
                    view /= sumw
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _normalize_recursive(v, key=k)

        for key, obj in data.items():
            _normalize_recursive(obj, key=key)

    return hists_dict


# ---------------------------------------------------------------------------
# Cluster context managers
# ---------------------------------------------------------------------------

# The data/ subdirectories the Condor workers read. wrcoffea/analysis_config.py
# resolves every data/... path of config.yaml against the package's own location,
# which on a worker is the job sandbox: Condor transfers each subdirectory there
# flat ("<subdir>/") and _CondorWorkerSetup links it back as data/<subdir>.
# tests/test_run_analysis.py checks that config.yaml uses no other subdirectory.
CONDOR_DATA_DIRS = ("lumis", "jsonpog", "reweights", "pdf_sets", "muon_highpt")

# Where lpcjobqueue has Condor put the worker logs (deleted after three days).
CONDOR_LOG_DIR_TEMPLATE = "/uscmst1b_scratch/lpc1/3DayLifetime/{user}/dask-logs"

# Condor worker size (measured 2026-10-14, docs/condor.md). The memory is the job's
# request_memory, a hard limit on the worker node, and the Dask worker's limit, whose pause
# (80 %) and restart (95 %) thresholds must stay above the worker's peak. Nominal workers
# peaked at 2.4-2.9 GB with 100k-event chunks (processing and merging), and the last merges
# of a composite with every signal point hold its whole output (about 1.2 GB): with 4 GB
# those merges killed four workers in a row and ended the run. With --systs the workers
# peaked at 3.9 GB. A 10 GB job waited minutes longer for a slot than a 2-4 GB one.
CONDOR_WORKER_MEMORY = "6GB"
CONDOR_WORKER_MEMORY_SYSTS = "8GB"
CONDOR_CHUNKSIZE = 100_000
LOCAL_CHUNKSIZE = 250_000

# The packages a Condor worker imports. A worker runs the coffea image with the shipped
# virtual environment (.env) in front of it, and nothing else: a package this process
# imports from the user site (~/.local) is missing or another version there. An
# lpcjobqueue in ~/.local, for example, makes the lpcjobqueue bootstrap skip installing it
# into .env, and every worker then dies at start ("No module named 'lpcjobqueue'", the
# workers preload lpcjobqueue.patch) while this process waits for them.
CONDOR_WORKER_MODULES = ("lpcjobqueue", "coffea", "awkward", "uproot", "dask", "distributed",
                         "numpy", "hist", "correctionlib")


def _check_condor_environment(modules=CONDOR_WORKER_MODULES):
    """Raise before any job is submitted if a package the workers need comes from the user site."""
    import importlib
    import site

    if not site.ENABLE_USER_SITE:
        return
    user_site = os.path.realpath(site.getusersitepackages())
    shadowed = []
    for name in modules:
        path = os.path.realpath(importlib.import_module(name).__file__)
        if path.startswith(user_site + os.sep):
            shadowed.append(os.path.dirname(path))
    if shadowed:
        raise RuntimeError(
            f"Imported from the user site {user_site}, which the Condor workers do not have: "
            f"{', '.join(shadowed)}. Delete these directories and their *.dist-info next to them, "
            "then start ./shell again (its .bashrc reinstalls lpcjobqueue into .env). See "
            "docs/condor.md."
        )


def _condor_job_states(cluster):
    """Condor states of the cluster's worker jobs (one job per worker): idle, running, held, and
    ended (completed, removed or gone from the queue), with one hold reason; None before any
    job is submitted. Queries the schedd lpcjobqueue submitted to, through its own thread."""
    from lpcjobqueue.schedd import SCHEDD, SCHEDD_POOL

    ids = {job.job_id for job in list(cluster.workers.values()) if getattr(job, "job_id", None)}
    if not ids:
        return None
    constraint = f"ClusterId >= {min(ids)} && ClusterId <= {max(ids)}"
    ads = SCHEDD_POOL.submit(
        lambda: SCHEDD().query(constraint=constraint, projection=["ClusterId", "JobStatus", "HoldReason"])
    ).result(timeout=120)
    states = {"idle": 0, "running": 0, "held": 0, "ended": 0, "hold_reason": None}
    seen = set()
    for ad in ads:
        cluster_id = int(ad["ClusterId"])
        if cluster_id not in ids:
            continue
        seen.add(cluster_id)
        status = int(ad["JobStatus"])   # 1 idle, 2 running, 3 removed, 4 completed, 5 held, 6 transferring
        if status == 1:
            states["idle"] += 1
        elif status in (2, 6, 7):
            states["running"] += 1
        elif status == 5:
            states["held"] += 1
            states["hold_reason"] = states["hold_reason"] or str(ad.get("HoldReason", ""))[:200]
        else:
            states["ended"] += 1
    states["ended"] += len(ids - seen)
    return states


def _port_in_use(exc) -> bool:
    """Whether a cluster start failed because its scheduler port was taken."""
    import errno

    while exc is not None:
        if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
            return True
        if "bind to local port" in str(exc):   # lpcjobqueue's own message for it
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _start_lpc_cluster(cluster_cls, *, attempts=5, **kwargs):
    """Start an LPCCondorCluster, retrying when its scheduler port is taken.

    lpcjobqueue draws the scheduler port at random from 10000-10100, the range open on the
    login nodes, so a run sharing the node with other Dask schedulers (yours or other
    users') can draw a busy one and fail with "Address already in use" (2026-10-14).
    """
    for attempt in range(1, attempts + 1):
        try:
            return cluster_cls(**kwargs)
        except (RuntimeError, OSError) as exc:
            if attempt == attempts or not _port_in_use(exc):
                raise
            logging.warning("Dask scheduler port already in use (attempt %d of %d); retrying.",
                            attempt, attempts)


def _start_condor_cluster(*, n_workers, memory):
    """Start an LPCCondorCluster of ``n_workers`` workers with the worker setup; return (cluster, client).

    lpcjobqueue copies wrcoffea/, bin/ and data/<CONDOR_DATA_DIRS> when the cluster starts, and
    every worker, also one that joins later, gets that copy.
    """
    from dask.distributed import Client, WorkerPlugin

    _check_condor_environment()
    from lpcjobqueue import LPCCondorCluster

    class _CondorWorkerSetup(WorkerPlugin):
        """Runs on every worker (including late arrivals) to fix paths."""
        data_dirs = CONDOR_DATA_DIRS

        def setup(self, worker):
            import sys, os
            for p in (".", "wrcoffea"):
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
            # Condor transfers each data/<subdir> as a flat "<subdir>/" but
            # the config references them under "data/...".  Recreate the
            # expected structure.
            os.makedirs("data", exist_ok=True)
            for subdir in self.data_dirs:
                src = os.path.abspath(subdir)
                dst = os.path.join("data", subdir)
                if os.path.isdir(src) and not os.path.exists(dst):
                    os.symlink(src, dst)

    repo_root = Path(__file__).resolve().parent.parent
    log_dir = CONDOR_LOG_DIR_TEMPLATE.format(user=os.environ["USER"])

    cluster = _start_lpc_cluster(
        LPCCondorCluster,
        ship_env=True,
        memory=memory,
        transfer_input_files=[
            str(repo_root / "wrcoffea"),
            str(repo_root / "bin"),
            *(str(repo_root / "data" / subdir) for subdir in CONDOR_DATA_DIRS),
        ],
        log_directory=log_dir,
    )
    cluster.scale(n_workers)

    client = Client(cluster)
    # Dask >=2025 deprecates register_worker_plugin in favor of register_plugin.
    if hasattr(client, "register_plugin"):
        client.register_plugin(_CondorWorkerSetup(), name="_CondorWorkerSetup")
    else:
        client.register_worker_plugin(_CondorWorkerSetup())
    return cluster, client


@contextmanager
def _condor_cluster(*, n_workers, wait_timeout_s, memory):
    """Start Condor workers for one run, yield the client, remove the workers on exit."""
    cluster, client = _start_condor_cluster(n_workers=n_workers, memory=memory)
    try:
        _wait_for_workers_or_raise(
            client,
            requested_workers=n_workers,
            min_workers=1,
            timeout_s=wait_timeout_s,
            label=f"condor_startup_{n_workers}workers",
            job_states=lambda: _condor_job_states(cluster),
        )
        logging.info(
            "Started with %d/%d workers; remaining will join dynamically.",
            len(client.scheduler_info()["workers"]), n_workers,
        )
        yield client
    finally:
        client.close()
        cluster.close()


# ---------------------------------------------------------------------------
# A long-lived cluster that several runs share (bin/condor_cluster.py)
# ---------------------------------------------------------------------------

# Scheduler metadata key under which bin/condor_cluster.py records what its workers run.
CLUSTER_IDENTITY_KEY = "wrcoffea-condor-cluster"
# Packages whose versions must be the same in a run and in the workers it uses.
CLUSTER_PACKAGES = ("coffea", "awkward", "uproot", "dask", "distributed", "numpy", "hist", "correctionlib")


def _worker_files(repo_root=None):
    """{path in the repository: sha256} of every file the Condor workers execute or read:
    wrcoffea/ (the code and config.yaml) and data/<CONDOR_DATA_DIRS> (payloads, golden JSONs).
    bin/ travels with them too, but runs only in the process that submits the work."""
    import hashlib

    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    files = {}
    for root in (repo_root / "wrcoffea", *(repo_root / "data" / d for d in CONDOR_DATA_DIRS)):
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(repo_root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def _cluster_identity(*, n_workers, memory, repo_root=None):
    """What a long-lived cluster's workers run: recorded on its scheduler when it starts."""
    from importlib.metadata import version

    return {
        "files": _worker_files(repo_root),
        "packages": {name: version(name) for name in CLUSTER_PACKAGES},
        "image": os.environ.get("COFFEA_IMAGE_FULL"),
        "workers": n_workers,
        "memory": memory,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _check_cluster_identity(identity, *, repo_root=None):
    """Refuse a cluster whose workers run other code, data or packages than this checkout.

    The workers keep the copy of wrcoffea/ and data/ made when the cluster started; the work this
    process submits names the analyzer by module, so the workers would run their old copy of it.
    """
    if not identity:
        raise RuntimeError("This scheduler records no identity, so it was not started by "
                           "bin/condor_cluster.py and its workers' code cannot be checked.")
    mine = _cluster_identity(n_workers=0, memory=None, repo_root=repo_root)
    theirs = identity.get("files", {})
    changed = sorted(p for p in set(theirs) | set(mine["files"]) if theirs.get(p) != mine["files"].get(p))
    problems = []
    if changed:
        shown = ", ".join(changed[:6]) + (f" and {len(changed) - 6} more" if len(changed) > 6 else "")
        problems.append(f"{len(changed)} file(s) of wrcoffea/ or data/ differ from the workers' copy: {shown}")
    packages = identity.get("packages", {})
    versions = [f"{name} {packages.get(name)} there, {ver} here" for name, ver in mine["packages"].items()
                if packages.get(name) != ver]
    if versions:
        problems.append("package versions differ: " + ", ".join(versions))
    if identity.get("image") != mine["image"]:
        problems.append(f"another image: {identity.get('image')} there, {mine['image']} here")
    if problems:
        raise RuntimeError(
            f"Cannot use the cluster started at {identity.get('started')}: " + "; ".join(problems)
            + ". Its workers keep what they got when it started: stop bin/condor_cluster.py and start "
            "it again from this checkout."
        )


@contextmanager
def _attached_cluster(scheduler, *, wait_timeout_s):
    """Use the workers of a cluster started by bin/condor_cluster.py (its scheduler address or
    scheduler file) after checking that they run this checkout; leave them running on exit."""
    from dask.distributed import Client

    if os.path.isfile(scheduler):
        client = Client(scheduler_file=scheduler, timeout=60)
    else:
        client = Client(scheduler, timeout=60)
    try:
        identity = client.get_metadata(CLUSTER_IDENTITY_KEY, None)
        _check_cluster_identity(identity)
        _wait_for_workers_or_raise(
            client,
            requested_workers=int(identity.get("workers") or 1),
            min_workers=1,
            timeout_s=wait_timeout_s,
            label="attached_cluster",
        )
        logging.info("Using the cluster started at %s: %d workers now.",
                     identity.get("started"), len(client.scheduler_info()["workers"]))
        yield client
    finally:
        client.close()


@contextmanager
def _local_cluster(*, n_workers, threads_per_worker):
    """Set up a local Dask cluster, yield client, clean up on exit."""
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def _xrd_fallback_enabled(args) -> bool:
    """Return whether preprocess redirector fallback is active for this run."""
    setting = getattr(args, "xrd_fallback", None)
    if setting is None:
        setting = getattr(args, "unskimmed", False)
    return bool(setting) and bool(getattr(args, "unskimmed", False))


def _rewrite_fileset_lfn_url(fileset: dict, *, lfn: str, new_url: str) -> int:
    """Rewrite all fileset URL keys whose LFN matches ``lfn``."""
    rewrites = 0
    for dataset in fileset.values():
        files = dataset.get("files")
        if not isinstance(files, dict):
            continue
        for old_url in list(files.keys()):
            if extract_lfn_from_url(old_url) != lfn or old_url == new_url:
                continue
            files[new_url] = files.pop(old_url)
            rewrites += 1
    return rewrites


def _preprocess_with_xrd_fallback(run, fileset: dict, *, treename: str, args):
    """Run coffea preprocess with redirector fallback for unskimmed filesets."""
    if not _xrd_fallback_enabled(args):
        return run.preprocess(fileset=fileset, treename=treename)

    timeout = int(getattr(args, "xrd_fallback_timeout", DEFAULT_TIMEOUT_SECONDS))
    retries = int(
        getattr(args, "xrd_fallback_retries_per_redirector", DEFAULT_RETRIES_PER_REDIRECTOR)
    )
    sleep_seconds = float(getattr(args, "xrd_fallback_sleep", DEFAULT_RETRY_SLEEP_SECONDS))

    resolved_cache: dict[str, str] = {}
    exhausted_cache: dict[str, str] = {}
    rewrite_count = 0
    redirector_hits: dict[str, int] = {}

    while True:
        try:
            preproc = run.preprocess(fileset=fileset, treename=treename)
            if resolved_cache:
                redirects = ", ".join(
                    f"{redirector}={count}"
                    for redirector, count in sorted(redirector_hits.items())
                )
                logging.info(
                    "XRootD fallback summary: resolved %d LFN(s), rewrote %d URL entr%s (%s)",
                    len(resolved_cache),
                    rewrite_count,
                    "y" if rewrite_count == 1 else "ies",
                    redirects or "no redirector substitutions",
                )
            return preproc
        except Exception as exc:
            failed_url = extract_root_url_from_error(exc)
            failed_lfn = extract_lfn_from_url(failed_url) if failed_url else None
            if failed_lfn is None:
                raise

            if failed_lfn in exhausted_cache:
                raise RuntimeError(
                    f"XRootD fallback already exhausted for {failed_lfn}: "
                    f"{exhausted_cache[failed_lfn]}"
                ) from exc

            if failed_lfn in resolved_cache:
                raise RuntimeError(
                    f"Preprocess still failing for {failed_lfn} after fallback to "
                    f"{resolved_cache[failed_lfn]}"
                ) from exc

            # Skip the redirector already present in the failing URL so we
            # always try a *different* route first.
            skip_redirectors = set()
            if failed_url:
                for r in REDIRECTORS:
                    if failed_url.startswith(r.rstrip("/") + "/"):
                        skip_redirectors.add(r)
                        break
            remaining = [r for r in REDIRECTORS if r not in skip_redirectors]

            probe_result = resolve_url_with_redirectors(
                failed_lfn,
                timeout=timeout,
                retries_per_redirector=retries,
                sleep_seconds=sleep_seconds,
                redirectors=remaining or None,
            )

            if (
                not probe_result.success
                or probe_result.resolved_url is None
                or probe_result.redirector is None
            ):
                summary = probe_result.failure_summary()
                exhausted_cache[failed_lfn] = summary
                raise RuntimeError(
                    f"XRootD fallback exhausted for {failed_lfn}: {summary}"
                ) from exc

            rewrites = _rewrite_fileset_lfn_url(
                fileset, lfn=failed_lfn, new_url=probe_result.resolved_url
            )

            resolved_cache[failed_lfn] = probe_result.resolved_url
            rewrite_count += rewrites
            redirector_hits[probe_result.redirector] = (
                redirector_hits.get(probe_result.redirector, 0) + 1
            )

            if rewrites <= 0:
                # Probe succeeded but the resolved URL matches the existing
                # fileset key (same redirector routed to a different SE, or a
                # transient issue cleared up).  Record and retry; the
                # resolved_cache guard above will raise on the next failure.
                logging.warning(
                    "XRootD fallback: probe succeeded for %s at %s but no "
                    "fileset rewrite needed (same URL); retrying preprocess.",
                    failed_lfn,
                    probe_result.redirector,
                )
                continue

            logging.warning(
                "XRootD fallback: %s -> %s (attempts=%d, rewritten=%d)",
                failed_lfn,
                probe_result.redirector,
                probe_result.total_attempts,
                rewrites,
            )


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _process_fileset(args, fileset, *, client, condor=False):
    """Preprocess and process a fileset, return histograms."""
    import uproot
    from coffea.nanoevents import NanoAODSchema
    from coffea.processor import Runner, DaskExecutor
    from coffea.processor.executor import UprootMissTreeError
    from wrcoffea.analyzer import WrAnalysis

    NanoAODSchema.warn_missing_crossrefs = False
    NanoAODSchema.error_missing_event_ids = False

    processor = WrAnalysis(
        mass_point=args.mass,
        enabled_systs=args.systs,
        region=args.region,
        compute_sumw=args.unskimmed,
        region_variants=resolve_studies(args.study),
        syst_observables=getattr(args, "syst_observables", None),
        omit_systs=getattr(args, "omit_systs", None),
        save_events=bool(getattr(args, "save_events", False)),
        dy_zpt_reweight=(getattr(args, "dy_reweight", None) or "none") != "none",
        signal_kfactor=not getattr(args, "no_signal_kfactor", False),
        dy_reshape=getattr(args, "dy_reweight", None) == "zpt+reshape",
        unblind=bool(getattr(args, "unblind", False)),
    )
    # compression=1 (coffea's default; it was None until 2026-10-14): each chunk's output stays
    # on its worker as lz4-compressed bytes, about 28 times smaller (a DY dataset's nominal
    # histograms: 3.9 MB pickled, 0.14 MB compressed), and a merge of the reduction tree
    # unpacks its inputs one at a time. Uncompressed, the merges of a composite with many
    # datasets held several large outputs at once and pushed 4 GB Condor workers over their
    # limit; a killed worker's merged results are recomputed from their chunks.
    run = Runner(
        executor=DaskExecutor(client=client, compression=1, retries=10, treereduction=10),
        chunksize=args.chunksize,
        maxchunks=args.maxchunks,
        # Skip bad files to continue processing with remaining files
        skipbadfiles=False,
        xrootdtimeout=getattr(args, "xrootd_timeout", None) or (60 if condor else 10),
        align_clusters=False,
        savemetrics=True,
        schema=NanoAODSchema,
    )

    label = args.sample
    if args.mass:
        label += f" ({args.mass})"

    logging.info("***PREPROCESSING %s***", label.upper())
    preproc = _preprocess_with_xrd_fallback(
        run, fileset, treename="Events", args=args
    )
    logging.info("Preprocessing completed")

    logging.info("***PROCESSING %s***", label.upper())
    hists, _ = run(preproc, treename="Events", processor_instance=processor)
    logging.info("Processing completed")
    return hists


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_local_composite(args, era, *, client, sample_to_group):
    """Sequentially process each sample of a composite mode on a local cluster.

    Output is saved through ``save_histograms_by_group`` with the same
    ``sample_to_group`` map used by the Condor path, so ``--split-groups``
    behaves identically in both execution modes.
    """
    from wrcoffea.save_hists import save_histograms_by_group

    composite_mode = args.sample
    samples = COMPOSITE_SAMPLES[composite_mode]
    has_signal = "Signal" in samples
    sig_points = (select_signal_points(era, getattr(args, "signal_points", None),
                                       getattr(args, "mwr_range", None)) if has_signal else [])

    for sample in samples:
        sample_runs = [(sample, None)]
        if sample == "Signal":
            sample_runs = [(sample, mp) for mp in sig_points]

        for sample_name, mass_point in sample_runs:
            label = f"{sample_name}" + (f" ({mass_point})" if mass_point else "")
            logging.info("=== Processing %s ===", label)

            # Temporarily set args for this sample.
            args.sample = sample_name
            args.mass = mass_point

            try:
                filepath = build_fileset_path(
                    era=era, sample=sample_name,
                    unskimmed=args.unskimmed, dy=args.dy,
                )
                sample_fileset = load_and_select_fileset(
                    filepath=filepath,
                    desired_process=sample_name,
                    mass=mass_point,
                    maxfiles=args.maxfiles,
                )
                hists = _process_fileset(args, sample_fileset, client=client, condor=False)

                if args.unskimmed:
                    hists = normalize_by_sumw(hists)

                if not args.debug:
                    save_histograms_by_group(hists, args, sample_to_group)

                logging.info("=== Finished %s ===", label)
            except Exception:
                logging.exception("Failed processing %s. Continuing...", label)

    args.sample = composite_mode
    args.mass = None


if __name__ == "__main__":
    ERA_CHOICES = list_eras()
    SAMPLE_CHOICES = list_samples()
    COMPOSITE_CHOICES = list(COMPOSITE_SAMPLES.keys())
    ALL_SAMPLE_CHOICES = SAMPLE_CHOICES + COMPOSITE_CHOICES
    DY_CHOICES = sorted(set(v for variants in DY_VARIANTS.values() for v in variants))

    parser = argparse.ArgumentParser(description="Processing script for WR analysis.")
    parser.add_argument("era", nargs="?", default=None, type=str, choices=ERA_CHOICES, help="Campaign to analyze.")
    parser.add_argument("sample", nargs="?", default=None, type=str, choices=ALL_SAMPLE_CHOICES, help="Sample to analyze (e.g., Signal, DYJets) or composite mode (all, data, bkg, mc, signal).")
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument("--dy", type=str, default=None, choices=DY_CHOICES, help="Specific DY sample to analyze (LO, NLO, etc)")
    optional.add_argument("--mass", type=str, default=None, help="Signal mass point to analyze.")
    optional.add_argument("--signal-points", nargs="+", default=None, metavar="POINT",
                          help="Signal points of a composite mode that includes Signal (signal, mc, all): "
                               "'default' (the 9-point grid, the default), 'all' (every point of "
                               "data/signal_points/<era>_mass_points.csv) or WR<wr>_N<n> tags.")
    optional.add_argument("--mwr-range", nargs=2, type=int, default=None, metavar=("MIN", "MAX"),
                          help="Keep only the signal points with MIN <= M_WR <= MAX (composite modes with Signal).")
    optional.add_argument("--fileset", type=Path, default=None, help="Override automatic fileset path with a custom fileset JSON.")
    optional.add_argument("--dir", type=str, default=None, help="Create a new output directory.")
    optional.add_argument("--name", type=str, default=None, help="Append the filenames of the output ROOT files.")
    optional.add_argument("--debug", action='store_true', help="Debug mode (don't compute histograms)")
    optional.add_argument("--unskimmed", action='store_true', help="Run on unskimmed files.")
    optional.add_argument("--condor", action='store_true', help="Run on Condor. Without this flag, composite modes run locally in sequential order.")
    optional.add_argument("--max-workers", type=int, default=None, help="Number of Dask workers (local default: 3, condor single-sample: 50, condor composite skimmed: 200, condor composite unskimmed: 500).")
    optional.add_argument("--worker-wait-timeout", type=int, default=3600,
                          help="Seconds to wait for the first Condor worker while the jobs are idle in a busy pool "
                               "(default: 3600). Worker jobs that end or are held before any worker connected "
                               "stop the run at once.")
    optional.add_argument("--scheduler", default=None, metavar="ADDRESS_OR_FILE",
                          help="Run on the workers of a cluster started with bin/condor_cluster.py: its scheduler "
                               "file (condor_scheduler.json) or address. Implies --condor and starts no workers; "
                               "refused if the workers' copy of wrcoffea/ and data/ differs from this checkout.")
    optional.add_argument("--worker-memory", default=None, metavar="SIZE",
                          help="Memory of each Condor worker: the job's request_memory and the Dask "
                               f"worker's memory limit (default: {CONDOR_WORKER_MEMORY}, {CONDOR_WORKER_MEMORY_SYSTS} "
                               "with --systs; see docs/condor.md).")
    optional.add_argument("--threads-per-worker", type=int, default=None, help="Threads per Dask worker for local runs (LocalCluster threads_per_worker).")
    optional.add_argument("--chunksize", type=int, default=None,
                          help=f"Number of events per processing chunk (default: {CONDOR_CHUNKSIZE} with --condor, "
                               f"{LOCAL_CHUNKSIZE} locally).")
    optional.add_argument("--maxchunks", type=int, default=None, help="Max chunks per dataset file (default: all). Use 1 for quick testing.")
    optional.add_argument("--maxfiles", type=int, default=None, help="Max files per dataset (default: all). Use 1 for quick testing.")
    optional.add_argument("--systs", nargs="*", default=[], choices=list(FAMILIES), metavar="FAMILY",
                          help=f"Enable systematic histogram variations (choices: {', '.join(FAMILIES)}). "
                               "Weight-based: pileup, sf, dyzpt, lsf, dyreshape, scale (QCD muR/muF), ps (ISR/FSR). Shape (re-run selections): jer, jes, elescale, "
                               "elesmear, muscale, musmear. Normalization (scaled at export): lumi. "
                               "A family the era cannot produce is an error, not a silent skip.")
    optional.add_argument("--omit-systs", nargs="*", default=[], metavar="SOURCE",
                          help="Deliberately leave out these systematic sources (e.g. ElectronTrigSf) "
                               "although their family is enabled. The omission is recorded in the "
                               "output manifest. Required when a source has no inputs for the era.")
    optional.add_argument("--syst-observables", nargs="*", default=None, metavar="OBS",
                          help="Histograms that receive the systematic variations (Nominal is always "
                               f"filled everywhere). Default: fit, the fit observables ({' '.join(FIT_OBSERVABLES)}). "
                               "'all' varies every histogram except the eta-phi maps (for the productions "
                               "behind the control plots), or list histogram names.")
    optional.add_argument("--region", type=str, default="both", choices=["resolved", "boosted", "both"], help="Analysis region to run: resolved, boosted, or both (default: both).")
    optional.add_argument("--unblind", action="store_true",
                          help="Fill the data signal regions (and their variants) and the full data cutflows. "
                               "Run 2 eras only. Default: data is blinded (no signal-region histograms or "
                               "events, ee/mumu cutflows stop before the mll cut); MC is never blinded.")
    optional.add_argument("--dy-reweight", choices=DY_REWEIGHT_CHOICES, default=None,
                          help="Corrections of the LO HT-binned DY sample, for eras with payloads in config.yaml "
                               "(dy_zpt_reweight, dy_reshape): none (default since 2026-10-12: the sample at "
                               "its NNLO-normalized cross section), zpt (the Z-pT QCD x EW reweight) or "
                               "zpt+reshape (also the data-driven jet-pT reshape, derived on top of the Z-pT "
                               "reweight). --systs dyzpt needs zpt or zpt+reshape, --systs dyreshape "
                               "zpt+reshape. The DY file is WRAnalyzer_DYJets.root for the default DY (lo_ht, "
                               "none), WRAnalyzer_DYJets_<variant>[_zpt|_zpt_reshape].root otherwise.")
    # Legacy switches from when both corrections were on by default (resolve_dy_reweight).
    optional.add_argument("--no-dy-zpt-reweight", action="store_true", help=argparse.SUPPRESS)
    optional.add_argument("--xrootd-timeout", type=int, default=None, metavar="SECONDS",
                          help="XRootD read timeout per chunk (default: 60 with --condor, 10 locally). Raise it when "
                               "streaming skims over a high-latency link ('Operation expired' read failures).")
    optional.add_argument("--no-dy-reshape", action="store_true", help=argparse.SUPPRESS)
    optional.add_argument("--no-signal-kfactor", action="store_true",
                          help="Do not apply the signal k(m_lN)/<k> shape reweighting for eras with a payload "
                               "in config.yaml (signal_kfactor).")
    optional.add_argument("--list-eras", action="store_true", help="Print available eras and exit.")
    optional.add_argument("--list-samples", action="store_true", help="Print available samples and exit.")
    optional.add_argument("--list-masses", action="store_true", help="Print available signal mass points for the given era (or all eras if none provided) and exit.")
    optional.add_argument("--study", nargs="*", default=[], choices=sorted(STUDIES), metavar="STUDY",
                          help="Add the extra regions of these studies to the output "
                               f"(choices: {', '.join(sorted(STUDIES))}). E.g. --study transfer-factor "
                               "adds SR/flavor-CR copies without the m > 800 GeV cut (suffix _nomass).")
    optional.add_argument("--split-groups", nargs="*", default=[], choices=SPLITTABLE_GROUPS, metavar="GROUP",
                          help=f"Composite runs: save these physics groups as one file per subgroup (choices: {', '.join(SPLITTABLE_GROUPS)}).")
    optional.add_argument("--save-events", action="store_true",
                          help="Also write the per-event (mass, nominal weight) of the signal regions as TTrees "
                               "events_<region> (resolved: mass_fourobject, boosted: mass_twoobject) into the output files.")
    optional.add_argument("--preflight-only", action="store_true", help="Validate fileset path/schema and selection, then exit without processing.")
    optional.add_argument(
        "--xrd-fallback",
        action="store_true",
        default=None,
        help="Enable XRootD redirector fallback during unskimmed preprocess (default: enabled for --unskimmed).",
    )
    optional.add_argument(
        "--xrd-fallback-timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Seconds per fallback probe open (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    optional.add_argument(
        "--xrd-fallback-retries-per-redirector",
        type=int,
        default=DEFAULT_RETRIES_PER_REDIRECTOR,
        help=f"Probe attempts per redirector during fallback (default: {DEFAULT_RETRIES_PER_REDIRECTOR}).",
    )
    optional.add_argument(
        "--xrd-fallback-sleep",
        type=float,
        default=DEFAULT_RETRY_SLEEP_SECONDS,
        help=f"Seconds between fallback retries (default: {DEFAULT_RETRY_SLEEP_SECONDS}).",
    )
    args = parser.parse_args()

    if args.xrd_fallback is None:
        args.xrd_fallback = bool(args.unskimmed)
    if args.scheduler:
        if args.max_workers is not None or args.worker_memory is not None:
            parser.error("--max-workers and --worker-memory size the workers of bin/condor_cluster.py, "
                         "not of a run with --scheduler.")
        args.condor = True
    if args.chunksize is None:
        args.chunksize = CONDOR_CHUNKSIZE if args.condor else LOCAL_CHUNKSIZE
    if args.worker_memory is None:
        args.worker_memory = CONDOR_WORKER_MEMORY_SYSTS if args.systs else CONDOR_WORKER_MEMORY

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
    args.dy_reweight = resolve_dy_reweight(args)

    # The mass-point CSV is read only for runs that involve Signal: a data-only era has none.
    MASS_CHOICES = mass_choices_for(args.era, args.sample)

    logging.info(f"Analyzing {args.era} - {args.sample} events")
    validate_arguments(args, MASS_CHOICES)
    if args.unblind:
        logging.warning("--unblind: data signal regions and full data cutflows will be filled.")
    run, year, era = get_era_details(args.era)

    is_composite = args.sample in COMPOSITE_SAMPLES
    is_condor = args.condor

    # --- Load fileset ---
    if is_composite:
        has_signal = "Signal" in COMPOSITE_SAMPLES[args.sample]
        sig_points = (select_signal_points(era, args.signal_points, args.mwr_range)
                      if has_signal else None)
        if sig_points:
            logging.info("Signal points: %s", ", ".join(sig_points))

        fileset = load_composite_fileset(
            era=era,
            composite_mode=args.sample,
            unskimmed=args.unskimmed,
            dy=args.dy,
            signal_points=sig_points,
            maxfiles=args.maxfiles,
        )
        sample_to_group = build_sample_to_group_map(fileset, signal_points=sig_points)

        # Optionally split parent groups (e.g. tt_tW -> TTbar, tW) into
        # separate output files.  The same map is used by both the Condor
        # and the local composite paths.
        split_groups = list(args.split_groups)
        if split_groups:
            sample_to_group = split_sample_groups(sample_to_group, split_groups)
            logging.info("Split groups %s into %s", split_groups,
                         sorted(set(sample_to_group.values())))

        n_files = sum(len(ds.get("files", {})) for ds in fileset.values())
        logging.info(
            "Composite '%s': merged %d dataset(s), %d file(s).",
            args.sample, len(fileset), n_files,
        )
    else:
        if args.fileset:
            filepath = args.fileset
        else:
            filepath = build_fileset_path(era=era, sample=args.sample, unskimmed=args.unskimmed, dy=args.dy)

        logging.info(f"Reading files from {filepath}")

        fileset = load_and_select_fileset(
            filepath=filepath,
            desired_process=args.sample,
            mass=args.mass,
            maxfiles=args.maxfiles,
        )

        n_files = sum(len(ds.get("files", {})) for ds in fileset.values())
        logging.info(
            "Selected %d dataset(s), %d file(s) after filtering.",
            len(fileset), n_files,
        )

    if args.preflight_only:
        logging.info("Preflight-only requested; exiting before processing.")
        raise SystemExit(0)

    # --- Process and save ---
    t0 = time.monotonic()

    if is_condor:
        if args.scheduler:
            cluster_context = _attached_cluster(args.scheduler, wait_timeout_s=args.worker_wait_timeout)
        else:
            if is_composite:
                default_workers = 500 if args.unskimmed else 200
            else:
                default_workers = 50
            n_workers = args.max_workers or default_workers
            cluster_context = _condor_cluster(n_workers=n_workers, wait_timeout_s=args.worker_wait_timeout,
                                              memory=args.worker_memory)
        with cluster_context as client:
            try:
                hists = _process_fileset(args, fileset, client=client, condor=True)

                if args.unskimmed:
                    hists = normalize_by_sumw(hists)

                if not args.debug:
                    if is_composite:
                        from wrcoffea.save_hists import save_histograms_by_group
                        save_histograms_by_group(hists, args, sample_to_group)
                    else:
                        from wrcoffea.save_hists import save_histograms
                        save_histograms(hists, args)

                if args.scheduler:
                    logging.info("All output saved. The shared cluster keeps its workers.")
                else:
                    logging.info("All output saved. Shutting down cluster (safe to Ctrl+C)...")
            except Exception:
                _dump_dask_diagnostics(
                    client,
                    label=f"run_analysis_{args.era}_{args.sample}_condor",
                )
                logging.exception("Condor processing failed.")
                raise
    else:
        n_workers = args.max_workers or 3
        with _local_cluster(n_workers=n_workers, threads_per_worker=args.threads_per_worker or 1) as client:
            if is_composite:
                _run_local_composite(args, era, client=client, sample_to_group=sample_to_group)
            else:
                try:
                    hists = _process_fileset(args, fileset, client=client, condor=False)

                    if args.unskimmed:
                        hists = normalize_by_sumw(hists)

                    if not args.debug:
                        from wrcoffea.save_hists import save_histograms
                        save_histograms(hists, args)
                except Exception:
                    _dump_dask_diagnostics(
                        client,
                        label=f"run_analysis_{args.era}_{args.sample}_local",
                    )
                    logging.exception("Local processing failed.")
                    raise

    exec_time = time.monotonic() - t0
    logging.info(f"Execution took {exec_time/60:.2f} minutes")
