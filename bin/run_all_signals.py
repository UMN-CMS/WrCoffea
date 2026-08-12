#!/usr/bin/env python3
"""Re-run a full block of signal mass points in a single Condor cluster.

The stock ``run_analysis.py signal`` composite mode only processes the 9
default points from ``SIGNAL_WR_GRID``.  This driver reuses the same tested
composite-condor machinery (one cluster, per-mass output ROOT files via
``save_histograms_by_group``) but feeds it *every* signal point in a chosen
M_WR range.

Default: RunIISummer20UL18, M_WR in [1000, 6000] (481 points), skimmed,
genWeight-only (whatever the current analyzer is configured to apply),
written to WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/<--dir>/.

Must be run from the repository root (uses relative data/ paths), e.g.:
    python bin/run_all_signals.py --dir 20260624_signals
"""
import os
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")

import argparse
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

from wrcoffea.era_utils import get_era_details
from wrcoffea.cli_utils import (
    build_sample_to_group_map,
    load_composite_fileset,
    load_masses_from_csv,
)
from wrcoffea.save_hists import save_histograms_by_group

# Reuse the exact cluster + processing helpers used by run_analysis.py.
from run_analysis import (
    _condor_cluster,
    _process_fileset,
    _dump_dask_diagnostics,
    normalize_by_sumw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_MASS_RE = re.compile(r"WR(\d+)_N(\d+)")


def select_mass_points(era: str, min_mwr: int, max_mwr: int) -> list[str]:
    """All signal points for ``era`` with min_mwr <= M_WR <= max_mwr."""
    masses = load_masses_from_csv(Path(f"data/signal_points/{era}_mass_points.csv"))
    pts = []
    for tag in masses:
        m = _MASS_RE.match(tag)
        if m and min_mwr <= int(m.group(1)) <= max_mwr:
            pts.append(tag)
    return pts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18", help="Campaign (default: RunIISummer20UL18).")
    p.add_argument("--dir", default="20260624_signals", help="Output subdir under rootfiles/<run>/<year>/<era>/.")
    p.add_argument("--min-mwr", type=int, default=1000, help="Lowest M_WR to include (default: 1000).")
    p.add_argument("--max-mwr", type=int, default=6000, help="Highest M_WR to include (default: 6000).")
    p.add_argument("--region", default="both", choices=["resolved", "boosted", "both"])
    p.add_argument("--unskimmed", action="store_true", help="Run on unskimmed files (efficiency-scaled via sumw).")
    p.add_argument("--max-workers", type=int, default=200, help="Condor workers (default: 200).")
    p.add_argument("--worker-wait-timeout", type=int, default=1200)
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--maxchunks", type=int, default=None)
    p.add_argument("--maxfiles", type=int, default=None, help="Cap files per dataset (use 1 for a quick test).")
    p.add_argument("--dry-run", action="store_true", help="Build the fileset and print the plan, then exit.")
    a = p.parse_args()

    run, year, era = get_era_details(a.era)

    sig_points = select_mass_points(era, a.min_mwr, a.max_mwr)
    if not sig_points:
        raise SystemExit(f"No signal points found for {era} in M_WR [{a.min_mwr}, {a.max_mwr}].")
    logging.info("Selected %d signal points (M_WR %d-%d) for %s",
                 len(sig_points), a.min_mwr, a.max_mwr, era)

    fileset = load_composite_fileset(
        era=era,
        composite_mode="signal",
        unskimmed=a.unskimmed,
        dy=None,
        signal_points=sig_points,
        maxfiles=a.maxfiles,
    )
    sample_to_group = build_sample_to_group_map(fileset, signal_points=sig_points)
    n_files = sum(len(ds.get("files", {})) for ds in fileset.values())
    logging.info("Loaded %d signal dataset(s), %d file(s). Output dir: %s",
                 len(fileset), n_files, a.dir)

    # args namespace consumed by _process_fileset / save_histograms_by_group.
    proc_args = SimpleNamespace(
        era=a.era, sample="signal", mass=None, dy=None,
        region=a.region, systs=[], tf_study=False,
        unskimmed=a.unskimmed, chunksize=a.chunksize, maxchunks=a.maxchunks,
        debug=False, dir=a.dir, name=None,
        # xrd fallback only matters for unskimmed preprocess
        xrd_fallback=bool(a.unskimmed), xrd_fallback_timeout=30,
        xrd_fallback_retries_per_redirector=3, xrd_fallback_sleep=5,
    )

    if a.dry_run:
        logging.info("Dry run: first 3 points %s ... last %s",
                     sig_points[:3], sig_points[-1])
        return

    t0 = time.monotonic()
    with _condor_cluster(n_workers=a.max_workers, wait_timeout_s=a.worker_wait_timeout) as client:
        try:
            hists = _process_fileset(proc_args, fileset, client=client, condor=True)
            if a.unskimmed:
                hists = normalize_by_sumw(hists)
            save_histograms_by_group(hists, proc_args, sample_to_group)
            logging.info("All %d signal outputs saved to rootfiles/%s/%s/%s/%s. "
                         "Safe to Ctrl+C once paths are printed.",
                         len(fileset), run, year, era, a.dir)
        except Exception:
            _dump_dask_diagnostics(client, label=f"run_all_signals_{era}")
            logging.exception("Signal batch processing failed.")
            raise
    logging.info("Execution took %.2f minutes", (time.monotonic() - t0) / 60)


if __name__ == "__main__":
    main()
