#!/usr/bin/env python3
"""Quick test: skim one TTto2L2Nu file with low memory, then compare to Wisconsin."""
import logging
import os

import uproot

from wrcoffea.skimmer import skim_single_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SRC = "root://cmsxrootd.fnal.gov//store/mc/RunIII2024Summer24NanoAODv15/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/100000/016a8301-8bde-443d-a9f2-9dd5837b3f6d.root"
DEST = "data/skims/test_TTto2L2Nu/016a8301-8bde-443d-a9f2-9dd5837b3f6d_skim.root"
WISC = "root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24/backgrounds/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8_skim0.root"

# ── Step 1: Skim using the library (now memory-efficient) ─────────────────
logger.info("Skimming %s", SRC.split("/")[-1])
result = skim_single_file(SRC, DEST)
logger.info("Result: %d -> %d events (%.1f%%), %.1f MB",
            result.n_events_before, result.n_events_after,
            result.efficiency, result.file_size_bytes / 1_048_576)


# ── Step 2: Compare to Wisconsin skim ────────────────────────────────────
logger.info("")
logger.info("=" * 70)
logger.info("COMPARISON: new skim vs Wisconsin skim")
logger.info("=" * 70)

new = uproot.open(DEST)
wisc = uproot.open(WISC)

# Tree names
logger.info("")
logger.info("Trees in new skim:  %s", sorted(new.keys(cycle=False, filter_classname="TTree")))
logger.info("Trees in Wisconsin: %s", sorted(wisc.keys(cycle=False, filter_classname="TTree")))

# Events tree comparison
new_ev = new["Events"]
wisc_ev = wisc["Events"]

new_branches = set(new_ev.keys())
wisc_branches = set(wisc_ev.keys())

logger.info("")
logger.info("Events branch count — new: %d, Wisconsin: %d", len(new_branches), len(wisc_branches))

only_new = sorted(new_branches - wisc_branches)
only_wisc = sorted(wisc_branches - new_branches)
common = sorted(new_branches & wisc_branches)

if only_new:
    logger.info("Branches ONLY in new skim (%d):", len(only_new))
    for b in only_new:
        logger.info("  + %s", b)
if only_wisc:
    logger.info("Branches ONLY in Wisconsin skim (%d):", len(only_wisc))
    for b in only_wisc:
        logger.info("  - %s", b)
if not only_new and not only_wisc:
    logger.info("Branch sets are IDENTICAL.")

# Runs tree comparison
logger.info("")
if "Runs" in new and "Runs" in wisc:
    new_runs = new["Runs"]
    wisc_runs = wisc["Runs"]
    new_rb = set(new_runs.keys())
    wisc_rb = set(wisc_runs.keys())
    logger.info("Runs branch count — new: %d, Wisconsin: %d", len(new_rb), len(wisc_rb))
    only_new_r = sorted(new_rb - wisc_rb)
    only_wisc_r = sorted(wisc_rb - new_rb)
    if only_new_r:
        logger.info("Runs branches ONLY in new: %s", only_new_r)
    if only_wisc_r:
        logger.info("Runs branches ONLY in Wisconsin: %s", only_wisc_r)
    if not only_new_r and not only_wisc_r:
        logger.info("Runs branch sets are IDENTICAL.")

    # Compare Runs values
    logger.info("")
    logger.info("Runs tree values:")
    for b in sorted(new_rb & wisc_rb):
        nv = new_runs[b].array(library="np")
        wv = wisc_runs[b].array(library="np")
        logger.info("  %-40s  new=%s  wisc=%s", b, nv, wv)
else:
    logger.info("Runs tree: new=%s, Wisconsin=%s",
                "present" if "Runs" in new else "MISSING",
                "present" if "Runs" in wisc else "MISSING")

# Event count comparison
new_nevt = new_ev.num_entries
wisc_nevt = wisc_ev.num_entries
logger.info("")
logger.info("Event counts — new: %d, Wisconsin: %d", new_nevt, wisc_nevt)
logger.info("(Wisconsin is a merged file so different event count is expected)")

# Spot-check a few branch dtypes
logger.info("")
logger.info("Dtype comparison for common branches (first 20):")
for b in common[:20]:
    nt = new_ev[b].typename
    wt = wisc_ev[b].typename
    match = "OK" if nt == wt else "MISMATCH"
    logger.info("  %-40s  new=%-30s wisc=%-30s [%s]", b, nt, wt, match)

new.close()
wisc.close()
logger.info("")
logger.info("Done.")
