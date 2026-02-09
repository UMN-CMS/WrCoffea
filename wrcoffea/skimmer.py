"""Core skimming library for the WR analysis.

Provides event selection, Runs tree handling, and single-file skimming.
All functions are pure (no CLI coupling, no print statements) and return
structured data for the caller to log or inspect.
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import awkward as ak
import dask_awkward as dak
import numpy as np
import uproot
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from dask import compute


logger = logging.getLogger(__name__)

NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.error_missing_event_ids = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XROOTD_TIMEOUT = 10  # seconds
os.environ.setdefault("XRD_REQUESTTIMEOUT", str(XROOTD_TIMEOUT))
os.environ.setdefault("XRD_CONNECTIONTIMEOUT", str(XROOTD_TIMEOUT))

REDIRECTORS = [
    "root://cmsxrootd.fnal.gov/",
    "root://cms-xrd-global.cern.ch/",
    "root://xrootd-cms.infn.it/",
]
MAX_RETRIES_PER_REDIRECTOR = 15

# Skim cuts are intentionally looser than analysis cuts (analysis_config.CUTS)
# so skimmed files remain usable as the analysis evolves.
SKIM_CUTS = {
    "lepton_pt_min": 45,
    "lepton_eta_max": 2.5,
    "lead_lepton_pt_min": 52,
    "sublead_lepton_pt_min": 45,
    "ak4_pt_min": 32,
    "ak4_eta_max": 2.5,
    "ak8_pt_min": 180,
    "ak8_eta_max": 2.5,
}

KEPT_BRANCHES = {
    "event", "run", "luminosityBlock",
    "Electron", "Muon", "GenDressedLepton",
    "Jet", "GenJet", "FatJet", "GenJetAK8",
    "GenPart", "genWeight",
    "HLT", "LHE", "LHEPart", "LHEScaleWeight",
    "MET", "Pileup", "bunchCrossing", "Generator",
    "PSWeight", "GenMET",
    "LHEPdfWeight", "LHEReweightingWeight", "LHEWeight",
    "Flag", "genTtbarId", "GenProton",
    "HLTriggerFinalPath", "HLTriggerFirstPath",
    "Rho", "PV", "SV", "SubGenJetAK8", "SubJet", "OtherPV",
}

# Runs-tree branches whose values should be *summed* when collapsing
# multi-entry Runs trees into a single entry.
SUMMED_RUNS_BRANCHES = {
    "genEventSumw", "genEventSumw_",
    "genEventSumw2", "genEventSumw2_",
    "genEventCount", "genEventCount_",
    "nEvents",
}


# ---------------------------------------------------------------------------
# Dataclass for skim results
# ---------------------------------------------------------------------------

@dataclass
class SkimResult:
    """Structured result from skimming a single file."""
    src_path: str
    dest_path: str
    n_events_before: int
    n_events_after: int
    file_size_bytes: int
    efficiency: float


# ---------------------------------------------------------------------------
# Branch filtering helpers
# ---------------------------------------------------------------------------

def _kept_branch_names(tree_keys):
    """Return the subset of TTree branch names matching KEPT_BRANCHES.

    KEPT_BRANCHES contains collection-level names (e.g. "Electron", "Jet")
    and scalar names (e.g. "event", "genWeight").  This maps them to actual
    NanoAOD branch names like ``Electron_pt``, ``nElectron``, etc.
    """
    kept = []
    for bname in tree_keys:
        # Exact match (scalar branches like "event", "run", "genWeight")
        if bname in KEPT_BRANCHES:
            kept.append(bname)
            continue
        # Collection count branch: "nElectron" -> check "Electron"
        # Must be exactly n{Collection} — no underscore suffix.
        if bname.startswith("n") and bname[1:] in KEPT_BRANCHES:
            kept.append(bname)
            continue
        # Collection sub-branch: "Electron_pt" -> check "Electron"
        prefix = bname.split("_")[0]
        if prefix in KEPT_BRANCHES:
            kept.append(bname)
    return kept


# ---------------------------------------------------------------------------
# Runs tree handling — BUG FIX: collapse multi-entry to single entry
# ---------------------------------------------------------------------------

def read_runs_tree(src_path: str) -> dict | None:
    """Read the Runs tree from a source NanoAOD file, collapsing to one entry.

    NanoAOD files that were themselves produced by hadd can contain multiple
    Runs entries (one per original production file).  Copying them all leads
    to inflated genEventSumw totals after a second hadd during skim merging.

    This function collapses:
    - ``genEventSumw``, ``genEventCount``, etc. → **summed** into one value
    - All other branches → first entry kept
    """
    try:
        with uproot.open(src_path) as fin:
            if "Runs" not in fin:
                return None
            runs = fin["Runs"].arrays(library="np")
            collapsed: dict = {}
            for branch, arr in runs.items():
                if arr.dtype.kind == "O":
                    continue  # skip object-dtype branches (strings, ragged) — not writable
                if any(s in branch for s in SUMMED_RUNS_BRANCHES):
                    collapsed[branch] = np.array([arr.sum()])
                else:
                    collapsed[branch] = arr[:1]
            return collapsed
    except Exception as e:
        logger.warning("Failed to read Runs from %s: %s", src_path, e)
        return None


# ---------------------------------------------------------------------------
# Skim selection
# ---------------------------------------------------------------------------

def apply_skim_selection(events):
    """Apply loose skim selection to events.

    Requires:
      - >= 2 leptons (leading > 52 GeV, subleading > 45 GeV)
        with per-lepton preselection: pT > 45 GeV, |eta| < 2.5
      - (>= 2 AK4 jets with pT > 32 GeV) OR (>= 1 FatJet with pT > 180 GeV)

    Returns
    -------
    skimmed : awkward array
        Events passing the selection.
    summary : dict
        Selection summary with counts for logging.
    """
    c = SKIM_CUTS

    # --- Lepton preselection ---
    e = events.Electron
    m = events.Muon

    loose_electrons = e[(abs(e.eta) < c["lepton_eta_max"]) & (e.pt > c["lepton_pt_min"])]
    loose_muons = m[(abs(m.eta) < c["lepton_eta_max"]) & (m.pt > c["lepton_pt_min"])]

    leptons = ak.with_name(
        ak.concatenate((loose_electrons, loose_muons), axis=1),
        "PtEtaPhiMCandidate",
    )
    leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
    leptons_padded = ak.pad_none(leptons, 2, clip=True)

    leading = leptons_padded[:, 0]
    subleading = leptons_padded[:, 1]

    lep_mask = (
        (ak.fill_none(leading.pt, 0) > c["lead_lepton_pt_min"])
        & (ak.fill_none(subleading.pt, 0) > c["sublead_lepton_pt_min"])
    )

    # --- Hadronic selection ---
    ak4_pass = (events.Jet.pt > c["ak4_pt_min"]) & (abs(events.Jet.eta) < c["ak4_eta_max"])
    two_ak4 = ak.sum(ak4_pass, axis=1) >= 2

    ak8_pass = (events.FatJet.pt > c["ak8_pt_min"]) & (abs(events.FatJet.eta) < c["ak8_eta_max"])
    one_ak8 = ak.sum(ak8_pass, axis=1) >= 1

    had_mask = two_ak4 | one_ak8

    # --- Final mask ---
    event_mask = lep_mask & had_mask
    skimmed = events[event_mask]

    summary = {
        "lead_pt_cut": c["lead_lepton_pt_min"],
        "sublead_pt_cut": c["sublead_lepton_pt_min"],
        "ak4_pt_cut": c["ak4_pt_min"],
        "ak8_pt_cut": c["ak8_pt_min"],
    }
    return skimmed, event_mask, summary


def select_branches(events, kept=None):
    """Keep only branches in *kept* (default: KEPT_BRANCHES).

    Returns
    -------
    filtered : awkward array
    kept_list : list[str]
    dropped_list : list[str]
    """
    if kept is None:
        kept = KEPT_BRANCHES
    available = ak.fields(events)
    keep = [f for f in available if f in kept]
    dropped = sorted(f for f in available if f not in kept)
    return events[keep], keep, dropped


# ---------------------------------------------------------------------------
# Single-file skim
# ---------------------------------------------------------------------------

def _extract_lfn(url: str) -> str | None:
    """Extract the logical file name from an XRootD URL.

    Returns None for local paths (no retry/redirector logic needed).
    """
    if not url.startswith("root://"):
        return None
    # root://host(:port)//store/... -> /store/...
    idx = url.find("//store/")
    if idx >= 0:
        return url[idx + 1:]
    return None


SKIM_STAGES = 4  # open, select, compute, write — used by callers for progress bars


def _skim_impl(src_path: str, dest_path: str, progress=None) -> SkimResult:
    """Core skim logic for a single file (no retry handling).

    Uses coffea/dask only for lightweight mask computation, then reads
    selected events directly with uproot (no dask overhead for the heavy
    branch-materialisation step).

    Parameters
    ----------
    progress : callable or None
        Called as ``progress(stage_label)`` after each stage completes.
    """
    use_log = progress is None  # log steps when no progress bar

    def _step(label):
        if progress is not None:
            progress(label)

    def _mem_mb():
        """Peak RSS in MB."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    # 1. Compute selection mask via coffea/dask (only touches pt/eta branches)
    events = NanoEventsFactory.from_root(
        {src_path: "Events"},
        mode="dask",
        schemaclass=NanoAODSchema,
    ).events()

    n_before = dak.num(events, axis=0).compute()
    if use_log:
        logger.info("  Opened — %d events  [peak RSS: %.0f MB]", n_before, _mem_mb())
    _step("selecting")

    if use_log:
        logger.info("  Computing selection mask...")
    _, event_mask, _ = apply_skim_selection(events)
    mask_np = event_mask.compute()
    n_after = int(np.sum(mask_np))

    # Free all dask/coffea objects before the heavy read
    del events, event_mask
    gc.collect()

    if use_log:
        logger.info("  Selected %d / %d events (%.1f%%)  [peak RSS: %.0f MB]",
                    n_after, n_before,
                    (n_after / n_before * 100) if n_before else 0, _mem_mb())
    _step("computing")

    # 2. Stream: read event chunks from source, write directly to output.
    #    Never holds more than one chunk of all branches in memory.
    if use_log:
        logger.info("  Streaming selected events to output...")
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    STEP = 50_000  # source entries per chunk (before cut)

    with uproot.open(src_path) as fin:
        tree = fin["Events"]
        total_entries = tree.num_entries
        kept = _kept_branch_names(tree.keys())
        if use_log:
            logger.info("    Keeping %d / %d branches", len(kept), len(tree.keys()))

        with uproot.recreate(dest_path) as fout:
            first_chunk = True
            chunk_num = 0
            for start in range(0, total_entries, STEP):
                stop = min(start + STEP, total_entries)
                chunk_mask = mask_np[start:stop]
                if not np.any(chunk_mask):
                    continue
                chunk = tree.arrays(
                    kept, entry_start=start, entry_stop=stop, library="ak"
                )
                chunk = chunk[chunk_mask]
                chunk_dict = {f: chunk[f] for f in chunk.fields}
                if first_chunk:
                    fout["Events"] = chunk_dict
                    first_chunk = False
                else:
                    fout["Events"].extend(chunk_dict)
                del chunk, chunk_dict, chunk_mask
                gc.collect()
                chunk_num += 1
                if use_log:
                    logger.info("    Chunk %d written  [peak RSS: %.0f MB]",
                                chunk_num, _mem_mb())

            # Runs tree
            runs_payload = read_runs_tree(src_path)
            if runs_payload is not None:
                fout["Runs"] = runs_payload
            else:
                logger.warning("No Runs tree in source; skipping Runs for %s", dest_path)

    del mask_np
    gc.collect()
    _step("writing")

    file_size = os.path.getsize(dest_path)
    efficiency = (n_after / n_before * 100) if n_before > 0 else 0.0
    _step("done")

    return SkimResult(
        src_path=src_path,
        dest_path=dest_path,
        n_events_before=n_before,
        n_events_after=n_after,
        file_size_bytes=file_size,
        efficiency=efficiency,
    )


def skim_single_file(src_path: str, dest_path: str, progress=None) -> SkimResult:
    """Skim one NanoAOD file with XRootD retry and redirector fallback.

    For XRootD URLs: retries up to MAX_RETRIES_PER_REDIRECTOR times per
    redirector, then falls back to the next redirector in REDIRECTORS.
    For local paths: calls the implementation directly with no retries.

    Parameters
    ----------
    src_path : str
        Input NanoAOD file (local path or XRootD URL).
    dest_path : str
        Output ROOT file path.
    progress : callable or None
        Forwarded to ``_skim_impl``; called as ``progress(stage_label)``
        after each processing stage completes.

    Returns
    -------
    SkimResult
    """
    lfn = _extract_lfn(src_path)
    if lfn is None:
        return _skim_impl(src_path, dest_path, progress=progress)

    for redir_idx, redirector in enumerate(REDIRECTORS):
        url = f"{redirector}{lfn}"
        for attempt in range(1, MAX_RETRIES_PER_REDIRECTOR + 1):
            try:
                logger.info(
                    "Attempt %d/%d via %s",
                    attempt, MAX_RETRIES_PER_REDIRECTOR, redirector,
                )
                return _skim_impl(url, dest_path, progress=progress)
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d failed (%s): %s",
                    attempt, MAX_RETRIES_PER_REDIRECTOR, redirector, exc,
                )
                time.sleep(1)
        if redir_idx < len(REDIRECTORS) - 1:
            logger.info("Switching to redirector: %s", REDIRECTORS[redir_idx + 1])

    raise RuntimeError(
        f"All retries exhausted for {lfn} across {len(REDIRECTORS)} redirectors"
    )

