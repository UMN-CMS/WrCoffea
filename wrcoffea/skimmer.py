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
from wrcoffea.xrootd_fallback import (
    DEFAULT_RETRIES_PER_REDIRECTOR,
    DEFAULT_RETRY_SLEEP_SECONDS,
    REDIRECTORS,
)


logger = logging.getLogger(__name__)

NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.error_missing_event_ids = False


# ---------------------------------------------------------------------------
# Monkey-patch: fix uproot shared-counter deduplication bug
# ---------------------------------------------------------------------------
# uproot's _cascadetree.Tree.__init__ has two bugs triggered when
# counter_name maps multiple jagged branches to the same counter
# (e.g. Jet_pt and Jet_eta both → nJet):
#
# Bug 1: The dedup code does `del _branch_data[_branch_lookup[name]]`
#   which shifts indices without updating _branch_lookup. With many
#   branches (1200+ in NanoAOD) this cascading index corruption deletes
#   data branches instead of counters. Result: stale counter references
#   cause `struct.pack(">...", None)` during write_anew.
#
# Bug 2: extend() validates shared counters via awkward.to_numpy() on a
#   big-endian numpy array (dtype ">u4"), which awkward rejects.
#
# Fix strategy: call __init__ with an identity counter_name (no
# collisions → no dedup → no corruption), then correctly merge counters.

def _patch_uproot_shared_counters():
    import uproot.writing._cascadetree as _ct

    # Guard: don't re-apply if already patched.
    if getattr(_ct.Tree.__init__, '_shared_counter_patched', False):
        return

    # Bug 2 fix: patch awkward.to_numpy to pass through numpy arrays.
    try:
        _ak = uproot.extras.awkward()
        _orig_to_numpy = _ak.to_numpy

        def _safe_to_numpy(array, *args, **kwargs):
            if isinstance(array, np.ndarray):
                return array
            return _orig_to_numpy(array, *args, **kwargs)

        _ak.to_numpy = _safe_to_numpy
    except ModuleNotFoundError:
        pass

    # Bug 1 fix: intercept __init__ to prevent buggy dedup.
    _original_init = _ct.Tree.__init__

    def _patched_init(self, directory, name, title, branch_types,
                      freesegments, counter_name, field_name,
                      initial_basket_capacity, resize_factor):
        # Quick check: does counter_name produce any collisions?
        if isinstance(branch_types, dict):
            items = list(branch_types.items())
        else:
            items = list(branch_types)

        seen_counters: dict = {}
        has_collisions = False
        for bname, btype in items:
            is_jagged = isinstance(btype, str) and btype.strip().startswith("var *")
            if is_jagged:
                cname = counter_name(bname)
                if cname in seen_counters:
                    has_collisions = True
                    break
                seen_counters[cname] = bname

        if not has_collisions:
            # No collisions → original code is safe, run unchanged.
            _original_init(self, directory, name, title, branch_types,
                           freesegments, counter_name, field_name,
                           initial_basket_capacity, resize_factor)
            return

        # Collisions exist → use identity counter_name so no dedup occurs.
        identity_counter = lambda counted: "n" + counted
        _original_init(self, directory, name, title, branch_types,
                       freesegments, identity_counter, field_name,
                       initial_basket_capacity, resize_factor)

        # Restore the real counter_name function.
        self._counter_name = counter_name

        # Now do correct dedup: merge per-field counters into shared ones.
        # e.g. nJet_pt, nJet_eta, nJet_mass → single nJet counter.
        canonical: dict = {}       # real_name → canonical counter dict
        to_remove: set = set()     # ids of duplicate counter dicts

        for datum in self._branch_data:
            if datum.get("kind") != "counter":
                continue
            # Per-field counter fName like "nJet_pt" → data branch "Jet_pt"
            data_branch_name = datum["fName"][1:]
            real_name = counter_name(data_branch_name)

            if real_name not in canonical:
                # First occurrence: rename to shared counter name.
                datum["fName"] = real_name
                letter = datum["fTitle"].rsplit("/", 1)[-1]
                datum["fTitle"] = f"{real_name}/{letter}"
                canonical[real_name] = datum
            else:
                # Duplicate: mark for removal.
                to_remove.add(id(datum))

        # Point all data branches to the canonical counter and fix titles.
        for datum in self._branch_data:
            if datum.get("counter") is None or datum.get("kind") == "counter":
                continue
            real_name = counter_name(datum["fName"])
            if real_name in canonical:
                datum["counter"] = canonical[real_name]
                # Fix fTitle: "Jet_pt[nJet_pt]/F" → "Jet_pt[nJet]/F"
                old = datum["fTitle"]
                bstart = old.find("[")
                bend = old.find("]")
                if bstart >= 0 and bend >= 0:
                    datum["fTitle"] = old[:bstart] + "[" + real_name + "]" + old[bend + 1:]

        # Remove duplicate counters.
        self._branch_data = [
            d for d in self._branch_data if id(d) not in to_remove
        ]

        # Reorder so each counter appears before its first data branch.
        placed: set = set()
        reordered: list = []
        for d in self._branch_data:
            if d.get("kind") == "counter":
                continue  # placed on demand below
            c = d.get("counter")
            if c is not None and id(c) not in placed:
                reordered.append(c)
                placed.add(id(c))
            reordered.append(d)
        # Append any orphan counters (shouldn't happen in practice).
        for c in canonical.values():
            if id(c) not in placed:
                reordered.append(c)
        self._branch_data = reordered

        # Rebuild _branch_lookup indices.
        self._branch_lookup = {}
        for i, d in enumerate(self._branch_data):
            key = d.get("fName") or d.get("name")
            if key:
                self._branch_lookup[key] = i

    _patched_init._shared_counter_patched = True
    _ct.Tree.__init__ = _patched_init


_patch_uproot_shared_counters()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XROOTD_TIMEOUT = 10  # seconds
os.environ.setdefault("XRD_REQUESTTIMEOUT", str(XROOTD_TIMEOUT))
os.environ.setdefault("XRD_CONNECTIONTIMEOUT", str(XROOTD_TIMEOUT))

MAX_RETRIES_PER_REDIRECTOR = DEFAULT_RETRIES_PER_REDIRECTOR
RETRY_SLEEP_SECONDS = DEFAULT_RETRY_SLEEP_SECONDS

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

# Jagged Runs branches that should be element-wise summed when collapsing.
SUMMED_RUNS_JAGGED = {"LHEScaleSumw", "LHEPdfSumw", "PSSumw"}


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
    status: str = "success"
    attempts: int = 1
    redirector: str | None = None
    failure_category: str | None = None
    failure_reason: str | None = None


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

def read_runs_tree(src_path: str) -> tuple[dict, dict] | None:
    """Read the Runs tree from a source NanoAOD file, collapsing to one entry.

    NanoAOD files that were themselves produced by hadd can contain multiple
    Runs entries (one per original production file).  Copying them all leads
    to inflated genEventSumw totals after a second hadd during skim merging.

    This function collapses:
    - ``genEventSumw``, ``genEventCount``, etc. → **summed** into one value
    - ``LHEScaleSumw``, ``LHEPdfSumw``, ``PSSumw`` → **element-wise summed**
    - All other branches → first entry kept

    Returns
    -------
    (collapsed_data, branch_types) or None
        collapsed_data : dict of awkward arrays, each length 1
        branch_types : dict of mktree-compatible type specs
    """
    try:
        with uproot.open(src_path) as fin:
            if "Runs" not in fin:
                return None
            runs_tree = fin["Runs"]

            # Build branch types from source tree (same logic as Events)
            branch_types: dict = {}
            for bname in runs_tree.keys():
                interp = runs_tree[bname].interpretation
                if hasattr(interp, "content") and hasattr(interp.content, "to_dtype"):
                    inner_dt = interp.content.to_dtype
                    if inner_dt.names is not None:
                        inner_dt = inner_dt[inner_dt.names[0]]
                    branch_types[bname] = "var * " + str(inner_dt)
                elif hasattr(interp, "to_dtype"):
                    dt = interp.to_dtype
                    if dt.names is not None:
                        dt = dt[dt.names[0]]
                    branch_types[bname] = dt
                else:
                    continue  # skip unrecognized interpretation types

            # Read with awkward library to properly handle jagged branches
            runs_ak = runs_tree.arrays(list(branch_types.keys()), library="ak")

            collapsed: dict = {}
            for bname in branch_types:
                arr = runs_ak[bname]
                if any(s in bname for s in SUMMED_RUNS_BRANCHES):
                    # Flat branches to sum (genEventSumw, genEventCount, etc.)
                    collapsed[bname] = ak.Array([ak.sum(arr)])
                elif bname in SUMMED_RUNS_JAGGED:
                    # Jagged branches to element-wise sum
                    collapsed[bname] = ak.Array([ak.sum(arr, axis=0)])
                else:
                    collapsed[bname] = arr[:1]

            return collapsed, branch_types
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

    if n_before == 0:
        del events
        gc.collect()
        if use_log:
            logger.info("  Empty file — skipping")
        return SkimResult(
            src_path=src_path, dest_path=dest_path,
            n_events_before=0, n_events_after=0, file_size_bytes=0,
            efficiency=0.0,
            status="empty_input",
        )

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

        # Detect shared NanoAOD counter branches (nJet, nMuon, etc.).
        # These are auto-generated by mktree via counter_name, so we
        # exclude them from branch_types and from the data we read/write.
        counter_branches: set[str] = set()
        kept_set = set(kept)
        for bname in kept:
            if not (bname.startswith("n") and len(bname) > 1 and bname[1].isupper()):
                continue
            coll = bname[1:]  # nJet → Jet
            interp = tree[bname].interpretation
            is_flat = hasattr(interp, "to_dtype") and not hasattr(interp, "content")
            if is_flat and any(b.startswith(coll + "_") for b in kept_set):
                counter_branches.add(bname)

        data_branches = [b for b in kept if b not in counter_branches]
        if use_log and counter_branches:
            logger.info("    %d shared counters detected (auto-generated by mktree)",
                        len(counter_branches))

        with uproot.recreate(dest_path) as fout:
            # Build TTree branch types from the source tree so the output
            # uses TTree (not RNTuple, which uproot 5.7+ writes by default
            # for dict assignment).  This preserves the original NanoAOD
            # float storage and avoids tiny precision changes at cut edges.
            branch_types = {}
            for bname in data_branches:
                branch = tree[bname]
                interp = branch.interpretation
                # AsJagged wrapping AsDtype: variable-length branch (e.g. Jet_pt)
                if hasattr(interp, "content") and hasattr(interp.content, "to_dtype"):
                    inner_dt = interp.content.to_dtype
                    if inner_dt.names is not None:
                        inner_dt = inner_dt[inner_dt.names[0]]
                    branch_types[bname] = "var * " + str(inner_dt)
                # AsDtype: flat scalar branch (e.g. run, genWeight)
                elif hasattr(interp, "to_dtype"):
                    dt = interp.to_dtype
                    if dt.names is not None:
                        dt = dt[dt.names[0]]
                    branch_types[bname] = dt
                else:
                    # Fallback: let uproot infer the type from the first chunk
                    branch_types[bname] = interp

            # Use shared NanoAOD counter names: Jet_pt → nJet (not nJet_pt).
            # This triggers the monkey-patched dedup in _patch_uproot_shared_counters.
            fout.mktree(
                "Events", branch_types,
                counter_name=lambda counted: "n" + counted.split("_")[0],
            )

            chunk_num = 0
            for start in range(0, total_entries, STEP):
                stop = min(start + STEP, total_entries)
                chunk_mask = mask_np[start:stop]
                if not np.any(chunk_mask):
                    continue
                chunk = tree.arrays(
                    data_branches, entry_start=start, entry_stop=stop, library="ak"
                )
                chunk = chunk[chunk_mask]
                chunk_dict = {f: chunk[f] for f in chunk.fields}
                fout["Events"].extend(chunk_dict)
                del chunk, chunk_dict, chunk_mask
                gc.collect()
                chunk_num += 1
                if use_log:
                    logger.info("    Chunk %d written  [peak RSS: %.0f MB]",
                                chunk_num, _mem_mb())

            # Runs tree — also written as TTree via mktree
            runs_result = read_runs_tree(src_path)
            if runs_result is not None:
                runs_data, runs_types = runs_result
                fout.mktree("Runs", runs_types)
                fout["Runs"].extend(runs_data)
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
        status="success",
    )


def _classify_skim_error(exc: Exception) -> tuple[str, bool]:
    """Classify skim errors and indicate whether retries are worthwhile.

    Returns
    -------
    (category, retryable)
        category: one of
          - "empty_input"
          - "corrupt_file"
          - "schema_error"
          - "network_error"
          - "unknown_error"
    """
    msg = str(exc).lower()

    # Empty-input patterns often show up as missing branch fields.
    empty_patterns = [
        "not in fields",
        "keyerror: 'events'",
        "keyerror: \"events\"",
    ]
    if any(p in msg for p in empty_patterns):
        return "empty_input", False

    # Corruption / unreadable object storage payloads.
    corrupt_patterns = [
        "received 0 bytes",
        "corrupt",
        "zlib",
        "decompression",
        "badseek",
        "basket",
    ]
    if any(p in msg for p in corrupt_patterns):
        return "corrupt_file", False

    # Structural incompatibility with expected NanoAOD schema.
    schema_patterns = [
        "missing branch",
        "cannot interpret",
        "cannot cast",
        "nanoevents",
        "schema",
    ]
    if any(p in msg for p in schema_patterns):
        return "schema_error", False

    # Transient transport / service failures are usually retryable.
    network_patterns = [
        "timeout",
        "timed out",
        "operation expired",
        "socket",
        "xrootd",
        "connection reset",
        "connection refused",
        "temporary failure",
        "service unavailable",
        "no servers are available",
    ]
    if any(p in msg for p in network_patterns):
        return "network_error", True

    # Keep unknown failures retryable to avoid false-negative hard failures.
    return "unknown_error", True


def _retry_sleep_seconds(_attempt: int) -> float:
    """Return fixed sleep between retries.

    The *attempt* index starts at 1.
    """
    return RETRY_SLEEP_SECONDS


def skim_single_file(src_path: str, dest_path: str, progress=None) -> SkimResult:
    """Skim one NanoAOD file with XRootD retry and redirector fallback.

    For XRootD URLs: retries up to MAX_RETRIES_PER_REDIRECTOR times per
    redirector with a fixed RETRY_SLEEP_SECONDS delay between retries,
    then falls back to the next redirector in REDIRECTORS.
    For local paths: calls the implementation directly with no retries.

    Schema errors are treated as non-retryable and are raised immediately.
    Empty-input and corrupt-read signatures trigger redirector failover; a
    file is only declared ``empty_input``/``corrupt_file`` if that signature
    is seen on every redirector.

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
        result = _skim_impl(src_path, dest_path, progress=progress)
        result.attempts = 1
        result.redirector = None
        return result

    total_attempts = 0
    corrupt_failures = {}
    empty_failures = {}
    for redir_idx, redirector in enumerate(REDIRECTORS):
        url = f"{redirector}{lfn}"
        for attempt in range(1, MAX_RETRIES_PER_REDIRECTOR + 1):
            total_attempts += 1
            try:
                logger.info(
                    "Attempt %d/%d via %s",
                    attempt, MAX_RETRIES_PER_REDIRECTOR, redirector,
                )
                result = _skim_impl(url, dest_path, progress=progress)
                result.attempts = total_attempts
                result.redirector = redirector
                return result
            except Exception as exc:
                category, retryable = _classify_skim_error(exc)
                logger.warning(
                    "Attempt %d/%d failed (%s, category=%s): %s",
                    attempt, MAX_RETRIES_PER_REDIRECTOR, redirector, category, exc,
                )
                # Empty/corrupt signatures can be redirector-specific.
                # Try other redirectors before declaring a global failure.
                if category in {"corrupt_file", "empty_input"}:
                    if category == "corrupt_file":
                        corrupt_failures[redirector] = str(exc)
                    else:
                        empty_failures[redirector] = str(exc)
                    logger.info(
                        "%s signature via %s; trying next redirector",
                        category, redirector,
                    )
                    break
                if not retryable:
                    raise RuntimeError(
                        f"Non-retryable [{category}] error for {lfn}: {exc}"
                    ) from exc
                if attempt < MAX_RETRIES_PER_REDIRECTOR:
                    sleep_s = _retry_sleep_seconds(attempt)
                    logger.info("Retrying in %.2f s", sleep_s)
                    time.sleep(sleep_s)
        if redir_idx < len(REDIRECTORS) - 1:
            logger.info("Switching to redirector: %s", REDIRECTORS[redir_idx + 1])

    if len(empty_failures) == len(REDIRECTORS):
        details = "; ".join(
            f"{redir}: {msg}" for redir, msg in empty_failures.items()
        )
        raise RuntimeError(
            f"Non-retryable [empty_input] error for {lfn} across all "
            f"{len(REDIRECTORS)} redirectors: {details}"
        )

    if len(corrupt_failures) == len(REDIRECTORS):
        details = "; ".join(
            f"{redir}: {msg}" for redir, msg in corrupt_failures.items()
        )
        raise RuntimeError(
            f"Non-retryable [corrupt_file] error for {lfn} across all "
            f"{len(REDIRECTORS)} redirectors: {details}"
        )

    raise RuntimeError(
        f"All retries exhausted for {lfn} across {len(REDIRECTORS)} redirectors"
    )
