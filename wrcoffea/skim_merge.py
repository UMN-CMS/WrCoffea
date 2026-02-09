"""Post-skim merging: extract tarballs, hadd ROOT files, validate.

Replaces the old hadd_dataset.sh, hadd_dataset2.sh, and the merge logic
in unzip_files.sh with testable, importable Python.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import uproot
from uproot.behaviors.TTree import TTree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Outcome of a merge_dataset() call."""
    dataset: str
    input_files: int
    output_files: int
    total_events_in: int
    total_events_out: int
    total_sumw_in: float
    total_sumw_out: float
    events_match: bool
    sumw_match: bool
    output_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tarball extraction
# ---------------------------------------------------------------------------

def extract_tarballs(directory: str | Path, dataset_name: str) -> int:
    """Extract ``{dataset_name}_*.tar.gz`` tarballs in *directory*, then delete them.

    Returns the number of tarballs extracted.
    """
    directory = Path(directory)
    pattern = f"{dataset_name}_*.tar.gz"
    tarballs = sorted(directory.glob(pattern))
    if not tarballs:
        logger.warning("No tarballs matching '%s' in %s", pattern, directory)
        return 0

    for tb in tarballs:
        logger.info("Extracting %s", tb.name)
        with tarfile.open(tb, "r:gz") as tf:
            tf.extractall(path=directory)
        tb.unlink()

    logger.info("Extracted %d tarballs", len(tarballs))
    return len(tarballs)


# ---------------------------------------------------------------------------
# ROOT file inspection
# ---------------------------------------------------------------------------

def _get_tree(path: str, tree_name: str = "Events"):
    """Open *path* and return the TTree (or None)."""
    with uproot.open(path) as f:
        if tree_name in f:
            obj = f[tree_name]
            if isinstance(obj, TTree):
                return obj
        # Fallback: first TTree
        for k, cls in f.classnames().items():
            if cls.startswith("TTree"):
                return f[k]
    return None


def count_events(filepath: str, tree_name: str = "Events") -> int:
    """Return the number of entries in the Events tree of *filepath*."""
    with uproot.open(filepath) as f:
        if tree_name in f:
            return int(f[tree_name].num_entries)
        for k, cls in f.classnames().items():
            if cls.startswith("TTree"):
                return int(f[k].num_entries)
    return 0


def read_runs_sumw(filepath: str) -> float:
    """Sum ``genEventSumw`` from the Runs tree (returns 0.0 if absent)."""
    try:
        with uproot.open(filepath) as f:
            if "Runs" not in f:
                return 0.0
            runs = f["Runs"]
            # Try genEventSumw_ first (uproot naming), then genEventSumw
            for branch_name in ("genEventSumw_", "genEventSumw"):
                if branch_name in runs.keys():
                    arr = runs[branch_name].array(library="np")
                    return float(arr.sum())
        return 0.0
    except Exception as e:
        logger.warning("Failed to read Runs sumw from %s: %s", filepath, e)
        return 0.0


# ---------------------------------------------------------------------------
# HLT-aware grouping
# ---------------------------------------------------------------------------

def get_hlt_signature(filepath: str, tree_name: str = "Events") -> tuple[str, ...]:
    """Return a sorted tuple of HLT_* branch names from a ROOT file."""
    with uproot.open(filepath) as f:
        if tree_name not in f:
            return ()
        tree = f[tree_name]
        return tuple(sorted(k for k in tree.keys() if k.startswith("HLT_")))


def group_by_hlt(
    root_files: list[str],
    tree_name: str = "Events",
) -> dict[tuple[str, ...], list[tuple[str, int]]]:
    """Group ROOT files by their HLT branch signature.

    Returns
    -------
    dict mapping HLT-signature-tuple -> list of (filepath, n_events).
    """
    groups: dict[tuple[str, ...], list[tuple[str, int]]] = defaultdict(list)
    for fpath in root_files:
        try:
            sig = get_hlt_signature(fpath, tree_name)
            nev = count_events(fpath, tree_name)
            groups[sig].append((fpath, nev))
        except Exception as e:
            logger.warning("Skipping %s (inspection failed): %s", fpath, e)
    return dict(groups)


# ---------------------------------------------------------------------------
# hadd wrapper
# ---------------------------------------------------------------------------

# hadd warning that indicates incompatible TTree schemas
_BAD_WARN_RE = re.compile(r"Warning in <TTree::CopyEntries>.*not present in the import TTree")


def merge_files(
    input_files: list[str],
    output_path: str,
    *,
    check_warnings: bool = True,
) -> bool:
    """Run ``hadd -f`` to merge *input_files* into *output_path*.

    If *check_warnings* is True, the ``CopyEntries`` warning is treated as
    fatal and the output file is removed.

    Returns True on success, False on failure.
    """
    cmd = ["hadd", "-f", output_path] + input_files
    logger.info("Merging %d files -> %s", len(input_files), Path(output_path).name)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("hadd failed (exit %d):\n%s", result.returncode, result.stderr[-500:])
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    if check_warnings and _BAD_WARN_RE.search(result.stdout + result.stderr):
        logger.error(
            "hadd produced a fatal CopyEntries warning — removing %s",
            output_path,
        )
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Wrote %s (%.1f MB)", Path(output_path).name, size_mb)
    return True


# ---------------------------------------------------------------------------
# Full merge pipeline
# ---------------------------------------------------------------------------

def merge_dataset(
    input_dir: str | Path,
    dataset_name: str,
    *,
    max_events: int = 1_000_000,
    hlt_aware: bool = True,
    output_dir: str | Path | None = None,
    file_pattern: str = "*_skim.root",
) -> MergeResult:
    """Full merge pipeline for one dataset.

    1. Discover input ROOT files matching *file_pattern* in *input_dir*.
    2. Optionally group by HLT branch signature.
    3. Chunk each group by *max_events*.
    4. hadd each chunk.
    5. Verify Events counts and Runs sumw.

    Returns a MergeResult summarizing the operation.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover input files
    input_files = sorted(str(p) for p in input_dir.glob(file_pattern))
    if not input_files:
        logger.warning("No files matching '%s' in %s", file_pattern, input_dir)
        return MergeResult(
            dataset=dataset_name, input_files=0, output_files=0,
            total_events_in=0, total_events_out=0,
            total_sumw_in=0.0, total_sumw_out=0.0,
            events_match=True, sumw_match=True,
        )

    # Tally input totals
    total_events_in = 0
    total_sumw_in = 0.0
    file_events: list[tuple[str, int]] = []

    for fpath in input_files:
        nev = count_events(fpath)
        sw = read_runs_sumw(fpath)
        total_events_in += nev
        total_sumw_in += sw
        file_events.append((fpath, nev))

    logger.info(
        "Input: %d files, %d events, sumw=%.2f",
        len(input_files), total_events_in, total_sumw_in,
    )

    # Group files
    if hlt_aware:
        groups = group_by_hlt(input_files)
        logger.info("Found %d distinct HLT signature group(s)", len(groups))
    else:
        # One big group
        groups = {(): file_events}

    # Merge within each group, chunking by max_events
    output_paths: list[str] = []
    part = 1

    for sig, members in groups.items():
        members = sorted(members, key=lambda x: x[0])
        chunk_files: list[str] = []
        chunk_events = 0

        for fpath, nev in members:
            if chunk_files and (chunk_events + nev > max_events):
                outpath = str(output_dir / f"{dataset_name}_part{part}.root")
                if merge_files(chunk_files, outpath):
                    output_paths.append(outpath)
                part += 1
                chunk_files = []
                chunk_events = 0

            chunk_files.append(fpath)
            chunk_events += nev

        # Flush remaining
        if chunk_files:
            outpath = str(output_dir / f"{dataset_name}_part{part}.root")
            if merge_files(chunk_files, outpath):
                output_paths.append(outpath)
            part += 1

    # Verify outputs
    total_events_out = 0
    total_sumw_out = 0.0
    for op in output_paths:
        total_events_out += count_events(op)
        total_sumw_out += read_runs_sumw(op)

    events_match = total_events_in == total_events_out
    sumw_match = (
        abs(total_sumw_in - total_sumw_out) < 1.0
        if total_sumw_in > 0
        else total_sumw_out == 0.0
    )

    if not events_match:
        logger.error(
            "Event count mismatch! in=%d out=%d (diff=%d)",
            total_events_in, total_events_out,
            total_events_out - total_events_in,
        )
    if not sumw_match:
        logger.error(
            "Runs sumw mismatch! in=%.2f out=%.2f (ratio=%.4f)",
            total_sumw_in, total_sumw_out,
            total_sumw_out / total_sumw_in if total_sumw_in else float("inf"),
        )

    return MergeResult(
        dataset=dataset_name,
        input_files=len(input_files),
        output_files=len(output_paths),
        total_events_in=total_events_in,
        total_events_out=total_events_out,
        total_sumw_in=total_sumw_in,
        total_sumw_out=total_sumw_out,
        events_match=events_match,
        sumw_match=sumw_match,
        output_paths=output_paths,
    )


def validate_merge(
    merged_files: list[str],
    *,
    expected_events: int | None = None,
    expected_sumw: float | None = None,
) -> MergeResult:
    """Standalone validation of already-merged files.

    Checks Events count and Runs sumw against expected values.
    """
    total_events = 0
    total_sumw = 0.0
    for fpath in merged_files:
        total_events += count_events(fpath)
        total_sumw += read_runs_sumw(fpath)

    events_match = (
        expected_events is None or total_events == expected_events
    )
    sumw_match = (
        expected_sumw is None or abs(total_sumw - expected_sumw) < 1.0
    )

    return MergeResult(
        dataset="validation",
        input_files=0,
        output_files=len(merged_files),
        total_events_in=expected_events or 0,
        total_events_out=total_events,
        total_sumw_in=expected_sumw or 0.0,
        total_sumw_out=total_sumw,
        events_match=events_match,
        sumw_match=sumw_match,
        output_paths=merged_files,
    )
