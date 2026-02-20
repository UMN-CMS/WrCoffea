"""Post-skim merging: extract tarballs, merge ROOT files, validate.

Replaces the old hadd_dataset.sh, hadd_dataset2.sh, and the merge logic
in unzip_files.sh with testable, importable Python.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import uproot
from uproot.behaviors.TTree import TTree

logger = logging.getLogger(__name__)


def _ensure_uproot_patch():
    """Activate the uproot shared-counter monkey-patch if not already applied.

    The patch (defined in wrcoffea.skimmer) fixes uproot's _cascadetree.Tree
    when counter_name maps multiple jagged branches to the same counter
    (e.g. Jet_pt and Jet_eta both -> nJet).  We import lazily to avoid
    pulling in coffea/dask when only merge functionality is needed.
    """
    import uproot.writing._cascadetree as _ct
    if getattr(_ct.Tree.__init__, '_shared_counter_patched', False):
        return
    from wrcoffea.skimmer import _patch_uproot_shared_counters
    _patch_uproot_shared_counters()


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-friendly dict."""
        return asdict(self)


def save_merge_summary(result: MergeResult, output_dir: str | Path) -> Path:
    """Write ``merge_summary.json`` next to the merged part files.

    Returns the path to the written file.
    """
    output_dir = Path(output_dir)
    summary_path = output_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    logger.info("Wrote %s", summary_path)
    return summary_path


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
            for member in tf.getmembers():
                # Strip leading directory so files land directly in *directory*
                member.name = Path(member.name).name
                if member.name and not member.isdir():
                    tf.extract(member, path=directory)
        tb.unlink()

    logger.info("Extracted %d tarballs", len(tarballs))
    return len(tarballs)


def extract_single_tarball(tarball: Path, directory: Path) -> list[str]:
    """Extract one tarball into *directory*, delete the tarball.

    Returns a list of paths to extracted ROOT files.
    """
    directory = Path(directory)
    extracted = []
    logger.info("Extracting %s", tarball.name)
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            member.name = Path(member.name).name
            if member.name and not member.isdir():
                tf.extract(member, path=directory)
                fpath = directory / member.name
                if fpath.suffix == ".root":
                    extracted.append(str(fpath))
    tarball.unlink()
    return extracted


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
            keys = runs.keys()
            for branch_name in ("genEventSumw_", "genEventSumw"):
                if branch_name not in keys:
                    continue
                try:
                    arr = runs[branch_name].array(library="np")
                    return float(arr.sum())
                except Exception:
                    # Variable-length branches can fail with library="np";
                    # fall back to awkward and flatten.
                    import awkward as ak
                    arr = runs[branch_name].array()
                    return float(ak.sum(ak.flatten(arr, axis=None)))
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
    total = len(root_files)
    for i, fpath in enumerate(root_files):
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == total:
            logger.info("HLT grouping: %d/%d files", i + 1, total)
        try:
            sig = get_hlt_signature(fpath, tree_name)
            nev = count_events(fpath, tree_name)
            groups[sig].append((fpath, nev))
        except Exception as e:
            logger.warning("Skipping %s (inspection failed): %s", fpath, e)
    return dict(groups)


# ---------------------------------------------------------------------------
# Merge via uproot (handles both TTree and RNTuple inputs)
# ---------------------------------------------------------------------------

# Branches that should be summed (not copied) when merging the Runs tree.
_SUMMED_RUNS_BRANCHES = ("genEventSumw", "genEventCount", "nEvents", "LHEWeight")

# Jagged Runs branches that should be element-wise summed when collapsing.
_SUMMED_RUNS_JAGGED = {"LHEScaleSumw", "LHEPdfSumw", "PSSumw"}

_CHUNK_SIZE = 100_000  # entries per read chunk during merge


def _native_endian(ak_array):
    """Convert an awkward array's underlying buffers to native byte order.

    RNTuple and older ROOT files may store data in big-endian format,
    which uproot's TTree writer cannot handle.  This function repacks
    the array through ``ak.from_buffers`` with native-endian buffers.
    """
    import awkward as ak
    import numpy as np

    form, length, container = ak.to_buffers(ak_array)
    native_container = {}
    for key, buf in container.items():
        if isinstance(buf, np.ndarray) and buf.dtype.byteorder not in ("=", "|", "<"):
            native_container[key] = buf.astype(buf.dtype.newbyteorder("="))
        else:
            native_container[key] = buf
    return ak.from_buffers(form, length, native_container)


def _is_counter_branch(name: str, all_names: set[str]) -> bool:
    """Return True if *name* looks like a NanoAOD counter (e.g. nMuon, nJet).

    Uproot auto-generates these from jagged arrays, so including them
    explicitly causes a dtype mismatch during extend().
    """
    if not name.startswith("n") or len(name) < 2 or not name[1].isupper():
        return False
    # NanoAOD convention: nX is the counter for X_* branches or single X branch.
    collection = name[1:]
    return (collection in all_names
            or any(n.startswith(collection + "_") for n in all_names))


def merge_files(
    input_files: list[str],
    output_path: str,
    *,
    check_warnings: bool = True,
) -> bool:
    """Merge *input_files* into *output_path* using uproot.

    Reads each input file in chunks and writes a TTree-format output,
    handling both TTree and RNTuple input files transparently.

    Returns True on success, False on failure.
    """
    import awkward as ak

    _ensure_uproot_patch()

    logger.info("Merging %d files -> %s", len(input_files), Path(output_path).name)
    try:
        with uproot.recreate(output_path) as fout:
            tree_created = False
            for fpath in input_files:
                with uproot.open(fpath) as fin:
                    if "Events" not in fin:
                        logger.warning("No Events tree in %s — skipping", fpath)
                        continue
                    tree = fin["Events"]
                    all_names = set(tree.keys())
                    counter_branches = {
                        bname for bname in tree.keys()
                        if _is_counter_branch(bname, all_names)
                    }
                    data_branches = [b for b in tree.keys()
                                     if b not in counter_branches]

                    # Create TTree on first file encountered.  mktree +
                    # counter_name auto-generates shared counters (nJet,
                    # nMuon, etc.) so we exclude them from branch_types.
                    if not tree_created:
                        branch_types = {}
                        for bname in data_branches:
                            interp = tree[bname].interpretation
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
                                branch_types[bname] = interp
                        fout.mktree(
                            "Events", branch_types,
                            counter_name=lambda counted: "n" + counted.split("_")[0],
                        )
                        tree_created = True

                    n_entries = tree.num_entries
                    for start in range(0, n_entries, _CHUNK_SIZE):
                        stop = min(start + _CHUNK_SIZE, n_entries)
                        arrays = tree.arrays(
                            data_branches,
                            entry_start=start,
                            entry_stop=stop,
                            library="ak",
                        )
                        data_dict = {
                            f: _native_endian(arrays[f])
                            for f in arrays.fields
                        }
                        fout["Events"].extend(data_dict)
                        del arrays, data_dict

            if not tree_created:
                logger.error("No events found in any input file — aborting %s", Path(output_path).name)
                return False

            # Merge Runs trees: sum genEventSumw/genEventCount, keep first for others
            _merge_runs(input_files, fout)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info("Wrote %s (%.1f MB)", Path(output_path).name, size_mb)
        return True

    except Exception:
        logger.exception("Merge failed for %s", Path(output_path).name)
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def _merge_runs(input_files: list[str], fout) -> None:
    """Merge Runs trees from *input_files*, writing a single collapsed entry to *fout*.

    Uses mktree + extend to produce a TTree (not RNTuple).  Jagged branches
    (LHEScaleSumw, LHEPdfSumw, PSSumw) are element-wise summed; scalar
    weight branches are summed; all others keep the first value.
    """
    import awkward as ak

    all_data: dict[str, list] = {}
    branch_types: dict | None = None

    for fpath in input_files:
        try:
            with uproot.open(fpath) as fin:
                if "Runs" not in fin:
                    continue
                runs_tree = fin["Runs"]

                # Build branch_types from first file encountered
                if branch_types is None:
                    branch_types = {}
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
                            continue

                runs_ak = runs_tree.arrays(list(branch_types.keys()), library="ak")
                for bname in branch_types:
                    arr = runs_ak[bname]
                    if bname not in all_data:
                        all_data[bname] = []
                    all_data[bname].append(arr)
        except Exception as e:
            logger.warning("Failed to read Runs from %s: %s", fpath, e)

    if branch_types is None or not all_data:
        return

    collapsed: dict = {}
    for bname in branch_types:
        if bname not in all_data:
            continue
        combined = ak.concatenate(all_data[bname])
        if any(s in bname for s in _SUMMED_RUNS_BRANCHES):
            collapsed[bname] = ak.Array([ak.sum(combined)])
        elif bname in _SUMMED_RUNS_JAGGED:
            collapsed[bname] = ak.Array([ak.sum(combined, axis=0)])
        else:
            collapsed[bname] = combined[:1]

    fout.mktree("Runs", branch_types)
    fout["Runs"].extend(collapsed)


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

    result = MergeResult(
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

    save_merge_summary(result, output_dir)
    return result


def merge_dataset_incremental(
    input_dir: str | Path,
    dataset_name: str,
    *,
    max_events: int = 1_000_000,
    hlt_aware: bool = True,
    output_dir: str | Path | None = None,
    file_pattern: str = "*_skim.root",
) -> MergeResult:
    """Merge pipeline: extract all tarballs, group by HLT, chunk, merge.

    1. Extract all ``{dataset_name}_*.tar.gz`` tarballs in *input_dir*.
    2. Discover all ROOT files matching *file_pattern* (includes
       pre-existing files from a previous interrupted run).
    3. Group files by HLT branch signature.
    4. Chunk each group by *max_events*.
    5. Merge each chunk and validate.
    6. Delete original skim files after all merges succeed.
    7. Write ``merge_summary.json`` next to the merged part files.

    Returns a MergeResult summarizing the operation.
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract all tarballs
    tarball_pattern = f"{dataset_name}_*.tar.gz"
    tarballs = sorted(input_dir.glob(tarball_pattern))
    existing_skims = sorted(str(p) for p in input_dir.glob(file_pattern))

    if not tarballs and not existing_skims:
        logger.warning(
            "No tarballs matching '%s' and no files matching '%s' in %s",
            tarball_pattern, file_pattern, input_dir,
        )
        return MergeResult(
            dataset=dataset_name, input_files=0, output_files=0,
            total_events_in=0, total_events_out=0,
            total_sumw_in=0.0, total_sumw_out=0.0,
            events_match=True, sumw_match=True,
        )

    if tarballs:
        logger.info("Extracting %d tarballs", len(tarballs))
        extract_tarballs(input_dir, dataset_name)

    # 2. Discover all skim files (pre-existing + freshly extracted)
    input_files = sorted(str(p) for p in input_dir.glob(file_pattern))
    if not input_files:
        logger.warning("No skim files found after extraction in %s", input_dir)
        return MergeResult(
            dataset=dataset_name, input_files=0, output_files=0,
            total_events_in=0, total_events_out=0,
            total_sumw_in=0.0, total_sumw_out=0.0,
            events_match=True, sumw_match=True,
        )

    # 3. Tally input totals
    total_events_in = 0
    total_sumw_in = 0.0
    for fpath in input_files:
        total_events_in += count_events(fpath)
        total_sumw_in += read_runs_sumw(fpath)

    logger.info(
        "Input: %d files, %d events, sumw=%.2f",
        len(input_files), total_events_in, total_sumw_in,
    )

    # 4. Group by HLT signature
    if hlt_aware:
        groups = group_by_hlt(input_files)
        logger.info("Found %d distinct HLT signature group(s)", len(groups))
    else:
        groups = {(): [(f, count_events(f)) for f in input_files]}

    # 5. Chunk within each group and merge
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

        # Flush remaining files in this HLT group
        if chunk_files:
            outpath = str(output_dir / f"{dataset_name}_part{part}.root")
            if merge_files(chunk_files, outpath):
                output_paths.append(outpath)
            part += 1

    # 6. Validate
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

    # 7. Clean up skim files if all merges succeeded
    if events_match and sumw_match:
        for f in input_files:
            try:
                os.remove(f)
            except OSError:
                pass
        logger.info("Cleaned up %d skim files", len(input_files))
    else:
        logger.error(
            "Merge validation failed — keeping %d skim files for retry",
            len(input_files),
        )

    logger.info(
        "Merge complete: %d input files -> %d output files, %d events",
        len(input_files), len(output_paths), total_events_out,
    )

    result = MergeResult(
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

    save_merge_summary(result, output_dir)
    return result


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
