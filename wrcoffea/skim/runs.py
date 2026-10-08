"""Collapsing and merging of the NanoAOD ``Runs`` and ``LuminosityBlocks`` trees.

MC ``Runs`` trees carry one entry per original production file with the
generator-weight sums.  A skim part made from many inputs collapses them to a
single entry:

- ``genEventCount``, ``genEventSumw``, ``genEventSumw2`` (and the old
  trailing-underscore spellings) are **summed**;
- ``LHEScaleSumw``, ``LHEPdfSumw``, ``PSSumw``, ``LHEReweightingSumw`` are
  stored by NanoAOD as *ratios* to ``genEventSumw``
  (``sum(genWeight * w_i) / genEventSumw``), so they are combined as the
  ``genEventSumw``-weighted element-wise mean, which keeps them ratios to the
  collapsed ``genEventSumw``;
- every other branch keeps its first value.

Data ``Runs`` trees only list run numbers; they are collapsed to one entry per
distinct run.  ``LuminosityBlocks`` (data) is the list of processed
lumisections and is merged with duplicate (run, lumi) rows removed, so the
luminosity actually processed can be recomputed from the skims.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import awkward as ak
import numpy as np

logger = logging.getLogger(__name__)

SUMMED_RUNS_BRANCHES = frozenset({
    "genEventCount", "genEventCount_",
    "genEventSumw", "genEventSumw_",
    "genEventSumw2", "genEventSumw2_",
})
RATIO_RUNS_BRANCHES = frozenset({
    "LHEScaleSumw", "LHEPdfSumw", "PSSumw", "LHEReweightingSumw",
})


def _sumw(arrays: Mapping[str, ak.Array]) -> np.ndarray | None:
    for name in ("genEventSumw", "genEventSumw_"):
        if name in arrays:
            return np.asarray(ak.to_numpy(arrays[name]), dtype=np.float64)
    return None


def _weighted_mean_jagged(name: str, arr: ak.Array, sumw: np.ndarray | None) -> np.ndarray:
    rows = [np.asarray(r, dtype=np.float64) for r in ak.to_list(arr)]
    if not rows:
        return np.zeros(0, dtype=np.float64)
    lengths = {len(r) for r in rows}
    if len(lengths) == 2 and 0 in lengths and sumw is not None and len(sumw) == len(rows):
        n = max(lengths)
        empty = [i for i, r in enumerate(rows) if len(r) == 0]
        if any(sumw[i] != 0 for i in empty):
            # Inputs that processed events but carry no such weights at all (a production quirk:
            # e.g. one 2022 ZZZ file without LHE scale/PDF weights): the mean is taken over the
            # inputs that have them; their events also lack the per-event weights.
            logger.warning("Runs branch %r is empty for %d of %d input(s) with events; averaging over the "
                           "other inputs", name, len(empty), len(rows))
        keep = [i for i, r in enumerate(rows) if len(r) == n]
        rows = [rows[i] for i in keep]
        sumw = sumw[keep]
        lengths = {n}
    if len(lengths) != 1:
        raise ValueError(
            f"Runs branch {name!r} has entries of different lengths {sorted(lengths)}; "
            "cannot collapse them into one entry"
        )
    mat = np.vstack(rows)
    if sumw is None or len(sumw) != len(rows) or not np.any(sumw):
        return mat.mean(axis=0)
    w = sumw[:, None]
    return (mat * w).sum(axis=0) / w.sum()


def is_mc_runs(arrays: Mapping[str, ak.Array]) -> bool:
    return any(k in SUMMED_RUNS_BRANCHES for k in arrays)


def collapse_runs(arrays: Mapping[str, ak.Array]) -> dict[str, ak.Array]:
    """Collapse a multi-entry Runs table (see module docstring)."""
    if not arrays:
        return {}
    if is_mc_runs(arrays):
        sumw = _sumw(arrays)
        # Columns that are neither summed nor averaged (counters such as nLHEScaleSumw) come from
        # the first row of an input that processed events: an empty input's row has empty vectors.
        first = 0
        if sumw is not None and len(sumw) and np.any(sumw != 0):
            ok = sumw != 0
            for name in RATIO_RUNS_BRANCHES:
                if name in arrays:
                    lens = np.asarray(ak.num(arrays[name], axis=1))
                    ok &= lens == lens.max()
            first = int(np.flatnonzero(ok)[0]) if np.any(ok) else int(np.flatnonzero(sumw != 0)[0])
        out: dict[str, ak.Array] = {}
        for name, arr in arrays.items():
            if name in SUMMED_RUNS_BRANCHES:
                vals = np.asarray(ak.to_numpy(arr))
                total = vals.sum(dtype=np.float64) if vals.dtype.kind == "f" else vals.sum()
                out[name] = ak.Array(np.array([total], dtype=vals.dtype))
            elif name in RATIO_RUNS_BRANCHES:
                inner = _weighted_mean_jagged(name, arr, sumw)
                dtype = ak.to_numpy(ak.flatten(arr, axis=None)).dtype if len(arr) else np.float32
                out[name] = ak.Array([inner.astype(dtype)])
            else:
                out[name] = arr[first:first + 1]
        return out
    # Data: one entry per distinct run, other columns from the first occurrence.
    if "run" not in arrays:
        return {name: arr[:1] for name, arr in arrays.items()}
    runs = np.asarray(ak.to_numpy(arrays["run"]))
    _, first_idx = np.unique(runs, return_index=True)
    first_idx = np.sort(first_idx)
    return {name: arr[first_idx] for name, arr in arrays.items()}


def concat_runs(tables: Sequence[Mapping[str, ak.Array]]) -> dict[str, ak.Array]:
    """Concatenate Runs tables that share the same branch set."""
    tables = [t for t in tables if t]
    if not tables:
        return {}
    names = list(tables[0].keys())
    for t in tables[1:]:
        if set(t.keys()) != set(names):
            raise ValueError(f"Runs tables have different branches: {sorted(set(t) ^ set(names))}")
    return {n: ak.concatenate([t[n] for t in tables]) for n in names}


def merge_lumiblocks(tables: Sequence[Mapping[str, ak.Array]]) -> dict[str, ak.Array]:
    """Merge LuminosityBlocks tables, dropping duplicate (run, luminosityBlock) rows."""
    tables = [t for t in tables if t]
    if not tables:
        return {}
    cat = concat_runs(tables)
    if "run" not in cat or "luminosityBlock" not in cat:
        return cat
    run = np.asarray(ak.to_numpy(cat["run"]), dtype=np.int64)
    lumi = np.asarray(ak.to_numpy(cat["luminosityBlock"]), dtype=np.int64)
    key = run * (1 << 32) + lumi
    _, first_idx = np.unique(key, return_index=True)
    order = first_idx[np.argsort(key[first_idx], kind="stable")]
    return {n: arr[order] for n, arr in cat.items()}


def lumi_pairs(table: Mapping[str, ak.Array]) -> set[tuple[int, int]]:
    """The set of (run, lumi) pairs in a LuminosityBlocks table."""
    if not table or "run" not in table:
        return set()
    run = ak.to_numpy(table["run"]).tolist()
    lumi = ak.to_numpy(table["luminosityBlock"]).tolist()
    return set(zip(run, lumi))
