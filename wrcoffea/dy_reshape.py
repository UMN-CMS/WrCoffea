"""Data-driven DY reshape vs jet pT, following the Run 2 HNWRAnalyzer.

Run 2 (AN-19-083, ``HNWRAnalyzer::GetDYReshape`` with the ``JetPt`` flag)
corrected the residual data/MC shape difference of the DY sample with a
ratio measured in the DY control regions:

    ratio = (data - non-DY MC) / DY MC

binned in the leading AK4 jet pT (resolved) or the DY-candidate AK8 jet pT
(boosted), with the ee and mumu control regions combined.  The ratio is then
applied to every DY event in every region as a function of the same
variable.  Two variations exist: the statistical uncertainty of the ratio
and the difference between the ee-only and mumu-only ratios.

The DY sample must carry the Z-pT reweighting when the ratio is derived and
when it is applied; the analyzer enforces that coupling.  Payloads come from
``scripts/derive_dy_reshape.py``.
"""

from __future__ import annotations

import json

import numpy as np

from wrcoffea.analysis_config import DY_RESHAPE_JSONS
from wrcoffea.dy_reweight import DY_PROCESS_GROUP  # noqa: F401  (re-exported for callers)

#: Region kinds and the jet-pT variable each one is binned in.
KINDS: dict[str, str] = {"resolved": "pt_leading_jet", "boosted": "pt_leading_AK8Jets"}

#: Systematic source names.
STAT_SOURCE = "DYReshapeStat"
CHANNEL_SOURCE = "DYReshapeChannel"
SOURCE_NAMES: tuple[str, ...] = (STAT_SOURCE, CHANNEL_SOURCE)

_PAYLOAD_CACHE: dict[str, dict] = {}


def load_payload(path: str) -> dict:
    cached = _PAYLOAD_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    payload = {}
    for kind in KINDS:
        block = raw[kind]
        arrays = {k: np.asarray(block[k], dtype=np.float64)
                  for k in ("edges", "ratio", "stat_err", "ee_ratio", "mumu_ratio", "channel_err")}
        n_bins = len(arrays["edges"]) - 1
        for k, arr in arrays.items():
            if k != "edges" and len(arr) != n_bins:
                raise ValueError(f"DY reshape payload '{path}': {kind}/{k} has {len(arr)} bins, expected {n_bins}")
        arrays["variable"] = block.get("variable", KINDS[kind])
        payload[kind] = arrays
    payload["requires_dy_zpt_reweight"] = bool(raw.get("requires_dy_zpt_reweight", True))
    _PAYLOAD_CACHE[path] = payload
    return payload


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin i covers [e_i, e_{i+1}); values beyond the last edge use the last bin."""
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def identity_weights(n: int) -> dict:
    ones = np.ones(n, dtype=np.float64)
    return {"nominal": ones, "variations": {name: (ones, ones) for name in SOURCE_NAMES}, "applied": False}


def evaluate(jet_pt: np.ndarray, block: dict) -> dict:
    idx = _bin_index(jet_pt, block["edges"])
    ratio = block["ratio"][idx]
    stat = block["stat_err"][idx]
    chan = block["channel_err"][idx]
    return {
        "nominal": ratio,
        "variations": {
            STAT_SOURCE: (ratio + stat, ratio - stat),
            CHANNEL_SOURCE: (ratio + chan, ratio - chan),
        },
        "applied": True,
    }


def dy_reshape_weight(jet_pt, era: str, kind: str, *, is_dy: bool) -> dict | None:
    """Per-event reshape weight for *era* and region *kind* (``resolved``/``boosted``).

    Returns ``None`` when the era has no payload (nothing to apply and no
    variation labels to produce), the identity for non-DY datasets, and the
    evaluated ratio with its variations for DY.
    """
    path = DY_RESHAPE_JSONS.get(era)
    if path is None:
        return None
    if kind not in KINDS:
        raise ValueError(f"Unknown DY reshape region kind '{kind}'")
    pt = np.asarray(jet_pt, dtype=np.float64)
    if not is_dy:
        return identity_weights(len(pt))
    return evaluate(pt, load_payload(path)[kind])
