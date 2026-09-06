"""Drell-Yan Z-pT reweighting ported from the Run 2 HNWRAnalyzer.

The Run 2 WR analysis (SKFlatAnalyzer ``HNWR/Run2Legacy_v4``,
``HNWRAnalyzer::executeEvent``) corrects the LO madgraphMLM HT-binned DY
sample with two multiplicative factors evaluated at the generator-level Z
pT:

* a QCD NLO/LO ratio (with QCD scale, PDF, and alpha_S variations), and
* an electroweak correction (with three uncertainty components).

Both are stored as one JSON payload per era (``dy_zpt_reweight`` in
``config.yaml``; produced by ``scripts/convert_run2_dy_reweight.py``) and
evaluated here with NumPy.  The generator-level Z follows the Run 2
``GenFinderForDY``: the first two hard-process final-state electrons or
muons (or decayed hard-process taus), required to be same flavour and
opposite sign.

The nominal weight is applied to every dataset whose ``physics_group`` is
``DYJets``; every other dataset receives the identity so that the
variations exist (as nominal copies) for coverage bookkeeping.
"""

from __future__ import annotations

import json
import logging

import awkward as ak
import numpy as np

from wrcoffea.analysis_config import DY_ZPT_REWEIGHT_JSONS

logger = logging.getLogger(__name__)

#: ``physics_group`` of the datasets the reweighting applies to.
DY_PROCESS_GROUP = "DYJets"

#: Systematic source names, in the order their labels are produced.  The
#: EW components keep the Run 2 numbering (``ZPtRwEW1/2/3``).
QCD_SOURCES: tuple[str, ...] = ("DYZptQcdScale", "DYZptQcdPdf", "DYZptQcdAlphaS")
EW_SOURCES: tuple[str, ...] = ("DYZptEw1", "DYZptEw2", "DYZptEw3")
SOURCE_NAMES: tuple[str, ...] = QCD_SOURCES + EW_SOURCES

# GenPart.statusFlags bit for ``fromHardProcess``.
_FROM_HARD_PROCESS_BIT = 8

_PAYLOAD_CACHE: dict[str, dict] = {}
_WARN_ONCE: set[str] = set()


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def load_payload(path: str) -> dict:
    """Load (and cache per process) a converted reweighting payload."""
    cached = _PAYLOAD_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    qcd, ew = raw["qcd"], raw["ew"]
    payload = {
        "qcd": {k: np.asarray(v, dtype=np.float64) for k, v in qcd.items()
                if isinstance(v, list)},
        "ew": {k: np.asarray(v, dtype=np.float64) for k, v in ew.items()
               if isinstance(v, list)},
        "qcd_pt_max": float(qcd["pt_max"]),
        "ew_pt_min": float(ew["pt_min"]),
        "ew_pt_max": float(ew["pt_max"]),
    }
    for block in ("qcd", "ew"):
        n_bins = len(payload[block]["edges"]) - 1
        for key, arr in payload[block].items():
            if key != "edges" and len(arr) != n_bins:
                raise ValueError(f"DY reweight payload '{path}': {block}/{key} has {len(arr)} bins, expected {n_bins}")
    _PAYLOAD_CACHE[path] = payload
    return payload


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """ROOT ``FindBin`` semantics on the in-range axis: bin i covers [e_i, e_{i+1})."""
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


# ---------------------------------------------------------------------------
# Generator-level Z
# ---------------------------------------------------------------------------

def find_gen_z(events):
    """Return ``(pt, mass, found)`` of the generator-level Z per event.

    Mirrors ``GenFinderForDY::Find``: candidates are electrons/muons with
    ``status == 1`` or taus with ``status == 2``, all flagged
    ``fromHardProcess``; the first two candidates (in GenPart order) form
    the Z when they are same flavour and opposite sign.  Events without such
    a pair have ``found == False`` and zeros for pt and mass.
    """
    gp = events.GenPart
    pid = abs(gp.pdgId)
    from_hard = (gp.statusFlags & (1 << _FROM_HARD_PROCESS_BIT)) != 0
    is_light = ((pid == 11) | (pid == 13)) & (gp.status == 1) & from_hard
    is_tau = (pid == 15) & (gp.status == 2) & from_hard
    cand = gp[is_light | is_tau]
    first_two = ak.pad_none(cand[:, :2], 2, axis=1)
    l1, l2 = first_two[:, 0], first_two[:, 1]

    has_two = ak.num(cand, axis=1) >= 2
    sfos = ak.fill_none((l1.pdgId + l2.pdgId) == 0, False) & has_two
    z = l1 + l2
    pt = np.asarray(ak.fill_none(ak.where(sfos, z.pt, 0.0), 0.0), dtype=np.float64)
    mass = np.asarray(ak.fill_none(ak.where(sfos, z.mass, 0.0), 0.0), dtype=np.float64)
    return pt, mass, np.asarray(sfos, dtype=bool)


# ---------------------------------------------------------------------------
# Weight evaluation
# ---------------------------------------------------------------------------

def identity_weights(n: int) -> dict:
    """Identity weight and variations for *n* events (non-DY or unconfigured era)."""
    ones = np.ones(n, dtype=np.float64)
    return {
        "nominal": ones,
        "variations": {name: (ones, ones) for name in SOURCE_NAMES},
        "gen_z_pt": None,
        "applied": False,
    }


def evaluate(gen_z_pt: np.ndarray, found: np.ndarray, payload: dict) -> dict:
    """Evaluate nominal and varied weights at *gen_z_pt* (NumPy arrays)."""
    qcd, ew = payload["qcd"], payload["ew"]

    pt_q = np.minimum(gen_z_pt, payload["qcd_pt_max"])
    iq = _bin_index(pt_q, qcd["edges"])
    q_c = qcd["central"][iq]
    q_pdf = qcd["pdf_err"][iq]
    q_scale_up, q_scale_down = qcd["scale_up"][iq], qcd["scale_down"][iq]
    q_as_up, q_as_down = qcd["alphas_up"][iq], qcd["alphas_down"][iq]

    # HNWRAnalyzer: ptZ < 30 -> 31 ; ptZ >= 6500 -> 6499.
    pt_e = np.where(gen_z_pt < 30.0, payload["ew_pt_min"], gen_z_pt)
    pt_e = np.where(pt_e >= 6500.0, payload["ew_pt_max"], pt_e)
    ie = _bin_index(pt_e, ew["edges"])
    e_c = ew["central"][ie]
    e1, e2, e3 = ew["e1_err"][ie], ew["e2_err"][ie], ew["e3_err"][ie]

    def _final(w):
        return np.where(found, w, 1.0)

    nominal = _final(q_c * e_c)
    variations = {
        "DYZptQcdScale": (_final(q_scale_up * e_c), _final(q_scale_down * e_c)),
        "DYZptQcdPdf": (_final((q_c + q_pdf) * e_c), _final((q_c - q_pdf) * e_c)),
        "DYZptQcdAlphaS": (_final(q_as_up * e_c), _final(q_as_down * e_c)),
        "DYZptEw1": (_final(q_c * (e_c + e1)), _final(q_c * (e_c - e1))),
        "DYZptEw2": (_final(q_c * (e_c + e2)), _final(q_c * (e_c - e2))),
        "DYZptEw3": (_final(q_c * (e_c + e3)), _final(q_c * (e_c - e3))),
    }
    return {"nominal": nominal, "variations": variations, "gen_z_pt": gen_z_pt, "applied": True}


def dy_zpt_weight(events, era: str, *, is_dy: bool) -> dict:
    """Per-event DY Z-pT weight for *era*.

    Returns ``{"nominal": array, "variations": {name: (up, down)}, ...}``.
    The identity is returned when the era has no payload or the dataset is
    not DY.  A DY dataset without a ``GenPart`` collection is an error: the
    correction cannot be evaluated and must not be skipped silently.
    """
    n = len(events)
    path = DY_ZPT_REWEIGHT_JSONS.get(era)
    if path is None or not is_dy:
        return identity_weights(n)
    if n == 0:
        return identity_weights(0)
    if getattr(events, "GenPart", None) is None:
        raise RuntimeError(
            f"DY Z-pT reweighting is configured for era '{era}' but the events have no "
            "GenPart collection; the correction cannot be evaluated."
        )
    payload = load_payload(path)
    pt, _mass, found = find_gen_z(events)
    n_missing = int(np.count_nonzero(~found))
    if n_missing:
        key = f"dy_zpt_no_gen_z::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "DY Z-pT reweight: %d of %d events in this chunk have no hard-process SFOS "
                "lepton pair; they keep weight 1 (reported once per worker).", n_missing, n,
            )
    return evaluate(pt, found, payload)
