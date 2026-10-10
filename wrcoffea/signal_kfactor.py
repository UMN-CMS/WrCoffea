"""Signal NLO/LO k-factor reweighting ported from the Run 2 HNWRAnalyzer.

AN-19-083 Appendix A: the QCD NLO/LO correction to WR production depends on
the hard scale, taken as the generator-level mass of the primary lepton plus
N system, ``m_lN``.  Run 2 reweighted every signal event by

    k(m_lN) / <k>

where ``k`` is a histogram in ``m_lN`` and ``<k>`` is the average of ``k``
over the generated sample of that mass point (per lepton channel).  The
signal cross sections used for normalization (the AN tables, copied into
our signal configs) are already LO x <k>, so the weight is a pure shape
correction that moves weight towards on-shell events.

The gen-level objects follow ``GenFinderForHNWRSignal``: N is the first
GenPart with |pdgId| 9900012/9900014/9900016; the primary lepton is the
first electron or muon whose mother is the WR (|pdgId| 9900024 or 34) when
an on-shell WR with a quark mother exists in the record, otherwise the
first electron or muon.  Payloads are produced by
``scripts/convert_run2_signal_kfactor.py``.
"""

from __future__ import annotations

import json
import logging
import re

import awkward as ak
import numpy as np

from wrcoffea.analysis_config import SIGNAL_KFACTOR_JSONS

logger = logging.getLogger(__name__)

#: ``physics_group`` of the datasets the k-factor applies to.
SIGNAL_PROCESS_GROUP = "Signal"

WR_PDGIDS = (9900024, 34)
N_PDGIDS = (9900012, 9900014, 9900016)
_N_CHANNEL = {9900012: "ee", 9900014: "mm"}

_MASS_POINT_RE = re.compile(r"M?WR(\d+)_M?N(\d+)")

_PAYLOAD_CACHE: dict[str, dict] = {}
_WARN_ONCE: set[str] = set()


def mass_point_key(name: str | None) -> str | None:
    """``WRtoNLtoLLJJ_MWR1000_MN100`` / ``WR1000_N100`` -> ``WR1000_N100``."""
    if not name:
        return None
    m = _MASS_POINT_RE.search(str(name))
    return f"WR{m.group(1)}_N{m.group(2)}" if m else None


def load_payload(path: str) -> dict:
    cached = _PAYLOAD_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    kf = raw["kfactor"]
    payload = {
        "edges": np.asarray(kf["edges"], dtype=np.float64),
        "values": np.asarray(kf["values"], dtype=np.float64),
        "mln_min": float(kf["mln_min"]),
        "averages": raw["averages"],
        # k(m_lN) of each mass point from its own NLO/LO ratio (docs/signal_kfactor.md); a point without one uses
        # the shared curve above
        "point_curves": {},
    }
    if len(payload["values"]) != len(payload["edges"]) - 1:
        raise ValueError(f"Signal k-factor payload '{path}': bin count mismatch")
    for name, c in (raw.get("point_curves") or {}).items():
        curve = {"edges": np.asarray(c["edges"], dtype=np.float64), "values": np.asarray(c["values"], dtype=np.float64),
                 "mln_min": float(c["mln_min"])}
        if len(curve["values"]) != len(curve["edges"]) - 1:
            raise ValueError(f"Signal k-factor payload '{path}': bin count mismatch in the curve of {name}")
        payload["point_curves"][name] = curve
    _PAYLOAD_CACHE[path] = payload
    return payload


def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, x, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)


def find_gen_ln(events):
    """Return ``(m_ln, channel_pdgid, found)`` per event.

    ``channel_pdgid`` is |pdgId| of the N (9900012 for ee, 9900014 for mumu,
    0 when absent).  ``found`` is False when no N or no primary lepton is
    present; those events get ``m_ln = 0``.
    """
    gp = events.GenPart
    pid = abs(gp.pdgId)
    mother_idx = gp.genPartIdxMother
    has_mother = mother_idx >= 0
    safe_idx = ak.where(has_mother, mother_idx, 0)
    mother_pid = ak.where(has_mother, abs(gp.pdgId[safe_idx]), 0)

    # On-shell primary WR: WR with a quark mother and a finite mass.
    is_wr = (pid == WR_PDGIDS[0]) | (pid == WR_PDGIDS[1])
    on_shell_wr = ak.any(is_wr & has_mother & (mother_pid <= 6) & (gp.mass == gp.mass), axis=1)

    is_n = (pid == N_PDGIDS[0]) | (pid == N_PDGIDS[1]) | (pid == N_PDGIDS[2])
    n_cand = ak.firsts(gp[is_n])

    is_lep = (pid == 11) | (pid == 13)
    mother_is_wr = (mother_pid == WR_PDGIDS[0]) | (mother_pid == WR_PDGIDS[1])
    lep_mask = (is_lep & mother_is_wr & on_shell_wr) | (is_lep & ~on_shell_wr)
    pri_lep = ak.firsts(gp[lep_mask])

    found = np.asarray(~ak.is_none(n_cand) & ~ak.is_none(pri_lep), dtype=bool)
    m_ln = np.asarray(ak.fill_none((pri_lep + n_cand).mass, 0.0), dtype=np.float64)
    m_ln = np.where(found, m_ln, 0.0)
    channel = np.asarray(ak.fill_none(abs(n_cand.pdgId), 0), dtype=np.int64)
    return m_ln, channel, found


def evaluate(m_ln: np.ndarray, channel: np.ndarray, found: np.ndarray, payload: dict, mass_point: str) -> np.ndarray:
    """``k(m_ln) / <k>`` per event; 1 where the gen objects were not found.  ``k`` comes from the mass point's own
    curve when the payload has one (``point_curves``), else from the shared curve."""
    try:
        avg = payload["averages"][mass_point]
    except KeyError as e:
        raise KeyError(
            f"Signal k-factor payload has no average for mass point '{mass_point}'"
        ) from e
    curve = payload.get("point_curves", {}).get(mass_point, payload)     # the point's own curve or the shared one
    m = np.where(m_ln < 200.0, curve["mln_min"], m_ln)
    k = curve["values"][_bin_index(m, curve["edges"])]
    # a point produced in one lepton channel only (e.g. the 2024 electron-only WR3200 points) has one average
    k_ee, k_mm = avg.get("ee", avg.get("mm")), avg.get("mm", avg.get("ee"))
    if k_ee is None:
        raise KeyError(f"Signal k-factor payload has no average value for mass point '{mass_point}'")
    avg_per_event = np.where(channel == 9900014, k_mm, k_ee)
    return np.where(found, k / avg_per_event, 1.0)


def signal_kfactor_weight(events, era: str, *, mass_point: str | None, is_signal: bool) -> dict:
    """Per-event signal k-factor weight for *era* and *mass_point*.

    Returns ``{"nominal": array, "applied": bool}``.  The identity is
    returned when the era has no payload or the dataset is not signal.  A
    signal dataset without ``GenPart`` or without a resolvable mass point is
    an error.
    """
    n = len(events)
    ones = np.ones(n, dtype=np.float64)
    path = SIGNAL_KFACTOR_JSONS.get(era)
    if path is None or not is_signal:
        return {"nominal": ones, "applied": False}
    if mass_point is None:
        raise ValueError(
            f"Signal k-factor is configured for era '{era}' but the mass point could not be "
            "determined from the dataset name or --mass."
        )
    if n == 0:
        return {"nominal": ones, "applied": True}
    if getattr(events, "GenPart", None) is None:
        raise RuntimeError(
            f"Signal k-factor is configured for era '{era}' but the events have no GenPart collection."
        )
    payload = load_payload(path)
    m_ln, channel, found = find_gen_ln(events)
    n_missing = int(np.count_nonzero(~found))
    if n_missing:
        key = f"signal_kfactor_missing::{era}::{mass_point}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Signal k-factor: %d of %d events in this chunk have no gen N or primary lepton; "
                "they keep weight 1 (reported once per worker).", n_missing, n,
            )
    return {"nominal": evaluate(m_ln, channel, found, payload, mass_point), "applied": True}
