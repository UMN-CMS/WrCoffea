"""Independent re-implementation of the skim selection, used for validation.

:func:`wrcoffea.skim.selection.skim_mask` is the production selection.  This
module computes the same decision with different primitives: every jagged
branch is flattened to a NumPy array plus per-event counts, and the per-event
quantities are built with ``np.bincount`` / ``np.maximum.at`` instead of
awkward's jagged operations.  The two implementations are compared event by
event on every file the campaign processes, so a bug in either one shows up
as a mismatch rather than as silently lost events.

It also provides the event-identifier digest written to the part sidecars.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import awkward as ak
import numpy as np

from wrcoffea.skim.selection import SKIM_CUTS, get_branch


def flatten(arr: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(counts, flat)`` NumPy views of a jagged array."""
    counts = np.asarray(ak.to_numpy(ak.num(arr, axis=1)), dtype=np.int64)
    flat = np.asarray(ak.to_numpy(ak.flatten(arr, axis=1)))
    return counts, flat


def _event_index(counts: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(counts)), counts)


def count_per_event(counts: np.ndarray, flat_bool: np.ndarray) -> np.ndarray:
    """Number of objects per event for which *flat_bool* is True."""
    n = len(counts)
    if flat_bool.size == 0:
        return np.zeros(n, dtype=np.int64)
    return np.bincount(_event_index(counts), weights=flat_bool.astype(np.float64), minlength=n).astype(np.int64)


def max_per_event(counts: np.ndarray, flat_vals: np.ndarray, flat_bool: np.ndarray) -> np.ndarray:
    """Largest *flat_vals* per event among objects where *flat_bool* holds (``-inf`` if none)."""
    out = np.full(len(counts), -np.inf, dtype=np.float64)
    if flat_vals.size == 0:
        return out
    idx = _event_index(counts)[flat_bool]
    np.maximum.at(out, idx, flat_vals[flat_bool].astype(np.float64))
    return out


def recount_mask(arrays, cuts: Mapping[str, float] = SKIM_CUTS) -> np.ndarray:
    """Per-event skim decision, computed independently of ``selection.skim_mask``.

    Formulation: an event passes when it has at least two leptons above the
    lepton threshold inside |eta| < eta_max, the largest of their pTs exceeds
    the leading-lepton threshold, and it has at least two AK4 jets or one AK8
    jet above the jet thresholds.  (With both lepton thresholds at the same
    value this is identical to "leading > 52 and subleading > 45" after
    sorting, which is how the production selection phrases it.)
    """
    c = cuts
    e_n, e_pt = flatten(arrays["Electron_pt"])
    _, e_eta = flatten(arrays["Electron_eta"])
    m_n, m_pt_pf = flatten(arrays["Muon_pt"])
    _, m_eta = flatten(arrays["Muon_eta"])
    rel_arr = get_branch(arrays, "Muon_tunepRelPt")
    if rel_arr is not None:
        _, rel = flatten(rel_arr)
        m_pt = np.maximum(m_pt_pf, m_pt_pf * rel)
    else:
        m_pt = m_pt_pf

    e_ok = (np.abs(e_eta) < c["lepton_eta_max"]) & (e_pt > c["lepton_pt_min"])
    m_ok = (np.abs(m_eta) < c["lepton_eta_max"]) & (m_pt > c["lepton_pt_min"])
    n_lep = count_per_event(e_n, e_ok) + count_per_event(m_n, m_ok)
    lead = np.maximum(max_per_event(e_n, e_pt, e_ok), max_per_event(m_n, m_pt, m_ok))
    lep_pass = (n_lep >= 2) & (lead > c["lead_lepton_pt_min"])
    # Guard for the general case where the subleading threshold differs from
    # the per-lepton one: require at least two leptons above the subleading cut.
    if c["sublead_lepton_pt_min"] != c["lepton_pt_min"]:
        n_sub = (count_per_event(e_n, e_ok & (e_pt > c["sublead_lepton_pt_min"]))
                 + count_per_event(m_n, m_ok & (m_pt > c["sublead_lepton_pt_min"])))
        lep_pass &= n_sub >= 2

    j_n, j_pt = flatten(arrays["Jet_pt"])
    _, j_eta = flatten(arrays["Jet_eta"])
    f_n, f_pt = flatten(arrays["FatJet_pt"])
    _, f_eta = flatten(arrays["FatJet_eta"])
    n_ak4 = count_per_event(j_n, (j_pt > c["ak4_pt_min"]) & (np.abs(j_eta) < c["ak4_eta_max"]))
    n_ak8 = count_per_event(f_n, (f_pt > c["ak8_pt_min"]) & (np.abs(f_eta) < c["ak8_eta_max"]))

    return lep_pass & ((n_ak4 >= 2) | (n_ak8 >= 1))


def event_ids(run, lumi, event) -> np.ndarray:
    """Structured array of (run, lumi, event) in entry order."""
    out = np.empty(len(run), dtype=[("run", "<u4"), ("lumi", "<u4"), ("event", "<u8")])
    out["run"] = np.asarray(run, dtype=np.uint32)
    out["lumi"] = np.asarray(lumi, dtype=np.uint32)
    out["event"] = np.asarray(event, dtype=np.uint64)
    return out


class IdDigest:
    """Incremental order-sensitive digest of event identifiers.

    Feed chunks of (run, lumi, event) in entry order; ``hexdigest()`` is the
    value stored in the sidecars.  Two files with the same events in the same
    order give the same digest.
    """

    def __init__(self):
        self._h = hashlib.blake2b(digest_size=16)
        self.count = 0

    def update(self, run, lumi, event) -> None:
        ids = event_ids(run, lumi, event)
        self._h.update(ids.tobytes())
        self.count += len(ids)

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def ids_digest(run, lumi, event) -> str:
    d = IdDigest()
    d.update(run, lumi, event)
    return d.hexdigest()
