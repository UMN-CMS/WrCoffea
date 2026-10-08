"""Skim event selection on raw NanoAOD branch arrays.

The selection is deliberately looser than the analysis cuts
(``wrcoffea/config.yaml``) so the skims stay usable when thresholds move or
when systematic variations shift object momenta:

- at least two leptons (electrons or muons) with |eta| < 2.5 and pT > 45 GeV,
  the leading one above 52 GeV;
- and at least two AK4 jets (pT > 32, |eta| < 2.5) or one AK8 jet (pT > 180,
  |eta| < 2.5).

Muons are ranked on the larger of the PF pT and the TuneP pT
(``Muon_pt * Muon_tunepRelPt``): the analysis selects high-pT muons on TuneP,
which can exceed the PF value.  No trigger, MET filter or golden-JSON
requirement is applied here; the analyzer does that.

This module works on the plain branch arrays uproot returns (no NanoEvents),
so it runs identically inside the skim worker and in tests.  An independent
re-implementation lives in :mod:`wrcoffea.skim.recount` and is used to
cross-check every file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import awkward as ak
import numpy as np

SKIM_CUTS: dict[str, float] = {
    "lepton_pt_min": 45.0,
    "lepton_eta_max": 2.5,
    "lead_lepton_pt_min": 52.0,
    "sublead_lepton_pt_min": 45.0,
    "ak4_pt_min": 32.0,
    "ak4_eta_max": 2.5,
    "ak8_pt_min": 180.0,
    "ak8_eta_max": 2.5,
}

# Branches the selection reads.  ``Muon_tunepRelPt`` is optional (absent in
# very old NanoAOD); every other one must exist.
REQUIRED_SELECTION_BRANCHES: tuple[str, ...] = (
    "Electron_pt", "Electron_eta",
    "Muon_pt", "Muon_eta",
    "Jet_pt", "Jet_eta",
    "FatJet_pt", "FatJet_eta",
)
OPTIONAL_SELECTION_BRANCHES: tuple[str, ...] = ("Muon_tunepRelPt",)
SELECTION_BRANCHES: tuple[str, ...] = REQUIRED_SELECTION_BRANCHES + OPTIONAL_SELECTION_BRANCHES

# Event identifiers, read alongside the selection for bookkeeping.
EVENT_ID_BRANCHES: tuple[str, ...] = ("run", "luminosityBlock", "event")


def selection_branches(available: Iterable[str]) -> list[str]:
    """Return the selection branches present in *available*.

    Raises ``KeyError`` naming every required branch that is missing.
    """
    avail = set(available)
    missing = [b for b in REQUIRED_SELECTION_BRANCHES if b not in avail]
    if missing:
        raise KeyError(f"input lacks selection branches: {missing}")
    return [b for b in SELECTION_BRANCHES if b in avail]


def get_branch(arrays, name: str, default=None):
    """Field lookup that works for both dicts and awkward record arrays."""
    if isinstance(arrays, ak.Array):
        return arrays[name] if name in arrays.fields else default
    return arrays.get(name, default)


def muon_skim_pt(pt: ak.Array, tunep_rel_pt: ak.Array | None) -> ak.Array:
    """Per-muon pT used by the skim: max(PF pT, TuneP pT)."""
    if tunep_rel_pt is None:
        return pt
    tunep = pt * tunep_rel_pt
    return ak.where(tunep > pt, tunep, pt)


def skim_mask(arrays, cuts: Mapping[str, float] = SKIM_CUTS) -> np.ndarray:
    """Boolean per-event mask of the skim selection.

    *arrays* is a dict (or awkward record array, as ``TTree.arrays`` returns)
    of the jagged per-object arrays listed in :data:`SELECTION_BRANCHES`.
    """
    c = cuts
    e_pt, e_eta = arrays["Electron_pt"], arrays["Electron_eta"]
    m_pt = muon_skim_pt(arrays["Muon_pt"], get_branch(arrays, "Muon_tunepRelPt"))
    m_eta = arrays["Muon_eta"]

    e_ok = (abs(e_eta) < c["lepton_eta_max"]) & (e_pt > c["lepton_pt_min"])
    m_ok = (abs(m_eta) < c["lepton_eta_max"]) & (m_pt > c["lepton_pt_min"])

    lep_pt = ak.concatenate([e_pt[e_ok], m_pt[m_ok]], axis=1)
    lep_pt = ak.sort(lep_pt, axis=1, ascending=False)
    padded = ak.pad_none(lep_pt, 2, axis=1, clip=True)
    lead = ak.fill_none(padded[:, 0], 0.0)
    sublead = ak.fill_none(padded[:, 1], 0.0)
    lep_mask = (lead > c["lead_lepton_pt_min"]) & (sublead > c["sublead_lepton_pt_min"])

    ak4_ok = (arrays["Jet_pt"] > c["ak4_pt_min"]) & (abs(arrays["Jet_eta"]) < c["ak4_eta_max"])
    ak8_ok = (arrays["FatJet_pt"] > c["ak8_pt_min"]) & (abs(arrays["FatJet_eta"]) < c["ak8_eta_max"])
    had_mask = (ak.sum(ak4_ok, axis=1) >= 2) | (ak.sum(ak8_ok, axis=1) >= 1)

    return np.asarray(ak.to_numpy(lep_mask & had_mask), dtype=bool)


def describe_cuts(cuts: Mapping[str, float] = SKIM_CUTS) -> str:
    c = cuts
    return (
        f">=2 leptons (e or mu) with pT > {c['lepton_pt_min']:g} GeV and |eta| < {c['lepton_eta_max']:g}, "
        f"leading lepton pT > {c['lead_lepton_pt_min']:g} GeV, subleading > {c['sublead_lepton_pt_min']:g} GeV; "
        f"AND (>=2 AK4 jets with pT > {c['ak4_pt_min']:g}, |eta| < {c['ak4_eta_max']:g} "
        f"OR >=1 AK8 jet with pT > {c['ak8_pt_min']:g}, |eta| < {c['ak8_eta_max']:g}). "
        "Muon pT = max(PF, TuneP)."
    )
