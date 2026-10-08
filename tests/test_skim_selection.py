"""Skim selection: production vs independent recount vs a brute-force reference."""

import numpy as np
import awkward as ak
import pytest
import uproot

from tests.nano_synth import write_nano_file, make_events
from wrcoffea.skim.selection import skim_mask, SKIM_CUTS, selection_branches, EVENT_ID_BRANCHES, muon_skim_pt
from wrcoffea.skim.recount import recount_mask, ids_digest, IdDigest


def reference_mask(ev: dict, cuts=SKIM_CUTS) -> np.ndarray:
    """Third, per-event pure-Python implementation."""
    c = cuts
    out = np.zeros(len(ev["run"]), dtype=bool)
    e_pt, e_eta = ak.to_list(ev["Electron_pt"]), ak.to_list(ev["Electron_eta"])
    m_pt, m_eta, m_rel = ak.to_list(ev["Muon_pt"]), ak.to_list(ev["Muon_eta"]), ak.to_list(ev["Muon_tunepRelPt"])
    j_pt, j_eta = ak.to_list(ev["Jet_pt"]), ak.to_list(ev["Jet_eta"])
    f_pt, f_eta = ak.to_list(ev["FatJet_pt"]), ak.to_list(ev["FatJet_eta"])
    for i in range(len(out)):
        leps = [p for p, e in zip(e_pt[i], e_eta[i]) if abs(e) < c["lepton_eta_max"] and p > c["lepton_pt_min"]]
        for p, e, r in zip(m_pt[i], m_eta[i], m_rel[i]):
            pt = max(p, p * r)
            if abs(e) < c["lepton_eta_max"] and pt > c["lepton_pt_min"]:
                leps.append(pt)
        leps.sort(reverse=True)
        lep_ok = len(leps) >= 2 and leps[0] > c["lead_lepton_pt_min"] and leps[1] > c["sublead_lepton_pt_min"]
        n4 = sum(1 for p, e in zip(j_pt[i], j_eta[i]) if p > c["ak4_pt_min"] and abs(e) < c["ak4_eta_max"])
        n8 = sum(1 for p, e in zip(f_pt[i], f_eta[i]) if p > c["ak8_pt_min"] and abs(e) < c["ak8_eta_max"])
        out[i] = lep_ok and (n4 >= 2 or n8 >= 1)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_three_implementations_agree(seed):
    ev = make_events(3000, seed)
    m1 = skim_mask(ev)
    m2 = recount_mask(ev)
    m3 = reference_mask(ev)
    assert m1.dtype == bool and len(m1) == 3000
    assert np.array_equal(m1, m2)
    assert np.array_equal(m1, m3)
    assert 0 < m1.mean() < 1


def test_agreement_with_shifted_cuts():
    ev = make_events(3000, 7)
    c = dict(SKIM_CUTS, lepton_pt_min=40.0, sublead_lepton_pt_min=48.0, lead_lepton_pt_min=70.0, ak4_pt_min=50.0)
    assert np.array_equal(skim_mask(ev, c), recount_mask(ev, c))
    assert np.array_equal(skim_mask(ev, c), reference_mask(ev, c))


def test_thresholds_are_strict():
    def one(e_pt, mu_pt, rel, j_pt, f_pt=()):
        ev = {
            "Electron_pt": ak.Array([list(e_pt)]), "Electron_eta": ak.Array([[0.1] * len(e_pt)]),
            "Muon_pt": ak.Array([list(mu_pt)]), "Muon_eta": ak.Array([[0.2] * len(mu_pt)]),
            "Muon_tunepRelPt": ak.Array([list(rel)]),
            "Jet_pt": ak.Array([list(j_pt)]), "Jet_eta": ak.Array([[0.3] * len(j_pt)]),
            "FatJet_pt": ak.Array([list(f_pt)]), "FatJet_eta": ak.Array([[0.4] * len(f_pt)]),
        }
        a, b = skim_mask(ev)[0], recount_mask(ev)[0]
        assert a == b
        return bool(a)
    assert one([60.0, 50.0], [], [], [40.0, 40.0])            # passes
    assert not one([52.0, 50.0], [], [], [40.0, 40.0])        # lead exactly at threshold fails
    assert not one([60.0, 45.0], [], [], [40.0, 40.0])        # sublead exactly at threshold fails
    assert not one([60.0, 50.0], [], [], [32.0, 40.0])        # jet exactly at threshold fails
    assert one([60.0, 50.0], [], [], [20.0], [181.0])         # AK8 path
    assert not one([60.0, 50.0], [], [], [20.0], [180.0])
    assert one([60.0], [40.0], [1.2], [40.0, 40.0])           # TuneP promotes the muon (48 > 45)
    assert not one([60.0], [40.0], [1.1], [40.0, 40.0])       # 44 < 45


def test_eta_boundary_and_missing_tunep():
    ev = make_events(500, 11)
    ev2 = dict(ev)
    ev2.pop("Muon_tunepRelPt")
    assert np.array_equal(skim_mask(ev2), recount_mask(ev2))
    # removing TuneP can only lose muons, never gain events
    assert not np.any(skim_mask(ev2) & ~skim_mask(ev))


def test_selection_branches_and_file_roundtrip(tmp_path):
    p = tmp_path / "nano.root"
    ev = write_nano_file(p, n=1500, seed=5)
    with uproot.open(p) as f:
        t = f["Events"]
        names = selection_branches(t.keys())
        assert "Muon_tunepRelPt" in names and "Electron_pt" in names
        arr = t.arrays(names + list(EVENT_ID_BRANCHES))
        assert np.array_equal(skim_mask(arr), reference_mask(ev))
        assert np.array_equal(recount_mask(arr), reference_mask(ev))
    with pytest.raises(KeyError):
        selection_branches(["Electron_pt", "Muon_pt"])


def test_id_digest_is_order_sensitive_and_incremental():
    run = np.array([1, 1, 1], np.uint32)
    lumi = np.array([5, 5, 6], np.uint32)
    event = np.array([10, 11, 12], np.uint64)
    d = IdDigest()
    d.update(run[:1], lumi[:1], event[:1])
    d.update(run[1:], lumi[1:], event[1:])
    assert d.hexdigest() == ids_digest(run, lumi, event)
    assert d.count == 3
    assert ids_digest(run, lumi, event[::-1]) != ids_digest(run, lumi, event)
