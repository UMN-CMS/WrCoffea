"""Unit tests for the WrAnalysis processor core logic.

These tests use lightweight mock awkward arrays to exercise selection methods
without requiring real NanoAOD files or correctionlib payloads.
"""

import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import vector

from wrcoffea.analysis_config import CUTS
from wrcoffea.analyzer import WrAnalysis

ak.behavior.update(vector.behavior)


# ---------------------------------------------------------------------------
# Helpers to build mock objects
# ---------------------------------------------------------------------------

def _make_collection(pts, etas, phis=None, masses=None, extra_fields=None):
    """Build a single-event jagged record array (like NanoAOD collections).

    Appends a sentinel element to guarantee proper type resolution in awkward
    (empty numpy arrays lose their dtype), then trims it off.
    """
    n = len(pts)
    if phis is None:
        phis = [0.5 * i for i in range(n)]
    if masses is None:
        masses = [0.105] * n
    fields = {
        "pt": [np.array(pts, dtype=np.float64)],
        "eta": [np.array(etas, dtype=np.float64)],
        "phi": [np.array(phis, dtype=np.float64)],
        "mass": [np.array(masses, dtype=np.float64)],
    }
    if extra_fields:
        for k, v in extra_fields.items():
            if isinstance(v, np.ndarray):
                fields[k] = [v]
            else:
                fields[k] = [np.array(v)]
    # Append a sentinel to each field to guarantee type resolution, then trim
    for k in fields:
        inner = fields[k][0]
        fields[k] = [np.concatenate([inner, np.zeros(1, dtype=inner.dtype)])]
    result = ak.zip(fields)
    return result[:, :n]


def _mock_events(n_muons=2, n_electrons=0, n_jets=2, n_fatjets=0,
                 muon_pts=None, electron_pts=None, jet_pts=None):
    """Build a minimal single-event mock events object."""
    mu_pts = muon_pts or [100.0] * n_muons
    el_pts = electron_pts or [100.0] * n_electrons
    j_pts = jet_pts or [100.0] * n_jets

    muon = _make_collection(
        mu_pts, [1.0] * n_muons,
        extra_fields={
            "highPtId": np.full(n_muons, 2, dtype=np.int32),
            "tkRelIso": np.full(n_muons, 0.05, dtype=np.float64),
            "cutBased_HEEP": np.zeros(n_muons, dtype=np.bool_),
            "cutBased": np.zeros(n_muons, dtype=np.int32),
        },
    )

    electron = _make_collection(
        el_pts, [1.0] * n_electrons,
        phis=[2.0 + 0.5 * i for i in range(n_electrons)],
        masses=[0.000511] * n_electrons,
        extra_fields={
            "highPtId": np.zeros(n_electrons, dtype=np.int32),
            "tkRelIso": np.zeros(n_electrons, dtype=np.float64),
            "cutBased_HEEP": np.ones(n_electrons, dtype=np.bool_),
            "cutBased": np.full(n_electrons, 4, dtype=np.int32),
        },
    )

    jet = _make_collection(
        j_pts, [1.0] * n_jets,
        phis=[3.0 + 0.5 * i for i in range(n_jets)],
        masses=[10.0] * n_jets,
        extra_fields={
            "isTightLeptonVeto": np.ones(n_jets, dtype=np.bool_),
        },
    )

    fatjet = _make_collection(
        [300.0] * n_fatjets, [1.0] * n_fatjets,
        phis=[0.0] * n_fatjets,
        masses=[80.0] * n_fatjets,
        extra_fields={
            "msoftdrop": np.full(n_fatjets, 80.0, dtype=np.float64),
        },
    )

    class MockEvents:
        pass

    ev = MockEvents()
    ev.Muon = muon
    ev.Electron = electron
    ev.Jet = jet
    ev.FatJet = fatjet
    return ev


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWrAnalysisInit:
    def test_default_region(self):
        proc = WrAnalysis(mass_point=None)
        assert proc._region == "both"

    def test_resolved_region(self):
        proc = WrAnalysis(mass_point=None, region="resolved")
        assert proc._region == "resolved"

    def test_invalid_region_raises(self):
        with pytest.raises(ValueError, match="Invalid region"):
            WrAnalysis(mass_point=None, region="invalid")

    def test_enabled_systs_parsed(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=["lumi", "pileup", "sf"])
        assert proc._enabled_systs == {"lumi", "pileup", "sf"}

    def test_empty_systs(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=[])
        assert proc._enabled_systs == set()

    def test_make_output_produces_hists(self):
        proc = WrAnalysis(mass_point=None)
        output = proc.make_output()
        assert "mass_dilepton" in output
        assert "pt_leading_lepton" in output
        assert "mass_fourobject" in output


class TestSelectLeptons:
    """Test the select_leptons method with mock events."""

    def test_two_tight_muons(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, n_electrons=0)
        tight, loose, masks = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]
        flavors = tight.flavor[0].tolist()
        assert all(f == "muon" for f in flavors)

    def test_two_tight_electrons(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=0, n_electrons=2)
        tight, loose, masks = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]
        flavors = tight.flavor[0].tolist()
        assert all(f == "electron" for f in flavors)

    def test_low_pt_muons_rejected(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, muon_pts=[10.0, 10.0])
        tight, loose, masks = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [0]

    def test_mixed_flavor(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=1, n_electrons=1)
        tight, loose, masks = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]

    def test_sorted_by_pt(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, muon_pts=[55.0, 200.0])
        tight, loose, masks = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]
        # Higher pT should be first
        assert tight[0, 0].pt > tight[0, 1].pt

    def test_masks_returned(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, n_electrons=1)
        tight, loose, masks = proc.select_leptons(ev)
        assert "ele_pteta" in masks
        assert "mu_pteta" in masks
        assert "ele_id" in masks
        assert "mu_id" in masks


class TestSelectJets:
    """Test the select_jets method."""

    def test_two_jets_pass(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_jets=2)
        ak4, ak8, masks = proc.select_jets(ev, "RunIISummer20UL18")
        assert ak.num(ak4, axis=1).tolist() == [2]

    def test_low_pt_jets_rejected(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_jets=2, jet_pts=[10.0, 20.0])
        ak4, ak8, masks = proc.select_jets(ev, "RunIISummer20UL18")
        assert ak.num(ak4, axis=1).tolist() == [0]

    def test_high_eta_jets_rejected(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_jets=0)
        ev.Jet = _make_collection(
            [100.0], [3.5],
            phis=[0.0], masses=[10.0],
            extra_fields={"isTightLeptonVeto": np.array([True])},
        )
        ak4, ak8, masks = proc.select_jets(ev, "RunIISummer20UL18")
        assert ak.num(ak4, axis=1).tolist() == [0]

    def test_ak8_jets_empty_by_default(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_jets=2, n_fatjets=0)
        ak4, ak8, masks = proc.select_jets(ev, "RunIISummer20UL18")
        assert ak.num(ak8, axis=1).tolist() == [0]


class TestTriggerMasks:
    """Test build_trigger_masks for different eras."""

    def _events_with_hlt(self, n=5, **hlt_paths):
        class HLT:
            pass
        hlt = HLT()
        for name, vals in hlt_paths.items():
            setattr(hlt, name, ak.Array(vals))

        class Events:
            pass
        ev = Events()
        ev.HLT = hlt
        type(ev).__len__ = lambda self: n
        return ev

    def test_ul18_electron_trigger(self):
        proc = WrAnalysis(mass_point=None)
        ev = self._events_with_hlt(
            n=3,
            Ele32_WPTight_Gsf=[True, False, False],
            Photon200=[False, True, False],
            Ele115_CaloIdVT_GsfTrkIdT=[False, False, False],
        )
        e_trig, mu_trig, emu_trig = proc.build_trigger_masks(ev, "RunIISummer20UL18")
        assert e_trig.tolist() == [True, True, False]

    def test_run3_2024_muon_trigger(self):
        proc = WrAnalysis(mass_point=None)
        ev = self._events_with_hlt(
            n=2,
            Mu50=[False, True],
            HighPtTkMu100=[True, False],
            CascadeMu100=[False, False],
        )
        e_trig, mu_trig, emu_trig = proc.build_trigger_masks(ev, "RunIII2024Summer24")
        assert mu_trig.tolist() == [True, True]

    def test_missing_hlt_path_defaults_false(self):
        proc = WrAnalysis(mass_point=None)
        ev = self._events_with_hlt(n=2)
        e_trig, mu_trig, emu_trig = proc.build_trigger_masks(ev, "RunIISummer20UL18")
        assert all(not v for v in e_trig.tolist())
        assert all(not v for v in mu_trig.tolist())


class TestElectronCut:
    """Test the ElectronCut bitmap decoder."""

    def test_all_pass(self):
        proc = WrAnalysis(mass_point=None)
        bitmap = sum(2 << (i * 3) for i in range(10))
        result = proc.ElectronCut(bitmap, id_level=2)
        assert result == True

    def test_one_fails(self):
        proc = WrAnalysis(mass_point=None)
        bitmap = 1  # cut 0 = 1
        bitmap += sum(2 << (i * 3) for i in range(1, 10))
        result = proc.ElectronCut(bitmap, id_level=2)
        assert result == False

    def test_isolation_ignored(self):
        proc = WrAnalysis(mass_point=None)
        bitmap = sum(2 << (i * 3) for i in range(10))
        bitmap &= ~(7 << (7 * 3))  # zero out isolation cut
        result = proc.ElectronCut(bitmap, id_level=2)
        assert result == True
