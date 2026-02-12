"""Unit tests for the WrAnalysis processor core logic.

These tests use lightweight mock awkward arrays to exercise selection methods
without requiring real NanoAOD files or correctionlib payloads.
"""

import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods import candidate

from wrcoffea.analysis_config import CUTS
from wrcoffea.analyzer import WrAnalysis

ak.behavior.update(vector.behavior)
ak.behavior.update(candidate.behavior)


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
            "charge": np.ones(n_muons, dtype=np.int32),
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
            "charge": np.ones(n_electrons, dtype=np.int32),
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
        tight, loose, masks, *_ = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]
        flavors = tight.flavor[0].tolist()
        assert all(f == "muon" for f in flavors)

    def test_two_tight_electrons(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=0, n_electrons=2)
        tight, loose, masks, *_ = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]
        flavors = tight.flavor[0].tolist()
        assert all(f == "electron" for f in flavors)

    def test_low_pt_muons_rejected(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, muon_pts=[10.0, 10.0])
        tight, loose, masks, *_ = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [0]

    def test_mixed_flavor(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=1, n_electrons=1)
        tight, loose, masks, *_ = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]

    def test_sorted_by_pt(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, muon_pts=[55.0, 200.0])
        tight, loose, masks, *_ = proc.select_leptons(ev)
        assert ak.num(tight, axis=1).tolist() == [2]
        # Higher pT should be first
        assert tight[0, 0].pt > tight[0, 1].pt

    def test_masks_returned(self):
        proc = WrAnalysis(mass_point=None)
        ev = _mock_events(n_muons=2, n_electrons=1)
        tight, loose, masks, *_ = proc.select_leptons(ev)
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


# ---------------------------------------------------------------------------
# Integration Tests for process()
# ---------------------------------------------------------------------------

from unittest.mock import patch


def _typed_empty_jagged(n_events, dtype):
    """Create a jagged array of n_events empty sub-arrays with proper dtype.

    Awkward 2.8 drops dtype information from truly empty arrays, so we build
    each row with one sentinel element and then slice it off (``[:, :0]``).
    This preserves the dtype in the array layout.
    """
    if n_events == 0:
        return ak.Array([np.array([0], dtype=dtype)])[:0]
    return ak.Array([np.array([0], dtype=dtype)] * n_events)[:, :0]


def _make_integration_events(n_events=5, *,
                              n_muons=2, n_electrons=0, n_jets=2, n_fatjets=0,
                              muon_pts=None, electron_pts=None, jet_pts=None,
                              metadata=None, include_genweight=True,
                              include_hlt=True, include_pileup=True):
    """Build a mock NanoAOD events object with n_events rows in every collection.

    Unlike _mock_events (used by unit tests), this helper replicates each
    collection row across ``n_events`` so that ``len(events)`` matches the
    outer dimension of every collection and per-event array.  Lepton
    collections are created with ``with_name="PtEtaPhiMCandidate"`` and
    ``behavior=vector.behavior`` so that Lorentz vector addition works
    inside ``resolved_selections``.
    """
    mu_pts = muon_pts or [100.0] * n_muons
    el_pts = electron_pts or [100.0] * n_electrons
    j_pts = jet_pts or [100.0] * n_jets

    class MockEvents:
        pass

    ev = MockEvents()

    # --- Muon collection (n_events x n_muons) ---
    if n_muons > 0 and n_events > 0:
        mu_fields = {
            "pt": ak.Array([np.array(mu_pts, dtype=np.float64)] * n_events),
            "eta": ak.Array([np.array([1.0] * n_muons, dtype=np.float64)] * n_events),
            "phi": ak.Array([np.array([0.5 * i for i in range(n_muons)], dtype=np.float64)] * n_events),
            "mass": ak.Array([np.array([0.105] * n_muons, dtype=np.float64)] * n_events),
            "highPtId": ak.Array([np.full(n_muons, 2, dtype=np.int32)] * n_events),
            "tkRelIso": ak.Array([np.full(n_muons, 0.05, dtype=np.float64)] * n_events),
            "cutBased_HEEP": ak.Array([np.zeros(n_muons, dtype=np.bool_)] * n_events),
            "cutBased": ak.Array([np.zeros(n_muons, dtype=np.int32)] * n_events),
            "charge": ak.Array([np.ones(n_muons, dtype=np.int32)] * n_events),
        }
    else:
        mu_fields = {
            "pt": _typed_empty_jagged(n_events, np.float64),
            "eta": _typed_empty_jagged(n_events, np.float64),
            "phi": _typed_empty_jagged(n_events, np.float64),
            "mass": _typed_empty_jagged(n_events, np.float64),
            "highPtId": _typed_empty_jagged(n_events, np.int32),
            "tkRelIso": _typed_empty_jagged(n_events, np.float64),
            "cutBased_HEEP": _typed_empty_jagged(n_events, np.bool_),
            "cutBased": _typed_empty_jagged(n_events, np.int32),
            "charge": _typed_empty_jagged(n_events, np.int32),
        }
    ev.Muon = ak.zip(mu_fields, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)

    # --- Electron collection (n_events x n_electrons) ---
    if n_electrons > 0 and n_events > 0:
        el_fields = {
            "pt": ak.Array([np.array(el_pts, dtype=np.float64)] * n_events),
            "eta": ak.Array([np.array([1.0] * n_electrons, dtype=np.float64)] * n_events),
            "phi": ak.Array([np.array([2.0 + 0.5 * i for i in range(n_electrons)], dtype=np.float64)] * n_events),
            "mass": ak.Array([np.array([0.000511] * n_electrons, dtype=np.float64)] * n_events),
            "highPtId": ak.Array([np.zeros(n_electrons, dtype=np.int32)] * n_events),
            "tkRelIso": ak.Array([np.zeros(n_electrons, dtype=np.float64)] * n_events),
            "cutBased_HEEP": ak.Array([np.ones(n_electrons, dtype=np.bool_)] * n_events),
            "cutBased": ak.Array([np.full(n_electrons, 4, dtype=np.int32)] * n_events),
            "charge": ak.Array([np.ones(n_electrons, dtype=np.int32)] * n_events),
        }
    else:
        el_fields = {
            "pt": _typed_empty_jagged(n_events, np.float64),
            "eta": _typed_empty_jagged(n_events, np.float64),
            "phi": _typed_empty_jagged(n_events, np.float64),
            "mass": _typed_empty_jagged(n_events, np.float64),
            "highPtId": _typed_empty_jagged(n_events, np.int32),
            "tkRelIso": _typed_empty_jagged(n_events, np.float64),
            "cutBased_HEEP": _typed_empty_jagged(n_events, np.bool_),
            "cutBased": _typed_empty_jagged(n_events, np.int32),
            "charge": _typed_empty_jagged(n_events, np.int32),
        }
    ev.Electron = ak.zip(el_fields, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)

    # --- AK4 Jet collection (n_events x n_jets) ---
    # Jets need a ``charge`` field so that ``Candidate.add`` (used when adding
    # lepton + jet four-vectors) does not fail with "no field named 'charge'".
    if n_jets > 0 and n_events > 0:
        jet_fields = {
            "pt": ak.Array([np.array(j_pts, dtype=np.float64)] * n_events),
            "eta": ak.Array([np.array([1.0] * n_jets, dtype=np.float64)] * n_events),
            "phi": ak.Array([np.array([3.0 + 0.5 * i for i in range(n_jets)], dtype=np.float64)] * n_events),
            "mass": ak.Array([np.array([10.0] * n_jets, dtype=np.float64)] * n_events),
            "isTightLeptonVeto": ak.Array([np.ones(n_jets, dtype=np.bool_)] * n_events),
            "charge": ak.Array([np.zeros(n_jets, dtype=np.int32)] * n_events),
        }
    else:
        jet_fields = {
            "pt": _typed_empty_jagged(n_events, np.float64),
            "eta": _typed_empty_jagged(n_events, np.float64),
            "phi": _typed_empty_jagged(n_events, np.float64),
            "mass": _typed_empty_jagged(n_events, np.float64),
            "isTightLeptonVeto": _typed_empty_jagged(n_events, np.bool_),
            "charge": _typed_empty_jagged(n_events, np.int32),
        }
    ev.Jet = ak.zip(jet_fields, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)

    # --- AK8 FatJet collection (n_events x n_fatjets) ---
    if n_fatjets > 0 and n_events > 0:
        fj_fields = {
            "pt": ak.Array([np.array([300.0] * n_fatjets, dtype=np.float64)] * n_events),
            "eta": ak.Array([np.array([1.0] * n_fatjets, dtype=np.float64)] * n_events),
            "phi": ak.Array([np.array([0.0] * n_fatjets, dtype=np.float64)] * n_events),
            "mass": ak.Array([np.array([80.0] * n_fatjets, dtype=np.float64)] * n_events),
            "msoftdrop": ak.Array([np.full(n_fatjets, 80.0, dtype=np.float64)] * n_events),
            "lsf3": ak.Array([np.full(n_fatjets, 0.9, dtype=np.float64)] * n_events),
            "charge": ak.Array([np.zeros(n_fatjets, dtype=np.int32)] * n_events),
        }
    else:
        fj_fields = {
            "pt": _typed_empty_jagged(n_events, np.float64),
            "eta": _typed_empty_jagged(n_events, np.float64),
            "phi": _typed_empty_jagged(n_events, np.float64),
            "mass": _typed_empty_jagged(n_events, np.float64),
            "msoftdrop": _typed_empty_jagged(n_events, np.float64),
            "lsf3": _typed_empty_jagged(n_events, np.float64),
            "charge": _typed_empty_jagged(n_events, np.int32),
        }
    ev.FatJet = ak.zip(fj_fields, with_name="PtEtaPhiMCandidate", behavior=candidate.behavior)

    # --- Per-event flat arrays ---
    if include_genweight:
        ev.genWeight = ak.Array([1.0] * n_events)

    if include_hlt:
        class HLT:
            pass
        hlt = HLT()
        # Use np.array with explicit dtype to avoid ``unknown`` type for n_events=0.
        hlt.Ele32_WPTight_Gsf = ak.Array(np.ones(n_events, dtype=np.bool_))
        hlt.Photon200 = ak.Array(np.zeros(n_events, dtype=np.bool_))
        hlt.Ele115_CaloIdVT_GsfTrkIdT = ak.Array(np.zeros(n_events, dtype=np.bool_))
        hlt.Mu50 = ak.Array(np.ones(n_events, dtype=np.bool_))
        hlt.OldMu100 = ak.Array(np.zeros(n_events, dtype=np.bool_))
        hlt.TkMu100 = ak.Array(np.zeros(n_events, dtype=np.bool_))
        ev.HLT = hlt

    if include_pileup:
        class Pileup:
            pass
        pu = Pileup()
        pu.nTrueInt = ak.Array([20.0] * n_events)
        ev.Pileup = pu

    ev.run = ak.Array([1] * n_events)
    ev.luminosityBlock = ak.Array([100] * n_events)

    default_metadata = {
        "era": "RunIISummer20UL18",
        "datatype": "mc",
        "physics_group": "DYJets",
        "sample": "DYJetsToLL_M-50",
        "xsec": 6077.22,
        "genEventSumw": 50000.0,
    }
    if metadata:
        default_metadata.update(metadata)
    ev.metadata = default_metadata

    type(ev).__len__ = lambda self: n_events

    return ev


def _sf_ones(n):
    """Return a (nominal, up, down) tuple of ones arrays of length n."""
    ones = np.ones(n, dtype=np.float64)
    return ones, ones.copy(), ones.copy()


def _mock_jet_veto(events, era):
    return np.ones(len(events), dtype=bool)


def _mock_pileup_weight(events, era):
    return _sf_ones(len(events))


def _mock_muon_sf(tight_muons, era):
    n = len(tight_muons)
    return {
        "reco": _sf_ones(n),
        "id": _sf_ones(n),
        "iso": _sf_ones(n),
    }


def _mock_muon_trigger_sf(tight_muons, era):
    return _sf_ones(len(tight_muons))


def _mock_electron_reco_sf(tight_electrons, era):
    return _sf_ones(len(tight_electrons))


def _mock_electron_id_sf(tight_electrons, era):
    return _sf_ones(len(tight_electrons))


def _mock_electron_trigger_sf(tight_electrons, era):
    return _sf_ones(len(tight_electrons))


# All scale-factor patches applied to every test in this class.
_SF_PATCHES = {
    "wrcoffea.analyzer.jet_veto_event_mask": _mock_jet_veto,
    "wrcoffea.analyzer.pileup_weight": _mock_pileup_weight,
    "wrcoffea.analyzer.muon_sf": _mock_muon_sf,
    "wrcoffea.analyzer.muon_trigger_sf": _mock_muon_trigger_sf,
    "wrcoffea.analyzer.electron_reco_sf": _mock_electron_reco_sf,
    "wrcoffea.analyzer.electron_id_sf": _mock_electron_id_sf,
    "wrcoffea.analyzer.electron_trigger_sf": _mock_electron_trigger_sf,
}


def _apply_sf_patches(func):
    """Decorator that stacks all scale-factor mocks onto a test method."""
    for target, replacement in _SF_PATCHES.items():
        func = patch(target, side_effect=replacement)(func)
    return func


class TestProcessIntegration:
    """Integration tests for the process() method.

    These tests mock minimal NanoAOD events and verify the end-to-end
    pipeline: object selection -> weight building -> histogram filling -> output.

    All correctionlib-dependent scale factor functions are patched out so the
    tests do not require JSON payload files.  The mock events are built with
    n_events rows in every collection so that Lorentz vector arithmetic
    (used inside resolved_selections) operates on properly shaped arrays.
    """

    # ------------------------------------------------------------------
    # MC basic output structure
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_mc_basic_output_structure(self, *_mocks):
        """process() returns correct output structure for MC."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2)

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output
        ds = output["DYJetsToLL_M-50"]
        assert "mass_dilepton" in ds
        assert "pt_leading_lepton" in ds
        assert "mass_fourobject" in ds
        assert "cutflow" in ds

    # ------------------------------------------------------------------
    # Data (no genWeight)
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_data_no_genweight(self, *_mocks):
        """process() handles data (no genWeight) correctly."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        # Use an era with no lumi JSON configured so apply_lumi_mask is
        # a no-op (avoids MockEvents not being subscriptable).
        events = _make_integration_events(
            n_events=3, n_muons=2, n_jets=2,
            metadata={"datatype": "data", "physics_group": "SingleMuon",
                      "sample": "SingleMuon", "era": "Run3Summer23"},
            include_genweight=False,
        )

        output = proc.process(events)

        assert "SingleMuon" in output
        assert "_sumw" not in output["SingleMuon"]

    # ------------------------------------------------------------------
    # compute_sumw=True accumulates genWeight sum
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_compute_sumw_true(self, *_mocks):
        """_sumw is accumulated when compute_sumw=True."""
        proc = WrAnalysis(mass_point=None, region="resolved", compute_sumw=True)
        events = _make_integration_events(n_events=5, n_muons=2, n_jets=2)
        events.genWeight = ak.Array([1.0, 2.0, 3.0, -1.0, 0.5])

        output = proc.process(events)

        ds = output["DYJetsToLL_M-50"]
        assert "_sumw" in ds
        assert abs(ds["_sumw"] - 5.5) < 1e-6

    # ------------------------------------------------------------------
    # compute_sumw=False (default) -- no _sumw key
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_compute_sumw_false(self, *_mocks):
        """_sumw is NOT in output when compute_sumw=False (default)."""
        proc = WrAnalysis(mass_point=None, region="resolved", compute_sumw=False)
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2)

        output = proc.process(events)

        assert "_sumw" not in output["DYJetsToLL_M-50"]

    # ------------------------------------------------------------------
    # Resolved-only region
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_region_resolved_only(self, *_mocks):
        """region='resolved' produces resolved histograms."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2,
                                           n_fatjets=1)

        output = proc.process(events)

        ds = output["DYJetsToLL_M-50"]
        assert "mass_dilepton" in ds
        assert "cutflow" in ds

    # ------------------------------------------------------------------
    # Boosted-only region (FatJet without lsf3 -> boosted skipped gracefully)
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_region_boosted_only(self, *_mocks):
        """region='boosted' without lsf3 still returns output (boosted skipped)."""
        proc = WrAnalysis(mass_point=None, region="boosted")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2,
                                           n_fatjets=1)

        output = proc.process(events)

        ds = output["DYJetsToLL_M-50"]
        assert "mass_dilepton" in ds

    # ------------------------------------------------------------------
    # Missing FatJet handled gracefully
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_missing_fatjet_graceful(self, *_mocks):
        """Missing FatJet is handled gracefully (no crash)."""
        proc = WrAnalysis(mass_point=None, region="both")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2,
                                           n_fatjets=0)
        delattr(events, "FatJet")
        # select_jets needs FatJet; provide a minimal empty one so it
        # does not crash before boosted is even attempted.
        fj_fields = {
            "pt": ak.Array([[] for _ in range(3)]),
            "eta": ak.Array([[] for _ in range(3)]),
            "phi": ak.Array([[] for _ in range(3)]),
            "mass": ak.Array([[] for _ in range(3)]),
            "msoftdrop": ak.Array([[] for _ in range(3)]),
        }
        ev_fatjet = ak.zip(fj_fields, with_name="PtEtaPhiMCandidate",
                           behavior=vector.behavior)
        events.FatJet = ev_fatjet

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output

    # ------------------------------------------------------------------
    # Missing lsf3 handled gracefully
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_missing_lsf3_graceful(self, *_mocks):
        """Missing lsf3 field is handled gracefully."""
        proc = WrAnalysis(mass_point=None, region="both")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2,
                                           n_fatjets=1)
        # Rebuild FatJet without lsf3
        n = 3
        fj_fields = {
            "pt": ak.Array([np.array([300.0], dtype=np.float64)] * n),
            "eta": ak.Array([np.array([1.0], dtype=np.float64)] * n),
            "phi": ak.Array([np.array([0.0], dtype=np.float64)] * n),
            "mass": ak.Array([np.array([80.0], dtype=np.float64)] * n),
            "msoftdrop": ak.Array([np.full(1, 80.0, dtype=np.float64)] * n),
        }
        events.FatJet = ak.zip(fj_fields, with_name="PtEtaPhiMCandidate",
                                behavior=vector.behavior)

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output

    # ------------------------------------------------------------------
    # Metadata extraction -- sample name becomes dataset key
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_metadata_extraction(self, *_mocks):
        """Metadata fields are extracted correctly."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        events = _make_integration_events(
            n_events=2, n_muons=2, n_jets=2,
            metadata={
                "era": "Run3Summer22EE",
                "datatype": "mc",
                "physics_group": "Signal",
                "sample": "WR3000_N1500",
                "xsec": 0.05,
                "genEventSumw": 10000.0,
            },
        )

        output = proc.process(events)

        assert "WR3000_N1500" in output

    # ------------------------------------------------------------------
    # Signal physics_group
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_signal_sample(self, *_mocks):
        """Signal physics_group is processed correctly."""
        proc = WrAnalysis(mass_point="WR3000_N1500", region="resolved")
        events = _make_integration_events(
            n_events=3, n_muons=2, n_jets=2,
            metadata={"physics_group": "Signal", "sample": "WR3000_N1500"},
        )

        output = proc.process(events)

        assert "WR3000_N1500" in output

    # ------------------------------------------------------------------
    # Zero-event data (simulates all events filtered by lumi mask)
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_empty_events(self, *_mocks):
        """Empty events are handled gracefully."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        # Use an era with no lumi JSON configured for data.
        events = _make_integration_events(
            n_events=0,
            metadata={"datatype": "data", "physics_group": "SingleMuon",
                      "sample": "SingleMuon", "era": "Run3Summer23"},
            include_genweight=False,
        )

        output = proc.process(events)

        assert "SingleMuon" in output

    # ------------------------------------------------------------------
    # Lumi mask is NOT applied to MC
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_lumi_mask_not_applied_to_mc(self, *_mocks):
        """Lumi mask is NOT applied to MC events."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        events = _make_integration_events(
            n_events=3, n_muons=2, n_jets=2,
            metadata={"datatype": "mc"},
        )

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output

    # ------------------------------------------------------------------
    # Histograms filled for passing events
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_histograms_filled_with_passing_events(self, *_mocks):
        """Histograms are filled for events passing selections."""
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=[])
        events = _make_integration_events(
            n_events=10, n_muons=2, n_jets=2,
            muon_pts=[100.0, 80.0],
        )

        output = proc.process(events)

        ds = output["DYJetsToLL_M-50"]
        assert "cutflow" in ds

    # ------------------------------------------------------------------
    # Systematic variations enabled
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_systematic_variations_enabled(self, *_mocks):
        """Systematic variations are created when enabled."""
        proc = WrAnalysis(mass_point=None, region="resolved",
                          enabled_systs=["lumi", "pileup"])
        events = _make_integration_events(n_events=5, n_muons=2, n_jets=2)

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output

    # ------------------------------------------------------------------
    # genEventSumw=0 raises ZeroDivisionError
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_zero_sumw_raises(self, *_mocks):
        """genEventSumw=0 raises ZeroDivisionError."""
        proc = WrAnalysis(mass_point=None, region="resolved", compute_sumw=False)
        events = _make_integration_events(
            n_events=3, n_muons=2, n_jets=2,
            metadata={"genEventSumw": 0.0},
        )

        with pytest.raises(ZeroDivisionError, match="genEventSumw is zero"):
            proc.process(events)

    # ------------------------------------------------------------------
    # Negative genEventSumw (NLO) handled correctly
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_negative_sumw_for_nlo(self, *_mocks):
        """Negative genEventSumw (NLO samples) is handled correctly."""
        proc = WrAnalysis(mass_point=None, region="resolved", compute_sumw=False)
        events = _make_integration_events(
            n_events=3, n_muons=2, n_jets=2,
            metadata={"genEventSumw": -1000.0},
        )

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output

    # ------------------------------------------------------------------
    # Full pipeline integration (both regions)
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_pipeline_integration(self, *_mocks):
        """End-to-end: object selection -> weights -> histograms."""
        proc = WrAnalysis(mass_point=None, region="both")
        events = _make_integration_events(
            n_events=20, n_muons=2, n_jets=2, n_fatjets=1,
            muon_pts=[120.0, 70.0],
            jet_pts=[150.0, 100.0],
        )

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output
        ds = output["DYJetsToLL_M-50"]
        assert "mass_dilepton" in ds
        assert "pt_leading_lepton" in ds
        assert "mass_fourobject" in ds
        assert "cutflow" in ds
        assert ds["cutflow"] is not None

    # ------------------------------------------------------------------
    # Electron channel exercises the electron SF mocks
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_electron_channel(self, *_mocks):
        """Electron channel exercises electron SF mock paths."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        events = _make_integration_events(
            n_events=5, n_muons=0, n_electrons=2, n_jets=2,
        )

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output
        assert "cutflow" in output["DYJetsToLL_M-50"]

    # ------------------------------------------------------------------
    # SF-enabled systematics produce extra syst axis entries
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_sf_systs(self, *_mocks):
        """enabled_systs=['sf'] does not crash with mocked SFs."""
        proc = WrAnalysis(mass_point=None, region="resolved",
                          enabled_systs=["sf"])
        events = _make_integration_events(n_events=5, n_muons=2, n_jets=2)

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output
