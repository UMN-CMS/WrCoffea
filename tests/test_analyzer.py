"""Unit tests for the WrAnalysis processor core logic.

These tests use lightweight mock awkward arrays to exercise selection methods
without requiring real NanoAOD files or correctionlib payloads.
"""

import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods import candidate

from wrcoffea.analysis_config import CUTS, LUMIS
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
            "pdgId": np.full(n_muons, 13, dtype=np.int32),
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
            "pdgId": np.full(n_electrons, 11, dtype=np.int32),
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
        def __len__(self):
            return len(self.run)

        def __getitem__(self, mask):
            filtered = type(self)()
            for name, value in vars(self).items():
                if name == "metadata":
                    setattr(filtered, name, value)
                elif isinstance(value, ak.Array):
                    setattr(filtered, name, value[mask])
                else:
                    collection = type(value)()
                    for field, values in vars(value).items():
                        setattr(collection, field, values[mask])
                    setattr(filtered, name, collection)
            return filtered

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
            "pdgId": ak.Array([np.full(n_muons, 13, dtype=np.int32)] * n_events),
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
            "pdgId": _typed_empty_jagged(n_events, np.int32),
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
            "pdgId": ak.Array([np.full(n_electrons, 11, dtype=np.int32)] * n_events),
            # All 10 vid cuts at level 2 ("all pass" for id_level=2).
            "vidNestedWPBitmap": ak.Array(
                [np.full(n_electrons, sum(2 << (i * 3) for i in range(10)),
                         dtype=np.int64)] * n_events),
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
            "pdgId": _typed_empty_jagged(n_events, np.int32),
            "vidNestedWPBitmap": _typed_empty_jagged(n_events, np.int64),
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
            # process() computes raw pT inline: pt * (1 - rawFactor).
            "rawFactor": ak.Array([np.zeros(n_jets, dtype=np.float64)] * n_events),
        }
    else:
        jet_fields = {
            "pt": _typed_empty_jagged(n_events, np.float64),
            "eta": _typed_empty_jagged(n_events, np.float64),
            "phi": _typed_empty_jagged(n_events, np.float64),
            "mass": _typed_empty_jagged(n_events, np.float64),
            "isTightLeptonVeto": _typed_empty_jagged(n_events, np.bool_),
            "charge": _typed_empty_jagged(n_events, np.int32),
            "rawFactor": _typed_empty_jagged(n_events, np.float64),
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
            "isTight": ak.Array([np.ones(n_fatjets, dtype=np.bool_)] * n_events),
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
            "isTight": _typed_empty_jagged(n_events, np.bool_),
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

    return ev


def _sf_ones(n):
    """Return a (nominal, up, down) tuple of ones arrays of length n."""
    ones = np.ones(n, dtype=np.float64)
    return ones, ones.copy(), ones.copy()


def _mock_jet_veto(events, ak4_id_mask, era):
    return np.ones(len(events), dtype=bool)


def _mock_pileup_weight(events, era):
    # (nom, up, down) per-event arrays.
    return _sf_ones(len(events))


def _mock_muon_sf(tight_muons, era):
    # muon_sf returns {"reco"|"id"|"iso": (nom, up, down)} with per-event
    # arrays (already product-reduced over the muons in each event).
    n = len(tight_muons)
    return {
        "reco": _sf_ones(n),
        "id": _sf_ones(n),
        "iso": _sf_ones(n),
    }


def _mock_muon_sf_loose(loose_muons, era):
    # muon_sf_loose returns {"reco"|"id": (nom, up, down)} — no ISO component.
    n = len(loose_muons)
    return {
        "reco": _sf_ones(n),
        "id": _sf_ones(n),
    }


def _mock_muon_trigger_sf(tight_muons, era):
    return _sf_ones(len(tight_muons))


def _mock_electron_reco_sf(tight_electrons, era):
    return _sf_ones(len(tight_electrons))


def _mock_electron_id_sf(tight_electrons, era):
    return _sf_ones(len(tight_electrons))


def _mock_electron_trigger_sf(tight_electrons, era):
    return _sf_ones(len(tight_electrons))


def _mock_apply_jet_corrections(events, era, isMC, variations=()):
    # Passthrough: no JEC/JER applied to mock events.
    return events


def _mock_apply_muon_scale_smearing(events, era, is_mc):
    # Falsy return -> process() keeps the original Muon kinematics.
    return None


# All correction patches applied to every test in this class.
def _mock_dy_zpt_weight(events, era, *, is_dy):
    from wrcoffea.dy_reweight import identity_weights
    return identity_weights(len(events))


def _mock_signal_kfactor_weight(events, era, *, mass_point, is_signal):
    return {"nominal": np.ones(len(events), dtype=np.float64), "applied": False}


def _mock_dy_reshape_weight(jet_pt, era, kind, *, is_dy):
    return None   # behave as an era without a reshape payload


_SF_PATCHES = {
    "wrcoffea.analyzer.jet_veto_event_mask": _mock_jet_veto,
    "wrcoffea.analyzer.dy_zpt_weight": _mock_dy_zpt_weight,
    "wrcoffea.analyzer.signal_kfactor_weight": _mock_signal_kfactor_weight,
    "wrcoffea.analyzer.dy_reshape_weight": _mock_dy_reshape_weight,
    "wrcoffea.analyzer.pileup_weight": _mock_pileup_weight,
    "wrcoffea.analyzer.muon_sf": _mock_muon_sf,
    "wrcoffea.analyzer.muon_sf_loose": _mock_muon_sf_loose,
    "wrcoffea.analyzer.muon_trigger_sf": _mock_muon_trigger_sf,
    "wrcoffea.analyzer.electron_reco_sf": _mock_electron_reco_sf,
    "wrcoffea.analyzer.electron_id_sf": _mock_electron_id_sf,
    "wrcoffea.analyzer.electron_trigger_sf": _mock_electron_trigger_sf,
    "wrcoffea.analyzer.apply_jet_corrections": _mock_apply_jet_corrections,
    "wrcoffea.analyzer.apply_muon_scale_smearing": _mock_apply_muon_scale_smearing,
    # Isolate the processor from lumi/noise filtering, which has separate
    # coverage and requires additional event fields or payloads.
    "wrcoffea.analyzer.WrAnalysis.apply_lumi_mask":
        lambda self, events, mc_campaign, is_data: events,
    "wrcoffea.analyzer.WrAnalysis.apply_noise_filter":
        lambda self, events, mc_campaign: events,
}


def _apply_sf_patches(func):
    """Decorator that stacks all scale-factor mocks onto a test method."""
    for target, replacement in _SF_PATCHES.items():
        func = patch(target, new=replacement)(func)
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
        # Physics hists carry the canonical categorical axes.
        assert [ax.name for ax in ds["mass_dilepton"].axes[:3]] == [
            "process", "region", "syst"]
        # All 3 events enter the cutflow.
        cum = ds["cutflow"]["mumu"]["cumulative_unweighted"]
        assert cum.values()[0] == 3.0  # no_cuts bin

    @pytest.mark.parametrize("systs, expected", [
        ([], set()),
        (["lumi", "sf"], set()),
        (["jer"], {"jer"}),
        (["jes", "elescale"], {"jes"}),
        (["jer", "jes", "pileup"], {"jer", "jes"}),
    ])
    @_apply_sf_patches
    def test_process_requests_only_enabled_jet_variations(self, systs, expected):
        """apply_jet_corrections is asked for exactly the enabled JER/JES variations."""
        proc = WrAnalysis(mass_point=None, enabled_systs=systs, region="resolved")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2)
        events.genWeight = ak.Array([1.0, 1.0, 1.0])

        # The identity mock attaches no shifted kinematics, so the shape passes
        # (which UL18 can now produce) are planned away: this test only checks
        # what apply_jet_corrections is asked for.
        with patch("wrcoffea.analyzer.apply_jet_corrections",
                   side_effect=lambda ev, era, is_mc, variations=(): ev) as jec, \
             patch.object(WrAnalysis, "_plan_shape_variations", return_value=([], [])):
            proc.process(events)

        assert jec.call_count == 1
        assert set(jec.call_args.kwargs["variations"]) == expected

    @pytest.mark.parametrize("keep_mask", [[True, False, True], [False, False, False]])
    @_apply_sf_patches
    def test_process_applies_jet_veto_mask(self, keep_mask):
        """Downstream corrections receive filtered events, including an empty chunk."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        events = _make_integration_events(n_events=3, n_muons=2, n_jets=2)
        events.genWeight = ak.Array([1.0, 2.0, 3.0])
        mask = np.asarray(keep_mask, dtype=bool)

        with patch("wrcoffea.analyzer.jet_veto_event_mask", return_value=mask), \
             patch("wrcoffea.analyzer.apply_electron_scale_smearing", return_value={}) as correction:
            output = proc.process(events)

        filtered = correction.call_args.args[0]
        assert filtered.metadata == events.metadata
        assert len(filtered) == np.count_nonzero(mask)
        np.testing.assert_array_equal(filtered.genWeight, np.asarray(events.genWeight)[mask])
        assert len(filtered.Jet) == len(filtered.Muon) == len(filtered.HLT.Mu50) == len(filtered)
        cum = output["DYJetsToLL_M-50"]["cutflow"]["mumu"]["cumulative_unweighted"]
        assert cum.values()[0] == np.count_nonzero(mask)

    # ------------------------------------------------------------------
    # Data (no genWeight)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("compute_sumw", [False, True])
    @_apply_sf_patches
    def test_process_data_no_genweight(self, compute_sumw):
        """process() handles data (no genWeight) correctly."""
        proc = WrAnalysis(mass_point=None, region="resolved", compute_sumw=compute_sumw)
        # apply_lumi_mask is patched to a passthrough so these synthetic
        # run/lumi values do not need to appear in the golden JSON.
        events = _make_integration_events(
            n_events=3, n_muons=2, n_jets=2,
            metadata={"datatype": "data", "physics_group": "SingleMuon",
                      "sample": "SingleMuon", "era": "Run3Summer23"},
            include_genweight=False,
        )

        output = proc.process(events)

        assert "SingleMuon" in output
        assert "_sumw" not in output["SingleMuon"]
        # Data uses unit weights: all 3 events counted through the trigger.
        cum = output["SingleMuon"]["cutflow"]["mumu"]["cumulative_unweighted"]
        counts = dict(zip(list(cum.axes[0]), cum.values()))
        assert counts["no_cuts"] == 3.0
        assert counts["mu_trigger"] == 3.0

    @pytest.mark.parametrize("flavor", ["ee", "mumu"])
    @pytest.mark.parametrize("topology", ["resolved", "boosted"])
    @pytest.mark.parametrize("nomass", [False, True], ids=["sr", "nomass"])
    @_apply_sf_patches
    def test_process_data_skips_lepton_sfs(self, flavor, topology, nomass):
        """Selected data retains unit weights without evaluating efficiency payloads."""
        from contextlib import ExitStack

        proc = WrAnalysis(
            mass_point=None, region="both", region_variants={"nomass"},
            enabled_systs=["sf", "lumi", "pileup", "jes", "jer", "elescale", "muscale"],
        )
        n = 3
        events = _make_integration_events(
            n_events=n, n_muons=2 if flavor == "mumu" else 0,
            n_electrons=2 if flavor == "ee" else 0, n_jets=2, n_fatjets=1,
            metadata={"datatype": "data", "physics_group": "Observed", "sample": "Observed"},
            include_genweight=False,
        )
        collection = "Muon" if flavor == "mumu" else "Electron"
        leptons = getattr(events, collection)
        leptons = ak.with_field(leptons, ak.Array([[300.0, 250.0]] * n), "pt")
        leptons = ak.with_field(leptons, ak.zeros_like(leptons.eta), "eta")
        leptons = ak.with_field(leptons, ak.Array([[0.0, np.pi]] * n), "phi")
        if topology == "boosted":
            # The second lepton is loose and embedded in the AK8 jet.
            if collection == "Muon":
                leptons = ak.with_field(leptons, ak.Array([[0.05, 0.2]] * n), "tkRelIso")
            else:
                leptons = ak.with_field(leptons, ak.Array([[True, False]] * n), "cutBased_HEEP")
                leptons = ak.with_field(leptons, ak.Array([[4, 2]] * n), "cutBased")
        setattr(events, collection, leptons)
        jet_pt = 60.0 if nomass else 300.0
        events.Jet = ak.with_field(events.Jet, ak.full_like(events.Jet.pt, jet_pt), "pt")
        events.Jet = ak.with_field(events.Jet, ak.zeros_like(events.Jet.eta), "eta")
        events.Jet = ak.with_field(events.Jet, ak.Array([[np.pi / 2, -np.pi / 2]] * n), "phi")
        events.FatJet = ak.with_field(
            events.FatJet, ak.full_like(events.FatJet.pt, 300.0 if nomass else 600.0), "pt",
        )
        events.FatJet = ak.with_field(events.FatJet, ak.zeros_like(events.FatJet.eta), "eta")
        events.FatJet = ak.with_field(events.FatJet, ak.full_like(events.FatJet.phi, np.pi), "phi")

        with ExitStack() as stack:
            for name in ("muon_sf", "muon_sf_loose", "muon_trigger_sf",
                         "electron_reco_sf", "electron_id_sf", "electron_trigger_sf"):
                stack.enter_context(patch(
                    f"wrcoffea.analyzer.{name}",
                    side_effect=AssertionError(f"{name} must not be called for data"),
                ))
            output = proc.process(events)["Observed"]

        suffix = "sr_nomass" if nomass else "sr"
        selected = output["mass_dilepton"][{
            "process": "Observed", "region": f"wr_{flavor}_{topology}_{suffix}", "syst": "Nominal",
        }]
        assert selected.sum(flow=True).value == pytest.approx(n)
        assert selected.sum(flow=True).variance == pytest.approx(n)
        for obj in output.values():
            if hasattr(obj, "axes"):
                # Booked but unfilled histograms can have an empty category axis.
                assert set(obj.axes["syst"]) <= {"Nominal"}
        assert "_sumw" not in output

    # ------------------------------------------------------------------
    # compute_sumw=True accumulates genWeight sum
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("noise_keep, veto_keep", [
        ([True] * 5, [True] * 5),
        ([True, False, True, False, True], [True] * 3),
        ([True] * 5, [True, False, True, False, True]),
        ([True, False, True, False, True], [True, False, True]),
        ([False] * 5, []),
        ([True] * 5, [False] * 5),
        ([], []),
    ], ids=["no-rejection", "noise", "jet-veto", "both", "all-noise", "all-veto", "empty"])
    @_apply_sf_patches
    def test_process_compute_sumw_true(self, noise_keep, veto_keep):
        """Normalization includes signed weights before noise filtering and jet veto."""
        proc = WrAnalysis(mass_point=None, region="resolved", compute_sumw=True)
        events = _make_integration_events(n_events=len(noise_keep), n_muons=2, n_jets=2)
        genweights = np.array([1.0, 2.0, 3.0, -1.0, 0.5])[:len(events)]
        events.genWeight = ak.Array(genweights)
        noise_mask = np.asarray(noise_keep, dtype=bool)
        veto_mask = np.asarray(veto_keep, dtype=bool)

        with patch.object(proc, "apply_noise_filter", side_effect=lambda ev, era: ev[noise_mask]), \
             patch("wrcoffea.analyzer.jet_veto_event_mask", return_value=veto_mask):
            output = proc.process(events)

        ds = output["DYJetsToLL_M-50"]
        assert ds["_sumw"] == pytest.approx(genweights.sum())
        cutflow = ds["cutflow"]["mumu"]
        assert cutflow["cumulative_unweighted"].values()[0] == np.count_nonzero(veto_mask)
        scale = events.metadata["xsec"] * LUMIS[events.metadata["era"]] * 1000.0
        assert cutflow["cumulative"].values()[0] == pytest.approx(
            genweights[noise_mask][veto_mask].sum() * scale
        )

    @_apply_sf_patches
    def test_process_sumw_normalization_across_filtered_chunks(self, monkeypatch):
        """Merged unskimmed output matches normalization using the full sample sumw."""
        from pathlib import Path
        from coffea.processor import accumulate

        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "bin"))
        from run_analysis import normalize_by_sumw

        events = _make_integration_events(n_events=4, metadata={"genEventSumw": 3.5})
        events.genWeight = ak.Array([3.0, -1.0, 2.0, -0.5])
        unskimmed = WrAnalysis(mass_point=None, region="resolved", compute_sumw=True)
        precomputed = WrAnalysis(mass_point=None, region="resolved", compute_sumw=False)
        chunks = (events[:1], events[1:2], events[2:])

        with patch("wrcoffea.analyzer.WrAnalysis.apply_noise_filter",
                   new=lambda self, ev, era: ev[ev.genWeight != -1.0]), \
             patch("wrcoffea.analyzer.jet_veto_event_mask",
                   side_effect=lambda ev, mask, era: ak.to_numpy(ev.genWeight > 0)):
            # The middle chunk is fully rejected but still contributes -1 to sumw.
            merged = accumulate(unskimmed.process(chunk) for chunk in chunks)
            reference = accumulate(precomputed.process(chunk) for chunk in chunks)

        dataset = events.metadata["sample"]
        assert merged[dataset]["_sumw"] == pytest.approx(3.5)
        normalized = normalize_by_sumw(merged)
        assert "_sumw" not in normalized[dataset]
        for name, expected in reference[dataset]["cutflow"]["mumu"].items():
            actual = normalized[dataset]["cutflow"]["mumu"][name]
            np.testing.assert_allclose(actual.values(), expected.values())
            if expected.variances() is None:
                assert actual.variances() is None
            else:
                np.testing.assert_allclose(actual.variances(), expected.variances())

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
    @pytest.mark.parametrize("flavor", ["ee", "mumu"])
    @pytest.mark.parametrize("region", ["dy_cr", "sr", "sr_nomass"])
    @_apply_sf_patches
    @patch("wrcoffea.analyzer.electron_id_sf", new=_mock_electron_reco_sf)
    def test_resolved_same_flavor_requires_two_total_tight_leptons(self, flavor, region):
        """Extra tight leptons are vetoed; an extra loose lepton remains allowed."""
        proc = WrAnalysis(mass_point=None, region="resolved", region_variants={"nomass"})
        events = _make_integration_events(
            n_events=4, n_muons=2, n_electrons=2,
            metadata={"xsec": 1.0, "genEventSumw": LUMIS["RunIISummer20UL18"] * 1000.0},
        )
        # Distinct weights identify which events reach the region histograms.
        events.genWeight = ak.Array([1.0, 10.0, 100.0, 1000.0])
        pair_name, extra_name = ("Electron", "Muon") if flavor == "ee" else ("Muon", "Electron")
        pair = getattr(events, pair_name)
        pts, phis = ([100.0, 90.0], [0.0, 1.0]) if region == "dy_cr" else (
            [300.0, 250.0], [0.0, np.pi],
        )
        pair = ak.with_field(pair, ak.Array([pts] * 4), "pt")
        pair = ak.with_field(pair, ak.Array([phis] * 4), "phi")
        pair = ak.with_field(pair, ak.zeros_like(pair.eta), "eta")
        setattr(events, pair_name, pair)

        extra = getattr(events, extra_name)
        extra = ak.with_field(extra, ak.Array([[70.0, 65.0]] * 4), "pt")
        if extra_name == "Muon":
            extra = ak.with_field(extra, ak.Array([
                [0.05] * 2, [0.05] * 2, [0.2] * 2, [0.05] * 2,
            ]), "tkRelIso")
        else:
            extra = ak.with_field(extra, ak.Array([
                [True] * 2, [True] * 2, [False] * 2, [True] * 2,
            ]), "cutBased_HEEP")
            extra = ak.with_field(extra, ak.full_like(extra.cutBased, 2), "cutBased")
        # Same-flavor pair plus: nothing, one tight, one loose, two tight.
        extra = extra[ak.Array([[False, False], [True, False], [True, False], [True, True]])]
        setattr(events, extra_name, extra)
        tight, loose, _ = proc.select_leptons(events)
        assert ak.to_list(ak.num(tight)) == [2, 3, 2, 4]
        assert ak.to_list(ak.num(loose)) == [0, 0, 1, 0]

        jet_pt = {"dy_cr": 500.0, "sr": 300.0, "sr_nomass": 60.0}[region]
        events.Jet = ak.with_field(events.Jet, ak.full_like(events.Jet.pt, jet_pt), "pt")
        events.Jet = ak.with_field(events.Jet, ak.zeros_like(events.Jet.eta), "eta")
        events.Jet = ak.with_field(events.Jet, ak.Array([[np.pi / 2, -np.pi / 2]] * 4), "phi")

        output = proc.process(events)[events.metadata["sample"]]
        # Shape variations rebuild selections and must enforce the same veto.
        proc._run_shape_systematic(
            events, output, events.metadata, True, "JERUp",
            proc.build_trigger_masks(events, events.metadata["era"]),
        )
        hist = output["mass_dilepton"]
        for syst in ("Nominal", "JERUp"):
            selected = hist[{
                "process": "DYJets", "region": f"wr_{flavor}_resolved_{region}", "syst": syst,
            }]
            assert selected.sum(flow=True).value == pytest.approx(101.0)

        cutflow = output["cutflow"][flavor]
        for name, expected in (("cumulative_unweighted", 2.0), ("cumulative", 101.0)):
            counts = dict(zip(list(cutflow[name].axes[0]), cutflow[name].values()))
            assert counts["two_tight_leptons"] == pytest.approx(expected)

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
        """Missing FatJet lsf3 is fine for a resolved-only run, an error otherwise."""
        proc = WrAnalysis(mass_point=None, region="resolved")
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
        with pytest.raises(RuntimeError, match="no 'lsf3' field"):
            WrAnalysis(mass_point=None, region="both").process(events)

    # ------------------------------------------------------------------
    # Missing lsf3 handled gracefully
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_missing_lsf3_graceful(self, *_mocks):
        """Missing lsf3 is fine for a resolved-only run; boosted requests fail loudly."""
        proc = WrAnalysis(mass_point=None, region="resolved")
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
        for mode in ("both", "boosted"):
            with pytest.raises(RuntimeError, match="no 'lsf3' field"):
                WrAnalysis(mass_point=None, region=mode).process(events)

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
        ds = output["WR3000_N1500"]
        assert "cutflow" in ds
        assert ds["cutflow"]["mumu"]["cumulative_unweighted"].values()[0] == 2.0

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
        ds = output["WR3000_N1500"]
        assert "cutflow" in ds
        assert ds["cutflow"]["mumu"]["cumulative_unweighted"].values()[0] == 3.0

    # ------------------------------------------------------------------
    # Zero-event data (simulates all events filtered by lumi mask)
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_empty_events(self, *_mocks):
        """Empty events are handled gracefully."""
        proc = WrAnalysis(mass_point=None, region="resolved")
        # apply_lumi_mask/apply_noise_filter are patched to passthroughs.
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
        # All 10 events survive object/trigger cuts; none pass mlljj > 800.
        cum = ds["cutflow"]["mumu"]["cumulative_unweighted"]
        counts = dict(zip(list(cum.axes[0]), cum.values()))
        assert counts["no_cuts"] == 10.0
        assert counts["mu_trigger"] == 10.0
        assert counts["mlljj_gt800"] == 0.0

    # ------------------------------------------------------------------
    # Systematic variations enabled
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_systematic_variations_enabled(self, *_mocks):
        """Systematic variations (lumi + pileup) run end-to-end."""
        proc = WrAnalysis(mass_point=None, region="resolved",
                          enabled_systs=["lumi", "pileup"])
        events = _make_integration_events(n_events=5, n_muons=2, n_jets=2)

        output = proc.process(events)

        assert "DYJetsToLL_M-50" in output

    # ------------------------------------------------------------------
    # Lumi-only systematics
    # ------------------------------------------------------------------
    @_apply_sf_patches
    def test_process_lumi_systematics_only(self, *_mocks):
        """enabled_systs=['lumi'] runs end-to-end with normalized weights."""
        proc = WrAnalysis(mass_point=None, region="resolved",
                          enabled_systs=["lumi"])
        events = _make_integration_events(n_events=5, n_muons=2, n_jets=2)

        output = proc.process(events)

        ds = output["DYJetsToLL_M-50"]
        cum = ds["cutflow"]["mumu"]["cumulative"]
        # no_cuts bin: 5 events x genWeight(=1) * xsec * lumi * 1000 / sumw,
        # with xsec/sumw from _make_integration_events default metadata.
        expected = 5 * 6077.22 * float(LUMIS["RunIISummer20UL18"]) * 1000.0 / 50000.0
        assert cum.values()[0] == pytest.approx(expected)

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
        # Boosted selections ran too (mock FatJet has lsf3).
        assert "cutflow_boosted" in ds
        cum = ds["cutflow"]["mumu"]["cumulative_unweighted"]
        assert cum.values()[0] == 20.0  # no_cuts bin sees all events

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
        # All 5 events reach the electron trigger step of the ee chain.
        cum = output["DYJetsToLL_M-50"]["cutflow"]["ee"]["cumulative_unweighted"]
        counts = dict(zip(list(cum.axes[0]), cum.values()))
        assert counts["no_cuts"] == 5.0
        assert counts["e_trigger"] == 5.0

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


# ---------------------------------------------------------------------------
# MET noise filters
# ---------------------------------------------------------------------------

_COMMON_NOISE_FLAGS = (
    "goodVertices", "globalSuperTightHalo2016Filter",
    "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter", "eeBadScFilter",
)


def _make_noise_filter_events(flag_values, *, run=1, met_pt=None, met_phi=None, jets=None):
    """Awkward events with a ``Flag`` record plus the run/event/Jet/MET fields
    that ``ecal_bad_calib_filter`` reads.  *flag_values* maps flag name to a
    list of booleans (one per event)."""
    n = len(next(iter(flag_values.values())))
    if jets is None:
        jets = {"pt": [[10.0]] * n, "eta": [[0.0]] * n, "phi": [[0.0]] * n,
                "neEmEF": [[0.0]] * n, "chEmEF": [[0.0]] * n}
    return ak.zip({
        "Flag": ak.zip({k: ak.Array(v) for k, v in flag_values.items()}),
        "run": ak.Array([run] * n),
        "event": ak.Array(list(range(n))),
        "Jet": ak.zip({k: ak.Array(v) for k, v in jets.items()}),
        "MET": ak.zip({
            "pt": ak.Array(met_pt if met_pt is not None else [0.0] * n),
            "phi": ak.Array(met_phi if met_phi is not None else [0.0] * n),
        }),
    }, depth_limit=1)


class TestNoiseFilter:
    """apply_noise_filter is applied to every sample type, signal included."""

    @staticmethod
    def _flags(n, **overrides):
        flags = {k: [True] * n for k in _COMMON_NOISE_FLAGS}
        flags.update(overrides)
        return flags

    def test_signal_is_filtered_like_background(self):
        proc = WrAnalysis(mass_point="WR2000_N1000", region="resolved")
        flags = self._flags(3, goodVertices=[True, False, True], ecalBadCalibFilter=[True] * 3)
        events = _make_noise_filter_events(flags)

        out = proc.apply_noise_filter(events, "RunIII2024Summer24")

        assert ak.to_list(out.event) == [0, 2]

    def test_ul18_requires_hbhe_and_ecal_flags(self):
        flags = self._flags(
            4,
            HBHENoiseFilter=[True, False, True, True],
            HBHENoiseIsoFilter=[True, True, False, True],
            ecalBadCalibFilter=[True, True, True, False],
        )
        events = _make_noise_filter_events(flags)

        out = WrAnalysis(mass_point=None).apply_noise_filter(events, "RunIISummer20UL18")

        assert ak.to_list(out.event) == [0]

    def test_optional_flags_applied_when_present(self):
        flags = self._flags(2, ecalBadCalibFilter=[True, True], hfNoisyHitsFilter=[True, False])
        events = _make_noise_filter_events(flags)

        out = WrAnalysis(mass_point=None).apply_noise_filter(events, "RunIII2024Summer24")

        assert ak.to_list(out.event) == [0]

    def test_missing_required_flag_passes_with_warning(self, caplog):
        import logging
        import wrcoffea.analyzer as analyzer_module

        flags = self._flags(2, ecalBadCalibFilter=[True, True])
        del flags["eeBadScFilter"]
        events = _make_noise_filter_events(flags)
        analyzer_module._WARN_ONCE.clear()

        with caplog.at_level(logging.WARNING, logger="wrcoffea.analyzer"):
            out = WrAnalysis(mass_point=None).apply_noise_filter(events, "RunIII2024Summer24")

        assert ak.to_list(out.event) == [0, 1]
        assert "eeBadScFilter" in caplog.text

    def test_run3_2022_applies_manual_ecal_veto(self):
        # Event 0: EM-dominated jet in the affected eta/phi window, back-to-back
        # with MET > 100 GeV, in the affected run range -> rejected.
        # Event 1: same jet but outside the eta window -> kept.
        flags = self._flags(2)
        jets = {"pt": [[80.0], [80.0]], "eta": [[-0.3], [1.0]], "phi": [[-2.0], [-2.0]],
                "neEmEF": [[0.95], [0.95]], "chEmEF": [[0.0], [0.0]]}
        events = _make_noise_filter_events(
            flags, run=363000, met_pt=[150.0, 150.0], met_phi=[1.2, 1.2], jets=jets,
        )

        out = WrAnalysis(mass_point=None).apply_noise_filter(events, "Run3Summer22")

        assert ak.to_list(out.event) == [1]


# ---------------------------------------------------------------------------
# Boosted SFs are evaluated on the selected candidates
# ---------------------------------------------------------------------------

def _cand(pt, pdg):
    """One boosted lepton candidate record."""
    return {"pt": pt, "eta": 0.1, "phi": 0.2, "mass": 0.0, "charge": -1,
            "pdgId": pdg, "flavor": "muon" if abs(pdg) == 13 else "electron",
            "deltaEtaSC": 0.0}


def _optional_candidates(specs):
    """Per-event optional candidate array from [(pt, pdgId) or None, ...]."""
    return ak.Array([_cand(*spec) if spec else None for spec in specs])


class TestBoostedSFCandidates:
    """_fill_boosted must hand the region's own lead/sublead objects to the SF code."""

    def test_candidates_by_flavor_splits_and_keeps_alignment(self):
        cand = _optional_candidates([(100.0, 11), (90.0, -13), None])
        mu, e = WrAnalysis._candidates_by_flavor(cand)
        assert ak.to_list(ak.num(mu)) == [0, 1, 0]
        assert ak.to_list(ak.num(e)) == [1, 0, 0]
        assert ak.to_list(ak.flatten(mu.pt)) == [90.0]
        assert ak.to_list(ak.flatten(e.pt)) == [100.0]

    def test_fill_boosted_uses_region_candidates(self):
        from coffea.analysis_tools import PackedSelection
        from wrcoffea.analysis_config import (
            SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
            SEL_ATLEAST1AK8_DPHI_GT2, SEL_AK8JETS_WITH_LSF,
            SEL_MUMU_DYCR, SEL_EE_DYCR, SEL_MUMU_SR, SEL_EE_SR, SEL_EMU_CR, SEL_MUE_CR,
            SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED,
        )
        n = 3
        # ev0: lead e; SF partner e; DY partner e; OF partner mu
        # ev1: lead mu; SF partner mu; DY partner mu; OF partner e
        # ev2: nothing selected
        tight_lep = _optional_candidates([(120.0, 11), (110.0, 13), None])
        sf_cand = _optional_candidates([(70.0, -11), (65.0, -13), None])
        dy_cand = _optional_candidates([(60.0, -11), (55.0, -13), None])
        of_cand = _optional_candidates([(80.0, -13), (75.0, -11), None])
        ak8 = ak.Array([{"pt": 300.0, "eta": 0.0, "phi": 1.0, "mass": 50.0}] * n)

        sel = PackedSelection()
        for name in (SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
                     SEL_ATLEAST1AK8_DPHI_GT2, SEL_AK8JETS_WITH_LSF,
                     SEL_MUMU_DYCR, SEL_EE_DYCR, SEL_MUMU_SR, SEL_EE_SR, SEL_EMU_CR, SEL_MUE_CR,
                     SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED):
            sel.add(name, np.ones(n, dtype=bool))
        payload = (sel, tight_lep, ak8, dy_cand, ak8, of_cand, sf_cand)

        captured = {}

        def fake_sfs(region, syst_weights, era, tight_muons, tight_electrons,
                     loose_muons=None, loose_electrons=None, *, is_mc):
            captured[region] = {
                "tight_mu": ak.to_list(ak.flatten(tight_muons.pt)),
                "tight_e": ak.to_list(ak.flatten(tight_electrons.pt)),
                "loose_mu": ak.to_list(ak.flatten(loose_muons.pt)),
                "loose_e": ak.to_list(ak.flatten(loose_electrons.pt)),
                "n_tight": ak.to_list(ak.num(tight_muons) + ak.num(tight_electrons)),
            }
            return dict(syst_weights)

        proc = WrAnalysis(mass_point=None, region="boosted")
        syst_weights = {"Nominal": np.ones(n)}
        with patch.object(proc, "_lepton_sfs_for_region", side_effect=fake_sfs), \
             patch("wrcoffea.analyzer.fill_boosted_histograms"), \
             patch("wrcoffea.analyzer.fill_boosted_cutflows"):
            proc._fill_boosted({}, payload, "DYJets", None, syst_weights, "RunIII2024Summer24", is_mc=True)

        # Lead lepton is the same object for every region: e in ev0, mu in ev1, none in ev2.
        for region, got in captured.items():
            assert got["tight_e"] == [120.0], region
            assert got["tight_mu"] == [110.0], region
            assert got["n_tight"] == [1, 1, 0], region

        # Sub-leading object depends on the region kind.
        assert captured["wr_ee_boosted_sr"]["loose_e"] == [70.0]
        assert captured["wr_ee_boosted_sr"]["loose_mu"] == [65.0]
        assert captured["wr_ee_boosted_dy_cr"]["loose_e"] == [60.0]
        assert captured["wr_mumu_boosted_dy_cr"]["loose_mu"] == [55.0]
        assert captured["wr_emu_boosted_flavor_cr"]["loose_mu"] == [80.0]
        assert captured["wr_emu_boosted_flavor_cr"]["loose_e"] == [75.0]
        assert set(captured) == {
            "wr_mumu_boosted_dy_cr", "wr_mumu_boosted_sr", "wr_ee_boosted_dy_cr",
            "wr_ee_boosted_sr", "wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr",
        }
