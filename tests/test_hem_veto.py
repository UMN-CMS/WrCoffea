"""HEM 15/16 electron veto (2018 only), ported from HNWRAnalyzer::isHEMVetoEvent."""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from tests.test_analyzer import _make_integration_events
from wrcoffea.analysis_config import HEM_VETO, SEL_HEM_VETO
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.regions import RESOLVED_REGIONS

UL18 = "RunIISummer20UL18"
RUN3 = "Run3Summer22"

# Inside the window: eta < -1.25 and -1.62 < phi < -0.82.
HEM_ETA, HEM_PHI = -1.8, -1.2   # eta outside the ECAL crack so the electron survives object selection


def _cands(rows):
    """Per-event optional lepton candidates from (pdgId, eta, phi) or None."""
    def field(i):
        return ak.Array([[r[i]] if r is not None else [] for r in rows])
    return ak.firsts(ak.zip({"pdgId": field(0), "eta": field(1), "phi": field(2)}))


class TestHelper:
    def test_window_is_configured_for_ul18_only(self):
        assert set(HEM_VETO) == {UL18}
        assert HEM_VETO[UL18] == {"eta_max": -1.25, "phi_min": -1.62, "phi_max": -0.82}

    def test_electron_in_window_vetoes(self):
        lead = _cands([(11, HEM_ETA, HEM_PHI), (11, HEM_ETA, HEM_PHI), (11, 1.0, 0.0), (11, -1.0, HEM_PHI)])
        sub = _cands([(11, 1.0, 0.0), (11, 1.0, 0.0), (11, HEM_ETA, HEM_PHI), (11, HEM_ETA, 0.0)])
        ok = WrAnalysis._hem_veto_ok(UL18, lead, sub)
        # lead in window; lead in window; sub in window; both outside (eta or phi off)
        assert ok.tolist() == [False, False, False, True]

    def test_muons_and_missing_candidates_never_veto(self):
        lead = _cands([(13, HEM_ETA, HEM_PHI), (11, 1.0, 0.0)])
        sub = _cands([(11, 1.0, 0.0), None])
        assert WrAnalysis._hem_veto_ok(UL18, lead, sub).tolist() == [True, True]

    def test_window_edges(self):
        lead = _cands([(11, -1.25, HEM_PHI), (11, -1.26, -1.62), (11, -1.26, -0.82), (11, -1.26, -1.61)])
        sub = _cands([None] * 4)
        assert WrAnalysis._hem_veto_ok(UL18, lead, sub).tolist() == [True, True, True, False]

    def test_other_eras_apply_no_veto(self):
        lead = _cands([(11, HEM_ETA, HEM_PHI)])
        for era in (RUN3, "RunIII2024Summer24", None):
            assert WrAnalysis._hem_veto_ok(era, lead, lead).tolist() == [True]


class TestResolvedSelection:
    def test_resolved_regions_require_the_veto(self):
        for cuts in RESOLVED_REGIONS.values():
            assert SEL_HEM_VETO in cuts

    @pytest.mark.parametrize("era, expected", [(UL18, [False, True]), (RUN3, [True, True])])
    def test_selection_flags_hem_electrons(self, era, expected):
        proc = WrAnalysis(mass_point=None, region="resolved")
        ev = _make_integration_events(n_events=2, n_muons=0, n_electrons=2, n_jets=2,
                                      metadata={"era": era})
        # Event 0: leading electron inside the HEM window; event 1: outside.
        ev.Electron = ak.with_field(ev.Electron, ak.Array([[HEM_ETA, 1.0], [1.0, 1.0]]), "eta")
        ev.Electron = ak.with_field(ev.Electron, ak.Array([[HEM_PHI, 2.5], [2.0, 2.5]]), "phi")
        tight, _loose, lep_masks = proc.select_leptons(ev)
        ak4, _ak8, jet_masks = proc.select_jets(ev, era)
        sel = proc.resolved_selections(tight, ak4, lepton_masks=lep_masks, jet_masks=jet_masks,
                                       triggers=proc.build_trigger_masks(ev, era), era=era)
        assert np.asarray(sel.all(SEL_HEM_VETO)).tolist() == expected

    def test_default_era_applies_no_veto(self):
        proc = WrAnalysis(mass_point=None, region="resolved")
        ev = _make_integration_events(n_events=1, n_muons=0, n_electrons=2, n_jets=2)
        ev.Electron = ak.with_field(ev.Electron, ak.Array([[HEM_ETA, 1.0]]), "eta")
        ev.Electron = ak.with_field(ev.Electron, ak.Array([[HEM_PHI, 2.5]]), "phi")
        tight, _loose, lep_masks = proc.select_leptons(ev)
        ak4, _ak8, jet_masks = proc.select_jets(ev, RUN3)
        sel = proc.resolved_selections(tight, ak4, lepton_masks=lep_masks, jet_masks=jet_masks)
        assert np.asarray(sel.all(SEL_HEM_VETO)).tolist() == [True]
