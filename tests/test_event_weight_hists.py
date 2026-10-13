"""Tests for the MC event-weight histograms.

``event_weights/<region>/{positive,negative}`` count events vs log10|w| of the
region's nominal weight; ``event_weights/<region>/components/<name>`` hold one
distribution per factor of that weight (generator weight and normalization on
a log10 axis, multiplicative corrections and their product on a linear axis).
"""

from contextlib import ExitStack
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
import uproot

from wrcoffea.analysis_config import DY_ZPT_REWEIGHT_JSONS
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.histograms import (
    CORRECTIONS_COMPONENT, EVENT_WEIGHT_AXIS, GENWEIGHT_COMPONENT, NORMALIZATION_COMPONENT,
    WEIGHT_FACTOR_AXIS, fill_event_weights,
)
from wrcoffea.save_hists import EVENT_WEIGHTS_DIR, _save_cutflows

from tests.test_analyzer import _SF_PATCHES, _make_integration_events


def _counts(h):
    """Total entries including flow bins."""
    return h.sum(flow=True).value


def _filled_at(h, axis, value):
    """Entries in the bin of *axis* that contains *value*."""
    return h.values()[h.axes[axis].index(value)]


# ---------------------------------------------------------------------------
# fill_event_weights
# ---------------------------------------------------------------------------

class TestFillEventWeights:

    def test_total_split_by_sign_with_unit_weights(self):
        acc = {}
        w = np.array([0.5, -2.0, 0.0, 1e-9, 10.0, -0.1])
        cut = np.array([True, True, True, True, True, False])
        fill_event_weights(acc, "wr_ee_resolved_sr", cut, w)

        hists = acc["wr_ee_resolved_sr"]
        assert set(hists) == {"positive", "negative"}
        pos, neg = hists["positive"], hists["negative"]
        # w == 0 is skipped; 1e-9 is below the axis and lands in the underflow.
        assert _counts(pos) == 3
        assert _counts(neg) == 1
        assert pos.sum(flow=True).variance == 3  # unit-weight fills
        assert pos.values(flow=True)[0] == 1
        assert _filled_at(pos, EVENT_WEIGHT_AXIS, np.log10(0.5)) == 1
        assert _filled_at(neg, EVENT_WEIGHT_AXIS, np.log10(2.0)) == 1

    def test_empty_region_is_booked(self):
        acc = {}
        factors = {GENWEIGHT_COMPONENT: np.ones(3), NORMALIZATION_COMPONENT: np.ones(3),
                   "pileup": np.ones(3)}
        fill_event_weights(acc, "wr_mumu_boosted_sr", np.zeros(3, dtype=bool), np.ones(3), factors)

        hists = acc["wr_mumu_boosted_sr"]
        assert _counts(hists["positive"]) == _counts(hists["negative"]) == 0
        assert set(hists["components"]) == {
            "genweight_positive", "genweight_negative", "normalization", "pileup", CORRECTIONS_COMPONENT,
        }
        assert all(_counts(h) == 0 for h in hists["components"].values())

    def test_without_factors_no_components(self):
        acc = {}
        fill_event_weights(acc, "r", np.ones(2, dtype=bool), np.ones(2))
        assert "components" not in acc["r"]

    def test_components_axes_and_corrections_product(self):
        acc = {}
        cut = np.array([True, True, True, False])
        factors = {
            GENWEIGHT_COMPONENT: np.array([2.0, -2.0, 400.0, 5.0]),
            NORMALIZATION_COMPONENT: np.full(4, 1e-3),
            "pileup": np.array([1.2, 0.8, 1.0, 2.5]),
            "MuonIdSf": np.array([0.95, 0.95, 0.9, 0.95]),
        }
        weights = np.prod(np.stack(list(factors.values())), axis=0)
        fill_event_weights(acc, "r", cut, weights, factors)

        comps = acc["r"]["components"]
        assert set(comps) == {"genweight_positive", "genweight_negative", "normalization",
                              "pileup", "MuonIdSf", CORRECTIONS_COMPONENT}
        assert comps["genweight_positive"].axes[0].name == EVENT_WEIGHT_AXIS
        assert _counts(comps["genweight_positive"]) == 2
        assert _counts(comps["genweight_negative"]) == 1
        assert _filled_at(comps["genweight_positive"], EVENT_WEIGHT_AXIS, np.log10(400.0)) == 1
        assert _filled_at(comps["normalization"], EVENT_WEIGHT_AXIS, -3.0 + 1e-6) == 3

        for name in ("pileup", "MuonIdSf", CORRECTIONS_COMPONENT):
            assert comps[name].axes[0].name == WEIGHT_FACTOR_AXIS
            assert _counts(comps[name]) == 3
        # corrections = pileup * MuonIdSf per event, not the product of the means.
        corr = comps[CORRECTIONS_COMPONENT]
        for value in (1.2 * 0.95, 0.8 * 0.95, 1.0 * 0.9):
            assert _filled_at(corr, WEIGHT_FACTOR_AXIS, value) >= 1

        # The total is still filled from the region weight.
        assert _counts(acc["r"]["positive"]) == 2
        assert _counts(acc["r"]["negative"]) == 1

    def test_fills_accumulate_across_chunks(self):
        acc = {}
        factors = {GENWEIGHT_COMPONENT: np.ones(2), "pileup": np.ones(2)}
        for _ in range(2):
            fill_event_weights(acc, "r", np.ones(2, dtype=bool), np.ones(2), factors)
        assert _counts(acc["r"]["positive"]) == 4
        assert _counts(acc["r"]["components"]["pileup"]) == 4

    def test_saved_layout(self, tmp_path):
        acc = {}
        fill_event_weights(acc, "wr_ee_resolved_sr", np.ones(2, dtype=bool), np.array([1.0, -1.0]),
                           {GENWEIGHT_COMPONENT: np.array([1.0, -1.0]), "pileup": np.ones(2)})
        path = tmp_path / "out.root"
        with uproot.recreate(path) as f:
            _save_cutflows(f, acc, f"/{EVENT_WEIGHTS_DIR}")
        with uproot.open(path) as f:
            base = f"{EVENT_WEIGHTS_DIR}/wr_ee_resolved_sr"
            for key in ("positive", "negative", "components/genweight_positive",
                        "components/genweight_negative", "components/pileup",
                        f"components/{CORRECTIONS_COMPONENT}"):
                assert f"{base}/{key}" in f


# ---------------------------------------------------------------------------
# Analyzer: where the factors come from
# ---------------------------------------------------------------------------

class _Events:
    def __init__(self, gen_weight):
        self.genWeight = ak.Array(gen_weight)

    def __len__(self):
        return len(self.genWeight)


def _pu(n, values):
    nom = np.asarray(values, dtype=np.float64)
    return nom, nom * 1.1, nom * 0.9


class TestAnalyzerFactors:

    @pytest.mark.parametrize("dy_zpt_reweight", [False, True])
    def test_base_factors_multiply_to_base_weight(self, dy_zpt_reweight):
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="both", dy_zpt_reweight=dy_zpt_reweight)
        events = _Events([1.5, -0.5, 2.0])
        meta = {"era": "RunIII2024Summer24", "xsec": 12.3, "genEventSumw": 4567.0, "sample": "s",
                "physics_group": "tt_tW"}
        with patch("wrcoffea.analyzer.pileup_weight", return_value=_pu(3, [1.1, 0.7, 1.3])):
            weights, _ = proc.build_event_weights(events, meta, is_mc=True)
            factors = proc._event_weight_factors(events, meta, weights)

        # with the reweight on, an era with a DY Z-pT payload gives every MC dataset the factor
        # (identity for non-DY); it is off by default since 2026-10-12
        zpt = {"dy_zpt"} if dy_zpt_reweight and meta["era"] in DY_ZPT_REWEIGHT_JSONS else set()
        assert set(factors) == {GENWEIGHT_COMPONENT, NORMALIZATION_COMPONENT, "pileup"} | zpt
        if zpt:
            np.testing.assert_allclose(factors["dy_zpt"], 1.0)
        np.testing.assert_allclose(factors[GENWEIGHT_COMPONENT], [1.5, -0.5, 2.0])
        np.testing.assert_allclose(factors["pileup"], [1.1, 0.7, 1.3])
        np.testing.assert_allclose(np.prod(np.stack(list(factors.values())), axis=0),
                                   weights.weight(), rtol=1e-12)

    def test_lepton_sfs_recorded_per_component(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="both")
        n = 2
        muons = ak.Array([[{"pt": 100.0, "eta": 0.0}]] * n)
        sfs = {"reco": (np.full(n, 0.99), np.ones(n), np.ones(n)),
               "id": (np.full(n, 0.95), np.ones(n), np.ones(n)),
               "iso": (np.full(n, 0.98), np.ones(n), np.ones(n))}
        trig = (np.array([0.9, 0.8]), np.ones(n), np.ones(n))
        base = {"Nominal": np.array([2.0, 3.0])}
        factors = {}
        with patch("wrcoffea.analyzer.muon_sf", return_value=sfs), \
             patch("wrcoffea.analyzer.muon_trigger_sf", return_value=trig):
            out = proc._lepton_sfs_for_region("wr_mumu_resolved_sr", base, "RunIII2024Summer24",
                                              muons, None, is_mc=True, factors=factors)

        assert set(factors) == {"MuonRecoSf", "MuonIdSf", "MuonIsoSf", "MuonTrigSf"}
        np.testing.assert_allclose(factors["MuonTrigSf"], [0.9, 0.8])
        np.testing.assert_allclose(base["Nominal"] * np.prod(np.stack(list(factors.values())), axis=0),
                                   out["Nominal"])

    def test_data_records_no_factors(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="both")
        factors = {}
        proc._lepton_sfs_for_region("wr_mumu_resolved_sr", {"Nominal": np.ones(2)}, "RunIII2024Summer24",
                                    None, None, is_mc=False, factors=factors)
        assert factors == {}

    @pytest.mark.parametrize("payload", [None, {"nominal": np.array([1.2, 0.8]), "variations": {}}])
    def test_dy_reshape_recorded_when_applied(self, payload):
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="both", dy_zpt_reweight=True, dy_reshape=True)
        factors = {}
        with patch("wrcoffea.analyzer.dy_reshape_weight", return_value=payload):
            proc._dy_reshape_for_region({"Nominal": np.ones(2)}, "resolved", np.full(2, 100.0),
                                        "RunIISummer20UL18", True, True, factors=factors)
        if payload is None:
            assert factors == {}
        else:
            np.testing.assert_allclose(factors["dy_reshape"], [1.2, 0.8])


# ---------------------------------------------------------------------------
# process(): the factors of every region multiply to the filled weight
# ---------------------------------------------------------------------------

def _mumu_resolved_sr_events(n, *, metadata=None, include_genweight=True):
    """Events whose two muons and two jets pass the resolved mumu SR."""
    events = _make_integration_events(n_events=n, n_muons=2, n_jets=2,
                                      metadata=metadata, include_genweight=include_genweight)
    muons = events.Muon
    muons = ak.with_field(muons, ak.Array([[300.0, 250.0]] * n), "pt")
    muons = ak.with_field(muons, ak.zeros_like(muons.eta), "eta")
    muons = ak.with_field(muons, ak.Array([[0.0, np.pi]] * n), "phi")
    events.Muon = muons
    events.Jet = ak.with_field(events.Jet, ak.full_like(events.Jet.pt, 300.0), "pt")
    events.Jet = ak.with_field(events.Jet, ak.zeros_like(events.Jet.eta), "eta")
    events.Jet = ak.with_field(events.Jet, ak.Array([[np.pi / 2, -np.pi / 2]] * n), "phi")
    return events


def _run_process(proc, events, overrides=()):
    """process() under the standard correction mocks plus *overrides*;
    returns (output, recorded fill_event_weights calls)."""
    calls = []
    real_fill = fill_event_weights

    def recording_fill(acc, region, cut, weights, factors=None):
        calls.append((region, np.asarray(cut, dtype=bool), np.asarray(weights), factors))
        return real_fill(acc, region, cut, weights, factors)

    with ExitStack() as stack:
        for target, replacement in _SF_PATCHES.items():
            stack.enter_context(patch(target, new=replacement))
        for target, replacement in overrides:
            stack.enter_context(patch(target, new=replacement))
        stack.enter_context(patch("wrcoffea.analyzer.fill_event_weights", new=recording_fill))
        output = proc.process(events)
    return output, calls


class TestProcessEventWeights:

    def test_mc_factors_multiply_to_region_weight(self):
        n = 3
        events = _mumu_resolved_sr_events(n)
        events.genWeight = ak.Array([1.0, -2.0, 0.5])

        def pileup(evts, era):
            return _pu(len(evts), np.linspace(0.8, 1.2, len(evts)))

        def muon_sf(muons, era):
            m = len(muons)
            return {"reco": (np.full(m, 0.99), np.ones(m), np.ones(m)),
                    "id": (np.full(m, 0.97), np.ones(m), np.ones(m)),
                    "iso": (np.linspace(0.9, 1.0, m), np.ones(m), np.ones(m))}

        def muon_trigger_sf(muons, era):
            m = len(muons)
            return np.full(m, 0.95), np.ones(m), np.ones(m)

        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=[])
        output, calls = _run_process(proc, events, overrides=(
            ("wrcoffea.analyzer.pileup_weight", pileup),
            ("wrcoffea.analyzer.muon_sf", muon_sf),
            ("wrcoffea.analyzer.muon_trigger_sf", muon_trigger_sf),
        ))

        assert calls, "MC nominal pass must fill the event-weight histograms"
        for region, _cut, weights, factors in calls:
            assert factors is not None, region
            assert {GENWEIGHT_COMPONENT, NORMALIZATION_COMPONENT, "pileup"} <= set(factors)
            product = np.prod(np.stack([np.asarray(v, dtype=np.float64) for v in factors.values()]),
                              axis=0)
            np.testing.assert_allclose(product, weights, rtol=1e-12, err_msg=region)

        sr = output["DYJetsToLL_M-50"]["event_weights"]["wr_mumu_resolved_sr"]
        assert _counts(sr["positive"]) == 2 and _counts(sr["negative"]) == 1
        comps = sr["components"]
        assert {"MuonRecoSf", "MuonIdSf", "MuonIsoSf", "MuonTrigSf", CORRECTIONS_COMPONENT} <= set(comps)
        assert _counts(comps["genweight_positive"]) + _counts(comps["genweight_negative"]) == n

    def test_data_has_no_event_weights(self):
        events = _mumu_resolved_sr_events(
            3, metadata={"datatype": "data", "physics_group": "Muon", "sample": "Muon",
                         "era": "Run3Summer23"},
            include_genweight=False,
        )
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=[])
        output, calls = _run_process(proc, events)
        assert calls == []
        assert "event_weights" not in output["Muon"]

    def test_compute_sumw_mode_has_no_event_weights(self):
        events = _mumu_resolved_sr_events(3)
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=[], compute_sumw=True)
        output, calls = _run_process(proc, events)
        assert calls == []
        assert "event_weights" not in output["DYJetsToLL_M-50"]

    def test_shape_passes_do_not_fill_event_weights(self):
        events = _mumu_resolved_sr_events(3)
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=[])
        _, nominal_calls = _run_process(proc, events)

        events = _mumu_resolved_sr_events(3)
        shape_pass = [("JESUp", object(), "up")]
        _, calls = _run_process(proc, events, overrides=(
            ("wrcoffea.analyzer.WrAnalysis._plan_shape_variations",
             lambda self, evts, era: (shape_pass, [])),
            ("wrcoffea.analyzer.WrAnalysis._apply_shape_variation",
             staticmethod(lambda evts, source, direction: evts)),
        ))
        assert [c[0] for c in calls] == [c[0] for c in nominal_calls]
