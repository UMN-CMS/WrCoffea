"""Unit tests for histogram creation and filling utilities."""

import awkward as ak
import hist as hist_mod
import numpy as np
import pytest
import vector
from coffea.analysis_tools import Weights

vector.register_awkward()

# Add coffea-style delta_phi method alias (the vector library calls it deltaphi).
vector.backends.awkward.MomentumArray4D.delta_phi = vector.backends.awkward.MomentumArray4D.deltaphi

_BEHAVIOR = vector.backends.awkward.behavior

from wrcoffea.histograms import (
    RESOLVED_HIST_SPECS,
    BOOSTED_HIST_SPECS,
    _booking_specs,
    create_hist,
    fill_resolved_histograms,
    fill_boosted_histograms,
)


def _vec_array(pts, etas, phis, masses, extra_fields=None):
    """Build an awkward array with Lorentz vector behavior."""
    fields = {"pt": pts, "eta": etas, "phi": phis, "mass": masses}
    if extra_fields:
        fields.update(extra_fields)
    return ak.zip(fields, with_name="Momentum4D", behavior=_BEHAVIOR)


class TestBookingSpecs:
    def test_contains_resolved_hists(self):
        specs = _booking_specs()
        for name, _, _, _ in RESOLVED_HIST_SPECS:
            assert name in specs

    def test_contains_boosted_hists(self):
        specs = _booking_specs()
        for name, _, _, _ in BOOSTED_HIST_SPECS:
            assert name in specs

    def test_contains_count(self):
        specs = _booking_specs()
        assert "count" in specs

    def test_no_empty_specs(self):
        specs = _booking_specs()
        assert len(specs) > 0
        for name, (bins, label) in specs.items():
            assert len(bins) == 3, f"{name}: bins tuple should have 3 elements"
            assert isinstance(label, str)


class TestCreateHist:
    def test_creates_weighted_hist(self):
        h = create_hist("test", (100, 0, 100), "Test label")
        assert isinstance(h, hist_mod.Hist)
        # Should have 4 axes: process, region, syst, value
        assert len(h.axes) == 4

    def test_axes_are_growable(self):
        h = create_hist("test", (50, 0, 50), "Test")
        h.fill(process="DYJets", region="sr", syst="Nominal", test=25.0, weight=1.0)
        h.fill(process="tt_tW", region="cr", syst="Nominal", test=30.0, weight=1.0)
        assert "DYJets" in h.axes["process"]
        assert "tt_tW" in h.axes["process"]


class TestFillResolvedHistograms:
    """Test fill_resolved_histograms with mock vector objects."""

    @pytest.fixture
    def setup(self):
        """Build mock objects for a 3-event chunk with 2 leptons and 2 jets each."""
        n = 3
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        leptons = _vec_array(
            pts=[[200.0, 100.0], [150.0, 80.0], [300.0, 60.0]],
            etas=[[0.5, -0.5], [1.0, -1.0], [0.3, 0.8]],
            phis=[[0.0, 1.0], [0.5, 1.5], [0.2, 1.2]],
            masses=[[0.105, 0.105], [0.105, 0.105], [0.105, 0.105]],
        )

        jets = _vec_array(
            pts=[[150.0, 80.0], [200.0, 90.0], [120.0, 70.0]],
            etas=[[0.3, -0.3], [0.8, -0.8], [1.0, -1.0]],
            phis=[[2.0, 3.0], [2.5, 3.5], [2.2, 3.2]],
            masses=[[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
        )

        weights = Weights(n)
        weights.add("test", np.ones(n))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True, True, True])

        return output, leptons, jets, weights, syst_weights, cut

    def test_fills_without_error(self, setup):
        output, leptons, jets, weights, syst_weights, cut = setup
        fill_resolved_histograms(output, "test_region", cut, "DYJets", jets, leptons, weights, syst_weights)

    def test_histograms_have_entries(self, setup):
        output, leptons, jets, weights, syst_weights, cut = setup
        fill_resolved_histograms(output, "test_region", cut, "DYJets", jets, leptons, weights, syst_weights)
        h = output["mass_dilepton"]
        assert h.sum().value > 0

    def test_process_label_correct(self, setup):
        output, leptons, jets, weights, syst_weights, cut = setup
        fill_resolved_histograms(output, "test_region", cut, "MyProcess", jets, leptons, weights, syst_weights)
        assert "MyProcess" in output["mass_dilepton"].axes["process"]

    def test_partial_cut_reduces_entries(self, setup):
        output, leptons, jets, weights, syst_weights, cut = setup
        partial_cut = np.array([True, False, False])
        fill_resolved_histograms(output, "r1", partial_cut, "DYJets", jets, leptons, weights, syst_weights)
        h_partial = output["pt_leading_lepton"].sum().value

        output2 = {name: create_hist(name, bins, label) for name, (bins, label) in _booking_specs().items()}
        fill_resolved_histograms(output2, "r1", cut, "DYJets", jets, leptons, weights, syst_weights)
        h_full = output2["pt_leading_lepton"].sum().value

        assert h_partial < h_full


class TestFillBoostedHistograms:
    """Test fill_boosted_histograms with mock per-event vector objects."""

    @pytest.fixture
    def setup(self):
        n = 3
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        leptons = _vec_array(
            pts=[200.0, 150.0, 300.0],
            etas=[0.5, 1.0, 0.3],
            phis=[0.0, 0.5, 0.2],
            masses=[0.105, 0.105, 0.105],
        )

        ak8jets = _vec_array(
            pts=[500.0, 400.0, 600.0],
            etas=[0.3, 0.8, 1.0],
            phis=[3.0, 3.5, 3.2],
            masses=[80.0, 90.0, 85.0],
            extra_fields={"lsf3": [0.9, 0.85, 0.95]},
        )

        loose_leptons = _vec_array(
            pts=[80.0, 60.0, 90.0],
            etas=[-0.5, -1.0, 0.8],
            phis=[1.0, 1.5, 1.2],
            masses=[0.105, 0.105, 0.105],
            extra_fields={"pdgId": [13, 11, 13]},
        )

        weights = Weights(n)
        weights.add("test", np.ones(n))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True, True, True])

        return output, leptons, ak8jets, loose_leptons, weights, syst_weights, cut

    def test_fills_without_error(self, setup):
        output, leptons, ak8jets, loose, weights, syst_weights, cut = setup
        fill_boosted_histograms(output, "wr_mumu_boosted_sr", cut, "DYJets",
                                leptons, ak8jets, loose, weights, syst_weights)

    def test_histograms_have_entries(self, setup):
        output, leptons, ak8jets, loose, weights, syst_weights, cut = setup
        fill_boosted_histograms(output, "wr_mumu_boosted_sr", cut, "DYJets",
                                leptons, ak8jets, loose, weights, syst_weights)
        assert output["mass_dilepton"].sum().value > 0
        assert output["pt_leading_AK8Jets"].sum().value > 0
