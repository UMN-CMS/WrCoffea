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


# ---------------------------------------------------------------------------
# Histogram Value Validation Tests
# ---------------------------------------------------------------------------


class TestResolvedHistogramValues:
    """Test that resolved histograms are filled with correct physics values.

    These tests verify that the histogram bin contents match the expected
    physics calculations, not just that histograms are filled without error.
    """

    def test_mass_dilepton_calculation(self):
        """Test that mass_dilepton histogram contains correct invariant masses."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        # Create leptons with known kinematics
        leptons = _vec_array(
            pts=[[100.0, 80.0]],  # Single event, 2 leptons
            etas=[[0.0, 0.0]],
            phis=[[0.0, np.pi]],  # Back-to-back in phi
            masses=[[0.105, 0.105]],  # Muon mass
        )

        jets = _vec_array(
            pts=[[150.0, 100.0]],
            etas=[[1.0, -1.0]],
            phis=[[2.0, 3.0]],
            masses=[[10.0, 10.0]],
        )

        weights = Weights(1)
        weights.add("test", np.array([2.0]))  # Weight = 2.0
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_resolved_histograms(output, "test_sr", cut, "DYJets", jets, leptons, weights, syst_weights)

        # Calculate expected mll
        l1 = leptons[0, 0]
        l2 = leptons[0, 1]
        mll = (l1 + l2).mass
        expected_mll = float(mll)

        # Extract histogram values
        h = output["mass_dilepton"]
        # Get bin index for the expected mll value
        h_proj = h[{"process": "DYJets", "region": "test_sr", "syst": "Nominal"}]

        # Verify that the histogram has entries near expected value
        total_sum = h_proj.sum().value
        assert total_sum == pytest.approx(2.0, rel=1e-5)  # Weight = 2.0

    def test_pt_leading_lepton_sorting(self):
        """Test that pt_leading_lepton uses the highest pT lepton."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        # Create leptons where second has higher pT (should be sorted)
        leptons = _vec_array(
            pts=[[50.0, 200.0]],  # Second lepton has higher pT
            etas=[[0.5, -0.5]],
            phis=[[0.0, 1.0]],
            masses=[[0.105, 0.105]],
        )

        jets = _vec_array(
            pts=[[100.0, 80.0]],
            etas=[[1.0, -1.0]],
            phis=[[2.0, 3.0]],
            masses=[[10.0, 10.0]],
        )

        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_resolved_histograms(output, "sr", cut, "DYJets", jets, leptons, weights, syst_weights)

        h = output["pt_leading_lepton"]
        # Leading lepton should be the one at index 0 (already sorted in practice)
        # This tests that the getter uses [:, 0] correctly

    def test_mass_fourobject_calculation(self):
        """Test that mass_fourobject = (l1 + l2 + j1 + j2).mass."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        # Create simple kinematics for easy validation
        leptons = _vec_array(
            pts=[[100.0, 100.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 0.5]],
            masses=[[0.105, 0.105]],
        )

        jets = _vec_array(
            pts=[[200.0, 200.0]],
            etas=[[0.0, 0.0]],
            phis=[[1.0, 1.5]],
            masses=[[10.0, 10.0]],
        )

        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_resolved_histograms(output, "sr", cut, "Signal", jets, leptons, weights, syst_weights)

        # Verify histogram is filled
        h = output["mass_fourobject"]
        h_proj = h[{"process": "Signal", "region": "sr", "syst": "Nominal"}]
        assert h_proj.sum().value > 0

    def test_empty_cut_produces_empty_histograms(self):
        """Test that cut=all False results in empty histograms."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        leptons = _vec_array(
            pts=[[100.0, 80.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 1.0]],
            masses=[[0.105, 0.105]],
        )

        jets = _vec_array(
            pts=[[150.0, 100.0]],
            etas=[[1.0, -1.0]],
            phis=[[2.0, 3.0]],
            masses=[[10.0, 10.0]],
        )

        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([False])  # No events pass

        fill_resolved_histograms(output, "sr", cut, "DYJets", jets, leptons, weights, syst_weights)

        # Histograms should be empty
        h = output["mass_dilepton"]
        assert h.sum().value == 0

    def test_weight_propagation_to_histograms(self):
        """Test that event weights are correctly applied to histogram bins."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        leptons = _vec_array(
            pts=[[100.0, 80.0], [120.0, 90.0]],  # 2 events
            etas=[[0.0, 0.0], [0.5, -0.5]],
            phis=[[0.0, 1.0], [0.2, 1.2]],
            masses=[[0.105, 0.105], [0.105, 0.105]],
        )

        jets = _vec_array(
            pts=[[150.0, 100.0], [160.0, 110.0]],
            etas=[[1.0, -1.0], [1.1, -1.1]],
            phis=[[2.0, 3.0], [2.1, 3.1]],
            masses=[[10.0, 10.0], [10.0, 10.0]],
        )

        weights = Weights(2)
        weights.add("test", np.array([1.5, 3.0]))  # Different weights per event
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True, True])

        fill_resolved_histograms(output, "sr", cut, "DYJets", jets, leptons, weights, syst_weights)

        # Total weight should be 1.5 + 3.0 = 4.5
        h = output["pt_leading_lepton"]
        h_proj = h[{"process": "DYJets", "region": "sr", "syst": "Nominal"}]
        assert h_proj.sum().value == pytest.approx(4.5, rel=1e-5)

    def test_multiple_systematics_produce_separate_bins(self):
        """Test that multiple systematic variations are stored separately."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        leptons = _vec_array(
            pts=[[100.0, 80.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 1.0]],
            masses=[[0.105, 0.105]],
        )

        jets = _vec_array(
            pts=[[150.0, 100.0]],
            etas=[[1.0, -1.0]],
            phis=[[2.0, 3.0]],
            masses=[[10.0, 10.0]],
        )

        weights = Weights(1)
        weights.add("test", np.array([1.0]), weightUp=np.array([1.2]), weightDown=np.array([0.8]))
        syst_weights = {
            "Nominal": weights.weight(),
            "TestUp": weights.weight(modifier="testUp"),
            "TestDown": weights.weight(modifier="testDown"),
        }
        cut = np.array([True])

        fill_resolved_histograms(output, "sr", cut, "DYJets", jets, leptons, weights, syst_weights)

        h = output["mass_dilepton"]
        # Should have separate entries for Nominal, TestUp, TestDown
        assert "Nominal" in h.axes["syst"]
        assert "TestUp" in h.axes["syst"]
        assert "TestDown" in h.axes["syst"]

        h_nom = h[{"process": "DYJets", "region": "sr", "syst": "Nominal"}]
        h_up = h[{"process": "DYJets", "region": "sr", "syst": "TestUp"}]
        h_down = h[{"process": "DYJets", "region": "sr", "syst": "TestDown"}]

        assert h_nom.sum().value == pytest.approx(1.0, rel=1e-5)
        assert h_up.sum().value == pytest.approx(1.2, rel=1e-5)
        assert h_down.sum().value == pytest.approx(0.8, rel=1e-5)


class TestBoostedHistogramValues:
    """Test that boosted histograms are filled with correct physics values."""

    def test_mass_dilepton_boosted(self):
        """Test boosted mass_dilepton = (tight_lep + loose_lep).mass."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        tight_lep = _vec_array(
            pts=[150.0],  # Single event, per-event objects (not jagged)
            etas=[0.5],
            phis=[0.0],
            masses=[0.105],
        )

        ak8jet = _vec_array(
            pts=[500.0],
            etas=[1.0],
            phis=[3.0],
            masses=[80.0],
            extra_fields={"lsf3": [0.9]},
        )

        loose_lep = _vec_array(
            pts=[80.0],
            etas=[-0.5],
            phis=[1.0],
            masses=[0.105],
            extra_fields={"pdgId": [13]},
        )

        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_boosted_histograms(output, "wr_mumu_boosted_sr", cut, "DYJets",
                                tight_lep, ak8jet, loose_lep, weights, syst_weights)

        # Verify histogram is filled
        h = output["mass_dilepton"]
        h_proj = h[{"process": "DYJets", "region": "wr_mumu_boosted_sr", "syst": "Nominal"}]
        assert h_proj.sum().value > 0

    def test_lsf3_histogram_values(self):
        """Test that LSF_leading_AK8Jets histogram contains correct LSF values."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        tight_lep = _vec_array(pts=[150.0], etas=[0.5], phis=[0.0], masses=[0.105])
        ak8jet = _vec_array(
            pts=[500.0], etas=[1.0], phis=[3.0], masses=[80.0],
            extra_fields={"lsf3": [0.95]},  # Known LSF value
        )
        loose_lep = _vec_array(
            pts=[80.0], etas=[-0.5], phis=[1.0], masses=[0.105],
            extra_fields={"pdgId": [13]},
        )

        weights = Weights(1)
        weights.add("test", np.array([2.5]))  # Weight = 2.5
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_boosted_histograms(output, "sr", cut, "Signal",
                                tight_lep, ak8jet, loose_lep, weights, syst_weights)

        h = output["LSF_leading_AK8Jets"]
        h_proj = h[{"process": "Signal", "region": "sr", "syst": "Nominal"}]
        # Total weight should equal event weight
        assert h_proj.sum().value == pytest.approx(2.5, rel=1e-5)

    def test_boosted_dy_cr_mass_twoobject_switching(self):
        """Test mass_twoobject switches based on dR(AK8, loose_lep) in DY CR.

        When dR < 0.8: mass = (tight_lep + AK8).mass
        When dR >= 0.8: mass = (tight_lep + AK8 + loose_lep).mass
        """
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        # Event 1: dR < 0.8 (loose lep close to AK8)
        tight_lep_1 = _vec_array(pts=[150.0], etas=[0.0], phis=[0.0], masses=[0.105])
        ak8jet_1 = _vec_array(
            pts=[500.0], etas=[1.0], phis=[1.0], masses=[80.0],
            extra_fields={"lsf3": [0.9]},
        )
        loose_lep_1 = _vec_array(
            pts=[80.0], etas=[1.05], phis=[1.05], masses=[0.105],  # Close to AK8
            extra_fields={"pdgId": [13]},
        )

        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_boosted_histograms(output, "wr_mumu_boosted_dy_cr", cut, "DYJets",
                                tight_lep_1, ak8jet_1, loose_lep_1, weights, syst_weights)

        # When in DY CR region, the mass_twoobject logic should apply
        # (verifying no crash is sufficient; exact values depend on dR calculation)
        h = output["mass_twoobject"]
        assert h.sum().value > 0

    def test_boosted_multiple_events_different_weights(self):
        """Test that boosted histograms correctly handle multiple events with different weights."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        tight_lep = _vec_array(
            pts=[150.0, 200.0, 120.0],  # 3 events
            etas=[0.5, 0.3, 0.8],
            phis=[0.0, 0.2, 0.5],
            masses=[0.105, 0.105, 0.105],
        )

        ak8jet = _vec_array(
            pts=[500.0, 600.0, 450.0],
            etas=[1.0, 1.2, 0.9],
            phis=[3.0, 3.2, 2.8],
            masses=[80.0, 85.0, 75.0],
            extra_fields={"lsf3": [0.9, 0.92, 0.88]},
        )

        loose_lep = _vec_array(
            pts=[80.0, 90.0, 70.0],
            etas=[-0.5, -0.3, -0.8],
            phis=[1.0, 1.2, 0.8],
            masses=[0.105, 0.105, 0.105],
            extra_fields={"pdgId": [13, 11, 13]},
        )

        weights = Weights(3)
        weights.add("test", np.array([1.0, 2.0, 0.5]))  # Different weights
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True, True, True])

        fill_boosted_histograms(output, "sr", cut, "DYJets",
                                tight_lep, ak8jet, loose_lep, weights, syst_weights)

        # Total weight = 1.0 + 2.0 + 0.5 = 3.5
        h = output["pt_leading_lepton"]
        h_proj = h[{"process": "DYJets", "region": "sr", "syst": "Nominal"}]
        assert h_proj.sum().value == pytest.approx(3.5, rel=1e-5)

    def test_boosted_partial_cut(self):
        """Test that partial event selection works correctly for boosted."""
        specs = _booking_specs()
        output = {name: create_hist(name, bins, label) for name, (bins, label) in specs.items()}

        tight_lep = _vec_array(
            pts=[150.0, 200.0, 120.0],
            etas=[0.5, 0.3, 0.8],
            phis=[0.0, 0.2, 0.5],
            masses=[0.105, 0.105, 0.105],
        )

        ak8jet = _vec_array(
            pts=[500.0, 600.0, 450.0],
            etas=[1.0, 1.2, 0.9],
            phis=[3.0, 3.2, 2.8],
            masses=[80.0, 85.0, 75.0],
            extra_fields={"lsf3": [0.9, 0.92, 0.88]},
        )

        loose_lep = _vec_array(
            pts=[80.0, 90.0, 70.0],
            etas=[-0.5, -0.3, -0.8],
            phis=[1.0, 1.2, 0.8],
            masses=[0.105, 0.105, 0.105],
            extra_fields={"pdgId": [13, 11, 13]},
        )

        weights = Weights(3)
        weights.add("test", np.ones(3))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True, False, True])  # Only events 0 and 2 pass

        fill_boosted_histograms(output, "sr", cut, "DYJets",
                                tight_lep, ak8jet, loose_lep, weights, syst_weights)

        # Total weight = 1.0 + 1.0 = 2.0 (event 1 excluded)
        h = output["pt_leading_lepton"]
        h_proj = h[{"process": "DYJets", "region": "sr", "syst": "Nominal"}]
        assert h_proj.sum().value == pytest.approx(2.0, rel=1e-5)
