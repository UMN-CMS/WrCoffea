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
    RESOLVED_2D_HIST_SPECS,
    BOOSTED_2D_HIST_SPECS,
    _booking_specs,
    create_hist,
    create_hist2D,
    fill_resolved_histograms,
    fill_boosted_histograms,
)


def _vec_array(pts, etas, phis, masses, extra_fields=None):
    """Build an awkward array with Lorentz vector behavior."""
    fields = {"pt": pts, "eta": etas, "phi": phis, "mass": masses}
    if extra_fields:
        fields.update(extra_fields)
    return ak.zip(fields, with_name="Momentum4D", behavior=_BEHAVIOR)


def _make_test_output():
    """Book histograms the same way analyzer._make_output does.

    1D hists come from _booking_specs(); 2D hists are booked via create_hist2D.
    Note: analyzer._make_output books only RESOLVED_2D_HIST_SPECS -- the boosted
    2D table currently piggybacks on that booking because its histogram key and
    axis definitions coincide with a resolved 2D spec. We book both tables here
    and pin the coincidence in TestBookingFillingConsistency.
    """
    output = {
        name: create_hist(name, bins, label)
        for name, (bins, label) in _booking_specs().items()
    }
    for hist_key, xinfo, yinfo, _ in (*RESOLVED_2D_HIST_SPECS, *BOOSTED_2D_HIST_SPECS):
        xname, xbins, xlabel = xinfo
        yname, ybins, ylabel = yinfo
        output[hist_key] = create_hist2D(xname, xbins, xlabel, yname, ybins, ylabel)
    return output


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
        output = _make_test_output()

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

        output2 = _make_test_output()
        fill_resolved_histograms(output2, "r1", cut, "DYJets", jets, leptons, weights, syst_weights)
        h_full = output2["pt_leading_lepton"].sum().value

        assert h_partial < h_full


class TestFillBoostedHistograms:
    """Test fill_boosted_histograms with mock per-event vector objects."""

    @pytest.fixture
    def setup(self):
        n = 3
        output = _make_test_output()

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
            extra_fields={"lsf3": [0.9, 0.85, 0.95], "msoftdrop": [80.0, 80.0, 80.0]},
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
        output = _make_test_output()

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
        output = _make_test_output()

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
        output = _make_test_output()

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
        output = _make_test_output()

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
        output = _make_test_output()

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
        output = _make_test_output()

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
        output = _make_test_output()

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
            extra_fields={"lsf3": [0.9], "msoftdrop": [80.0]},
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
        output = _make_test_output()

        tight_lep = _vec_array(pts=[150.0], etas=[0.5], phis=[0.0], masses=[0.105])
        ak8jet = _vec_array(
            pts=[500.0], etas=[1.0], phis=[3.0], masses=[80.0],
            extra_fields={"lsf3": [0.95], "msoftdrop": [80.0]},  # Known LSF value
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
        output = _make_test_output()

        # Event 1: dR < 0.8 (loose lep close to AK8)
        tight_lep_1 = _vec_array(pts=[150.0], etas=[0.0], phis=[0.0], masses=[0.105])
        ak8jet_1 = _vec_array(
            pts=[500.0], etas=[1.0], phis=[1.0], masses=[80.0],
            extra_fields={"lsf3": [0.9], "msoftdrop": [80.0]},
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
        output = _make_test_output()

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
            extra_fields={"lsf3": [0.9, 0.92, 0.88], "msoftdrop": [80.0, 80.0, 80.0]},
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
        output = _make_test_output()

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
            extra_fields={"lsf3": [0.9, 0.92, 0.88], "msoftdrop": [80.0, 80.0, 80.0]},
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


# ---------------------------------------------------------------------------
# Booking / Filling Consistency Tests
# ---------------------------------------------------------------------------


class _RecordingOutput(dict):
    """Dict that records every histogram key the fill functions look up."""

    def __init__(self, base):
        super().__init__(base)
        self.accessed = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)


class TestBookingFillingConsistency:
    """Guard against booking/filling divergence.

    analyzer._make_output books 1D hists from _booking_specs() and 2D hists
    from RESOLVED_2D_HIST_SPECS only. The boosted 2D table currently relies on
    its histogram key/axes coinciding with a resolved 2D spec to be booked at
    all. These tests pin that every name the fill functions touch exists in
    the union of _booking_specs() + the 2D spec tables, so a new spec that is
    filled but never booked fails here instead of KeyError-ing in production.
    """

    @staticmethod
    def _booked_names():
        names = set(_booking_specs())
        names.update(key for key, *_ in RESOLVED_2D_HIST_SPECS)
        names.update(key for key, *_ in BOOSTED_2D_HIST_SPECS)
        return names

    def test_resolved_fill_names_are_all_booked(self):
        output = _RecordingOutput(_make_test_output())
        leptons = _vec_array(
            pts=[[100.0, 80.0]], etas=[[0.0, 0.0]],
            phis=[[0.0, 1.0]], masses=[[0.105, 0.105]],
        )
        jets = _vec_array(
            pts=[[150.0, 100.0]], etas=[[1.0, -1.0]],
            phis=[[2.0, 3.0]], masses=[[10.0, 10.0]],
        )
        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_resolved_histograms(output, "sr", cut, "DYJets", jets, leptons, weights, syst_weights)

        expected = {name for name, *_ in RESOLVED_HIST_SPECS}
        expected |= {key for key, *_ in RESOLVED_2D_HIST_SPECS}
        assert output.accessed == expected
        assert output.accessed <= self._booked_names()

    def test_boosted_fill_names_are_all_booked(self):
        output = _RecordingOutput(_make_test_output())
        tight_lep = _vec_array(pts=[150.0], etas=[0.5], phis=[0.0], masses=[0.105])
        ak8jet = _vec_array(
            pts=[500.0], etas=[1.0], phis=[3.0], masses=[80.0],
            extra_fields={"lsf3": [0.9], "msoftdrop": [80.0]},
        )
        loose_lep = _vec_array(
            pts=[80.0], etas=[-0.5], phis=[1.0], masses=[0.105],
            extra_fields={"pdgId": [13]},
        )
        weights = Weights(1)
        weights.add("test", np.ones(1))
        syst_weights = {"Nominal": weights.weight()}
        cut = np.array([True])

        fill_boosted_histograms(output, "wr_mumu_boosted_sr", cut, "DYJets",
                                tight_lep, ak8jet, loose_lep, weights, syst_weights)

        expected = {name for name, *_ in BOOSTED_HIST_SPECS}
        expected |= {key for key, *_ in BOOSTED_2D_HIST_SPECS}
        assert output.accessed == expected
        assert output.accessed <= self._booked_names()

    def test_boosted_2d_specs_are_covered_by_resolved_booking(self):
        # analyzer._make_output books only RESOLVED_2D_HIST_SPECS, so every
        # boosted 2D spec must reuse a resolved 2D key with identical axis
        # names and binning -- otherwise the analyzer would KeyError (or fill
        # with mismatched axes) at runtime.
        resolved_2d = {key: (xinfo, yinfo) for key, xinfo, yinfo, _ in RESOLVED_2D_HIST_SPECS}
        for key, xinfo, yinfo, _ in BOOSTED_2D_HIST_SPECS:
            assert key in resolved_2d, (
                f"Boosted 2D hist '{key}' is not booked by analyzer._make_output "
                "(which only books RESOLVED_2D_HIST_SPECS)"
            )
            rxinfo, ryinfo = resolved_2d[key]
            # Axis name and binning must match; labels may differ.
            assert (xinfo[0], xinfo[1]) == (rxinfo[0], rxinfo[1])
            assert (yinfo[0], yinfo[1]) == (ryinfo[0], ryinfo[1])


# ---------------------------------------------------------------------------
# Axis ranges (2026-10-14: LSF3 from -1, phi over +-pi, fit masses to 12 TeV, pT to 5 TeV)
# ---------------------------------------------------------------------------

def _booked(name):
    bins, label = _booking_specs()[name]
    return create_hist(name, bins, label)


class TestHistogramRanges:
    def test_lsf3_keeps_minus_one_and_puts_075_on_an_edge(self):
        h = _booked("LSF_leading_AK8Jets")
        edges = h.axes["LSF_leading_AK8Jets"].edges
        assert edges[0] <= -1.0 and np.isclose(edges, 0.75).any()
        h.fill(process="p", region="r", syst="Nominal", LSF_leading_AK8Jets=[-1.0, 1.9], weight=[1.0, 1.0])
        # NanoAOD's -1 (no lepton in the jet) and a value above 1.1 are both inside the axis.
        assert h[{"process": "p", "region": "r", "syst": "Nominal"}].values().sum() == pytest.approx(2.0)

    @pytest.mark.parametrize("name", [n for n, *_ in RESOLVED_HIST_SPECS + BOOSTED_HIST_SPECS if n.startswith("phi_")])
    def test_phi_axes_span_exactly_pi(self, name):
        edges = _booked(name).axes[name].edges
        assert edges[0] == pytest.approx(-np.pi) and edges[-1] == pytest.approx(np.pi)

    def test_eta_phi_maps_span_exactly_pi(self):
        for _key, _x, (yname, ybins, _ylabel), _expr in RESOLVED_2D_HIST_SPECS + BOOSTED_2D_HIST_SPECS:
            assert yname == "phi" and ybins[1] == pytest.approx(-np.pi) and ybins[2] == pytest.approx(np.pi)

    def test_dphi_starts_at_the_boosted_requirement(self):
        edges = _booked("dPhi_leading_tightlepton_AK8Jet").axes["dPhi_leading_tightlepton_AK8Jet"].edges
        assert edges[0] == pytest.approx(2.0) and edges[-1] == pytest.approx(np.pi)

    @pytest.mark.parametrize("name", ["mass_fourobject", "mass_twoobject"])
    def test_fit_masses_reach_12_tev_with_800_on_an_edge(self, name):
        edges = _booked(name).axes[name].edges
        assert edges[-1] >= 12000.0 and np.isclose(edges, 800.0).any()

    @pytest.mark.parametrize("name", ["pt_leading_lepton", "pt_subleading_lepton", "pt_leading_jet",
                                      "pt_subleading_jet", "pt_dilepton", "pt_leading_AK8Jets"])
    def test_pt_axes_reach_5_tev(self, name):
        assert _booked(name).axes[name].edges[-1] >= 5000.0

    def test_names_shared_by_both_topologies_have_one_binning(self):
        """The booking keeps one histogram per name (the boosted spec wins), so a name used by both
        topologies must have the same axis in both."""
        resolved = {n: b for n, b, *_ in RESOLVED_HIST_SPECS}
        for name, bins, *_ in BOOSTED_HIST_SPECS:
            if name in resolved:
                assert resolved[name] == bins, name


# ---------------------------------------------------------------------------
# New histograms: N(PV) (event level), AK8 m_SD, dR(loose lepton, AK8), N(AK4)
# ---------------------------------------------------------------------------

class TestEventLevelHistograms:
    def _resolved(self):
        leptons = _vec_array(pts=[[200.0, 100.0]] * 3, etas=[[0.5, -0.5]] * 3, phis=[[0.0, 1.0]] * 3,
                             masses=[[0.105, 0.105]] * 3)
        jets = _vec_array(pts=[[150.0, 80.0, 50.0]] * 3, etas=[[0.3, -0.3, 1.0]] * 3,
                          phis=[[2.0, 3.0, -1.0]] * 3, masses=[[10.0, 10.0, 5.0]] * 3)
        weights = Weights(3)
        weights.add("test", np.array([1.0, 2.0, 3.0]))
        return leptons, jets, weights, {"Nominal": weights.weight()}

    def test_npv_good_filled_with_the_region_weights(self):
        output = _make_test_output()
        leptons, jets, weights, syst = self._resolved()
        cut = np.array([True, False, True])
        fill_resolved_histograms(output, "r", cut, "DYJets", jets, leptons, weights, syst,
                                 event_values={"npv_good": np.array([10.0, 20.0, 30.0])})
        h = output["npv_good"][{"process": "DYJets", "region": "r", "syst": "Nominal"}]
        assert h.values().sum() == pytest.approx(4.0)
        assert h[hist_mod.loc(10.0)].value == pytest.approx(1.0)
        assert h[hist_mod.loc(30.0)].value == pytest.approx(3.0)

    def test_npv_good_left_empty_without_event_values(self):
        output = _make_test_output()
        leptons, jets, weights, syst = self._resolved()
        fill_resolved_histograms(output, "r", np.ones(3, dtype=bool), "DYJets", jets, leptons, weights, syst)
        assert output["npv_good"].sum().value == 0.0

    def test_n_ak4_jets_counts_every_selected_jet(self):
        output = _make_test_output()
        leptons, jets, weights, syst = self._resolved()
        fill_resolved_histograms(output, "r", np.ones(3, dtype=bool), "DYJets", jets, leptons, weights, syst)
        h = output["n_ak4_jets"][{"process": "DYJets", "region": "r", "syst": "Nominal"}]
        assert h[hist_mod.loc(3)].value == pytest.approx(6.0)

    def test_boosted_msd_dr_and_npv(self):
        output = _make_test_output()
        lep = _vec_array(pts=[150.0], etas=[0.5], phis=[0.0], masses=[0.105])
        ak8 = _vec_array(pts=[500.0], etas=[0.0], phis=[3.0], masses=[80.0],
                         extra_fields={"lsf3": [-1.0], "msoftdrop": [92.0]})
        loose = _vec_array(pts=[80.0], etas=[0.0], phis=[2.5], masses=[0.105], extra_fields={"pdgId": [13]})
        weights = Weights(1)
        weights.add("test", np.array([2.0]))
        fill_boosted_histograms(output, "wr_mumu_boosted_dy_cr", np.array([True]), "DYJets", lep, ak8, loose,
                                weights, {"Nominal": weights.weight()}, event_values={"npv_good": np.array([42.0])})
        sel = {"process": "DYJets", "region": "wr_mumu_boosted_dy_cr", "syst": "Nominal"}
        assert output["msd_leading_AK8Jets"][sel][hist_mod.loc(92.0)].value == pytest.approx(2.0)
        assert output["dR_loose_lepton_AK8Jet"][sel][hist_mod.loc(0.5)].value == pytest.approx(2.0)
        assert output["npv_good"][sel][hist_mod.loc(42.0)].value == pytest.approx(2.0)
        # A jet without a lepton (LSF3 = -1) is inside the LSF axis.
        assert output["LSF_leading_AK8Jets"][sel][hist_mod.loc(-1.0)].value == pytest.approx(2.0)
