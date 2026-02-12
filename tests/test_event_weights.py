"""
Tests for event weight computation in WrAnalysis.build_event_weights()

Critical functionality:
- MC normalization (genWeight * xsec * lumi * 1000 / genEventSumw)
- Pileup reweighting
- Lepton scale factors (RECO, ID, ISO, trigger)
- Systematic variations (lumi, pileup, SF)
- Data vs MC path separation
"""

import pytest
import numpy as np
import awkward as ak
from coffea.analysis_tools import Weights
from unittest.mock import patch, MagicMock

from wrcoffea.analyzer import WrAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_EVENTS = 100

# Use an era that IS configured in config.yaml for muon, electron, pileup JSONs.
ERA = "RunIII2024Summer24"

# Expected lumi from config.yaml (fb^-1).  The code multiplies by 1000 to
# get pb^-1 inside build_event_weights.
LUMI_FB = 109.08


class MockEvents:
    """Minimal mock that looks like a NanoAOD events object."""
    pass


def _make_mock_events(n=N_EVENTS, gen_weights=None):
    """Return a MockEvents object with genWeight and len() support."""
    ev = MockEvents()
    if gen_weights is None:
        gen_weights = np.random.default_rng(42).normal(1.0, 0.1, n)
    ev.genWeight = ak.Array(gen_weights)
    # Make len(ev) work.
    type(ev).__len__ = lambda self: len(self.genWeight)
    return ev


def _make_metadata(era=ERA, xsec=1234.5, sumw=50000.0, sample="TestSample"):
    """Return metadata dict that build_event_weights expects."""
    return {
        "era": era,
        "xsec": xsec,
        "genEventSumw": sumw,
        "sample": sample,
    }


def _make_tight_muons(n=N_EVENTS):
    """Return an awkward record array mimicking tight muons with pt/eta."""
    return ak.zip({
        "pt": ak.Array([[50.0, 40.0]] * n),
        "eta": ak.Array([[1.0, -1.0]] * n),
    })


def _make_tight_electrons(n=N_EVENTS):
    """Return an awkward record array mimicking tight electrons with pt/eta/deltaEtaSC."""
    return ak.zip({
        "pt": ak.Array([[45.0, 35.0]] * n),
        "eta": ak.Array([[1.2, -1.5]] * n),
        "deltaEtaSC": ak.Array([[0.01, -0.02]] * n),
    })


def _sf_triple(n, nom_val=1.0, up_val=1.01, down_val=0.99):
    """Return a (nominal, up, down) tuple of flat numpy arrays."""
    return (
        np.full(n, nom_val, dtype=np.float64),
        np.full(n, up_val, dtype=np.float64),
        np.full(n, down_val, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """WrAnalysis with no systematics enabled."""
    return WrAnalysis(mass_point=None, enabled_systs=[], region="both")


@pytest.fixture
def mock_events():
    """Mock events object with genWeight."""
    return _make_mock_events()


@pytest.fixture
def metadata():
    """Standard MC metadata."""
    return _make_metadata()


# ---------------------------------------------------------------------------
# MC weight tests
# ---------------------------------------------------------------------------

class TestEventWeightsMC:
    """Test event weight computation for MC samples."""

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_mc_normalization_formula(self, mock_pu, analyzer, mock_events, metadata):
        """Test MC weight = genWeight * xsec * lumi * 1000 / genEventSumw."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n, 1.0, 1.0, 1.0)

        weights, syst_weights = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
        )

        nominal = weights.weight()

        expected_factor = (
            metadata["xsec"] * LUMI_FB * 1000.0 / metadata["genEventSumw"]
        )
        expected = np.asarray(mock_events.genWeight) * expected_factor

        np.testing.assert_allclose(
            nominal, expected, rtol=1e-5,
            err_msg="MC normalization formula incorrect",
        )

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_return_type_is_tuple(self, mock_pu, analyzer, mock_events, metadata):
        """build_event_weights returns (Weights, dict)."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)

        result = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
        )

        assert isinstance(result, tuple) and len(result) == 2
        weights, syst_weights = result
        assert isinstance(weights, Weights)
        assert isinstance(syst_weights, dict)
        assert "Nominal" in syst_weights

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_zero_sumw_raises(self, mock_pu, analyzer, mock_events):
        """Zero genEventSumw should raise ZeroDivisionError."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)
        bad_meta = _make_metadata(sumw=0.0)

        with pytest.raises(ZeroDivisionError):
            analyzer.build_event_weights(mock_events, bad_meta, is_mc=True)

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_pileup_reweighting_applied(self, mock_pu, analyzer, mock_events, metadata):
        """Pileup weight is multiplied into the total weight."""
        n = len(mock_events)
        pu_nom = np.random.default_rng(7).uniform(0.8, 1.2, n)
        pu_up = pu_nom * 1.05
        pu_down = pu_nom * 0.95
        mock_pu.return_value = (pu_nom, pu_up, pu_down)

        weights, _ = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
        )

        # The pileup weight should be registered in the Weights object.
        assert "pileup" in weights.weightStatistics

        # Nominal weight should include pileup.
        factor = metadata["xsec"] * LUMI_FB * 1000.0 / metadata["genEventSumw"]
        expected = np.asarray(mock_events.genWeight) * factor * pu_nom

        np.testing.assert_allclose(
            weights.weight(), expected, rtol=1e-5,
            err_msg="Pileup weights not applied correctly",
        )

    @patch("wrcoffea.analyzer.muon_trigger_sf")
    @patch("wrcoffea.analyzer.muon_sf")
    @patch("wrcoffea.analyzer.pileup_weight")
    def test_muon_sf_components(self, mock_pu, mock_mu_sf, mock_mu_trig,
                                analyzer, mock_events, metadata):
        """Muon RECO, ID, ISO, trigger SFs are registered as independent weights."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)

        # muon_sf returns dict of component -> (nom, up, down)
        mock_mu_sf.return_value = {
            "reco": _sf_triple(n, 0.98, 0.99, 0.97),
            "id":   _sf_triple(n, 0.95, 0.96, 0.94),
            "iso":  _sf_triple(n, 0.97, 0.98, 0.96),
        }
        mock_mu_trig.return_value = _sf_triple(n, 0.93, 0.94, 0.92)

        tight_muons = _make_tight_muons(n)

        weights, syst_weights = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
            tight_muons=tight_muons,
        )

        # All muon SF components should be in weight statistics.
        assert "muon_reco_sf" in weights.weightStatistics
        assert "muon_id_sf" in weights.weightStatistics
        assert "muon_iso_sf" in weights.weightStatistics
        assert "muon_trig_sf" in weights.weightStatistics

        # Nominal weight should include product of all SFs.
        factor = metadata["xsec"] * LUMI_FB * 1000.0 / metadata["genEventSumw"]
        sf_product = 0.98 * 0.95 * 0.97 * 0.93
        expected = np.asarray(mock_events.genWeight) * factor * sf_product

        np.testing.assert_allclose(
            weights.weight(), expected, rtol=1e-4,
            err_msg="Muon SF components not stacked correctly",
        )

    @patch("wrcoffea.analyzer.electron_trigger_sf")
    @patch("wrcoffea.analyzer.electron_id_sf")
    @patch("wrcoffea.analyzer.electron_reco_sf")
    @patch("wrcoffea.analyzer.pileup_weight")
    def test_electron_sf_components(self, mock_pu, mock_e_reco, mock_e_id,
                                    mock_e_trig, analyzer, mock_events, metadata):
        """Electron RECO, ID, trigger SFs are registered as independent weights."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)
        mock_e_reco.return_value = _sf_triple(n, 0.98, 0.99, 0.97)
        mock_e_id.return_value = _sf_triple(n, 0.97, 0.98, 0.96)
        mock_e_trig.return_value = _sf_triple(n, 0.96, 0.97, 0.95)

        tight_electrons = _make_tight_electrons(n)

        weights, syst_weights = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
            tight_electrons=tight_electrons,
        )

        assert "electron_reco_sf" in weights.weightStatistics
        assert "electron_id_sf" in weights.weightStatistics
        assert "electron_trig_sf" in weights.weightStatistics

        factor = metadata["xsec"] * LUMI_FB * 1000.0 / metadata["genEventSumw"]
        sf_product = 0.98 * 0.97 * 0.96
        expected = np.asarray(mock_events.genWeight) * factor * sf_product

        np.testing.assert_allclose(
            weights.weight(), expected, rtol=1e-4,
            err_msg="Electron SF components not stacked correctly",
        )


# ---------------------------------------------------------------------------
# Data weight tests
# ---------------------------------------------------------------------------

class TestEventWeightsData:
    """Test event weight computation for data samples."""

    def test_data_weights_all_ones(self, analyzer):
        """Data weights are exactly 1.0."""
        n = N_EVENTS
        ev = _make_mock_events(n)
        meta = _make_metadata()

        weights, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=False,
        )

        np.testing.assert_array_equal(
            weights.weight(), np.ones(n),
            err_msg="Data weights should all be 1.0",
        )
        assert "Nominal" in syst_weights

    def test_data_no_sf_called(self, analyzer):
        """Scale factor functions are never called for data."""
        n = N_EVENTS
        ev = _make_mock_events(n)
        meta = _make_metadata()
        tight_muons = _make_tight_muons(n)
        tight_electrons = _make_tight_electrons(n)

        with patch("wrcoffea.analyzer.muon_sf") as mock_mu, \
             patch("wrcoffea.analyzer.electron_reco_sf") as mock_e_reco, \
             patch("wrcoffea.analyzer.pileup_weight") as mock_pu:

            weights, _ = analyzer.build_event_weights(
                ev, meta, is_mc=False,
                tight_muons=tight_muons, tight_electrons=tight_electrons,
            )

            mock_mu.assert_not_called()
            mock_e_reco.assert_not_called()
            mock_pu.assert_not_called()

        np.testing.assert_array_equal(weights.weight(), np.ones(n))

    def test_data_no_systematic_variations(self, analyzer):
        """Data syst_weights dict contains only Nominal."""
        n = N_EVENTS
        ev = _make_mock_events(n)
        meta = _make_metadata()

        _, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=False,
        )

        assert set(syst_weights.keys()) == {"Nominal"}


# ---------------------------------------------------------------------------
# Systematic weight tests
# ---------------------------------------------------------------------------

class TestSystematicWeights:
    """Test systematic weight variations."""

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_lumi_systematic_variations(self, mock_pu):
        """Lumi up/down variations are present and have correct direction."""
        n = N_EVENTS
        mock_pu.return_value = _sf_triple(n)
        ev = _make_mock_events(n)
        meta = _make_metadata()

        analyzer = WrAnalysis(mass_point=None, enabled_systs=["lumi"], region="both")
        weights, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=True,
        )

        assert "LumiUp" in syst_weights, "LumiUp variation missing"
        assert "LumiDown" in syst_weights, "LumiDown variation missing"

        nominal = np.asarray(syst_weights["Nominal"])
        lumi_up = np.asarray(syst_weights["LumiUp"])
        lumi_down = np.asarray(syst_weights["LumiDown"])

        # Lumi up should be > nominal, lumi down should be < nominal
        # (for events with positive weights).
        pos_mask = nominal > 0
        assert np.all(lumi_up[pos_mask] > nominal[pos_mask]), "LumiUp should increase weights"
        assert np.all(lumi_down[pos_mask] < nominal[pos_mask]), "LumiDown should decrease weights"

        # Check typical lumi uncertainty magnitude (~1.4% for RunIII2024Summer24).
        relative_up = (lumi_up[pos_mask] - nominal[pos_mask]) / nominal[pos_mask]
        np.testing.assert_allclose(relative_up, 0.014, atol=0.005)

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_pileup_systematic_variations(self, mock_pu):
        """Pileup up/down variations are present in syst_weights."""
        n = N_EVENTS
        pu_nom = np.ones(n)
        pu_up = np.ones(n) * 1.05
        pu_down = np.ones(n) * 0.95
        mock_pu.return_value = (pu_nom, pu_up, pu_down)

        ev = _make_mock_events(n)
        meta = _make_metadata()

        analyzer = WrAnalysis(mass_point=None, enabled_systs=["pileup"], region="both")
        weights, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=True,
        )

        assert "PileupUp" in syst_weights
        assert "PileupDown" in syst_weights

        # Verify the up/down variations reflect the pileup changes.
        factor = meta["xsec"] * LUMI_FB * 1000.0 / meta["genEventSumw"]
        gen = np.asarray(ev.genWeight)
        expected_up = gen * factor * pu_up
        expected_down = gen * factor * pu_down

        np.testing.assert_allclose(
            np.asarray(syst_weights["PileupUp"]), expected_up, rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(syst_weights["PileupDown"]), expected_down, rtol=1e-5,
        )

    @patch("wrcoffea.analyzer.muon_trigger_sf")
    @patch("wrcoffea.analyzer.muon_sf")
    @patch("wrcoffea.analyzer.pileup_weight")
    def test_sf_systematic_variations(self, mock_pu, mock_mu_sf, mock_mu_trig):
        """Scale-factor systematic variations appear in syst_weights."""
        n = N_EVENTS
        mock_pu.return_value = _sf_triple(n)
        mock_mu_sf.return_value = {
            "reco": _sf_triple(n, 0.98, 0.99, 0.97),
            "id":   _sf_triple(n, 1.0, 1.0, 1.0),
            "iso":  _sf_triple(n, 1.0, 1.0, 1.0),
        }
        mock_mu_trig.return_value = _sf_triple(n, 1.0, 1.0, 1.0)

        ev = _make_mock_events(n)
        meta = _make_metadata()
        tight_muons = _make_tight_muons(n)

        analyzer = WrAnalysis(mass_point=None, enabled_systs=["sf"], region="both")
        weights, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=True,
            tight_muons=tight_muons,
        )

        assert "MuonRecoSfUp" in syst_weights
        assert "MuonRecoSfDown" in syst_weights
        assert "MuonTrigSfUp" in syst_weights
        assert "MuonTrigSfDown" in syst_weights


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_no_leptons_passed(self, mock_pu, analyzer, mock_events, metadata):
        """When no tight leptons are passed, only event_weight + pileup are set."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)

        weights, syst_weights = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
            # tight_muons=None, tight_electrons=None  (defaults)
        )

        assert len(weights.weight()) == n
        assert not np.any(np.isnan(np.asarray(weights.weight())))
        assert "Nominal" in syst_weights

    @patch("wrcoffea.analyzer.electron_trigger_sf")
    @patch("wrcoffea.analyzer.electron_id_sf")
    @patch("wrcoffea.analyzer.electron_reco_sf")
    @patch("wrcoffea.analyzer.muon_trigger_sf")
    @patch("wrcoffea.analyzer.muon_sf")
    @patch("wrcoffea.analyzer.pileup_weight")
    def test_both_lepton_flavors(self, mock_pu, mock_mu_sf, mock_mu_trig,
                                 mock_e_reco, mock_e_id, mock_e_trig,
                                 analyzer, mock_events, metadata):
        """Both muon and electron SFs are applied when both collections are provided."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)
        mock_mu_sf.return_value = {
            "reco": _sf_triple(n, 0.95, 1.0, 0.90),
            "id":   _sf_triple(n, 1.0, 1.0, 1.0),
            "iso":  _sf_triple(n, 1.0, 1.0, 1.0),
        }
        mock_mu_trig.return_value = _sf_triple(n, 0.94, 1.0, 0.88)
        mock_e_reco.return_value = _sf_triple(n, 0.98, 1.0, 0.96)
        mock_e_id.return_value = _sf_triple(n, 0.97, 1.0, 0.94)
        mock_e_trig.return_value = _sf_triple(n, 0.96, 1.0, 0.92)

        tight_muons = _make_tight_muons(n)
        tight_electrons = _make_tight_electrons(n)

        weights, syst_weights = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
            tight_muons=tight_muons, tight_electrons=tight_electrons,
        )

        weight_names = set(weights.weightStatistics.keys())
        assert any("muon" in name for name in weight_names), "No muon SF applied"
        assert any("electron" in name for name in weight_names), "No electron SF applied"

        assert len(weights.weight()) == n
        assert not np.any(np.isnan(np.asarray(weights.weight())))

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_compute_sumw_mode(self, mock_pu):
        """compute_sumw=True omits the /sumw division."""
        n = N_EVENTS
        mock_pu.return_value = _sf_triple(n)
        ev = _make_mock_events(n)
        meta = _make_metadata(sumw=50000.0)

        analyzer_sumw = WrAnalysis(
            mass_point=None, enabled_systs=[], region="both", compute_sumw=True,
        )
        weights, _ = analyzer_sumw.build_event_weights(
            ev, meta, is_mc=True,
        )

        # Without /sumw, weight = genWeight * xsec * lumi * 1000
        expected = np.asarray(ev.genWeight) * meta["xsec"] * LUMI_FB * 1000.0

        np.testing.assert_allclose(
            weights.weight(), expected, rtol=1e-5,
            err_msg="compute_sumw mode should omit /sumw normalization",
        )

    def test_unconfigured_era_no_pileup(self, analyzer):
        """An era not in PILEUP_JSONS should skip pileup weight without error."""
        n = N_EVENTS
        ev = _make_mock_events(n)
        # Use an era that is in LUMIS but not in PILEUP_JSONS.
        meta = _make_metadata(era="Run3Summer22")

        weights, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=True,
        )

        # Pileup should NOT be in the weight statistics.
        assert "pileup" not in weights.weightStatistics
        assert len(weights.weight()) == n
