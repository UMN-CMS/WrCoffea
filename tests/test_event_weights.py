"""
Tests for event weight computation in WrAnalysis.

Current API (after the per-region SF refactor):
- build_event_weights(events, metadata, is_mc) computes BASE weights only:
  MC normalization (genWeight * xsec * lumi * 1000 / genEventSumw), pileup,
  and the optional lumi systematic.  It no longer accepts lepton collections.
- Lepton SFs moved to _lepton_sfs_for_region(region, syst_weights, era,
  tight_muons, tight_electrons, loose_muons=None, loose_electrons=None),
  which multiplies region-appropriate SFs into the syst_weights dict.
"""

import pytest
import numpy as np
import awkward as ak
from coffea.analysis_tools import Weights
from unittest.mock import patch, MagicMock

from wrcoffea.analyzer import WrAnalysis
from wrcoffea.analysis_config import LUMIS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_EVENTS = 100

# Use an era that IS configured in config.yaml for muon, electron, pileup JSONs.
ERA = "RunIII2024Summer24"

# Lumi comes straight from the analysis config (fb^-1) so the expectation
# cannot drift from config.yaml.  The code multiplies by 1000 to get pb^-1
# inside build_event_weights.
LUMI_FB = float(LUMIS[ERA])

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


def _base_syst_weights(n=N_EVENTS, value=1.0):
    """Synthetic base syst_weights dict, as produced by build_event_weights."""
    return {"Nominal": np.full(n, value, dtype=np.float64)}


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


# ---------------------------------------------------------------------------
# Per-region lepton SF tests (_lepton_sfs_for_region)
# ---------------------------------------------------------------------------

class TestLeptonSFsForRegion:
    """Lepton SFs are applied per region via _lepton_sfs_for_region.

    These tests pin the RESTORED behavior with synthetic syst_weights and
    monkeypatched SF functions; the method currently early-returns
    dict(syst_weights), so they carry the strict xfail mark.
    """

    @patch("wrcoffea.analyzer.muon_sf")
    @patch("wrcoffea.analyzer.muon_trigger_sf")
    def test_muon_sf_components(self, mock_mu_trig, mock_mu_sf, analyzer):
        """Muon RECO, ID, ISO, trigger SFs multiply the region syst_weights."""
        n = N_EVENTS
        # muon_sf returns dict of component -> (nom, up, down)
        mock_mu_sf.return_value = {
            "reco": _sf_triple(n, 0.98, 0.99, 0.97),
            "id":   _sf_triple(n, 0.95, 0.96, 0.94),
            "iso":  _sf_triple(n, 0.97, 0.98, 0.96),
        }
        mock_mu_trig.return_value = _sf_triple(n, 0.93, 0.94, 0.92)

        tight_muons = _make_tight_muons(n)
        base = _base_syst_weights(n, 2.0)

        result = analyzer._lepton_sfs_for_region(
            "mumu_resolved_sr", base, ERA, tight_muons, None,
        )

        sf_product = 0.98 * 0.95 * 0.97 * 0.93
        np.testing.assert_allclose(
            np.asarray(result["Nominal"]), np.asarray(base["Nominal"]) * sf_product,
            rtol=1e-4,
            err_msg="Muon SF components not stacked correctly",
        )
        mock_mu_sf.assert_called_once()
        mock_mu_trig.assert_called_once()

    @patch("wrcoffea.analyzer.electron_id_sf")
    @patch("wrcoffea.analyzer.electron_reco_sf")
    def test_electron_sf_components(self, mock_e_reco, mock_e_id, analyzer):
        """Electron RECO and HEEP ID SFs multiply the ee-region syst_weights."""
        n = N_EVENTS
        mock_e_reco.return_value = _sf_triple(n, 0.98, 0.99, 0.97)
        mock_e_id.return_value = _sf_triple(n, 0.95, 0.96, 0.94)

        tight_electrons = _make_tight_electrons(n)
        base = _base_syst_weights(n, 3.0)

        result = analyzer._lepton_sfs_for_region(
            "ee_resolved_sr", base, ERA, None, tight_electrons,
        )

        np.testing.assert_allclose(
            np.asarray(result["Nominal"]), np.asarray(base["Nominal"]) * 0.98 * 0.95,
            rtol=1e-4,
            err_msg="Electron RECO x ID SF not applied correctly",
        )
        mock_e_reco.assert_called_once()
        mock_e_id.assert_called_once()

    @patch("wrcoffea.analyzer.electron_id_sf")
    @patch("wrcoffea.analyzer.electron_reco_sf")
    @patch("wrcoffea.analyzer.muon_trigger_sf")
    @patch("wrcoffea.analyzer.muon_sf")
    def test_both_lepton_flavors(self, mock_mu_sf, mock_mu_trig, mock_e_reco, mock_e_id, analyzer):
        """Flavor CR applies both muon and electron SFs."""
        n = N_EVENTS
        mock_mu_sf.return_value = {
            "reco": _sf_triple(n, 0.95, 1.0, 0.90),
            "id":   _sf_triple(n, 1.0, 1.0, 1.0),
            "iso":  _sf_triple(n, 1.0, 1.0, 1.0),
        }
        mock_mu_trig.return_value = _sf_triple(n, 0.94, 1.0, 0.88)
        mock_e_reco.return_value = _sf_triple(n, 0.98, 1.0, 0.96)
        mock_e_id.return_value = _sf_triple(n, 0.97, 1.0, 0.94)

        tight_muons = _make_tight_muons(n)
        tight_electrons = _make_tight_electrons(n)
        base = _base_syst_weights(n, 1.0)

        result = analyzer._lepton_sfs_for_region(
            "flavor_cr_resolved", base, ERA, tight_muons, tight_electrons,
        )

        sf_product = 0.95 * 0.94 * 0.98 * 0.97
        np.testing.assert_allclose(
            np.asarray(result["Nominal"]), np.full(n, sf_product), rtol=1e-4,
            err_msg="Both-flavor SF product incorrect in flavor CR",
        )
        mock_mu_sf.assert_called_once()
        mock_e_reco.assert_called_once()
        mock_e_id.assert_called_once()

    @patch("wrcoffea.analyzer.muon_trigger_sf")
    @patch("wrcoffea.analyzer.muon_sf")
    def test_sf_systematic_variations(self, mock_mu_sf, mock_mu_trig):
        """Per-component SF up/down variations appear in the returned dict."""
        n = N_EVENTS
        mock_mu_sf.return_value = {
            "reco": _sf_triple(n, 0.98, 0.99, 0.97),
            "id":   _sf_triple(n, 1.0, 1.0, 1.0),
            "iso":  _sf_triple(n, 1.0, 1.0, 1.0),
        }
        mock_mu_trig.return_value = _sf_triple(n, 1.0, 1.0, 1.0)

        tight_muons = _make_tight_muons(n)
        base = _base_syst_weights(n, 1.0)

        analyzer = WrAnalysis(mass_point=None, enabled_systs=["sf"], region="both")
        result = analyzer._lepton_sfs_for_region(
            "mumu_resolved_sr", base, ERA, tight_muons, None,
        )

        assert "MuonRecoSfUp" in result
        assert "MuonRecoSfDown" in result
        assert "MuonTrigSfUp" in result
        assert "MuonTrigSfDown" in result

        # Value-level: varied = Nominal * (varied_component / nominal_component).
        nominal = np.asarray(result["Nominal"])
        np.testing.assert_allclose(
            np.asarray(result["MuonRecoSfUp"]), nominal * (0.99 / 0.98), rtol=1e-5,
        )
        np.testing.assert_allclose(
            np.asarray(result["MuonRecoSfDown"]), nominal * (0.97 / 0.98), rtol=1e-5,
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
        """Scale factor / pileup functions are never called for data."""
        n = N_EVENTS
        ev = _make_mock_events(n)
        meta = _make_metadata()

        with patch("wrcoffea.analyzer.muon_sf") as mock_mu, \
             patch("wrcoffea.analyzer.electron_reco_sf") as mock_e_reco, \
             patch("wrcoffea.analyzer.pileup_weight") as mock_pu:

            weights, _ = analyzer.build_event_weights(
                ev, meta, is_mc=False,
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


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_base_weights_no_nans(self, mock_pu, analyzer, mock_events, metadata):
        """Base MC weights are finite and length-matched (no lepton args now)."""
        n = len(mock_events)
        mock_pu.return_value = _sf_triple(n)

        weights, syst_weights = analyzer.build_event_weights(
            mock_events, metadata, is_mc=True,
        )

        assert len(weights.weight()) == n
        assert not np.any(np.isnan(np.asarray(weights.weight())))
        assert "Nominal" in syst_weights

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

    @patch("wrcoffea.analyzer.PILEUP_JSONS", {})
    @patch("wrcoffea.scale_factors.PILEUP_JSONS", {})
    def test_unconfigured_era_no_pileup(self, analyzer):
        """An era not in PILEUP_JSONS should skip pileup weight without error."""
        n = N_EVENTS
        ev = _make_mock_events(n)
        meta = _make_metadata()

        weights, syst_weights = analyzer.build_event_weights(
            ev, meta, is_mc=True,
        )

        # Pileup should NOT be in the weight statistics.
        assert "pileup" not in weights.weightStatistics
        assert len(weights.weight()) == n
