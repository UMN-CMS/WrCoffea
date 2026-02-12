"""
Tests for event weight computation in WrAnalysis.build_event_weights()

Critical functionality:
- MC normalization (xsec × lumi / genEventSumw)
- Pileup reweighting
- Lepton scale factors (RECO, ID, ISO, trigger)
- Systematic variations
- Data vs MC path separation
"""

import pytest
import numpy as np
import awkward as ak
from coffea.analysis_tools import Weights
from unittest.mock import Mock, patch, MagicMock

from wrcoffea.analyzer import WrAnalysis


@pytest.fixture
def mock_events():
    """Create mock NanoAOD events for testing weight computation."""
    n_events = 100
    return {
        "genWeight": ak.Array(np.random.normal(1.0, 0.1, n_events)),
        "Pileup_nTrueInt": ak.Array(np.random.uniform(10, 50, n_events)),
        "Electron": {
            "pt": ak.Array([[45.0, 35.0], [50.0], [], [60.0, 40.0]] * 25),
            "eta": ak.Array([[1.2, -1.5], [0.8], [], [2.0, -2.1]] * 25),
        },
        "Muon": {
            "pt": ak.Array([[55.0], [65.0, 45.0], [70.0, 50.0, 40.0], []] * 25),
            "eta": ak.Array([[0.5], [1.0, -1.2], [1.5, -0.8, 2.0], []] * 25),
        },
    }


@pytest.fixture
def mock_metadata():
    """Create mock metadata for MC normalization."""
    return {
        "xsec": 1234.5,  # pb
        "genEventSumw": 50000.0,
        "is_mc": True,
    }


@pytest.fixture
def analyzer():
    """Create WrAnalysis instance for testing."""
    return WrAnalysis(era="RunIII2024Summer24", region="both", systematics=[])


class TestEventWeightsMC:
    """Test event weight computation for MC samples."""

    def test_mc_normalization_formula(self, analyzer, mock_events, mock_metadata):
        """Test MC weight = genWeight × xsec × lumi / genEventSumw."""
        n_events = len(mock_events["genWeight"])

        # Mock tight leptons (empty for simplicity)
        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
            )

        # Extract nominal weights
        nominal_weights = weights.weight()

        # Verify formula: genWeight × xsec × lumi / genEventSumw
        lumi = 35000.0  # RunIII2024Summer24 lumi in pb^-1
        expected_factor = mock_metadata["xsec"] * lumi / mock_metadata["genEventSumw"]
        expected_weights = mock_events["genWeight"] * expected_factor

        np.testing.assert_allclose(
            nominal_weights,
            expected_weights,
            rtol=1e-5,
            err_msg="MC normalization formula incorrect"
        )

    def test_zero_sumw_handling(self, analyzer, mock_events):
        """Test graceful handling of zero genEventSumw (avoid division by zero)."""
        metadata_zero_sumw = {"xsec": 100.0, "genEventSumw": 0.0, "is_mc": True}
        n_events = len(mock_events["genWeight"])

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, metadata_zero_sumw
            )

        # Should return all zeros or handle gracefully (not crash)
        nominal_weights = weights.weight()
        assert len(nominal_weights) == n_events
        assert not np.any(np.isnan(nominal_weights)), "NaN weights from zero sumw"
        assert not np.any(np.isinf(nominal_weights)), "Inf weights from zero sumw"

    def test_pileup_reweighting_applied(self, analyzer, mock_events, mock_metadata):
        """Test that pileup weights are multiplied into total weight."""
        n_events = len(mock_events["genWeight"])
        pileup_weights = np.random.uniform(0.8, 1.2, n_events)

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=pileup_weights):
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
            )

        # Check pileup weight is in the Weights object
        assert "pileup" in weights.weightStatistics

        # Nominal weight should include pileup
        lumi = 35000.0
        mc_factor = mock_metadata["xsec"] * lumi / mock_metadata["genEventSumw"]
        expected_weights = mock_events["genWeight"] * mc_factor * pileup_weights

        np.testing.assert_allclose(
            weights.weight(),
            expected_weights,
            rtol=1e-5,
            err_msg="Pileup weights not applied correctly"
        )

    @patch("wrcoffea.scale_factors.muon_sf")
    @patch("wrcoffea.scale_factors.muon_trigger_sf")
    def test_muon_sf_components_stacked(
        self, mock_trig_sf, mock_muon_sf, analyzer, mock_events, mock_metadata
    ):
        """Test muon SF components (RECO, ID, ISO) are multiplied together."""
        n_events = len(mock_events["genWeight"])

        # Mock SF returns: (nominal, up, down) for each component
        sf_reco = (np.ones(n_events) * 0.98, np.ones(n_events) * 0.99, np.ones(n_events) * 0.97)
        sf_id = (np.ones(n_events) * 0.95, np.ones(n_events) * 0.96, np.ones(n_events) * 0.94)
        sf_iso = (np.ones(n_events) * 0.97, np.ones(n_events) * 0.98, np.ones(n_events) * 0.96)
        sf_trig = (np.ones(n_events) * 0.93, np.ones(n_events) * 0.94, np.ones(n_events) * 0.92)

        mock_muon_sf.return_value = (sf_reco, sf_id, sf_iso)
        mock_trig_sf.return_value = sf_trig

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {
            "pt": ak.Array([[50.0] for _ in range(n_events)]),
            "eta": ak.Array([[1.0] for _ in range(n_events)]),
        }
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
            )

        # Check all muon SF components are present
        assert "muon_sf_RECO" in weights.weightStatistics
        assert "muon_sf_ID" in weights.weightStatistics
        assert "muon_sf_ISO" in weights.weightStatistics
        assert "muon_trigger_sf" in weights.weightStatistics

        # Nominal weight should include product of all SFs
        lumi = 35000.0
        mc_factor = mock_metadata["xsec"] * lumi / mock_metadata["genEventSumw"]
        expected_sf_product = sf_reco[0] * sf_id[0] * sf_iso[0] * sf_trig[0]
        expected_weights = mock_events["genWeight"] * mc_factor * expected_sf_product

        np.testing.assert_allclose(
            weights.weight(),
            expected_weights,
            rtol=1e-4,
            err_msg="Muon SF components not stacked correctly"
        )


class TestEventWeightsData:
    """Test event weight computation for data samples."""

    def test_data_weights_all_ones(self, analyzer, mock_events):
        """Test that data weights are exactly 1.0 (no genWeight, SF, pileup)."""
        metadata_data = {"is_mc": False}
        n_events = len(mock_events["genWeight"])

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        weights = analyzer.build_event_weights(
            mock_events, tight_electrons, tight_muons, ak4_jets, metadata_data
        )

        # Data should have weight = 1.0 for all events
        np.testing.assert_array_equal(
            weights.weight(),
            np.ones(n_events),
            err_msg="Data weights should all be 1.0"
        )

        # Should have no weight variations for data
        assert len(weights.variations) == 0 or all(
            "nominal" in var for var in weights.variations
        ), "Data should not have systematic weight variations"

    def test_data_no_sf_applied(self, analyzer, mock_events):
        """Test that scale factors are not applied to data."""
        metadata_data = {"is_mc": False}
        n_events = len(mock_events["genWeight"])

        tight_electrons = {
            "pt": ak.Array([[50.0] for _ in range(n_events)]),
            "eta": ak.Array([[1.0] for _ in range(n_events)]),
        }
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.electron_id_sf") as mock_sf:
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, metadata_data
            )

            # SF functions should never be called for data
            mock_sf.assert_not_called()

        np.testing.assert_array_equal(weights.weight(), np.ones(n_events))


class TestSystematicWeights:
    """Test systematic weight variations."""

    def test_lumi_systematic_variations(self, mock_events, mock_metadata):
        """Test luminosity up/down variations (typically ±2.5%)."""
        analyzer = WrAnalysis(era="RunIII2024Summer24", region="both", systematics=["lumi"])
        n_events = len(mock_events["genWeight"])

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
            )

        # Check lumi variations exist
        variations = weights.variations
        assert "lumiUp" in variations, "lumiUp variation missing"
        assert "lumiDown" in variations, "lumiDown variation missing"

        # Lumi up should be > nominal, lumi down should be < nominal
        nominal = weights.weight()
        lumi_up = weights.weight("lumiUp")
        lumi_down = weights.weight("lumiDown")

        assert np.all(lumi_up > nominal), "lumiUp should increase weights"
        assert np.all(lumi_down < nominal), "lumiDown should decrease weights"

        # Check typical lumi uncertainty magnitude (~2.5%)
        relative_up = (lumi_up - nominal) / nominal
        relative_down = (nominal - lumi_down) / nominal
        np.testing.assert_allclose(relative_up, 0.025, atol=0.01)
        np.testing.assert_allclose(relative_down, 0.025, atol=0.01)

    @patch("wrcoffea.scale_factors.pileup_weight")
    def test_pileup_systematic_variations(self, mock_pu_weight, mock_events, mock_metadata):
        """Test pileup up/down variations."""
        analyzer = WrAnalysis(era="RunIII2024Summer24", region="both", systematics=["pileup"])
        n_events = len(mock_events["genWeight"])

        # Mock pileup weight returns: (nominal, up, down)
        pu_nominal = np.ones(n_events)
        pu_up = np.ones(n_events) * 1.05
        pu_down = np.ones(n_events) * 0.95
        mock_pu_weight.return_value = (pu_nominal, pu_up, pu_down)

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        weights = analyzer.build_event_weights(
            mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
        )

        # Check pileup variations exist
        assert "pileupUp" in weights.variations
        assert "pileupDown" in weights.variations

        # Verify up/down weights match expected values
        nominal = weights.weight()
        weight_up = weights.weight("pileupUp")
        weight_down = weights.weight("pileupDown")

        # Up variation should use pu_up, down should use pu_down
        lumi = 35000.0
        mc_factor = mock_metadata["xsec"] * lumi / mock_metadata["genEventSumw"]
        expected_up = mock_events["genWeight"] * mc_factor * pu_up
        expected_down = mock_events["genWeight"] * mc_factor * pu_down

        np.testing.assert_allclose(weight_up, expected_up, rtol=1e-5)
        np.testing.assert_allclose(weight_down, expected_down, rtol=1e-5)

    @patch("wrcoffea.scale_factors.muon_sf")
    def test_sf_systematic_variations(self, mock_sf, mock_events, mock_metadata):
        """Test scale factor systematic variations."""
        analyzer = WrAnalysis(era="RunIII2024Summer24", region="both", systematics=["sf"])
        n_events = len(mock_events["genWeight"])

        # Mock SF with variations: (nominal, up, down) for each component
        sf_reco = (np.ones(n_events) * 0.98, np.ones(n_events) * 0.99, np.ones(n_events) * 0.97)
        sf_id = (np.ones(n_events), np.ones(n_events), np.ones(n_events))
        sf_iso = (np.ones(n_events), np.ones(n_events), np.ones(n_events))
        mock_sf.return_value = (sf_reco, sf_id, sf_iso)

        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {
            "pt": ak.Array([[50.0] for _ in range(n_events)]),
            "eta": ak.Array([[1.0] for _ in range(n_events)]),
        }
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            with patch("wrcoffea.scale_factors.muon_trigger_sf", return_value=(np.ones(n_events), np.ones(n_events), np.ones(n_events))):
                weights = analyzer.build_event_weights(
                    mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
                )

        # Check SF variations exist
        assert "muon_sf_RECOUp" in weights.variations
        assert "muon_sf_RECODown" in weights.variations


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_lepton_collections(self, analyzer, mock_events, mock_metadata):
        """Test handling of events with no tight leptons."""
        n_events = len(mock_events["genWeight"])

        # All empty lepton collections
        tight_electrons = {"pt": ak.Array([[] for _ in range(n_events)])}
        tight_muons = {"pt": ak.Array([[] for _ in range(n_events)])}
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            weights = analyzer.build_event_weights(
                mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
            )

        # Should not crash, weights should be valid
        assert len(weights.weight()) == n_events
        assert not np.any(np.isnan(weights.weight()))

    def test_mixed_electron_muon_events(self, analyzer, mock_events, mock_metadata):
        """Test events with both electrons and muons (should apply both SFs)."""
        n_events = len(mock_events["genWeight"])

        tight_electrons = {
            "pt": ak.Array([[50.0] if i % 2 == 0 else [] for i in range(n_events)]),
            "eta": ak.Array([[1.0] if i % 2 == 0 else [] for i in range(n_events)]),
        }
        tight_muons = {
            "pt": ak.Array([[60.0] if i % 2 == 1 else [] for i in range(n_events)]),
            "eta": ak.Array([[1.2] if i % 2 == 1 else [] for i in range(n_events)]),
        }
        ak4_jets = {"pt": ak.Array([[] for _ in range(n_events)])}

        with patch("wrcoffea.scale_factors.pileup_weight", return_value=np.ones(n_events)):
            with patch("wrcoffea.scale_factors.electron_id_sf", return_value=(np.ones(n_events) * 0.97, np.ones(n_events), np.ones(n_events))):
                with patch("wrcoffea.scale_factors.electron_reco_sf", return_value=(np.ones(n_events) * 0.98, np.ones(n_events), np.ones(n_events))):
                    with patch("wrcoffea.scale_factors.electron_trigger_sf", return_value=(np.ones(n_events) * 0.96, np.ones(n_events), np.ones(n_events))):
                        with patch("wrcoffea.scale_factors.muon_sf", return_value=((np.ones(n_events) * 0.95, np.ones(n_events), np.ones(n_events)), (np.ones(n_events), np.ones(n_events), np.ones(n_events)), (np.ones(n_events), np.ones(n_events), np.ones(n_events)))):
                            with patch("wrcoffea.scale_factors.muon_trigger_sf", return_value=(np.ones(n_events) * 0.94, np.ones(n_events), np.ones(n_events))):
                                weights = analyzer.build_event_weights(
                                    mock_events, tight_electrons, tight_muons, ak4_jets, mock_metadata
                                )

        # Both electron and muon SFs should be in weight statistics
        weight_names = set(weights.weightStatistics.keys())
        assert any("electron" in name for name in weight_names), "No electron SF applied"
        assert any("muon" in name for name in weight_names), "No muon SF applied"

        # Weights should still be valid
        assert len(weights.weight()) == n_events
        assert not np.any(np.isnan(weights.weight()))
