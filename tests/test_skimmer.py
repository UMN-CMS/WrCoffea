"""Unit tests for the wrcoffea.skimmer module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wrcoffea.analysis_config import CUTS
from wrcoffea.skimmer import (
    KEPT_BRANCHES,
    SKIM_CUTS,
    SUMMED_RUNS_BRANCHES,
    SkimResult,
    read_runs_tree,
)


# ---------------------------------------------------------------------------
# Skim cuts vs analysis cuts
# ---------------------------------------------------------------------------


class TestSkimCuts:
    """Skim cuts must always be looser than (or equal to) analysis cuts."""

    def test_lepton_pt(self):
        assert SKIM_CUTS["lepton_pt_min"] <= CUTS["lepton_pt_min"]

    def test_lepton_eta(self):
        assert SKIM_CUTS["lepton_eta_max"] >= CUTS["lepton_eta_max"]

    def test_ak4_pt(self):
        assert SKIM_CUTS["ak4_pt_min"] <= CUTS["ak4_pt_min"]

    def test_ak4_eta(self):
        assert SKIM_CUTS["ak4_eta_max"] >= CUTS["ak4_eta_max"]

    def test_ak8_pt(self):
        assert SKIM_CUTS["ak8_pt_min"] <= CUTS["ak8_pt_min"]

    def test_lead_lepton_pt(self):
        assert SKIM_CUTS["lead_lepton_pt_min"] <= CUTS["lead_lepton_pt_min"]

    def test_sublead_lepton_pt(self):
        assert SKIM_CUTS["sublead_lepton_pt_min"] <= CUTS["sublead_lepton_pt_min"]


# ---------------------------------------------------------------------------
# Kept branches
# ---------------------------------------------------------------------------


class TestKeptBranches:
    """Verify that critical branches are in the kept set."""

    @pytest.mark.parametrize(
        "branch",
        [
            "genWeight", "HLT", "Muon", "Electron", "Jet", "FatJet",
            "MET", "Pileup", "PV", "Flag", "Generator", "GenPart",
            "event", "run", "luminosityBlock",
        ],
    )
    def test_essential_branches_kept(self, branch):
        assert branch in KEPT_BRANCHES


# ---------------------------------------------------------------------------
# read_runs_tree
# ---------------------------------------------------------------------------


class TestReadRunsTree:
    """Test Runs tree collapsing (the genEventSumw inflation fix)."""

    def _mock_uproot_open(self, runs_data):
        """Create a mock for uproot.open that returns *runs_data*."""
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key == "Runs"
        mock_runs_tree = MagicMock()
        mock_runs_tree.arrays.return_value = runs_data
        mock_file.__getitem__ = lambda self, key: mock_runs_tree
        cm = MagicMock()
        cm.__enter__ = lambda self: mock_file
        cm.__exit__ = lambda self, *a: None
        return cm

    def test_single_entry_unchanged(self):
        runs_data = {
            "genEventSumw_": np.array([1000.0]),
            "run": np.array([1]),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(runs_data)):
            result = read_runs_tree("fake.root")

        assert result is not None
        assert result["genEventSumw_"] == pytest.approx(np.array([1000.0]))
        assert result["run"].tolist() == [1]

    def test_multi_entry_sumw_summed(self):
        runs_data = {
            "genEventSumw_": np.array([100.0, 200.0, 300.0]),
            "genEventCount_": np.array([10, 20, 30]),
            "run": np.array([1, 1, 1]),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(runs_data)):
            result = read_runs_tree("fake.root")

        assert result is not None
        assert result["genEventSumw_"] == pytest.approx(np.array([600.0]))
        assert result["genEventCount_"] == pytest.approx(np.array([60]))
        assert len(result["run"]) == 1

    def test_missing_runs_returns_none(self):
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: False
        cm = MagicMock()
        cm.__enter__ = lambda self: mock_file
        cm.__exit__ = lambda self, *a: None
        with patch("wrcoffea.skimmer.uproot.open", return_value=cm):
            result = read_runs_tree("fake.root")

        assert result is None



# ---------------------------------------------------------------------------
# SkimResult
# ---------------------------------------------------------------------------


class TestSkimResult:
    def test_construction(self):
        r = SkimResult(
            src_path="a.root", dest_path="b.root",
            n_events_before=1000, n_events_after=500,
            file_size_bytes=1024, efficiency=50.0,
        )
        assert r.n_events_before == 1000
        assert r.efficiency == 50.0
