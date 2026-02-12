"""Unit tests for the wrcoffea.skimmer module."""

from unittest.mock import MagicMock, patch

import awkward as ak
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

    class _MockAsDtype:
        """Mock for uproot AsDtype interpretation."""
        def __init__(self, dtype):
            self._dtype = np.dtype(dtype)

        @property
        def to_dtype(self):
            return self._dtype

    class _MockAsJagged:
        """Mock for uproot AsJagged interpretation."""
        def __init__(self, content_dtype):
            self.content = TestReadRunsTree._MockAsDtype(content_dtype)

    def _mock_uproot_open(self, branch_data, branch_dtypes):
        """Create a mock for uproot.open with proper branch interpretations.

        branch_data : dict of branch_name -> list/array values
        branch_dtypes : dict of branch_name -> (dtype_str, is_jagged)
        """
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key == "Runs"

        mock_runs_tree = MagicMock()
        mock_runs_tree.keys.return_value = list(branch_data.keys())

        # Mock branch interpretations
        interps = {}
        for bname, (dtype_str, is_jagged) in branch_dtypes.items():
            if is_jagged:
                interps[bname] = TestReadRunsTree._MockAsJagged(dtype_str)
            else:
                interps[bname] = TestReadRunsTree._MockAsDtype(dtype_str)

        def branch_getitem(key):
            mock_branch = MagicMock()
            mock_branch.interpretation = interps[key]
            return mock_branch

        # MagicMock passes (self, key) to assigned dunder methods
        mock_runs_tree.__getitem__ = lambda _self, key: branch_getitem(key)

        # Return awkward array from .arrays()
        ak_data = ak.Array({k: v for k, v in branch_data.items()})
        mock_runs_tree.arrays.return_value = ak_data

        mock_file.__getitem__ = lambda _self, key: mock_runs_tree

        cm = MagicMock()
        cm.__enter__ = lambda self: mock_file
        cm.__exit__ = lambda self, *a: None
        return cm

    def test_single_entry_unchanged(self):
        branch_data = {
            "genEventSumw_": [1000.0],
            "run": [np.uint32(1)],
        }
        branch_dtypes = {
            "genEventSumw_": ("float64", False),
            "run": ("uint32", False),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(branch_data, branch_dtypes)):
            result = read_runs_tree("fake.root")

        assert result is not None
        data, types = result
        assert float(data["genEventSumw_"][0]) == pytest.approx(1000.0)
        assert int(data["run"][0]) == 1

    def test_multi_entry_sumw_summed(self):
        branch_data = {
            "genEventSumw_": [100.0, 200.0, 300.0],
            "genEventCount_": [10, 20, 30],
            "run": [np.uint32(1), np.uint32(1), np.uint32(1)],
        }
        branch_dtypes = {
            "genEventSumw_": ("float64", False),
            "genEventCount_": ("int64", False),
            "run": ("uint32", False),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(branch_data, branch_dtypes)):
            result = read_runs_tree("fake.root")

        assert result is not None
        data, types = result
        assert float(data["genEventSumw_"][0]) == pytest.approx(600.0)
        assert int(data["genEventCount_"][0]) == 60
        assert len(data["run"]) == 1

    def test_jagged_branches_summed_elementwise(self):
        branch_data = {
            "genEventSumw_": [100.0, 200.0],
            "nLHEScaleSumw": [np.int32(3), np.int32(3)],
            "LHEScaleSumw": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "run": [np.uint32(1), np.uint32(1)],
        }
        branch_dtypes = {
            "genEventSumw_": ("float64", False),
            "nLHEScaleSumw": ("int32", False),
            "LHEScaleSumw": ("float64", True),
            "run": ("uint32", False),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(branch_data, branch_dtypes)):
            result = read_runs_tree("fake.root")

        assert result is not None
        data, types = result
        # LHEScaleSumw should be element-wise summed: [5.0, 7.0, 9.0]
        summed = ak.to_list(data["LHEScaleSumw"])
        assert summed == [[5.0, 7.0, 9.0]]
        # nLHEScaleSumw is not in SUMMED_RUNS_BRANCHES, just take first
        assert int(data["nLHEScaleSumw"][0]) == 3
        # Branch types should include jagged spec
        assert types["LHEScaleSumw"] == "var * float64"

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
