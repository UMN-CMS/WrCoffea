"""Unit tests for the wrcoffea.skim_merge module."""

import json
from unittest.mock import patch

import numpy as np
import pytest
import uproot

from wrcoffea.skim_merge import MergeResult, save_merge_summary


class TestMergeResult:
    def test_events_and_sumw_match(self):
        r = MergeResult(
            dataset="ds", input_files=10, output_files=2,
            total_events_in=5000, total_events_out=5000,
            total_sumw_in=100.0, total_sumw_out=100.0,
            events_match=True, sumw_match=True,
        )
        assert r.events_match
        assert r.sumw_match

    def test_events_mismatch(self):
        r = MergeResult(
            dataset="ds", input_files=10, output_files=2,
            total_events_in=5000, total_events_out=4999,
            total_sumw_in=100.0, total_sumw_out=100.0,
            events_match=False, sumw_match=True,
        )
        assert not r.events_match
        assert r.sumw_match

    def test_sumw_mismatch(self):
        r = MergeResult(
            dataset="ds", input_files=10, output_files=2,
            total_events_in=5000, total_events_out=5000,
            total_sumw_in=100.0, total_sumw_out=112.0,
            events_match=True, sumw_match=False,
        )
        assert r.events_match
        assert not r.sumw_match

    def test_output_paths_default_empty(self):
        r = MergeResult(
            dataset="ds", input_files=0, output_files=0,
            total_events_in=0, total_events_out=0,
            total_sumw_in=0.0, total_sumw_out=0.0,
            events_match=True, sumw_match=True,
        )
        assert r.output_paths == []

    def test_to_dict(self):
        r = MergeResult(
            dataset="ds", input_files=3, output_files=1,
            total_events_in=100, total_events_out=100,
            total_sumw_in=50.0, total_sumw_out=50.0,
            events_match=True, sumw_match=True,
            output_paths=["/tmp/ds_part1.root"],
        )
        d = r.to_dict()
        assert d["dataset"] == "ds"
        assert d["input_files"] == 3
        assert d["output_paths"] == ["/tmp/ds_part1.root"]
        # Should be JSON-serializable
        json.dumps(d)


class TestSaveMergeSummary:
    def test_writes_json(self, tmp_path):
        r = MergeResult(
            dataset="myds", input_files=5, output_files=2,
            total_events_in=1000, total_events_out=1000,
            total_sumw_in=42.0, total_sumw_out=42.0,
            events_match=True, sumw_match=True,
            output_paths=["a.root", "b.root"],
        )
        path = save_merge_summary(r, tmp_path)
        assert path.name == "merge_summary.json"
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["dataset"] == "myds"
        assert data["total_events_in"] == 1000
        assert data["events_match"] is True
        assert data["output_paths"] == ["a.root", "b.root"]


def _make_skim_file(path, n_events, hlt_branches):
    """Create a minimal ROOT skim file with an Events TTree and given HLT branches."""
    branch_types = {"run": np.dtype("int32")}
    for hlt in hlt_branches:
        branch_types[hlt] = np.dtype("bool")
    branch_data = {"run": np.ones(n_events, dtype=np.int32)}
    for hlt in hlt_branches:
        branch_data[hlt] = np.ones(n_events, dtype=np.bool_)
    with uproot.recreate(str(path)) as f:
        f.mktree("Events", branch_types)
        f["Events"].extend(branch_data)


@pytest.fixture(autouse=False)
def _skip_uproot_patch():
    """Disable the uproot shared-counter patch (requires dask_awkward)."""
    with patch("wrcoffea.skim_merge._ensure_uproot_patch"):
        yield


class TestMergeDatasetChunking:
    """Test that merge_dataset groups by HLT first, then chunks within each group."""

    @pytest.mark.usefixtures("_skip_uproot_patch")
    def test_same_hlt_files_combined(self, tmp_path):
        """Files with the same HLT signature and total < max_events
        should be merged into a single output file per HLT group."""
        from wrcoffea.skim_merge import merge_dataset

        hlt_a = ["HLT_Mu50", "HLT_Mu55"]
        hlt_b = ["HLT_Mu50", "HLT_Mu55", "HLT_Ele32"]

        # Create 3 small files with hlt_a (150 events total)
        _make_skim_file(tmp_path / "f1_skim.root", 50, hlt_a)
        _make_skim_file(tmp_path / "f2_skim.root", 40, hlt_a)
        _make_skim_file(tmp_path / "f3_skim.root", 60, hlt_a)

        # Create 2 small files with hlt_b (90 events total)
        _make_skim_file(tmp_path / "f4_skim.root", 50, hlt_b)
        _make_skim_file(tmp_path / "f5_skim.root", 40, hlt_b)

        result = merge_dataset(
            tmp_path, "testds",
            max_events=1_000_000,
            hlt_aware=True,
        )

        assert result.events_match
        assert result.total_events_in == 240
        assert result.total_events_out == 240
        # Should produce exactly 2 output files (one per HLT group),
        # not 5 tiny files.
        assert result.output_files == 2

        # Summary should exist
        summary = tmp_path / "merge_summary.json"
        assert summary.exists()
        data = json.loads(summary.read_text())
        assert data["output_files"] == 2

    @pytest.mark.usefixtures("_skip_uproot_patch")
    def test_chunking_within_hlt_group(self, tmp_path):
        """When a single HLT group exceeds max_events, it should be
        split into multiple chunks."""
        from wrcoffea.skim_merge import merge_dataset

        hlt = ["HLT_Mu50"]

        # 3 files with 100 events each, max_events=150
        _make_skim_file(tmp_path / "f1_skim.root", 100, hlt)
        _make_skim_file(tmp_path / "f2_skim.root", 100, hlt)
        _make_skim_file(tmp_path / "f3_skim.root", 100, hlt)

        result = merge_dataset(
            tmp_path, "testds",
            max_events=150,
            hlt_aware=True,
        )

        assert result.events_match
        assert result.total_events_in == 300
        assert result.total_events_out == 300
        # f1 (100) fits in chunk 1, f2 would push to 200 > 150 so flush,
        # then f2 (100) in chunk 2, f3 would push to 200 > 150 so flush,
        # then f3 (100) as remaining -> 3 output files
        assert result.output_files == 3
