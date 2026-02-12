"""Unit tests for the wrcoffea.skim_merge module."""

import pytest

from wrcoffea.skim_merge import MergeResult


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
