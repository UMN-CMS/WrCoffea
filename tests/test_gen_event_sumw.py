"""Tests for the ratio-valued theory-weight sums in scripts/gen_event_sumw.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "gen_event_sumw", str(Path(__file__).resolve().parent.parent / "scripts" / "gen_event_sumw.py"))
ges = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ges  # dataclasses resolve postponed annotations via sys.modules
_spec.loader.exec_module(ges)


class TestWeightedLheSum:
    def test_single_entry(self):
        out = ges.weighted_lhe_sum([100.0], [[0.9, 1.1]])
        np.testing.assert_allclose(out, [90.0, 110.0])

    def test_multi_entry_file(self):
        out = ges.weighted_lhe_sum([100.0, 300.0], [[1.0, 2.0], [2.0, 4.0]])
        np.testing.assert_allclose(out, [700.0, 1400.0])

    def test_finalize_gives_dataset_ratio(self):
        total = ges.weighted_lhe_sum([100.0, 300.0], [[1.0, 2.0], [2.0, 4.0]])
        ratios = ges.finalize_lhe({"LHEScaleSumw": total}, 400.0)
        assert ratios == {"LHEScaleSumw": [pytest.approx(1.75), pytest.approx(3.5)]}

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            ges.weighted_lhe_sum([1.0, 1.0], [[1.0, 2.0], [1.0]])

    def test_zero_total_sumw_writes_nothing(self):
        assert ges.finalize_lhe({"LHEScaleSumw": np.zeros(2)}, 0.0) == {}
