"""Tests for WrAnalysis._theory_variations (QCD-scale and parton-shower weights).

Covers the three input situations met in the Run 3 skims:
- every event carries the weights (the normal case);
- the dataset has no LHEScaleWeight branch at all (pure-Pythia 8 dibosons);
- some events of a dataset have no weights (one production file each in 2022 ZZZ
  and the 22EE WR MWR6000_N1500 point): those events get the nominal weight.
"""

import logging

import awkward as ak
import numpy as np
import pytest

import wrcoffea.analyzer as analyzer_mod
from wrcoffea.analyzer import WrAnalysis

ERA = "Run3Summer22"


def _analyzer():
    return WrAnalysis(mass_point=None, enabled_systs=["scale", "ps"], region="both")


def _events(scale_rows, ps_rows=None, *, with_scale=True):
    n = len(scale_rows)
    if ps_rows is None:
        ps_rows = [[1.0, 1.0, 1.0, 1.0]] * n
    fields = {"PSWeight": ak.Array(ps_rows)}
    if with_scale:
        fields["LHEScaleWeight"] = ak.Array(scale_rows)
    return ak.zip(fields, depth_limit=1)


def _metadata(scale_ratios, ps_ratios=(1.0, 1.0, 1.0, 1.0), sample="TestSample"):
    return {"sample": sample, "LHEScaleSumw": list(scale_ratios), "PSSumw": list(ps_ratios)}


@pytest.fixture(autouse=True)
def _reset_warnings():
    analyzer_mod._WARN_ONCE_ANALYZER.clear()
    yield
    analyzer_mod._WARN_ONCE_ANALYZER.clear()


def test_all_events_with_weights_are_divided_by_the_dataset_ratio():
    rng = np.random.default_rng(1)
    rows = rng.uniform(0.8, 1.2, size=(5, 9)).tolist()
    ratios = rng.uniform(0.9, 1.1, size=9)
    out = _analyzer()._theory_variations(_events(rows), _metadata(ratios), ERA)
    w = np.asarray(rows)
    up, down = out["ScaleMuR"]          # indices (7, 1)
    np.testing.assert_allclose(up, w[:, 7] / ratios[7])
    np.testing.assert_allclose(down, w[:, 1] / ratios[1])


def test_events_without_scale_weights_get_exactly_nominal(caplog):
    rng = np.random.default_rng(2)
    full = rng.uniform(0.8, 1.2, size=(3, 9)).tolist()
    rows = [full[0], [], full[1], [], full[2]]
    ratios = rng.uniform(0.9, 1.1, size=9)
    with caplog.at_level(logging.WARNING, logger="wrcoffea.analyzer"):
        out = _analyzer()._theory_variations(_events(rows), _metadata(ratios, sample="ZZZ"), ERA)
        out2 = _analyzer()._theory_variations(_events(rows), _metadata(ratios, sample="ZZZ"), ERA)
    missing = np.array([False, True, False, True, False])
    w = np.asarray([full[0], [0] * 9, full[1], [0] * 9, full[2]])
    for name, (iu, idn) in WrAnalysis._SCALE_INDEX.items():
        up, down = out[name]
        np.testing.assert_array_equal(up[missing], 1.0)
        np.testing.assert_array_equal(down[missing], 1.0)
        np.testing.assert_allclose(up[~missing], w[~missing, iu] / ratios[iu])
        np.testing.assert_allclose(down[~missing], w[~missing, idn] / ratios[idn])
        np.testing.assert_allclose(out2[name][0], up)
    msgs = [r.getMessage() for r in caplog.records if "NOMINAL" in r.getMessage()]
    assert len(msgs) == 1                                  # once per dataset
    assert "2 of 5 events of 'ZZZ'" in msgs[0]


def test_nominal_fill_keeps_the_generator_level_yield():
    """The dataset ratio comes from the events that have weights, so giving the others a
    factor of 1 leaves sum(genWeight * factor) == sum(genWeight) for every variation."""
    rng = np.random.default_rng(3)
    n_with, n_without = 400, 7
    s = rng.uniform(0.7, 1.3, size=(n_with, 9))
    g_with = rng.uniform(0.5, 1.5, size=n_with)
    g_without = rng.uniform(0.5, 1.5, size=n_without)
    ratios = (g_with[:, None] * s).sum(axis=0) / g_with.sum()
    rows = s.tolist() + [[]] * n_without
    g = np.concatenate([g_with, g_without])
    out = _analyzer()._theory_variations(_events(rows), _metadata(ratios), ERA)
    for name in WrAnalysis._SCALE_INDEX:
        for factor in out[name]:
            assert np.isclose((g * factor).sum(), g.sum(), rtol=1e-12)


def test_other_weight_count_mismatches_still_raise():
    rows = [[1.0] * 9, [1.0] * 8]
    with pytest.raises(ValueError, match="entries per event, expected 9"):
        _analyzer()._theory_variations(_events(rows), _metadata([1.0] * 9), ERA)


def test_dataset_without_the_branch_gets_nominal(caplog):
    rows = [[1.0] * 9] * 4                                  # only used for the event count
    with caplog.at_level(logging.WARNING, logger="wrcoffea.analyzer"):
        out = _analyzer()._theory_variations(_events(rows, with_scale=False),
                                             _metadata([1.0] * 9, sample="WW"), ERA)
    for name in WrAnalysis._SCALE_INDEX:
        np.testing.assert_array_equal(out[name][0], 1.0)
        np.testing.assert_array_equal(out[name][1], 1.0)
    assert any("has no LHEScaleWeight branch" in r.getMessage() for r in caplog.records)


def test_eight_weight_samples_use_the_eight_weight_order():
    """TTLNu-1Jets stores 8 scale weights (no nominal): [0.5,0.5] [0.5,1] [0.5,2] [1,0.5]
    [1,2] [2,0.5] [2,1] [2,2] in (muR, muF)."""
    rng = np.random.default_rng(4)
    rows = rng.uniform(0.8, 1.2, size=(6, 8))
    ratios = rng.uniform(0.9, 1.1, size=8)
    out = _analyzer()._theory_variations(_events(rows.tolist()), _metadata(ratios), ERA)
    expected = {"ScaleMuR": (6, 1), "ScaleMuF": (4, 3), "ScaleMuRF": (7, 0)}
    assert WrAnalysis._SCALE_INDEX_8 == expected
    for name, (iu, idn) in expected.items():
        np.testing.assert_allclose(out[name][0], rows[:, iu] / ratios[iu])
        np.testing.assert_allclose(out[name][1], rows[:, idn] / ratios[idn])


def test_eight_weight_order_matches_the_nine_weight_order_without_the_nominal():
    nine = [(0.5, 0.5), (0.5, 1), (0.5, 2), (1, 0.5), (1, 1), (1, 2), (2, 0.5), (2, 1), (2, 2)]
    eight = [p for p in nine if p != (1, 1)]
    for name in WrAnalysis._SCALE_INDEX:
        for i9, i8 in zip(WrAnalysis._SCALE_INDEX[name], WrAnalysis._SCALE_INDEX_8[name]):
            assert nine[i9] == eight[i8]


def test_metadata_and_event_weight_counts_must_agree():
    rows = [[1.0] * 9] * 3
    with pytest.raises(ValueError, match="entries per event, expected 8"):
        _analyzer()._theory_variations(_events(rows), _metadata([1.0] * 8), ERA)
    with pytest.raises(ValueError, match="9 or 8-entry 'LHEScaleSumw'"):
        _analyzer()._theory_variations(_events(rows), _metadata([1.0] * 7), ERA)
