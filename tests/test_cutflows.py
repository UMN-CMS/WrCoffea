"""Regression tests for selection yields, statistical errors, and ROOT consumers."""

from pathlib import Path
import runpy

import awkward as ak
import hist
import numpy as np
import pytest
import uproot
from coffea.analysis_tools import PackedSelection, Weights

from tests.test_analyzer import _make_integration_events
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.histograms import _cutflow_histograms
from wrcoffea.save_hists import _save_cutflows


def _set_leptons(events, collection, records):
    """Each record is (pt, eta, phi, passes_tight_id)."""
    leptons = getattr(events, collection)[:, :len(records)]
    for index, field in enumerate(("pt", "eta", "phi")):
        values = np.array([[record[index] for record in records]], dtype=float)
        leptons = ak.with_field(leptons, ak.Array(values), field)
    tight = np.array([[record[3] for record in records]], dtype=bool)
    if collection == "Muon":
        leptons = ak.with_field(leptons, ak.Array(np.where(tight, 0.05, 0.2)), "tkRelIso")
    else:
        leptons = ak.with_field(leptons, ak.Array(tight), "cutBased_HEEP")
    setattr(events, collection, leptons)


def _resolved_output(flavor="mumu", case="baseline"):
    events = _make_integration_events(n_events=1, n_muons=3, n_electrons=3)
    lead, sublead = (300., 0., 0., True), (250., 0., np.pi, True)
    if case == "low_lead_pt":
        # Large eta separation keeps mll > 400 even though the lead fails pT.
        lead, sublead = (59., 2.3, 0., True), (58., -2.3, np.pi, True)
    muons = [lead, sublead] if flavor == "mumu" else ([lead] if flavor == "em" else [])
    electrons = [lead, sublead] if flavor == "ee" else ([sublead] if flavor == "em" else [])
    extras = {
        "extra_loose": (70., 0., 1., False),
        "extra_low_pt": (20., 0., 1., True),
        "extra_outside_eta": (70., 3., 1., True),
        "extra_tight": (70., 0., 1., True),
    }
    if case in extras:
        (electrons if flavor == "ee" else muons).append(extras[case])
    if case == "mixed_id_collections":
        # Two muons pass kinematics and two pass ID, but only one passes both.
        muons = [lead, extras["extra_loose"], extras["extra_low_pt"]]
        electrons = [sublead]
    _set_leptons(events, "Muon", muons)
    _set_leptons(events, "Electron", electrons)
    for field, values in {"pt": [[300., 300.]], "eta": [[0., 0.]],
                          "phi": [[np.pi / 2, -np.pi / 2]]}.items():
        events.Jet = ak.with_field(events.Jet, ak.Array(values), field)

    proc = WrAnalysis(None, region="resolved")
    tight, _, lepton_masks = proc.select_leptons(events)
    jets, _, jet_masks = proc.select_jets(events, events.metadata["era"])
    selections = proc.resolved_selections(
        tight, jets, lepton_masks=lepton_masks, jet_masks=jet_masks,
        triggers=proc.build_trigger_masks(events, events.metadata["era"]),
    )
    weights = Weights(1)
    weights.add("base", np.ones(1))
    output = proc.make_output()
    proc._fill_resolved(
        output, selections, "Test", jets, tight, weights,
        {"Nominal": weights.weight()}, events.metadata["era"], None, None, is_mc=False,
    )
    return output


@pytest.mark.parametrize("flavor", ["ee", "mumu", "em"])
@pytest.mark.parametrize("case", ["baseline", "extra_loose", "extra_low_pt",
                                  "extra_outside_eta", "extra_tight", "low_lead_pt"])
def test_resolved_cutflow_matches_selected_region(flavor, case):
    output = _resolved_output(flavor, case)
    expected = float(case not in ("extra_tight", "low_lead_pt"))
    regions = {"ee": "wr_ee_resolved_sr", "mumu": "wr_mumu_resolved_sr",
               "em": "wr_resolved_flavor_cr"}
    for channel, region in regions.items():
        count = expected if channel == flavor else 0.
        histogram = output["mass_dilepton"]
        observed = 0.
        if region in list(histogram.axes["region"]):
            observed = histogram[{
                "process": "Test", "region": region, "syst": "Nominal",
            }].sum(flow=True).value
        assert observed == count
        for name in ("cumulative", "cumulative_unweighted"):
            assert output["cutflow"][channel][name].values()[-1] == count


def test_mixed_flavor_event_is_retained_only_in_flavor_cr():
    output = _resolved_output("em", "mixed_id_collections")
    assert output["mass_dilepton"][{
        "process": "Test", "region": "wr_resolved_flavor_cr", "syst": "Nominal",
    }].sum(flow=True).value == 1.
    for flavor, expected in (("ee", 0.), ("mumu", 0.), ("em", 1.)):
        assert output["cutflow"][flavor]["cumulative_unweighted"].values()[-1] == expected


def _crossing_cuts(indices):
    selections = PackedSelection()
    selections.add("a", np.array([True, False, True, False])[indices])
    selections.add("b", np.array([False, True, True, False])[indices])
    weights = Weights(len(np.arange(4)[indices]))
    weights.add("base", np.array([2., 3., -1., 4.])[indices])
    return _cutflow_histograms(selections, weights, ["a", "b"])


def test_onecut_and_cumulative_moments_are_independent_of_chunks():
    together = _crossing_cuts(slice(None))
    first = _crossing_cuts(slice(0, 2))
    second = _crossing_cuts(slice(2, 4))
    np.testing.assert_array_equal(together["onecut"].values(), [8., 1., 2.])
    np.testing.assert_array_equal(together["onecut"].variances(), [30., 5., 10.])
    np.testing.assert_array_equal(together["cumulative"].values(), [8., 1., -1.])
    np.testing.assert_array_equal(together["cumulative"].variances(), [30., 5., 1.])
    np.testing.assert_array_equal(together["onecut_unweighted"].values(), [4., 2., 2.])
    np.testing.assert_array_equal(together["cumulative_unweighted"].values(), [4., 2., 1.])
    for name, expected in together.items():
        merged = first[name] + second[name]
        np.testing.assert_array_equal(merged.values(), expected.values())
        np.testing.assert_array_equal(merged.variances(), expected.variances())


def test_empty_chunk_has_zero_counts_and_variances():
    for histogram in _crossing_cuts(slice(0, 0)).values():
        np.testing.assert_array_equal(histogram.values(), np.zeros(3))
        np.testing.assert_array_equal(histogram.variances(), np.zeros(3))


@pytest.fixture(params=["make_cutflow_tables.py", "make_signal_cutflows.py"])
def table_reader(request, monkeypatch):
    plotter = Path(__file__).resolve().parents[1] / "WR_Plotter"
    script = plotter / "bin" / request.param
    if not script.exists():
        pytest.skip("WR_Plotter submodule is not checked out")
    monkeypatch.syspath_prepend(str(plotter))
    return runpy.run_path(str(script))


def test_analyzer_root_output_matches_table_sequences(table_reader, tmp_path):
    output = _resolved_output()
    path = tmp_path / "cutflows.root"
    with uproot.recreate(path) as root_file:
        _save_cutflows(root_file, output["cutflow"], "cutflow_resolved")
    for flavor, sequence in table_reader["_CUT_SEQUENCES"].items():
        expected_keys = [key for key, _ in sequence]
        for name in ("cumulative", "onecut", "cumulative_unweighted", "onecut_unweighted"):
            histogram = output["cutflow"][flavor][name]
            assert list(histogram.axes[0]) == expected_keys
            # Both readers accept a fully qualified hist_type suffix.
            actual = table_reader["_read_cutflow"](
                path, flavor, hist_type=name, expected_keys=expected_keys,
            )
            np.testing.assert_array_equal(actual, histogram.values())
            with uproot.open(path) as root_file:
                saved = root_file[f"cutflow_resolved/{flavor}/{name}"]
                np.testing.assert_array_equal(saved.variances(), histogram.variances())
    if "_CUT_LABELS_GENERIC" in table_reader:
        assert len(table_reader["_CUT_LABELS_GENERIC"][False]) == len(expected_keys)


def test_table_reader_rejects_different_sequence_with_same_bin_count(table_reader, tmp_path):
    expected = [key for key, _ in table_reader["_CUT_SEQUENCES"]["mumu"]]
    labels = expected.copy()
    labels[-2], labels[-3] = labels[-3], labels[-2]
    path = tmp_path / "old_sequence.root"
    with uproot.recreate(path) as root_file:
        root_file["cutflow_resolved/mumu/cumulative"] = hist.Hist(
            hist.axis.StrCategory(labels, name="cut"), storage=hist.storage.Weight(),
        )
    assert table_reader["_read_cutflow"](path, "mumu", expected_keys=expected) is None
