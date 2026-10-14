"""Regression tests for selection yields, statistical errors, and ROOT consumers."""

from pathlib import Path
import json
from types import SimpleNamespace

import awkward as ak
import hist
import numpy as np
import pytest
import uproot
from coffea.analysis_tools import PackedSelection, Weights

from tests.test_analyzer import _apply_sf_patches, _make_integration_events
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.histograms import CUTFLOW_SF_ROW, _cutflow_histograms, fill_boosted_cutflows, fill_cutflows
from wrcoffea.regions import is_signal_region
from wrcoffea.save_hists import _save_cutflows, save_histograms

# ROOT directory -> key of the analyzer payload holding that cutflow.
_CUTFLOW_KEYS = {"cutflow_resolved": "cutflow", "cutflow_boosted": "cutflow_boosted"}
_CUTFLOW_NAMES = ("cumulative", "onecut", "cumulative_unweighted", "onecut_unweighted")
# Blinded data drops these steps and everything after them in ee and mumu
# (the plotter contract); em, the flavor CR, is never blinded.
_BLINDED_FROM = {"cutflow_resolved": "mll_gt200", "cutflow_boosted": "mll_gt200_boosted"}
# The last step of each ee/mumu chain is the signal region.
_SR_LAST_CUT = {"cutflow_resolved": "mll_gt400", "cutflow_boosted": "mlj_gt800_boosted"}
# The chains of each directory: boosted me (the mu(e-jet) flavour CR) since 2026-10-14; em and me are never blinded.
_CHANNELS = {"cutflow_resolved": {"ee", "mumu", "em"}, "cutflow_boosted": {"ee", "mumu", "em", "me"}}
_HEM_ROWS = {"cutflow_resolved": {"ee": "hem_veto", "mumu": "hem_veto", "em": "hem_veto"},
             "cutflow_boosted": {"ee": "hem_veto_boosted_sr", "mumu": "hem_veto_boosted_sr",
                                 "em": "hem_veto_boosted_cr", "me": "hem_veto_boosted_cr"}}


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


def _resolved_output(flavor="mumu", case="baseline", *, unblind=False):
    """Fill the resolved regions and cutflows of one data event (era UL18).

    Data is blinded unless *unblind*, which the fixture's Run 2 era allows.
    """
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

    proc = WrAnalysis(None, region="resolved", unblind=unblind)
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
    # Unblinded, so the last ee/mumu rows can be compared with the SRs.
    output = _resolved_output(flavor, case, unblind=True)
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

    # Blinded data (the default) fills no SR and keeps the leading cutflow rows.
    blinded = _resolved_output(flavor, case)
    assert set(blinded["mass_dilepton"].axes["region"]) == {
        r for r in output["mass_dilepton"].axes["region"] if not is_signal_region(r)}
    for channel in regions:
        for name in _CUTFLOW_NAMES:
            full = output["cutflow"][channel][name]
            cut = blinded["cutflow"][channel][name]
            labels = list(full.axes[0])
            n = len(labels) if channel == "em" else labels.index(_BLINDED_FROM["cutflow_resolved"])
            assert list(cut.axes[0]) == labels[:n]
            np.testing.assert_array_equal(cut.values(), full.values()[:n])
            np.testing.assert_array_equal(cut.variances(), full.variances()[:n])


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


class _PassAll:
    """Selections stand-in that passes every cut (one event)."""

    def all(self, *names):
        return np.ones(1, dtype=bool)


def _analyzer_cut_chains():
    """The full cut chains the analyzer books, per ROOT directory and flavor."""
    weights = Weights(1)
    weights.add("base", np.ones(1))
    output = {}
    fill_cutflows(output, _PassAll(), weights)
    fill_boosted_cutflows(output, _PassAll(), weights)
    return {
        directory: {flavor: list(hists["cumulative"].axes[0]) for flavor, hists in output[key].items()}
        for directory, key in _CUTFLOW_KEYS.items()
    }


@_apply_sf_patches
def _process_both_topologies(datatype, *, unblind=False):
    """One analyzer chunk (era UL18) of mumu resolved-SR events, run with the
    resolved and boosted regions so that both cutflows are booked."""
    n = 3
    metadata = None
    if datatype == "data":
        metadata = {"datatype": "data", "physics_group": "Observed", "sample": "Observed"}
    events = _make_integration_events(n_events=n, n_muons=2, n_jets=2, n_fatjets=1,
                                      metadata=metadata, include_genweight=datatype == "mc")
    for collection, fields in (
        ("Muon", {"pt": [[300., 250.]], "eta": [[0., 0.]], "phi": [[0., np.pi]]}),
        ("Jet", {"pt": [[300., 300.]], "eta": [[0., 0.]], "phi": [[np.pi / 2, -np.pi / 2]]}),
    ):
        objects = getattr(events, collection)
        for field, values in fields.items():
            objects = ak.with_field(objects, ak.Array(values * n), field)
        setattr(events, collection, objects)
    return WrAnalysis(None, region="both", unblind=unblind).process(events)


def test_analyzer_cut_chains_end_in_the_signal_regions():
    for directory, chains in _analyzer_cut_chains().items():
        assert set(chains) == _CHANNELS[directory]
        for flavor, row in _HEM_ROWS[directory].items():
            assert row in chains[flavor]
        if "me" in chains:
            assert chains["me"][:3] == ["no_cuts", "boosted_tag", "lead_is_muon"]
        for flavor in ("ee", "mumu"):
            assert chains[flavor][0] == "no_cuts"
            assert chains[flavor][-2] == _SR_LAST_CUT[directory]
            assert chains[flavor][-1] == CUTFLOW_SF_ROW
            assert _BLINDED_FROM[directory] in chains[flavor]


@pytest.mark.parametrize("datatype, unblind", [("mc", False), ("data", False), ("data", True)],
                         ids=["mc", "data-blinded", "data-unblinded"])
def test_analyzer_root_output_matches_cut_chains(datatype, unblind, tmp_path, monkeypatch):
    """Written cutflows follow the analyzer's chains: complete for MC and
    unblinded data; for blinded data (the default), ee/mumu stop before the
    mll cut and em is complete."""
    monkeypatch.chdir(tmp_path)
    blinded = datatype == "data" and not unblind
    chains = _analyzer_cut_chains()
    hists = _process_both_topologies(datatype, unblind=unblind)
    (payload,) = hists.values()
    args = SimpleNamespace(era="RunIISummer20UL18", sample="DYJets" if datatype == "mc" else "Muon",
                           mass=None, dir=None, name=None, systs=[], syst_observables=None,
                           omit_systs=[])
    path = save_histograms(hists, args)

    with uproot.open(path) as root_file:
        for directory, key in _CUTFLOW_KEYS.items():
            assert set(root_file[directory].keys(recursive=False, cycle=False)) == _CHANNELS[directory]
            for flavor, labels in chains[directory].items():
                if blinded and flavor in ("ee", "mumu"):
                    labels = labels[:labels.index(_BLINDED_FROM[directory])]
                for name in _CUTFLOW_NAMES:
                    saved = root_file[f"{directory}/{flavor}/{name}"]
                    assert list(saved.axis().labels()) == labels
                    in_memory = payload[key][flavor][name]
                    assert list(in_memory.axes[0]) == labels
                    np.testing.assert_array_equal(saved.values(), in_memory.values())
                    np.testing.assert_array_equal(saved.variances(), in_memory.variances())
        folders = set(root_file.keys(recursive=False, cycle=False))

    # Blinded data files carry no signal-region folders (nor variants of them).
    signal_folders = {folder for folder in folders if is_signal_region(folder)}
    if blinded:
        assert not signal_folders
    else:
        assert "wr_mumu_resolved_sr" in signal_folders
    manifest = json.loads(path.with_name(path.stem + "_systematics.json").read_text())
    assert manifest["blinding"]["applied"] is blinded


@pytest.fixture
def plotter_cutflow(monkeypatch):
    """The plotter's cutflow reader (WR_Plotter/wrplotter/cutflow.py), which
    takes the cut sequence from the bin labels of the analyzer's ROOT output."""
    plotter = Path(__file__).resolve().parents[1] / "WR_Plotter"
    if not (plotter / "wrplotter" / "cutflow.py").exists():
        pytest.skip("WR_Plotter submodule is not checked out")
    monkeypatch.syspath_prepend(str(plotter))
    from wrplotter import cutflow
    from wrplotter.histogram_cache import clear_cache
    yield cutflow
    clear_cache()  # close the plotter's cached file handles


@pytest.mark.parametrize("datatype", ["mc", "data"], ids=["mc", "data-blinded"])
def test_plotter_reads_back_the_analyzer_cut_chains(plotter_cutflow, datatype, tmp_path, monkeypatch):
    """From a file written by save_histograms the plotter reads exactly the
    analyzer's chains and contents: complete for MC; for blinded data, ee/mumu
    stop before the mll cut and em is complete."""
    monkeypatch.chdir(tmp_path)
    chains = _analyzer_cut_chains()
    hists = _process_both_topologies(datatype)
    (payload,) = hists.values()
    args = SimpleNamespace(era="RunIISummer20UL18", sample="DYJets" if datatype == "mc" else "Muon",
                           mass=None, dir=None, name=None, systs=[], syst_observables=None,
                           omit_systs=[])
    path = save_histograms(hists, args)

    for directory, key in _CUTFLOW_KEYS.items():
        topology = directory.removeprefix("cutflow_")
        for flavor, labels in chains[directory].items():
            if datatype == "data" and flavor in ("ee", "mumu"):
                labels = labels[:labels.index(_BLINDED_FROM[directory])]
            for name in _CUTFLOW_NAMES:
                kind = name.removesuffix("_unweighted")
                cutflow = plotter_cutflow.read_cutflow(path, topology, flavor, kind,
                                                       unweighted=name != kind)
                assert cutflow.cuts == tuple(labels)
                written = payload[key][flavor][name]
                np.testing.assert_array_equal(cutflow.values, written.values())
                np.testing.assert_array_equal(cutflow.variances, written.variances())


def test_plotter_refuses_to_sum_sequences_that_only_share_a_bin_count(plotter_cutflow, tmp_path):
    (payload,) = _process_both_topologies("mc").values()
    analyzer_file = tmp_path / "analyzer.root"
    with uproot.recreate(analyzer_file) as root_file:
        _save_cutflows(root_file, payload["cutflow"], "cutflow_resolved")
    labels = _analyzer_cut_chains()["cutflow_resolved"]["mumu"].copy()
    labels[-2], labels[-3] = labels[-3], labels[-2]
    other_file = tmp_path / "old_sequence.root"
    with uproot.recreate(other_file) as root_file:
        root_file["cutflow_resolved/mumu/cumulative"] = hist.Hist(
            hist.axis.StrCategory(labels, name="cut"), storage=hist.storage.Weight(),
        )
    # The labels come from the file, not from an assumed sequence ...
    assert plotter_cutflow.read_cutflow(other_file, "resolved", "mumu", "cumulative").cuts == tuple(labels)
    # ... so summing it with the analyzer's (e.g. across eras) is refused, not misaligned.
    with pytest.raises(ValueError, match="old_sequence.root"):
        plotter_cutflow.read_files([analyzer_file, other_file], "resolved", "mumu", "cumulative")


def test_cutflows_leave_out_the_dy_zpt_weight_and_end_with_the_region_sfs():
    """Every row uses the base weight without the DY Z-pT factor; the last row multiplies the chain's region
    factor (the lepton and LSF SFs) in, and is the base weight where no factor is given (data)."""
    weights = Weights(1, storeIndividual=True)
    weights.add("base", np.array([2.0]))
    weights.add("dy_zpt", np.array([1.5]))
    output = {}
    fill_cutflows(output, _PassAll(), weights, sf_factors={"ee": np.array([0.8])})
    fill_boosted_cutflows(output, _PassAll(), weights, sf_factors={"mumu": np.array([0.5])})
    for key, flavor, factor in (("cutflow", "ee", 0.8), ("cutflow", "mumu", 1.0), ("cutflow_boosted", "mumu", 0.5)):
        for name in ("cumulative", "onecut"):
            hist_ = output[key][flavor][name]
            assert list(hist_.axes[0])[-1] == CUTFLOW_SF_ROW
            np.testing.assert_allclose(hist_.values()[:-1], 2.0)
            assert hist_.values()[-1] == pytest.approx(2.0 * factor)
            assert hist_.variances()[-1] == pytest.approx((2.0 * factor) ** 2)
        assert output[key][flavor]["cumulative_unweighted"].values()[-1] == 1


def test_blinded_chains_have_no_region_sf_row():
    weights = Weights(1)
    weights.add("base", np.ones(1))
    output = {}
    fill_cutflows(output, _PassAll(), weights, blind=True)
    assert CUTFLOW_SF_ROW not in list(output["cutflow"]["ee"]["cumulative"].axes[0])
    assert list(output["cutflow"]["em"]["cumulative"].axes[0])[-1] == CUTFLOW_SF_ROW


@_apply_sf_patches
def _mumu_resolved_sr_chunk(trigger_sf):
    """One analyzer chunk (era UL18, MC) of three mumu resolved-SR events, with a muon trigger SF of *trigger_sf*."""
    from unittest.mock import patch
    n = 3
    events = _make_integration_events(n_events=n, n_muons=2, n_jets=2, n_fatjets=1, include_genweight=True)
    for collection, fields in (
        ("Muon", {"pt": [[300., 250.]], "eta": [[0., 0.]], "phi": [[0., np.pi]]}),
        ("Jet", {"pt": [[300., 300.]], "eta": [[0., 0.]], "phi": [[np.pi / 2, -np.pi / 2]]}),
    ):
        objects = getattr(events, collection)
        for field, values in fields.items():
            objects = ak.with_field(objects, ak.Array(values * n), field)
        setattr(events, collection, objects)

    def trigger(tight_muons, era, *args, **kwargs):
        m = len(tight_muons)
        return np.full(m, trigger_sf), np.full(m, trigger_sf), np.full(m, trigger_sf)

    with patch("wrcoffea.analyzer.muon_trigger_sf", new=trigger):
        return WrAnalysis(None, region="resolved").process(events)


def test_region_sf_row_equals_the_signal_region_yield():
    (payload,) = _mumu_resolved_sr_chunk(0.9).values()
    cumulative = payload["cutflow"]["mumu"]["cumulative"]
    labels, values = list(cumulative.axes[0]), cumulative.values()
    assert labels[-1] == CUTFLOW_SF_ROW and values[-2] > 0
    sr = payload["mass_fourobject"][{"region": "wr_mumu_resolved_sr", "syst": "Nominal"}].sum(flow=True).value
    assert values[-1] == pytest.approx(sr)
    assert values[-1] == pytest.approx(0.9 * values[-2])


def test_signal_flavour_masks_from_lhe_leptons_or_the_dataset_name():
    from wrcoffea.analyzer import signal_flavour_masks
    lhe = ak.zip({"pdgId": ak.Array([[2, -1, 11, -11, 1, 2], [2, -1, 13, -13, 1, 2], [2, -1, 1, 2]]),
                  "status": ak.Array([[-1, -1, 1, 1, 1, 1], [-1, -1, 1, 1, 1, 1], [-1, -1, 1, 1]])})
    masks = signal_flavour_masks(lhe, 3)
    assert masks["ee"].tolist() == [True, False, False]
    assert masks["mumu"].tolist() == [False, True, False]
    masks = signal_flavour_masks(None, 2, "WRtoNMutoMuMuJJ_MWR4000_N2100_TuneCP5_13p6TeV_madgraph-pythia8")
    assert masks["ee"].tolist() == [False, False] and masks["mumu"].tolist() == [True, True]
    assert signal_flavour_masks(None, 2, "WRtoNLtoLLJJ_MWR4000_MN2100") is None


def test_channel_masks_restrict_the_same_flavour_chains():
    """Two events: the ee chains count only the first, the mumu chains only the second, em both."""
    weights = Weights(2)
    weights.add("base", np.array([2.0, 3.0]))

    class _PassAll2:
        def all(self, *names):
            return np.ones(2, dtype=bool)

    output = {}
    masks = {"ee": np.array([True, False]), "mumu": np.array([False, True])}
    fill_cutflows(output, _PassAll2(), weights, channel_masks=masks)
    fill_boosted_cutflows(output, _PassAll2(), weights, channel_masks=masks)
    for key in ("cutflow", "cutflow_boosted"):
        for flavor, expected, count in (("ee", 2.0, 1), ("mumu", 3.0, 1), ("em", 5.0, 2)):
            for name in ("cumulative", "onecut"):
                np.testing.assert_allclose(output[key][flavor][name].values(), expected)
            np.testing.assert_array_equal(output[key][flavor]["cumulative_unweighted"].values(), count)


@_apply_sf_patches
def _signal_mu_chunk():
    """One analyzer chunk (era UL18) of three mumu resolved-SR events from a muon-flavour signal dataset."""
    n = 3
    metadata = {"datatype": "mc", "physics_group": "Signal", "era": "RunIISummer20UL18", "xsec": 1.0,
                "genEventSumw": 3.0, "sample": "WRtoNMutoMuMuJJ_MWR4000_N2100_TuneCP5_13TeV_madgraph-pythia8"}
    events = _make_integration_events(n_events=n, n_muons=2, n_jets=2, n_fatjets=1, metadata=metadata,
                                      include_genweight=True)
    for collection, fields in (
        ("Muon", {"pt": [[300., 250.]], "eta": [[0., 0.]], "phi": [[0., np.pi]]}),
        ("Jet", {"pt": [[300., 300.]], "eta": [[0., 0.]], "phi": [[np.pi / 2, -np.pi / 2]]}),
    ):
        objects = getattr(events, collection)
        for field, values in fields.items():
            objects = ak.with_field(objects, ak.Array(values * n), field)
        setattr(events, collection, objects)
    return WrAnalysis("WR4000_N2100", region="both", signal_kfactor=False).process(events)


def test_signal_ee_chains_skip_the_other_flavour():
    (payload,) = _signal_mu_chunk().values()
    for key in ("cutflow", "cutflow_boosted"):
        assert payload[key]["ee"]["cumulative_unweighted"].values().sum() == 0
        assert payload[key]["mumu"]["cumulative_unweighted"].values()[0] == 3
        assert payload[key]["em"]["cumulative_unweighted"].values()[0] == 3
    # the region histograms keep every event
    sr = payload["mass_fourobject"][{"region": "wr_mumu_resolved_sr", "syst": "Nominal"}].sum(flow=True).value
    assert sr == pytest.approx(payload["cutflow"]["mumu"]["cumulative"].values()[-1])
