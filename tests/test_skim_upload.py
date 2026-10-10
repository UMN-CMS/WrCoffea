"""Tests for the fileset writer's normalisation: sums from the skims, samples split over several datasets."""
import pytest

from wrcoffea.skim.plan import DatasetPlan
from wrcoffea.skim.upload import combine_extension_sumw, dataset_metadata_from_parts


def _entry(sample, sumw, xsec=19.308, count=10, datatype="mc"):
    return {"files": {f"{sample}.root": "Events"},
            "metadata": {"sample": sample, "xsec": xsec, "genEventSumw": sumw, "genEventCount": count,
                         "datatype": datatype}}


NOM = "/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM"
EXT = "/TWminustoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM"
OTHER = "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM"


def test_nominal_and_extension_under_different_names_share_one_normalisation():
    fs = {NOM: _entry("TWminustoLNu2Q", 167.1), EXT: _entry("TWminustoLNu2Q_ext1", 176.3), OTHER: _entry("TTto2L2Nu", 5.0, xsec=98.0)}
    notes = combine_extension_sumw(fs)
    assert len(notes) == 1
    for k in (NOM, EXT):
        assert fs[k]["metadata"]["genEventSumw"] == pytest.approx(343.4)
        assert fs[k]["metadata"]["genEventCount"] == 20
        assert fs[k]["metadata"]["normalised_with"] == ["TWminustoLNu2Q", "TWminustoLNu2Q_ext1"]
    assert fs[NOM]["metadata"]["genEventSumw_own"] == 167.1 and fs[EXT]["metadata"]["genEventSumw_own"] == 176.3
    assert fs[OTHER]["metadata"]["genEventSumw"] == 5.0 and "normalised_with" not in fs[OTHER]["metadata"]
    # the sample's total weight is xsec x L once: sum over entries of (own sumw / combined sumw) = 1
    assert sum(fs[k]["metadata"]["genEventSumw_own"] / fs[k]["metadata"]["genEventSumw"] for k in (NOM, EXT)) == pytest.approx(1.0)


def test_combining_twice_is_stable():
    fs = {NOM: _entry("TWminustoLNu2Q", 167.1), EXT: _entry("TWminustoLNu2Q_ext1", 176.3)}
    combine_extension_sumw(fs)
    combine_extension_sumw(fs)
    assert fs[NOM]["metadata"]["genEventSumw"] == pytest.approx(343.4)


def test_different_cross_sections_for_one_sample_are_an_error():
    fs = {NOM: _entry("TWminustoLNu2Q", 1.0, xsec=19.3), EXT: _entry("TWminustoLNu2Q_ext1", 1.0, xsec=20.0)}
    with pytest.raises(ValueError, match="different cross sections"):
        combine_extension_sumw(fs)


def test_data_entries_are_left_alone():
    a = "/Muon/Run2022C-22Sep2023-v1/NANOAOD"
    b = "/Muon/Run2022C-22Sep2023-v2/NANOAOD"
    fs = {a: _entry("Muon_Run2022C", None, xsec=None, datatype="data"), b: _entry("Muon_Run2022C_v2", None, xsec=None, datatype="data")}
    assert combine_extension_sumw(fs) == []


def _plan(datatype="mc", metadata=None):
    return DatasetPlan(key="TTto2L2Nu", era="RunIII2024Summer24", run="Run3", year="2024", datatype=datatype,
                       physics_group="tt_tW" if datatype == "mc" else "Muon", category="backgrounds",
                       das_paths=["/TTto2L2Nu/RunIII2024Summer24/NANOAODSIM"], metadata=dict(metadata or {}),
                       files=[], groups=[], schema_id="s")


def _sidecar(n_events, runs=None):
    part = {"path": "part.root", "n_events": n_events}
    if runs is not None:
        part["runs"] = runs
    return {"campaign": {"campaign": "v2"}, "part": part}


def test_mc_sum_comes_from_the_skims_never_from_the_plan():
    """A genEventSumw left in the plan metadata (old configs carried one) is ignored."""
    ds = _plan(metadata={"xsec": 98.0, "genEventSumw": 1.0})
    meta = dataset_metadata_from_parts(ds, [_sidecar(10, {"genEventSumw": 300.0, "genEventCount": 30}),
                                            _sidecar(5, {"genEventSumw": 200.0, "genEventCount": 20})])
    assert meta["genEventSumw"] == 500.0 and meta["genEventCount"] == 50 and meta["nevts"] == 15


def test_mc_without_skim_sums_is_an_error():
    """No fallback: a dataset whose parts carry no Runs sums stops the fileset step."""
    ds = _plan(metadata={"xsec": 98.0, "genEventSumw": 1.0})
    with pytest.raises(ValueError, match="no genEventSumw"):
        dataset_metadata_from_parts(ds, [_sidecar(10), _sidecar(5, {"genEventCount": 5})])


def test_data_needs_no_sum():
    meta = dataset_metadata_from_parts(_plan("data"), [_sidecar(10)])
    assert "genEventSumw" not in meta and meta["nevts"] == 10
