"""Theory-weight sums derived from the events when the input's Runs tree lacks them (PSSumw in UL
NanoAODv9): the accumulation, the derived Runs columns, the worker end to end, the collapse over
inputs (including an empty one) and the native branches left alone."""
import numpy as np
import awkward as ak
import pytest
import uproot

from tests.nano_synth import write_nano_file
from wrcoffea.skim.runs import collapse_runs, concat_runs, pad_ratio_branches
from wrcoffea.skim.schema import branch_types_of_tree, build_schema
from wrcoffea.skim.weight_sums import EventWeightSums, RATIO_FROM_EVENTS, add_derived_runs_columns, derivable
from wrcoffea.skim.worker import skim_input
from wrcoffea.skim.writer import InputRecord, PartWriter, read_sidecar


def _expected(path, branch="PSWeight"):
    """(mean of genWeight * w_i over ALL events / sum genWeight, sum genWeight) of one input file."""
    with uproot.open(path) as f:
        a = f["Events"].arrays(["genWeight", branch])
    gw = np.asarray(ak.to_numpy(a["genWeight"]), dtype=np.float64)
    w = np.asarray(ak.to_numpy(a[branch]), dtype=np.float64)
    return (w * gw[:, None]).sum(axis=0) / gw.sum(), float(gw.sum())


def _schema(*paths):
    tables = []
    for p in paths:
        with uproot.open(p) as f:
            tables.append(branch_types_of_tree(f["Events"]))
    return build_schema(tables)


def _writer(tmp_path, **kw):
    kw.setdefault("part_max_bytes", 1e12)
    return PartWriter(tmp_path / "out", "DS", kw.pop("schema"), is_data=False, compression="zlib:1",
                      flush_events=500, campaign={"campaign": "t"}, **kw)


# ---------------------------------------------------------------- unit level

def test_derivable_needs_genweight_and_the_event_branch():
    assert derivable(["genEventSumw", "LHEScaleSumw"], ["genWeight", "PSWeight", "LHEScaleWeight"]) == ["PSSumw"]
    assert derivable([], ["genWeight", "PSWeight", "LHEScaleWeight", "LHEPdfWeight"]) == list(RATIO_FROM_EVENTS)
    assert derivable(["genEventSumw"], ["PSWeight"]) == []            # data-like: no genWeight
    assert derivable(["PSSumw"], ["genWeight", "PSWeight"]) == []       # already there


def test_sums_over_chunks_equal_the_direct_mean():
    rng = np.random.default_rng(1)
    gw = rng.choice([-1.0, 1.0, 1.0], 900) * 3.0
    ps = rng.uniform(0.8, 1.2, (900, 4))
    sums = EventWeightSums(["PSSumw"])
    assert sums.branches == ["genWeight", "PSWeight"]
    for lo in range(0, 900, 250):
        sums.add({"genWeight": ak.Array(gw[lo:lo + 250]), "PSWeight": ak.Array(ps[lo:lo + 250])})
    assert sums.n_events == 900 and sums.sumw == pytest.approx(gw.sum())
    np.testing.assert_allclose(sums.ratios()["PSSumw"], (ps * gw[:, None]).sum(axis=0) / gw.sum())


def test_varying_entry_count_is_refused():
    sums = EventWeightSums(["PSSumw"])
    w = ak.unflatten(np.ones(9, dtype=np.float32), np.array([4, 4, 1]))
    with pytest.raises(ValueError, match="entries per event"):
        sums.add({"genWeight": ak.Array(np.ones(3)), "PSWeight": w})
    sums2 = EventWeightSums(["PSSumw"])
    sums2.add({"genWeight": ak.Array(np.ones(2)), "PSWeight": ak.Array(np.ones((2, 4)))})
    with pytest.raises(ValueError, match="after 4 earlier"):
        sums2.add({"genWeight": ak.Array(np.ones(2)), "PSWeight": ak.Array(np.ones((2, 2)))})
    with pytest.raises(ValueError, match="not derivable"):
        EventWeightSums(["genEventSumw"])


def test_no_events_gives_empty_vectors():
    sums = EventWeightSums(["PSSumw"])
    assert sums.ratios()["PSSumw"].size == 0
    sums.add({"genWeight": ak.Array(np.zeros(0)), "PSWeight": ak.unflatten(np.zeros(0, np.float32), np.zeros(0, np.int64))})
    assert sums.n_events == 0 and sums.ratios()["PSSumw"].size == 0


def test_add_derived_runs_columns_one_identical_row_per_entry():
    sums = EventWeightSums(["PSSumw", "LHEScaleSumw"])
    sums.add({"genWeight": ak.Array([1.0, 3.0]), "PSWeight": ak.Array([[1.0, 2.0], [1.0, 0.0]]),
              "LHEScaleWeight": ak.Array([[0.5], [0.5]])})
    table = {"genEventSumw": ak.Array([4.0, 4.0]), "LHEScaleSumw": ak.Array([[0.9], [0.9]])}
    out = add_derived_runs_columns(table, sums)
    assert out["PSSumw"].ndim == 2 and ak.to_list(ak.num(out["PSSumw"], axis=1)) == [2, 2]
    np.testing.assert_allclose(ak.to_numpy(ak.flatten(out["PSSumw"], axis=1)), [1.0, 0.5, 1.0, 0.5])
    assert ak.to_numpy(ak.flatten(out["PSSumw"], axis=1)).dtype == np.float32
    assert ak.to_list(out["LHEScaleSumw"]) == [[0.9], [0.9]]      # native branch untouched
    assert add_derived_runs_columns({}, sums) == {}


# ------------------------------------------------------------ worker + writer

def test_skim_input_derives_pssumw_over_all_events_when_runs_lacks_it(tmp_path):
    p = tmp_path / "in.root"
    write_nano_file(p, n=3000, seed=5, runs_drop=("PSSumw",), n_runs_entries=2)
    with uproot.open(p) as f:
        assert "PSSumw" not in f["Runs"].keys()
    schema = _schema(p)
    w = _writer(tmp_path, schema=schema)
    rec = skim_input(str(p), InputRecord(lfn="/store/x/in.root"), w, schema, is_data=False, chunk_entries=700)
    (part,) = w.close()
    expected, sumw = _expected(p)
    assert rec.derived_runs_branches == ["PSSumw"]
    assert 0 < rec.n_pass < rec.n_in == 3000
    with uproot.open(part.path) as f:
        runs = f["Runs"].arrays(library="ak")
        assert len(runs) == 1
        np.testing.assert_allclose(ak.to_numpy(runs["PSSumw"][0]), expected, rtol=1e-6)
        assert int(runs["nPSSumw"][0]) == 4
        assert float(runs["genEventSumw"][0]) == pytest.approx(sumw)
        np.testing.assert_allclose(ak.to_numpy(runs["LHEScaleSumw"][0]), np.linspace(0.9, 1.1, 9), rtol=1e-6)
        # the mean is over every input event, not over the selected ones
        sel = f["Events"].arrays(["genWeight", "PSWeight"])
        gw = np.asarray(ak.to_numpy(sel["genWeight"]), dtype=np.float64)
        sel_mean = (np.asarray(ak.to_numpy(sel["PSWeight"]), dtype=np.float64) * gw[:, None]).sum(axis=0) / gw.sum()
        assert np.abs(sel_mean - expected).max() > 1e-6
    sc = read_sidecar(part.sidecar)
    assert sc["part"]["runs"]["PSSumw"] == pytest.approx(expected.tolist(), rel=1e-6)
    assert sc["part"]["inputs"][0]["derived_runs_branches"] == ["PSSumw"]


def test_native_pssumw_is_kept_not_recomputed(tmp_path):
    p = tmp_path / "in.root"
    write_nano_file(p, n=1500, seed=6)
    schema = _schema(p)
    w = _writer(tmp_path, schema=schema)
    rec = skim_input(str(p), InputRecord(lfn="/store/x/in.root"), w, schema, is_data=False, chunk_entries=700)
    (part,) = w.close()
    assert rec.derived_runs_branches == []
    with uproot.open(part.path) as f:
        np.testing.assert_allclose(ak.to_numpy(f["Runs"]["PSSumw"].array()[0]), [1.0, 1.01, 0.99, 1.0], rtol=1e-6)


def test_data_input_derives_nothing(tmp_path):
    p = tmp_path / "data.root"
    write_nano_file(p, n=800, seed=7, is_data=True)
    schema = _schema(p)
    w = PartWriter(tmp_path / "out", "DATA", schema, is_data=True, compression="zlib:1", flush_events=500,
                   part_max_bytes=1e12, campaign={"campaign": "t"})
    rec = skim_input(str(p), InputRecord(lfn="/store/x/data.root"), w, schema, is_data=True, chunk_entries=300)
    (part,) = w.close()
    assert rec.derived_runs_branches == []
    with uproot.open(part.path) as f:
        assert "PSSumw" not in f["Runs"].keys()


def test_inputs_collapse_with_their_geneventsumw_as_weights(tmp_path):
    paths = [tmp_path / "a.root", tmp_path / "b.root"]
    write_nano_file(paths[0], n=2000, seed=8, runs_drop=("PSSumw",))
    write_nano_file(paths[1], n=1000, seed=9, runs_drop=("PSSumw",))
    schema = _schema(*paths)
    w = _writer(tmp_path, schema=schema)
    for i, p in enumerate(paths):
        skim_input(str(p), InputRecord(lfn=f"/store/x/{p.name}"), w, schema, is_data=False, chunk_entries=600)
    (part,) = w.close()                       # one part holding both inputs
    (ra, sa), (rb, sb) = _expected(paths[0]), _expected(paths[1])
    with uproot.open(part.path) as f:
        runs = f["Runs"].arrays(library="ak")
        assert len(runs) == 1 and float(runs["genEventSumw"][0]) == pytest.approx(sa + sb)
        np.testing.assert_allclose(ak.to_numpy(runs["PSSumw"][0]), (ra * sa + rb * sb) / (sa + sb), rtol=1e-6)


def test_empty_input_without_the_derived_branch_is_padded(tmp_path):
    derived = {"genEventSumw": ak.Array([100.0]), "genEventCount": ak.Array(np.array([10])),
               "LHEScaleSumw": ak.Array([[1.0] * 9]), "PSSumw": ak.Array([[1.1, 0.9, 1.0, 1.0]])}
    empty = {"genEventSumw": ak.Array([0.0]), "genEventCount": ak.Array(np.array([0])),
             "LHEScaleSumw": ak.unflatten(np.zeros(0, np.float32), np.array([0]))}
    padded = pad_ratio_branches([derived, empty])
    assert ak.to_list(padded[1]["PSSumw"]) == [[]]
    out = collapse_runs(concat_runs(padded))
    assert ak.to_list(out["PSSumw"][0]) == pytest.approx([1.1, 0.9, 1.0, 1.0])
    assert float(out["genEventSumw"][0]) == 100.0
    # through the writer: an empty input after one that derived PSSumw.  An empty NanoAOD file's Runs
    # tree has the same branches as a full one (zero sums, empty vectors), minus the derived PSSumw.
    p = tmp_path / "in.root"
    write_nano_file(p, n=1200, seed=11, runs_drop=("PSSumw",))
    with uproot.open(p) as f:
        runs_in = f["Runs"].arrays(library="ak")
    empty_file = {}
    for name in runs_in.fields:
        arr = runs_in[name]
        if arr.ndim > 1:
            inner = ak.to_numpy(ak.flatten(arr, axis=1)).dtype
            empty_file[name] = ak.unflatten(np.zeros(0, dtype=inner), np.array([0]))
        else:
            empty_file[name] = ak.Array(np.zeros(1, dtype=ak.to_numpy(arr).dtype))
    assert "PSSumw" not in empty_file
    schema = _schema(p)
    w = _writer(tmp_path, schema=schema)
    skim_input(str(p), InputRecord(lfn="/store/x/in.root"), w, schema, is_data=False, chunk_entries=400)
    w.begin_input(InputRecord(lfn="/store/x/empty.root"))
    w.end_input(empty_file, None)
    (part,) = w.close()
    expected, _ = _expected(p)
    with uproot.open(part.path) as f:
        np.testing.assert_allclose(ak.to_numpy(f["Runs"]["PSSumw"].array()[0]), expected, rtol=1e-6)
