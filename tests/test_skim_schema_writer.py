"""Branch policy, output schema, Runs collapsing and the part writer."""

import json

import numpy as np
import awkward as ak
import pytest
import uproot

from tests.nano_synth import write_nano_file
from wrcoffea.skim.schema import build_schema, branch_types_of_tree, load_policy, collection_of, OutputSchema
from wrcoffea.skim.runs import collapse_runs, merge_lumiblocks, concat_runs
from wrcoffea.skim.writer import PartWriter, InputRecord, read_sidecar, compression_from_spec
from wrcoffea.skim.selection import skim_mask, selection_branches, EVENT_ID_BRANCHES
from wrcoffea.skim.recount import ids_digest, IdDigest


def _table(path):
    with uproot.open(path) as f:
        return branch_types_of_tree(f["Events"])


def test_collection_of():
    assert collection_of("Jet_pt") == "Jet"
    assert collection_of("nJet") == "Jet"
    assert collection_of("HLT_Mu50") == "HLT"
    assert collection_of("run") is None
    assert collection_of("SoftActivityJetHT") is None


def test_schema_policy_decisions(tmp_path):
    a = tmp_path / "a.root"
    b = tmp_path / "b.root"
    write_nano_file(a, n=200, seed=1, extra_hlt=("HLT_Mu55",))
    write_nano_file(b, n=200, seed=2, extra_hlt=("HLT_Photon200",))
    s = build_schema([_table(a), _table(b)])
    names = set(s.names)
    # kept collections and flat branches
    assert {"run", "luminosityBlock", "event", "genWeight", "Jet_pt", "Muon_tunepRelPt", "FatJet_msoftdrop",
            "PSWeight", "Flag_goodVertices", "fixedGridRhoFastjetAll"} <= names
    # dropped: Tau and L1 bits
    assert "Tau_pt" not in names and "L1_SingleMu22" not in names
    assert "Tau_pt" in s.dropped and "L1_SingleMu22" in s.dropped
    # counters are never explicit branches
    assert not any(n.startswith("nJet") or n == "nMuon" for n in names)
    # union of trigger patterns across files, plus every required path (synthetic when absent)
    assert "HLT_Mu55" in names and "HLT_Photon200" in names
    assert "HLT_OldMu100" in names and "HLT_OldMu100" in s.synthetic
    assert "HLT_Mu50" in names and "HLT_Mu50" not in s.synthetic
    # unknown branch kept and reported
    assert "SomethingNew_x" in names and "SomethingNew_x" in s.unknown_kept
    # event ids first
    assert s.names[:3] == ["run", "luminosityBlock", "event"]
    types = s.uproot_types()
    assert types["Jet"].startswith("var * {") and "genJetIdx: int16" in types["Jet"]
    assert types["PSWeight"] == "var * float32"
    assert np.dtype(types["HLT_Mu55"]) == np.dtype(bool)
    # serialization round trip keeps the digest
    s2 = OutputSchema.from_dict(json.loads(json.dumps(s.to_dict())))
    assert s2.digest() == s.digest() and s2.synthetic == s.synthetic


def test_schema_drop_unknown_when_policy_says_so(tmp_path):
    a = tmp_path / "a.root"
    write_nano_file(a, n=50, seed=1)
    pol = load_policy()
    pol["unknown_policy"] = "drop"
    s = build_schema([_table(a)], pol)
    assert "SomethingNew_x" not in s.names and "SomethingNew_x" in s.dropped


def test_collapse_runs_mc_and_data():
    runs = {
        "run": ak.Array([1, 1, 1]),
        "genEventCount": ak.Array([100, 200, 300]),
        "genEventSumw": ak.Array([100.0, 200.0, 300.0]),
        "genEventSumw2": ak.Array([1.0, 2.0, 3.0]),
        "LHEScaleSumw": ak.Array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
    }
    c = collapse_runs(runs)
    assert ak.to_list(c["genEventCount"]) == [600]
    assert ak.to_list(c["genEventSumw"]) == [600.0]
    # weighted mean of ratios: (1*100+2*200+3*300)/600 = 2.3333
    assert ak.to_list(c["LHEScaleSumw"])[0] == pytest.approx([1400 / 600, 2800 / 600])
    assert ak.to_list(c["run"]) == [1]
    with pytest.raises(ValueError):
        collapse_runs({"genEventSumw": ak.Array([1.0, 1.0]), "LHEScaleSumw": ak.Array([[1.0], [1.0, 2.0]])})
    data = {"run": ak.Array([300, 301, 300, 302, 301])}
    d = collapse_runs(data)
    assert ak.to_list(d["run"]) == [300, 301, 302]


def test_merge_lumiblocks_dedupes():
    t1 = {"run": ak.Array([1, 1, 2]), "luminosityBlock": ak.Array([5, 6, 1]), "x": ak.Array([0.1, 0.2, 0.3])}
    t2 = {"run": ak.Array([1, 2]), "luminosityBlock": ak.Array([6, 2]), "x": ak.Array([9.9, 0.4])}
    m = merge_lumiblocks([t1, t2])
    assert list(zip(ak.to_list(m["run"]), ak.to_list(m["luminosityBlock"]))) == [(1, 5), (1, 6), (2, 1), (2, 2)]
    assert ak.to_list(m["x"]) == [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError):
        concat_runs([{"a": ak.Array([1])}, {"b": ak.Array([1])}])


def test_compression_spec():
    assert compression_from_spec("zstd:5").level == 5
    assert compression_from_spec("zlib").level == 1
    assert compression_from_spec("none") is None
    with pytest.raises(ValueError):
        compression_from_spec("bzip2:3")


def _skim_into(writer, path, lfn, schema, extra_missing=()):
    """Mimic the worker: select, digest, add chunks, end input."""
    with uproot.open(path) as f:
        t = f["Events"]
        keys = [k for k in t.keys() if k not in extra_missing]
        rec = InputRecord(lfn=lfn, n_in=t.num_entries, events_dbs=t.num_entries)
        writer.begin_input(rec, available_branches=keys)
        d = IdDigest()
        kept = [n for n in schema.names if n in keys]
        for start in range(0, t.num_entries, 700):
            stop = min(start + 700, t.num_entries)
            sel = t.arrays(selection_branches(keys) + list(EVENT_ID_BRANCHES), entry_start=start, entry_stop=stop)
            m = skim_mask(sel)
            if m.any():
                ids = sel[m]
                d.update(ids["run"], ids["luminosityBlock"], ids["event"])
                writer.add_chunk(t.arrays(kept, entry_start=start, entry_stop=stop)[m], int(m.sum()))
        rec.ids_digest = rec.recount_digest = d.hexdigest()
        runs = f["Runs"].arrays(library="ak")
        runs = {k: runs[k] for k in runs.fields}
        lumis = None
        if "LuminosityBlocks" in f:
            lb = f["LuminosityBlocks"].arrays(library="ak")
            lumis = {k: lb[k] for k in lb.fields}
    return writer.end_input(runs, lumis)


def test_writer_mc_rolls_parts_fills_and_verifies(tmp_path):
    inputs = []
    for i in range(3):
        p = tmp_path / f"in{i}.root"
        write_nano_file(p, n=2500, seed=10 + i, extra_hlt=("HLT_Mu55",) if i == 0 else ())
        inputs.append(p)
    schema = build_schema([_table(p) for p in inputs])
    w = PartWriter(tmp_path / "out", "DS", schema, is_data=False, compression="zlib:1", flush_events=900,
                   part_max_bytes=1, campaign={"campaign": "t"}, name_fmt="{base}_g{gid:04d}p{index:02d}.root",
                   name_fields={"gid": 3})
    recs = [_skim_into(w, p, f"/store/x/in{i}.root", schema) for i, p in enumerate(inputs)]
    parts = w.close()
    # part_max_bytes=1 -> one part per input (roll at every input boundary)
    assert len(parts) == 3 and all(len(p.inputs) == 1 for p in parts)
    assert parts[0].path.endswith("DS_g0003p01.root")
    for part, rec, p in zip(parts, recs, inputs):
        with uproot.open(part.path) as f:
            t = f["Events"]
            assert t.num_entries == rec.n_pass == part.n_events
            assert set(t.keys()) >= set(schema.names)
            ids = t.arrays(list(EVENT_ID_BRANCHES))
            assert ids_digest(ids["run"], ids["luminosityBlock"], ids["event"]) == rec.ids_digest
            runs = f["Runs"].arrays(library="ak")
            assert len(runs) == 1 and float(runs["genEventSumw"][0]) == pytest.approx(rec.gen_sumw)
            assert "LuminosityBlocks" not in f
            # the filled HLT_Mu55 is all False where the input lacked it
            if "HLT_Mu55" in rec.filled_branches:
                assert not np.any(t["HLT_Mu55"].array(library="np"))
            # dropped branches really gone, unknown kept
            assert "Tau_pt" not in t.keys() and "L1_SingleMu22" not in t.keys() and "SomethingNew_x" in t.keys()
        sc = read_sidecar(part.sidecar)
        assert sc["part"]["adler32"] == part.adler32 and sc["schema"]["digest"] == schema.digest()
        assert sc["part"]["inputs"][0]["lfn"] == rec.lfn
    assert "HLT_Mu55" in recs[1].filled_branches and "HLT_Mu55" not in recs[0].filled_branches
    # NanoEvents can read the output
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    NanoAODSchema.warn_missing_crossrefs = False
    ev = NanoEventsFactory.from_root({parts[0].path: "Events"}, schemaclass=NanoAODSchema, mode="virtual").events()
    assert "Jet" in ev.fields and "tunepRelPt" in ev.Muon.fields and "Mu55" in ev.HLT.fields


def test_writer_data_merges_lumis_and_runs(tmp_path):
    a = tmp_path / "a.root"
    b = tmp_path / "b.root"
    write_nano_file(a, n=1200, seed=20, is_data=True, run=300, lumi_start=1)
    write_nano_file(b, n=1200, seed=21, is_data=True, run=301, lumi_start=10)
    schema = build_schema([_table(a), _table(b)])
    w = PartWriter(tmp_path / "out", "Muon0_X", schema, is_data=True, flush_events=10_000, part_max_bytes=1e12)
    ra = _skim_into(w, a, "/store/d/a.root", schema)
    rb = _skim_into(w, b, "/store/d/b.root", schema)
    parts = w.close()
    assert len(parts) == 1 and parts[0].n_events == ra.n_pass + rb.n_pass
    assert (ra.entry_start, ra.entry_stop, rb.entry_start, rb.entry_stop) == (0, ra.n_pass, ra.n_pass, ra.n_pass + rb.n_pass)
    with uproot.open(parts[0].path) as f:
        runs = f["Runs"].arrays(library="np")
        assert sorted(runs["run"].tolist()) == [300, 301]
        lb = f["LuminosityBlocks"].arrays(library="np")
        assert len(lb["run"]) == ra.n_lumis + rb.n_lumis == parts[0].n_lumis
        assert "genWeight" not in f["Events"].keys()


def test_writer_abort_leaves_nothing(tmp_path):
    a = tmp_path / "a.root"
    write_nano_file(a, n=300, seed=30)
    schema = build_schema([_table(a)])
    w = PartWriter(tmp_path / "out", "DS", schema, is_data=False)
    with uproot.open(a) as f:
        t = f["Events"]
        w.begin_input(InputRecord(lfn="x"), available_branches=t.keys())
        w.add_chunk(t.arrays([n for n in schema.names if n in t.keys()]), t.num_entries)
    w.abort()
    assert list((tmp_path / "out").glob("*")) == []


def test_collapse_runs_ignores_rows_of_empty_inputs():
    """A zero-event input contributes a Runs row with genEventSumw 0 and empty weight vectors."""
    table = {
        "run": ak.Array(np.array([1, 1, 1], dtype=np.uint32)),
        "genEventCount": ak.Array(np.array([0, 100, 300], dtype=np.int64)),
        "genEventSumw": ak.Array(np.array([0.0, 100.0, 300.0])),
        "genEventSumw2": ak.Array(np.array([0.0, 100.0, 300.0])),
        "nLHEScaleSumw": ak.Array(np.array([0, 3, 3], dtype=np.int32)),
        "LHEScaleSumw": ak.Array([[], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    }
    out = collapse_runs(table)
    assert int(out["genEventCount"][0]) == 400 and float(out["genEventSumw"][0]) == 400.0
    assert int(out["nLHEScaleSumw"][0]) == 3                      # from an input that processed events
    np.testing.assert_allclose(ak.to_numpy(out["LHEScaleSumw"][0]), [2.5, 2.0, 1.5])   # 1:3 weighted, empty row ignored
    # a genuinely inconsistent table (non-empty rows of different lengths) still fails loudly
    bad = dict(table, LHEScaleSumw=ak.Array([[1.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]))
    with pytest.raises(ValueError, match="different lengths"):
        collapse_runs(bad)


def test_collapse_runs_input_with_events_but_no_lhe_weights():
    """A 2022 ZZZ file had events but no LHE scale/PDF weights (nLHEScaleSumw 0)."""
    table = {
        "run": ak.Array(np.array([1, 1, 1], dtype=np.uint32)),
        "genEventCount": ak.Array(np.array([50, 100, 300], dtype=np.int64)),
        "genEventSumw": ak.Array(np.array([50.0, 100.0, 300.0])),
        "genEventSumw2": ak.Array(np.array([50.0, 100.0, 300.0])),
        "nLHEScaleSumw": ak.Array(np.array([0, 3, 3], dtype=np.int32)),
        "LHEScaleSumw": ak.Array([[], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    }
    out = collapse_runs(table)
    assert float(out["genEventSumw"][0]) == 450.0                 # every input counts for the normalisation
    assert int(out["nLHEScaleSumw"][0]) == 3                      # counter from an input that has the weights
    np.testing.assert_allclose(ak.to_numpy(out["LHEScaleSumw"][0]), [2.5, 2.0, 1.5])


def _scan_runs_table(models, sumw, scale):
    """One input's Runs table as in the pre-legacy NanoAODv7 scans: run + per-model branches only."""
    t = {"run": ak.Array(np.array([1], np.uint32))}
    for m, w, sc in zip(models, sumw, scale):
        t[f"genEventCount_{m}"] = ak.Array(np.array([100], np.uint64))
        t[f"genEventSumw_{m}"] = ak.Array(np.array([w], np.float64))
        t[f"genEventSumw2_{m}"] = ak.Array(np.array([w * 2.0], np.float64))
        t[f"nLHEScaleSumw_{m}"] = ak.Array(np.array([3], np.uint32))
        t[f"LHEScaleSumw_{m}"] = ak.Array([np.array(sc, np.float32)])
    return t


def test_scan_runs_with_different_models_per_input_are_padded_and_collapsed_per_model():
    from wrcoffea.skim.runs import is_scan_runs, pad_scan_runs, collapse_scan_runs, per_model_branch
    a, b, c = ("WRtoNLtoLLJJ_MWR5200_MN1000_TuneCP5_13TeV_madgraph_pythia8",
               "WRtoNLtoLLJJ_MWR5200_MN1200_TuneCP5_13TeV_madgraph_pythia8",
               "WRtoNLtoLLJJ_MWR5400_MN1000_TuneCP5_13TeV_madgraph_pythia8")
    t1 = _scan_runs_table([a, b], [10.0, 30.0], [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    t2 = _scan_runs_table([b, c], [10.0, 5.0], [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]])
    assert is_scan_runs(t1) and not is_scan_runs({"genEventSumw": ak.Array([1.0]), "genEventSumw_": ak.Array([1.0])})
    assert per_model_branch(f"genEventSumw2_{a}") == ("genEventSumw2", a) and per_model_branch("genEventSumw_") is None
    with pytest.raises(ValueError):
        concat_runs([t1, t2])          # what the worker used to hit
    padded = pad_scan_runs([t1, t2])
    assert set(padded[0]) == set(padded[1])
    out = collapse_scan_runs(concat_runs(padded))
    assert int(out["run"][0]) == 1
    assert float(out[f"genEventSumw_{a}"][0]) == 10.0 and float(out[f"genEventSumw_{b}"][0]) == 40.0
    assert float(out[f"genEventSumw2_{c}"][0]) == 10.0
    assert int(out[f"genEventCount_{b}"][0]) == 200 and out[f"genEventCount_{b}"].type.content.primitive == "uint64"
    assert f"nLHEScaleSumw_{c}" not in out   # counters are written by uproot from the vectors
    # ratios: mean weighted by the model's own sumw over the inputs that contain it
    assert np.allclose(ak.to_numpy(out[f"LHEScaleSumw_{a}"][0]), [1.0, 2.0, 3.0])
    assert np.allclose(ak.to_numpy(out[f"LHEScaleSumw_{b}"][0]), [(30 * 1 + 10 * 3) / 40.0] * 3)
    assert np.allclose(ak.to_numpy(out[f"LHEScaleSumw_{c}"][0]), [2.0, 2.0, 2.0])
