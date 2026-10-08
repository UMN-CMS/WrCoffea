"""Scan samples: per-model sums in the worker and the split into per-model datasets."""

import json

import numpy as np
import awkward as ak
import uproot

from tests.nano_synth import make_events, _collections, _types
from wrcoffea.skim.catalog import FileRecord
from wrcoffea.skim.plan import Plan, DatasetPlan, make_groups
from wrcoffea.skim.schema import build_schema, branch_types_of_tree
from wrcoffea.skim.worker import process_group, load_group_status
from wrcoffea.skim.scan import split_scan, model_name
from wrcoffea.skim.validate import validate_dataset
from wrcoffea.skim.writer import read_sidecar
from wrcoffea.skim.fetch import adler32_of

MODELS = ["GenModel_WRtoNLtoLLJJ_MWR1000_MN100_TuneCP5_13TeV_madgraph_pythia8",
          "GenModel_WRtoNLtoLLJJ_MWR1000_MN500_TuneCP5_13TeV_madgraph_pythia8",
          "GenModel_WRtoNLtoLLJJ_MWR2000_MN100_TuneCP5_13TeV_madgraph_pythia8"]


def _write_scan_file(path, n, seed, models_present):
    ev = make_events(n, seed)
    rng = np.random.default_rng(seed)
    which = rng.integers(0, len(models_present), n)
    for i, m in enumerate(models_present):
        ev[m] = which == i
    data = _collections(ev)
    with uproot.recreate(str(path)) as f:
        f.mktree("Events", _types(data)); f["Events"].extend(data)
        w = ev["genWeight"].astype(np.float64)
        runs = {"run": np.array([1], np.uint32), "genEventCount": np.array([n], np.int64), "genEventSumw": np.array([w.sum()]),
                "genEventSumw2": np.array([(w * w).sum()]), "LHEScaleSumw": ak.Array([np.ones(9, np.float32)]), "PSSumw": ak.Array([np.ones(4, np.float32)])}
        f.mktree("Runs", _types(runs)); f["Runs"].extend(runs)
    return ev


def test_scan_sums_and_split(tmp_path):
    # file 0 carries models 0,1; file 1 carries models 1,2 (flags differ between files, as in the real scans)
    (tmp_path / "in").mkdir()
    ev0 = _write_scan_file(tmp_path / "in" / "s0.root", 1500, 1, MODELS[:2])
    ev1 = _write_scan_file(tmp_path / "in" / "s1.root", 1500, 2, MODELS[1:])
    files, tables = [], []
    for p in sorted((tmp_path / "in").glob("*.root")):
        with uproot.open(p) as f:
            tables.append(branch_types_of_tree(f["Events"]))
        files.append(FileRecord(lfn=str(p), size=p.stat().st_size, events=1500, adler32=adler32_of(str(p)), valid=True, dataset="scan"))
    schema = build_schema(tables)
    assert all(m in schema.names for m in MODELS)   # union across files, kept as unknown flat branches
    ds = DatasetPlan(key="WRscan", era="RunIISummer20UL18", run="RunII", year="2018", datatype="mc", physics_group="Signal",
                     category="signals", das_paths=["/scan/x/NANOAODSIM"], metadata={}, files=[f.to_dict() for f in files],
                     groups=make_groups(files, 1e12), schema_id=schema.digest(), dbs_summary={"num_file": 2, "num_event": 3000, "file_size": 0},
                     out_dir="RunII/2018/RunIISummer20UL18/signals/WRscan", scan=True)
    from wrcoffea.skim.selection import SKIM_CUTS
    plan = Plan(campaign="unit", created="now", code_version="t", cuts=dict(SKIM_CUTS), policy_version=2,
                settings={"compression": "zlib:1", "flush_events": 500, "part_max_bytes": 1e12}, datasets={ds.key: ds}, schemas={schema.digest(): schema.to_dict()})
    plan.save(tmp_path / "plan"); plan = Plan.load(tmp_path / "plan")
    out = tmp_path / "out"
    st = process_group(plan, "WRscan", 0, out_root=out, scratch=tmp_path / "scr", prefetch=False)
    assert st.status == "done", st.error
    sc = read_sidecar(st.parts[0]["sidecar"])
    recs = {r["lfn"].split("/")[-1]: r for r in sc["part"]["inputs"]}
    # generated sums per model match the truth (before any selection)
    for name, ev, present in (("s0.root", ev0, MODELS[:2]), ("s1.root", ev1, MODELS[1:])):
        for m in present:
            flag = ev[m]
            assert recs[name]["scan_counts"][m] == int(flag.sum())
            assert np.isclose(recs[name]["scan_sumw"][m], float(ev["genWeight"][flag].sum()))
    # split
    priv = {model_name(m): {"xsec": 1.0 + i, "physics_group": "Signal"} for i, m in enumerate(MODELS)}
    new = split_scan(plan, "WRscan", out, model_meta=priv, models_per_pass=2, flush_events=100, plan_dir=tmp_path / "plan")
    assert sorted(new) == sorted(model_name(m) for m in MODELS)
    total = 0
    for name in new:
        d = plan.datasets[name]
        assert d.metadata["xsec"] == priv[name]["xsec"] and d.category == "signals"
        stm = load_group_status(out / d.out_dir, 0)
        assert stm is not None and stm.status == "done"
        rep = validate_dataset(plan, name, out, reread=True)
        assert rep["ok"], rep["problems"]
        with uproot.open(stm.parts[0]["path"]) as f:
            assert not any(k.startswith("GenModel_") for k in f["Events"].keys())
            runs = f["Runs"].arrays(library="np")
            m = next(mm for mm in MODELS if model_name(mm) == name)
            expected = sum(int(r["scan_counts"].get(m, 0)) for r in sc["part"]["inputs"])
            assert int(runs["genEventCount"].sum()) == expected
        total += stm.events_out
    assert total == st.events_out   # every skimmed event lands in exactly one model dataset
