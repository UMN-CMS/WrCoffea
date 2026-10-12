"""`skim.py fileset`: the selection flags, merging into existing family files, and unfinished datasets."""
import argparse
import copy
import json

import pytest

from tests.test_skim_worker import _local_plan
from wrcoffea.skim.plan import Plan
from wrcoffea.skim.upload import write_filesets
from wrcoffea.skim.worker import process_group

SUB = "Run3/2022/Run3Summer22"
DAS = {"TTtest": "/TTtest/Run3Summer22NanoAODv12-x/NANOAODSIM", "TTtest2": "/TTtest2/Run3Summer22NanoAODv12-x/NANOAODSIM",
       "Muon0_RunX": "/Muon0/Run2022X-x/NANOAOD", "Optional": "/Optional/Run3Summer22NanoAODv12-x/NANOAODSIM"}


@pytest.fixture
def campaign(tmp_path):
    """Plan with TTtest (finished), TTtest2 (not started) and Muon0_RunX (finished), and their configs."""
    (tmp_path / "mc").mkdir()
    (tmp_path / "data").mkdir()
    mc_plan, mc_key = _local_plan(tmp_path / "mc", n_files=1)
    data_plan, data_key = _local_plan(tmp_path / "data", n_files=1, is_data=True)
    ds2 = copy.deepcopy(mc_plan.datasets[mc_key])
    ds2.key, ds2.out_dir = "TTtest2", ds2.out_dir.replace("TTtest", "TTtest2")
    plan = mc_plan
    plan.datasets.update({"TTtest2": ds2, data_key: data_plan.datasets[data_key]})
    plan.schemas.update(data_plan.schemas)
    plan.save(tmp_path / "plan")
    plan = Plan.load(tmp_path / "plan")
    out = tmp_path / "out"
    for key in (mc_key, data_key):
        for gid in range(len(plan.datasets[key].groups)):
            assert process_group(plan, key, gid, out_root=out, scratch=tmp_path / "scr", prefetch=False).status == "done"
    cfg = tmp_path / "configs" / SUB
    cfg.mkdir(parents=True)
    mc = {"physics_group": "tt_tW", "datatype": "mc", "xsec": 1.0}
    (cfg / "Run3Summer22_mc_dy_lo_ht.json").write_text(json.dumps({
        DAS["TTtest"]: {**mc, "dataset": "TTtest"}, DAS["TTtest2"]: {**mc, "dataset": "TTtest2"},
        DAS["Optional"]: {**mc, "dataset": "Optional", "optional": True}}))
    (cfg / "Run3Summer22_data.json").write_text(json.dumps({
        DAS["Muon0_RunX"]: {"physics_group": "Muon", "datatype": "data", "dataset": "Muon0_RunX"}}))
    return tmp_path


def _run(root, **kw):
    args = argparse.Namespace(plan_dir=str(root / "plan"), out=str(root / "out"), remote="root://r//store/skims",
                              local=False, fileset_dir=str(root / "filesets"), config_root=str(root / "configs"),
                              era=["Run3Summer22"], dataset=None, physics_group=None, datatype=None, exclude=None,
                              allow_missing=False)
    for k, v in kw.items():
        setattr(args, k, v)
    return write_filesets(args)


def _read(root, family):
    p = root / "filesets" / SUB / "skimmed" / f"Run3Summer22_{family}_fileset.json"
    return json.loads(p.read_text()) if p.is_file() else None


def test_a_family_with_an_unfinished_dataset_is_refused(campaign, caplog):
    assert _run(campaign) == 1
    assert _read(campaign, "mc_dy_lo_ht") is None                       # TTtest2 unfinished: not written
    assert "TTtest2 (0/1 groups done)" in caplog.text and "NOT written" in caplog.text
    assert "Optional" not in caplog.text.split("NOT written")[1]       # optional, unplanned: not missing
    assert set(_read(campaign, "data")) == {DAS["Muon0_RunX"]}          # complete family still written


def test_allow_missing_writes_without_the_unfinished_datasets(campaign, caplog):
    assert _run(campaign, allow_missing=True) == 0
    assert set(_read(campaign, "mc_dy_lo_ht")) == {DAS["TTtest"]}
    assert "WITHOUT 1 unfinished" in caplog.text and "TTtest2" in caplog.text


def test_datatype_selects_families(campaign):
    assert _run(campaign, datatype="data") == 0
    assert _read(campaign, "mc_dy_lo_ht") is None and set(_read(campaign, "data")) == {DAS["Muon0_RunX"]}
    (campaign / "filesets").rename(campaign / "filesets_data")
    assert _run(campaign, datatype="mc") == 1                           # MC family incomplete, data untouched
    assert _read(campaign, "data") is None


def test_exclude_and_physics_group(campaign):
    assert _run(campaign, exclude=["TTtest2"]) == 0
    assert set(_read(campaign, "mc_dy_lo_ht")) == {DAS["TTtest"]}
    (campaign / "filesets").rename(campaign / "filesets_1")
    assert _run(campaign, physics_group=["Muon"]) == 0
    assert _read(campaign, "mc_dy_lo_ht") is None and set(_read(campaign, "data")) == {DAS["Muon0_RunX"]}


def test_a_narrowed_run_merges_into_the_existing_family_file(campaign):
    p = campaign / "filesets" / SUB / "skimmed" / "Run3Summer22_mc_dy_lo_ht_fileset.json"
    p.parent.mkdir(parents=True)
    old = {DAS["TTtest"]: {"files": {"old.root": "Events"}, "metadata": {"sample": "TTtest", "xsec": 1.0}},
           DAS["TTtest2"]: {"files": {"kept.root": "Events"}, "metadata": {"sample": "TTtest2", "xsec": 1.0}}}
    p.write_text(json.dumps(old))
    assert _run(campaign, dataset=["TTtest"], exclude=["TTtest2"]) == 0
    new = _read(campaign, "mc_dy_lo_ht")
    assert new[DAS["TTtest2"]] == old[DAS["TTtest2"]]                   # not selected: kept as it was
    assert list(new[DAS["TTtest"]]["files"]) != ["old.root"]            # selected: rewritten from the campaign
    assert all(f.startswith("root://r//store/skims/") for f in new[DAS["TTtest"]]["files"])
    assert new[DAS["TTtest"]]["metadata"]["genEventSumw"] > 0


def test_a_full_run_rewrites_the_family_file(campaign):
    _run(campaign, allow_missing=True)
    p = campaign / "filesets" / SUB / "skimmed" / "Run3Summer22_mc_dy_lo_ht_fileset.json"
    stale = json.loads(p.read_text())
    stale["/Stale/x/NANOAODSIM"] = {"files": {}, "metadata": {"sample": "Stale"}}
    p.write_text(json.dumps(stale))
    assert _run(campaign, allow_missing=True) == 0
    assert "/Stale/x/NANOAODSIM" not in _read(campaign, "mc_dy_lo_ht")


def _sel_args(root, **kw):
    a = argparse.Namespace(plan_dir=str(root / "plan"), out=str(root / "out"), era=["Run3Summer22"], dataset=None,
                           physics_group=None, datatype="mc", exclude=None)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_validate_honours_exclude(campaign):
    from wrcoffea.skim.validate import validate_campaign
    base = dict(skip_reread=False, remote=None, remote_lumis=False, accept_bad_files=False, report=str(campaign / "rep.json"))
    assert validate_campaign(_sel_args(campaign, **base)) == 1          # TTtest2 is unfinished
    assert {d["dataset"] for d in json.loads((campaign / "rep.json").read_text())["datasets"]} == {"TTtest", "TTtest2"}
    assert validate_campaign(_sel_args(campaign, exclude=["TTtest2"], **base)) == 0
    assert {d["dataset"] for d in json.loads((campaign / "rep.json").read_text())["datasets"]} == {"TTtest"}


def test_upload_honours_exclude(campaign, capsys):
    from wrcoffea.skim.upload import upload_campaign
    base = dict(remote="root://r//store/skims", dry_run=True, workers=1)
    upload_campaign(_sel_args(campaign, **base))
    assert "from 1 dataset(s) to upload; 1 incomplete dataset(s) skipped" in capsys.readouterr().out
    upload_campaign(_sel_args(campaign, exclude=["TTtest2"], **base))
    assert "from 1 dataset(s) to upload; 0 incomplete dataset(s) skipped" in capsys.readouterr().out
