"""Condor submission tooling: plan slices, job preparation, shipping and mirroring (fake xrootd tools)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_skim_worker import _local_plan
from wrcoffea.skim import condor
from wrcoffea.skim.plan import Plan
from wrcoffea.skim.worker import process_group, load_group_status
from wrcoffea.skim.validate import validate_dataset

FAKE_XRDCP = '''#!/usr/bin/env python3
import os, shutil, sys
args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.startswith("adler32:")]
src, dst = args[-2], args[-1]
def local(u): return u.replace("root://fake/", "", 1) if u.startswith("root://fake/") else u
src, dst = local(src), local(dst)
if not os.path.exists(src):
    print("[ERROR] Server responded with an error: [3011] Too many DFS read attempts; operation terminated", file=sys.stderr); sys.exit(54)
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
shutil.copyfile(src, dst)
'''
FAKE_XRDFS = '''#!/usr/bin/env python3
import os, sys
host, cmd = sys.argv[1], sys.argv[2]
if cmd == "stat":
    p = sys.argv[3]
    if not os.path.exists(p): print("[ERROR] not found"); sys.exit(54)
    print(f"Path:   {p}\\nSize:   {os.path.getsize(p)}")
elif cmd == "ls":
    root = sys.argv[-1]
    if not os.path.isdir(root): print("[ERROR] [3011] No such file or directory"); sys.exit(54)
    for d, _, fs in os.walk(root):
        for f in fs: print(os.path.join(d, f))
'''


@pytest.fixture
def fake_xrootd(tmp_path, monkeypatch):
    b = tmp_path / "fakebin"
    b.mkdir()
    for name, body in (("xrdcp", FAKE_XRDCP), ("xrdfs", FAKE_XRDFS)):
        f = b / name
        f.write_text(body)
        f.chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    monkeypatch.setattr(condor.shutil, "which", lambda name: None if name == "condor_q" else f"{b}/{name}")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    return f"root://fake/{remote_root}"


def test_plan_slice_round_trip(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)
    condor.write_plan_slice(plan, key, tmp_path / "slice")
    sl = Plan.load(tmp_path / "slice")
    assert list(sl.datasets) == [key] and sl.datasets[key].groups == plan.datasets[key].groups
    assert sl.schemas.keys() == {plan.datasets[key].schema_id} and sl.cuts == plan.cuts


def test_prepare_jobs_writes_submission(tmp_path, monkeypatch):
    plan, key = _local_plan(tmp_path, n_files=3, group_bytes=1)   # three groups
    monkeypatch.setattr(condor, "inflight_groups", lambda *a: {("Run3Summer22", key, 1), ("Run3Summer23BPix", key, 2)})
    repo = tmp_path / "repo"
    (repo / "bin").mkdir(parents=True)
    (repo / "wrcoffea" / "skim" / "__pycache__").mkdir(parents=True)
    (repo / "bin" / "skim.py").write_text("print('x')\n")
    (repo / "wrcoffea" / "__init__.py").write_text("")
    (repo / "wrcoffea" / "skim" / "worker.py").write_text("")
    (repo / "wrcoffea" / "skim" / "__pycache__" / "worker.cpython-312.pyc").write_bytes(b"\x00")
    jobs = tmp_path / "jobs"
    summary = condor.prepare_jobs(plan, [key], out_root=tmp_path / "out", remote="root://x//store/u/skims_v2",
                                  jobs_dir=jobs, repo_root=repo, proxy="/tmp/x509up_u1", max_groups=None)
    assert summary["groups"] == 2 and summary["skipped_inflight"] == 1
    assert (jobs / "groups.txt").read_text() == f"{key} 0 Run3Summer22\n{key} 2 Run3Summer22\n"
    jdl = (jobs / "submit.jdl").read_text()
    assert "+SingularityImage" in jdl and condor.DEFAULT_IMAGE in jdl and "x509userproxy           = /tmp/x509up_u1" in jdl
    assert "queue key,gid,era from groups.txt" in jdl and "root://x//store/u/skims_v2" in jdl
    assert '+SkimGroup              = "$(era)/$(key):$(gid)"' in jdl
    assert "job_machine_attrs       = Machine" in jdl and "TARGET.Machine =!= MY.MachineAttrMachine1" in jdl
    assert "periodic_hold           = (JobStatus == 2) && ((time() - EnteredCurrentStatus) > 28800)" in jdl
    assert "periodic_release        = (HoldReasonCode == 3) && (NumJobStarts < 4)" in jdl
    assert os.access(jobs / "run_group.sh", os.X_OK)
    assert Plan.load(jobs / "plans" / key).datasets[key].key == key
    import tarfile
    names = tarfile.open(jobs / "skim_code.tgz").getnames()
    assert "bin/skim.py" in names and "wrcoffea/skim/worker.py" in names and not any("__pycache__" in n for n in names)
    # a second preparation with everything done produces no jobs
    out_root = tmp_path / "out"
    for gid in range(3):
        process_group(plan, key, gid, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    monkeypatch.setattr(condor, "inflight_groups", lambda *a: set())
    assert condor.prepare_jobs(plan, [key], out_root=out_root, remote="root://x//r", jobs_dir=tmp_path / "jobs2",
                               repo_root=repo, proxy="/p")["groups"] == 0


def test_ship_then_mirror_reproduces_a_valid_tree(tmp_path, fake_xrootd):
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)
    job_out = tmp_path / "job_out"
    for gid in (0, 1):
        assert process_group(plan, key, gid, out_root=job_out, scratch=tmp_path / "scratch", prefetch=False).status == "done"
        res = condor.ship_group(plan, key, gid, out_root=job_out, remote=fake_xrootd)
        assert res["status"] == "done" and res["files"][-1] == f"_groups/g{gid:04d}.json"
    remote_dir = Path(fake_xrootd.replace("root://fake/", "")) / plan.datasets[key].out_dir
    assert sorted(p.name for p in remote_dir.glob("*.root")) == sorted(p.name for p in (job_out / plan.datasets[key].out_dir).glob("*.root"))
    # the remote tree is a campaign root in its own right
    assert load_group_status(remote_dir, 0).status == "done"
    assert validate_dataset(plan, key, Path(fake_xrootd.replace("root://fake/", "")), reread=True)["ok"]
    # a laptop root gets the JSON files only, enough for status and filesets
    local_root = tmp_path / "laptop_root"
    res = condor.mirror_metadata(plan, [key], out_root=local_root, remote=fake_xrootd)
    assert res["files"] == 4 and res["probed"] == 2     # two sidecars + two statuses, probed by name
    # a second mirror fetches nothing new; refresh fetches everything again
    assert condor.mirror_metadata(plan, [key], out_root=local_root, remote=fake_xrootd)["probed"] == 0
    assert condor.mirror_metadata(plan, [key], out_root=local_root, remote=fake_xrootd, refresh=True)["files"] == 4
    st = load_group_status(local_root / plan.datasets[key].out_dir, 1)
    assert st.status == "done" and Path(st.parts[0]["sidecar"]).is_file()
    assert not (local_root / plan.datasets[key].out_dir / Path(st.parts[0]["path"]).name).exists()   # parts stay remote


def test_ship_fails_on_size_mismatch(tmp_path, fake_xrootd, monkeypatch):
    plan, key = _local_plan(tmp_path, n_files=1)
    job_out = tmp_path / "job_out"
    process_group(plan, key, 0, out_root=job_out, scratch=tmp_path / "scratch", prefetch=False)
    monkeypatch.setattr(condor, "remote_size", lambda url, timeout=120: 1)
    monkeypatch.setattr(condor.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="could not ship"):
        condor.ship_group(plan, key, 0, out_root=job_out, remote=fake_xrootd)


def test_inflight_groups_parses_condor_q(monkeypatch, tmp_path):
    monkeypatch.setattr(condor.shutil, "which", lambda name: "/usr/bin/condor_q")
    # a legacy job (no era in its tag) whose submit dir holds the plan slice
    (tmp_path / "plans" / "WZ_TuneCP5").mkdir(parents=True)
    (tmp_path / "plans" / "WZ_TuneCP5" / "plan.json").write_text(json.dumps({"datasets": {"WZ_TuneCP5": {"era": "Run3Summer23"}}}))
    out = ('"Run3Summer23BPix/TTto2L2Nu_TuneCP5:12" 2 /x\n"TTto2L2Nu_TuneCP5:13" 1 /nowhere\n"Muon0_Run2024I:3" 4 /x\n'
           f'undefined 2 /x\n"WZ_TuneCP5:0" 2 {tmp_path}\n')
    monkeypatch.setattr(condor.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": out})())
    assert condor.inflight_groups() == {("Run3Summer23BPix", "TTto2L2Nu_TuneCP5", 12), (None, "TTto2L2Nu_TuneCP5", 13),
                                        ("Run3Summer23", "WZ_TuneCP5", 0)}


def test_mirror_skips_datasets_not_shipped_yet(tmp_path, fake_xrootd):
    plan, key = _local_plan(tmp_path, n_files=1)
    res = condor.mirror_metadata(plan, [key], out_root=tmp_path / "root", remote=fake_xrootd)
    assert res["files"] == 0 and res["datasets"] == 1 and res["probed"] == 1
    assert condor._is_missing("[ERROR] Server responded with an error: [3011] Too many DFS read attempts; operation terminated")
    assert condor._is_missing("[ERROR] [3011] No such file or directory") and not condor._is_missing("[ERROR] [3010] permission denied")


def test_inflight_groups_from_saved_text():
    text = '"Run3Summer23/WW_TuneCP5_13p6TeV_pythia8:0" 5 /x\n"Run3Summer23/Muon0_Run2023C:3" 4 /x\n'
    assert condor.inflight_groups(text) == {("Run3Summer23", "WW_TuneCP5_13p6TeV_pythia8", 0)}
    assert condor.inflight_groups("") == set()


def test_same_dataset_key_in_another_era_is_not_treated_as_in_flight(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)       # era Run3Summer22, two groups
    repo = tmp_path / "repo"; (repo / "bin").mkdir(parents=True); (repo / "wrcoffea").mkdir()
    (repo / "bin" / "skim.py").write_text(""); (repo / "wrcoffea" / "__init__.py").write_text("")
    text = f'"Run3Summer23/{key}:0" 2 /x\n"Run3Summer22/{key}:1" 2 /x\n'
    summary = condor.prepare_jobs(plan, [key], out_root=tmp_path / "out", remote="root://x//r", jobs_dir=tmp_path / "j",
                                  repo_root=repo, proxy="/p", inflight_text=text)
    assert (tmp_path / "j" / "groups.txt").read_text() == f"{key} 0 Run3Summer22\n"    # g1 of THIS era is in flight
    assert summary["groups"] == 1


def test_ship_tolerates_a_busy_server_for_checksummed_files(tmp_path, fake_xrootd, monkeypatch):
    plan, key = _local_plan(tmp_path, n_files=1)
    job_out = tmp_path / "job_out"
    process_group(plan, key, 0, out_root=job_out, scratch=tmp_path / "scratch", prefetch=False)
    calls = {"n": 0}

    def slow_stat(url, timeout=120, attempts=3):
        calls["n"] += 1
        if url.endswith(".root"):
            raise condor.RemoteUnavailable("stat timed out")      # part: checksum-verified copy is accepted
        return Path(url.replace("root://fake/", "")).stat().st_size   # JSONs: stat answers
    monkeypatch.setattr(condor, "remote_size", slow_stat)
    monkeypatch.setattr(condor.time, "sleep", lambda s: None)
    res = condor.ship_group(plan, key, 0, out_root=job_out, remote=fake_xrootd)
    assert res["status"] == "done" and calls["n"] >= 3
    # a JSON (no checksum) whose stat keeps timing out is a hard failure
    monkeypatch.setattr(condor, "remote_size", lambda url, timeout=120, attempts=3: (_ for _ in ()).throw(condor.RemoteUnavailable("x")))
    with pytest.raises(RuntimeError, match="could not ship"):
        condor.ship_file(tmp_path / "job_out" / plan.datasets[key].out_dir / "_groups" / "g0000.json", f"{fake_xrootd}/x/g0000.json")


def test_xrd_timeout_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise condor.subprocess.TimeoutExpired(a[0], k.get("timeout", 1))
    monkeypatch.setattr(condor.subprocess, "run", boom)
    rc, msg = condor._xrd(["xrdfs", "root://x", "stat", "/p"], 5)
    assert rc == -9 and "timed out" in msg
    with pytest.raises(condor.RemoteUnavailable):
        condor.remote_size("root://x//p", timeout=1, attempts=2)


def test_validate_and_filesets_on_a_mirrored_tree_with_remote_parts(tmp_path, fake_xrootd, monkeypatch):
    """Condor output: parts only at the remote root, JSON mirrored locally; validation stats the parts,
    reads data lumisections remotely, and filesets point at the remote files."""
    import argparse
    from wrcoffea.skim import validate as validate_mod
    from wrcoffea.skim.upload import finished_parts
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1, is_data=True)
    job_out = tmp_path / "job_out"
    for gid in (0, 1):
        assert process_group(plan, key, gid, out_root=job_out, scratch=tmp_path / "scratch", prefetch=False).status == "done"
        condor.ship_group(plan, key, gid, out_root=job_out, remote=fake_xrootd)
    laptop = tmp_path / "laptop"
    condor.mirror_metadata(plan, [key], out_root=laptop, remote=fake_xrootd)
    remote_dir = fake_xrootd.replace("root://fake/", "")
    monkeypatch.setattr(validate_mod, "_open_part",
                        lambda path: validate_mod.uproot.open(path.replace("root://fake/", "", 1)))
    # strict local validation fails (parts are not here) ...
    rep = validate_dataset(plan, key, laptop, reread=False)
    assert not rep["ok"] and any("part missing" in p for p in rep["problems"])
    # ... remote-aware validation passes, and data lumisections are read from the remote parts
    rep = validate_dataset(plan, key, laptop, reread=False, remote=fake_xrootd, remote_lumis=True)
    assert rep["ok"], rep["problems"]
    assert rep["remote_parts"] == 2 and rep["lumis_processed"] > 0
    # a missing remote part is caught
    victim = next(Path(remote_dir).rglob("*_g0001p*.root"))
    victim.unlink()
    rep = validate_dataset(plan, key, laptop, reread=False, remote=fake_xrootd)
    assert any("remote part missing" in p for p in rep["problems"])


def test_finished_parts_accepts_permanent_bad_inputs_only(tmp_path):
    from wrcoffea.skim.upload import finished_parts
    from wrcoffea.skim.worker import write_group_status
    plan, key = _local_plan(tmp_path, n_files=1)
    out_root = tmp_path / "out"
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scr", prefetch=False)
    out_dir = out_root / plan.datasets[key].out_dir
    st.bad_files = [{"lfn": "/store/x.root", "category": "unavailable"}]
    write_group_status(out_dir, st)
    assert finished_parts(plan, key, out_root) is not None             # permanent: dataset is final
    st.bad_files = [{"lfn": "/store/y.root", "attempts": [{"category": "timeout"}]}]
    write_group_status(out_dir, st)
    assert finished_parts(plan, key, out_root) is None                 # transient: will be redone


def test_prepare_jobs_avoids_hosts_and_sets_runtime_limit(tmp_path, monkeypatch):
    plan, key = _local_plan(tmp_path, n_files=1)
    monkeypatch.setattr(condor, "inflight_groups", lambda *a: set())
    repo = tmp_path / "repo"; (repo / "bin").mkdir(parents=True); (repo / "wrcoffea").mkdir()
    (repo / "bin" / "skim.py").write_text(""); (repo / "wrcoffea" / "__init__.py").write_text("")
    condor.prepare_jobs(plan, [key], out_root=tmp_path / "out", remote="root://x//r", jobs_dir=tmp_path / "j",
                        repo_root=repo, proxy="/p", max_hours=3, avoid_hosts=("cmswn2230.fnal.gov", "cmswn2245.fnal.gov"))
    jdl = (tmp_path / "j" / "submit.jdl").read_text()
    assert '&& (TARGET.Machine =!= "cmswn2230.fnal.gov") && (TARGET.Machine =!= "cmswn2245.fnal.gov")' in jdl
    assert "> 10800)" in jdl and "longer than 3 h" in jdl


def test_prepare_jobs_sets_the_rucio_account_of_the_jobs(tmp_path, monkeypatch):
    """The jobs' Rucio fallback authenticates as the submitter's account, not skim.py's default (Billy's)."""
    plan, key = _local_plan(tmp_path, n_files=1, group_bytes=1)
    monkeypatch.setattr(condor, "inflight_groups", lambda *a: set())
    repo = tmp_path / "repo"
    (repo / "bin").mkdir(parents=True)
    (repo / "wrcoffea").mkdir()
    (repo / "bin" / "skim.py").write_text("")
    (repo / "wrcoffea" / "__init__.py").write_text("")
    condor.prepare_jobs(plan, [key], out_root=tmp_path / "out", remote="root://x//r", jobs_dir=tmp_path / "a",
                        repo_root=repo, proxy="/p", rucio_account="someone")
    assert 'environment             = "RUCIO_ACCOUNT=someone"' in (tmp_path / "a" / "submit.jdl").read_text()
    condor.prepare_jobs(plan, [key], out_root=tmp_path / "out", remote="root://x//r", jobs_dir=tmp_path / "b",
                        repo_root=repo, proxy="/p")
    assert "environment" not in (tmp_path / "b" / "submit.jdl").read_text()
