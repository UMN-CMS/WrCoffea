"""Group worker, plan grouping, validation and fetch helpers on local synthetic files."""

import json
import os
import zlib
from pathlib import Path

import numpy as np
import awkward as ak
import pytest
import uproot

from tests.nano_synth import write_nano_file
from wrcoffea.skim import worker as worker_mod
from wrcoffea.skim.plan import Plan, DatasetPlan, make_groups, dataset_key_for
from wrcoffea.skim.catalog import FileRecord
from wrcoffea.skim.schema import build_schema, branch_types_of_tree
from wrcoffea.skim.worker import process_group, load_group_status, VerificationError, verify_part
from wrcoffea.skim.validate import validate_dataset, golden_lumis
from wrcoffea.skim.fetch import classify_xrdcp_error, adler32_of
from wrcoffea.skim.writer import PartRecord, InputRecord


def _local_plan(tmp_path, *, is_data=False, n_files=3, group_bytes=1e12, n=1500):
    files = []
    tables = []
    for i in range(n_files):
        p = tmp_path / "inputs" / f"in{i}.root"
        p.parent.mkdir(exist_ok=True)
        write_nano_file(p, n=n, seed=100 + i, is_data=is_data, run=300 + i if is_data else 1, lumi_start=1,
                        extra_hlt=("HLT_Mu55",) if i == 1 else ())
        with uproot.open(p) as f:
            tables.append(branch_types_of_tree(f["Events"]))
        files.append(FileRecord(lfn=str(p), size=p.stat().st_size, events=n, adler32=adler32_of(str(p)), valid=True, dataset="local"))
    schema = build_schema(tables)
    ds = DatasetPlan(key="Muon0_RunX" if is_data else "TTtest", era="Run3Summer22", run="Run3", year="2022",
                     datatype="data" if is_data else "mc", physics_group="Muon" if is_data else "tt_tW",
                     category="data" if is_data else "backgrounds", das_paths=["/local/x/NANOAOD"],
                     metadata={"xsec": 1.0, "genEventSumw": None}, files=[f.to_dict() for f in files],
                     groups=make_groups(files, group_bytes), schema_id=schema.digest(),
                     dbs_summary={"num_file": n_files, "num_event": n * n_files, "file_size": sum(f.size for f in files)},
                     out_dir=f"Run3/2022/Run3Summer22/{'data' if is_data else 'backgrounds'}/{'Muon0_RunX' if is_data else 'TTtest'}")
    plan = Plan(campaign="unit", created="now", code_version="test", cuts={}, policy_version=2,
                settings={"compression": "zlib:1", "flush_events": 400, "part_max_bytes": 1},
                datasets={ds.key: ds}, schemas={schema.digest(): schema.to_dict()})
    from wrcoffea.skim.selection import SKIM_CUTS
    plan.cuts = dict(SKIM_CUTS)
    plan.save(tmp_path / "plan")
    return Plan.load(tmp_path / "plan"), ds.key


def test_make_groups_and_dataset_key():
    files = [FileRecord(f"/store/f{i}.root", 10, 1, "", True, "d") for i in range(7)]
    assert [len(g) for g in make_groups(files, 25)] == [2, 2, 2, 1]
    assert [len(g) for g in make_groups(files, 1e9)] == [7]
    assert dataset_key_for("/Muon0/Run2024C-x-v1/NANOAOD", {"dataset": "Muon0_Run2024C"}) == "Muon0_Run2024C"
    assert dataset_key_for("/TTto2L2Nu_X/Run3/NANOAODSIM", {}) == "TTto2L2Nu_X"


def test_private_sample_plan_keeps_file_bookkeeping_out_of_metadata(tmp_path):
    from wrcoffea.skim.plan import build_plan
    url = "root://site.example//store/user/x/WRtoNMutoMuMuJJ_MWR_2000_MN_100_merged.root"
    name = "WRtoNMutoMuMuJJ_MWR2000_N100_TuneCP5_13p6TeV_madgraph-pythia8"
    cfg = tmp_path / "RunIII2024Summer24_signal.json"
    cfg.write_text(json.dumps({name: {
        "das_name": name, "run": "Run3", "year": "2024", "era": "RunIII2024Summer24", "dataset": name,
        "physics_group": "Signal", "xsec": 0.1672, "genEventSumw": 39000.0, "datatype": "mc",
        "files": [url], "file_sizes": {url: 4.5e8}, "file_events": {url: 39000}, "file_adler32": {url: "62b1df76"},
    }}))
    plan = build_plan([cfg], "unit", dbs=object(), cache_dir=tmp_path / "cache")
    ds = plan.datasets[name]
    assert ds.category == "signals" and ds.schema_id == "lazy" and ds.das_paths == [name]
    assert ds.files == [{"lfn": url, "size": 450000000, "events": 39000, "adler32": "62b1df76", "valid": True,
                         "dataset": name}]
    assert ds.dbs_summary == {"num_file": 1, "num_event": 39000, "file_size": 450000000}
    assert not {"files", "file_sizes", "file_events", "file_adler32"} & set(ds.metadata)
    assert ds.metadata["xsec"] == 0.1672
    assert any("1 with adler32" in n for n in ds.notes)


def test_process_group_mc_end_to_end(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=3)
    out_root = tmp_path / "out"
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert st.status == "done", st.error
    assert st.n_done == 3 and st.events_in == 4500 and 0 < st.events_out < 4500
    assert len(st.parts) == 3  # part_max_bytes=1 -> roll after every input
    assert load_group_status(out_root / plan.datasets[key].out_dir, 0).status == "done"
    rep = validate_dataset(plan, key, out_root, reread=True)
    assert rep["ok"], rep["problems"]
    assert rep["events_in"] == 4500 and rep["events_out"] == st.events_out
    assert rep["gen_count"] == 4500
    # idempotent: a second call skips
    st2 = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert st2.status == "done" and st2.finished == st.finished
    # the sidecar tells which input filled HLT_Mu55 (only in1 had it)
    from wrcoffea.skim.writer import read_sidecar
    fills = {os.path.basename(i["lfn"]): i["filled_branches"] for p in st.parts for i in read_sidecar(p["sidecar"])["part"]["inputs"]}
    assert "HLT_Mu55" in fills["in0.root"] and "HLT_Mu55" not in fills["in1.root"]


def test_validate_detects_corruption_and_missing_group(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)  # one group per file
    out_root = tmp_path / "out"
    ds = plan.datasets[key]
    assert len(ds.groups) == 2
    st0 = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert st0.status == "done"
    rep = validate_dataset(plan, key, out_root)
    assert not rep["ok"] and any("never ran" in p for p in rep["problems"])
    st1 = process_group(plan, key, 1, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert validate_dataset(plan, key, out_root)["ok"]
    # flip a byte in a part -> adler32 mismatch
    path = st1.parts[0]["path"]
    data = bytearray(open(path, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(path, "wb").write(bytes(data))
    rep = validate_dataset(plan, key, out_root)
    assert not rep["ok"] and any("adler32" in p for p in rep["problems"])


def test_selection_mismatch_fails_group(tmp_path, monkeypatch):
    plan, key = _local_plan(tmp_path, n_files=1)
    real = worker_mod.recount_mask

    def flipped(arrays, cuts=None):
        m = real(arrays, cuts) if cuts is not None else real(arrays)
        m = m.copy()
        m[0] = not m[0]
        return m
    monkeypatch.setattr(worker_mod, "recount_mask", flipped)
    out_root = tmp_path / "out"
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert st.status == "failed" and "SelectionMismatch" in st.error
    assert not list((out_root / plan.datasets[key].out_dir).glob("*.root"))


def test_groups_use_separate_scratch_dirs(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)
    out_root = tmp_path / "out"
    for gid in (0, 1):
        st = process_group(plan, key, gid, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False, keep_inputs=True)
        assert st.status == "done", st.error
    # keep_inputs leaves each group's scratch in its own directory
    assert sorted(p.name for p in (tmp_path / "scratch" / key).iterdir()) == ["g0000", "g0001"]


def test_dbs_event_count_mismatch_fails_group(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=1)
    plan.datasets[key].files[0]["events"] = 999
    st = process_group(plan, key, 0, out_root=tmp_path / "out", scratch=tmp_path / "scratch", prefetch=False)
    assert st.status == "failed" and "DBS lists 999" in st.error


def test_process_group_data_with_golden_coverage(tmp_path):
    plan, key = _local_plan(tmp_path, is_data=True, n_files=2, n=1000)
    out_root = tmp_path / "out"
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert st.status == "done", st.error
    with uproot.open(st.parts[0]["path"]) as f:
        assert "LuminosityBlocks" in f and "genWeight" not in f["Events"].keys()
    # golden JSON covering all lumis of both runs (1000 events / 50 per lumi = 20 lumis each)
    gold = {"300": [[1, 20]], "301": [[1, 20]]}
    gpath = tmp_path / "golden.json"
    gpath.write_text(json.dumps(gold))
    rep = validate_dataset(plan, key, out_root, golden=str(gpath), dbs_runs=[300, 301])
    assert rep["ok"], rep["problems"]
    assert rep["golden_lumis_missing"] == 0 and rep["lumis_processed"] == 40
    # a golden JSON claiming more lumis than were processed is reported
    gold["301"] = [[1, 25]]
    gpath.write_text(json.dumps(gold))
    rep = validate_dataset(plan, key, out_root, golden=str(gpath), dbs_runs=[300, 301])
    assert not rep["ok"] and rep["golden_lumis_missing"] == 5


def test_verify_part_catches_bad_digest(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=1)
    st = process_group(plan, key, 0, out_root=tmp_path / "out", scratch=tmp_path / "scratch", prefetch=False)
    from wrcoffea.skim.writer import read_sidecar
    sc = read_sidecar(st.parts[0]["sidecar"])
    part = PartRecord(path=sc["part"]["path"], index=1, n_events=sc["part"]["n_events"], size=sc["part"]["size"],
                      adler32=sc["part"]["adler32"], inputs=[InputRecord(**{k: v for k, v in i.items()}) for i in sc["part"]["inputs"]],
                      runs=sc["part"]["runs"], n_lumis=None, schema_digest="", sidecar=sc["part"]["sidecar"])
    verify_part(part, False)
    part.inputs[0].ids_digest = "0" * 32
    with pytest.raises(VerificationError):
        verify_part(part, False)


def test_fetch_helpers(tmp_path):
    assert classify_xrdcp_error(53, "") == ("checksum", False)
    assert classify_xrdcp_error(54, "[3011] No servers are available to read the file") == ("missing", False)
    assert classify_xrdcp_error(54, "Run: [ERROR] Operation expired") == ("timeout", True)
    assert classify_xrdcp_error(1, "weird")[1] is True
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world" * 1000)
    assert adler32_of(str(p)) == f"{zlib.adler32(p.read_bytes()) & 0xFFFFFFFF:08x}"
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"1": [[1, 3], [7, 7]]}))
    assert golden_lumis(g) == {1: {1, 2, 3, 7}}


def _dead_pid() -> int:
    import subprocess
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_running_group_with_live_worker_is_not_run_twice(tmp_path):
    import socket
    plan, key = _local_plan(tmp_path, n_files=1)
    out_root = tmp_path / "out"
    out_dir = out_root / plan.datasets[key].out_dir
    out_dir.mkdir(parents=True)
    import subprocess
    import sys
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])   # a live python worker
    try:
        live = worker_mod.GroupStatus(dataset=key, gid=0, status="running", host=socket.gethostname(), pid=other.pid)
        worker_mod.write_group_status(out_dir, live)
        assert worker_mod.group_in_progress(live)
        with pytest.raises(RuntimeError, match="being processed"):
            process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
        assert load_group_status(out_dir, 0).status == "running"   # untouched
    finally:
        other.kill()
        other.wait()
    # the same status with a dead pid is a crashed attempt and is redone
    dead = worker_mod.GroupStatus(dataset=key, gid=0, status="running", host=socket.gethostname(), pid=_dead_pid())
    worker_mod.write_group_status(out_dir, dead)
    assert not worker_mod.group_in_progress(dead)
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False)
    assert st.status == "done", st.error
    # done / failed statuses are never "in progress"
    assert not worker_mod.group_in_progress(load_group_status(out_dir, 0))


def test_fetch_stop_aborts_running_transfer(tmp_path, monkeypatch):
    import threading
    import time
    from wrcoffea.skim import fetch
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "xrdcp"
    fake.write_text("#!/bin/sh\nsleep 30\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    fetch.STOP.clear()
    outcome = {}

    def run():
        try:
            fetch.fetch_file("/store/x.root", str(tmp_path / "x.root"), ["root://a//store/x.root", "root://b//store/x.root"],
                             adler32=None, size=1, base_timeout=60)
            outcome["result"] = "returned"
        except Exception as e:
            outcome["result"] = e

    t = threading.Thread(target=run)
    t0 = time.monotonic()
    t.start()
    for _ in range(100):           # wait for the fake transfer to be registered
        if fetch._ACTIVE:
            break
        time.sleep(0.05)
    assert fetch._ACTIVE, "transfer was not registered"
    fetch.STOP.set()
    assert fetch.abort_active() == 1
    t.join(10)
    fetch.STOP.clear()
    assert not t.is_alive()
    assert isinstance(outcome["result"], fetch.FetchAborted), outcome
    assert time.monotonic() - t0 < 10          # did not wait for the 30 s sleep or retry the second source
    assert not fetch._ACTIVE


def test_stalled_transfer_is_killed_and_next_source_used(tmp_path, monkeypatch):
    import time
    from wrcoffea.skim import fetch
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "xrdcp"
    # xrdcp -f --nopbar --streams N URL DEST: source "a" writes a few bytes and then hangs, "b" completes
    fake.write_text('#!/bin/sh\nurl="$5"; dest="$6"\ncase "$url" in\n  root://a/*) echo stalled > "$dest"; exec sleep 60;;\n'
                    '  *) echo good > "$dest"; exit 0;;\nesac\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setattr(fetch, "_POLL_SECONDS", 0.2)
    fetch.STOP.clear()
    t0 = time.monotonic()
    res = fetch.fetch_file("/store/x.root", str(tmp_path / "x.root"), ["root://a//store/x.root", "root://b//store/x.root"],
                           adler32=None, size=1, base_timeout=60, stall_timeout=1.0, backoff=(0.0, 0.0))
    assert res.ok and res.source == "root://b//store/x.root"
    assert [a.category for a in res.attempts] == ["stall", "ok"]     # one stalled attempt, then the next source
    assert "no progress" in res.attempts[0].message
    assert time.monotonic() - t0 < 20
    assert (tmp_path / "x.root").read_text() == "good\n"
    assert not fetch._ACTIVE


def test_output_tree_is_relocatable(tmp_path):
    """Statuses and sidecars store names relative to the dataset directory: the whole tree can move."""
    import shutil
    from wrcoffea.skim.writer import read_sidecar
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)
    out_root = tmp_path / "out"
    for gid in (0, 1):
        assert process_group(plan, key, gid, out_root=out_root, scratch=tmp_path / "scratch", prefetch=False).status == "done"
    ds_dir = out_root / plan.datasets[key].out_dir
    raw = json.loads((ds_dir / "_groups" / "g0000.json").read_text())
    assert all(not os.path.isabs(p["path"]) and not os.path.isabs(p["sidecar"]) for p in raw["parts"])
    sc_raw = json.loads(Path(ds_dir / raw["parts"][0]["sidecar"]).read_text())
    assert not os.path.isabs(sc_raw["part"]["path"])
    # move the whole campaign root: everything still resolves and validates
    moved = tmp_path / "elsewhere" / "skims"
    shutil.move(str(out_root), str(moved))
    st = load_group_status(moved / plan.datasets[key].out_dir, 0)
    assert Path(st.parts[0]["path"]).is_file() and str(moved) in st.parts[0]["path"]
    sc = read_sidecar(st.parts[0]["sidecar"])
    assert Path(sc["part"]["path"]).is_file() and sc["part"]["path"] == st.parts[0]["path"]
    rep = validate_dataset(plan, key, moved, reread=True)
    assert rep["ok"], rep["problems"]
    # an old-style status with absolute paths still loads unchanged
    old = json.loads((moved / plan.datasets[key].out_dir / "_groups" / "g0001.json").read_text())
    for p in old["parts"]:
        p["path"] = str(moved / plan.datasets[key].out_dir / p["path"]); p["sidecar"] = str(moved / plan.datasets[key].out_dir / p["sidecar"])
    (moved / plan.datasets[key].out_dir / "_groups" / "g0001.json").write_text(json.dumps(old))
    assert load_group_status(moved / plan.datasets[key].out_dir, 1).parts == old["parts"]


def test_threaded_decompression_gives_identical_output(tmp_path):
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1e12)
    outs = []
    for threads in (1, 3):
        root = tmp_path / f"out{threads}"
        st = process_group(plan, key, 0, out_root=root, scratch=tmp_path / f"scr{threads}", prefetch=False,
                           decompression_threads=threads)
        assert st.status == "done", st.error
        with uproot.open(st.parts[0]["path"]) as f:
            outs.append((st.events_in, st.events_out, f["Events"]["event"].array(library="np").tolist()))
    assert outs[0] == outs[1]


def test_threaded_decompression_survives_uproot_closing_the_executor(tmp_path, monkeypatch):
    """uproot 5.7.6 shuts the supplied decompression executor down when the file is closed: every
    input must get its own executor (the first LPC test job died at input 2 with
    'cannot schedule new futures after shutdown')."""
    import uproot as _uproot
    real_open = _uproot.open

    def closing_open(path, **kw):
        f = real_open(path, **kw)
        ex = kw.get("decompression_executor")
        if ex is not None:
            orig_close = f.close

            def close_and_shutdown(*a, **k):
                orig_close(*a, **k)
                ex.shutdown(wait=True)
            f.close = close_and_shutdown
        return f

    monkeypatch.setattr(worker_mod.uproot, "open", closing_open)
    plan, key = _local_plan(tmp_path, n_files=3, group_bytes=1e12)
    st = process_group(plan, key, 0, out_root=tmp_path / "out", scratch=tmp_path / "scr", prefetch=False,
                       decompression_threads=2)
    assert st.status == "done", st.error
    assert st.n_done == 3


def _corrupt_file_in_place(path: Path) -> None:
    """Flip bytes in the middle of a ROOT file: a basket no longer decompresses (checksums would still match DBS)."""
    data = bytearray(path.read_bytes())
    mid = len(data) // 2
    for i in range(mid, mid + 2048):
        data[i] ^= 0xFF
    path.write_bytes(bytes(data))


def test_corrupt_input_is_recorded_and_skipped_on_retry(tmp_path):
    from wrcoffea.skim.worker import pending_groups, needs_retry
    from wrcoffea.skim.condor import corrupt_inputs_from_statuses, write_plan_slice
    plan, key = _local_plan(tmp_path, n_files=3, group_bytes=1e12)      # one group, three inputs
    bad = Path(plan.datasets[key].groups[0][1])
    _corrupt_file_in_place(bad)
    out_root = tmp_path / "out"
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scr", prefetch=False)
    assert st.status == "failed" and "CorruptInput" in st.error
    assert [b["category"] for b in st.bad_files] == ["corrupt"] and st.bad_files[0]["lfn"] == str(bad)
    assert not list((out_root / plan.datasets[key].out_dir).glob("*.root"))      # partial parts discarded
    # the retry skips the corrupt input and finishes with it recorded
    assert pending_groups(plan, out_root, [key]) == [(key, 0)]
    st2 = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scr", prefetch=False)
    assert st2.status == "done" and st2.n_done == 2
    assert [b["lfn"] for b in st2.bad_files] == [str(bad)] and st2.bad_files[0]["category"] == "corrupt"
    assert not needs_retry(st2)
    assert pending_groups(plan, out_root, [key]) == []                           # corrupt-only: not retried forever
    rep = validate_dataset(plan, key, out_root, reread=True)
    assert rep["bad_files"] == [str(bad)]
    # Condor: the plan slice for a retry carries the corrupt input, so a fresh job skips it too
    assert corrupt_inputs_from_statuses(plan, key, out_root) == [str(bad)]
    write_plan_slice(plan, key, tmp_path / "slice", bad_inputs=corrupt_inputs_from_statuses(plan, key, out_root))
    sliced = Plan.load(tmp_path / "slice")
    assert sliced.datasets[key].bad_inputs == [str(bad)]
    st3 = process_group(sliced, key, 0, out_root=tmp_path / "out_job", scratch=tmp_path / "scr2", prefetch=False)
    assert st3.status == "done" and st3.n_done == 2 and st3.bad_files[0]["category"] == "corrupt"


def test_unreachable_without_replica_becomes_unavailable_and_is_not_retried(tmp_path, monkeypatch):
    from wrcoffea.skim.worker import classify_unreachable, pending_groups, needs_retry, write_group_status, GroupStatus
    from wrcoffea.skim.condor import corrupt_inputs_from_statuses
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1)
    ds = plan.datasets[key]
    out_dir = tmp_path / "out" / ds.out_dir
    missing = ds.groups[0][0]
    attempts = [{"url": f"root://{h}/{missing}", "rc": 54, "seconds": 20.0, "category": "missing", "message": "[3011] No servers"}
                for h in ("cmsxrootd.fnal.gov", "cms-xrd-global.cern.ch")]
    st = GroupStatus(dataset=key, gid=0, status="done", n_inputs=1, bad_files=[{"lfn": missing, "attempts": attempts}])
    write_group_status(out_dir, st)
    transient = GroupStatus(dataset=key, gid=1, status="done", n_inputs=1,
                            bad_files=[{"lfn": ds.groups[1][0], "attempts": [dict(attempts[0], category="timeout")]}])
    write_group_status(out_dir, transient)
    assert pending_groups(plan, tmp_path / "out", [key]) == [(key, 0), (key, 1)]

    class FakeRucio:
        def replicas(self, lfn):
            return [] if lfn == missing else [{"rse": "T1_US_FNAL_Disk", "state": "AVAILABLE", "pfn": "x"}]
    res = classify_unreachable(plan, tmp_path / "out", [key], FakeRucio())
    assert res[key]["unavailable"] == 1 and res[key]["events"] == 1500 and res[key]["still_unreachable"] == 1
    st0 = worker_mod.load_group_status(out_dir, 0)
    assert st0.bad_files[0]["category"] == "unavailable" and not needs_retry(st0)
    assert pending_groups(plan, tmp_path / "out", [key]) == [(key, 1)]        # only the transient one is retried
    assert corrupt_inputs_from_statuses(plan, key, tmp_path / "out") == [missing]


def test_mark_corrupt_cli_records_input_for_the_retry(tmp_path):
    import subprocess, sys
    from wrcoffea.skim.worker import GroupStatus, write_group_status, pending_groups
    from wrcoffea.skim.condor import corrupt_inputs_from_statuses
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1e12)
    ds = plan.datasets[key]
    out_root = tmp_path / "out"
    lfn = ds.groups[0][1]
    write_group_status(out_root / ds.out_dir, GroupStatus(dataset=key, gid=0, status="failed", error="DecompressionError: lzma data error"))
    repo = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, str(repo / "bin" / "skim.py"), "bad-files", "--plan-dir", str(tmp_path / "plan"), "--out", str(out_root),
                        "--mark-corrupt", lfn, "--reason", "verified unreadable"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "marked corrupt" in r.stdout
    assert corrupt_inputs_from_statuses(plan, key, out_root) == [lfn]
    assert pending_groups(plan, out_root, [key]) == [(key, 0)]
    st = process_group(plan, key, 0, out_root=out_root, scratch=tmp_path / "scr", prefetch=False)
    assert st.status == "done" and st.n_done == 1 and st.bad_files[0]["category"] == "corrupt"


def test_empty_input_file_is_recorded_not_fatal(tmp_path):
    """DBS lists some NanoAOD files with 0 events; they may lack every event branch."""
    plan, key = _local_plan(tmp_path, n_files=2, group_bytes=1e12)
    ds = plan.datasets[key]
    empty = tmp_path / "inputs" / "empty.root"
    with uproot.recreate(empty) as f:
        f.mktree("Events", {"run": "uint32", "luminosityBlock": "uint32", "event": "uint64"})
        from tests.nano_synth import _write_tree
        _write_tree(f, "Runs", {                     # what an empty NanoAOD file carries: zero sums, empty vectors
            "run": np.array([1], np.uint32), "genEventCount": np.array([0], np.int64),
            "genEventSumw": np.array([0.0]), "genEventSumw2": np.array([0.0]),
            "LHEScaleSumw": ak.Array([np.zeros(0, np.float32)]), "PSSumw": ak.Array([np.zeros(0, np.float32)]),
        })
    rec = FileRecord(lfn=str(empty), size=empty.stat().st_size, events=0, adler32=adler32_of(str(empty)), valid=True, dataset="local")
    ds.files.append(rec.to_dict())
    ds.groups[0].insert(0, str(empty))
    ds.dbs_summary["num_file"] += 1
    st = process_group(plan, key, 0, out_root=tmp_path / "out", scratch=tmp_path / "scr", prefetch=False)
    assert st.status == "done", st.error
    assert st.n_done == 3 and st.events_in == 3000
    rep = validate_dataset(plan, key, tmp_path / "out", reread=True)
    assert rep["ok"], rep["problems"]
