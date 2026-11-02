"""Tests for bin/run_production.py: the commands of an era's production, how they run and how they are reported."""

import json
import os
import sys
from contextlib import contextmanager

import pytest

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

import run_production  # noqa: E402
from run_production import Command, Result, plan_commands, select_commands, with_extra_systs  # noqa: E402


def names_and_args(era, extra=()):
    return [(c.name, " ".join(c.args)) for c in plan_commands(era, extra)]


def fake_popen(code_of, *, polls=2):
    """A Popen stand-in whose processes exit with ``code_of(argv)`` after ``polls`` polls, writing a
    log line; ``state`` records every argv and the peak number alive per phase (preflight / run)."""
    state = {"argvs": [], "live": {"preflight": 0, "run": 0}, "peak": {"preflight": 0, "run": 0}}

    class Proc:
        def __init__(self, argv, stdout=None, stderr=None):
            self.code = code_of(argv)
            self.left = polls
            self.phase = "preflight" if "--preflight-only" in argv else "run"
            state["argvs"].append(list(argv))
            state["live"][self.phase] += 1
            state["peak"][self.phase] = max(state["peak"][self.phase], state["live"][self.phase])
            stdout.write("Execution took 1.00 minutes\n" if self.code == 0 else "Traceback\nRuntimeError: boom\n")

        def poll(self):
            self.left -= 1
            if self.left > 0:
                return None
            if self.left == 0:
                state["live"][self.phase] -= 1
            return self.code

        def wait(self, timeout=None):
            return self.code

        def terminate(self):
            pass

    return Proc, state


def test_2024_production_is_the_seven_commands_of_the_readme():
    assert names_and_args("RunIII2024Summer24") == [
        ("nominal", "all --signal-points all"),
        ("dy_zpt", "DYJets --dy-reweight zpt"),
        ("dy_zpt_reshape", "DYJets --dy-reweight zpt+reshape"),
        ("dy_lo_inc", "DYJets --dy lo_inc"),
        ("dy_nlo_inc", "DYJets --dy nlo_inc"),
        ("dy_nlo_mll", "DYJets --dy nlo_mll"),
        ("dy_nlo_njet", "DYJets --dy nlo_njet"),
    ]


def test_commands_follow_the_era_config():
    """Every dy_variants entry but the default and nlo_ptll, the DY corrections where the era has payloads."""
    run3_2022 = dict(names_and_args("Run3Summer22"))
    assert set(run3_2022) == {"nominal", "dy_zpt", "dy_zpt_reshape", "dy_lo_inc", "dy_nlo_inc",
                              "dy_lo_njet", "dy_lo_ptll", "dy_nlo_mll", "dy_nlo_njet"}
    ul18 = dict(names_and_args("RunIISummer20UL18"))
    assert set(ul18) == {"nominal", "dy_zpt", "dy_zpt_reshape", "dy_lo_inc", "dy_nlo_inc",
                         "dy_nlo_mll", "dy_nlo_njet", "dy_nlo_njet_mll", "dy_lo_njet"}
    for era in run_production.list_eras():
        names = dict(names_and_args(era))
        assert "dy_nlo_ptll" not in names and "dy_lo_ht" not in names
        assert names["nominal"] == "all --signal-points all"
    # 2025 runs the 2024 MC with its Z-pT payload; it has no reshape payload
    assert "dy_zpt" in dict(names_and_args("RunIII2025Summer25"))
    assert "dy_zpt_reshape" not in dict(names_and_args("RunIII2025Summer25"))


def test_extra_flags_reach_every_command_and_systs_gain_the_dy_nuisances():
    extra = ["--systs", "lumi", "pileup", "--xrootd-timeout", "120"]
    args = {c.name: list(c.args) for c in plan_commands("RunIII2024Summer24", extra)}
    assert args["nominal"] == ["all", "--signal-points", "all", *extra]
    assert args["dy_lo_inc"] == ["DYJets", "--dy", "lo_inc", *extra]
    assert args["dy_zpt"] == ["DYJets", "--dy-reweight", "zpt",
                              "--systs", "lumi", "pileup", "dyzpt", "--xrootd-timeout", "120"]
    assert args["dy_zpt_reshape"] == ["DYJets", "--dy-reweight", "zpt+reshape",
                                      "--systs", "lumi", "pileup", "dyzpt", "dyreshape", "--xrootd-timeout", "120"]


def test_with_extra_systs():
    assert with_extra_systs([], ["dyzpt"]) == []                                         # no systematics
    assert with_extra_systs(["--chunksize", "50000"], ["dyzpt"]) == ["--chunksize", "50000"]
    assert with_extra_systs(["--systs", "lumi"], ["dyzpt"]) == ["--systs", "lumi", "dyzpt"]  # --systs last
    assert with_extra_systs(["--systs", "lumi", "dyzpt"], ["dyzpt"]) == ["--systs", "lumi", "dyzpt"]
    assert with_extra_systs(["--systs"], ["dyzpt"]) == ["--systs", "dyzpt"]


def test_select_commands():
    commands = plan_commands("RunIII2024Summer24")
    assert [c.name for c in select_commands(commands, only=["dy_nlo_mll", "nominal"])] == ["nominal", "dy_nlo_mll"]
    assert [c.name for c in select_commands(commands, skip=["nominal"])][0] == "dy_zpt"
    with pytest.raises(ValueError, match="dy_nlo_ptll"):
        select_commands(commands, only=["dy_nlo_ptll"])


def test_resolve_eras():
    assert run_production.resolve_eras(["RunIII2024Summer24", "Run3Summer22", "RunIII2024Summer24"]) == \
        ["RunIII2024Summer24", "Run3Summer22"]
    every = run_production.resolve_eras(["all"])
    assert "RunIII2024Summer24" in every and "RunIISummer20UL18" in every
    assert "RunIII2025Summer25" not in every        # no 2025 data fileset yet
    with pytest.raises(ValueError, match="Run3Summer24"):
        run_production.resolve_eras(["Run3Summer24"])


def test_command_lines_logs_and_output_dir():
    command = Command("RunIII2024Summer24", "dy_zpt", ("DYJets", "--dy-reweight", "zpt"))
    argv = run_production.command_argv(command, tag="20261102", scheduler="condor_scheduler.json", wait_timeout=3600)
    assert argv[0] == sys.executable
    assert argv[1:] == ["bin/run_analysis.py", "RunIII2024Summer24", "DYJets", "--dy-reweight", "zpt",
                        "--dir", "20261102", "--scheduler", "condor_scheduler.json", "--worker-wait-timeout", "3600"]
    assert run_production.command_argv(command, tag="t")[1:] == \
        ["bin/run_analysis.py", "RunIII2024Summer24", "DYJets", "--dy-reweight", "zpt", "--dir", "t"]
    preflight = run_production.command_argv(command, tag="t", scheduler="f", wait_timeout=1, preflight=True)
    assert preflight[-1] == "--preflight-only" and "--scheduler" not in preflight
    assert str(run_production.log_path(command, tag="t")) == "logs/t/RunIII2024Summer24_dy_zpt.log"
    assert str(run_production.log_path(command, tag="t", preflight=True)) == \
        "logs/t/RunIII2024Summer24_dy_zpt.preflight.log"
    assert str(run_production.output_dir("RunIII2024Summer24", "t")) == \
        "WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/t"


def test_run_commands_limits_concurrency_and_reports_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    commands = plan_commands("RunIII2024Summer24")
    Proc, state = fake_popen(lambda argv: 1 if "nlo_mll" in argv else 0)
    ticks = []
    results = run_production.run_commands(
        commands, argv_of=lambda c: run_production.command_argv(c, tag="t"),
        log_of=lambda c: run_production.log_path(c, tag="t"),
        jobs=3, tick=lambda: ticks.append(1), tick_s=0, poll_s=0, popen=Proc)
    assert state["peak"]["run"] == 3 and len(results) == 7 and ticks
    failed = [r for r in results if r.returncode != 0]
    assert [r.command.name for r in failed] == ["dy_nlo_mll"]
    assert run_production.last_line(failed[0].log) == "RuntimeError: boom"
    assert (tmp_path / "logs/t/RunIII2024Summer24_nominal.log").read_text().startswith("Execution took")


def test_summary_names_the_failed_commands_and_how_to_rerun_them(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    commands = plan_commands("RunIII2024Summer24")
    out = run_production.output_dir("RunIII2024Summer24", "t")
    out.mkdir(parents=True)
    (out / "WRAnalyzer_DYJets.root").write_bytes(b"")
    log = tmp_path / "fail.log"
    log.write_text("Traceback\nRuntimeError: boom\n")
    results = [Result(c, 1 if c.name in ("dy_nlo_mll", "dy_nlo_njet") else 0, 2.0, log) for c in commands]
    text = run_production.summary_text(results, tag="t", wall_minutes=50.4,
                                       rerun_line=lambda era, names: f"RERUN {era} {' '.join(names)}")
    assert text.startswith("5 of 7 command(s) succeeded in 50 min, 2 FAILED.")
    assert "RunIII2024Summer24 dy_nlo_mll" in text and "RuntimeError: boom" in text
    assert "1 files in WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/t/" in text
    assert text.endswith("Rerun: RERUN RunIII2024Summer24 dy_nlo_mll dy_nlo_njet")


def test_main_plan_prints_the_commands_and_runs_nothing(capsys):
    assert run_production.main(["RunIII2024Summer24", "--plan", "--dir", "t", "--", "--systs", "lumi"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Production t: 7 command(s), logs in logs/t/")
    assert "RunIII2024Summer24 dy_zpt_reshape  DYJets --dy-reweight zpt+reshape --systs lumi dyzpt dyreshape" in out
    assert "output WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/t/" in out


def test_main_rejects_unknown_names(capsys):
    with pytest.raises(SystemExit, match="Unknown command name"):
        run_production.main(["RunIII2024Summer24", "--plan", "--only", "dy_nlo_ptll"])
    with pytest.raises(SystemExit, match="Unknown era"):
        run_production.main(["Run3Summer24", "--plan"])


def test_main_local_runs_one_command_at_a_time_after_the_preflight(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_production.time, "sleep", lambda s: None)
    Proc, state = fake_popen(lambda argv: 0)
    assert run_production.main(["RunIII2024Summer24", "--dir", "t", "--only", "dy_lo_inc", "dy_nlo_inc"], popen=Proc) == 0
    preflights = [a for a in state["argvs"] if "--preflight-only" in a]
    runs = [a for a in state["argvs"] if "--preflight-only" not in a]
    assert len(preflights) == 2 and len(runs) == 2
    assert state["peak"] == {"preflight": 2, "run": 1}
    assert all("--scheduler" not in a and a[1:4] == ["bin/run_analysis.py", "RunIII2024Summer24", "DYJets"] for a in runs)


def test_main_stops_when_a_preflight_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_production.time, "sleep", lambda s: None)
    Proc, state = fake_popen(lambda argv: 1 if "--preflight-only" in argv and "nlo_mll" in argv else 0)
    assert run_production.main(["RunIII2024Summer24", "--dir", "t"], popen=Proc) == 2
    assert all("--preflight-only" in a for a in state["argvs"])     # nothing started
    assert (tmp_path / "logs/t/RunIII2024Summer24_dy_nlo_mll.preflight.log").exists()


def test_main_condor_starts_the_workers_and_attaches_every_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_production.time, "sleep", lambda s: None)
    Proc, state = fake_popen(lambda argv: 1 if "nlo_mll" in argv and "--preflight-only" not in argv else 0)
    events = []

    @contextmanager
    def fake_cluster(*, workers, memory, wait_timeout):
        events.append(("start", workers, memory, wait_timeout))
        yield lambda: events.append("tick")
        events.append("stop")

    rc = run_production.main(["RunIII2024Summer24", "--condor", "--dir", "t", "--worker-wait-timeout", "21600",
                              "--", "--systs", "lumi"], popen=Proc, cluster=fake_cluster)
    assert rc == 1
    assert events[0] == ("start", 250, "8GB", 21600) and events[-1] == "stop"
    runs = [a for a in state["argvs"] if "--preflight-only" not in a]
    assert len(runs) == 7 and state["peak"]["run"] == 7
    assert all(a[-4:] == ["--scheduler", "condor_scheduler.json", "--worker-wait-timeout", "21600"] for a in runs)


def test_main_summary_rerun_line(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_production.time, "sleep", lambda s: None)
    Proc, state = fake_popen(lambda argv: 1 if "nlo_mll" in argv and "--preflight-only" not in argv else 0)
    rc = run_production.main(["RunIII2024Summer24", "--scheduler", "condor_scheduler.json", "--dir", "t",
                              "--", "--chunksize", "50000"], popen=Proc)
    assert rc == 1
    out = capsys.readouterr().out
    assert out.rstrip().endswith("Rerun: python bin/run_production.py RunIII2024Summer24 --scheduler condor_scheduler.json "
                                 "--dir t --only dy_nlo_mll -- --chunksize 50000")
    assert state["peak"]["run"] == 7      # every command of the era at once on the cluster


def test_scheduler_excludes_worker_sizing():
    with pytest.raises(SystemExit):
        run_production.parse_args(["RunIII2024Summer24", "--scheduler", "f", "--workers", "10"])


def test_production_cluster_writes_the_scheduler_file_and_stops_the_workers(tmp_path, monkeypatch):
    calls = []

    class Cluster:
        scheduler_address = "tcp://node:10042"

        def scale(self, n):
            calls.append(("scale", n))

        def close(self):
            calls.append("cluster.close")

    class Client:
        def set_metadata(self, key, value):
            calls.append(("metadata", key, value["workers"]))

        def scheduler_info(self):
            return {"workers": {"w1": {}}}

        def close(self):
            calls.append("client.close")

    monkeypatch.setattr(run_production.run_analysis, "_cluster_identity",
                        lambda **kw: {"workers": kw["n_workers"], "memory": kw["memory"], "started": "now"})
    monkeypatch.setattr(run_production.run_analysis, "_start_condor_cluster", lambda **kw: (Cluster(), Client()))
    monkeypatch.setattr(run_production.run_analysis, "_wait_for_workers_or_raise",
                        lambda client, **kw: calls.append(("wait", kw["timeout_s"])))
    scheduler_file = tmp_path / "condor_scheduler.json"
    with run_production.production_cluster(workers=5, memory="6GB", wait_timeout=7, scheduler_file=scheduler_file) as keep_alive:
        assert json.loads(scheduler_file.read_text())["address"] == "tcp://node:10042"
        keep_alive()
    assert not scheduler_file.exists()
    assert calls == [("metadata", run_production.run_analysis.CLUSTER_IDENTITY_KEY, 5), ("wait", 7),
                     ("scale", 5), "client.close", "cluster.close"]
