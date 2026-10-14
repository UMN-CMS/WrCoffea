"""Tests for bin/run_analysis.py helpers."""

import logging
import os
import sys
from types import SimpleNamespace

import pytest

import coffea.processor as coffea_processor
import wrcoffea.analyzer as analyzer_mod


BIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "bin",
)
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

import run_analysis
from wrcoffea.xrootd_fallback import RedirectorProbeResult


def test_process_fileset_runner_config(monkeypatch):
    """Pin the Runner configuration built by _process_fileset.

    Since the virtual-arrays migration (c10ebe5), skipbadfiles is
    intentionally False: bad-file errors must propagate so
    _preprocess_with_xrd_fallback can rewrite the URL and retry instead
    of silently dropping files.
    """

    captured = {}

    class FakeDaskExecutor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def preprocess(self, fileset, treename):
            return {"preprocessed": True}

        def __call__(self, preproc, treename, processor_instance):
            return {}, {"chunks": 0}

    class FakeWrAnalysis:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(coffea_processor, "DaskExecutor", FakeDaskExecutor)
    monkeypatch.setattr(coffea_processor, "Runner", FakeRunner)
    monkeypatch.setattr(analyzer_mod, "WrAnalysis", FakeWrAnalysis)

    args = SimpleNamespace(
        sample="DYJets",  # _process_fileset reads args.sample for its log label
        mass=None,
        systs=[],
        region="both",
        unskimmed=True,
        study=[],
        chunksize=1000,
        maxchunks=None,
    )

    run_analysis._process_fileset(args, fileset={}, client=object(), condor=False)

    # Bad files are NOT skipped; failures surface to the xrd fallback layer.
    assert captured["skipbadfiles"] is False
    # Chunk outputs and merges stay compressed on the workers (memory of large composites).
    assert captured["executor"].kwargs["compression"] == 1
    assert captured["chunksize"] == 1000
    assert captured["maxchunks"] is None
    assert captured["savemetrics"] is True
    assert captured["xrootdtimeout"] == 10  # condor=False path


def test_dump_dask_diagnostics_writes_file(tmp_path):
    class FakeClient:
        def scheduler_info(self):
            return {"workers": {"tcp://worker-a:1234": {}, "tcp://worker-b:1234": {}}}

        def get_scheduler_logs(self, n=300):
            return [("info", "scheduler log line")]

        def get_worker_logs(self, n=300):
            return {
                "tcp://worker-a:1234": [("warning", "worker-a log line")],
                "tcp://worker-b:1234": "worker-b raw log",
            }

    out = run_analysis._dump_dask_diagnostics(
        FakeClient(),
        label="unit_test",
        out_dir=tmp_path,
        max_entries=5,
    )
    assert out is not None
    text = out.read_text(encoding="utf-8")
    assert "Scheduler workers visible at failure: 2" in text
    assert "scheduler log line" in text
    assert "worker-a log line" in text
    assert "worker-b raw log" in text


class _NoWorkerClient:
    """wait_for_workers that never sees a worker; advances the fake clock by its timeout."""

    def __init__(self, now, arrive_on_call=None):
        self.now, self.calls, self.arrive_on_call = now, 0, arrive_on_call

    def wait_for_workers(self, min_workers, timeout):
        self.calls += 1
        if self.calls == self.arrive_on_call:
            return
        self.now[0] += float(timeout.rstrip("s"))
        raise TimeoutError("no workers")


def _wait_setup(monkeypatch, tmp_path):
    now = [0.0]
    monkeypatch.setattr(run_analysis.time, "monotonic", lambda: now[0])
    diag_file = tmp_path / "diag.log"
    diag_file.write_text("stub\n", encoding="utf-8")
    monkeypatch.setattr(run_analysis, "_dump_dask_diagnostics", lambda client, label: diag_file)
    return now


def _states(**counts):
    return {"idle": 0, "running": 0, "held": 0, "ended": 0, "hold_reason": None, **counts}


def test_wait_for_workers_keeps_waiting_while_the_pool_is_busy(monkeypatch, tmp_path, caplog):
    """Jobs idle in the queue: wait out the timeout, report progress, then explain the timeout."""
    now = _wait_setup(monkeypatch, tmp_path)
    client = _NoWorkerClient(now)
    with caplog.at_level("INFO"), pytest.raises(RuntimeError, match="Timed out waiting.*200 idle.*no free slot"):
        run_analysis._wait_for_workers_or_raise(
            client, requested_workers=200, min_workers=1, timeout_s=1200, label="busy",
            job_states=lambda: _states(idle=200), poll_s=60, report_s=300)
    assert now[0] >= 1200 and client.calls == 20
    assert "Still waiting for Condor workers" in caplog.text


def test_wait_for_workers_stops_when_worker_jobs_end_without_connecting(monkeypatch, tmp_path):
    """Workers that die at start (2026-10-14: lpcjobqueue missing from the shipped env) end their
    jobs: stop at the first check instead of waiting out the timeout."""
    now = _wait_setup(monkeypatch, tmp_path)
    client = _NoWorkerClient(now)
    with pytest.raises(RuntimeError, match="cannot start.*dask-logs"):
        run_analysis._wait_for_workers_or_raise(
            client, requested_workers=100, min_workers=1, timeout_s=3600, label="dying",
            job_states=lambda: _states(idle=90, ended=10), poll_s=60)
    assert client.calls == 1
    held = _NoWorkerClient(now)
    with pytest.raises(RuntimeError, match="hold reason: proxy expired"):
        run_analysis._wait_for_workers_or_raise(
            held, requested_workers=2, min_workers=1, timeout_s=3600, label="held",
            job_states=lambda: _states(held=2, hold_reason="proxy expired"), poll_s=60)


def test_wait_for_workers_returns_when_a_worker_arrives(monkeypatch, tmp_path):
    now = _wait_setup(monkeypatch, tmp_path)
    client = _NoWorkerClient(now, arrive_on_call=3)
    run_analysis._wait_for_workers_or_raise(
        client, requested_workers=50, min_workers=1, timeout_s=3600, label="ok",
        job_states=lambda: _states(idle=50), poll_s=60)
    assert client.calls == 3


def test_wait_for_workers_survives_a_failing_schedd_query(monkeypatch, tmp_path):
    now = _wait_setup(monkeypatch, tmp_path)

    def broken():
        raise OSError("schedd unreachable")

    with pytest.raises(RuntimeError, match="Timed out waiting.*job states unknown"):
        run_analysis._wait_for_workers_or_raise(
            _NoWorkerClient(now), requested_workers=10, min_workers=1, timeout_s=600, label="q",
            job_states=broken, poll_s=60)

def test_preprocess_with_xrd_fallback_rewrites_fileset_and_retries(monkeypatch):
    fileset = {
        "dataset": {
            "files": {
                "root://cmsxrootd.fnal.gov//store/mc/a.root": "Events",
            },
            "metadata": {"sample": "Dummy"},
        }
    }

    class FakeRun:
        def __init__(self):
            self.calls = 0
            self.urls_seen = []

        def preprocess(self, fileset, treename):
            self.calls += 1
            self.urls_seen.append(next(iter(fileset["dataset"]["files"].keys())))
            if self.calls == 1:
                raise ValueError(
                    "not a ROOT file: first four bytes are b'\\x00\\x00\\x00\\x00'\n"
                    "in file root://cmsxrootd.fnal.gov//store/mc/a.root"
                )
            return {"preprocessed": True}

    resolve_kwargs = {}

    def _fake_resolve(*args, **kwargs):
        resolve_kwargs.update(kwargs)
        return RedirectorProbeResult(
            success=True,
            lfn="/store/mc/a.root",
            resolved_url="root://cms-xrd-global.cern.ch//store/mc/a.root",
            redirector="root://cms-xrd-global.cern.ch/",
            total_attempts=2,
            failures_by_redirector={"root://cmsxrootd.fnal.gov/": "ValueError: bad header"},
        )

    monkeypatch.setattr(run_analysis, "resolve_url_with_redirectors", _fake_resolve)
    fake_run = FakeRun()
    args = SimpleNamespace(
        unskimmed=True,
        xrd_fallback=True,
        xrd_fallback_timeout=1,
        xrd_fallback_retries_per_redirector=1,
        xrd_fallback_sleep=0,
    )

    out = run_analysis._preprocess_with_xrd_fallback(
        fake_run,
        fileset,
        treename="Events",
        args=args,
    )
    assert out == {"preprocessed": True}
    assert fake_run.calls == 2
    assert fake_run.urls_seen[0] == "root://cmsxrootd.fnal.gov//store/mc/a.root"
    assert fake_run.urls_seen[1] == "root://cms-xrd-global.cern.ch//store/mc/a.root"
    assert (
        next(iter(fileset["dataset"]["files"].keys()))
        == "root://cms-xrd-global.cern.ch//store/mc/a.root"
    )
    # The failing redirector (cmsxrootd.fnal.gov) should be excluded from the probe
    redirectors_used = resolve_kwargs.get("redirectors")
    assert redirectors_used is not None
    assert "root://cmsxrootd.fnal.gov/" not in redirectors_used


def test_preprocess_with_xrd_fallback_retries_when_same_redirector_resolves(monkeypatch):
    """When probe succeeds at the same redirector, retry preprocess instead of raising."""
    fileset = {
        "dataset": {
            "files": {
                "root://cmsxrootd.fnal.gov//store/mc/a.root": "Events",
            },
            "metadata": {"sample": "Dummy"},
        }
    }

    class FakeRun:
        def __init__(self):
            self.calls = 0

        def preprocess(self, fileset, treename):
            self.calls += 1
            if self.calls == 1:
                raise ValueError(
                    "not a ROOT file: first four bytes are b'\\x00\\x00\\x00\\x00'\n"
                    "in file root://cmsxrootd.fnal.gov//store/mc/a.root"
                )
            return {"preprocessed": True}

    def _fake_resolve(*args, **kwargs):
        # Same redirector resolves (e.g. routed to a different SE)
        return RedirectorProbeResult(
            success=True,
            lfn="/store/mc/a.root",
            resolved_url="root://cmsxrootd.fnal.gov//store/mc/a.root",
            redirector="root://cmsxrootd.fnal.gov/",
            total_attempts=1,
        )

    monkeypatch.setattr(run_analysis, "resolve_url_with_redirectors", _fake_resolve)
    fake_run = FakeRun()
    args = SimpleNamespace(
        unskimmed=True,
        xrd_fallback=True,
        xrd_fallback_timeout=1,
        xrd_fallback_retries_per_redirector=1,
        xrd_fallback_sleep=0,
    )

    out = run_analysis._preprocess_with_xrd_fallback(
        fake_run,
        fileset,
        treename="Events",
        args=args,
    )
    assert out == {"preprocessed": True}
    assert fake_run.calls == 2
    # Fileset key should be unchanged (no rewrite needed)
    assert next(iter(fileset["dataset"]["files"].keys())) == "root://cmsxrootd.fnal.gov//store/mc/a.root"


def _dy_args(era, dy):
    return SimpleNamespace(
        era=era,
        sample="DYJets",
        mass=None,
        dy=dy,
        max_workers=None,
        threads_per_worker=None,
        worker_wait_timeout=1200,
        chunksize=250_000,
        xrd_fallback_timeout=10,
        xrd_fallback_retries_per_redirector=2,
        xrd_fallback_sleep=1,
    )


@pytest.mark.parametrize(
    "era",
    ["Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix"],
)
def test_validate_dy_lo_ht_accepted_for_run3(era):
    """--dy lo_ht is now valid for Run3 eras (dy_variants in config.yaml)."""
    # Should not raise: Run3 DY LO HT configs/filesets exist now.
    run_analysis.validate_arguments(_dy_args(era, "lo_ht"), [])


def test_validate_dy_unknown_variant_rejected():
    """A DY variant not listed in DY_VARIANTS for the era must still raise."""
    with pytest.raises(
        ValueError, match="--dy nlo_ht is not available for Run3Summer22"
    ):
        run_analysis.validate_arguments(_dy_args("Run3Summer22", "nlo_ht"), [])


def test_validate_dy_lo_ht_accepted_for_ul18():
    """--dy lo_ht should be accepted for RunIISummer20UL18."""
    # Should not raise
    run_analysis.validate_arguments(_dy_args("RunIISummer20UL18", "lo_ht"), [])


def test_validate_dy_nlo_ptll_warns(caplog):
    """--dy nlo_ptll runs, with a warning that the unstitched samples are no DY prediction."""
    with caplog.at_level(logging.WARNING):
        run_analysis.validate_arguments(_dy_args("RunIII2024Summer24", "nlo_ptll"), [])
    assert "--dy nlo_ptll" in caplog.text and "unstitched" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run_analysis.validate_arguments(_dy_args("RunIII2024Summer24", "nlo_inc"), [])
    assert "nlo_ptll" not in caplog.text


@pytest.mark.parametrize("flags, expected", [
    ({}, "none"),  # the default since 2026-10-12
    ({"dy_reweight": "zpt"}, "zpt"),
    ({"dy_reweight": "zpt+reshape"}, "zpt+reshape"),
    # The legacy switches keep the output they gave when both corrections were on by default.
    ({"no_dy_zpt_reweight": True}, "none"),
    ({"no_dy_reshape": True}, "zpt"),
    ({"no_dy_zpt_reweight": True, "no_dy_reshape": True}, "none"),
])
def test_resolve_dy_reweight(flags, expected):
    assert run_analysis.resolve_dy_reweight(SimpleNamespace(**flags)) == expected


@pytest.mark.parametrize("dy, ok", [(None, True), ("lo_ht", True), ("nlo_inc", False), ("lo_inc", False)])
def test_validate_dy_reweight_only_for_lo_ht(dy, ok):
    args = _dy_args("RunIII2024Summer24", dy)
    args.dy_reweight = "zpt"
    if ok:
        run_analysis.validate_arguments(args, [])
    else:
        with pytest.raises(ValueError, match="has nothing to reweight"):
            run_analysis.validate_arguments(args, [])


def test_resolve_dy_reweight_legacy_flag_conflicts():
    with pytest.raises(ValueError, match="--no-dy-reshape cannot be combined with --dy-reweight"):
        run_analysis.resolve_dy_reweight(SimpleNamespace(dy_reweight="zpt+reshape", no_dy_reshape=True))


@pytest.mark.parametrize("systs, dy_reweight, error", [
    (["dyzpt"], None, "--dy-reweight zpt"),
    (["dyreshape"], None, "--dy-reweight zpt\\+reshape"),
    (["dyzpt", "dyreshape"], "zpt", "--dy-reweight zpt\\+reshape"),
    (["dyzpt"], "zpt", None),
    (["dyzpt", "dyreshape"], "zpt+reshape", None),
])
def test_validate_dy_systs_need_the_reweight(systs, dy_reweight, error):
    args = _dy_args("RunIISummer20UL18", None)
    args.systs, args.dy_reweight = systs, dy_reweight
    if error is None:
        run_analysis.validate_arguments(args, [])
    else:
        with pytest.raises(ValueError, match=error):
            run_analysis.validate_arguments(args, [])


def _unblind_args(era, unblind=True):
    args = _dy_args(era, None)
    args.sample = "Muon"
    args.unblind = unblind
    return args


@pytest.mark.parametrize(
    "era",
    ["Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix",
     "RunIII2024Summer24", "RunIII2025Summer25"],
)
def test_validate_unblind_rejected_for_run3(era):
    """Run 3 data signal regions stay blind: --unblind is refused up front."""
    with pytest.raises(ValueError, match=f"--unblind is not allowed for {era}"):
        run_analysis.validate_arguments(_unblind_args(era), [])
    # The blinded default needs no permission.
    run_analysis.validate_arguments(_unblind_args(era, unblind=False), [])


def test_validate_unblind_accepted_for_ul18():
    """--unblind is allowed for Run 2 (UL18)."""
    run_analysis.validate_arguments(_unblind_args("RunIISummer20UL18"), [])


class _ReachedProcessing(Exception):
    pass


def _run_cli(monkeypatch, *argv):
    """Run bin/run_analysis.py as a script until it starts the actual run."""
    import runpy
    import wrcoffea.era_utils as era_utils

    def stop(era):
        raise _ReachedProcessing(era)

    # The script looks up the era details right after validating its arguments
    # (the wrcoffea modules bound their own reference at import time).
    monkeypatch.setattr(era_utils, "get_era_details", stop)
    monkeypatch.chdir(os.path.dirname(BIN_DIR))
    monkeypatch.setattr(sys, "argv", ["run_analysis.py", *argv])
    runpy.run_path(os.path.join(BIN_DIR, "run_analysis.py"), run_name="__main__")


def test_cli_unblind_flag(monkeypatch, caplog):
    """--unblind on the command line is refused for Run 3 and accepted for UL18."""
    with pytest.raises(ValueError, match="--unblind is not allowed for RunIII2024Summer24"):
        _run_cli(monkeypatch, "RunIII2024Summer24", "Muon", "--unblind")
    with pytest.raises(_ReachedProcessing):
        _run_cli(monkeypatch, "RunIII2024Summer24", "Muon")
    assert "--unblind" not in caplog.text
    with pytest.raises(_ReachedProcessing):
        _run_cli(monkeypatch, "RunIISummer20UL18", "Muon", "--unblind")
    assert "--unblind: data signal regions" in caplog.text


@pytest.mark.parametrize("extra, expected", [({}, False), ({"unblind": False}, False),
                                             ({"unblind": True}, True)])
def test_process_fileset_forwards_unblind(monkeypatch, extra, expected):
    """_process_fileset hands --unblind to the processor (blinded by default)."""
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def preprocess(self, fileset, treename):
            return {}

        def __call__(self, preproc, treename, processor_instance):
            return {}, {}

    class FakeWrAnalysis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(coffea_processor, "DaskExecutor", lambda **kwargs: None)
    monkeypatch.setattr(coffea_processor, "Runner", FakeRunner)
    monkeypatch.setattr(analyzer_mod, "WrAnalysis", FakeWrAnalysis)
    args = SimpleNamespace(sample="Muon", mass=None, systs=[], region="both", unskimmed=False,
                           study=[], chunksize=1000, maxchunks=None, **extra)

    run_analysis._process_fileset(args, fileset={}, client=object(), condor=False)

    assert captured["unblind"] is expected


@pytest.mark.parametrize("dy_reweight, zpt, reshape", [(None, False, False), ("none", False, False),
                                                       ("zpt", True, False), ("zpt+reshape", True, True)])
def test_process_fileset_forwards_dy_reweight(monkeypatch, dy_reweight, zpt, reshape):
    """_process_fileset hands --dy-reweight to the processor (no DY reweighting by default)."""
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            pass

        def preprocess(self, fileset, treename):
            return {}

        def __call__(self, preproc, treename, processor_instance):
            return {}, {}

    class FakeWrAnalysis:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(coffea_processor, "DaskExecutor", lambda **kwargs: None)
    monkeypatch.setattr(coffea_processor, "Runner", FakeRunner)
    monkeypatch.setattr(analyzer_mod, "WrAnalysis", FakeWrAnalysis)
    args = SimpleNamespace(sample="DYJets", mass=None, systs=[], region="both", unskimmed=False,
                           study=[], chunksize=1000, maxchunks=None, dy_reweight=dy_reweight)

    run_analysis._process_fileset(args, fileset={}, client=object(), condor=False)

    assert captured["dy_zpt_reweight"] is zpt
    assert captured["dy_reshape"] is reshape


def test_preprocess_with_xrd_fallback_raises_after_all_redirectors_fail(monkeypatch):
    fileset = {
        "dataset": {
            "files": {
                "root://cmsxrootd.fnal.gov//store/mc/b.root": "Events",
            },
            "metadata": {"sample": "Dummy"},
        }
    }

    class FakeRun:
        def preprocess(self, fileset, treename):
            raise ValueError(
                "not a ROOT file: first four bytes are b'\\x00\\x00\\x00\\x00'\n"
                "in file root://cmsxrootd.fnal.gov//store/mc/b.root"
            )

    def _fake_resolve(*args, **kwargs):
        return RedirectorProbeResult(
            success=False,
            lfn="/store/mc/b.root",
            resolved_url=None,
            redirector=None,
            total_attempts=3,
            failures_by_redirector={
                "root://cmsxrootd.fnal.gov/": "ValueError: bad header",
                "root://cms-xrd-global.cern.ch/": "OSError: timeout",
                "root://xrootd-cms.infn.it/": "OSError: timeout",
            },
        )

    monkeypatch.setattr(run_analysis, "resolve_url_with_redirectors", _fake_resolve)
    args = SimpleNamespace(
        unskimmed=True,
        xrd_fallback=True,
        xrd_fallback_timeout=1,
        xrd_fallback_retries_per_redirector=1,
        xrd_fallback_sleep=0,
    )

    with pytest.raises(RuntimeError, match="XRootD fallback exhausted for /store/mc/b.root"):
        run_analysis._preprocess_with_xrd_fallback(
            FakeRun(),
            fileset,
            treename="Events",
            args=args,
        )


# ---------------------------------------------------------------------------
# Group splitting and local/Condor composite output parity
# ---------------------------------------------------------------------------

def test_split_sample_groups_remaps_subgroups_and_rejects_unknown():
    from wrcoffea.cli_utils import split_sample_groups

    mapping = {
        "TTto2L2Nu_TuneCP5": "tt_tW",
        "TWminus_2L": "tt_tW",
        "TbarWplus_2L": "tt_tW",
        "TTtoLNu2Q": "tt_tW",  # no subgroup prefix -> stays in the parent
        "DYto2L": "DYJets",
    }
    out = split_sample_groups(mapping, ["tt_tW"])
    assert out == {
        "TTto2L2Nu_TuneCP5": "TTbar",
        "TWminus_2L": "tW",
        "TbarWplus_2L": "tW",
        "TTtoLNu2Q": "tt_tW",
        "DYto2L": "DYJets",
    }
    # No split requested -> identity copy.
    assert split_sample_groups(mapping, []) == mapping
    with pytest.raises(ValueError):
        split_sample_groups(mapping, ["DYJets"])


def test_local_composite_saves_through_group_map(monkeypatch):
    """The local composite loop must honor the same group map as Condor.

    Regression: before the fix, the local path called save_histograms(args)
    per sample, so --split-groups (tt_tW -> TTbar/tW) only worked on Condor.
    """
    import wrcoffea.save_hists as save_hists_mod

    monkeypatch.setattr(run_analysis, "COMPOSITE_SAMPLES", {"bkg": ["DYJets", "tt_tW"]})
    monkeypatch.setattr(run_analysis, "build_fileset_path", lambda **kw: f"{kw['sample']}.json")
    monkeypatch.setattr(run_analysis, "load_and_select_fileset", lambda **kw: {"fs": kw["desired_process"]})

    per_sample_hists = {
        "DYJets": {"DYto2L": {"h": 1}},
        "tt_tW": {"TTto2L2Nu_TuneCP5": {"h": 2}, "TWminus_2L": {"h": 3}},
    }
    monkeypatch.setattr(
        run_analysis, "_process_fileset",
        lambda args, fileset, *, client, condor=False: per_sample_hists[args.sample],
    )

    saved_groups = []
    monkeypatch.setattr(
        save_hists_mod, "save_histograms",
        lambda hists, group_args: saved_groups.append((group_args.sample, sorted(hists))),
    )

    sample_to_group = {
        "DYto2L": "DYJets",
        "TTto2L2Nu_TuneCP5": "TTbar",
        "TWminus_2L": "tW",
    }
    args = SimpleNamespace(sample="bkg", mass=None, unskimmed=False, dy=None,
                           maxfiles=None, debug=False)

    run_analysis._run_local_composite(args, "RunIII2024Summer24", client=object(),
                                      sample_to_group=sample_to_group)

    assert sorted(saved_groups) == [
        ("DYJets", ["DYto2L"]),
        ("TTbar", ["TTto2L2Nu_TuneCP5"]),
        ("tW", ["TWminus_2L"]),
    ]
    # The loop restores the composite mode after temporarily rebinding args.sample.
    assert args.sample == "bkg" and args.mass is None


def test_validate_mass_outside_the_csv_needs_fileset():
    """Signal --mass must be a point of the era's CSV, unless --fileset names the fileset that holds it."""
    args = _dy_args("RunIISummer20UL18", None)
    args.sample, args.mass = "Signal", "WR3800_N1600"
    with pytest.raises(ValueError, match="Invalid mass"):
        run_analysis.validate_arguments(args, ["WR4000_N2100"])
    args.fileset = "data/filesets/RunII/2018/RunIISummer20UL18/skimmed/RunIISummer20UL18_signal_scan_fileset.json"
    run_analysis.validate_arguments(args, ["WR4000_N2100"])


def test_validate_signal_points_only_for_composites_with_signal():
    """--signal-points / --mwr-range pick the points of signal/mc/all; elsewhere they are refused."""
    single = _dy_args("RunIII2024Summer24", None)
    single.signal_points = ["all"]
    with pytest.raises(ValueError, match="use --mass"):
        run_analysis.validate_arguments(single, [])

    bkg = _dy_args("RunIII2024Summer24", None)
    bkg.sample, bkg.mwr_range = "bkg", [1000, 6000]
    with pytest.raises(ValueError, match="includes Signal"):
        run_analysis.validate_arguments(bkg, [])

    sig = _dy_args("RunIII2024Summer24", None)
    sig.sample, sig.signal_points, sig.mwr_range = "signal", ["all"], [1000, 6000]
    run_analysis.validate_arguments(sig, [])


# ---------------------------------------------------------------------------
# Condor workers: the files they read must be shipped to them
# ---------------------------------------------------------------------------

def _config_data_paths():
    """Every ``data/...`` string value of wrcoffea/config.yaml, as written there."""
    import yaml
    from wrcoffea.analysis_config import REPO_ROOT

    with open(REPO_ROOT / "wrcoffea" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    found = []

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str) and node.startswith("data/"):
            found.append(node)

    walk(cfg)
    return found


def test_condor_ships_every_data_dir_the_config_uses():
    """A data/ subdirectory of config.yaml missing from CONDOR_DATA_DIRS fails on every Condor worker.

    data/reweights (DY Z-pT, reshape, signal k-factor), data/pdf_sets and data/muon_highpt were
    not shipped until 2026-10-12, so DY, signal and 2024 runs with --condor could not open them.
    """
    paths = _config_data_paths()
    assert paths
    used = {p.split("/")[1] for p in paths}
    assert used <= set(run_analysis.CONDOR_DATA_DIRS), sorted(used - set(run_analysis.CONDOR_DATA_DIRS))


def test_condor_worker_sandbox_resolves_every_config_path(monkeypatch, tmp_path):
    """Build the cluster with a fake lpcjobqueue, then replay a worker: Condor puts each transferred
    directory flat in the sandbox, the pickled worker plugin links the data ones back under data/,
    and every config.yaml data path that exists in the repository must exist in the sandbox."""
    import types
    import cloudpickle
    import dask.distributed
    from wrcoffea.analysis_config import REPO_ROOT

    captured = {}

    class FakeCluster:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def scale(self, n):
            pass

        def close(self):
            pass

    class FakeClient:
        def __init__(self, cluster):
            pass

        def register_plugin(self, plugin, name=None):
            captured["plugin"] = plugin

        def scheduler_info(self):
            return {"workers": {}}

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "lpcjobqueue", types.SimpleNamespace(LPCCondorCluster=FakeCluster))
    monkeypatch.setattr(dask.distributed, "Client", FakeClient)
    monkeypatch.setattr(run_analysis, "_wait_for_workers_or_raise", lambda *a, **k: None)
    monkeypatch.setattr(run_analysis, "_check_condor_environment", lambda *a, **k: None)
    monkeypatch.setenv("USER", "tester")
    with run_analysis._condor_cluster(n_workers=1, wait_timeout_s=1, memory="3GB"):
        pass
    assert captured["memory"] == "3GB"

    shipped = [os.path.realpath(p) for p in captured["transfer_input_files"]]
    for subdir in run_analysis.CONDOR_DATA_DIRS:
        assert os.path.realpath(REPO_ROOT / "data" / subdir) in shipped

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    for path in shipped:
        os.symlink(path, sandbox / os.path.basename(path))
    monkeypatch.chdir(sandbox)
    monkeypatch.setattr(sys, "path", list(sys.path))   # setup() prepends "." and "wrcoffea"
    plugin = cloudpickle.loads(cloudpickle.dumps(captured["plugin"]))
    plugin.setup(worker=None)

    missing = [p for p in _config_data_paths() if (REPO_ROOT / p).exists() and not (sandbox / p).exists()]
    assert not missing, missing


def test_condor_environment_check_refuses_user_site_packages(monkeypatch, tmp_path):
    """A worker package imported from the user site (~/.local) is missing on every Condor worker:
    stop before submitting instead of waiting for workers that die at start (2026-10-14: an
    lpcjobqueue in ~/.local, so the bootstrap left .env without it)."""
    import site
    import types

    user_site = tmp_path / "user-site"
    (user_site / "fakepkg").mkdir(parents=True)
    module = types.ModuleType("fakepkg")
    module.__file__ = str(user_site / "fakepkg" / "__init__.py")
    monkeypatch.setitem(sys.modules, "fakepkg", module)
    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(user_site))

    with pytest.raises(RuntimeError, match="user site"):
        run_analysis._check_condor_environment(modules=("fakepkg",))
    run_analysis._check_condor_environment(modules=("json",))   # the standard library is fine

    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)          # PYTHONNOUSERSITE=1
    run_analysis._check_condor_environment(modules=("fakepkg",))


def test_condor_cluster_start_retries_a_busy_scheduler_port():
    """lpcjobqueue draws the scheduler port from 10000-10100 at random: a run sharing the login
    node with other schedulers can draw a busy one. Retry that case, raise anything else."""
    import errno

    calls = []

    def busy_then_free(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            try:
                raise OSError(errno.EADDRINUSE, "Address already in use")
            except OSError as exc:
                raise RuntimeError("Scheduler failed to start.") from exc
        return "cluster"

    assert run_analysis._start_lpc_cluster(busy_then_free, memory="4GB") == "cluster"
    assert len(calls) == 3 and calls[-1] == {"memory": "4GB"}

    def broken(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("$COFFEA_IMAGE_FULL environment not set")

    calls.clear()
    with pytest.raises(RuntimeError, match="COFFEA_IMAGE_FULL"):
        run_analysis._start_lpc_cluster(broken)
    assert len(calls) == 1

    def always_busy(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("Likely failed to bind to local port 10042, try rerunning")

    calls.clear()
    with pytest.raises(RuntimeError, match="bind to local port"):
        run_analysis._start_lpc_cluster(always_busy, attempts=2)
    assert len(calls) == 2


def test_condor_job_states_counts_only_this_clusters_jobs(monkeypatch):
    """One Condor job per worker: count their states; a job gone from the queue has ended."""
    import concurrent.futures
    import types

    class FakeSchedd:
        def query(self, constraint, projection):
            assert constraint == "ClusterId >= 101 && ClusterId <= 107"
            return [{"ClusterId": 101, "JobStatus": 1}, {"ClusterId": 103, "JobStatus": 2},
                    {"ClusterId": 105, "JobStatus": 5, "HoldReason": "memory exceeded"},
                    {"ClusterId": 104, "JobStatus": 1}]   # another submission in the range

    pool = concurrent.futures.ThreadPoolExecutor(1)
    fake = types.SimpleNamespace(SCHEDD=lambda: FakeSchedd(), SCHEDD_POOL=pool)
    monkeypatch.setitem(sys.modules, "lpcjobqueue", types.SimpleNamespace(schedd=fake))
    monkeypatch.setitem(sys.modules, "lpcjobqueue.schedd", fake)
    cluster = types.SimpleNamespace(workers={f"w{i}": types.SimpleNamespace(job_id=i) for i in (101, 103, 105, 107)})
    cluster.workers["pending"] = types.SimpleNamespace(job_id=None)   # not submitted yet

    states = run_analysis._condor_job_states(cluster)
    assert states == {"idle": 1, "running": 1, "held": 1, "ended": 1, "hold_reason": "memory exceeded"}
    assert run_analysis._condor_job_states(types.SimpleNamespace(workers={})) is None
    pool.shutdown()


# ---------------------------------------------------------------------------
# A long-lived cluster shared by several runs (bin/condor_cluster.py, --scheduler)
# ---------------------------------------------------------------------------

def _fake_repo(root, analyzer_text="x = 1\n"):
    (root / "wrcoffea" / "__pycache__").mkdir(parents=True)
    (root / "wrcoffea" / "analyzer.py").write_text(analyzer_text)
    (root / "wrcoffea" / "config.yaml").write_text("a: 1\n")
    (root / "wrcoffea" / "__pycache__" / "analyzer.cpython-312.pyc").write_bytes(b"stale")
    for subdir in run_analysis.CONDOR_DATA_DIRS:
        (root / "data" / subdir).mkdir(parents=True)
        (root / "data" / subdir / "payload.json").write_text(subdir)
    (root / "bin").mkdir()
    (root / "bin" / "run_analysis.py").write_text("client side\n")
    return root


def test_worker_files_cover_what_the_workers_run(tmp_path):
    """wrcoffea/ and the shipped data/ subdirectories, without __pycache__ or bin/ (client side);
    two checkouts with the same contents give the same files."""
    a = run_analysis._worker_files(_fake_repo(tmp_path / "a"))
    assert set(a) == {"wrcoffea/analyzer.py", "wrcoffea/config.yaml",
                      *(f"data/{d}/payload.json" for d in run_analysis.CONDOR_DATA_DIRS)}
    assert run_analysis._worker_files(_fake_repo(tmp_path / "b")) == a
    assert run_analysis._worker_files(_fake_repo(tmp_path / "c", "x = 2\n")) != a


def test_cluster_identity_check(tmp_path, monkeypatch):
    """A run may use a long-lived cluster only if the workers' copy, packages and image are its own."""
    monkeypatch.setenv("COFFEA_IMAGE_FULL", "/cvmfs/image-a")
    repo = _fake_repo(tmp_path / "repo")
    identity = run_analysis._cluster_identity(n_workers=5, memory="6GB", repo_root=repo)
    run_analysis._check_cluster_identity(identity, repo_root=repo)

    (repo / "wrcoffea" / "analyzer.py").write_text("x = 2\n")
    (repo / "data" / "jsonpog" / "new.json").write_text("{}")
    with pytest.raises(RuntimeError, match=r"2 file\(s\).*data/jsonpog/new.json, wrcoffea/analyzer.py.*start it again"):
        run_analysis._check_cluster_identity(identity, repo_root=repo)

    fresh = run_analysis._cluster_identity(n_workers=5, memory="6GB", repo_root=repo)
    fresh["packages"]["coffea"] = "0.0.1"
    with pytest.raises(RuntimeError, match="coffea 0.0.1 there"):
        run_analysis._check_cluster_identity(fresh, repo_root=repo)
    other_image = run_analysis._cluster_identity(n_workers=5, memory="6GB", repo_root=repo)
    monkeypatch.setenv("COFFEA_IMAGE_FULL", "/cvmfs/image-b")
    with pytest.raises(RuntimeError, match="another image"):
        run_analysis._check_cluster_identity(other_image, repo_root=repo)
    with pytest.raises(RuntimeError, match="not started by bin/condor_cluster.py"):
        run_analysis._check_cluster_identity(None, repo_root=repo)


def test_attached_cluster_checks_identity_and_leaves_the_workers(monkeypatch, tmp_path):
    """--scheduler: connect (address or scheduler file), check, wait; on exit close only the client."""
    import dask.distributed

    events = []
    identity = {"started": "2026-10-14T20:00:00+00:00", "workers": 40}

    class FakeClient:
        def __init__(self, address=None, scheduler_file=None, timeout=None):
            events.append(("connect", address, scheduler_file))

        def get_metadata(self, key, default=None):
            assert key == run_analysis.CLUSTER_IDENTITY_KEY
            return identity

        def scheduler_info(self):
            return {"workers": {"w1": {}, "w2": {}}}

        def close(self):
            events.append(("close",))

    monkeypatch.setattr(dask.distributed, "Client", FakeClient)
    monkeypatch.setattr(run_analysis, "_check_cluster_identity", lambda ident: events.append(("check", ident)))
    monkeypatch.setattr(run_analysis, "_wait_for_workers_or_raise",
                        lambda client, **kw: events.append(("wait", kw["requested_workers"])))

    with run_analysis._attached_cluster("tcp://node:10042", wait_timeout_s=10):
        events.append(("work",))
    assert events == [("connect", "tcp://node:10042", None), ("check", identity), ("wait", 40),
                      ("work",), ("close",)]

    events.clear()
    scheduler_file = tmp_path / "condor_scheduler.json"
    scheduler_file.write_text('{"address": "tcp://node:10042"}')
    with run_analysis._attached_cluster(str(scheduler_file), wait_timeout_s=10):
        pass
    assert events[0] == ("connect", None, str(scheduler_file))

    events.clear()
    monkeypatch.setattr(run_analysis, "_check_cluster_identity",
                        lambda ident: (_ for _ in ()).throw(RuntimeError("differs")))
    with pytest.raises(RuntimeError, match="differs"):
        with run_analysis._attached_cluster("tcp://node:10042", wait_timeout_s=10):
            pass
    assert events[-1] == ("close",)


def test_condor_runs_refuse_a_config_overlay(monkeypatch):
    """The workers read config.yaml as shipped, never the overlay."""
    args = _dy_args("RunIII2024Summer24", None)
    monkeypatch.setenv("WRCOFFEA_CONFIG_OVERLAY", "/tmp/overlay.yaml")
    run_analysis.validate_arguments(args, [])            # local: allowed
    args.condor = True
    with pytest.raises(ValueError, match="never read it"):
        run_analysis.validate_arguments(args, [])
    args.condor, args.scheduler = False, "condor_scheduler.json"
    with pytest.raises(ValueError, match="never read it"):
        run_analysis.validate_arguments(args, [])


def test_scheduler_option_rules():
    """--scheduler starts no workers, so it refuses the options that size them."""
    import subprocess

    script = os.path.join(BIN_DIR, "run_analysis.py")
    out = subprocess.run([sys.executable, script, "RunIII2024Summer24", "DYJets", "--scheduler", "x.json",
                          "--max-workers", "3", "--preflight-only"], capture_output=True, text=True,
                         cwd=os.path.dirname(BIN_DIR))
    assert out.returncode == 2 and "size the workers of bin/condor_cluster.py" in out.stderr
