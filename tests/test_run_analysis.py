"""Tests for bin/run_analysis.py helpers."""

import os
import sys
from types import SimpleNamespace

import pytest
import uproot

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


def test_process_fileset_skipbadfiles_includes_missing_events_exceptions(monkeypatch):
    """Ensure missing Events-tree files are skippable in preprocess/processing."""

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
        mass=None,
        systs=[],
        region="both",
        unskimmed=True,
        tf_study=False,
        chunksize=1000,
        maxchunks=None,
    )

    run_analysis._process_fileset(args, fileset={}, client=object(), condor=False)

    skipbadfiles = captured["skipbadfiles"]
    if skipbadfiles is not True:
        assert OSError in skipbadfiles
        assert ValueError in skipbadfiles
        assert any(
            issubclass(exc, uproot.exceptions.KeyInFileError) for exc in skipbadfiles
        )


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


def test_wait_for_workers_or_raise_timeout_has_diagnostics(monkeypatch, tmp_path):
    class FakeClient:
        def wait_for_workers(self, min_workers, timeout):
            raise TimeoutError("no workers")

    diag_file = tmp_path / "diag.log"
    diag_file.write_text("stub\n", encoding="utf-8")
    monkeypatch.setattr(run_analysis, "_dump_dask_diagnostics", lambda client, label: diag_file)

    with pytest.raises(RuntimeError, match="Timed out waiting for at least 1 Condor worker"):
        run_analysis._wait_for_workers_or_raise(
            FakeClient(),
            requested_workers=200,
            min_workers=1,
            timeout_s=600,
            label="unit_test_timeout",
        )


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


def test_validate_dy_lo_ht_rejected_for_run3():
    """--dy lo_ht should only be valid for RunIISummer20UL18."""
    args = SimpleNamespace(
        era="Run3Summer22",
        sample="DYJets",
        mass=None,
        reweight=None,
        dy="lo_ht",
        max_workers=None,
        threads_per_worker=None,
        worker_wait_timeout=1200,
        chunksize=250_000,
        xrd_fallback_timeout=10,
        xrd_fallback_retries_per_redirector=2,
        xrd_fallback_sleep=1,
    )
    with pytest.raises(ValueError, match="--dy lo_ht is not available for Run3Summer22"):
        run_analysis.validate_arguments(args, [])


def test_validate_dy_lo_ht_accepted_for_ul18():
    """--dy lo_ht should be accepted for RunIISummer20UL18."""
    args = SimpleNamespace(
        era="RunIISummer20UL18",
        sample="DYJets",
        mass=None,
        reweight=None,
        dy="lo_ht",
        max_workers=None,
        threads_per_worker=None,
        worker_wait_timeout=1200,
        chunksize=250_000,
        xrd_fallback_timeout=10,
        xrd_fallback_retries_per_redirector=2,
        xrd_fallback_sleep=1,
    )
    # Should not raise
    run_analysis.validate_arguments(args, [])


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
