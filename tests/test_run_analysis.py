"""Tests for bin/run_analysis.py helpers."""

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
