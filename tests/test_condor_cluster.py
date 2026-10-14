"""Tests for bin/condor_cluster.py: when the shared cluster counts as busy, and its scheduler file."""

import json
import os
import sys
from types import SimpleNamespace

BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)

import condor_cluster  # noqa: E402


def _scheduler(*, queued=(), unrunnable=None, processing=(), clients=("own", "fire-and-forget")):
    return SimpleNamespace(queued=list(queued), unrunnable=dict(unrunnable or {}),
                           workers={"w1": SimpleNamespace(processing=set(processing))},
                           clients={c: None for c in clients})


def test_busy_while_work_runs_or_a_run_is_attached():
    assert not condor_cluster.scheduler_busy(_scheduler(), "own")
    assert condor_cluster.scheduler_busy(_scheduler(queued=["t"]), "own")
    assert condor_cluster.scheduler_busy(_scheduler(unrunnable={"t": 1}), "own")
    assert condor_cluster.scheduler_busy(_scheduler(processing=["t"]), "own")
    # A run_analysis.py attached, e.g. while it saves its histograms between computations.
    assert condor_cluster.scheduler_busy(_scheduler(clients=("own", "fire-and-forget", "Client-run")), "own")


def test_scheduler_file(tmp_path):
    """dask's format (Client(scheduler_file=...) reads "address"); removed only by its own cluster."""
    path = tmp_path / "condor_scheduler.json"
    identity = {"started": "2026-10-14T20:00:00+00:00", "workers": 200, "memory": "6GB"}
    condor_cluster.write_scheduler_file(path, "tcp://node:10042", identity)
    record = json.loads(path.read_text())
    assert record["address"] == "tcp://node:10042" and record["workers"] == 200

    condor_cluster.remove_scheduler_file(path, "tcp://node:10099")   # another cluster's: kept
    assert path.exists()
    condor_cluster.remove_scheduler_file(path, "tcp://node:10042")
    assert not path.exists()
    condor_cluster.remove_scheduler_file(path, "tcp://node:10042")   # already gone: no error


def test_busy_checks_do_not_flood_the_log():
    import logging

    quiet = condor_cluster.QuietBusyChecks()
    make = lambda msg, *args: logging.LogRecord("distributed.scheduler", logging.INFO, "", 0, msg, args, None)
    assert not quiet.filter(make("Run out-of-band function %r", "scheduler_busy"))
    assert quiet.filter(make("Run out-of-band function %r", "peek"))
    assert quiet.filter(make("Register worker addr: %s", "tcp://w:10000"))
