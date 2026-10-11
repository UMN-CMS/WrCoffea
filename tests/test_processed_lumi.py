"""scripts/processed_lumi.py: channel luminosity from per-lumisection luminosities and processed masks."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("processed_lumi", REPO / "scripts" / "processed_lumi.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)


def _ls(run, a, b):
    return {(run, l) for l in range(a, b + 1)}


def test_halved_family_gives_mean_of_halves():
    byls = {x: 1.0 for x in _ls(100, 1, 10)}
    golden = set(byls)
    masks = {"Muon0_Run2023C": _ls(100, 1, 10), "Muon1_Run2023C": _ls(100, 1, 8)}     # half 1 lacks LS 9-10
    r = pl.compute(byls, golden, masks)
    assert r["channel_by_period"]["muon"]["Run2023C"] == pytest.approx(9.0)              # (10 + 8) / 2
    assert r["per_stream"]["Muon1_Run2023C"]["lumi_fb"] == pytest.approx(8.0)


def test_disjoint_families_add_and_overlaps_are_not_double_counted():
    byls = {x: 1.0 for x in _ls(200, 1, 10) | _ls(201, 1, 10)}
    golden = set(byls)
    # 2022C: SingleMuon for run 200, Muon for run 201 (and, wrongly, one LS of 200 too)
    masks = {"SingleMuon_Run2022C": _ls(200, 1, 10), "Muon_Run2022C": _ls(201, 1, 10) | {(200, 5)},
             "EGamma_Run2022C": _ls(200, 1, 10) | _ls(201, 1, 9)}
    r = pl.compute(byls, golden, masks)
    assert r["channel_by_period"]["muon"]["Run2022C"] == pytest.approx(20.0)      # union, LS (200,5) counted once
    assert r["channel_by_period"]["electron"]["Run2022C"] == pytest.approx(19.0)
    assert r["golden_runs_in_no_dataset"] == []


def test_non_golden_lumisections_do_not_count():
    byls = {x: 2.0 for x in _ls(300, 1, 4)}
    golden = _ls(300, 1, 2)
    r = pl.compute(byls, golden, {"EGamma0_Run2024B": _ls(300, 1, 4), "EGamma1_Run2024B": _ls(300, 1, 4)})
    assert r["channel_by_period"]["electron"]["Run2024B"] == pytest.approx(4.0)


def test_version_splits_of_one_period_are_a_union_not_halves():
    """2016B comes as SingleMuon_Run2016B_ver1 (runs before the golden JSON) and _ver2 (the golden runs): they are
    disjoint run ranges, not Run 3-style halves, so a lumisection held by one of them counts fully."""
    byls = {(273150, i): 1.0 for i in range(1, 11)}
    golden = set(byls)
    masks = {"SingleMuon_Run2016B_ver1": set(_ls(272007, 1, 10)),       # not golden at all
             "SingleMuon_Run2016B_ver2": set(_ls(273150, 1, 10))}
    res = pl.compute(byls, golden, masks)
    assert res["channel_total_fb"]["muon"] == pytest.approx(10.0)
    assert res["per_stream"]["SingleMuon_Run2016B_ver1"]["golden_ls"] == 0

