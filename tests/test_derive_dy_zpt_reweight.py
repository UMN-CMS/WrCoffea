"""Run 3 DY Z-pT derivation script: dataset bookkeeping, per-file filling, stitching, payload."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import awkward as ak
import numpy as np
import pytest
import uproot

from wrcoffea import dy_reweight

REPO = Path(__file__).resolve().parents[1]
ERA = "RunIII2024Summer24"
PAYLOAD = REPO / "data/reweights/Run3/2024/RunIII2024Summer24/dy_zpt_reweight_run3_20261008.json"


@pytest.fixture(scope="module")
def s():
    spec = importlib.util.spec_from_file_location("derive_dy_zpt_reweight",
                                                  REPO / "scripts" / "derive_dy_zpt_reweight.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def test_pick_files_spreads_and_caps(s):
    files = [SimpleNamespace(lfn=f"/f{i:03d}.root", events=100) for i in range(50)]
    picked = s.pick_files(files, target_events=350, max_files=8)
    assert [f.lfn for f in picked] == ["/f000.root", "/f016.root", "/f033.root", "/f049.root"]
    assert len(s.pick_files(files, target_events=10**9, max_files=8)) == 8
    assert len(s.pick_files(files[:1], target_events=10**9, max_files=8)) == 1


def test_2024_datasets_and_slices(s):
    ds = s.select_datasets(ERA)
    assert len(ds["lo_ht"]) == 14 and len(ds["nlo_ptll"]) == 4
    assert all("MLL-50_" in d["sample"] for d in ds["lo_inc"] + ds["nlo_inc"])
    assert len(ds["lo_inc"]) == 2 and len(ds["nlo_inc"]) == 2
    edges = s.slice_edges(ds)
    assert edges["LHE_HT"] == [0.0, 40.0, 70.0, 100.0, 400.0, 800.0, 1500.0, 2500.0, math.inf]
    assert edges["LHE_Vpt"] == [0.0, 100.0, 200.0, 400.0, 600.0, math.inf]
    ptll = {d["sample"].split("PTLL-")[1].split("_")[0]: s.kept_slice(d, edges["LHE_Vpt"]) for d in ds["nlo_ptll"]}
    assert ptll == {"100": (100.0, 200.0), "200": (200.0, 400.0), "400": (400.0, 600.0), "600": (600.0, math.inf)}
    assert s.kept_slice(ds["lo_inc"][0], edges["LHE_HT"]) == (0.0, 40.0)
    assert s.kept_slice(ds["nlo_inc"][0], edges["LHE_Vpt"]) == (0.0, 100.0)


# ---------------------------------------------------------------------------
# fill_file on a synthetic NanoAOD file
# ---------------------------------------------------------------------------

def _lepton_pair(pt, pdg, status=1):
    """Two back-to-back hard-process leptons of transverse momentum *pt* (Z pT = 0 is avoided below)."""
    return {"pt": [pt, pt], "eta": [0.0, 0.0], "phi": [0.0, 3.0], "mass": [0.0, 0.0],
            "pdgId": [pdg, -pdg], "status": [status, status], "statusFlags": [256, 256],
            "genPartIdxMother": [-1, -1]}


def _write_nano(path, events):
    """TTrees laid out like NanoAOD (a dict assignment would write an RNTuple)."""
    data = {f"GenPart_{k}": ak.Array([e["gen"][k] for e in events]) for k in events[0]["gen"]}
    data["genWeight"] = np.array([e["w"] for e in events], dtype=np.float32)
    data["LHE_Vpt"] = np.array([e["vpt"] for e in events], dtype=np.float32)
    data["LHEScaleWeight"] = ak.Array([[1.0 + 0.1 * (i - 4) for i in range(9)] for _ in events])
    data["LHEPdfWeight"] = ak.Array([[1.0] * 101 + [0.9, 1.1] for _ in events])
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: v.type if isinstance(v, ak.Array) else v.dtype for k, v in data.items()})
        f["Events"].extend(data)
        f.mktree("Runs", {"genEventSumw": np.float64})
        f["Runs"].extend({"genEventSumw": np.array([sum(e["w"] for e in events)])})


def test_fill_file_flavour_slices_and_variations(s, tmp_path, monkeypatch):
    events = [
        {"gen": _lepton_pair(100.0, 11), "w": 1.0, "vpt": 50.0},     # ee, slice 0
        {"gen": _lepton_pair(100.0, 13), "w": 2.0, "vpt": 150.0},    # mumu, slice 1
        {"gen": _lepton_pair(100.0, 15, 2), "w": 1.0, "vpt": 150.0}, # tautau: nominal only
        {"gen": _lepton_pair(10.0, 13), "w": 1.0, "vpt": 50.0},      # m(Z) < 50: dropped
    ]
    _write_nano(tmp_path / "nano.root", events)
    monkeypatch.setattr(s, "REDIRECTORS", [""])
    out = s.fill_file(str(tmp_path / "nano.root"), "LHE_Vpt", [0.0, 100.0, math.inf], with_lhe=True)
    h = {k: np.asarray(v) for k, v in out["hist"].items()}
    st = out["stats"]
    assert st["events"] == 4 and st["sumw_all"] == pytest.approx(5.0) and st["mz_lt_50"] == 1
    assert st["runs_sumw"] == pytest.approx(5.0)
    # pair pT: two leptons at phi 0 and 3 rad -> |p1 + p2| = 2 pt cos(1.5)
    zpt = 2 * 100.0 * math.cos(1.5)
    ib = int(np.searchsorted(s.PT_EDGES, zpt, side="right") - 1)
    assert h["sumw"][0, 0, ib] == pytest.approx(1.0)
    assert h["sumw"][1, 1, ib] == pytest.approx(2.0)
    assert h["sumw"][2, 1, ib] == pytest.approx(1.0)
    assert h["sumw"].sum() == pytest.approx(4.0)
    # LHE variations hold ee + mumu only
    assert h["scale"][:, 4].sum() == pytest.approx(3.0)
    assert h["scale"][1, 8, ib] == pytest.approx(2.0 * 1.4)
    assert h["pdf"][0, 102, ib] == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Stitching
# ---------------------------------------------------------------------------

def _row(role, sample, xsec, sumw, edges, kept, per_slice):
    """A cached dataset whose ee events sit in pT bin 0 with the given per-slice weights."""
    nb, ns = len(s_edges := s_mod().PT_EDGES) - 1, len(edges) - 1
    h = np.zeros((3, ns, nb))
    h[0, :, 0] = per_slice
    return {"role": role, "sample": sample, "xsec": xsec, "stats": {"sumw_all": sumw},
            "slice_edges": edges, "kept_slice": kept, "slice_var": "LHE_Vpt",
            "hist": {"sumw": h, "sumw2": h.copy(), "entries": h.copy()}}


def s_mod():
    spec = importlib.util.spec_from_file_location("derive_dy_zpt_reweight_b",
                                                  REPO / "scripts" / "derive_dy_zpt_reweight.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_open_ended_samples_are_not_added_up(s):
    edges = [0.0, 100.0, 200.0, math.inf]
    inc = _row("nlo_inc", "inc_MLL-50_", 10.0, 10.0, edges, [0.0, 100.0], [8.0, 1.5, 0.5])
    p100 = _row("nlo_ptll", "PTLL-100_", 2.0, 2.0, edges, [100.0, 200.0], [0.0, 1.5, 0.5])   # pT > 100
    p200 = _row("nlo_ptll", "PTLL-200_", 0.5, 0.5, edges, [200.0, math.inf], [0.0, 0.0, 0.5])  # pT > 200
    # stitched: inclusive below 100, PTLL-100 in [100, 200), PTLL-200 above
    assert s.spectrum([inc, p100, p200])[0] == pytest.approx(8.0 + 1.5 + 0.5)
    lo = _row("lo_inc", "lo_MLL-50_", 10.0, 10.0, edges, [0.0, 100.0], [8.0, 1.5, 0.5])
    lo_b = _row("lo_ht", "Bin-HT-100", 2.0, 2.0, edges, [100.0, math.inf], [0.0, 1.5, 0.5])
    checks = {tuple(c["slice"]): c["inclusive_over_binned"] for c in s.stitch_checks([inc, p100, p200, lo, lo_b])
              if c["slice_var"] == "LHE_Vpt" and c["binned_pb"] > 0}
    assert checks[(100.0, 200.0)] == pytest.approx(1.0)
    assert checks[(200.0, "inf")] == pytest.approx(1.0)        # PTLL-100's tail is not added in


# ---------------------------------------------------------------------------
# The committed 2024 payload
# ---------------------------------------------------------------------------

def test_2024_payload_loads_and_evaluates():
    payload = dy_reweight.load_payload(str(PAYLOAD))
    raw = json.loads(PAYLOAD.read_text())
    assert raw["provenance"]["era"] == ERA
    q = payload["qcd"]
    assert len(q["central"]) == len(q["edges"]) - 1 == 28
    assert np.all(q["scale_down"] <= q["central"] + 1e-12) and np.all(q["central"] <= q["scale_up"] + 1e-12)
    assert abs(raw["provenance"]["integrated"]["nlo_over_lo"] - 1) < 0.02
    w = dy_reweight.evaluate(np.array([85.0, 9000.0]), np.array([True, False]), payload)
    i = int(np.searchsorted(q["edges"], 85.0, side="right") - 1)
    e = int(np.searchsorted(payload["ew"]["edges"], 85.0, side="right") - 1)
    assert w["nominal"][0] == pytest.approx(q["central"][i] * payload["ew"]["central"][e])
    assert w["nominal"][1] == 1.0                               # no generator pair
