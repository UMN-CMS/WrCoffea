"""DY Z-pT derivation script (the AN-19-083 recipe): dataset bookkeeping, per-file filling, the ratio, payload."""

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

def test_pick_files_spreads_and_counts_the_picked_files(s):
    files = [SimpleNamespace(lfn=f"/f{i:03d}.root", events=100) for i in range(50)]
    picked = s.pick_files(files, target_events=350, max_files=8)
    assert [f.lfn for f in picked] == ["/f000.root", "/f016.root", "/f033.root", "/f049.root"]
    assert len(s.pick_files(files, target_events=10**9, max_files=8)) == 8
    assert len(s.pick_files(files[:1], target_events=10**9, max_files=8)) == 1
    # uneven files: the spread grows until the picked files themselves hold the target, whatever the mean
    uneven = [SimpleNamespace(lfn=f"/g{i:02d}.root", events=(1_000_000 if i % 2 else 10_000)) for i in range(10)]
    picked = s.pick_files(uneven, target_events=500_000, max_files=10)
    assert sum(f.events for f in picked) >= 500_000
    # the default reads everything, in LFN order
    assert [f.lfn for f in s.pick_files(files)] == [f.lfn for f in files]
    assert len(s.pick_files(files, target_events=350)) == 4           # no cap needed for a spread


def test_2024_datasets_are_the_inclusive_samples(s):
    """Run 3: the inclusive LO (4Jets MLM) and NLO (2Jets FxFx) MLL-50 samples, one per flavour; the MLL-10to50
    companions and every binned sample stay out."""
    ds = s.select_datasets(ERA)
    assert set(ds) == {"lo", "nlo"}
    assert len(ds["lo"]) == 2 and len(ds["nlo"]) == 2
    assert all("MLL-50_" in d["sample"] and "madgraphMLM" in d["sample"] for d in ds["lo"])
    assert all("MLL-50_" in d["sample"] and "amcatnloFXFX" in d["sample"] for d in ds["nlo"])
    assert not any("Bin-HT" in d["sample"] or "PTLL" in d["sample"] for d in sum(ds.values(), []))


def test_ul_datasets_drop_low_mass_and_count_ext_once(s):
    """Run 2 UL: M-50 madgraphMLM (two DAS paths, counted once) and M-50 amcatnloFXFX; M-10to50 dropped."""
    assert s._is_low_mass("DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8")
    assert not s._is_low_mass("DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8")
    era = "RunIISummer20UL17"
    if not s.config_path(era, "dy_nlo_inc").exists():
        pytest.skip("UL17 DY configs not installed")
    ds = s.select_datasets(era)
    assert [d["sample"] for d in ds["lo"]] == ["DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8"]
    assert [d["sample"] for d in ds["nlo"]] == ["DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"]
    assert ds["lo"][0]["xsec"] == pytest.approx(ds["nlo"][0]["xsec"])      # both at the inclusive NNLO value


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
    data["LHEScaleWeight"] = ak.Array([[1.0 + 0.1 * (i - 4) for i in range(9)] for _ in events])
    data["LHEPdfWeight"] = ak.Array([[1.0] * 101 + [0.9, 1.1] for _ in events])
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: v.type if isinstance(v, ak.Array) else v.dtype for k, v in data.items()})
        f["Events"].extend(data)
        f.mktree("Runs", {"genEventSumw": np.float64})
        f["Runs"].extend({"genEventSumw": np.array([sum(e["w"] for e in events)])})


def test_fill_file_flavours_and_variations(s, tmp_path, monkeypatch):
    events = [
        {"gen": _lepton_pair(100.0, 11), "w": 1.0},         # ee
        {"gen": _lepton_pair(100.0, 13), "w": 2.0},         # mumu
        {"gen": _lepton_pair(100.0, 15, 2), "w": 1.0},      # tautau: nominal only
        {"gen": _lepton_pair(10.0, 13), "w": 1.0},          # m(Z) < 50: dropped
    ]
    _write_nano(tmp_path / "nano.root", events)
    monkeypatch.setattr(s, "REDIRECTORS", [""])
    out = s.fill_file(str(tmp_path / "nano.root"), with_lhe=True)
    h = {k: np.asarray(v) for k, v in out["hist"].items()}
    st = out["stats"]
    assert st["events"] == 4 and st["sumw_all"] == pytest.approx(5.0) and st["mz_lt_50"] == 1
    assert st["runs_sumw"] == pytest.approx(5.0)
    assert h["sumw"].shape == (3, 28) and h["scale"].shape == (9, 28) and h["pdf"].shape == (103, 28)
    # pair pT: two leptons at phi 0 and 3 rad -> |p1 + p2| = 2 pt cos(1.5)
    zpt = 2 * 100.0 * math.cos(1.5)
    ib = int(np.searchsorted(s.PT_EDGES, zpt, side="right") - 1)
    assert h["sumw"][0, ib] == pytest.approx(1.0)
    assert h["sumw"][1, ib] == pytest.approx(2.0)
    assert h["sumw"][2, ib] == pytest.approx(1.0)
    assert h["sumw"].sum() == pytest.approx(4.0)
    # LHE variations hold ee + mumu only
    assert h["scale"][4].sum() == pytest.approx(3.0)
    assert h["scale"][8, ib] == pytest.approx(3.0 * 1.4)
    assert h["pdf"][102, ib] == pytest.approx(3.0 * 1.1)
    # nominal-only fill (an LO sample) has no variation arrays
    assert set(s.fill_file(str(tmp_path / "nano.root"), with_lhe=False)["hist"]) == {"sumw", "sumw2", "entries"}


# ---------------------------------------------------------------------------
# The ratio
# ---------------------------------------------------------------------------

def _row(s, role, xsec, sumw, ee_bins):
    """A cached dataset whose ee events fill the first pT bins with *ee_bins* (summed genWeight)."""
    nb = len(s.PT_EDGES) - 1
    h = np.zeros((3, nb))
    h[0, :len(ee_bins)] = ee_bins
    hist = {"sumw": h, "sumw2": h.copy(), "entries": h.copy()}
    if role == "nlo":
        hist["scale"] = np.repeat(h[0][None, :], 9, axis=0) * np.array([1.0] * 8 + [1.4])[:, None]   # index 8: x1.4
        pdf = np.repeat(h[0][None, :], 103, axis=0)
        pdf[101] *= 0.9
        pdf[102] *= 1.1
        hist["pdf"] = pdf
    return {"role": role, "sample": f"{role}_sample", "xsec": xsec, "stats": {"sumw_all": sumw, "events": 1},
            "files": [{"lfn": f"/{role}.root"}], "hist": hist}


def test_merge_eras_adds_the_same_sample_and_keeps_distinct_ones(s):
    a = _row(s, "lo", 10.0, 10.0, [8.0, 2.0])                 # era A: per pb 8, 2
    b = _row(s, "lo", 10.0, 30.0, [12.0, 18.0])               # era B, same sample: 3x the events
    nlo_b = _row(s, "nlo", 10.0, 20.0, [12.0, 8.0])
    nlo_b["sample"] = "nlo_other"
    rows = s.merge_eras({"A": [a], "B": [b, nlo_b]})
    lo = [r for r in rows if r["role"] == "lo"]
    assert len(lo) == 1 and lo[0]["eras"] == ["A", "B"] and len(lo[0]["files"]) == 2
    assert lo[0]["stats"]["sumw_all"] == 40.0 and lo[0]["stats"]["events"] == 2
    assert s.spectrum(lo)[:2] == pytest.approx([5.0, 5.0])    # (8+12, 2+18) x 10 pb / 40, not the mean of per-pb spectra
    assert [r["sample"] for r in rows if r["role"] == "nlo"] == ["nlo_other"]
    c = _row(s, "lo", 11.0, 10.0, [1.0])
    with pytest.raises(ValueError, match="cross section differs"):
        s.merge_eras({"A": [a], "C": [c]})


def test_qcd_block_is_the_per_pb_ratio_with_unrenormalized_variations(s):
    lo = _row(s, "lo", 10.0, 10.0, [8.0, 2.0])         # per pb: 8, 2
    nlo = _row(s, "nlo", 10.0, 20.0, [12.0, 8.0])      # per pb: 6, 4
    qcd, checks = s.qcd_block([lo], [nlo])
    assert qcd["central"][:2] == pytest.approx([0.75, 2.0])
    assert checks["integrated"]["nlo_over_lo"] == pytest.approx(1.0)
    assert qcd["scale_up"][:2] == pytest.approx([0.75 * 1.4, 2.0 * 1.4])
    assert qcd["scale_down"][:2] == pytest.approx([0.75, 2.0])
    assert qcd["alphas_up"][:2] == pytest.approx([0.75 * 1.1, 2.0 * 1.1])
    assert qcd["alphas_down"][:2] == pytest.approx([0.75 * 0.9, 2.0 * 0.9])
    assert qcd["pdf_err"][:2] == pytest.approx([0.0, 0.0])
    assert qcd["central"][2:] == pytest.approx([2.0] * 26)      # empty bins take the nearest filled one
    assert checks["per_flavour_ratio"]["ee"][:2] == pytest.approx([0.75, 2.0])
    assert math.isnan(checks["per_flavour_ratio"]["mumu"][0])
    assert checks["entries"]["lo"][:2] == [8.0, 2.0]


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
