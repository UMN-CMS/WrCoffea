"""Tests for wrcoffea.muon_momentum (MuonScaRe below 200 GeV, GE + MC smearing above).

These tests write a synthetic payload with made-up numbers and the real
layout; test_configured_payload_loads checks the payloads config.yaml names.
"""

import json

import awkward as ak
import numpy as np
import pytest

import wrcoffea.muon_momentum as mm
import wrcoffea.scale_factors as sf
from wrcoffea.MuonScaRe import pt_scale

ERA = "RunIII2024Summer24"


@pytest.fixture
def payload(tmp_path, monkeypatch):
    kappa = np.zeros((2, 6))
    kappa[1, 3] = -0.1            # eta 0..2.4, phi 0..pi/3 (made-up value)
    cfg = {
        "threshold_gev": 200.0,
        "ge_kappa": {"units": "1/TeV", "eta_edges": [-2.4, 0.0, 2.4],
                     "phi_edges": list(np.linspace(-np.pi, np.pi, 7)),
                     "kappa": kappa.tolist(), "kappa_unc": np.full((2, 6), 0.01).tolist()},
        "mc_smearing": {"eta_split": 1.2, "sigma_barrel": [0.01, 2e-5, 0.0, 0.0],
                        "sigma_endcap": [0.02, 4e-5, 0.0, 0.0], "sigma_p_max_gev": 2500.0,
                        "extra_smear": 0.2, "syst": 0.05},
    }
    path = tmp_path / "highpt.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setitem(mm.MUON_HIGHPT, ERA, str(path))
    mm.highpt_config.cache_clear()
    yield cfg
    mm.highpt_config.cache_clear()


@pytest.mark.parametrize("era", sorted(mm.MUON_HIGHPT))
def test_configured_payload_loads(era):
    """Each payload config.yaml names is in the repository and passes the loader's checks."""
    mm.highpt_config.cache_clear()
    try:
        cfg = mm.highpt_config(era)
    finally:
        mm.highpt_config.cache_clear()
    assert cfg["era"] == era
    assert cfg["threshold_gev"] > 0
    assert len(cfg["mc_smearing"]["sigma_barrel"]) == len(cfg["mc_smearing"]["sigma_endcap"])


def test_smear_factor_is_sqrt_of_one_plus_x_squared_minus_one():
    assert mm.smear_factor(0.05) == pytest.approx(0.32, abs=0.005)
    assert mm.smear_factor(0.10) == pytest.approx(0.46, abs=0.005)
    assert mm.smear_factor(0.15) == pytest.approx(0.568, abs=0.001)
    assert mm.smear_factor(0.0) == 0.0


def test_ge_correction_sign_convention():
    """kappa is the data bias, q/pT_meas = q/pT_true + kappa: data gets q/pT - kappa.
    With kappa = -0.1/TeV at 1 TeV a mu+ goes down to 909 GeV and a mu- up to 1111 GeV."""
    out = mm.shift_curvature(np.array([1000.0, 1000.0]), np.array([1, -1]), np.array([0.1, 0.1]))
    np.testing.assert_allclose(out, [1000 / 1.1, 1000 / 0.9])


def test_curvature_flip_leaves_the_muon_unchanged():
    out = mm.shift_curvature(np.array([10000.0]), np.array([1]), np.array([-0.2]))
    assert out[0] == 10000.0


def test_kappa_lookup_and_resolution_freeze(payload):
    k, u = mm.kappa_lookup(np.array([0.5, -0.5, 2.39]), np.array([0.5, 0.5, 0.5]), payload)
    np.testing.assert_allclose(k, [-0.1, 0.0, -0.1])
    np.testing.assert_allclose(u, 0.01)
    s = mm.resolution(np.array([1000.0, 2500.0, 5000.0]), np.array([0.5, 0.5, 2.0]), payload)
    np.testing.assert_allclose(s, [0.01 + 0.02, 0.01 + 0.05, 0.02 + 0.1])   # 5 TeV frozen at 2.5


def _events(pts, etas, phis, charges, event=7):
    n = len(pts)
    muon = ak.zip({
        "pt": [pts], "tunepRelPt": [[1.0] * n], "eta": [etas], "phi": [phis],
        "charge": [charges], "nTrackerLayers": [[12] * n],
    })
    return ak.zip({"Muon": muon, "event": np.array([event], dtype=np.uint64),
                   "luminosityBlock": np.array([3], dtype=np.uint32)}, depth_limit=1)


def test_data_low_pt_uses_muonscare_and_high_pt_uses_ge(payload):
    ev = _events([80.0, 1000.0, 1000.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, -1])
    out = mm.correct_muons(ev, ERA, is_mc=False)
    pt = ak.to_numpy(ak.flatten(out["pt"]))
    cset = mm._scare_cset(ERA)
    expected_low = ak.to_numpy(ak.flatten(pt_scale(1, ev.Muon.pt, ev.Muon.eta, ev.Muon.phi, ev.Muon.charge,
                                                   cset, nested=True)))[0]
    assert pt[0] == pytest.approx(expected_low)
    assert pt[0] != 80.0
    np.testing.assert_allclose(pt[1:], [1000 / 1.1, 1000 / 0.9])
    assert set(out) == {"pt"}


def test_mc_high_pt_smearing_is_deterministic_and_ordered(payload):
    ev = _events([500.0, 1500.0, 90.0], [0.3, 1.8, 0.3], [0.2, -2.0, 1.0], [1, -1, 1])
    a = mm.correct_muons(ev, ERA, is_mc=True)
    b = mm.correct_muons(ev, ERA, is_mc=True)
    for key in ("pt", "pt_scale_up", "pt_scale_down", "pt_smear_up", "pt_smear_down"):
        np.testing.assert_array_equal(ak.to_numpy(ak.flatten(a[key])), ak.to_numpy(ak.flatten(b[key])))
    pt0 = np.array([500.0, 1500.0])
    nom = ak.to_numpy(ak.flatten(a["pt"]))[:2]
    up = ak.to_numpy(ak.flatten(a["pt_smear_up"]))[:2]
    down = ak.to_numpy(ak.flatten(a["pt_smear_down"]))[:2]
    # same random number: the shift grows with the smearing fraction (x - syst < x < x + syst)
    assert np.all(np.abs(down - pt0) < np.abs(nom - pt0))
    assert np.all(np.abs(nom - pt0) < np.abs(up - pt0))
    np.testing.assert_allclose((up - pt0) / (nom - pt0), mm.smear_factor(0.25) / mm.smear_factor(0.2))
    # MC gets no kappa; its scale variation is the kappa uncertainty (+-0.01/TeV) on q/pT
    sc_up = ak.to_numpy(ak.flatten(a["pt_scale_up"]))[:2]
    np.testing.assert_allclose(sc_up, mm.shift_curvature(nom, np.array([1, -1]), np.array([-0.01, -0.01])))


def test_mc_smearing_width(payload):
    """Over many events the relative shift has width sigma(p) * f(x)."""
    n = 20000
    muon = ak.zip({"pt": np.full((n, 1), 1000.0), "tunepRelPt": np.ones((n, 1)), "eta": np.zeros((n, 1)),
                   "phi": np.full((n, 1), 0.5), "charge": np.ones((n, 1), dtype=int),
                   "nTrackerLayers": np.full((n, 1), 12)})
    ev = ak.zip({"Muon": muon, "event": np.arange(n, dtype=np.uint64),
                 "luminosityBlock": np.ones(n, dtype=np.uint32)}, depth_limit=1)
    rel = ak.to_numpy(ak.flatten(mm.correct_muons(ev, ERA, is_mc=True)["pt"])) / 1000.0 - 1
    assert rel.std() == pytest.approx(0.03 * mm.smear_factor(0.2), rel=0.03)
    assert abs(rel.mean()) < 0.001


def test_unlisted_era_keeps_the_tunep_placeholder(monkeypatch):
    monkeypatch.setattr(sf, "MUON_HIGHPT", {})
    ev = _events([1000.0], [0.5], [0.5], [1])
    out = sf.apply_muon_scale_smearing(ev, ERA, True)
    assert ak.to_numpy(ak.flatten(out["pt"]))[0] == 1000.0
    assert ak.to_numpy(ak.flatten(out["pt_scale_up"]))[0] == pytest.approx(1050.0)


def test_listed_era_dispatches_to_the_new_recipe(payload, monkeypatch):
    monkeypatch.setattr(sf, "MUON_HIGHPT", {ERA: "x"})
    ev = _events([1000.0], [0.5], [0.5], [1])
    out = sf.apply_muon_scale_smearing(ev, ERA, False)
    assert ak.to_numpy(ak.flatten(out["pt"]))[0] == pytest.approx(1000 / 1.1)


# ---------------------------------------------------------------------------
# 2022/2023 payloads (Muon POG Run 3 tables) and barrel/endcap smearing fractions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("era,eta,phi,kappa,unc", [
    ("Run3Summer22", -2.3, -2.0, -0.16, 0.10),      # POG table: row -180..-60 deg, column -2.4..-2.1
    ("Run3Summer22EE", 2.2, 0.0, -0.48, 0.05),      # row -60..60, column 2.1..2.4
    ("Run3Summer23", 0.5, -0.5, 0.030, 0.027),      # row -60..60, column 0..1.2
    ("Run3Summer23BPix", 2.2, 2.0, -0.20, 0.07),    # row 60..180, column 2.1..2.4
])
def test_2022_2023_kappa_maps_follow_the_pog_tables(era, eta, phi, kappa, unc):
    """The POG tables have phi rows and eta columns; the payloads store kappa[eta][phi]."""
    mm.highpt_config.cache_clear()
    try:
        cfg = mm.highpt_config(era)
    finally:
        mm.highpt_config.cache_clear()
    k, u = mm.kappa_lookup(np.array([eta]), np.array([phi]), cfg)
    assert k[0] == pytest.approx(kappa) and u[0] == pytest.approx(unc)


@pytest.mark.parametrize("era,barrel,endcap", [
    ("Run3Summer22", 0.0, 0.05), ("Run3Summer22EE", 0.05, 0.10),
    ("Run3Summer23", 0.05, 0.10), ("Run3Summer23BPix", 0.0, 0.15),
])
def test_2022_2023_smearing_fractions(era, barrel, endcap):
    mm.highpt_config.cache_clear()
    try:
        m = mm.highpt_config(era)["mc_smearing"]
    finally:
        mm.highpt_config.cache_clear()
    assert m["extra_smear"] == {"barrel": barrel, "endcap": endcap}
    assert m["syst"] == 0.1


@pytest.fixture
def region_payload(payload, tmp_path, monkeypatch):
    """The synthetic payload with a smearing fraction per region (0 in the barrel), as in 2022 and 2023BPix."""
    cfg = json.loads(json.dumps(payload))
    cfg["mc_smearing"]["extra_smear"] = {"barrel": 0.0, "endcap": 0.15}
    cfg["mc_smearing"]["syst"] = 0.1
    path = tmp_path / "highpt_regions.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setitem(mm.MUON_HIGHPT, ERA, str(path))
    mm.highpt_config.cache_clear()
    yield cfg
    mm.highpt_config.cache_clear()


def test_smearing_fraction_per_region(region_payload):
    """A barrel fraction of 0 leaves the barrel MC unsmeared (MuSmear down too) while MuSmear up smears it by 10 %;
    the endcap uses its own fraction."""
    ev = _events([500.0, 500.0], [0.3, 1.8], [0.2, 0.2], [1, -1])
    out = mm.correct_muons(ev, ERA, is_mc=True)
    nom, up, down = (ak.to_numpy(ak.flatten(out[k])) for k in ("pt", "pt_smear_up", "pt_smear_down"))
    assert nom[0] == 500.0 and down[0] == 500.0 and up[0] != 500.0
    assert nom[1] != 500.0
    np.testing.assert_allclose((up[1] - 500.0) / (nom[1] - 500.0), mm.smear_factor(0.25) / mm.smear_factor(0.15))
    np.testing.assert_allclose((down[1] - 500.0) / (nom[1] - 500.0), mm.smear_factor(0.05) / mm.smear_factor(0.15))
