"""Muon momentum corrections for the high-pT ID muons (TuneP), Muon POG Run 3 recipe.

Eras listed in ``config.yaml`` ``muon_highpt`` get, per muon, with p_T the TuneP
transverse momentum (``Muon_pt * Muon_tunepRelPt``) before any correction:

- p_T < threshold (200 GeV), medium-pT regime: the MuonScaRe scale (data and
  MC) and resolution smearing (MC) of the era's ``ScaleNSmear`` payload, with
  its statistical scale and resolution variations.  Below 200 GeV TuneP takes
  the tracker-only fit for nearly all high-pT-ID muons and equals the PF
  p_T the kit is calibrated on (checked on 2024 DY MC and 2024G data).
- p_T >= threshold, high-pT regime:
  - data: the generalized-endpoint (GE) bias is removed,
    q/p_T -> q/p_T - kappa(eta, phi), kappa in 1/TeV;
  - MC: p -> p (1 + N(0, sigma(p, |eta|) f)), f = sqrt((1 + x)^2 - 1) with the
    era's extra smearing x; ``MuSmear`` up/down use x +- syst with the SAME
    random number, ``MuScale`` up/down shift q/p_T by -+ the kappa uncertainty
    (the uncertainty of the data correction, carried by MC).

The numbers (kappa map, sigma(p) coefficients, x) come from CMS-internal
documents and live in the era's JSON payload (``data/muon_highpt/``, sources
in docs/muon_momentum.md); this module holds only the formulas.  Eras not
listed keep the old TuneP placeholder in
``scale_factors.apply_muon_scale_smearing``.

The random numbers are deterministic: correctionlib's counter-based hashprng of
(event, luminosityBlock, phi), so reruns and shape-variation passes see the same
smearing.  The high-pT draw uses a distinct salt from MuonScaRe's.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from functools import lru_cache
from pathlib import Path

import awkward as ak
import numpy as np

from wrcoffea.analysis_config import MUON_HIGHPT, MUON_JSONS, REPO_ROOT
from wrcoffea.MuonScaRe import pt_resol, pt_resol_var, pt_scale, pt_scale_var

logger = logging.getLogger(__name__)

_WARN_ONCE: set[str] = set()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@lru_cache(maxsize=None)
def highpt_config(era: str) -> dict:
    """The era's high-pT payload (GE kappa map, MC smearing)."""
    path = MUON_HIGHPT.get(era)
    if not path:
        raise KeyError(f"No muon_highpt payload configured for era '{era}'")
    p = _resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"muon_highpt payload {p} is missing (config.yaml muon_highpt, docs/muon_momentum.md).")
    with open(p) as f:
        cfg = json.load(f)
    ge = cfg["ge_kappa"]
    kappa = np.asarray(ge["kappa"], dtype=np.float64)
    unc = np.asarray(ge["kappa_unc"], dtype=np.float64)
    n_eta, n_phi = len(ge["eta_edges"]) - 1, len(ge["phi_edges"]) - 1
    if kappa.shape != (n_eta, n_phi) or unc.shape != (n_eta, n_phi):
        raise ValueError(f"{p}: kappa map shape {kappa.shape} does not match the edges ({n_eta}, {n_phi})")
    if ge.get("units") != "1/TeV":
        raise ValueError(f"{p}: kappa units must be 1/TeV, got {ge.get('units')!r}")
    return cfg


@lru_cache(maxsize=None)
def _scare_cset(era: str):
    import correctionlib
    return correctionlib.CorrectionSet.from_file(str(_resolve(MUON_JSONS[era]["ScaleNSmear"])))


@lru_cache(maxsize=None)
def _normal_rng():
    """Deterministic standard-normal draws of (event, lumi, phi, salt)."""
    from correctionlib import schemav2 as cs
    corr = cs.Correction(
        name="highpt_smear_rng", version=1,
        inputs=[cs.Variable(name="evt", type="int"), cs.Variable(name="lumi", type="int"),
                cs.Variable(name="phi", type="real"), cs.Variable(name="salt", type="int")],
        output=cs.Variable(name="r", type="real"),
        data=cs.HashPRNG(nodetype="hashprng", inputs=["evt", "lumi", "phi", "salt"],
                         distribution="stdnormal"),
    )
    return cs.CorrectionSet(schema_version=2, corrections=[corr]).to_evaluator()["highpt_smear_rng"]


# Salt of the high-pT smearing draw (MuonScaRe's own draw has no salt input).
HIGHPT_SALT = 2024


def smear_factor(x: float) -> float:
    """Width multiplier f for an extra smearing fraction x: f = sqrt((1 + x)^2 - 1)."""
    if x <= 0:
        return 0.0
    return float(np.sqrt((1.0 + x) ** 2 - 1.0))


def resolution(p, abseta, cfg: dict) -> np.ndarray:
    """sigma(p, |eta|) of the high-pT smearing (cubic in p, frozen above sigma_p_max_gev)."""
    m = cfg["mc_smearing"]
    p = np.minimum(np.asarray(p, dtype=np.float64), float(m.get("sigma_p_max_gev", np.inf)))
    barrel = np.asarray(abseta) < float(m["eta_split"])
    out = np.empty_like(p)
    for mask, coeffs in ((barrel, m["sigma_barrel"]), (~barrel, m["sigma_endcap"])):
        c = np.asarray(coeffs, dtype=np.float64)
        pp = p[mask]
        out[mask] = c[0] + c[1] * pp + c[2] * pp ** 2 + c[3] * pp ** 3
    return np.maximum(out, 0.0)


def kappa_lookup(eta, phi, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(kappa, kappa_unc) in 1/TeV for each (eta, phi); edges clamp (|eta| <= 2.4 by selection)."""
    ge = cfg["ge_kappa"]
    eta_edges = np.asarray(ge["eta_edges"], dtype=np.float64)
    phi_edges = np.asarray(ge["phi_edges"], dtype=np.float64)
    ie = np.clip(np.searchsorted(eta_edges, eta, side="right") - 1, 0, len(eta_edges) - 2)
    ip = np.clip(np.searchsorted(phi_edges, phi, side="right") - 1, 0, len(phi_edges) - 2)
    return (np.asarray(ge["kappa"], dtype=np.float64)[ie, ip],
            np.asarray(ge["kappa_unc"], dtype=np.float64)[ie, ip])


def shift_curvature(pt, charge, shift_per_tev) -> np.ndarray:
    """p_T after q/p_T -> q/p_T + shift (shift in 1/TeV).

    A shift that would flip the sign of the curvature (p_T above ~1/|kappa|)
    leaves that muon unchanged.
    """
    pt = np.asarray(pt, dtype=np.float64)
    c = np.asarray(charge, dtype=np.float64) * 1000.0 / pt          # q/pT in 1/TeV
    c_new = c + np.asarray(shift_per_tev, dtype=np.float64)
    flip = (np.sign(c_new) != np.sign(c)) | (np.abs(c_new) < 1e-9)
    if np.any(flip):
        key = "ge_flip"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning("GE curvature shift would flip the charge of %d muon(s) (p_T up to %.0f GeV); "
                           "their p_T is left unchanged.", int(flip.sum()), float(pt[flip].max()))
    return np.where(flip, pt, 1000.0 / np.abs(np.where(flip, 1.0, c_new)))


def _medium_pt(events, muons, pt0, charge, cset, is_mc):
    """MuonScaRe on the TuneP p_T: (corrected p_T, {variation field: p_T}) (variations MC only)."""
    eta, phi = muons.eta, muons.phi
    if not is_mc:
        return pt_scale(1, pt0, eta, phi, charge, cset, nested=True), {}
    pt_sc = pt_scale(0, pt0, eta, phi, charge, cset, nested=True)
    event = ak.values_astype(events.event, np.int64)
    lumi = ak.values_astype(events.luminosityBlock, np.int64)
    low = pt_resol(pt_sc, eta, phi, muons.nTrackerLayers, event, lumi, cset, nested=True)
    return low, {
        "pt_scale_up": pt_scale_var(low, eta, phi, charge, "up", cset, nested=True),
        "pt_scale_down": pt_scale_var(low, eta, phi, charge, "dn", cset, nested=True),
        "pt_smear_up": pt_resol_var(pt_sc, low, eta, "up", cset, nested=True),
        "pt_smear_down": pt_resol_var(pt_sc, low, eta, "dn", cset, nested=True),
    }


def correct_muons(events, era: str, is_mc: bool) -> dict:
    """Corrected TuneP p_T (and MC variations) of ``events.Muon``.

    Returns ``{"pt": ...}`` for data and, for MC, also ``pt_scale_up/down`` and
    ``pt_smear_up/down`` (the fields the ``muscale`` / ``musmear`` shape
    systematics read), all jagged like ``events.Muon``.
    """
    muons = events.Muon
    counts = ak.num(muons)
    if ak.sum(counts) == 0:
        return {}
    cfg = highpt_config(era)
    threshold = float(cfg["threshold_gev"])
    cset = _scare_cset(era)

    pt0 = muons.pt * muons.tunepRelPt
    # TuneP charge where NanoAOD has it (not in every 2024 data stream).
    charge = muons.tuneP_charge if "tuneP_charge" in muons.fields else muons.charge
    eta, phi = muons.eta, muons.phi

    # Medium pT: MuonScaRe (it leaves p_T outside [26, 200] GeV unchanged by itself,
    # and prints a line per call saying so: silenced).
    with contextlib.redirect_stdout(io.StringIO()):
        low, low_vars = _medium_pt(events, muons, pt0, charge, cset, is_mc)

    # High pT: flat arrays over all muons, then unflattened.
    f_pt0 = np.asarray(ak.flatten(pt0), dtype=np.float64)
    f_q = np.asarray(ak.flatten(charge), dtype=np.float64)
    f_eta = np.asarray(ak.flatten(eta), dtype=np.float64)
    f_phi = np.asarray(ak.flatten(phi), dtype=np.float64)
    kappa, kappa_unc = kappa_lookup(f_eta, f_phi, cfg)
    if is_mc:
        m = cfg["mc_smearing"]
        x_nom, x_syst = float(m["extra_smear"]), float(m["syst"])
        evt = np.repeat(np.asarray(events.event, dtype=np.int64), np.asarray(counts))
        lumi = np.repeat(np.asarray(events.luminosityBlock, dtype=np.int64), np.asarray(counts))
        r = _normal_rng().evaluate(evt, lumi, f_phi, np.full(len(f_phi), HIGHPT_SALT, dtype=np.int64))
        sigma = resolution(f_pt0 * np.cosh(f_eta), np.abs(f_eta), cfg)
        high = {
            "pt": f_pt0 * (1.0 + r * sigma * smear_factor(x_nom)),
            "pt_smear_up": f_pt0 * (1.0 + r * sigma * smear_factor(x_nom + x_syst)),
            "pt_smear_down": f_pt0 * (1.0 + r * sigma * smear_factor(max(x_nom - x_syst, 0.0))),
        }
        # The data correction's uncertainty, carried by MC: kappa +- unc in data
        # is equivalent to q/pT -/+ unc in MC.
        high["pt_scale_up"] = shift_curvature(high["pt"], f_q, -kappa_unc)
        high["pt_scale_down"] = shift_curvature(high["pt"], f_q, kappa_unc)
        # Smearing can push a few muons to p_T <= 0 in a far tail: keep them unsmeared.
        for key in list(high):
            high[key] = np.where(high[key] > 0, high[key], f_pt0)
    else:
        high = {"pt": shift_curvature(f_pt0, f_q, -kappa)}

    is_high = f_pt0 >= threshold
    out = {}
    for key in ["pt", *(("pt_scale_up", "pt_scale_down", "pt_smear_up", "pt_smear_down") if is_mc else ())]:
        f_low = np.asarray(ak.flatten(low if key == "pt" else low_vars[key]), dtype=np.float64)
        out[key] = ak.unflatten(np.where(is_high, high[key], f_low), counts)
    return out
