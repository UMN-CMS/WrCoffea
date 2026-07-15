"""Histogram specification, creation, and filling for the WR analysis.

Canonical naming choice:
  - Histogram key in the output dict == numeric axis name == ROOT stem

Each spec is: (name, bins, label, getter)
  - For resolved: getter(tight_leptons, ak4_jets) -> values
  - For boosted:  getter(tight_lepton, ak8_jet, loose_lepton) -> values
"""

import logging
from typing import Callable

import awkward as ak
import hist
import numpy as np

from wrcoffea.analysis_config import (
    CUTS,
    SEL_MIN_TWO_AK4_JETS_PTETA, SEL_MIN_TWO_AK4_JETS_ID,
    SEL_TWO_PTETA_ELECTRONS, SEL_TWO_PTETA_MUONS, SEL_TWO_PTETA_EM,
    SEL_TWO_ID_ELECTRONS, SEL_TWO_ID_MUONS, SEL_TWO_ID_EM,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_MLL_GT200, SEL_MLLJJ_GT800, SEL_MLL_GT400,
    #SEL_JET_VETO_MAP,
    # Boosted selections
    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_AK8JETS_WITH_LSF,
    SEL_MUMU_SR, SEL_EE_SR, SEL_EMU_CR,
    SEL_LEAD_IS_ELECTRON, SEL_LEAD_IS_MUON,
    SEL_NO_DY_PAIR, SEL_NO_EXTRA_TIGHT_SR, SEL_NO_EXTRA_TIGHT_CR,
    SEL_SF_LEPTON_IN_AK8, SEL_NO_OF_LEPTON_IN_AK8,
    SEL_OF_LEPTON_IN_AK8, SEL_NO_SF_LEPTON_IN_AK8,
    SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED,
)

logger = logging.getLogger(__name__)
 

ResolvedGetter = Callable[[ak.Array, ak.Array, ak.Array, ak.Array], ak.Array]
BoostedGetter = Callable[[ak.Array, ak.Array, ak.Array], ak.Array]

# Resolved getters receive (L, J, LL, JJ) where LL = dilepton 4-vec, JJ = dijet 4-vec
# (pre-computed once per chunk in fill_resolved_histograms to avoid redundant additions).
RESOLVED_HIST_SPECS: list[tuple[str, tuple[int, float, float], str, ResolvedGetter]] = [
    ("pt_leading_lepton",           (200,   0, 2000), r"$p_{T}$ of the leading lepton [GeV]",           lambda L, J, LL, JJ: L[:, 0].pt),
    ("eta_leading_lepton",          (60,   -3,    3), r"$\\eta$ of the leading lepton",                lambda L, J, LL, JJ: L[:, 0].eta),
    ("phi_leading_lepton",          (80,   -4,    4), r"$\\phi$ of the leading lepton",                lambda L, J, LL, JJ: L[:, 0].phi),
    ("pt_subleading_lepton",        (200,   0, 2000), r"$p_{T}$ of the subleading lepton [GeV]",        lambda L, J, LL, JJ: L[:, 1].pt),
    ("eta_subleading_lepton",       (60,   -3,    3), r"$\\eta$ of the subleading lepton",             lambda L, J, LL, JJ: L[:, 1].eta),
    ("phi_subleading_lepton",       (80,   -4,    4), r"$\\phi$ of the subleading lepton",             lambda L, J, LL, JJ: L[:, 1].phi),
    ("pt_leading_jet",              (200,   0, 2000), r"$p_{T}$ of the leading jet [GeV]",              lambda L, J, LL, JJ: J[:, 0].pt),
    ("eta_leading_jet",             (60,   -3,    3), r"$\\eta$ of the leading jet",                   lambda L, J, LL, JJ: J[:, 0].eta),
    ("phi_leading_jet",             (80,   -4,    4), r"$\\phi$ of the leading jet",                   lambda L, J, LL, JJ: J[:, 0].phi),
    ("pt_subleading_jet",           (200,   0, 2000), r"$p_{T}$ of the subleading jet [GeV]",           lambda L, J, LL, JJ: J[:, 1].pt),
    ("eta_subleading_jet",          (60,   -3,    3), r"$\\eta$ of the subleading jet",                lambda L, J, LL, JJ: J[:, 1].eta),
    ("phi_subleading_jet",          (80,   -4,    4), r"$\\phi$ of the subleading jet",                lambda L, J, LL, JJ: J[:, 1].phi),
    ("mass_dilepton",               (5000,  0, 5000), r"$m_{\\ell\\ell}$ [GeV]",                     lambda L, J, LL, JJ: LL.mass),
    ("pt_dilepton",                 (200,   0, 2000), r"$p_{T,\\ell\\ell}$ [GeV]",                   lambda L, J, LL, JJ: LL.pt),
    ("mass_dijet",                  (500,   0, 5000), r"$m_{jj}$ [GeV]",                               lambda L, J, LL, JJ: JJ.mass),
    ("pt_dijet",                    (500,   0, 5000), r"$p_{T,jj}$ [GeV]",                             lambda L, J, LL, JJ: JJ.pt),
    ("mass_threeobject_leadlep",    (800,   0, 8000), r"$m_{\\ell jj}$ [GeV]",                        lambda L, J, LL, JJ: (L[:, 0] + JJ).mass),
    ("pt_threeobject_leadlep",      (800,   0, 8000), r"$p_{T,\\ell jj}$ [GeV]",                      lambda L, J, LL, JJ: (L[:, 0] + JJ).pt),
    ("mass_threeobject_subleadlep", (800,   0, 8000), r"$m_{\\ell jj}$ [GeV]",                        lambda L, J, LL, JJ: (L[:, 1] + JJ).mass),
    ("pt_threeobject_subleadlep",   (800,   0, 8000), r"$p_{T,\\ell jj}$ [GeV]",                      lambda L, J, LL, JJ: (L[:, 1] + JJ).pt),
    ("mass_fourobject",             (800,   0, 8000), r"$m_{\\ell\\ell jj}$ [GeV]",                 lambda L, J, LL, JJ: (LL + JJ).mass),
    ("pt_fourobject",               (800,   0, 8000), r"$p_{T,\\ell\\ell jj}$ [GeV]",               lambda L, J, LL, JJ: (LL + JJ).pt),
]


smallneta = [
    (0.996101364522417, -2.07703281027104),
    (1.69785575048733, -1.97432239657632),
    (1.99805068226121, -1.83737517831669),
    (2.30214424951267, -1.64621968616262),
    (2.69980506822612, -1.38944365192582),
    (3.00389863547758, -1.28958630527817),
    (3.30409356725146, -1.20970042796006),
    (3.47953216374269, -1.16690442225392),
    (3.70175438596491, -1.07275320970043),
]

largeneta = [
    (1.00334448160535, -1.77714285714286),
    (1.70234113712375, -1.66),
    (2.00334448160535, -1.54285714285714),
    (2.30434782608696, -1.35714285714286),
    (2.7056856187291, -1.09142857142857),
    (3.00334448160535, -0.937142857142857),
    (3.30434782608696, -0.791428571428571),
]

def _interpolate_muon_resolution(pT, points):
    pT_values, res_values = zip(*points)
    pT_values = np.array(pT_values, dtype=float)
    res_values = np.array(res_values, dtype=float)

    pT_array = np.asarray(pT, dtype=float)
    pT_flat = pT_array.ravel()
    out = np.empty_like(pT_flat, dtype=float)

    for idx, pT_value in enumerate(pT_flat):
        if pT_value <= pT_values[0]:
            slope = (res_values[1] - res_values[0]) / (pT_values[1] - pT_values[0])
            out[idx] = res_values[0] + slope * (pT_value - pT_values[0])
        elif pT_value >= pT_values[-1]:
            slope = (res_values[-1] - res_values[-2]) / (pT_values[-1] - pT_values[-2])
            out[idx] = res_values[-1] + slope * (pT_value - pT_values[-1])
        else:
            out[idx] = np.interp(pT_value, pT_values, res_values)

    out = out.reshape(pT_array.shape)
    if pT_array.ndim == 0:
        return float(out[()])
    return out


def getsigma(pT, eta):
    pT_array = np.asarray(pT, dtype=float)
    pT_safe = np.where(pT_array > 0, pT_array, 1e-6)
    x = np.log10(pT_safe)

    eta_array = np.asarray(eta, dtype=float)
    if eta_array.ndim == 0:
        if abs(float(eta_array)) < 1:
            return np.power(10, _interpolate_muon_resolution(x, smallneta))
        return np.power(10, _interpolate_muon_resolution(x, largeneta))

    result = np.empty_like(pT_safe, dtype=float)
    small_mask = np.abs(eta_array) < 1.0
    result[small_mask] = np.power(10, _interpolate_muon_resolution(x[small_mask], smallneta))
    result[~small_mask] = np.power(10, _interpolate_muon_resolution(x[~small_mask], largeneta))
    return result


def dotprod_T(p1, p2):
    return p1.x * p2.x + p1.y * p2.y


def compute_s1_s2_modchi(l1, l2, j1, j2, x=0.5):

    # ========================================================
    # TRANSVERSE MOMENTA
    # ========================================================

    pT1 = np.sqrt(dotprod_T(l1, l1))
    pT2 = np.sqrt(dotprod_T(l2, l2))
    pT3 = np.sqrt(dotprod_T(j1, j1))
    pT4 = np.sqrt(dotprod_T(j2, j2))

    pT1 = np.atleast_1d(np.asarray(pT1, dtype=float)).astype(float)
    pT2 = np.atleast_1d(np.asarray(pT2, dtype=float)).astype(float)
    pT3 = np.atleast_1d(np.asarray(pT3, dtype=float)).astype(float)
    pT4 = np.atleast_1d(np.asarray(pT4, dtype=float)).astype(float)

    pTt = np.sqrt(dotprod_T(l1 + l2 + j1 + j2, l1 + l2 + j1 + j2))
    pTt = np.atleast_1d(np.asarray(pTt, dtype=float)).astype(float)
    pTt = np.where(pTt > 1e-12, pTt, 1e-12)

    # ========================================================
    # JET RESOLUTION
    # ========================================================

    S = 0.92
    C = 0.04

    sig1 = pT1 * getsigma(pT1, np.asarray(l1.eta, dtype=float).astype(float))
    sig2 = pT2 * getsigma(pT2, np.asarray(l2.eta, dtype=float).astype(float))
    sig3 = pT3 * np.sqrt(S * S / pT3 + C * C)
    sig4 = pT4 * np.sqrt(S * S / pT4 + C * C)

    sigt = np.sqrt(
        np.asarray(
            dotprod_T(
                (l1 + l2 + j1 + j2) / pTt,
                sig1 * l1 / pT1 + sig2 * l2 / pT2 + sig3 * j1 / pT3 + sig4 * j2 / pT4,
            ),
            dtype=float,
        )
    )
    sigt = np.atleast_1d(np.asarray(sigt, dtype=float)).astype(float)

    # ========================================================
    # MATRIX COEFFICIENTS
    # ========================================================

    n_events = max(pT1.shape[0], pT2.shape[0], pT3.shape[0], pT4.shape[0])
    matrix = np.zeros((n_events, 4, 4), dtype=float)
    rhs = np.zeros((n_events, 4), dtype=float)

    matrix[:, 0, 0] = np.asarray(x * pT1**2 / sig1**2 + (1 - x) * dotprod_T(l1, l1) / sigt**2, dtype=float)
    matrix[:, 1, 1] = np.asarray(x * pT2**2 / sig2**2 + (1 - x) * dotprod_T(l2, l2) / sigt**2, dtype=float)
    matrix[:, 2, 2] = np.asarray(x * pT3**2 / sig3**2 + (1 - x) * dotprod_T(j1, j1) / sigt**2, dtype=float)
    matrix[:, 3, 3] = np.asarray(x * pT4**2 / sig4**2 + (1 - x) * dotprod_T(j2, j2) / sigt**2, dtype=float)

    matrix[:, 0, 1] = np.asarray((1 - x) * dotprod_T(l1, l2) / sigt**2, dtype=float)
    matrix[:, 0, 2] = np.asarray((1 - x) * dotprod_T(l1, j1) / sigt**2, dtype=float)
    matrix[:, 0, 3] = np.asarray((1 - x) * dotprod_T(l1, j2) / sigt**2, dtype=float)
    matrix[:, 1, 0] = np.asarray((1 - x) * dotprod_T(l2, l1) / sigt**2, dtype=float)
    matrix[:, 1, 2] = np.asarray((1 - x) * dotprod_T(l2, j1) / sigt**2, dtype=float)
    matrix[:, 1, 3] = np.asarray((1 - x) * dotprod_T(l2, j2) / sigt**2, dtype=float)
    matrix[:, 2, 0] = np.asarray((1 - x) * dotprod_T(j1, l1) / sigt**2, dtype=float)
    matrix[:, 2, 1] = np.asarray((1 - x) * dotprod_T(j1, l2) / sigt**2, dtype=float)
    matrix[:, 2, 3] = np.asarray((1 - x) * dotprod_T(j1, j2) / sigt**2, dtype=float)
    matrix[:, 3, 0] = np.asarray((1 - x) * dotprod_T(j2, l1) / sigt**2, dtype=float)
    matrix[:, 3, 1] = np.asarray((1 - x) * dotprod_T(j2, l2) / sigt**2, dtype=float)
    matrix[:, 3, 2] = np.asarray((1 - x) * dotprod_T(j2, j1) / sigt**2, dtype=float)

    rhs[:, 0] = np.asarray(x * pT1**2 / sig1**2, dtype=float)
    rhs[:, 1] = np.asarray(x * pT2**2 / sig2**2, dtype=float)
    rhs[:, 2] = np.asarray(x * pT3**2 / sig3**2, dtype=float)
    rhs[:, 3] = np.asarray(x * pT4**2 / sig4**2, dtype=float)

    # ========================================================
    # SOLVE FOR s1, s2, s3, s4
    # ========================================================

    solutions = np.empty((matrix.shape[0], 4), dtype=matrix.dtype)

    for i in range(matrix.shape[0]):
        try:
            solutions[i] = np.linalg.solve(matrix[i], rhs[i])
        except np.linalg.LinAlgError:
            solutions[i] = np.linalg.lstsq(matrix[i], rhs[i], rcond=None)[0]

    s1 = solutions[:, 0]
    s2 = solutions[:, 1]
    s3 = solutions[:, 2]
    s4 = solutions[:, 3]

    if n_events == 1:
        return float(s1[0]), float(s2[0]), float(s3[0]), float(s4[0])

    return s1, s2, s3, s4

def _default_muon_scale_factors(l1, l2, jets):
    j1 = jets[:, 0]
    j2 = jets[:, 1]
    s1, s2, s3, s4 = compute_s1_s2_modchi(l1, l2, j1, j2)
    return s1, s2, s3, s4


def _get_muon_and_exact4_mask(L, J):
    l1 = L[:, 0]
    l2 = L[:, 1]
    both_mu = (l1.flavor == "muon") & (l2.flavor == "muon")
    object_count = ak.num(L, axis=1) + ak.num(J, axis=1)
    return both_mu & (object_count == 4)


def _make_muon_scaled_fourobj_getter(scale_factor_fn, attr):
    def getter(L, J, LL, JJ):
        l1 = L[:, 0]
        l2 = L[:, 1]

        both_mu = (l1.flavor == "muon") & (l2.flavor == "muon")
        if not ak.any(both_mu):
            return getattr(LL + JJ, attr)

        l1_sf, l2_sf, j1_sf, j2_sf = scale_factor_fn(l1, l2, J)

        l1_scaled = ak.zip({
            "pt": l1.pt * l1_sf,
            "eta": l1.eta,
            "phi": l1.phi,
            "mass": l1.mass,
            "charge": l1.charge if hasattr(l1, "charge") else (ak.where(l1.pdgId > 0, -1, 1) if "pdgId" in getattr(l1, "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        l2_scaled = ak.zip({
            "pt": l2.pt * l2_sf,
            "eta": l2.eta,
            "phi": l2.phi,
            "mass": l2.mass,
            "charge": l2.charge if hasattr(l2, "charge") else (ak.where(l2.pdgId > 0, -1, 1) if "pdgId" in getattr(l2, "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        j1_scaled = ak.zip({
            "pt": J[:, 0].pt * j1_sf,
            "eta": J[:, 0].eta,
            "phi": J[:, 0].phi,
            "mass": J[:, 0].mass,
            "charge": J[:, 0].charge if hasattr(J[:, 0], "charge") else (ak.where(J[:, 0].pdgId > 0, -1, 1) if "pdgId" in getattr(J[:, 0], "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        j2_scaled = ak.zip({
            "pt": J[:, 1].pt * j2_sf,
            "eta": J[:, 1].eta,
            "phi": J[:, 1].phi,
            "mass": J[:, 1].mass,
            "charge": J[:, 1].charge if hasattr(J[:, 1], "charge") else (ak.where(J[:, 1].pdgId > 0, -1, 1) if "pdgId" in getattr(J[:, 1], "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        scaled_obj = l1_scaled + l2_scaled + j1_scaled + j2_scaled
        nominal_obj = LL + JJ
        return ak.where(both_mu, getattr(scaled_obj, attr), getattr(nominal_obj, attr))

    return getter


def _make_muon_exact4_scaled_fourobj_getter(scale_factor_fn, attr):
    def getter(L, J, LL, JJ):
        mask = _get_muon_and_exact4_mask(L, J)
        if not ak.any(mask):
            return getattr(LL + JJ, attr)

        l1 = L[:, 0]
        l2 = L[:, 1]
        l1_sf, l2_sf, j1_sf, j2_sf = scale_factor_fn(l1, l2, J)

        l1_scaled = ak.zip({
            "pt": l1.pt * l1_sf,
            "eta": l1.eta,
            "phi": l1.phi,
            "mass": l1.mass,
            "charge": l1.charge if hasattr(l1, "charge") else (ak.where(l1.pdgId > 0, -1, 1) if "pdgId" in getattr(l1, "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        l2_scaled = ak.zip({
            "pt": l2.pt * l2_sf,
            "eta": l2.eta,
            "phi": l2.phi,
            "mass": l2.mass,
            "charge": l2.charge if hasattr(l2, "charge") else (ak.where(l2.pdgId > 0, -1, 1) if "pdgId" in getattr(l2, "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        j1_scaled = ak.zip({
            "pt": J[:, 0].pt * j1_sf,
            "eta": J[:, 0].eta,
            "phi": J[:, 0].phi,
            "mass": J[:, 0].mass,
            "charge": J[:, 0].charge if hasattr(J[:, 0], "charge") else (ak.where(J[:, 0].pdgId > 0, -1, 1) if "pdgId" in getattr(J[:, 0], "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        j2_scaled = ak.zip({
            "pt": J[:, 1].pt * j2_sf,
            "eta": J[:, 1].eta,
            "phi": J[:, 1].phi,
            "mass": J[:, 1].mass,
            "charge": J[:, 1].charge if hasattr(J[:, 1], "charge") else (ak.where(J[:, 1].pdgId > 0, -1, 1) if "pdgId" in getattr(J[:, 1], "fields", []) else 0),
        }, with_name="PtEtaPhiMCandidate")
        scaled_obj = l1_scaled + l2_scaled + j1_scaled + j2_scaled
        nominal_obj = LL + JJ
        return ak.where(mask, getattr(scaled_obj, attr), getattr(nominal_obj, attr))

    return getter


muon_scaled_fourobj_getter = _make_muon_scaled_fourobj_getter(_default_muon_scale_factors, "mass")
muon_scaled_fourobj_pt_getter = _make_muon_scaled_fourobj_getter(_default_muon_scale_factors, "pt")
muon_exact4_scaled_fourobj_getter = _make_muon_exact4_scaled_fourobj_getter(_default_muon_scale_factors, "mass")
muon_exact4_scaled_fourobj_pt_getter = _make_muon_exact4_scaled_fourobj_getter(_default_muon_scale_factors, "pt")

RESOLVED_HIST_SPECS.append(
    (
        "mass_fourobj_muon_corr",
        (800, 0, 8000),
        r"$m_{\\ell\\ell jj}$ (muon-scaled pt)",
        muon_scaled_fourobj_getter,
    )
)
RESOLVED_HIST_SPECS.append(
    (
        "pt_total_fourobj_muon_corr",
        (800, 0, 8000),
        r"$p_{T}^{\mathrm{tot}}$ (muon-scaled pt) [GeV]",
        muon_scaled_fourobj_pt_getter,
    )
)
RESOLVED_HIST_SPECS.append(
    (
        "mass_fourobj_muon_corr_exact4",
        (800, 0, 8000),
        r"$m_{\ell\ell jj}$ (muon-scaled pt, exact 4 objects)",
        muon_exact4_scaled_fourobj_getter,
    )
)
RESOLVED_HIST_SPECS.append(
    (
        "pt_total_fourobj_muon_corr_exact4",
        (800, 0, 8000),
        r"$p_{T}^{\mathrm{tot}}$ (muon-scaled pt, exact 4 objects) [GeV]",
        muon_exact4_scaled_fourobj_pt_getter,
    )
)

for idx, name in enumerate(["s1", "s2", "s3", "s4"], start=1):
    RESOLVED_HIST_SPECS.append(
        (
            f"{name}_correction",
            (400, 0, 2),
            rf"${name}$ correction factor",
            lambda L, J, LL, JJ, idx=idx: _default_muon_scale_factors(L[:, 0], L[:, 1], J)[idx - 1],
        )
    )
    RESOLVED_HIST_SPECS.append(
        (
            f"{name}_correction_exact4",
            (400, 0, 2),
            rf"${name}$ correction factor (exact 4 objects)",
            lambda L, J, LL, JJ, idx=idx: ak.where(
                _get_muon_and_exact4_mask(L, J),
                _default_muon_scale_factors(L[:, 0], L[:, 1], J)[idx - 1],
                0.0,
            ),
        )
    )

RESOLVED_2D_HIST_SPECS: list[
    tuple[
        str,               
        str, tuple[int,float,float], str,   # x axis
        str, tuple[int,float,float], str,   # y axis
        ResolvedGetter,
    ]
] = [
    ("etavsPhi_leadingJet",("eta", (60, -3, 3), r"$\\eta$"),("phi", (80, -4, 4), r"$\\phi$"), lambda L, J, LL, JJ: (J[:, 0].eta,J[:, 0].phi),),
    ("etavsPhi_subleadingJet",("eta", (60, -3, 3), r"$\\eta$"),("phi", (80, -4, 4), r"$\\phi$"), lambda L, J, LL, JJ: (J[:, 1].eta,J[:, 1].phi),),
]


BOOSTED_HIST_SPECS: list[tuple[str, tuple[int, float, float], str, BoostedGetter]] = [
    ("pt_leading_lepton",               (200,   0, 2000), r"$p_{T}$ of the leading lepton [GeV]",  lambda lep, ak8, loose: lep.pt),
    ("eta_leading_lepton",              (60,   -3,    3), r"$\\eta$ of the leading lepton",       lambda lep, ak8, loose: lep.eta),
    ("phi_leading_lepton",              (80,   -4,    4), r"$\\phi$ of the leading lepton",       lambda lep, ak8, loose: lep.phi),
    ("pt_subleading_lepton",            (200,   0, 2000), r"$p_{T}$ of the subleading lepton [GeV]", lambda lep, ak8, loose: loose.pt),
    ("eta_subleading_lepton",           (60,   -3,    3), r"$\\eta$ of the subleading lepton",    lambda lep, ak8, loose: loose.eta),
    ("phi_subleading_lepton",           (80,   -4,    4), r"$\\phi$ of the subleading lepton",    lambda lep, ak8, loose: loose.phi),
    ("pt_leading_AK8Jets",              (200,   0, 2000), r"$p_{T}$ of the leading  AK8Jets [GeV]", lambda lep, ak8, loose: ak8.pt),
    ("eta_leading_AK8Jets",             (60,   -3,    3), r"$\\eta$ of theleading  AK8Jets",       lambda lep, ak8, loose: ak8.eta),
    ("phi_leading_AK8Jets",             (80,   -4,    4), r"$\\phi$ of theleading  AK8Jets",       lambda lep, ak8, loose: ak8.phi),
    ("mass_dilepton",                   (5000,  0, 5000), r"$m_{\\ell\\ell}$ [GeV]",             lambda lep, ak8, loose: (lep + loose).mass),
    ("pt_dilepton",                     (200,   0, 2000), r"$p_{T,\\ell\\ell}$ [GeV]",           lambda lep, ak8, loose: (lep + loose).pt),
    ("mass_twoobject",                  (800,   0, 8000), r"$m_{\\ell\\ell jj}$ [GeV]",          lambda lep, ak8, loose: (lep + ak8).mass),
    ("pt_twoobject",                    (800,   0, 8000), r"$p_{T,\\ell\\ell jj}$ [GeV]",        lambda lep, ak8, loose: (lep + ak8).pt),
    ("LSF_leading_AK8Jets",             (200,   0, 1.1),  r"LSF of leading AK8Jets",                lambda lep, ak8, loose: ak8.lsf3),
    ("dPhi_leading_tightlepton_AK8Jet", (80,   -4,    4), r"$d\\phi$ (leading Tight lepton, AK8 Jet)", lambda lep, ak8, loose: abs(ak8.delta_phi(lep))),
]
BOOSTED_2D_HIST_SPECS: list[
    tuple[str,                  # histogram key                                                                                                     
          str, tuple[int,float,float], str,   # x axis                                                                                              
          str, tuple[int,float,float], str,   # y axis                                                                                              
          BoostedGetter,        # returns (xvals, yvals)
    ]
] = [

 ("etavsPhi_leadingJet",("eta", (60, -3, 3), r"$\\eta$"),("phi", (80, -4, 4), r"$\\phi$"), lambda lep, ak8,loose: (ak8.eta, ak8.phi),),
]
def _booking_specs() -> dict[str, tuple[tuple[int, float, float], str]]:
    """Return histogram booking metadata keyed by canonical histogram name."""
    specs: dict[str, tuple[tuple[int, float, float], str]] = {}
    for name, bins, label, _ in RESOLVED_HIST_SPECS:
        specs[name] = (bins, label)
    for name, bins, label, _ in BOOSTED_HIST_SPECS:
        specs[name] = (bins, label)
    # Misc always-available hists
    specs.setdefault("count", ((100, 0, 100), r"count"))
    return specs


def create_hist(name, bins, label):
    """Create a single physics histogram with standard categorical axes."""
    return (
        hist.Hist.new
        .StrCat([], name="process", label="Process", growth=True)
        .StrCat([], name="region",  label="Analysis Region", growth=True)
        .StrCat([], name="syst",    label="Systematic", growth=True)
        .Reg(*bins, name=name, label=label)
        .Weight()
    )
def create_hist2D(name_x, bins_x, label_x,
                  name_y, bins_y, label_y):
    return (
        hist.Hist.new
        .StrCat([], name="process", label="Process", growth=True)
        .StrCat([], name="region",  label="Analysis Region", growth=True)
        .StrCat([], name="syst",    label="Systematic", growth=True)
        .Reg(*bins_x, name=name_x, label=label_x)
        .Reg(*bins_y, name=name_y, label=label_y)
        .Weight()
    )
def fill_resolved_histograms(output, region, cut, process_name, jets, leptons, weights, syst_weights):
    """Fill all resolved-region histograms for a given region selection mask."""
    leptons_cut = leptons[cut]
    jets_cut    = jets[cut]
    syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}

    # Pre-compute common 4-vectors once per region instead of per histogram.
    dilepton = leptons_cut[:, 0] + leptons_cut[:, 1]
    dijet    = jets_cut[:, 0] + jets_cut[:, 1]
    #print("Events entering histogram filling:", len(leptons_cut[:,0]))
    for hist_name, _bins, _label, expr in RESOLVED_HIST_SPECS:
        vals = expr(leptons_cut, jets_cut, dilepton, dijet)
        for syst_label, sw in syst_weights_cut.items():
            output[hist_name].fill(
                process=process_name,
                region=region,
                syst=syst_label,
                **{hist_name: vals},
                weight=sw,
            )

    for hist_key, xinfo, yinfo, expr in RESOLVED_2D_HIST_SPECS:
        xvals, yvals = expr(leptons_cut, jets_cut, dilepton, dijet)
        
        xname, _, _ = xinfo
        yname, _, _ = yinfo
        
        for syst_label, sw in syst_weights_cut.items():
            output[hist_key].fill(
                process=process_name,
                region=region,
                syst=syst_label,
                **{
                    xname: xvals,
                    yname: yvals,
                },
                weight=sw,
            )
            
def fill_boosted_histograms(output, region, cut, process_name, leptons, ak8jets, looseleptons, weights, syst_weights):
    """Fill all boosted-region histograms for a given region selection mask."""
    syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}

    # Evaluate all base boosted quantities from the spec table.
    # Note: in boosted mode, the inputs are already per-event objects (not padded collections).
    value_map = {
        hist_name: expr(leptons, ak8jets, looseleptons)
        for hist_name, _bins, _label, expr in BOOSTED_HIST_SPECS
    }

    # Special-case DY CR: mass_twoobject / pt_twoobject switch depending on dR.
    if "boosted_dy_cr" in region:
        dr_dy = ak8jets.deltaR(looseleptons)
        value_map["mass_twoobject"] = ak.where(
            dr_dy < CUTS["dr_ak8_loose"],
            (leptons + ak8jets).mass,
            (leptons + ak8jets + looseleptons).mass,
        )
        value_map["pt_twoobject"] = ak.where(
            dr_dy < CUTS["dr_ak8_loose"],
            (leptons + ak8jets).pt,
            (leptons + ak8jets + looseleptons).pt,
        )

    for hist_name, vals_all in value_map.items():
        vals = vals_all[cut]
        for syst_label, sw in syst_weights_cut.items():
            output[hist_name].fill(
                process=process_name,
                region=region,
                syst=syst_label,
                **{hist_name: vals},
                weight=sw,
            )

    for (
            hist_key,
            xinfo,
            yinfo,
            expr
    ) in BOOSTED_2D_HIST_SPECS:
        xname, _, _ = xinfo
        yname, _, _ = yinfo
        # Evaluate full arrays first (same logic as 1D)
        xvals_all, yvals_all = expr(leptons, ak8jets, looseleptons)
        
        # Apply region cut
        xvals = xvals_all[cut]
        yvals = yvals_all[cut]
        
        for syst_label, sw in syst_weights_cut.items():
            output[hist_key].fill(
                process=process_name,
                region=region,
                syst=syst_label,
                **{
                    xname: xvals,
                    yname: yvals,
                },
                weight=sw,
            )
def _relabel_cutflow(h_raw, cut_names):
    """Convert an Integer-axis cutflow histogram to one with StrCategory axis.

    This embeds the cut names as bin labels in the ROOT file, making it
    self-documenting and robust against ordering changes.
    """
    h = hist.Hist(
        hist.axis.StrCategory(cut_names, name="cut"),
        storage=h_raw.storage_type(),
    )
    h.view(flow=False)[...] = h_raw.view(flow=False)
    return h


def fill_cutflows(output, selections, weights):
    """Build cumulative cutflows for ee, mumu, and em channels.

    Output layout (keys under ``output["cutflow"]``):
        - per-flavor: ``ee``, ``mumu``, ``em``
            - ``onecut`` / ``cumulative`` (and unweighted variants)
              Multi-bin histograms with StrCategory axis (bin labels are
              the cut names, e.g. "no_cuts", "min_two_ak4_jets_pteta", ...).
    """
    output.setdefault("cutflow", {})

    # --- Define cumulative chains per flavor
    chains = {
        "ee": [
            #SEL_JET_VETO_MAP,
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_TWO_PTETA_ELECTRONS,
            SEL_TWO_ID_ELECTRONS,
            SEL_E_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT200,
            SEL_MLL_GT400,
        ],
        "mumu": [
            #SEL_JET_VETO_MAP,
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_TWO_PTETA_MUONS,
            SEL_TWO_ID_MUONS,
            SEL_MU_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT200,
            SEL_MLL_GT400,
        ],
        "em": [
            #SEL_JET_VETO_MAP,
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_TWO_PTETA_EM,
            SEL_TWO_ID_EM,
            SEL_EMU_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT200,
            SEL_MLL_GT400,
        ],
    }

    # --- Per-flavor: multi-bin onecut / cumulative histograms
    for flavor, steps in chains.items():
        output["cutflow"].setdefault(flavor, {})
        bucket = output["cutflow"][flavor]

        # Cut names for axis labels: "no_cuts" + the selection step names
        cut_names = ["no_cuts"] + list(steps)

        cf = selections.cutflow(*steps, weights=weights)
        h_onecut_raw, h_cum_raw, _labels = cf.yieldhist(weighted=True)
        bucket["onecut"] = _relabel_cutflow(h_onecut_raw, cut_names)
        bucket["cumulative"] = _relabel_cutflow(h_cum_raw, cut_names)

        h_onecut_unw, h_cum_unw, _labels = cf.yieldhist(weighted=False)
        bucket["onecut_unweighted"] = _relabel_cutflow(h_onecut_unw, cut_names)
        bucket["cumulative_unweighted"] = _relabel_cutflow(h_cum_unw, cut_names)


def fill_boosted_cutflows(output, selections, weights):
    """Build cumulative cutflows for boosted ee, mumu, and em channels.

    Similar to fill_cutflows but for boosted topology. Shows progression
    through boosted-specific selections (SR progression for ee/mumu,
    flavor CR for em).

    Output layout (keys under ``output["cutflow_boosted"]``):
        - per-flavor: ``ee``, ``mumu``, ``em``
            - ``onecut`` / ``cumulative`` (and unweighted variants)
    """
    output.setdefault("cutflow_boosted", {})

    # Boosted cutflow chains - expanded SR progression for ee/mumu, flavor CR for em
    chains = {
        "ee": [
            #SEL_JET_VETO_MAP,
            SEL_BOOSTEDTAG,
            SEL_LEAD_IS_ELECTRON,
            SEL_LEAD_TIGHT_PT60_BOOSTED,
            SEL_E_TRIGGER,
            SEL_NO_DY_PAIR,
            SEL_AK8JETS_WITH_LSF,
            SEL_NO_EXTRA_TIGHT_SR,
            SEL_SF_LEPTON_IN_AK8,
            SEL_NO_OF_LEPTON_IN_AK8,
            SEL_MLL_GT200_BOOSTED,
            SEL_MLJ_GT800_BOOSTED,
        ],
        "mumu": [
            #SEL_JET_VETO_MAP,
            SEL_BOOSTEDTAG,
            SEL_LEAD_IS_MUON,
            SEL_LEAD_TIGHT_PT60_BOOSTED,
            SEL_MU_TRIGGER,
            SEL_NO_DY_PAIR,
            SEL_AK8JETS_WITH_LSF,
            SEL_NO_EXTRA_TIGHT_SR,
            SEL_SF_LEPTON_IN_AK8,
            SEL_NO_OF_LEPTON_IN_AK8,
            SEL_MLL_GT200_BOOSTED,
            SEL_MLJ_GT800_BOOSTED,
        ],
        "em": [
            #SEL_JET_VETO_MAP,
            SEL_BOOSTEDTAG,
            SEL_LEAD_IS_ELECTRON,
            SEL_LEAD_TIGHT_PT60_BOOSTED,
            SEL_E_TRIGGER,
            SEL_NO_DY_PAIR,
            SEL_AK8JETS_WITH_LSF,
            SEL_NO_EXTRA_TIGHT_CR,
            SEL_NO_SF_LEPTON_IN_AK8,
            SEL_OF_LEPTON_IN_AK8,
            SEL_MLL_GT200_BOOSTED,
            SEL_MLJ_GT800_BOOSTED,
        ],
    }

    # Generate cutflow histograms per flavor
    for flavor, steps in chains.items():
        output["cutflow_boosted"].setdefault(flavor, {})
        bucket = output["cutflow_boosted"][flavor]

        cut_names = ["no_cuts"] + list(steps)

        cf = selections.cutflow(*steps, weights=weights)
        h_onecut_raw, h_cum_raw, _labels = cf.yieldhist(weighted=True)
        bucket["onecut"] = _relabel_cutflow(h_onecut_raw, cut_names)
        bucket["cumulative"] = _relabel_cutflow(h_cum_raw, cut_names)

        h_onecut_unw, h_cum_unw, _labels = cf.yieldhist(weighted=False)
        bucket["onecut_unweighted"] = _relabel_cutflow(h_onecut_unw, cut_names)
        bucket["cumulative_unweighted"] = _relabel_cutflow(h_cum_unw, cut_names)
