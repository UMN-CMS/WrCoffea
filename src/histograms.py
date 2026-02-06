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
)

logger = logging.getLogger(__name__)

ResolvedGetter = Callable[[ak.Array, ak.Array], ak.Array]
BoostedGetter = Callable[[ak.Array, ak.Array, ak.Array], ak.Array]


RESOLVED_HIST_SPECS: list[tuple[str, tuple[int, float, float], str, ResolvedGetter]] = [
    ("pt_leading_lepton",           (200,   0, 2000), r"$p_{T}$ of the leading lepton [GeV]",           lambda L, J: L[:, 0].pt),
    ("eta_leading_lepton",          (60,   -3,    3), r"$\\eta$ of the leading lepton",                lambda L, J: L[:, 0].eta),
    ("phi_leading_lepton",          (80,   -4,    4), r"$\\phi$ of the leading lepton",                lambda L, J: L[:, 0].phi),
    ("pt_subleading_lepton",        (200,   0, 2000), r"$p_{T}$ of the subleading lepton [GeV]",        lambda L, J: L[:, 1].pt),
    ("eta_subleading_lepton",       (60,   -3,    3), r"$\\eta$ of the subleading lepton",             lambda L, J: L[:, 1].eta),
    ("phi_subleading_lepton",       (80,   -4,    4), r"$\\phi$ of the subleading lepton",             lambda L, J: L[:, 1].phi),
    ("pt_leading_jet",              (200,   0, 2000), r"$p_{T}$ of the leading jet [GeV]",              lambda L, J: J[:, 0].pt),
    ("eta_leading_jet",             (60,   -3,    3), r"$\\eta$ of the leading jet",                   lambda L, J: J[:, 0].eta),
    ("phi_leading_jet",             (80,   -4,    4), r"$\\phi$ of the leading jet",                   lambda L, J: J[:, 0].phi),
    ("pt_subleading_jet",           (200,   0, 2000), r"$p_{T}$ of the subleading jet [GeV]",           lambda L, J: J[:, 1].pt),
    ("eta_subleading_jet",          (60,   -3,    3), r"$\\eta$ of the subleading jet",                lambda L, J: J[:, 1].eta),
    ("phi_subleading_jet",          (80,   -4,    4), r"$\\phi$ of the subleading jet",                lambda L, J: J[:, 1].phi),
    ("mass_dilepton",               (5000,  0, 5000), r"$m_{\\ell\\ell}$ [GeV]",                     lambda L, J: (L[:, 0] + L[:, 1]).mass),
    ("pt_dilepton",                 (200,   0, 2000), r"$p_{T,\\ell\\ell}$ [GeV]",                   lambda L, J: (L[:, 0] + L[:, 1]).pt),
    ("mass_dijet",                  (500,   0, 5000), r"$m_{jj}$ [GeV]",                               lambda L, J: (J[:, 0] + J[:, 1]).mass),
    ("pt_dijet",                    (500,   0, 5000), r"$p_{T,jj}$ [GeV]",                             lambda L, J: (J[:, 0] + J[:, 1]).pt),
    ("mass_threeobject_leadlep",    (800,   0, 8000), r"$m_{\\ell jj}$ [GeV]",                        lambda L, J: (L[:, 0] + J[:, 0] + J[:, 1]).mass),
    ("pt_threeobject_leadlep",      (800,   0, 8000), r"$p_{T,\\ell jj}$ [GeV]",                      lambda L, J: (L[:, 0] + J[:, 0] + J[:, 1]).pt),
    ("mass_threeobject_subleadlep", (800,   0, 8000), r"$m_{\\ell jj}$ [GeV]",                        lambda L, J: (L[:, 1] + J[:, 0] + J[:, 1]).mass),
    ("pt_threeobject_subleadlep",   (800,   0, 8000), r"$p_{T,\\ell jj}$ [GeV]",                      lambda L, J: (L[:, 1] + J[:, 0] + J[:, 1]).pt),
    ("mass_fourobject",             (800,   0, 8000), r"$m_{\\ell\\ell jj}$ [GeV]",                 lambda L, J: (L[:, 0] + L[:, 1] + J[:, 0] + J[:, 1]).mass),
    ("pt_fourobject",               (800,   0, 8000), r"$p_{T,\\ell\\ell jj}$ [GeV]",               lambda L, J: (L[:, 0] + L[:, 1] + J[:, 0] + J[:, 1]).pt),
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


def fill_resolved_histograms(output, region, cut, process_name, jets, leptons, weights, syst_weights):
    """Fill all resolved-region histograms for a given region selection mask."""
    leptons_cut = leptons[cut]
    jets_cut    = jets[cut]
    syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}

    for hist_name, _bins, _label, expr in RESOLVED_HIST_SPECS:
        vals = expr(leptons_cut, jets_cut)
        for syst_label, sw in syst_weights_cut.items():
            output[hist_name].fill(
                process=process_name,
                region=region,
                syst=syst_label,
                **{hist_name: vals},
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


def fill_cutflows(output, selections, weights):
    """Build flat and cumulative cutflows for ee, mumu, and em channels.

    Also store one-bin histograms for pT/eta-only vs ID-only (jets & leptons).

    Output layout (keys under ``output["cutflow"]``):
        - top-level: ``no_cuts``, plus a few global one-bin counters
        - per-flavor: ``ee``, ``mumu``, ``em``
            - one-bin counters for each step (weighted + ``_unweighted``)
            - ``onecut`` / ``cumulative`` (and unweighted variants) from PackedSelection.cutflow
    """
    output.setdefault("cutflow", {})

    required = [
        SEL_MIN_TWO_AK4_JETS_ID,
        SEL_MIN_TWO_AK4_JETS_PTETA,
        SEL_TWO_PTETA_ELECTRONS,
        SEL_TWO_PTETA_MUONS,
        SEL_TWO_PTETA_EM,
        SEL_TWO_ID_ELECTRONS,
        SEL_TWO_ID_MUONS,
        SEL_TWO_ID_EM,
        SEL_E_TRIGGER,
        SEL_MU_TRIGGER,
        SEL_EMU_TRIGGER,
        SEL_DR_ALL_PAIRS_GT0P4,
        SEL_MLL_GT200,
        SEL_MLLJJ_GT800,
        SEL_MLL_GT400,
    ]

    # Fail fast with a helpful message if resolved_selections does not define expected keys.
    missing = []
    for name in required:
        try:
            selections.all(name)
        except Exception:
            missing.append(name)
    if missing:
        available = getattr(selections, "names", None)
        raise KeyError(
            "fill_cutflows: missing selection keys: "
            f"{missing}. Available keys: {sorted(available) if available else 'unknown'}"
        )

    mask = selections.all

    m_j2_id = mask(SEL_MIN_TWO_AK4_JETS_ID)
    m_j2_pteta = mask(SEL_MIN_TWO_AK4_JETS_PTETA)

    m_two_pteta_e = mask(SEL_TWO_PTETA_ELECTRONS)
    m_two_pteta_mu = mask(SEL_TWO_PTETA_MUONS)
    m_two_pteta_em = mask(SEL_TWO_PTETA_EM)

    m_two_id_e = mask(SEL_TWO_ID_ELECTRONS)
    m_two_id_mu = mask(SEL_TWO_ID_MUONS)
    m_two_id_em = mask(SEL_TWO_ID_EM)

    m_e_trig = mask(SEL_E_TRIGGER)
    m_mu_trig = mask(SEL_MU_TRIGGER)
    m_em_trig = mask(SEL_EMU_TRIGGER)

    m_dr = mask(SEL_DR_ALL_PAIRS_GT0P4)
    m_mll200 = mask(SEL_MLL_GT200)
    m_mlljj8 = mask(SEL_MLLJJ_GT800)
    m_mll400 = mask(SEL_MLL_GT400)

    w = weights.weight()  # nominal weight

    def _onebin_hist(lastcut_name: str, mask, use_weights=True):
        """1 bin in [0,1), named by the last cut."""
        h = hist.Hist(
            hist.axis.Regular(1, 0, 1, name=lastcut_name, label=lastcut_name),
            storage=hist.storage.Weight(),
        )
        n_pass = int(ak.count_nonzero(mask))
        if use_weights:
            w_pass = ak.to_numpy(w[mask])
        else:
            # unweighted: count passing events
            w_pass = np.ones(n_pass, dtype="f8")
        if n_pass:
            coords = np.full(w_pass.size, 0.5, dtype=float)
            h.fill(**{lastcut_name: coords}, weight=w_pass)
        return h

    def _store_both_in(container: dict, name: str, mask):
        """Store weighted and unweighted variants into a given dict container."""
        container[name] = _onebin_hist(name, mask, use_weights=True)
        container[f"{name}_unweighted"] = _onebin_hist(name, mask, use_weights=False)

    # --- Top-level: no_cuts only
    n_events = len(w)
    mask_all = np.ones(n_events, dtype=bool)
    _store_both_in(output["cutflow"], "no_cuts", mask_all)
    _store_both_in(output["cutflow"], SEL_MIN_TWO_AK4_JETS_PTETA, m_j2_pteta)

    # --- Prepare flavor folders (+ jets folder for pt/eta vs ID)
    output["cutflow"].setdefault("ee", {})
    output["cutflow"].setdefault("mumu", {})
    output["cutflow"].setdefault("em", {})

    # --- Define cumulative chains per flavor (kept as-is)
    chains = {
        "ee":   [
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
        "em":   [
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

    name2mask = {
        SEL_MIN_TWO_AK4_JETS_ID: m_j2_id,
        SEL_MIN_TWO_AK4_JETS_PTETA: m_j2_pteta,

        SEL_TWO_PTETA_ELECTRONS: m_two_pteta_e,
        SEL_TWO_PTETA_MUONS: m_two_pteta_mu,
        SEL_TWO_PTETA_EM: m_two_pteta_em,

        SEL_TWO_ID_ELECTRONS: m_two_id_e,
        SEL_TWO_ID_MUONS: m_two_id_mu,
        SEL_TWO_ID_EM: m_two_id_em,

        SEL_E_TRIGGER: m_e_trig,
        SEL_MU_TRIGGER: m_mu_trig,
        SEL_EMU_TRIGGER: m_em_trig,

        SEL_DR_ALL_PAIRS_GT0P4: m_dr,
        SEL_MLL_GT200: m_mll200,
        SEL_MLLJJ_GT800: m_mlljj8,
        SEL_MLL_GT400: m_mll400,
    }

    # --- Step-by-step cumulative counters per flavor
    for flavor, steps in chains.items():
        cumu = name2mask[steps[0]].copy()
        bucket = output["cutflow"][flavor]
        for step in steps:
            if step != steps[0]:
                cumu = cumu & name2mask[step]
            _store_both_in(bucket, step, cumu)

    # --- Region cutflows: onecut + cumulative into flavor folders
    for flavor, order in chains.items():
        cf = selections.cutflow(*order, weights=weights)

        # weighted
        res = cf.yieldhist(weighted=True)
        h_onecut, h_cum = res[0], res[1]
        bucket = output["cutflow"][flavor]
        bucket["onecut"] = h_onecut
        bucket["cumulative"] = h_cum

        # unweighted
        res_unw = cf.yieldhist(weighted=False)
        h_onecut_unw, h_cum_unw = res_unw[0], res_unw[1]
        bucket["onecut_unweighted"] = h_onecut_unw
        bucket["cumulative_unweighted"] = h_cum_unw
