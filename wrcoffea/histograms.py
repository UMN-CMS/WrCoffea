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

from wrcoffea.analysis_config import (
    CUTS,
    SEL_MIN_TWO_AK4_JETS_PTETA, SEL_MIN_TWO_AK4_JETS_ID,
    SEL_TWO_PTETA_ELECTRONS, SEL_TWO_PTETA_MUONS, SEL_TWO_PTETA_EM,
    SEL_TWO_ID_ELECTRONS, SEL_TWO_ID_MUONS, SEL_TWO_ID_EM,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_MLL_GT200, SEL_MLLJJ_GT800, SEL_MLL_GT400,
    SEL_JET_VETO_MAP,
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

    # Pre-compute common 4-vectors once per region instead of per histogram.
    dilepton = leptons_cut[:, 0] + leptons_cut[:, 1]
    dijet    = jets_cut[:, 0] + jets_cut[:, 1]

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
            SEL_JET_VETO_MAP,
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
            SEL_JET_VETO_MAP,
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
            SEL_JET_VETO_MAP,
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
            SEL_JET_VETO_MAP,
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
            SEL_JET_VETO_MAP,
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
            SEL_JET_VETO_MAP,
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
