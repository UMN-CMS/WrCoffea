"""Histogram specification, creation, and filling for the WR analysis.

Canonical naming choice:
  - Histogram key in the output dict == numeric axis name == ROOT stem

Each spec is: (name, bins, label, getter)
  - For resolved: getter(tight_leptons, ak4_jets) -> values
  - For boosted:  getter(tight_lepton, ak8_jet, loose_lepton) -> values

Systematic coverage:
  ``fill_*_histograms`` take ``syst_observables``: ``None`` fills every
  histogram for every entry of ``syst_weights``; a set of names restricts the
  non-Nominal entries to those histograms (Nominal is always filled
  everywhere).  Both return the set of syst labels that were completed for
  the region so the analyzer can record chunk coverage.

Event-weight distributions (``fill_event_weights``) and cutflows are plain
nested dicts of Hists outside this booking: no process/region/syst axes and
no systematic variations.
"""

import logging
from typing import Callable

import awkward as ak
import hist
import numpy as np
from coffea.processor import column_accumulator

from wrcoffea.regions import blind_cutflow_steps
from wrcoffea.pdf_weights import PDF_OBSERVABLES, fill_members
from wrcoffea.systematics import is_varied
from wrcoffea.analysis_config import (
    CUTS,
    SEL_MIN_TWO_AK4_JETS_PTETA, SEL_MIN_TWO_AK4_JETS_ID,
    SEL_TWO_PTETA_ELECTRONS, SEL_TWO_PTETA_MUONS, SEL_TWO_PTETA_EM,
    SEL_TWO_ID_ELECTRONS, SEL_TWO_ID_MUONS, SEL_TWO_ID_EM,
    SEL_LEAD_TIGHT_PT60, SEL_NO_THIRD_TIGHT_LEPTON,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_HEM_VETO, SEL_MLL_GT200, SEL_MLLJJ_GT800, SEL_MLL_GT400,
    SEL_HEM_VETO_BOOSTED_SR, SEL_HEM_VETO_BOOSTED_CR,
    #SEL_JET_VETO_MAP,
    # Boosted selections
    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_AK8JETS_WITH_LSF,
    SEL_LEAD_IS_ELECTRON, SEL_LEAD_IS_MUON,
    SEL_NO_DY_PAIR, SEL_NO_EXTRA_TIGHT_SR, SEL_NO_EXTRA_TIGHT_CR,
    SEL_SF_LEPTON_IN_AK8, SEL_NO_OF_LEPTON_IN_AK8,
    SEL_OF_LEPTON_IN_AK8, SEL_NO_SF_LEPTON_IN_AK8,
    SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED,
)

logger = logging.getLogger(__name__)

NOMINAL = "Nominal"


def _labels_for(hist_name, syst_weights, syst_observables):
    """Which entries of ``syst_weights`` are filled into ``hist_name``."""
    if is_varied(hist_name, syst_observables):
        return list(syst_weights)
    return [k for k in syst_weights if k == NOMINAL]


def _fill_pdf_members(output, pdf, hist_name, region, cut, vals, syst_weights_cut, syst_observables):
    """Fill the PDF members of *hist_name* (a fit mass) when this pass carries them."""
    if pdf is None or hist_name not in PDF_OBSERVABLES or NOMINAL not in syst_weights_cut:
        return
    if not is_varied(hist_name, syst_observables):
        return
    store, members = pdf
    axis = output[hist_name].axes[hist_name]
    bins = (axis.size, axis.edges[0], axis.edges[-1])
    fill_members(store, hist_name, bins, region, vals, syst_weights_cut[NOMINAL], members.weights(cut))


def all_histogram_names():
    """Every histogram key booked by the analyzer (1D and 2D)."""
    names = set(_booking_specs())
    names.update(key for key, *_ in RESOLVED_2D_HIST_SPECS)
    names.update(key for key, *_ in BOOSTED_2D_HIST_SPECS)
    return frozenset(names)

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

EVENT_WEIGHT_AXIS = "log10_abs_weight"
EVENT_WEIGHT_BINS = (200, -8, 2)

# Weight components (event_weights/<region>/components/<name>).  The generator
# weight and the xsec * lumi / sumw normalization span orders of magnitude and
# use a log10 axis (the generator weight split by sign, like the total); every
# other component is a multiplicative correction near 1 on a linear axis.
# "corrections" is the product of the linear factors, so
# total = genweight * normalization * corrections.
WEIGHT_COMPONENT_LOG_BINS = (400, -12, 8)
WEIGHT_FACTOR_AXIS = "factor"
WEIGHT_FACTOR_BINS = (600, 0.0, 3.0)
GENWEIGHT_COMPONENT = "genweight"
NORMALIZATION_COMPONENT = "normalization"
CORRECTIONS_COMPONENT = "corrections"


def create_event_weight_hist(bins=EVENT_WEIGHT_BINS):
    """Event count vs log10|w|, the axis of the per-region weight distributions."""
    return hist.Hist(
        hist.axis.Regular(*bins, name=EVENT_WEIGHT_AXIS, label=r"$\log_{10}|w|$"),
        storage=hist.storage.Weight(),
    )


def create_weight_factor_hist():
    """Event count vs the value of one multiplicative weight factor."""
    return hist.Hist(
        hist.axis.Regular(*WEIGHT_FACTOR_BINS, name=WEIGHT_FACTOR_AXIS, label="Weight factor"),
        storage=hist.storage.Weight(),
    )


def fill_event_weights(event_weights, region, cut, weights, factors=None):
    """Fill log10|w| of the events in *region*, split by the sign of w.

    *weights* is the region's nominal per-event weight for the whole chunk
    (``region_syst[NOMINAL]``: lepton SFs and DY reshape included), *cut* the
    region mask.  Each entry counts one event (unit weight); events with
    w == 0 are not filled, and values outside the axis land in the flow bins.
    ``event_weights`` is the ``{region: {"positive": Hist, "negative": Hist}}``
    accumulator dict; both histograms are booked even when the region is
    empty in this chunk.

    *factors*, when given, is ``{name: per-event array}`` of the factors whose
    product is *weights*; each is filled under ``[region]["components"]``
    (see the ``WEIGHT_COMPONENT_*`` constants for the axes).
    """
    mask = np.asarray(cut, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)[mask]
    hists = event_weights.get(region)
    if hists is None:
        hists = event_weights[region] = {
            "positive": create_event_weight_hist(),
            "negative": create_event_weight_hist(),
        }
    _fill_signed_log(hists, "positive", "negative", w)
    if factors is not None:
        _fill_weight_components(hists.setdefault("components", {}), mask, factors)


def _booked(hists, key, create):
    """hists[key], booked with *create()* on first use."""
    h = hists.get(key)
    if h is None:
        h = hists[key] = create()
    return h


def _fill_signed_log(hists, pos_key, neg_key, values, bins=EVENT_WEIGHT_BINS):
    """Fill log10|v| into hists[pos_key] / hists[neg_key] by sign (v == 0 skipped)."""
    for key, selected in ((pos_key, values[values > 0]), (neg_key, values[values < 0])):
        _booked(hists, key, lambda: create_event_weight_hist(bins)).fill(
            **{EVENT_WEIGHT_AXIS: np.log10(np.abs(selected))})


def _fill_weight_components(hists, mask, factors):
    """Fill one distribution per weight factor of the masked events, plus the
    product of the linear factors (``corrections``)."""
    corrections = None
    for name, values in factors.items():
        v = np.asarray(values, dtype=np.float64)[mask]
        if name == GENWEIGHT_COMPONENT:
            _fill_signed_log(hists, f"{name}_positive", f"{name}_negative", v,
                             bins=WEIGHT_COMPONENT_LOG_BINS)
        elif name == NORMALIZATION_COMPONENT:
            _booked(hists, name, lambda: create_event_weight_hist(WEIGHT_COMPONENT_LOG_BINS)).fill(
                **{EVENT_WEIGHT_AXIS: np.log10(v[v > 0])})
        else:
            _booked(hists, name, create_weight_factor_hist).fill(**{WEIGHT_FACTOR_AXIS: v})
            corrections = v if corrections is None else corrections * v
    if corrections is not None:
        _booked(hists, CORRECTIONS_COMPONENT, create_weight_factor_hist).fill(
            **{WEIGHT_FACTOR_AXIS: corrections})


def record_events(event_lists, region, values, weights):
    """Append the per-event (mass, nominal weight) of one region to the
    ``event_lists`` accumulator dict ({region: {"mass": column_accumulator,
    "weight": column_accumulator}}), used by --save-events."""
    vals = np.asarray(ak.to_numpy(values), dtype=np.float64)
    wts = np.asarray(ak.to_numpy(weights), dtype=np.float64)
    cols = event_lists.setdefault(region, {})
    for key, arr in (("mass", vals), ("weight", wts)):
        if key in cols:
            cols[key].add(column_accumulator(arr))
        else:
            cols[key] = column_accumulator(arr)


def fill_resolved_histograms(output, region, cut, process_name, jets, leptons, weights, syst_weights,
                             syst_observables=None, event_lists=None, pdf=None):
    """Fill all resolved-region histograms for a given region selection mask.

    *pdf*, when given, is ``(store, PdfMembers)``: the PDF members of the fit
    mass are filled into ``store`` (nominal pass only).
    Returns the set of syst labels completed for *region*.
    """
    leptons_cut = leptons[cut]
    jets_cut    = jets[cut]
    syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}

    # Pre-compute common 4-vectors once per region instead of per histogram.
    dilepton = leptons_cut[:, 0] + leptons_cut[:, 1]
    dijet    = jets_cut[:, 0] + jets_cut[:, 1]
    #print("Events entering histogram filling:", len(leptons_cut[:,0]))
    for hist_name, _bins, _label, expr in RESOLVED_HIST_SPECS:
        labels = _labels_for(hist_name, syst_weights_cut, syst_observables)
        if not labels:
            continue
        vals = expr(leptons_cut, jets_cut, dilepton, dijet)
        if (event_lists is not None and hist_name == "mass_fourobject"
                and region.endswith("_sr") and NOMINAL in syst_weights_cut):
            record_events(event_lists, region, vals, syst_weights_cut[NOMINAL])
        _fill_pdf_members(output, pdf, hist_name, region, cut, vals, syst_weights_cut, syst_observables)
        for syst_label in labels:
            sw = syst_weights_cut[syst_label]
            output[hist_name].fill(
                process=process_name,
                region=region,
                syst=syst_label,
                **{hist_name: vals},
                weight=sw,
            )

    for hist_key, xinfo, yinfo, expr in RESOLVED_2D_HIST_SPECS:
        labels = _labels_for(hist_key, syst_weights_cut, syst_observables)
        if not labels:
            continue
        xvals, yvals = expr(leptons_cut, jets_cut, dilepton, dijet)

        xname, _, _ = xinfo
        yname, _, _ = yinfo

        for syst_label in labels:
            sw = syst_weights_cut[syst_label]
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

    return set(syst_weights_cut)


def fill_boosted_histograms(output, region, cut, process_name, leptons, ak8jets, looseleptons, weights, syst_weights,
                            syst_observables=None, event_lists=None, pdf=None):
    """Fill all boosted-region histograms for a given region selection mask.

    *pdf* as in :func:`fill_resolved_histograms`.
    Returns the set of syst labels completed for *region*.
    """
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
        labels = _labels_for(hist_name, syst_weights_cut, syst_observables)
        if not labels:
            continue
        vals = vals_all[cut]
        if (event_lists is not None and hist_name == "mass_twoobject"
                and region.endswith("_sr") and NOMINAL in syst_weights_cut):
            record_events(event_lists, region, vals, syst_weights_cut[NOMINAL])
        _fill_pdf_members(output, pdf, hist_name, region, cut, vals, syst_weights_cut, syst_observables)
        for syst_label in labels:
            sw = syst_weights_cut[syst_label]
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
        labels = _labels_for(hist_key, syst_weights_cut, syst_observables)
        if not labels:
            continue
        # Evaluate full arrays first (same logic as 1D)
        xvals_all, yvals_all = expr(leptons, ak8jets, looseleptons)

        # Apply region cut
        xvals = xvals_all[cut]
        yvals = yvals_all[cut]

        for syst_label in labels:
            sw = syst_weights_cut[syst_label]
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

    return set(syst_weights_cut)


# The last row of every complete cutflow chain: the cutflow weight times the lepton (and LSF) scale factors of the
# region the chain ends in, so it equals that region's yield (Billy, 2026-10-14).  No DY correction is in any
# cutflow: the cutflow weight leaves out the DY Z-pT reweight, and the DY jet-pT reshape is a per-region weight.
CUTFLOW_SF_ROW = "with_region_sfs"
CUTFLOW_EXCLUDED_WEIGHTS = ("dy_zpt",)
# The region each chain ends in (boosted em: the e(mu-jet) flavour CR, me: the mu(e-jet) one).
CUTFLOW_FINAL_REGIONS = {
    "resolved": {"ee": "wr_ee_resolved_sr", "mumu": "wr_mumu_resolved_sr", "em": "wr_resolved_flavor_cr"},
    "boosted": {"ee": "wr_ee_boosted_sr", "mumu": "wr_mumu_boosted_sr", "em": "wr_emu_boosted_flavor_cr",
                "me": "wr_mue_boosted_flavor_cr"},
}


def cutflow_weight(weights) -> np.ndarray:
    """The per-event weight of the cutflows: the base weight without CUTFLOW_EXCLUDED_WEIGHTS."""
    if not hasattr(weights, "weight"):
        return np.asarray(weights, dtype=np.float64)
    present = [n for n in CUTFLOW_EXCLUDED_WEIGHTS if n in getattr(weights, "weightStatistics", {})]
    return np.asarray(weights.partial_weight(exclude=present) if present else weights.weight(), dtype=np.float64)


def _cutflow_histograms(selections, weights, steps, sf_weights=None, start=None):
    """Build eager cutflows with event-level sum(w) and sum(w**2).

    Coffea 2025.12's eager ``yieldhist`` fills each bin with its total weight,
    giving (sum(w))**2 as the variance. Accumulate the second moment from
    individual events instead, so uncertainties are independent of chunking.

    *weights* is a coffea ``Weights`` (its cutflow weight is taken, see
    :func:`cutflow_weight`) or a per-event array.  With *sf_weights* (per event)
    a last row CUTFLOW_SF_ROW holds sum(sf_weights) over the events passing
    every step, in ``cumulative`` and in ``onecut`` alike (it is not a cut).
    With *start* (a per-event mask) only those events enter any row, ``no_cuts``
    included: the signal ee / mumu chains count their own generator flavour.
    """
    labels = ["no_cuts", *steps] + ([CUTFLOW_SF_ROW] if sf_weights is not None else [])
    axis = hist.axis.StrCategory(labels, name="cut")
    bucket = {}
    for name in ("onecut", "cumulative"):
        bucket[name] = hist.Hist(axis, storage=hist.storage.Weight())
        bucket[f"{name}_unweighted"] = hist.Hist(axis, storage=hist.storage.Weight())

    event_weights = cutflow_weight(weights)
    included = (np.ones(len(event_weights), dtype=bool) if start is None
                else np.asarray(start, dtype=bool))
    cumulative = included.copy()

    def record(name, index, mask, values=event_weights):
        selected = values[mask]
        view = bucket[name].view()
        view.value[index] = np.sum(selected)
        view.variance[index] = np.sum(selected**2)
        count = np.count_nonzero(mask)
        unweighted = bucket[f"{name}_unweighted"].view()
        unweighted.value[index] = count
        unweighted.variance[index] = count

    record("onecut", 0, cumulative)
    record("cumulative", 0, cumulative)
    for index, step in enumerate(steps, 1):
        mask = selections.all(step)
        cumulative &= mask
        record("onecut", index, mask & included)
        record("cumulative", index, cumulative)
    if sf_weights is not None:
        sf = np.asarray(sf_weights, dtype=np.float64)
        record("onecut", len(steps) + 1, cumulative, sf)
        record("cumulative", len(steps) + 1, cumulative, sf)
    return bucket


def _chain_histograms(selections, base, kind, flavor, steps, *, blind, sf_factors, channel_masks=None):
    """The cutflow histograms of one chain: blinded data stops before its mll cut; a complete chain ends with
    the CUTFLOW_SF_ROW row (base x the factor of *sf_factors* for *flavor*, 1 when none is given); with a
    mask in *channel_masks* for *flavor*, only those events enter the chain."""
    full = list(steps)
    if blind:
        steps = blind_cutflow_steps(kind, flavor, full)
    sf_weights = None
    if list(steps) == full:
        factor = (sf_factors or {}).get(flavor)
        sf_weights = base * (np.asarray(factor, dtype=np.float64) if factor is not None else 1.0)
    start = (channel_masks or {}).get(flavor)
    return _cutflow_histograms(selections, base, steps, sf_weights, start)


def fill_cutflows(output, selections, weights, *, blind=False, sf_factors=None, channel_masks=None):
    """Build cumulative cutflows for ee, mumu, and em channels.

    Order: jets (pT/eta, then ID), leptons (pT/eta, then pT/eta and ID together,
    i.e. tight, then no third tight lepton), leading pT, trigger, dR, mlljj,
    mll, then CUTFLOW_SF_ROW.  The object-level rows are "at least" counts; the
    no-third-tight-lepton row pins the tight leptons to exactly two, so the last
    cut equals the region selection.  Extra non-tight leptons are allowed.

    ``no_cuts`` starts after upstream event filters (and any input skim).
    Weighted bins use the cutflow weight (:func:`cutflow_weight`: the base
    event weight without the DY Z-pT reweight), before per-region lepton SFs;
    a complete chain ends with CUTFLOW_SF_ROW, which adds *sf_factors[flavor]*
    (the lepton and LSF SFs of the chain's region, CUTFLOW_FINAL_REGIONS), so it
    equals that region's yield without the DY corrections.

    *channel_masks* (signal): ``{"ee": mask, "mumu": mask}`` of the events of
    each generator flavour; those chains then count only their flavour, so a
    file holding both flavour samples of a mass point gives single-flavour
    cutflows (the em chain keeps every event: both flavours reach the CR).

    With *blind* (data of a blinded run) the ee and mumu chains stop before
    the mll cut (``wrcoffea.regions.BLINDED_CUTFLOW_STEPS``); em is complete.

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
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_TWO_PTETA_ELECTRONS,
            SEL_TWO_ID_ELECTRONS,
            SEL_NO_THIRD_TIGHT_LEPTON,
            SEL_LEAD_TIGHT_PT60,
            SEL_E_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_HEM_VETO,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT200,
            SEL_MLL_GT400,
        ],
        "mumu": [
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_TWO_PTETA_MUONS,
            SEL_TWO_ID_MUONS,
            SEL_NO_THIRD_TIGHT_LEPTON,
            SEL_LEAD_TIGHT_PT60,
            SEL_MU_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_HEM_VETO,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT200,
            SEL_MLL_GT400,
        ],
        "em": [
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_TWO_PTETA_EM,
            SEL_TWO_ID_EM,
            SEL_NO_THIRD_TIGHT_LEPTON,
            SEL_LEAD_TIGHT_PT60,
            SEL_EMU_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_HEM_VETO,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT200,
            SEL_MLL_GT400,
        ],
    }

    # --- Per-flavor: multi-bin onecut / cumulative histograms
    base = cutflow_weight(weights)
    for flavor, steps in chains.items():
        output["cutflow"].setdefault(flavor, {}).update(
            _chain_histograms(selections, base, "resolved", flavor, steps, blind=blind, sf_factors=sf_factors,
                              channel_masks=channel_masks)
        )


def fill_boosted_cutflows(output, selections, weights, *, blind=False, sf_factors=None, channel_masks=None):
    """Build cumulative cutflows for boosted ee, mumu, and em channels.

    Similar to fill_cutflows but for boosted topology. Shows progression
    through boosted-specific selections (SR progression for ee/mumu, the
    e(mu-jet) flavor CR for em and the mu(e-jet) one for me). As in the resolved cutflow, ``no_cuts`` follows
    upstream filters, the weighted bins use the cutflow weight and a complete
    chain ends with CUTFLOW_SF_ROW (*sf_factors*: the lepton and LSF SFs of the
    chain's region).  With *blind* the ee and mumu chains stop before the
    boosted mll cut.

    Output layout (keys under ``output["cutflow_boosted"]``):
        - per-flavor: ``ee``, ``mumu``, ``em``, ``me``
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
            SEL_HEM_VETO_BOOSTED_SR,
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
            SEL_HEM_VETO_BOOSTED_SR,
            SEL_MLL_GT200_BOOSTED,
            SEL_MLJ_GT800_BOOSTED,
        ],
        # e(mu-jet) flavour CR: leading electron, electron trigger
        "em": [
            SEL_BOOSTEDTAG,
            SEL_LEAD_IS_ELECTRON,
            SEL_LEAD_TIGHT_PT60_BOOSTED,
            SEL_E_TRIGGER,
            SEL_NO_DY_PAIR,
            SEL_AK8JETS_WITH_LSF,
            SEL_NO_EXTRA_TIGHT_CR,
            SEL_NO_SF_LEPTON_IN_AK8,
            SEL_OF_LEPTON_IN_AK8,
            SEL_HEM_VETO_BOOSTED_CR,
            SEL_MLL_GT200_BOOSTED,
            SEL_MLJ_GT800_BOOSTED,
        ],
        # mu(e-jet) flavour CR: leading muon, muon trigger (since 2026-10-14)
        "me": [
            SEL_BOOSTEDTAG,
            SEL_LEAD_IS_MUON,
            SEL_LEAD_TIGHT_PT60_BOOSTED,
            SEL_MU_TRIGGER,
            SEL_NO_DY_PAIR,
            SEL_AK8JETS_WITH_LSF,
            SEL_NO_EXTRA_TIGHT_CR,
            SEL_NO_SF_LEPTON_IN_AK8,
            SEL_OF_LEPTON_IN_AK8,
            SEL_HEM_VETO_BOOSTED_CR,
            SEL_MLL_GT200_BOOSTED,
            SEL_MLJ_GT800_BOOSTED,
        ],
    }

    # Generate cutflow histograms per flavor
    base = cutflow_weight(weights)
    for flavor, steps in chains.items():
        output["cutflow_boosted"].setdefault(flavor, {}).update(
            _chain_histograms(selections, base, "boosted", flavor, steps, blind=blind, sf_factors=sf_factors,
                              channel_masks=channel_masks)
        )
