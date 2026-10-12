"""Scale factor evaluation for muon and electron corrections.

Uses correctionlib CorrectionSet payloads. Caches per worker process
to avoid re-reading JSON every chunk.
"""

import logging

import awkward as ak
import numpy as np
from functools import lru_cache

from wrcoffea.analysis_config import (
    AK8_MSOFTDROP_MODES, CUTS, ELECTRON_HLT_ERA_KEYS, ELECTRON_HLT_PATHS, ELECTRON_JSONS, ELECTRON_RECO_CONFIG,
    ELECTRON_SF_ERA_KEYS, LSF_SF,
    JETVETO_CORRECTION_NAMES, JETVETO_JSONS, JETVETO_MODES, MUON_HIGHPT, MUON_HLT_CORRECTIONS, MUON_JSONS,
    PILEUP_CORRECTION_NAMES, PILEUP_JSONS, JERC_JSONS
)
import vector
vector.register_awkward()
logger = logging.getLogger(__name__)
from wrcoffea.MuonScaRe import pt_resol, pt_scale, pt_resol_var, pt_scale_var
# Cache correctionlib payloads per worker process (avoid re-reading JSON every chunk).
_CORRECTIONSET_CACHE = {}

# Warn-once cache (per worker process) to avoid log spam.
_WARN_ONCE: set[str] = set()


def _has_correction(cset, name):
    """Return True if *name* is a correction in the correctionlib CorrectionSet.

    ``CorrectionSet.__getitem__`` raises ``IndexError`` (not ``KeyError``) for
    an unknown name, which breaks the Mapping ``in`` operator and also
    ``name in cset.keys()`` since the KeysView delegates to it.  Iterating the
    set only touches the names, so that is the safe membership test.
    """
    return name in set(cset)


def _unflatten_and_product(flat_nom, flat_up, flat_down, counts):
    """Unflatten per-object SFs, take product over objects per event.

    Parameters
    ----------
    flat_nom, flat_up, flat_down : numpy arrays
        Flat per-object SF values (nominal / up / down).
    counts : awkward array
        Number of objects per event (from ``ak.num``).

    Returns
    -------
    (event_nom, event_up, event_down) : tuple of numpy float64 arrays
        Per-event SF products. Events with no objects get SF = 1.0.
    """
    nom = np.asarray(ak.fill_none(ak.prod(ak.unflatten(flat_nom, counts), axis=1), 1.0), dtype=np.float64)
    up = np.asarray(ak.fill_none(ak.prod(ak.unflatten(flat_up, counts), axis=1), 1.0), dtype=np.float64)
    down = np.asarray(ak.fill_none(ak.prod(ak.unflatten(flat_down, counts), axis=1), 1.0), dtype=np.float64)
    return nom, up, down


def pileup_weight(events, era):
    """Compute per-event pileup reweighting using correctionlib.

    Evaluates the pileup weight correction on ``Pileup.nTrueInt``.
    Returns (nominal, up, down) arrays of shape (n_events,).
    """
    n_events = len(events)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in PILEUP_JSONS:
        key = f"pileup_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No pileup JSON configured for era '%s'; using weight=1.", era)
        return ones, ones.copy(), ones.copy()

    import correctionlib

    json_path = PILEUP_JSONS[era]
    ceval = _CORRECTIONSET_CACHE.get(json_path)
    if ceval is None:
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        _CORRECTIONSET_CACHE[json_path] = ceval

    corr_name = PILEUP_CORRECTION_NAMES[era]
    corr = ceval[corr_name]

    nTrueInt = np.asarray(events.Pileup.nTrueInt, dtype=np.float64)

    nom = corr.evaluate(nTrueInt, "nominal")
    up = corr.evaluate(nTrueInt, "up")
    down = corr.evaluate(nTrueInt, "down")

    return nom, up, down


#: How the JME jet veto map is applied (JME "Jet Veto Map Recommendations").
#: Run 3 ("event", the default): reject the event if any jet passing the minimal
#: selection lies in a vetoed (eta, phi) cell.  Run 2 ("jet"): drop only the jets
#: whose axis lies in a vetoed cell and keep the event.  The two runs differ on
#: purpose; ``jetveto_mode`` in config.yaml selects the mode per era.
JETVETO_MODE_CHOICES = ("event", "jet")


def jet_veto_mode(era):
    """Return the jet veto application mode for *era*: ``"event"`` or ``"jet"``."""
    mode = JETVETO_MODES.get(era, "event")
    if mode not in JETVETO_MODE_CHOICES:
        raise ValueError(
            f"jetveto_mode for era '{era}' is {mode!r}; expected one of {JETVETO_MODE_CHOICES}"
        )
    return mode


def _jet_veto_hits(events, ak4_id_mask, era):
    """Per-jet boolean (aligned with ``events.Jet``): passes the minimal selection
    and lies in a vetoed cell.  Returns None when no map is configured for *era*.

    Minimal selection: pT > ``jet_veto_pt_min``, (chEmEF + neEmEF) <
    ``jet_veto_em_neFrac`` and the AK4 ID in *ak4_id_mask* (TightLeptonVeto).
    """
    if era not in JETVETO_JSONS:
        key = f"jetveto_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No jet veto map configured for era '%s'; passing all events.", era)
        return None

    import correctionlib

    json_path = JETVETO_JSONS[era]
    ceval = _CORRECTIONSET_CACHE.get(json_path)
    if ceval is None:
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        _CORRECTIONSET_CACHE[json_path] = ceval

    corr = ceval[JETVETO_CORRECTION_NAMES[era]]
    jets = events.Jet
    pt_mask = jets.pt > CUTS["jet_veto_pt_min"]
    jet_emf_mask = (jets.chEmEF + jets.neEmEF < CUTS["jet_veto_em_neFrac"])
    jet_mask = jet_emf_mask & pt_mask & ak4_id_mask

    # Evaluate the map only on jets passing the minimal selection (non-zero →
    # vetoed region) and scatter the result back onto all jets.
    counts = ak.num(jets)
    flat_sel = np.asarray(ak.flatten(jet_mask), dtype=bool)
    flat_hits = np.zeros(len(flat_sel), dtype=bool)
    if flat_sel.any():
        flat_eta = np.asarray(ak.flatten(jets.eta), dtype=np.float64)[flat_sel]
        flat_phi = np.asarray(ak.flatten(jets.phi), dtype=np.float64)[flat_sel]
        flat_hits[flat_sel] = corr.evaluate("jetvetomap", flat_eta, flat_phi) > 0
    return ak.unflatten(flat_hits, counts)


def jet_veto_event_mask(events, ak4_id_mask, era):
    """Return a per-event boolean mask (True = pass) using JME jet veto maps.

    Run 3 rule: events containing at least one jet that passes the minimal
    selection (see ``_jet_veto_hits``) in a vetoed (eta, phi) region are
    flagged False.  Applies to both data and MC.  Always returns a NumPy
    boolean array of shape ``(len(events),)``; the caller filters events.
    """
    n_events = len(events)
    if n_events == 0:
        return np.ones(0, dtype=bool)

    hits = _jet_veto_hits(events, ak4_id_mask, era)
    if hits is None:
        return np.ones(n_events, dtype=bool)
    return ak.to_numpy(~ak.any(hits, axis=1))


def jet_veto_jet_mask(events, ak4_id_mask, era):
    """Return a per-jet boolean mask (True = keep), aligned with ``events.Jet``.

    Run 2 rule: drop the jets that pass the minimal selection and lie in a
    vetoed (eta, phi) region; the event itself is kept.  As in the JME Run 2
    loose selection, jets with a PF muon within dR < 0.2 are not vetoed.  The
    PU jet ID requirement for CHS jets below 50 GeV is not applied: it only
    concerns jets below the 40 GeV AK4 threshold or at its edge.
    """
    hits = _jet_veto_hits(events, ak4_id_mask, era)
    if hits is None:
        return ak.ones_like(events.Jet.pt, dtype=bool)
    if "Muon" in events.fields and len(events) > 0:
        muons = events.Muon
        if "isPFcand" in muons.fields:
            muons = muons[muons.isPFcand]
        pairs = ak.cartesian([events.Jet, muons], axis=1, nested=True)
        jet_part, mu_part = ak.unzip(pairs)
        deta = jet_part.eta - mu_part.eta
        dphi = (jet_part.phi - mu_part.phi + np.pi) % (2 * np.pi) - np.pi
        near_pf_muon = ak.fill_none(ak.any(np.hypot(deta, dphi) < 0.2, axis=2), False)
        hits = hits & ~near_pf_muon
    return ~hits


def _get_muon_ceval(era,key):
    """Load (and cache) a correctionlib CorrectionSet for the muon SF file."""
    import correctionlib

    json_path = MUON_JSONS[era][key]
    ceval = _CORRECTIONSET_CACHE.get(json_path)
    if ceval is None:
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        _CORRECTIONSET_CACHE[json_path] = ceval
    return ceval


def muon_sf(tight_muons, era, is_loose=False):
    """Compute per-event muon RECO, ID, and ISO scale factors independently.

    Returns a dict of three independent components, each a (nominal, up, down) tuple
    of arrays with shape (n_events,). Register each component as a separate weight
    in the Coffea Weights object so that variations are handled independently.

    Events with no tight muons get SF = 1.0 for all components.
    """
    n_events = len(tight_muons)
    ones = np.ones(n_events, dtype=np.float64)
    identity = (ones, ones.copy(), ones.copy())

    if era not in MUON_JSONS:
        key = f"muon_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No muon SF JSON configured for era '%s'; using SF=1.", era)
        return {"reco": identity, "id": identity, "iso": identity}

    ceval = _get_muon_ceval(era,"RECO")
    
    # Flatten muon arrays for correctionlib evaluation.
    counts = ak.num(tight_muons)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_muons.eta, 0.0)), dtype=np.float64)
    flat_pt = np.asarray(ak.flatten(ak.fill_none(tight_muons.pt, 0.0)), dtype=np.float64)

    if len(flat_pt) == 0:
        return {"reco": identity, "id": identity, "iso": identity}

    # Compute momentum p = pt * cosh(eta) for RECO.
    flat_p = flat_pt * np.cosh(flat_eta)

    # Clip inputs to valid bin edges; eta in each correction's convention (_muon_eta_input).
    reco_p = np.clip(flat_p, 50.001, 1e9)
    idiso_pt = np.clip(flat_pt, 50.001, 1e9)

    def _eval_component(corr, pt_or_p):
        eta = _muon_eta_input(corr, flat_eta)
        nom = corr.evaluate(eta, pt_or_p, "nominal")
        up = corr.evaluate(eta, pt_or_p, "systup")
        down = corr.evaluate(eta, pt_or_p, "systdown")
        return _unflatten_and_product(nom, up, down, counts)

    # RECO SF (uses momentum p, not pT)
    reco = _eval_component(ceval["NUM_GlobalMuons_DEN_TrackerMuonProbes"], reco_p)
    # ID SF
    id_sf = _eval_component(ceval["NUM_HighPtID_DEN_GlobalMuonProbes"], idiso_pt)
    # ISO SF
    if is_loose:
        # Loose muons carry no ISO requirement: per-event identity SF.
        iso = (np.ones(n_events), np.ones(n_events), np.ones(n_events))
    else:
        iso = _eval_component(ceval["NUM_probe_LooseRelTkIso_DEN_HighPtProbes"], idiso_pt)

    return {"reco": reco, "id": id_sf, "iso": iso}


def muon_sf_loose(loose_muons, era):
    """Compute per-event muon RECO and ID SFs for loose muons (no ISO)."""
    n_events = len(loose_muons)
    ones = np.ones(n_events, dtype=np.float64)
    identity = (ones, ones.copy(), ones.copy())

    if era not in MUON_JSONS:
        return {"reco": identity, "id": identity}

    ceval = _get_muon_ceval(era, "RECO")

    counts = ak.num(loose_muons)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(loose_muons.eta, 0.0)), dtype=np.float64)
    flat_pt = np.asarray(ak.flatten(ak.fill_none(loose_muons.pt, 0.0)), dtype=np.float64)

    if len(flat_pt) == 0:
        return {"reco": identity, "id": identity}

    flat_p = flat_pt * np.cosh(flat_eta)

    reco_p = np.clip(flat_p, 50.001, 1e9)
    idiso_pt = np.clip(flat_pt, 50.001, 1e9)

    def _eval_component(corr, pt_or_p):
        eta = _muon_eta_input(corr, flat_eta)
        nom = corr.evaluate(eta, pt_or_p, "nominal")
        up = corr.evaluate(eta, pt_or_p, "systup")
        down = corr.evaluate(eta, pt_or_p, "systdown")
        return _unflatten_and_product(nom, up, down, counts)

    reco = _eval_component(ceval["NUM_GlobalMuons_DEN_TrackerMuonProbes"], reco_p)
    id_sf = _eval_component(ceval["NUM_HighPtID_DEN_GlobalMuonProbes"], idiso_pt)

    return {"reco": reco, "id": id_sf}


_MUON_HLT_CORRECTION = "NUM_HLT_DEN_HighPtLooseRelIsoProbes"


def _muon_eta_input(corr, flat_eta):
    """Eta in the convention a ``muon_HighPt`` reco/ID/iso correction is binned in: signed ``eta``
    (UL, 2022 to 2024) or ``abseta`` (the 2025 payload), inside the |eta| < 2.4 acceptance."""
    inputs = getattr(corr, "inputs", None) or []
    first_input = getattr(inputs[0], "name", "eta") if len(inputs) else "eta"
    if first_input == "abseta":
        return np.clip(np.abs(flat_eta), 0.0, 2.399)
    return np.clip(flat_eta, -2.399, 2.399)


def _muon_hlt_eta_input(corr, flat_eta):
    """Return eta in the convention the HLT correction is binned in.

    The per-era high-pT HLT payloads (schemaV2) take ``abseta``; the POG
    ``muon_HighPt`` payloads take signed ``eta``.
    """
    inputs = getattr(corr, "inputs", None) or []
    first_input = getattr(inputs[0], "name", "abseta") if len(inputs) else "abseta"
    if first_input == "abseta":
        return np.clip(np.abs(flat_eta), 0.0, 2.4)
    return np.clip(flat_eta, -2.399, 2.399)


def _muon_trigger_sf_from_efficiencies(data_eff_corr, mc_eff_corr, flat_eta, hlt_pt, counts):
    """Event trigger SF from per-lepton data/MC efficiencies.

    Implements the dilepton OR formula ``e(l1,l2) = 1 - (1-e(l1))(1-e(l2))``
    separately for data and MC, then ``SF = e_data / e_MC``.
    """
    hlt_eta = _muon_hlt_eta_input(data_eff_corr, flat_eta)

    def _event_eff(corr, variation):
        per_lepton = ak.unflatten(corr.evaluate(hlt_eta, hlt_pt, variation), counts)
        return 1.0 - ak.prod(1.0 - per_lepton, axis=1)

    def _safe_ratio(num, den):
        num_np = np.asarray(ak.fill_none(num, 0.0), dtype=np.float64)
        den_np = np.asarray(ak.fill_none(den, 0.0), dtype=np.float64)
        out = np.ones_like(num_np)
        np.divide(num_np, den_np, out=out, where=den_np > 0)
        return out

    # Variation as the Run 2 analysis (SKFlat MCCorrection::MuonTrigger_SF): the data efficiency moved up
    # while the MC efficiency moves down, and the reverse, so the two errors add in the ratio.  Moving both
    # the same way (as this code did until 2026-10-12) cancels them and left the variation at ~0.01 %.
    nom = _safe_ratio(_event_eff(data_eff_corr, "nominal"), _event_eff(mc_eff_corr, "nominal"))
    up = _safe_ratio(_event_eff(data_eff_corr, "systup"), _event_eff(mc_eff_corr, "systdown"))
    down = _safe_ratio(_event_eff(data_eff_corr, "systdown"), _event_eff(mc_eff_corr, "systup"))
    return nom, up, down


def _muon_trigger_sf_from_leading_sf(sf_corr, pt_jagged, flat_eta, hlt_pt, counts):
    """Event trigger SF from a per-lepton SF payload (no efficiencies).

    The single-muon trigger OR is dominated by the leading muon, and without
    data/MC efficiencies the two legs cannot be combined properly, so the SF
    of the leading-pT tight muon is applied once per event.
    """
    hlt_eta = _muon_hlt_eta_input(sf_corr, flat_eta)
    lead = ak.argmax(pt_jagged, axis=1, keepdims=True)

    def _leading(variation):
        per_lepton = ak.unflatten(sf_corr.evaluate(hlt_eta, hlt_pt, variation), counts)
        return np.asarray(ak.fill_none(ak.firsts(per_lepton[lead]), 1.0), dtype=np.float64)

    return _leading("nominal"), _leading("systup"), _leading("systdown")


@lru_cache(maxsize=None)
def _correction_valtypes(path: str, name: str) -> frozenset:
    """The keys of the ``scale_factors`` category of correction *name* in the JSON at *path* (empty if none)."""
    import gzip
    import json as _json
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        doc = _json.load(fh)
    for c in doc.get("corrections", []):
        if c["name"] != name:
            continue
        keys: set = set()

        def walk(node):
            if isinstance(node, dict):
                if node.get("nodetype") == "category":
                    if node["input"] == "scale_factors":
                        keys.update(k["key"] for k in node["content"])
                    for k in node["content"]:
                        walk(k["value"])
                elif node.get("nodetype") in ("binning", "multibinning"):
                    content = node["content"]
                    for ch in (content if isinstance(content, list) else [content]):
                        walk(ch)
                elif node.get("nodetype") == "transform":
                    walk(node["content"])

        walk(c["data"])
        return frozenset(keys)
    return frozenset()


# The tag-and-probe fit variations a muon_Z correction stores at SF level; with the SF's stat, in quadrature,
# they make the POG's per-muon SF uncertainty.
_MUON_Z_FIT_KEYS = ("AltBkg", "AltSig", "massBin", "massRange", "tagIso")


def _muon_trigger_sf_from_valtype_efficiencies(corr, flat_eta, hlt_pt, counts, valtypes=frozenset()):
    """Event trigger SF from one POG ``muon_Z`` correction that holds the data and MC efficiencies of the
    trigger OR as ``scale_factors`` values (``nominal_DATAeff``, ``nominal_MCeff``; CAT muon_Z of 2026-06-18,
    2022 and 2023).  Same dilepton OR formula as the two-correction form.

    Variation (since 2026-10-12): each muon's data efficiency moves by the relative uncertainty of the POG
    per-muon SF, its ``stat`` and the tag-and-probe fit variations (``_MUON_Z_FIT_KEYS``) in quadrature, with
    the MC efficiency kept nominal; the OR then dilutes it per event.  The efficiency-level errors are not used:
    ``syst_DATAeff`` / ``syst_MCeff`` are zero in these payloads, and the SF-level ``syst`` key is not always
    the quadrature of its own components (zero in some bins where they are not)."""
    hlt_eta = _muon_hlt_eta_input(corr, flat_eta)
    pt = np.clip(hlt_pt, 52.001, 1e9)          # the muon_Z HLT tables start at 52 GeV

    sf_nom = corr.evaluate(hlt_eta, pt, "nominal")
    err2 = corr.evaluate(hlt_eta, pt, "stat") ** 2
    for key in _MUON_Z_FIT_KEYS:
        if key in valtypes:
            err2 = err2 + corr.evaluate(hlt_eta, pt, key) ** 2
    rel_err = np.divide(np.sqrt(err2), sf_nom, out=np.zeros_like(sf_nom, dtype=np.float64), where=sf_nom > 0)

    def _per_lepton(kind, sign=0):
        eff = corr.evaluate(hlt_eta, pt, f"nominal_{kind}eff")
        if sign:
            eff = np.clip(eff * (1.0 + sign * rel_err), 0.0, 1.0)
        return ak.unflatten(eff, counts)

    def _event_eff(kind, sign=0):
        return 1.0 - ak.prod(1.0 - _per_lepton(kind, sign), axis=1)

    def _safe_ratio(num, den):
        num_np = np.asarray(ak.fill_none(num, 0.0), dtype=np.float64)
        den_np = np.asarray(ak.fill_none(den, 0.0), dtype=np.float64)
        out = np.ones_like(num_np)
        np.divide(num_np, den_np, out=out, where=den_np > 0)
        return out

    eff_mc = _event_eff("MC")
    nom = _safe_ratio(_event_eff("DATA"), eff_mc)
    up = _safe_ratio(_event_eff("DATA", +1), eff_mc)
    down = _safe_ratio(_event_eff("DATA", -1), eff_mc)
    return nom, up, down


def muon_trigger_sf(tight_muons, era):
    """Compute per-event muon trigger SF.

    Three payload shapes are supported; the treatment is chosen per era from
    what the configured ``TRIGGER`` JSON contains (``config.yaml``
    ``muon_hlt_correction`` names the correction, default
    ``NUM_HLT_DEN_HighPtLooseRelIsoProbes``):

    1. Data and MC efficiencies as separate corrections (``<name>_DATAeff`` /
       ``<name>_MCeff``, the 2024 and 2025 high-pT HLT payloads): the event
       efficiency is built with the dilepton OR formula
       ``e(l1,l2) = 1 - (1-e(l1))(1-e(l2))`` and ``SF = e_data / e_MC``.
    2. Data and MC efficiencies as ``scale_factors`` values of one correction
       (the POG ``muon_Z`` trigger-OR entries of 2022-2025,
       ``NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose``):
       the same OR formula.
    3. Per-lepton scale factors only (the Run 2 UL ``muon_HighPt`` files and
       the 2022/2023 high-pT HLT payloads): the SF of the leading-pT tight
       muon is applied once per event, as Run 2 did.

    The up/down variation of forms 1 and 2 follows the Run 2 analysis
    (SKFlat ``MCCorrection::MuonTrigger_SF``, AN-19-083 Sec. 2.4.1): every
    muon's data efficiency is moved up by its error while its MC efficiency is
    moved down, and the reverse for "down", so the two errors add in the
    ratio.  Form 3 uses the payload's own systup/systdown of the leading muon.
    Decided 2026-10-12 (the same-direction shift used before cancelled the
    errors and left a ~0.01 % variation).

    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight muons get SF = 1.0.
    """

    n_events = len(tight_muons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in MUON_JSONS or "TRIGGER" not in MUON_JSONS.get(era, {}):
        key = f"muon_trig_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No muon trigger SF JSON configured for era '%s'; using SF=1.", era)
        return ones, ones.copy(), ones.copy()

    ceval = _get_muon_ceval(era, "TRIGGER")

    counts = ak.num(tight_muons)
    flat_pt = np.asarray(ak.flatten(ak.fill_none(tight_muons.pt, 0.0)), dtype=np.float64)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_muons.eta, 0.0)), dtype=np.float64)

    if len(flat_pt) == 0:
        return ones, ones.copy(), ones.copy()

    hlt_pt = np.clip(flat_pt, 50.001, 1e9)
    corr_name = MUON_HLT_CORRECTIONS.get(era, _MUON_HLT_CORRECTION)

    data_key = f"{corr_name}_DATAeff"
    mc_key = f"{corr_name}_MCeff"
    if _has_correction(ceval, data_key) and _has_correction(ceval, mc_key):
        return _muon_trigger_sf_from_efficiencies(
            ceval[data_key], ceval[mc_key], flat_eta, hlt_pt, counts
        )

    valtypes = _correction_valtypes(str(MUON_JSONS[era]["TRIGGER"]), corr_name)
    if {"nominal", "stat", "nominal_DATAeff", "nominal_MCeff"} <= valtypes:
        return _muon_trigger_sf_from_valtype_efficiencies(ceval[corr_name], flat_eta, hlt_pt, counts, valtypes)

    key = f"muon_trig_sf_only::{era}"
    if key not in _WARN_ONCE:
        _WARN_ONCE.add(key)
        logger.info(
            "Muon trigger payload for era '%s' has no data/MC efficiencies; "
            "applying the leading-muon per-lepton SF.", era
        )
    return _muon_trigger_sf_from_leading_sf(
        ceval[corr_name], tight_muons.pt, flat_eta, hlt_pt, counts
    )

logger = logging.getLogger(__name__)

# Assuming you have a _get_muon_ceval or use your existing _get_electron_ceval equivalent
def apply_muon_scale_smearing(events, era, is_mc):
    """
    Apply Scale and Smearing (Resolution) corrections to muons.
    Uses the MuonScaRe wrapper for nested awkward arrays.
    
    Args:
        events: Awkward array of events (must contain .Muon, .event, and .luminosityBlock).
        era: The dataset era (e.g., 'RunIII2024Summer24').
        is_mc: Boolean indicating if the sample is Monte Carlo.
        
    Returns:
        A dictionary containing the corrected awkward arrays for 'pt',
        and (if MC) systematic variations.
    """
    if era in MUON_HIGHPT:
        # Muon POG Run 3 recipe: MuonScaRe below 200 GeV, GE + MC smearing above.
        from wrcoffea.muon_momentum import correct_muons
        return correct_muons(events, era, is_mc)

    muons = events.Muon
    counts = ak.num(muons)
    
    # Early exit if there are no muons in this chunk
    if ak.sum(counts) == 0:
        return {}

    result = {}

    # ---------------------------------------------------------
    # HIGH-pT REGIME: Use TuneP momentum if available
    # ---------------------------------------------------------
    if "tunepRelPt" in muons.fields:
        # tunepRelPt is tunePpt / pt. So tuneP_pt = pt * tunepRelPt
        tuneP_pt = muons.pt * muons.tunepRelPt
        result["pt"] = tuneP_pt
        if is_mc:
            # -----------------------------------------------------
            # High-pT Systematics (Sagitta Bias / Momentum Scale)
            # -----------------------------------------------------
            # Replace '0.05' with the exact Run-3 POG recommendation (e.g., 5% per TeV)
            # Formula: uncertainty = pT * (fractional error per TeV) * (pT in TeV) -
            ## ---- only a pplace holder - to be checked when we do systematics)
            fractional_error_per_tev = 0.05 
            tuneP_unc = tuneP_pt*fractional_error_per_tev * (tuneP_pt / 1000.0)
            
            result["pt_scale_up"] = tuneP_pt + tuneP_unc
            result["pt_scale_down"] = tuneP_pt - tuneP_unc
            
            # High-pT TuneP generally does not use a separate stochastic resolution 
            # smearing envelope like the Z-peak method does. 
            # We map the smear variations to the nominal TuneP pT so downstream 
            # code expecting these dictionary keys doesn't crash.
            result["pt_smear_up"] = tuneP_pt
            result["pt_smear_down"] = tuneP_pt
        return result
    ##  --- place holder for recchestor corrections but might not be needed)
    # Load evaluator using your existing caching function (adapt the key as needed)
    try:
        cset = _get_muon_ceval(era, "ScaleNSmear") 
    except KeyError:
        logger.warning(f"No Muon ScaleNSmear JSON configured for era '{era}'; skipping correction.")
        return {}

    if not is_mc:
        # -------------------
        # DATA PROCEDURE
        # -------------------
        # Data only gets the scale correction to the Z peak
        corrected_pt = pt_scale(
            1, # 1 designates Data
            muons.pt, 
            muons.eta, 
            muons.phi, 
            muons.charge, 
            cset, 
            nested=True
        )
        result["pt"] = corrected_pt

    else:
        # -------------------
        # MC PROCEDURE
        # -------------------
        # 1. Scale correction to MC
        ptscalecorr = pt_scale(
            0, # 0 designates MC
            muons.pt, 
            muons.eta, 
            muons.phi, 
            muons.charge, 
            cset, 
            nested=True
        )
        
        # 2. Resolution (Smearing) correction applied on top of the scaled pT
        corrected_pt = pt_resol(
            ptscalecorr, 
            muons.eta, 
            muons.phi, 
            muons.nTrackerLayers, 
            events.event, 
            events.luminosityBlock, 
            cset, 
            nested=True
        )
        
        result["pt"] = corrected_pt

        # 3. Systematics
        # Note: Scale variations take the fully corrected pT as input
        result["pt_scale_up"] = pt_scale_var(
            corrected_pt, muons.eta, muons.phi, muons.charge, "up", cset, nested=True
        )
        result["pt_scale_down"] = pt_scale_var(
            corrected_pt, muons.eta, muons.phi, muons.charge, "dn", cset, nested=True
        )
        
        # Note: Resolution variations take both the scaled-only and fully corrected pT
        result["pt_smear_up"] = pt_resol_var(
            ptscalecorr, corrected_pt, muons.eta, "up", cset, nested=True
        )
        result["pt_smear_down"] = pt_resol_var(
            ptscalecorr, corrected_pt, muons.eta, "dn", cset, nested=True
        )
    return result

def verify_muon_scale_smearing(original_muons, corrected_dict, is_mc):
    """
    Prints a statistical summary to verify Muon Scale and Smearing (Resolution).
    """
    # Handle empty chunks gracefully
    if len(corrected_dict) == 0 or ak.sum(ak.num(original_muons)) == 0:
        return

    # Flatten arrays to compute global statistics
    orig_pt = np.asarray(ak.flatten(original_muons.pt), dtype=np.float64)
    corr_pt = np.asarray(ak.flatten(corrected_dict["pt"]), dtype=np.float64)

    print("\n" + "="*70)
    print(f"--- MUON SCALE & SMEARING VERIFICATION ({'MC' if is_mc else 'DATA'}) ---")
    print("="*70)
    
    print(f"{'Variable':<20} | {'Original':<15} | {'Corrected':<15} | {'Change':<15}")
    print("-" * 70)

    if not is_mc:
        # DATA: Expect mean pT to shift slightly (Scale correction)
        pt_shift = np.mean(corr_pt) / np.mean(orig_pt) if np.mean(orig_pt) > 0 else 1.0
        print(f"{'Mean pT':<20} | {np.mean(orig_pt):<15.4f} | {np.mean(corr_pt):<15.4f} | {pt_shift:<10.4f} (ratio)")
        print(f"{'Std Dev pT':<20} | {np.std(orig_pt):<15.4f} | {np.std(corr_pt):<15.4f} |")
    
    else:
        # MC: Expect mean pT to shift (MC Scale) AND std dev to increase (Resolution Smearing)
        pt_diff = np.mean(corr_pt) - np.mean(orig_pt)
        print(f"{'Mean pT':<20} | {np.mean(orig_pt):<15.4f} | {np.mean(corr_pt):<15.4f} | {pt_diff:<10.4f} (diff)")
        print(f"{'Std Dev pT':<20} | {np.std(orig_pt):<15.4f} | {np.std(corr_pt):<15.4f} | Broadened")
        
        # Systematics verification
        print("\n--- MC Systematics Envelopes ---")
        pt_smear_up = np.asarray(ak.flatten(corrected_dict["pt_smear_up"]), dtype=np.float64)
        pt_smear_down = np.asarray(ak.flatten(corrected_dict["pt_smear_down"]), dtype=np.float64)
        pt_scale_up = np.asarray(ak.flatten(corrected_dict["pt_scale_up"]), dtype=np.float64)
        pt_scale_down = np.asarray(ak.flatten(corrected_dict["pt_scale_down"]), dtype=np.float64)
        
        print(f"{'Smearing Std Dev':<20} | Down: {np.std(pt_smear_down):<10.4f} | Nom: {np.std(corr_pt):<10.4f} | Up: {np.std(pt_smear_up):<10.4f}")
        print(f"{'Scale Mean pT':<20} | Down: {np.mean(pt_scale_down):<10.4f} | Nom: {np.mean(corr_pt):<10.4f} | Up: {np.mean(pt_scale_up):<10.4f}")

    print("="*70 + "\n")
    

def _get_electron_ceval(era, key):
    """Load (and cache) a correctionlib CorrectionSet for an electron SF file."""
    import correctionlib

    json_path = ELECTRON_JSONS[era][key]
    ceval = _CORRECTIONSET_CACHE.get(json_path)
    if ceval is None:
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        _CORRECTIONSET_CACHE[json_path] = ceval
    return ceval

def electron_trigger_sf(tight_electrons, era):
    """Compute per-event electron trigger SF using dilepton efficiency formula.

    Implements: e(l1,l2) = 1 - (1-e(l1))(1-e(l2))
    Then SF = e_data(event) / e_MC(event)

    This correctly accounts for the fact that we need at least one electron
    to fire the trigger (OR logic), not both (AND logic).

    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight electrons get SF = 1.0.

    Note: the Run 3 EGM HLT payloads only provide the single-electron
    ``HLT_SF_Ele30_TightID`` efficiencies, which are used as a proxy for the
    full Run 3 electron trigger OR (Ele30_WPTight || Photon200 || Ele115 ||
    Ele50_PFJet165).  UL18 uses the Run 2 WR measurement of the trigger OR on
    HEEP electrons (``WREGammaTrigger``).  The path name per era comes from
    ``electron_hlt_paths`` in ``config.yaml``.  Efficiencies are evaluated at
    supercluster eta.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in ELECTRON_JSONS or "TRIGGER" not in ELECTRON_JSONS.get(era, {}):
        key = f"electron_trig_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No electron trigger SF JSON configured for era '%s'; using SF=1.", era)
        return ones, ones.copy(), ones.copy()

    # The trigger payload's own era key: an era that borrows another era's file (2025) uses that key.
    sf_era_key = ELECTRON_HLT_ERA_KEYS.get(era, ELECTRON_SF_ERA_KEYS.get(era))
    if sf_era_key is None:
        logger.warning("No electron trigger SF era key for '%s'; returning SF=1.", era)
        return ones, ones.copy(), ones.copy()

    ceval = _get_electron_ceval(era, "TRIGGER")
    data_eff_corr = ceval["Electron-HLT-DataEff"]
    mc_eff_corr = ceval["Electron-HLT-McEff"]

    counts = ak.num(tight_electrons)
    flat_pt = np.asarray(ak.flatten(ak.fill_none(tight_electrons.pt, 0.0)), dtype=np.float64)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_electrons.eta, 0.0)), dtype=np.float64)

    if len(flat_pt) == 0:
        return ones, ones.copy(), ones.copy()

    # Supercluster eta = eta + deltaEtaSC (fall back to eta if the branch is absent).
    try:
        flat_deltaEtaSC = np.asarray(ak.flatten(ak.fill_none(tight_electrons.deltaEtaSC, 0.0)), dtype=np.float64)
    except AttributeError:
        key = f"missing_deltaEtaSC_trig::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning("Electron `deltaEtaSC` branch missing; using eta as fallback for trigger SF.")
        flat_deltaEtaSC = np.zeros_like(flat_eta)

    # Clip inputs to valid ranges.
    hlt_eta = np.clip(flat_eta + flat_deltaEtaSC, -2.499, 2.499)
    hlt_pt = np.clip(flat_pt, 30.001, 1e6)

    trigger_path = ELECTRON_HLT_PATHS.get(era)
    if trigger_path is None:
        raise KeyError(f"No electron_hlt_paths entry in config.yaml for era '{era}'")

    # Get per-electron efficiencies in data and MC.
    eff_data_nom = data_eff_corr.evaluate(sf_era_key, "nom", trigger_path, hlt_eta, hlt_pt)
    eff_data_up = data_eff_corr.evaluate(sf_era_key, "up", trigger_path, hlt_eta, hlt_pt)
    eff_data_down = data_eff_corr.evaluate(sf_era_key, "down", trigger_path, hlt_eta, hlt_pt)

    eff_mc_nom = mc_eff_corr.evaluate(sf_era_key, "nom", trigger_path, hlt_eta, hlt_pt)
    eff_mc_up = mc_eff_corr.evaluate(sf_era_key, "up", trigger_path, hlt_eta, hlt_pt)
    eff_mc_down = mc_eff_corr.evaluate(sf_era_key, "down", trigger_path, hlt_eta, hlt_pt)

    # Unflatten per-electron efficiencies.
    eff_data_nom_jagged = ak.unflatten(eff_data_nom, counts)
    eff_data_up_jagged = ak.unflatten(eff_data_up, counts)
    eff_data_down_jagged = ak.unflatten(eff_data_down, counts)

    eff_mc_nom_jagged = ak.unflatten(eff_mc_nom, counts)
    eff_mc_up_jagged = ak.unflatten(eff_mc_up, counts)
    eff_mc_down_jagged = ak.unflatten(eff_mc_down, counts)

    # Apply dilepton trigger efficiency formula: e(l1,l2) = 1 - (1-e1)(1-e2)
    # For events with 1 electron: e(event) = e1
    # For events with 2+ electrons: e(event) = 1 - (1-e1)(1-e2)...(1-en)
    # This is equivalent to: 1 - product(1 - ei)
    event_eff_data_nom = 1.0 - ak.prod(1.0 - eff_data_nom_jagged, axis=1)
    event_eff_data_up = 1.0 - ak.prod(1.0 - eff_data_up_jagged, axis=1)
    event_eff_data_down = 1.0 - ak.prod(1.0 - eff_data_down_jagged, axis=1)

    event_eff_mc_nom = 1.0 - ak.prod(1.0 - eff_mc_nom_jagged, axis=1)
    event_eff_mc_up = 1.0 - ak.prod(1.0 - eff_mc_up_jagged, axis=1)
    event_eff_mc_down = 1.0 - ak.prod(1.0 - eff_mc_down_jagged, axis=1)

    # Compute event-level SF = e_data(event) / e_MC(event).
    # For events with no electrons both efficiencies are 0 → 0/0;
    # use np.divide(where=) to avoid the RuntimeWarning and default to 1.
    def _safe_ratio(num, den):
        num_np = np.asarray(ak.fill_none(num, 0.0), dtype=np.float64)
        den_np = np.asarray(ak.fill_none(den, 0.0), dtype=np.float64)
        out = np.ones_like(num_np)
        np.divide(num_np, den_np, out=out, where=den_np > 0)
        return out

    trig_sf_nom = _safe_ratio(event_eff_data_nom, event_eff_mc_nom)
    trig_sf_up = _safe_ratio(event_eff_data_up, event_eff_mc_up)
    trig_sf_down = _safe_ratio(event_eff_data_down, event_eff_mc_down)

    return trig_sf_nom, trig_sf_up, trig_sf_down

# Flat HEEP ID scale factors, keyed by the EGM era key: (value, stat, syst) for the barrel (|eta_SC| < 1.4442) and
# the endcaps.  Used when the era's electron payload has no HEEPID working point (every Run 2 UL file and the
# 2022/2023 Run 3 files; 2024 has one); other eras fall back to 2018.
# - Run 2 UL: HEEP V7.0 (EGM twiki EgammaUL2016To2018, "HEEP ID Scale Factor for UL", UL2016 updated 2023-03-03; saved
#   as docs/pog_recommendations/egamma/CMS_EgammaUL2016To2018.pdf).  Both UL16 eras take "Full 2016".
# - 2022/2023: the central EGM values, constant in pT (EGM twiki EgammSFandSSRun3, "HEEP ID SFs for 2022" (pT > 80
#   GeV) and "... for 2023" (pT > 35 GeV), saved as docs/pog_recommendations/egamma/CMS_EgammSFandSSRun3.pdf pp. 7 and
#   14; the same values as AN-25-107 Table 6). The single quoted uncertainty is carried as syst.
HEEP_V7_FLAT_SF = {
    "2016preVFP": {"barrel": (0.985, 0.001, 0.004), "endcap": (0.990, 0.003, 0.007)},
    "2016postVFP": {"barrel": (0.985, 0.001, 0.004), "endcap": (0.990, 0.003, 0.007)},
    "2017": {"barrel": (0.979, 0.001, 0.005), "endcap": (0.987, 0.002, 0.010)},
    "2018": {"barrel": (0.973, 0.001, 0.004), "endcap": (0.980, 0.002, 0.011)},
    "2022Re-recoBCD": {"barrel": (1.007, 0.0, 0.009), "endcap": (1.019, 0.0, 0.015)},
    "2022Re-recoE+PromptFG": {"barrel": (0.997, 0.0, 0.008), "endcap": (0.997, 0.0, 0.013)},
    "2023PromptC": {"barrel": (1.006, 0.0, 0.004), "endcap": (0.988, 0.0, 0.005)},
    "2023PromptD": {"barrel": (1.009, 0.0, 0.005), "endcap": (0.988, 0.0, 0.004)},
}


def electron_id_sf(tight_electrons, era):
    """Compute per-event HEEP electron ID scale factor using JSON POG payloads.

    Extracts nominal, up, and down scale factors directly from the official EGamma 
    correctionlib JSON files configured in analysis_config.

    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight electrons get SF = 1.0.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    # 1. Check if the era is configured at all
    if era not in ELECTRON_JSONS:
        key = f"electron_id_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No electron ID SF JSON configured for era '%s'; using SF=1.", era)
        return ones, ones.copy(), ones.copy()

    counts = ak.num(tight_electrons)
    if ak.sum(counts) == 0:
        return ones, ones.copy(), ones.copy()

    # 2. Extract flat pT and eta
    flat_pt = np.asarray(ak.flatten(ak.fill_none(tight_electrons.pt, 0.0)), dtype=np.float64)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_electrons.eta, 0.0)), dtype=np.float64)

    # 3. Supercluster eta = eta + deltaEtaSC.
    try:
        flat_deltaEtaSC = np.asarray(ak.flatten(ak.fill_none(tight_electrons.deltaEtaSC, 0.0)), dtype=np.float64)
        flat_sc_eta = flat_eta + flat_deltaEtaSC
    except AttributeError:
        key = f"missing_deltaEtaSC_id::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Electron `deltaEtaSC` branch missing; using eta as fallback for HEEP ID SF."
            )
        flat_sc_eta = flat_eta

    # 4. Fetch lookup keys using your imported config dictionaries
    sf_era_key = ELECTRON_SF_ERA_KEYS.get(era)
    if sf_era_key is None:
        logger.warning("No electron ID SF era key for '%s'; returning SF=1.", era)
        return ones, ones.copy(), ones.copy()

    reco_cfg = ELECTRON_RECO_CONFIG.get(era)
    if reco_cfg is None:
        logger.warning("No ELECTRON_RECO_CONFIG for '%s'; returning SF=1.", era)
        return ones, ones.copy(), ones.copy()

    # The HEEP ID JSON bins start at 35.0 GeV and flow is set to "error".
    # Clip to prevent correctionlib from throwing out-of-bounds exceptions.
    # At the HIGH end the Run 3 payloads carry a placeholder bin above the
    # last measurement (HEEPID: pt > 1000 GeV -> SF = 1 +- 20%); clip to the
    # last measured bin (`id_pt_max` in electron_reco_config) so TeV
    # electrons take the last measured SF and uncertainty instead.
    id_pt_max = reco_cfg.get("id_pt_max")
    safe_pt = np.maximum(flat_pt, 35.01)
    if id_pt_max is not None:
        safe_pt = np.minimum(safe_pt, float(id_pt_max) - 0.01)

    correction_name = reco_cfg["correction"]
    
    # 5. Load ceval using the existing helper function in your file
    ceval = _get_electron_ceval(era, "RECO")
    
    if not _has_correction(ceval, correction_name):
        key = f"missing_correction_name::{era}_{correction_name}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.error("Correction '%s' not found in ceval for era %s. Using SF=1.", correction_name, era)
        return ones, ones.copy(), ones.copy()
        
    evaluator = ceval[correction_name]

    # 6. Evaluate SFs using the correctionlib signature: [year, ValType, WorkingPoint, eta, pt]
    try:
        sf_nom = evaluator.evaluate(sf_era_key, "sf", "HEEPID", flat_sc_eta, safe_pt)
        sf_up = evaluator.evaluate(sf_era_key, "sfup", "HEEPID", flat_sc_eta, safe_pt)
        sf_down = evaluator.evaluate(sf_era_key, "sfdown", "HEEPID", flat_sc_eta, safe_pt)
    except (IndexError, RuntimeError, ValueError):
        # JSON for this era has no HEEPID working point (e.g. Run2 UL and
        # pre-2024 Run3 payloads). Fall back to the flat UL2018 HEEP V7.0
        # barrel/endcap SFs (stat + syst in quadrature).
        key = f"heepid_wp_missing::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Electron JSON for era '%s' has no HEEPID working point; "
                "using the flat HEEP SFs of %s.", era, sf_era_key if sf_era_key in HEEP_V7_FLAT_SF else "2018"
            )
        abs_sc_eta = np.abs(flat_sc_eta)
        heep = HEEP_V7_FLAT_SF.get(sf_era_key, HEEP_V7_FLAT_SF["2018"])
        barrel_sf, barrel_unc = heep["barrel"][0], np.sqrt(heep["barrel"][1]**2 + heep["barrel"][2]**2)
        endcap_sf, endcap_unc = heep["endcap"][0], np.sqrt(heep["endcap"][1]**2 + heep["endcap"][2]**2)
        is_barrel = abs_sc_eta < 1.4442
        sf_nom = np.where(is_barrel, barrel_sf, endcap_sf)
        sf_unc = np.where(is_barrel, barrel_unc, endcap_unc)
        sf_up = sf_nom + sf_unc
        sf_down = sf_nom - sf_unc

    return _unflatten_and_product(sf_nom, sf_up, sf_down, counts)
# def electron_id_sf(tight_electrons, era):
#     """Compute per-event HEEP electron ID scale factor.

#     Uses flat barrel/endcap SFs from the Run2 UL2018 HEEP V7.0 measurement
#     as a proxy for Run3, per EGamma POG recommendation (Run3 HEEP SFs not
#     yet available).

#     Source: EGamma POG twiki – "HEEP ID Scale Factor for UL"
#       Barrel (|eta_SC| < 1.4442): 0.973 +/- 0.001 (stat) +/- 0.004 (syst)
#       Endcap (1.566 < |eta_SC| < 2.5): 0.980 +/- 0.002 (stat) +/- 0.011 (syst)

#     Returns (nominal, up, down) arrays of shape (n_events,).
#     Events with no tight electrons get SF = 1.0.
#     """
#     n_events = len(tight_electrons)
#     ones = np.ones(n_events, dtype=np.float64)

#     if era not in ELECTRON_JSONS:
#         key = f"electron_id_sf_unconfigured::{era}"
#         if key not in _WARN_ONCE:
#             _WARN_ONCE.add(key)
#             logger.info("No electron ID SF JSON configured for era '%s'; using SF=1.", era)
#         return ones, ones.copy(), ones.copy()

#     counts = ak.num(tight_electrons)

#     if ak.sum(counts) == 0:
#         return ones, ones.copy(), ones.copy()

#     # Supercluster eta = eta + deltaEtaSC.
#     try:
#         delta_eta_sc = tight_electrons.deltaEtaSC
#     except AttributeError:
#         key = f"missing_deltaEtaSC_id::{era}"
#         if key not in _WARN_ONCE:
#             _WARN_ONCE.add(key)
#             logger.warning(
#                 "Electron `deltaEtaSC` branch missing; using eta as fallback for HEEP ID SF."
#             )
#         delta_eta_sc = ak.zeros_like(tight_electrons.eta)

#     flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_electrons.eta, 0.0)), dtype=np.float64)
#     flat_deltaEtaSC = np.asarray(ak.flatten(ak.fill_none(delta_eta_sc, 0.0)), dtype=np.float64)
#     flat_sc_eta = np.abs(flat_eta + flat_deltaEtaSC)

#     # UL2018 HEEP V7.0 SFs (stat + syst added in quadrature).
#     barrel_sf, barrel_unc = 0.973, np.sqrt(0.001**2 + 0.004**2)  # 0.00412
#     endcap_sf, endcap_unc = 0.980, np.sqrt(0.002**2 + 0.011**2)  # 0.01118

#     is_barrel = flat_sc_eta < 1.4442

#     sf_nom = np.where(is_barrel, barrel_sf, endcap_sf)
#     sf_unc = np.where(is_barrel, barrel_unc, endcap_unc)

#     sf_up = sf_nom + sf_unc
#     sf_down = sf_nom - sf_unc

#     return _unflatten_and_product(sf_nom, sf_up, sf_down, counts)


def electron_reco_sf(tight_electrons, era):
    """Compute per-event electron Reco scale factor for tight electrons.

    Uses two working points depending on pT, with era-specific configuration
    from ELECTRON_RECO_CONFIG (correction name, WP names, and pT split point
    differ between UL and Run3 JSONs).

    The correction uses supercluster eta (eta + deltaEtaSC) and pT.
    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight electrons get SF = 1.0.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in ELECTRON_JSONS:
        key = f"electron_reco_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No electron reco SF JSON configured for era '%s'; using SF=1.", era)
        return ones, ones.copy(), ones.copy()

    sf_era_key = ELECTRON_SF_ERA_KEYS.get(era)
    if sf_era_key is None:
        logger.warning("No electron reco SF era key for '%s'; returning SF=1.", era)
        return ones, ones.copy(), ones.copy()

    reco_cfg = ELECTRON_RECO_CONFIG.get(era)
    if reco_cfg is None:
        logger.warning("No ELECTRON_RECO_CONFIG for '%s'; returning SF=1.", era)
        return ones, ones.copy(), ones.copy()

    ceval = _get_electron_ceval(era, "RECO")
    corr = ceval[reco_cfg["correction"]]

    counts = ak.num(tight_electrons)
    flat_pt = np.asarray(ak.flatten(tight_electrons.pt), dtype=np.float64)

    if len(flat_pt) == 0:
        return ones, ones.copy(), ones.copy()

    # Supercluster eta = eta + deltaEtaSC.
    # Some NanoAOD variants may not have `deltaEtaSC`; fall back to 0.0 and warn once.
    try:
        delta_eta_sc = tight_electrons.deltaEtaSC
    except AttributeError:
        key = f"missing_deltaEtaSC::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Electron `deltaEtaSC` branch missing; using eta as a fallback for electron SF evaluation."
            )
        delta_eta_sc = ak.zeros_like(tight_electrons.eta)

    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_electrons.eta, 0.0)), dtype=np.float64)
    flat_deltaEtaSC = np.asarray(ak.flatten(ak.fill_none(delta_eta_sc, 0.0)), dtype=np.float64)
    flat_sc_eta = flat_eta + flat_deltaEtaSC

    # Clip sc_eta to valid bin edges.
    flat_sc_eta = np.clip(flat_sc_eta, -2.499, 2.499)

    # Split into low-pT and high-pT groups at the era-specific threshold.
    pt_split = reco_cfg["pt_split"]
    wp_low = reco_cfg["wp_low"]
    wp_high = reco_cfg["wp_high"]
    is_low_pt = flat_pt < pt_split

    # Clip pT to valid bin ranges for each WP. The Run 3 payloads end their
    # measured RecoAbove75 bins at `pt_max` (500 GeV) and carry a placeholder
    # SF = 1 +- 20% above it; clip to the last measured bin so TeV electrons
    # take its SF and uncertainty (the POG prescription) instead.
    pt_max = float(reco_cfg.get("pt_max", 1e6))
    pt_low = np.clip(flat_pt, 10.001, pt_split - 0.001)
    pt_high = np.clip(flat_pt, pt_split + 0.001, pt_max - 0.001)

    # Check if correction requires phi (Run3Summer23+ format)
    # Run3Summer22/22EE: 5 inputs (year, ValType, WorkingPoint, eta, pt)
    # Run3Summer23+: 6 inputs (year, ValType, WorkingPoint, eta, pt, phi)
    n_inputs = len(corr.inputs)
    if n_inputs == 6:
        # Need to include phi for Run3Summer23+
        flat_phi = np.asarray(ak.flatten(tight_electrons.phi), dtype=np.float64)
        sf_low_nom = corr.evaluate(sf_era_key, "sf", wp_low, flat_sc_eta, pt_low, flat_phi)
        sf_low_up = corr.evaluate(sf_era_key, "sfup", wp_low, flat_sc_eta, pt_low, flat_phi)
        sf_low_down = corr.evaluate(sf_era_key, "sfdown", wp_low, flat_sc_eta, pt_low, flat_phi)

        sf_high_nom = corr.evaluate(sf_era_key, "sf", wp_high, flat_sc_eta, pt_high, flat_phi)
        sf_high_up = corr.evaluate(sf_era_key, "sfup", wp_high, flat_sc_eta, pt_high, flat_phi)
        sf_high_down = corr.evaluate(sf_era_key, "sfdown", wp_high, flat_sc_eta, pt_high, flat_phi)
    else:
        # Run3Summer22/22EE format (no phi)
        sf_low_nom = corr.evaluate(sf_era_key, "sf", wp_low, flat_sc_eta, pt_low)
        sf_low_up = corr.evaluate(sf_era_key, "sfup", wp_low, flat_sc_eta, pt_low)
        sf_low_down = corr.evaluate(sf_era_key, "sfdown", wp_low, flat_sc_eta, pt_low)

        sf_high_nom = corr.evaluate(sf_era_key, "sf", wp_high, flat_sc_eta, pt_high)
        sf_high_up = corr.evaluate(sf_era_key, "sfup", wp_high, flat_sc_eta, pt_high)
        sf_high_down = corr.evaluate(sf_era_key, "sfdown", wp_high, flat_sc_eta, pt_high)

    sf_nom = np.where(is_low_pt, sf_low_nom, sf_high_nom)
    sf_up = np.where(is_low_pt, sf_low_up, sf_high_up)
    sf_down = np.where(is_low_pt, sf_low_down, sf_high_down)

    return _unflatten_and_product(sf_nom, sf_up, sf_down, counts)



logger = logging.getLogger(__name__)

# Salts of the per-object draws, distinct from each other and from the high-pT muon smearing's
# (muon_momentum.HIGHPT_SALT), so that no two corrections share random numbers.
ELECTRON_SMEAR_SALT = 11
JER_SALT_AK4 = 4
JER_SALT_AK8 = 8
JER_SALT_SUBJET = 5


@lru_cache(maxsize=None)
def _object_normal_rng():
    """correctionlib's counter-based hashprng: a standard-normal draw of (event, lumi, phi, salt)."""
    from correctionlib import schemav2 as cs
    corr = cs.Correction(
        name="object_normal_rng", version=1,
        inputs=[cs.Variable(name="evt", type="int"), cs.Variable(name="lumi", type="int"),
                cs.Variable(name="phi", type="real"), cs.Variable(name="salt", type="int")],
        output=cs.Variable(name="r", type="real"),
        data=cs.HashPRNG(nodetype="hashprng", inputs=["evt", "lumi", "phi", "salt"], distribution="stdnormal"),
    )
    return cs.CorrectionSet(schema_version=2, corrections=[corr]).to_evaluator()["object_normal_rng"]


def object_normal_draws(events, objects, salt):
    """One standard-normal draw per object, flat in object order: a deterministic function of the
    event number, the luminosity block, the object's phi and ``salt``.  The same object gets the same
    draw in the nominal and in every variation (its phi does not change), and the objects of one event
    get independent draws.  The random numbers used until 2026-10-12, a hash of the event number and the
    float bits of phi, gave objects with |phi| in the same power-of-two range nearly the same draw."""
    counts = np.asarray(ak.num(objects), dtype=np.int64)
    evt = np.repeat(np.asarray(events.event, dtype=np.int64), counts)
    lumi = np.repeat(np.asarray(events.luminosityBlock, dtype=np.int64), counts)
    phi = np.asarray(ak.flatten(objects.phi), dtype=np.float64)
    salts = np.full(len(phi), salt, dtype=np.int64)
    return np.asarray(_object_normal_rng().evaluate(evt, lumi, phi, salts), dtype=np.float64)


def apply_electron_scale_smearing(events, era, is_mc):
    """
    Apply Et-dependent Scale and Smearing corrections to electrons.
    Uses cached correctionlib evaluators and deterministic seeding for MC.
    
    Args:
        events: Awkward array of events (must contain .Electron and .event/.run).
        era: The dataset era (e.g., 'RunIII2024Summer24').
        is_mc: Boolean indicating if the sample is Monte Carlo.
        
    Returns:
        A dictionary containing the corrected awkward arrays for 'pt', 'energyErr',
        and (if MC) systematic variations.
    """
    electrons = events.Electron
    counts = ak.num(electrons)
    
    # Early exit if there are no electrons in this chunk
    if ak.sum(counts) == 0:
        return {}

    try:
        cset = _get_electron_ceval(era, "ScaleNSmear")
    except KeyError:
        logger.warning(f"No ScaleNSmear JSON configured for era '{era}'; skipping correction.")
        return {}

    scale_evaluator = cset.compound["Scale"]
    smear_and_syst_evaluator = cset["SmearAndSyst"]

    # 1. Extract and flatten required variables
    flat_pt = np.asarray(ak.flatten(electrons.pt), dtype=np.float64)
    flat_r9 = np.asarray(ak.flatten(electrons.r9), dtype=np.float64)
    flat_eta = np.asarray(ak.flatten(electrons.eta), dtype=np.float64)
    flat_energyErr = np.asarray(ak.flatten(electrons.energyErr), dtype=np.float64)
    
    # Handle missing deltaEtaSC gracefully
    try:
        flat_deltaEtaSC = np.asarray(ak.flatten(electrons.deltaEtaSC), dtype=np.float64)
    except AttributeError:
        flat_deltaEtaSC = np.zeros_like(flat_eta)
        
    flat_sc_eta = flat_eta + flat_deltaEtaSC
    flat_energy = flat_pt * np.cosh(flat_eta)
    
    result = {}

    if not is_mc:
        # -------------------
        # DATA PROCEDURE
        # -------------------
        flat_seedGain = np.asarray(ak.flatten(electrons.seedGain), dtype=np.int64)
        
        # Broadcast run numbers to match the electrons per event, then flatten
        runs_broadcasted = ak.broadcast_arrays(events.run, electrons.pt)[0]
        flat_run = np.asarray(ak.flatten(runs_broadcasted), dtype=np.int64)

        # Evaluate and apply scale
        scale = scale_evaluator.evaluate("scale", flat_run, flat_sc_eta, flat_r9, flat_pt, flat_seedGain)
        corrected_pt = flat_pt * scale
        
        # Evaluate smearing width (affects energy uncertainty in data)
        smear = smear_and_syst_evaluator.evaluate("smear", corrected_pt, flat_r9, flat_sc_eta)
        corrected_energyErr = np.sqrt(flat_energyErr**2 + (flat_energy * smear)**2) * scale

        result["pt"] = ak.unflatten(corrected_pt, counts)
        result["energyErr"] = ak.unflatten(corrected_energyErr, counts)

    else:
        # -------------------
        # MC PROCEDURE
        # -------------------
        # Evaluate nominal smearing width
        smear = smear_and_syst_evaluator.evaluate("smear", flat_pt, flat_r9, flat_sc_eta)
        
        # One standard-normal draw per electron, shared by the nominal and the smearing variations,
        # independent between the electrons of an event (object_normal_draws).
        random_numbers = object_normal_draws(events, electrons, ELECTRON_SMEAR_SALT)
        
        # Apply nominal smearing
        smearing_nom = 1.0 + smear * random_numbers
        corrected_pt_nom = flat_pt * smearing_nom
        corrected_energyErr = np.sqrt(flat_energyErr**2 + (flat_energy * smear)**2) * smearing_nom
        
        # Evaluate Systematics
        smear_up = smear_and_syst_evaluator.evaluate("smear_up", flat_pt, flat_r9, flat_sc_eta)
        smear_down = smear_and_syst_evaluator.evaluate("smear_down", flat_pt, flat_r9, flat_sc_eta)
        scale_up = smear_and_syst_evaluator.evaluate("scale_up", flat_pt, flat_r9, flat_sc_eta)
        scale_down = smear_and_syst_evaluator.evaluate("scale_down", flat_pt, flat_r9, flat_sc_eta)

        # Pack results, unflattening back to awkward arrays
        result["pt"] = ak.unflatten(corrected_pt_nom, counts)
        result["energyErr"] = ak.unflatten(corrected_energyErr, counts)
        result["pt_smear_up"] = ak.unflatten(flat_pt * (1.0 + smear_up * random_numbers), counts)
        result["pt_smear_down"] = ak.unflatten(flat_pt * (1.0 + smear_down * random_numbers), counts)
        result["pt_scale_up"] = ak.unflatten(corrected_pt_nom * scale_up, counts)
        result["pt_scale_down"] = ak.unflatten(corrected_pt_nom * scale_down, counts)

    return result

import awkward as ak
import numpy as np

def verify_scale_smearing(original_electrons, corrected_dict, is_mc):
    """
    Prints a statistical summary to verify EGM scale and smearing.
    """
    # Flatten arrays to compute global statistics
    orig_pt = np.asarray(ak.flatten(original_electrons.pt))
    orig_err = np.asarray(ak.flatten(original_electrons.energyErr))
    
    corr_pt = np.asarray(ak.flatten(corrected_dict["pt"]))
    corr_err = np.asarray(ak.flatten(corrected_dict["energyErr"]))

    print("\n" + "="*70)
    print(f"--- EGM SCALE & SMEARING VERIFICATION ({'MC' if is_mc else 'DATA'}) ---")
    print("="*70)
    
    print(f"{'Variable':<20} | {'Original':<15} | {'Corrected':<15} | {'Change':<15}")
    print("-" * 70)

    if not is_mc:
        # DATA: Expect mean pT to shift (Scale), and energyErr to increase (Smearing width)
        pt_shift = np.mean(corr_pt) / np.mean(orig_pt) if np.mean(orig_pt) > 0 else 1.0
        print(f"{'Mean pT':<20} | {np.mean(orig_pt):<15.4f} | {np.mean(corr_pt):<15.4f} | {pt_shift:<10.4f} (ratio)")
        print(f"{'Std Dev pT':<20} | {np.std(orig_pt):<15.4f} | {np.std(corr_pt):<15.4f} |")
        print(f"{'Mean energyErr':<20} | {np.mean(orig_err):<15.4f} | {np.mean(corr_err):<15.4f} | Inflated")
    
    else:
        # MC: Expect mean pT to be stable, std dev pT to increase, energyErr to increase
        pt_diff = np.mean(corr_pt) - np.mean(orig_pt)
        print(f"{'Mean pT':<20} | {np.mean(orig_pt):<15.4f} | {np.mean(corr_pt):<15.4f} | {pt_diff:<10.4f} (diff)")
        print(f"{'Std Dev pT':<20} | {np.std(orig_pt):<15.4f} | {np.std(corr_pt):<15.4f} | Broadened")
        print(f"{'Mean energyErr':<20} | {np.mean(orig_err):<15.4f} | {np.mean(corr_err):<15.4f} | Inflated")
        
        # Systematics verification
        print("\n--- MC Systematics Envelopes ---")
        pt_smear_up = np.asarray(ak.flatten(corrected_dict["pt_smear_up"]))
        pt_smear_down = np.asarray(ak.flatten(corrected_dict["pt_smear_down"]))
        pt_scale_up = np.asarray(ak.flatten(corrected_dict["pt_scale_up"]))
        pt_scale_down = np.asarray(ak.flatten(corrected_dict["pt_scale_down"]))
        
        print(f"{'Smearing Std Dev':<20} | Down: {np.std(pt_smear_down):<10.4f} | Nom: {np.std(corr_pt):<10.4f} | Up: {np.std(pt_smear_up):<10.4f}")
        print(f"{'Scale Mean pT':<20} | Down: {np.mean(pt_scale_down):<10.4f} | Nom: {np.mean(corr_pt):<10.4f} | Up: {np.mean(pt_scale_up):<10.4f}")

    print("="*70 + "\n")

    
import correctionlib


#_CORRECTIONSET_CACHE = {}
def _safe_eval(corr_obj, jets, events, pt_current, rho_val):
    """
    Dynamically maps Awkward arrays to the exact inputs requested by the correctionlib object.
    """
    # Ask the correction object what inputs it requires, in what order
    expected_inputs = [inp.name for inp in corr_obj.inputs]
    
    args = []
    for inp in expected_inputs:
        if inp == "JetA":
            args.append(jets.area)
        elif inp == "JetEta":
            args.append(jets.eta)
        elif inp == "JetPhi":
            args.append(jets.phi)
        elif inp == "JetPt":
            args.append(pt_current)
        elif inp == "Rho":
            args.append(rho_val)
        elif inp == "run":
            args.append(events.run)
        else:
            raise ValueError(f"Unknown input requested by JEC JSON: {inp}")
            
    # Unpack the dynamically built list of arguments into evaluate()
    return corr_obj.evaluate(*args)

def _get_ceval(json_path):
    if json_path not in _CORRECTIONSET_CACHE:
        _CORRECTIONSET_CACHE[json_path] = correctionlib.CorrectionSet.from_file(json_path)
    return _CORRECTIONSET_CACHE[json_path]

def _select_data_tag(tag_cfg, runs):
    """Generic data tag selector that handles single strings (2023), complex nibs (2024), and legacy dicts (2018, 2022EE)."""    
    n_events = len(runs)
    # Case 1: Simple string tag (e.g., Run3Summer23)
    if isinstance(tag_cfg, str):
        return ak.Array([tag_cfg] * n_events)
    # Case 2: Complex 'nibs' configuration (e.g., 2024)
    if "nibs" in tag_cfg:
        default_tag = tag_cfg.get("key_tag", tag_cfg["nibs"][0]["tag"])
        tags = ak.Array([default_tag] * n_events)       
        for nib in tag_cfg["nibs"]:
            if "first_run" in nib and "last_run" in nib:
                mask = (runs >= nib["first_run"]) & (runs <= nib["last_run"])
                new_tag_arr = ak.Array([nib["tag"]] * n_events)
                tags = ak.where(mask, new_tag_arr, tags)
        return tags

    # Case 3: Legacy Dict (Run 2, Run 3 2022EE)
    run_boundaries = {
        "Run_A": (315252, 316995),
        "Run_B": (317080, 319310),
        "Run_C": (319337, 320065),
        "Run_D": (320673, 325175), ## for 2018 - need to confirm these run start numbers
        "Run_E": (359022, 360331),
        "Run_F": (360332, 362180),
        "Run_G": (362350, 362760), # taken from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis?rev=163#2022_Era_definition
    }
    
    first_val = list(tag_cfg.values())[0] if tag_cfg else "none"
    tags = ak.Array([str(first_val)] * n_events)
    
    for era_key, tag_val in tag_cfg.items():
        if era_key in run_boundaries:
            first_run, last_run = run_boundaries[era_key]
            mask = (runs >= first_run) & (runs <= last_run)
            new_tag_arr = ak.Array([str(tag_val)] * n_events)
            tags = ak.where(mask, new_tag_arr, tags)            
    return tags

def get_jerc_key(tag, algo, cat_name, reduction_level):
    """Maps CAT name (e.g. CMS_scale_j_Absolute) to JSON key."""
    if reduction_level == "total":
        return f"{tag}_Total_{algo}"
    
    if reduction_level == "reduced":
        source = cat_name.replace("CMS_scale_j_", "")
        return f"{tag}_Regrouped_{source}_{algo}"
    
    if reduction_level == "full":
        source = cat_name.replace("CMS_scale_j_", "")
        return f"{tag}_{source}_{algo}"

def _jer_scale_factor(ceval, tag, algo, eta, pt, sys_var):
    """Evaluate the JER scale factor for ``sys_var`` in ("nom", "up", "down").

    Two payload schemas exist:
      - a ``systematic`` input on the ScaleFactor correction (UL, Run 3 up to
        Summer23 JRV1), optionally with a JetPt input;
      - no ``systematic`` input (Summer24 JRV2 onward): the ScaleFactor gives
        the nominal and a separate ``SFUncertainty`` correction gives the
        absolute uncertainty, so up/down are nominal +/- uncertainty.
    """
    sf = ceval[f"{tag}_MC_ScaleFactor_{algo}"]
    names = [i.name for i in sf.inputs]
    args = {"JetEta": eta, "JetPt": pt, "systematic": sys_var}
    if "systematic" in names:
        return sf.evaluate(*[args[n] for n in names])

    nominal = sf.evaluate(*[args[n] for n in names])
    if sys_var == "nom":
        return nominal
    unc_key = f"{tag}_MC_SFUncertainty_{algo}"
    if not _has_correction(ceval, unc_key):
        raise KeyError(
            f"JER payload '{tag}' has no systematic input on the ScaleFactor and no "
            f"'{unc_key}' correction; cannot build the '{sys_var}' variation."
        )
    unc = ceval[unc_key]
    unc_val = unc.evaluate(*[args[n] for n in [i.name for i in unc.inputs]])
    return nominal + unc_val if sys_var == "up" else nominal - unc_val


# Smallest energy a jet is smeared to (GeV), as in coffea's jetmet_tools.
_MIN_JET_ENERGY = 1e-2


def _gen_jet_match(jets, gen_idx, genjets, max_dr):
    """(gen pT, matched) of each jet: matched when the NanoAOD gen-jet index is valid and the gen jet
    lies within ``max_dr`` = R/2 of the jet (0.2 for AK4, 0.4 for AK8; JME JER recipe).  The second
    JME condition, |pT - pT_gen| < 3 sigma_JER pT, depends on the pT being smeared and is applied in
    _get_smeared_kinematics.  Unmatched jets get a gen pT of 0."""
    valid = (gen_idx >= 0) & (gen_idx < ak.num(genjets.pt))
    safe_idx = ak.where(valid, gen_idx, ak.zeros_like(gen_idx))
    gen = ak.pad_none(genjets, 1, axis=1)[safe_idx]
    dphi = (jets.phi - gen.phi + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt((jets.eta - gen.eta) ** 2 + dphi ** 2)
    matched = valid & ak.fill_none(dr < max_dr, False)
    genpt = ak.where(matched, ak.fill_none(gen.pt, 0.0), 0.0)
    return genpt, matched


def _get_smeared_kinematics(jets, pt_to_smear, mass_to_smear, genpt, matched, events, ceval, tag, algo, sys_var="nom",
                            salt=None):
    """Returns smeared pt and mass without modifying the original jet collection.  *salt* picks the random
    stream (default: the AK4 or AK8 one by *algo*)."""
    if ak.sum(ak.num(jets)) == 0:
        return pt_to_smear, mass_to_smear

    if "Rho" in events.fields:
        rho = events.Rho.fixedGridRhoFastjetAll
    elif "fixedGridRhoFastjetAll" in events.fields:
        rho = events.fixedGridRhoFastjetAll
    else:
        return pt_to_smear, mass_to_smear
    res = ceval[f"{tag}_MC_PtResolution_{algo}"]

    sigma = res.evaluate(jets.eta, pt_to_smear, rho)
    sf_val = _jer_scale_factor(ceval, tag, algo, jets.eta, pt_to_smear, sys_var)

    # JME hybrid method: the scaling method for a jet matched to a gen jet (dR < R/2, _gen_jet_match)
    # whose pT is within 3 sigma_JER of the gen pT, the stochastic method otherwise.
    matched = matched & (np.abs(pt_to_smear - genpt) < 3.0 * sigma * pt_to_smear)
    cjer_match = 1 + (sf_val - 1) * (pt_to_smear - genpt) / pt_to_smear

    # One standard-normal draw per jet: the same jet gets the same draw in the nominal and in every
    # JES/JER variation, and the jets of an event get independent draws (object_normal_draws).
    counts = ak.num(jets, axis=1)
    if salt is None:
        salt = JER_SALT_AK8 if algo.upper().startswith("AK8") else JER_SALT_AK4
    rand = ak.unflatten(object_normal_draws(events, jets, salt), counts)
    cjer_unmatch = 1 + rand * sigma * np.sqrt(np.maximum(sf_val**2 - 1, 0))

    cjer = ak.where(matched, cjer_match, cjer_unmatch)
    # As coffea's jer_smear: a jet is never smeared below an energy of _MIN_JET_ENERGY.
    min_pt = _MIN_JET_ENERGY / np.cosh(jets.eta)
    cjer = ak.where(pt_to_smear * cjer < min_pt, min_pt / pt_to_smear, cjer)

    return pt_to_smear * cjer, mass_to_smear * cjer

def _apply_jec(jets, events, ceval, tag, algo, isMC):
    if "Rho" in events.fields:
        rho = events.Rho.fixedGridRhoFastjetAll
    elif "fixedGridRhoFastjetAll" in events.fields:
        rho = events.fixedGridRhoFastjetAll
    else:
        return jets

    good_jet_mask = jets.rawFactor <= 0.9
    jets = jets[good_jet_mask]

    if ak.sum(ak.num(jets)) == 0:
        return jets

    pt_raw = jets.pt * (1 - jets.rawFactor)
    mass_raw = jets.mass * (1 - jets.rawFactor)
    
    pt_final = pt_raw
    mass_final = mass_raw

    unique_tags = set(ak.to_list(tag)) if not isMC else [tag if isinstance(tag, str) else tag[0]]
    valid_tags = [t for t in unique_tags if t not in ["", "none"]]

    if len(valid_tags) > 0:
        for t in valid_tags:            
            l1_key = f"{t}_L1FastJet_{algo}"
            c1 = _safe_eval(ceval[l1_key], jets, events, pt_raw, rho) if _has_correction(ceval, l1_key) else 1.0
            pt1 = pt_raw * c1
            m1 = mass_raw * c1

            l2_key = f"{t}_L2Relative_{algo}"
            c2 = _safe_eval(ceval[l2_key], jets, events, pt1, rho) if _has_correction(ceval, l2_key) else 1.0
            pt2 = pt1 * c2
            m2 = m1 * c2
            
            res_key = f"{t}_L2L3Residual_{algo}"
            c4 = _safe_eval(ceval[res_key], jets, events, pt2, rho) if (not isMC and _has_correction(ceval, res_key)) else 1.0

            pt_out = pt2 * c4
            m_out = m2 * c4
            
            if isMC:
                pt_final, mass_final = pt_out, m_out
            else:
                jet_mask = (tag == t) & ak.ones_like(jets.pt, dtype=bool)
                pt_final = ak.where(jet_mask, pt_out, pt_final)
                mass_final = ak.where(jet_mask, m_out, mass_final)
    else:
        pt_final, mass_final = pt_raw, mass_raw

    jets = ak.with_field(jets, pt_final, "pt")
    jets = ak.with_field(jets, mass_final, "mass")
    return jets

# Soft-drop mass, JME jet mass recommendation (JMAR page, 2026-10-12): for all eras, apply the AK4 JEC and JER
# to both soft-drop subjets (AK4 PUPPI jets) and take m_SD = mass(p1 + p2); propagate JES and JER through both
# subjets; residual JMS (1 +- 0.01) m_SD; residual JMR: the m_SD resolution width +- 2 %, without moving the
# mass scale.  2024-2026 provisional; validated for 50 <= m_SD < 300 GeV and pT >= 100 GeV.
SUBJET_JEC_ALGO = "AK4PFPuppi"
SUBJET_GEN_MAX_DR = 0.2          # R/2 of the AK4 JER matching
JMS_UNC = 0.01
JMR_UNC = 0.02


def _event_rho(events):
    if "Rho" in events.fields:
        return events.Rho.fixedGridRhoFastjetAll
    if "fixedGridRhoFastjetAll" in events.fields:
        return events.fixedGridRhoFastjetAll
    return None


def has_subjet_jec(era):
    """Does *era*'s AK4 JERC payload carry the AK4 PUPPI corrections the subjets need?  (The UL payloads hold
    AK4PFchs only.)"""
    cfg = JERC_JSONS[era]["JERC_AK4"]
    return _has_correction(_get_ceval(cfg["json"]), f"{cfg['tag_MC']}_L2Relative_{SUBJET_JEC_ALGO}")


def _correct_subjets(subjets, events, ceval, tag, algo, isMC):
    """AK4 JEC of the soft-drop subjets: the production JEC is undone with each subjet's own ``rawFactor`` and
    L1FastJet x L2Relative (x L2L3Residual on data) applied to the raw pT, as _apply_jec does for the AK4
    jets.  No subjet is dropped: ``FatJet_subJetIdx1/2`` index this collection.

    NanoAODv9 and v12 store no ``SubJet_area``.  Only L1FastJet reads it; without it that level is skipped,
    which is exact for a PUPPI payload, whose L1FastJet is 1 (checked on the chunk's subjets)."""
    rho = _event_rho(events)
    if rho is None:
        return subjets
    pt_raw = subjets.pt * (1 - subjets.rawFactor)
    mass_raw = subjets.mass * (1 - subjets.rawFactor)
    pt_final, mass_final = pt_raw, mass_raw
    tags = [tag if isinstance(tag, str) else tag[0]] if isMC else sorted(set(ak.to_list(tag)) - {"", "none"})
    for t in tags:
        c1 = 1.0
        l1_key = f"{t}_L1FastJet_{algo}"
        if _has_correction(ceval, l1_key):
            if "area" in subjets.fields:
                c1 = _safe_eval(ceval[l1_key], subjets, events, pt_raw, rho)
            else:
                for area in (0.0, 0.5):
                    probe = ak.with_field(subjets, ak.full_like(subjets.pt, area), "area")
                    l1 = np.asarray(ak.flatten(_safe_eval(ceval[l1_key], probe, events, pt_raw, rho), axis=None))
                    if not np.allclose(l1, 1.0):
                        raise RuntimeError(
                            f"{l1_key} is not 1 and the input has no SubJet_area, so the subjet JEC "
                            "cannot be applied."
                        )
        pt1, m1 = pt_raw * c1, mass_raw * c1
        l2_key = f"{t}_L2Relative_{algo}"
        c2 = _safe_eval(ceval[l2_key], subjets, events, pt1, rho)
        pt2, m2 = pt1 * c2, m1 * c2
        res_key = f"{t}_L2L3Residual_{algo}"
        c4 = _safe_eval(ceval[res_key], subjets, events, pt2, rho) if (not isMC and _has_correction(ceval, res_key)) else 1.0
        if isMC:
            pt_final, mass_final = pt2 * c4, m2 * c4
        else:
            in_era = (tag == t) & ak.ones_like(subjets.pt, dtype=bool)
            pt_final = ak.where(in_era, pt2 * c4, pt_final)
            mass_final = ak.where(in_era, m2 * c4, mass_final)
    subjets = ak.with_field(subjets, pt_final, "pt")
    return ak.with_field(subjets, mass_final, "mass")


def _subjet_gen_index(subjets, events):
    """Index into ``SubGenJetAK8`` of each subjet's gen subjet: NanoAOD's ``SubJet_subGenJetAK8Idx`` where it
    exists (v15), otherwise the nearest gen subjet in dR (v9, v12); -1 when there is none.  The dR < R/2
    condition is _gen_jet_match's."""
    if "subGenJetAK8Idx" in subjets.fields:
        return subjets.subGenJetAK8Idx
    gen = events.SubGenJetAK8
    pairs = ak.cartesian({"j": subjets, "g": gen}, axis=1, nested=True)
    dphi = (pairs.j.phi - pairs.g.phi + np.pi) % (2 * np.pi) - np.pi
    dr2 = (pairs.j.eta - pairs.g.eta) ** 2 + dphi ** 2
    return ak.fill_none(ak.argmin(dr2, axis=2), -1)


def _fatjet_subjet_pair(fatjets, n_subjets):
    """(idx1, idx2, has_both) of each AK8 jet; an index is unusable when negative or out of range."""
    i1, i2 = fatjets.subJetIdx1, fatjets.subJetIdx2
    has = (i1 >= 0) & (i2 >= 0) & (i1 < n_subjets) & (i2 < n_subjets)
    return ak.mask(i1, has), ak.mask(i2, has), has


def _pair_mass(a, b):
    """Invariant mass of two (pt, eta, phi, mass) records."""
    px = a.pt * np.cos(a.phi) + b.pt * np.cos(b.phi)
    py = a.pt * np.sin(a.phi) + b.pt * np.sin(b.phi)
    pz = a.pt * np.sinh(a.eta) + b.pt * np.sinh(b.eta)
    e = (np.sqrt((a.pt * np.cosh(a.eta)) ** 2 + a.mass ** 2)
         + np.sqrt((b.pt * np.cosh(b.eta)) ** 2 + b.mass ** 2))
    return np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0))


def _softdrop_mass(fatjets, subjets, sj_pt, sj_mass):
    """m_SD = mass(p1 + p2) of each AK8 jet's two soft-drop subjets with the given subjet pT and mass;
    NanoAOD's ``msoftdrop`` for a jet without two usable subjets."""
    i1, i2, has = _fatjet_subjet_pair(fatjets, ak.num(subjets.pt))
    sj = ak.zip({"pt": sj_pt, "eta": subjets.eta, "phi": subjets.phi, "mass": sj_mass})
    m = ak.fill_none(_pair_mass(sj[i1], sj[i2]), 0.0)
    return ak.to_packed(ak.values_astype(ak.where(has, m, fatjets.msoftdrop), "float32"))


def _gen_softdrop_mass(fatjets, subjets, events):
    """Generator-level m_SD of each AK8 jet, from the gen subjets matched to its two subjets (dR < R/2);
    None where a subjet has no match."""
    gen = events.SubGenJetAK8
    idx = _subjet_gen_index(subjets, events)
    _, matched = _gen_jet_match(subjets, idx, gen, max_dr=SUBJET_GEN_MAX_DR)
    safe = ak.where(matched, idx, 0)
    g = ak.pad_none(gen, 1, axis=1)[safe]
    g = ak.zip({"pt": g.pt, "eta": g.eta, "phi": g.phi, "mass": g.mass, "ok": matched})
    i1, i2, has = _fatjet_subjet_pair(fatjets, ak.num(subjets.pt))
    g1, g2 = g[i1], g[i2]
    return ak.mask(_pair_mass(g1, g2), has & ak.fill_none(g1.ok & g2.ok, False))


def _softdrop_from_subjets(fatjets, events, ceval, tag, cfg, isMC, variations):
    """``FatJet`` with ``msoftdrop`` from the AK4-corrected (and, on MC, smeared) subjets, and on MC the
    requested JER / JES variations of it as ``msoftdrop_<var>_{up,down}``: each varies both subjets with the
    same nuisance as the jets (JES: the AK4 PUPPI uncertainty of the source of ``jes_unc_mapping``)."""
    algo = SUBJET_JEC_ALGO
    subjets = _correct_subjets(events.SubJet, events, ceval, tag, algo, isMC)
    pt_jec, mass_jec = subjets.pt, subjets.mass
    if not isMC:
        return ak.with_field(fatjets, _softdrop_mass(fatjets, subjets, pt_jec, mass_jec), "msoftdrop")

    genpt, matched = _gen_jet_match(subjets, _subjet_gen_index(subjets, events), events.SubGenJetAK8,
                                    max_dr=SUBJET_GEN_MAX_DR)

    def msd(pt, mass, sys_var="nom"):
        pt_s, mass_s = _get_smeared_kinematics(subjets, pt, mass, genpt, matched, events, ceval, cfg["JR_tag_MC"],
                                               algo, sys_var=sys_var, salt=JER_SALT_SUBJET)
        return _softdrop_mass(fatjets, subjets, pt_s, mass_s)

    out = ak.with_field(fatjets, msd(pt_jec, mass_jec), "msoftdrop")
    if "jer" in variations:
        for d in ("up", "down"):
            out = ak.with_field(out, msd(pt_jec, mass_jec, d), f"msoftdrop_jer_{d}")
    if "jes" in variations:
        for cat_name, source_name in cfg.get("jes_unc_mapping", {}).items():
            key = f"{tag}_{source_name}_{algo}"
            if not _has_correction(ceval, key):
                raise KeyError(f"JES source '{source_name}' has no {algo} uncertainty ('{key}') for the subjets")
            shift = ceval[key].evaluate(subjets.eta, pt_jec)
            for d, sign in (("up", 1.0), ("down", -1.0)):
                c = 1.0 + sign * shift
                out = ak.with_field(out, msd(pt_jec * c, mass_jec * c), f"msoftdrop_{cat_name}_{d}")
    return out


def _jet_mass_variations(fatjets, events, variations):
    """Residual JMS and JMR of the soft-drop mass (MC): ``msoftdrop_jms_{up,down}`` = (1 +- JMS_UNC) m_SD;
    ``msoftdrop_jmr_{up,down}`` widens / narrows the m_SD response by JMR_UNC around the generator m_SD,
    m_gen + (m - m_gen)(1 +- JMR_UNC), so the scale stays (jets whose subjets have no gen match keep m_SD)."""
    m = fatjets.msoftdrop
    if "jms" in variations:
        for d, sign in (("up", 1.0), ("down", -1.0)):
            fatjets = ak.with_field(fatjets, ak.values_astype(m * (1.0 + sign * JMS_UNC), "float32"), f"msoftdrop_jms_{d}")
    if "jmr" in variations:
        m_gen = _gen_softdrop_mass(fatjets, events.SubJet, events)
        for d, sign in (("up", 1.0), ("down", -1.0)):
            m_var = ak.where(ak.is_none(m_gen, axis=-1), m,
                             ak.fill_none(m_gen + (m - m_gen) * (1.0 + sign * JMR_UNC), 0.0))
            m_var = ak.values_astype(np.maximum(m_var, 0.0), "float32")
            fatjets = ak.with_field(fatjets, ak.to_packed(m_var), f"msoftdrop_jmr_{d}")
    return fatjets


#: Which AK8 soft-drop mass the selection cuts on.  "corrected" (the default) is
#: NanoAOD's ``FatJet_msoftdrop``, the mass of the JEC-corrected soft-drop
#: subjets.  "raw" is the mass of the uncorrected subjets, the SDMass the Run 2
#: analysis cut on (SKFlatMaker / HNWRAnalyzer).  ``ak8_msoftdrop_mode`` in
#: config.yaml selects it per era.
AK8_MSOFTDROP_MODE_CHOICES = ("corrected", "raw")


def ak8_msoftdrop_mode(era):
    """Return which AK8 soft-drop mass *era* cuts on: ``"corrected"`` or ``"raw"``."""
    mode = AK8_MSOFTDROP_MODES.get(era, "corrected")
    if mode not in AK8_MSOFTDROP_MODE_CHOICES:
        raise ValueError(
            f"ak8_msoftdrop_mode for era '{era}' is {mode!r}; expected one of {AK8_MSOFTDROP_MODE_CHOICES}"
        )
    return mode


def _raw_softdrop_mass(fatjets, subjets):
    """Soft-drop mass of each AK8 jet from its uncorrected subjets.

    Sums the p4 of the subjets at ``subJetIdx1`` / ``subJetIdx2`` with the JEC
    undone (pt and mass x (1 - rawFactor)), as SKFlatMaker summed
    ``subjets("SoftDropPuppi")`` with ``correctedP4("Uncorrected")``.  A jet
    with one subjet gets that subjet's mass, a jet with none gets 0.
    """
    px = py = pz = e = 0.0
    for idx_name in ("subJetIdx1", "subJetIdx2"):
        idx = fatjets[idx_name]
        sj = subjets[ak.mask(idx, idx >= 0)]
        raw = 1.0 - sj.rawFactor
        pt, mass = sj.pt * raw, sj.mass * raw
        sx, sy, sz = pt * np.cos(sj.phi), pt * np.sin(sj.phi), pt * np.sinh(sj.eta)
        se = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2 + mass ** 2)
        px, py, pz, e = (tot + ak.fill_none(c, 0.0) for tot, c in zip((px, py, pz, e), (sx, sy, sz, se)))
    return ak.values_astype(np.sqrt(np.maximum(e ** 2 - px ** 2 - py ** 2 - pz ** 2, 0.0)), "float32")


def apply_jet_corrections(events, era, isMC, variations=()):
    """Apply JEC (data and MC) and nominal JER smearing (MC) to Jet and FatJet.

    *variations* selects which MC shape variations are additionally attached
    as ``pt_<var>_{up,down}`` / ``mass_<var>_{up,down}`` fields:
      - ``"jer"``: JER scale-factor up/down (``pt_jer_up`` ...)
      - ``"jes"``: each source in the era's ``jes_unc_mapping``
        (``pt_CMS_scale_j_Total_up`` ...)
      - ``"jms"``, ``"jmr"``: the residual soft-drop mass scale and resolution
        (``msoftdrop_jms_up`` ...; FatJet only)
    Nothing is computed for variations that are not requested.

    ``FatJet.msoftdrop`` follows the JME jet mass recommendation where the AK4
    JERC payload has AK4 PUPPI corrections (Run 3): the subjets get the AK4 JEC
    and, on MC, the AK4 JER, and m_SD is their summed mass; JER and JES also
    vary it (``msoftdrop_jer_up`` ...).  Without them (UL) NanoAOD's m_SD is
    kept.  For an era whose ``ak8_msoftdrop_mode`` is ``"raw"``, ``msoftdrop``
    is the soft-drop mass of the uncorrected input subjets instead.
    """
    msd_mode = ak8_msoftdrop_mode(era)
    input_subjets = events.SubJet if "SubJet" in events.fields else None
    if msd_mode == "raw" and (input_subjets is None or "rawFactor" not in input_subjets.fields):
        raise RuntimeError(
            f"ak8_msoftdrop_mode for era '{era}' is 'raw', but the input has no SubJet collection with "
            "rawFactor, so the uncorrected soft-drop mass cannot be built."
        )
    cfgAK4 = JERC_JSONS[era]["JERC_AK4"]
    cfgAK8 = JERC_JSONS[era]["JERC_AK8"]

    cevalAK4 = _get_ceval(cfgAK4["json"])
    cevalAK8 = _get_ceval(cfgAK8["json"])

    algoAK4, algoAK8 = cfgAK4["algo"], cfgAK8["algo"]
    jets, fatjets = events.Jet, events.FatJet

    # Every correction level below needs the event rho.  Without it the
    # helpers return their inputs unchanged, which would silently leave the
    # NanoAOD-level JEC in place, skip JER smearing, and make any JER
    # variation an exact copy of nominal.  Say so once, and refuse to
    # produce JER templates that cannot be real.
    if "Rho" not in events.fields and "fixedGridRhoFastjetAll" not in events.fields:
        key = f"jec_no_rho::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Input for era '%s' has no rho branch (Rho or fixedGridRhoFastjetAll): jet "
                "energy corrections are not re-applied and JER smearing is skipped. Re-skim "
                "with the current skimmer to restore them.", era,
            )
        if "jer" in variations:
            raise RuntimeError(
                f"JER variations were requested for era '{era}' but the input has no rho "
                "branch, so JER smearing cannot run and the JER templates would be copies of "
                "nominal. Re-skim with the current skimmer or drop 'jer' from --systs."
            )

    if isMC:
        tagAK4, tagAK8 = cfgAK4["tag_MC"], cfgAK8["tag_MC"]
    else:
        tagAK4 = _select_data_tag(cfgAK4["tag_Data"], events.run)
        tagAK8 = _select_data_tag(cfgAK8["tag_Data"], events.run)

    # ==========================================
    # STEP 1: Compute Nominal JEC
    # ==========================================
    jets = _apply_jec(jets, events, cevalAK4, tagAK4, algoAK4, isMC)
    fatjets = _apply_jec(fatjets, events, cevalAK8, tagAK8, algoAK8, isMC)

    if isMC:
        # --- AK4 GEN MATCHING ---
        genpt, valid_match = _gen_jet_match(jets, jets.genJetIdx, events.GenJet, max_dr=0.2)
        
        # Save baseline JEC-only kinematics
        pt_jec_ak4, mass_jec_ak4 = jets.pt, jets.mass

        # ==========================================
        # STEP 2: Compute Nominal JER
        # ==========================================
        pt_nom_ak4, mass_nom_ak4 = _get_smeared_kinematics(
            jets, pt_jec_ak4, mass_jec_ak4, genpt, valid_match, events, cevalAK4, cfgAK4["JR_tag_MC"], algoAK4, sys_var="nom"
        )
        jets = ak.with_field(jets, pt_nom_ak4, "pt")
        jets = ak.with_field(jets, mass_nom_ak4, "mass")

        # --- AK8 GEN MATCHING & SMEARING ---
        genpt8, valid_match8 = _gen_jet_match(fatjets, fatjets.genJetAK8Idx, events.GenJetAK8, max_dr=0.4)

        pt_jec_ak8, mass_jec_ak8 = fatjets.pt, fatjets.mass

        pt_nom_ak8, mass_nom_ak8 = _get_smeared_kinematics(
            fatjets, pt_jec_ak8, mass_jec_ak8, genpt8, valid_match8, events, cevalAK8, cfgAK8["JR_tag_MC"], algoAK8, sys_var="nom"
        )
        fatjets = ak.with_field(fatjets, pt_nom_ak8, "pt")
        fatjets = ak.with_field(fatjets, mass_nom_ak8, "mass")

        # ==========================================
        # STEP 4: Requested shape variations only
        # ==========================================
        want_jer = "jer" in variations
        want_jes = "jes" in variations

        if want_jer:
            for jer_dir in ["up", "down"]:
                pt_jer, mass_jer = _get_smeared_kinematics(
                    jets, pt_jec_ak4, mass_jec_ak4, genpt, valid_match, events, cevalAK4, cfgAK4["JR_tag_MC"], algoAK4, sys_var=jer_dir
                )
                jets = ak.with_field(jets, pt_jer, f"pt_jer_{jer_dir}")
                jets = ak.with_field(jets, mass_jer, f"mass_jer_{jer_dir}")

                pt_jer8, mass_jer8 = _get_smeared_kinematics(
                    fatjets, pt_jec_ak8, mass_jec_ak8, genpt8, valid_match8, events, cevalAK8, cfgAK8["JR_tag_MC"], algoAK8, sys_var=jer_dir
                )
                fatjets = ak.with_field(fatjets, pt_jer8, f"pt_jer_{jer_dir}")
                fatjets = ak.with_field(fatjets, mass_jer8, f"mass_jer_{jer_dir}")

        if want_jes:
            for cat_name, source_name in cfgAK4.get("jes_unc_mapping", {}).items():
                exact_json_key = f"{tagAK4}_{source_name}_{algoAK4}"
                if _has_correction(cevalAK4, exact_json_key):
                    shift = cevalAK4[exact_json_key].evaluate(jets.eta, pt_jec_ak4)
                    for jes_dir in ["up", "down"]:
                        c_jes = 1.0 + shift if jes_dir == "up" else 1.0 - shift
                        pt_final, mass_final = _get_smeared_kinematics(
                            jets, pt_jec_ak4 * c_jes, mass_jec_ak4 * c_jes, genpt, valid_match, events, cevalAK4, cfgAK4["JR_tag_MC"], algoAK4, sys_var="nom"
                        )
                        jets = ak.with_field(jets, pt_final, f"pt_{cat_name}_{jes_dir}")
                        jets = ak.with_field(jets, mass_final, f"mass_{cat_name}_{jes_dir}")

            for cat_name, source_name in cfgAK8.get("jes_unc_mapping", {}).items():
                exact_json_key = f"{tagAK8}_{source_name}_{algoAK8}"
                if _has_correction(cevalAK8, exact_json_key):
                    shift = cevalAK8[exact_json_key].evaluate(fatjets.eta, pt_jec_ak8)
                    for jes_dir in ["up", "down"]:
                        c_jes = 1.0 + shift if jes_dir == "up" else 1.0 - shift
                        pt_final, mass_final = _get_smeared_kinematics(
                            fatjets, pt_jec_ak8 * c_jes, mass_jec_ak8 * c_jes, genpt8, valid_match8, events, cevalAK8, cfgAK8["JR_tag_MC"], algoAK8, sys_var="nom"
                        )
                        fatjets = ak.with_field(fatjets, pt_final, f"pt_{cat_name}_{jes_dir}")
                        fatjets = ak.with_field(fatjets, mass_final, f"mass_{cat_name}_{jes_dir}")

    # ==========================================
    # STEP 3: Soft-drop mass (JME jet mass recommendation)
    # ==========================================
    events = ak.with_field(events, jets, "Jet")
    if msd_mode == "raw":
        fatjets = ak.with_field(fatjets, _raw_softdrop_mass(fatjets, input_subjets), "msoftdrop")
    elif input_subjets is not None and "rawFactor" in input_subjets.fields and has_subjet_jec(era):
        fatjets = _softdrop_from_subjets(fatjets, events, cevalAK4, tagAK4, cfgAK4, isMC, variations)
    else:
        key = f"subjet_jec::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Era '%s': no %s JEC in the AK4 JERC payload (or no SubJet collection); FatJet.msoftdrop is "
                "NanoAOD's and the JES/JER variations do not move it.", era, SUBJET_JEC_ALGO,
            )
    if isMC and ({"jms", "jmr"} & set(variations)):
        fatjets = _jet_mass_variations(fatjets, events, variations)
    return ak.with_field(events, fatjets, "FatJet")


def lsf_sf(era, jet_lepton_flavor, n_events):
    """Flat LSF scale factor for the lepton-in-jet flavour of a boosted region.

    Returns ``(nominal, up, down)`` arrays of length *n_events*, or ``None``
    when *era* has no ``lsf_sf`` entry in ``config.yaml``.  *jet_lepton_flavor*
    is ``"electron"`` or ``"muon"``: the flavour of the sub-leading lepton
    inside the AK8 jet (the one the LSF requirement acts on).  The values
    are per-year constants measured with the injected-lepton method
    (AN-19-083 Appendix M); the uncertainties are asymmetric.
    """
    table = LSF_SF.get(era)
    if table is None:
        return None
    try:
        entry = table[jet_lepton_flavor]
    except KeyError as e:
        raise KeyError(f"lsf_sf for era '{era}' has no entry for flavour '{jet_lepton_flavor}'") from e
    value = float(entry["value"])
    nom = np.full(n_events, value, dtype=np.float64)
    up = np.full(n_events, value + float(entry["err_up"]), dtype=np.float64)
    down = np.full(n_events, value - float(entry["err_down"]), dtype=np.float64)
    return nom, up, down
