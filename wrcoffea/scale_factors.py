"""Scale factor evaluation for muon and electron corrections.

Uses correctionlib CorrectionSet payloads. Caches per worker process
to avoid re-reading JSON every chunk.
"""

import logging

import awkward as ak
import numpy as np

from wrcoffea.analysis_config import ELECTRON_JSONS, ELECTRON_SF_ERA_KEYS, MUON_JSONS

logger = logging.getLogger(__name__)

# Cache correctionlib payloads per worker process (avoid re-reading JSON every chunk).
_CORRECTIONSET_CACHE = {}

# Warn-once cache (per worker process) to avoid log spam.
_WARN_ONCE: set[str] = set()


def _get_muon_ceval(era):
    """Load (and cache) a correctionlib CorrectionSet for the muon SF file."""
    import correctionlib

    json_path = MUON_JSONS[era]
    ceval = _CORRECTIONSET_CACHE.get(json_path)
    if ceval is None:
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        _CORRECTIONSET_CACHE[json_path] = ceval
    return ceval


def muon_sf(tight_muons, era):
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
        return {"reco": identity, "id": identity, "iso": identity}

    ceval = _get_muon_ceval(era)

    # Flatten muon arrays for correctionlib evaluation.
    counts = ak.num(tight_muons)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_muons.eta, 0.0)), dtype=np.float64)
    flat_pt = np.asarray(ak.flatten(ak.fill_none(tight_muons.pt, 0.0)), dtype=np.float64)

    if len(flat_pt) == 0:
        return {"reco": identity, "id": identity, "iso": identity}

    # Compute momentum p = pt * cosh(eta) for RECO.
    flat_p = flat_pt * np.cosh(flat_eta)

    # Clip inputs to valid bin edges (signed eta).
    reco_eta = np.clip(flat_eta, -2.399, 2.399)
    reco_p = np.clip(flat_p, 50.001, 1e9)
    idiso_eta = np.clip(flat_eta, -2.399, 2.399)
    idiso_pt = np.clip(flat_pt, 50.001, 1e9)

    def _eval_and_unflatten(corr, eta, pt_or_p):
        nom = corr.evaluate(eta, pt_or_p, "nominal")
        up = corr.evaluate(eta, pt_or_p, "systup")
        down = corr.evaluate(eta, pt_or_p, "systdown")
        # Unflatten per-muon SFs and take product over muons per event.
        ev_nom = np.asarray(ak.fill_none(ak.prod(ak.unflatten(nom, counts), axis=1), 1.0), dtype=np.float64)
        ev_up = np.asarray(ak.fill_none(ak.prod(ak.unflatten(up, counts), axis=1), 1.0), dtype=np.float64)
        ev_down = np.asarray(ak.fill_none(ak.prod(ak.unflatten(down, counts), axis=1), 1.0), dtype=np.float64)
        return ev_nom, ev_up, ev_down

    # RECO SF (uses momentum p, not pT)
    reco = _eval_and_unflatten(ceval["NUM_GlobalMuons_DEN_TrackerMuonProbes"], reco_eta, reco_p)
    # ID SF
    id_sf = _eval_and_unflatten(ceval["NUM_HighPtID_DEN_GlobalMuonProbes"], idiso_eta, idiso_pt)
    # ISO SF
    iso = _eval_and_unflatten(ceval["NUM_probe_TightRelTkIso_DEN_HighPtProbes"], idiso_eta, idiso_pt)

    return {"reco": reco, "id": id_sf, "iso": iso}


def muon_trigger_sf(tight_muons, era):
    """Compute per-event trigger SF as product of per-muon HLT SFs.

    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight muons get SF = 1.0.

    IMPORTANT: This uses the product of per-muon SFs, which is an approximation
    for dilepton events. The proper dilepton formula is:
        e(l1,l2) = 1 - (1-e1)(1-e2)
    However, the muon JSON file only provides SFs (not raw efficiencies),
    so we cannot implement the exact formula. This approximation is commonly
    used in analyses when raw efficiencies are unavailable.
    """
    n_events = len(tight_muons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in MUON_JSONS:
        return ones, ones.copy(), ones.copy()

    ceval = _get_muon_ceval(era)
    sf_corr = ceval["NUM_HLT_DEN_HighPtTightRelIsoProbes"]

    flat_pt = np.asarray(ak.flatten(ak.fill_none(tight_muons.pt, 0.0)), dtype=np.float64)
    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_muons.eta, 0.0)), dtype=np.float64)
    counts = ak.num(tight_muons)

    if len(flat_pt) == 0:
        return ones, ones.copy(), ones.copy()

    # Clip inputs (signed eta).
    hlt_eta = np.clip(flat_eta, -2.399, 2.399)
    hlt_pt = np.clip(flat_pt, 50.001, 1e9)

    sf_nom_flat = sf_corr.evaluate(hlt_eta, hlt_pt, "nominal")
    sf_up_flat = sf_corr.evaluate(hlt_eta, hlt_pt, "systup")
    sf_down_flat = sf_corr.evaluate(hlt_eta, hlt_pt, "systdown")

    # Unflatten and take product over muons per event.
    sf_nom_jagged = ak.unflatten(sf_nom_flat, counts)
    sf_up_jagged = ak.unflatten(sf_up_flat, counts)
    sf_down_jagged = ak.unflatten(sf_down_flat, counts)

    trig_sf_nom = np.asarray(ak.fill_none(ak.prod(sf_nom_jagged, axis=1), 1.0), dtype=np.float64)
    trig_sf_up = np.asarray(ak.fill_none(ak.prod(sf_up_jagged, axis=1), 1.0), dtype=np.float64)
    trig_sf_down = np.asarray(ak.fill_none(ak.prod(sf_down_jagged, axis=1), 1.0), dtype=np.float64)

    return trig_sf_nom, trig_sf_up, trig_sf_down


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

    Note: Uses HLT_SF_Ele30_TightID as proxy for Ele32_WPTight_Gsf trigger.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in ELECTRON_JSONS or "TRIGGER" not in ELECTRON_JSONS[era]:
        return ones, ones.copy(), ones.copy()

    sf_era_key = ELECTRON_SF_ERA_KEYS.get(era)
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

    # Clip inputs to valid ranges.
    hlt_eta = np.clip(flat_eta, -2.499, 2.499)
    hlt_pt = np.clip(flat_pt, 30.001, 1e6)

    # Use Ele30_TightID as proxy for Ele32_WPTight_Gsf.
    trigger_path = "HLT_SF_Ele30_TightID"

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
    # Protect against division by zero.
    trig_sf_nom = np.asarray(ak.fill_none(
        ak.where(event_eff_mc_nom > 0, event_eff_data_nom / event_eff_mc_nom, 1.0), 1.0
    ), dtype=np.float64)
    trig_sf_up = np.asarray(ak.fill_none(
        ak.where(event_eff_mc_up > 0, event_eff_data_up / event_eff_mc_up, 1.0), 1.0
    ), dtype=np.float64)
    trig_sf_down = np.asarray(ak.fill_none(
        ak.where(event_eff_mc_down > 0, event_eff_data_down / event_eff_mc_down, 1.0), 1.0
    ), dtype=np.float64)

    return trig_sf_nom, trig_sf_up, trig_sf_down


def electron_reco_sf(tight_electrons, era):
    """Compute per-event electron Reco scale factor for tight electrons.

    Uses two working points depending on pT:
      - "Reco20to75" for electrons with 20 < pT < 75 GeV
      - "RecoAbove75" for electrons with pT >= 75 GeV

    The correction uses supercluster eta (eta + deltaEtaSC) and pT.
    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight electrons get SF = 1.0.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in ELECTRON_JSONS:
        return ones, ones.copy(), ones.copy()

    sf_era_key = ELECTRON_SF_ERA_KEYS.get(era)
    if sf_era_key is None:
        logger.warning("No electron reco SF era key for '%s'; returning SF=1.", era)
        return ones, ones.copy(), ones.copy()

    ceval = _get_electron_ceval(era, "RECO")
    corr = ceval["Electron-ID-SF"]

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

    # Clip sc_eta to valid bin edges: (-inf, inf) handled by correctionlib,
    # but we clip to avoid float edge issues at +/-2.5.
    flat_sc_eta = np.clip(flat_sc_eta, -2.499, 2.499)

    # Evaluate per-electron SF using the appropriate working point.
    # Split into low-pT (20-75) and high-pT (>=75) groups.
    is_low_pt = flat_pt < 75.0

    # Clip pT to valid bin ranges for each WP.
    # Reco20to75: pt [20, 75, inf) -- clip to [20.001, ...)
    # RecoAbove75: pt [75, 100, 500, inf) -- clip to [75.001, ...)
    pt_low = np.clip(flat_pt, 20.001, 74.999)
    pt_high = np.clip(flat_pt, 75.001, 1e6)

    # Evaluate both WPs on all electrons, then select per-electron.
    sf_low_nom = corr.evaluate(sf_era_key, "sf", "Reco20to75", flat_sc_eta, pt_low)
    sf_low_up = corr.evaluate(sf_era_key, "sfup", "Reco20to75", flat_sc_eta, pt_low)
    sf_low_down = corr.evaluate(sf_era_key, "sfdown", "Reco20to75", flat_sc_eta, pt_low)

    sf_high_nom = corr.evaluate(sf_era_key, "sf", "RecoAbove75", flat_sc_eta, pt_high)
    sf_high_up = corr.evaluate(sf_era_key, "sfup", "RecoAbove75", flat_sc_eta, pt_high)
    sf_high_down = corr.evaluate(sf_era_key, "sfdown", "RecoAbove75", flat_sc_eta, pt_high)

    sf_nom = np.where(is_low_pt, sf_low_nom, sf_high_nom)
    sf_up = np.where(is_low_pt, sf_low_up, sf_high_up)
    sf_down = np.where(is_low_pt, sf_low_down, sf_high_down)

    # Unflatten and take product over electrons per event.
    sf_nom_jagged = ak.unflatten(sf_nom, counts)
    sf_up_jagged = ak.unflatten(sf_up, counts)
    sf_down_jagged = ak.unflatten(sf_down, counts)

    event_sf_nom = np.asarray(ak.fill_none(ak.prod(sf_nom_jagged, axis=1), 1.0), dtype=np.float64)
    event_sf_up = np.asarray(ak.fill_none(ak.prod(sf_up_jagged, axis=1), 1.0), dtype=np.float64)
    event_sf_down = np.asarray(ak.fill_none(ak.prod(sf_down_jagged, axis=1), 1.0), dtype=np.float64)

    return event_sf_nom, event_sf_up, event_sf_down
