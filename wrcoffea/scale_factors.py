"""Scale factor evaluation for muon and electron corrections.

Uses correctionlib CorrectionSet payloads. Caches per worker process
to avoid re-reading JSON every chunk.
"""

import logging

import awkward as ak
import numpy as np

from wrcoffea.analysis_config import (
    CUTS, ELECTRON_JSONS, ELECTRON_RECO_CONFIG, ELECTRON_SF_ERA_KEYS,
    JETVETO_CORRECTION_NAMES, JETVETO_JSONS, MUON_JSONS,
    PILEUP_CORRECTION_NAMES, PILEUP_JSONS, JERC_JSONS
)
import vector
vector.register_awkward()
from scipy.special import ndtri
logger = logging.getLogger(__name__)
from wrcoffea.MuonScaRe import pt_resol, pt_scale, pt_resol_var, pt_scale_var
# Cache correctionlib payloads per worker process (avoid re-reading JSON every chunk).
_CORRECTIONSET_CACHE = {}

# Warn-once cache (per worker process) to avoid log spam.
_WARN_ONCE: set[str] = set()


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


def jet_veto_event_mask(events, ak4_id_mask, era):
    """Return a per-event boolean mask (True = pass) using JME jet veto maps.

    Evaluates the ``jetvetomap`` correction on all jets with pT > 15 GeV.
    Events containing at least one jet in a vetoed (eta, phi) region are
    flagged False.  Applies to both data and MC.
    """
    n_events = len(events)
    if era not in JETVETO_JSONS:
        key = f"jetveto_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No jet veto map configured for era '%s'; passing all events.", era)
        return np.ones(n_events, dtype=bool)

    import correctionlib

    json_path = JETVETO_JSONS[era]
    ceval = _CORRECTIONSET_CACHE.get(json_path)
    if ceval is None:
        ceval = correctionlib.CorrectionSet.from_file(json_path)
        _CORRECTIONSET_CACHE[json_path] = ceval

    corr = ceval[JETVETO_CORRECTION_NAMES[era]]
    # Select all jets above the veto map pT threshold, EMfrac threshold and they should satisfy tight lepton veto ID.
    jets = events.Jet
    pt_mask = jets.pt > CUTS["jet_veto_pt_min"]
    jet_emf_mask = (jets.chEmEF + jets.neEmEF < CUTS["jet_veto_em_neFrac"])
    jet_mask = jet_emf_mask & pt_mask & ak4_id_mask
    sel_jets = jets[jet_mask]
    
    counts = ak.num(sel_jets)
    flat_eta = np.asarray(ak.flatten(sel_jets.eta), dtype=np.float64)
    flat_phi = np.asarray(ak.flatten(sel_jets.phi), dtype=np.float64)

    if len(flat_eta) == 0:
        return np.ones(n_events, dtype=bool)

    # Evaluate: non-zero → vetoed region.
    veto_vals = corr.evaluate("jetvetomap", flat_eta, flat_phi)

    # Unflatten and check if any jet per event is vetoed.
    vetoed_per_jet = ak.unflatten(veto_vals > 0, counts)
    any_vetoed = ak.any(vetoed_per_jet, axis=1)
    return events[~any_vetoed]


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

    # Clip inputs to valid bin edges (signed eta).
    reco_eta = np.clip(flat_eta, -2.399, 2.399)
    reco_p = np.clip(flat_p, 50.001, 1e9)
    idiso_eta = np.clip(flat_eta, -2.399, 2.399)
    idiso_pt = np.clip(flat_pt, 50.001, 1e9)

    def _eval_component(corr, eta, pt_or_p):
        nom = corr.evaluate(eta, pt_or_p, "nominal")
        up = corr.evaluate(eta, pt_or_p, "systup")
        down = corr.evaluate(eta, pt_or_p, "systdown")
        return _unflatten_and_product(nom, up, down, counts)

    # RECO SF (uses momentum p, not pT)
    reco = _eval_component(ceval["NUM_GlobalMuons_DEN_TrackerMuonProbes"], reco_eta, reco_p)
    # ID SF
    id_sf = _eval_component(ceval["NUM_HighPtID_DEN_GlobalMuonProbes"], idiso_eta, idiso_pt)
    # ISO SF
    if is_loose:
        # Create arrays of 1.0 with the exact same jagged shape as the muons
        ones = ak.ones_like(idiso_pt)
        # Return a tuple of (nominal, up, down) to match the expected format
        iso = (ones, ones, ones)
    else:
        iso = _eval_component(ceval["NUM_probe_LooseRelTkIso_DEN_HighPtProbes"], idiso_eta, idiso_pt)   
    #iso = _eval_component(ceval["NUM_probe_LooseRelTkIso_DEN_HighPtProbes"], idiso_eta, idiso_pt)
    return {"reco": reco, "id": id_sf, "iso": iso}

def muon_trigger_sf(tight_muons, era):
    """Compute per-event muon trigger SF using dilepton efficiency formula.

    Implements:
        e(l1,l2) = 1 - (1-e(l1))(1-e(l2))

    Event SF:
        SF = e_data(event) / e_MC(event)

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

    data_eff_corr = ceval["NUM_HLT_DEN_HighPtLooseRelIsoProbes_DATAeff"]
    mc_eff_corr   = ceval["NUM_HLT_DEN_HighPtLooseRelIsoProbes_MCeff"]

    counts = ak.num(tight_muons)

    flat_pt = np.asarray(
        ak.flatten(ak.fill_none(tight_muons.pt, 0.0)),
        dtype=np.float64
    )
    flat_eta = np.asarray(
        ak.flatten(ak.fill_none(tight_muons.eta, 0.0)),
        dtype=np.float64
    )

    if len(flat_pt) == 0:
        return ones, ones.copy(), ones.copy()

    # Clip inputs
    #    hlt_eta = np.clip(flat_eta, -2.399, 2.399)
    hlt_eta = np.clip(np.abs(flat_eta), 0.0, 2.4)
    hlt_pt  = np.clip(flat_pt, 50.001, 1e9)

    # Evaluate efficiencies
    eff_data_nom  = data_eff_corr.evaluate(hlt_eta, hlt_pt, "nominal")
    eff_data_up   = data_eff_corr.evaluate(hlt_eta, hlt_pt, "systup")
    eff_data_down = data_eff_corr.evaluate(hlt_eta, hlt_pt, "systdown")

    eff_mc_nom  = mc_eff_corr.evaluate(hlt_eta, hlt_pt, "nominal")
    eff_mc_up   = mc_eff_corr.evaluate(hlt_eta, hlt_pt, "systup")
    eff_mc_down = mc_eff_corr.evaluate(hlt_eta, hlt_pt, "systdown")

    # Restore event structure
    eff_data_nom_jag = ak.unflatten(eff_data_nom, counts)
    eff_data_up_jag = ak.unflatten(eff_data_up, counts)
    eff_data_down_jag = ak.unflatten(eff_data_down, counts)

    eff_mc_nom_jag = ak.unflatten(eff_mc_nom, counts)
    eff_mc_up_jag = ak.unflatten(eff_mc_up, counts)
    eff_mc_down_jag = ak.unflatten(eff_mc_down, counts)

    # Event efficiencies using OR logic
    event_eff_data_nom  = 1.0 - ak.prod(1.0 - eff_data_nom_jag, axis=1)
    event_eff_data_up   = 1.0 - ak.prod(1.0 - eff_data_up_jag, axis=1)
    event_eff_data_down = 1.0 - ak.prod(1.0 - eff_data_down_jag, axis=1)

    event_eff_mc_nom  = 1.0 - ak.prod(1.0 - eff_mc_nom_jag, axis=1)
    event_eff_mc_up   = 1.0 - ak.prod(1.0 - eff_mc_up_jag, axis=1)
    event_eff_mc_down = 1.0 - ak.prod(1.0 - eff_mc_down_jag, axis=1)

    # Safe ratio
    def _safe_ratio(num, den):
        num_np = np.asarray(ak.fill_none(num, 0.0), dtype=np.float64)
        den_np = np.asarray(ak.fill_none(den, 0.0), dtype=np.float64)

        out = np.ones_like(num_np)
        np.divide(num_np, den_np, out=out, where=den_np > 0)
        return out

    trig_sf_nom  = _safe_ratio(event_eff_data_nom, event_eff_mc_nom)
    trig_sf_up   = _safe_ratio(event_eff_data_up, event_eff_mc_up)
    trig_sf_down = _safe_ratio(event_eff_data_down, event_eff_mc_down)

    return trig_sf_nom, trig_sf_up, trig_sf_down

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

    Note: Uses HLT_SF_Ele30_TightID as proxy for Ele32_WPTight_Gsf trigger.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in ELECTRON_JSONS or "TRIGGER" not in ELECTRON_JSONS.get(era, {}):
        key = f"electron_trig_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No electron trigger SF JSON configured for era '%s'; using SF=1.", era)
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


def electron_id_sf(tight_electrons, era):
    """Compute per-event HEEP electron ID scale factor.

    Uses flat barrel/endcap SFs from the Run2 UL2018 HEEP V7.0 measurement
    as a proxy for Run3, per EGamma POG recommendation (Run3 HEEP SFs not
    yet available).

    Source: EGamma POG twiki – "HEEP ID Scale Factor for UL"
      Barrel (|eta_SC| < 1.4442): 0.973 +/- 0.001 (stat) +/- 0.004 (syst)
      Endcap (1.566 < |eta_SC| < 2.5): 0.980 +/- 0.002 (stat) +/- 0.011 (syst)

    Returns (nominal, up, down) arrays of shape (n_events,).
    Events with no tight electrons get SF = 1.0.
    """
    n_events = len(tight_electrons)
    ones = np.ones(n_events, dtype=np.float64)

    if era not in ELECTRON_JSONS:
        key = f"electron_id_sf_unconfigured::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.info("No electron ID SF JSON configured for era '%s'; using SF=1.", era)
        return ones, ones.copy(), ones.copy()

    counts = ak.num(tight_electrons)

    if ak.sum(counts) == 0:
        return ones, ones.copy(), ones.copy()

    # Supercluster eta = eta + deltaEtaSC.
    try:
        delta_eta_sc = tight_electrons.deltaEtaSC
    except AttributeError:
        key = f"missing_deltaEtaSC_id::{era}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "Electron `deltaEtaSC` branch missing; using eta as fallback for HEEP ID SF."
            )
        delta_eta_sc = ak.zeros_like(tight_electrons.eta)

    flat_eta = np.asarray(ak.flatten(ak.fill_none(tight_electrons.eta, 0.0)), dtype=np.float64)
    flat_deltaEtaSC = np.asarray(ak.flatten(ak.fill_none(delta_eta_sc, 0.0)), dtype=np.float64)
    flat_sc_eta = np.abs(flat_eta + flat_deltaEtaSC)

    # UL2018 HEEP V7.0 SFs (stat + syst added in quadrature).
    barrel_sf, barrel_unc = 0.973, np.sqrt(0.001**2 + 0.004**2)  # 0.00412
    endcap_sf, endcap_unc = 0.980, np.sqrt(0.002**2 + 0.011**2)  # 0.01118

    is_barrel = flat_sc_eta < 1.4442

    sf_nom = np.where(is_barrel, barrel_sf, endcap_sf)
    sf_unc = np.where(is_barrel, barrel_unc, endcap_unc)

    sf_up = sf_nom + sf_unc
    sf_down = sf_nom - sf_unc

    return _unflatten_and_product(sf_nom, sf_up, sf_down, counts)


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

    # Clip pT to valid bin ranges for each WP.
    pt_low = np.clip(flat_pt, 10.001, pt_split - 0.001)
    pt_high = np.clip(flat_pt, pt_split + 0.001, 1e6)

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
        
        # 1. Broadcast event numbers to match electrons, then flatten
        flat_event = np.asarray(ak.flatten(ak.broadcast_arrays(events.event, electrons.pt)[0]), dtype=np.uint64)
        
        # 2. Extract phi as a 32-bit float, then view it as an integer for bitwise operations.
        # This distinguishes multiple electrons within the exact same event.
        flat_phi = np.asarray(ak.flatten(electrons.phi), dtype=np.float32).view(np.uint32)
        
        # 3. Fast Vectorized Hash: Multiply event by a large prime and XOR with electron phi
        # 2654435761 is a standard prime used in integer hashing
        hash_val = (flat_event * np.uint64(2654435761)) ^ flat_phi
        
        # 4. Map the hash to a uniform distribution between (0, 1)
        # We clip it slightly away from 0 and 1 to prevent infinity when converting to Gaussian
        uniform_random = (hash_val % np.uint64(2**31 - 1)) / float(2**31 - 1)
        uniform_random = np.clip(uniform_random, 1e-8, 1.0 - 1e-8)
        
        # 5. Transform uniform to Standard Normal Gaussian N(0, 1)
        random_numbers = ndtri(uniform_random)
        
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
    
def _apply_jec(jets, events, ceval, tag, algo, isMC):
    if "Rho" in events.fields:
        rho = events.Rho.fixedGridRhoFastjetAll    # Run 3 schema
    elif "fixedGridRhoFastjetAll" in events.fields:
        rho = events.fixedGridRhoFastjetAll
    else:
        return jets
    # As per Twiki: Reject jets with Jet_rawFactor > 0.9
    good_jet_mask = jets.rawFactor <= 0.9
    jets = jets[good_jet_mask]

    #rho = events.Rho.fixedGridRhoFastjetAll

    pt_raw = jets.pt * (1 - jets.rawFactor)
    mass_raw = jets.mass * (1 - jets.rawFactor)
    
    pt_final = pt_raw
    mass_final = mass_raw

    unique_tags = set(ak.to_list(tag)) if not isMC else [tag if isinstance(tag, str) else tag[0]]
    # print(f"JEC EVALUATION FOR: {'MC' if isMC else 'DATA'}")
    # print(f"Unique Tags found in this chunk: {unique_tags}")
    
    # Filter out empty tags and 'none' so we only look at real keys
    valid_tags = [t for t in unique_tags if t not in ["", "none"]]

    if len(valid_tags) > 0:
        # print(f"JEC EVALUATION FOR: {'MC' if isMC else 'DATA'}")
        # print(f"Valid Tags found in this chunk: {valid_tags}")        
        for t in valid_tags:           
            # 1. L1 FastJet
            l1_key = f"{t}_L1FastJet_{algo}"
            #print(f"Looking for L1 Key:  {l1_key}")
            if l1_key in ceval:
                #print("  --> L1 Key FOUND! Evaluating...")
                c1 = _safe_eval(ceval[l1_key], jets, events, pt_raw, rho)
            else:
                #print("  --> L1 Key MISSING! Falling back to 1.0")
                #print("  --> First 3 available keys in JSON:", list(ceval.keys())[:3])
                c1 = 1.0
                
            pt1 = pt_raw * c1
            m1 = mass_raw * c1

            # 2. L2 Relative
            l2_key = f"{t}_L2Relative_{algo}"
            #print(f"Looking for L2 Key:  {l2_key}")
            if l2_key in ceval:
                c2 = _safe_eval(ceval[l2_key], jets, events, pt1, rho)
            else:
                #print("  --> L2 Key MISSING! Falling back to 1.0")
                c2 = 1.0
                
            pt2 = pt1 * c2
            m2 = m1 * c2
            
            # # 3. L3 Absolute
            # l3_key = f"{t}_L3Absolute_{algo}"
            # if l3_key in ceval:
            #     c3 = _safe_eval(ceval[l3_key], jets, events, pt2, rho)
            # else:
            #     c3 = 1.0
                
            # pt3 = pt2 * c3
            # m3 = m2 * c3

            # 4. Residuals
            res_key = f"{t}_L2L3Residual_{algo}"
            if not isMC:
                #print(f"Looking for Res Key: {res_key}")
                if res_key in ceval:
                    #print("  --> Residual Key FOUND! Evaluating...")
                    c4 = _safe_eval(ceval[res_key], jets, events, pt2, rho)
                else:
                    #print("  --> Residual Key MISSING! Falling back to 1.0")
                    c4 = 1.0
            else:
                c4 = 1.0

            pt_out = pt2 * c4
            m_out = m2 * c4
            
            # 5. Masking and Accumulation
            if isMC:
                pt_final = pt_out
                mass_final = m_out
            else:
                jet_mask = (tag == t) & ak.ones_like(jets.pt, dtype=bool)
                pt_final = ak.where(jet_mask, pt_out, pt_final)
                mass_final = ak.where(jet_mask, m_out, mass_final)

        #print("="*50 + "\n")
    else:
        # If there are no valid tags, just assign the uncorrected arrays
        pt_final = pt_raw
        mass_final = mass_raw

    jets = ak.with_field(jets, pt_final, "pt")
    jets = ak.with_field(jets, mass_final, "mass")
    return jets

def _get_smeared_kinematics(jets, pt_to_smear, mass_to_smear, genpt, matched, events, ceval, tag, algo, sys_var="nom"):
    """Returns smeared pt and mass without modifying the original jet collection"""
    if "Rho" in events.fields:
        rho = events.Rho.fixedGridRhoFastjetAll
    elif "fixedGridRhoFastjetAll" in events.fields:
        rho = events.fixedGridRhoFastjetAll
    else:
        return pt_to_smear, mass_to_smear
    res = ceval[f"{tag}_MC_PtResolution_{algo}"]
    sf  = ceval[f"{tag}_MC_ScaleFactor_{algo}"]

    sigma = res.evaluate(jets.eta, pt_to_smear, rho)
    sf_val = sf.evaluate(jets.eta, pt_to_smear, sys_var)

    cjer_match = 1 + (sf_val - 1) * (pt_to_smear - genpt) / pt_to_smear

    counts = ak.num(jets, axis=1)
    # Broadcast event IDs to match the number of jets
    flat_event = np.asarray(ak.flatten(ak.broadcast_arrays(events.event, jets.pt)[0]), dtype=np.uint64)
    # Use phi to distinguish jets within the same event
    flat_phi = np.asarray(ak.flatten(jets.phi), dtype=np.float32).view(np.uint32)

    # 2. Vectorized Hash (Deterministic "Randomness")
    # Using the same prime you used for electrons: 2654435761
    hash_val = (flat_event * np.uint64(2654435761)) ^ flat_phi
    
    # 3. Map to Standard Normal N(0, 1)
    uniform_random = (hash_val % np.uint64(2**31 - 1)) / float(2**31 - 1)
    uniform_random = np.clip(uniform_random, 1e-8, 1.0 - 1e-8)
    rand_flat = ndtri(uniform_random)
    
    # 4. Unflatten to match jet structure
    rand = ak.unflatten(rand_flat, counts)
    # rand = np.random.normal(0, 1, size=int(ak.sum(counts)))
    # rand = ak.unflatten(rand, counts)
    cjer_unmatch = 1 + rand * sigma * np.sqrt(np.maximum(sf_val**2 - 1, 0))

    cjer = ak.where(matched, cjer_match, cjer_unmatch)

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

    pt_raw = jets.pt * (1 - jets.rawFactor)
    mass_raw = jets.mass * (1 - jets.rawFactor)
    
    pt_final = pt_raw
    mass_final = mass_raw

    unique_tags = set(ak.to_list(tag)) if not isMC else [tag if isinstance(tag, str) else tag[0]]
    valid_tags = [t for t in unique_tags if t not in ["", "none"]]

    if len(valid_tags) > 0:
        for t in valid_tags:            
            l1_key = f"{t}_L1FastJet_{algo}"
            c1 = _safe_eval(ceval[l1_key], jets, events, pt_raw, rho) if l1_key in ceval else 1.0
            pt1 = pt_raw * c1
            m1 = mass_raw * c1

            l2_key = f"{t}_L2Relative_{algo}"
            c2 = _safe_eval(ceval[l2_key], jets, events, pt1, rho) if l2_key in ceval else 1.0
            pt2 = pt1 * c2
            m2 = m1 * c2
            
            res_key = f"{t}_L2L3Residual_{algo}"
            c4 = _safe_eval(ceval[res_key], jets, events, pt2, rho) if (not isMC and res_key in ceval) else 1.0

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

def _correct_subjets(subjets, events, ceval, tag, algo, isMC):
    if "Rho" in events.fields:
        rho = events.Rho.fixedGridRhoFastjetAll    # Run 3 schema                                                                                                           
    elif "fixedGridRhoFastjetAll" in events.fields:
        rho = events.fixedGridRhoFastjetAll
    else:
        return subjets

    #rho = events.Rho.fixedGridRhoFastjetAll
    # Raw quantities
    pt_raw = subjets.pt * (1 - subjets.rawFactor)
    mass_raw = subjets.mass * (1 - subjets.rawFactor)
    
    # Initialize final arrays with raw values to accumulate into
    pt_final = pt_raw
    mass_final = mass_raw

    # Unify MC and Data tag iteration
    unique_tags = set(ak.to_list(tag)) if not isMC else [tag if isinstance(tag, str) else tag[0]]
    
    # Filter out empty tags and 'none' so we only look at real keys
    valid_tags = [t for t in unique_tags if t not in ["", "none"]]

    if len(valid_tags) > 0:
        for t in valid_tags:
            
            # 1. L1 FastJet
            l1_key = f"{t}_L1FastJet_{algo}"
            if l1_key in ceval.keys():  # <-- .keys() fix applied
                # _safe_eval is smart enough to extract area, eta, etc., from subjets natively!
                c1 = _safe_eval(ceval[l1_key], subjets, events, pt_raw, rho)
            else:
                c1 = 1.0
                
            pt1 = pt_raw * c1
            m1 = mass_raw * c1

            # 2. L2 Relative
            l2_key = f"{t}_L2Relative_{algo}"
            if l2_key in ceval.keys():  # <-- .keys() fix applied
                c2 = _safe_eval(ceval[l2_key], subjets, events, pt1, rho)
            else:
                c2 = 1.0
                
            pt2 = pt1 * c2
            m2 = m1 * c2
            
            # # 3. L3 Absolute (Commented out as in apply_jec)
            # l3_key = f"{t}_L3Absolute_{algo}"
            # if l3_key in ceval.keys():
            #     c3 = _safe_eval(ceval[l3_key], subjets, events, pt2, rho)
            # else:
            #     c3 = 1.0
                
            # pt3 = pt2 * c3
            # m3 = m2 * c3

            # 4. Residuals
            res_key = f"{t}_L2L3Residual_{algo}"
            if not isMC:
                if res_key in ceval.keys():  # <-- .keys() fix applied
                    # Notice we pass pt2 here since L3 is commented out
                    c4 = _safe_eval(ceval[res_key], subjets, events, pt2, rho)
                else:
                    c4 = 1.0
            else:
                c4 = 1.0

            pt_out = pt2 * c4
            m_out = m2 * c4
            
            # 5. Masking and Accumulation
            if isMC:
                pt_final = pt_out
                mass_final = m_out
            else:
                # Create a nested mask matching the subjets collection dimensions
                subjet_mask = (tag == t) & ak.ones_like(subjets.pt, dtype=bool)
                
                # Accumulate the correctly matched subjets
                pt_final = ak.where(subjet_mask, pt_out, pt_final)
                mass_final = ak.where(subjet_mask, m_out, mass_final)

    else:
        # If there are no valid tags, just assign the uncorrected arrays
        pt_final = pt_raw
        mass_final = mass_raw

    # Update fields and return the updated subjets collection
    subjets = ak.with_field(subjets, pt_final, "pt")
    subjets = ak.with_field(subjets, mass_final, "mass")
    
    return subjets

def _recompute_softdrop(events):
    fatjets = events.FatJet
    subjets = events.SubJet

    base_mass = fatjets.msoftdrop if "msoftdrop" in fatjets.fields else fatjets.mass
    
    # 1. Define the mask
    has_subjets = (fatjets.subJetIdx1 >= 0) & (fatjets.subJetIdx2 >= 0)

    # 2. Use ak.mask to keep the shape consistent (249054 * var)
    # This ensures idx1 has the same length as fatjets, with None where has_subjets is False
    idx1 = ak.mask(fatjets.subJetIdx1, has_subjets)
    idx2 = ak.mask(fatjets.subJetIdx2, has_subjets)

    # 3. Indexing with a mask propagates the None values
    
    sj1 = subjets[idx1]
    sj2 = subjets[idx2]
    sj1_vec = ak.zip({"pt": sj1.pt, "eta": sj1.eta, "phi": sj1.phi, "mass": sj1.mass}, with_name="PtEtaPhiMLorentzVector")
    sj2_vec = ak.zip({"pt": sj2.pt, "eta": sj2.eta, "phi": sj2.phi, "mass": sj2.mass}, with_name="PtEtaPhiMLorentzVector")
    
    # Adding vectors handles all the px, py, pz, E math in optimized C++
    mass_new = (sj1_vec + sj2_vec).mass
    # # 4. Math operations automatically ignore None and return None at those positions
    # px = sj1.pt * np.cos(sj1.phi) + sj2.pt * np.cos(sj2.phi)
    # py = sj1.pt * np.sin(sj1.phi) + sj2.pt * np.sin(sj2.phi)
    # pz = sj1.pt * np.sinh(sj1.eta) + sj2.pt * np.sinh(sj2.eta)

    # e1 = np.sqrt((sj1.pt*np.cosh(sj1.eta))**2 + sj1.mass**2)
    # e2 = np.sqrt((sj2.pt*np.cosh(sj2.eta))**2 + sj2.mass**2)

    # # mass_new now has the same shape as fatjets, but with None where there are no subjets
    # mass_new = np.sqrt(np.maximum((e1 + e2)**2 - px**2 - py**2 - pz**2, 0))
    mass_new = ak.values_astype(mass_new, "float32")
    base_mass = ak.values_astype(base_mass, "float32")
    # 5. Fill the Nones in mass_new with the original base_mass values
    # ak.fill_none is often cleaner than ak.where for this specific pattern
    updated_msoftdrop = ak.where(has_subjets, mass_new, base_mass)
    # updated_msoftdrop = ak.fill_none(mass_new, base_mass)
    updated_msoftdrop = ak.to_packed(updated_msoftdrop)
    # Return the updated fatjets record
    return ak.with_field(fatjets, updated_msoftdrop, "msoftdrop")

def apply_jet_corrections(events, era, isMC,save_all_variations=False):
    cfgAK4 = JERC_JSONS[era]["JERC_AK4"]
    cfgAK8 = JERC_JSONS[era]["JERC_AK8"]

    cevalAK4 = _get_ceval(cfgAK4["json"])
    cevalAK8 = _get_ceval(cfgAK8["json"])

    algoAK4, algoAK8 = cfgAK4["algo"], cfgAK8["algo"]
    jets, fatjets = events.Jet, events.FatJet

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
        gen_idx = jets.genJetIdx
        valid_match = gen_idx >= 0
        safe_idx = ak.where(valid_match, gen_idx, ak.zeros_like(gen_idx))
        genpt = ak.where(valid_match, ak.pad_none(events.GenJet.pt, 1, axis=1)[safe_idx], 0)
        
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
        gen_idx8 = fatjets.genJetAK8Idx
        valid_match8 = gen_idx8 >= 0
        safe_idx8 = ak.where(valid_match8, gen_idx8, ak.zeros_like(gen_idx8))
        genpt8 = ak.where(valid_match8, ak.pad_none(events.GenJetAK8.pt, 1, axis=1)[safe_idx8], 0)

        pt_jec_ak8, mass_jec_ak8 = fatjets.pt, fatjets.mass

        pt_nom_ak8, mass_nom_ak8 = _get_smeared_kinematics(
            fatjets, pt_jec_ak8, mass_jec_ak8, genpt8, valid_match8, events, cevalAK8, cfgAK8["JR_tag_MC"], algoAK8, sys_var="nom"
        )
        fatjets = ak.with_field(fatjets, pt_nom_ak8, "pt")
        fatjets = ak.with_field(fatjets, mass_nom_ak8, "mass")

        # ==========================================
        # STEP 4: Save All Variations (if flagged)
        # ==========================================
        if era == "RunIISummer20UL18":
            save_all_variations = False
        if save_all_variations:
            # --- JER SYSTEMATICS (AK4) ---
            for jer_dir in ["up", "down"]:
                pt_jer, mass_jer = _get_smeared_kinematics(
                    jets, pt_jec_ak4, mass_jec_ak4, genpt, valid_match, events, cevalAK4, cfgAK4["JR_tag_MC"], algoAK4, sys_var=jer_dir
                )
                jets = ak.with_field(jets, pt_jer, f"pt_jer_{jer_dir}")
                jets = ak.with_field(jets, mass_jer, f"mass_jer_{jer_dir}")

            # --- JES SYSTEMATICS (AK4) ---
            for cat_name, source_name in cfgAK4.get("jes_unc_mapping", {}).items():
                exact_json_key = f"{tagAK4}_{source_name}_{algoAK4}"
                if exact_json_key in cevalAK4:
                    shift = cevalAK4[exact_json_key].evaluate(jets.eta, pt_jec_ak4)
                    for jes_dir in ["up", "down"]:
                        c_jes = 1.0 + shift if jes_dir == "up" else 1.0 - shift
                        pt_final, mass_final = _get_smeared_kinematics(
                            jets, pt_jec_ak4 * c_jes, mass_jec_ak4 * c_jes, genpt, valid_match, events, cevalAK4, cfgAK4["JR_tag_MC"], algoAK4, sys_var="nom"
                        )
                        jets = ak.with_field(jets, pt_final, f"pt_{cat_name}_{jes_dir}")
                        jets = ak.with_field(jets, mass_final, f"mass_{cat_name}_{jes_dir}")

            # --- JER SYSTEMATICS (AK8) ---
            for jer_dir in ["up", "down"]:
                pt_jer, mass_jer = _get_smeared_kinematics(
                    fatjets, pt_jec_ak8, mass_jec_ak8, genpt8, valid_match8, events, cevalAK8, cfgAK8["JR_tag_MC"], algoAK8, sys_var=jer_dir
                )
                fatjets = ak.with_field(fatjets, pt_jer, f"pt_jer_{jer_dir}")
                fatjets = ak.with_field(fatjets, mass_jer, f"mass_jer_{jer_dir}")

            # --- JES SYSTEMATICS (AK8) ---
            for cat_name, source_name in cfgAK8.get("jes_unc_mapping", {}).items():
                exact_json_key = f"{tagAK8}_{source_name}_{algoAK8}"
                if exact_json_key in cevalAK8:
                    shift = cevalAK8[exact_json_key].evaluate(fatjets.eta, pt_jec_ak8)
                    for jes_dir in ["up", "down"]:
                        c_jes = 1.0 + shift if jes_dir == "up" else 1.0 - shift
                        pt_final, mass_final = _get_smeared_kinematics(
                            fatjets, pt_jec_ak8 * c_jes, mass_jec_ak8 * c_jes, genpt8, valid_match8, events, cevalAK8, cfgAK8["JR_tag_MC"], algoAK8, sys_var="nom"
                        )
                        fatjets = ak.with_field(fatjets, pt_final, f"pt_{cat_name}_{jes_dir}")
                        fatjets = ak.with_field(fatjets, mass_final, f"mass_{cat_name}_{jes_dir}")

    # ==========================================
    # STEP 3: Correct Subjets
    # ==========================================
    events = ak.with_field(events, jets, "Jet")
    events = ak.with_field(events, fatjets, "FatJet")
    
    if "SubJet" in events.fields and "area" in events.SubJet.fields:
        subjets = _correct_subjets(events.SubJet, events, cevalAK8, tagAK8, algoAK8, isMC)
        events = ak.with_field(events, subjets, "SubJet")
        fatjets = _recompute_softdrop(events)
        events = ak.with_field(events, fatjets, "FatJet")
        
    return events
