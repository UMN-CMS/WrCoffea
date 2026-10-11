"""WR Coffea analysis processor.

This module implements the Coffea `ProcessorABC` for the WR→Nℓ→ℓℓjj analysis.

High-level flow per chunk:
    1) Apply lumi mask for data (golden JSON).
    2) Build tight/loose lepton collections + AK4/AK8 jet collections.
    3) Build resolved and (optionally) boosted PackedSelections.
    4) Build nominal weights (+ optional systematic variations).
    5) Fill histograms and cutflows.

Output conventions:
    - Canonical histogram naming: output dict key == numeric axis name == ROOT stem.
    - All physics histograms carry categorical axes: (process, region, syst).
    - Cutflows contain labeled one-cut and cumulative histograms with raw
        counts and base-weight yields/variances, before per-region lepton SFs.
    - Data is blinded unless ``unblind``: no signal regions, and the ee/mumu
        cutflows stop before the mll cut (policy in ``wrcoffea.regions``).
    - MC carries ``event_weights[region]["positive"|"negative"]``: event
        counts vs log10|w| of the region's nominal weight (not with compute_sumw),
        and ``event_weights[region]["components"]``: one distribution per
        factor of that weight (see ``histograms.fill_event_weights``).

Notes for distributed execution (Dask/Condor):
    - correctionlib payloads are cached per worker process.
    - noisy but expected fallbacks (e.g. missing JetID inputs) are logged once per
        worker process via `_WARN_ONCE`.
"""

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
import awkward as ak
import numpy as np
import copy
import re
import os
import logging
from coffea.nanoevents.methods import vector
from wrcoffea.analysis_config import (
    CUTS, DY_RESHAPE_JSONS, DY_ZPT_REWEIGHT_JSONS, ELECTRON_JSONS, HEM_VETO, JME_JSONS, LUMI_JSONS, LUMI_UNC, LUMIS,
    DY_REWEIGHT_DATASETS, MUON_JSONS, PILEUP_JSONS, PREFIRING_ERAS, SIGNAL_KFACTOR_JSONS,
    SEL_MIN_TWO_AK4_JETS_PTETA, SEL_MIN_TWO_AK4_JETS_ID,
    SEL_TWO_PTETA_ELECTRONS, SEL_TWO_PTETA_MUONS, SEL_TWO_PTETA_EM,
    SEL_TWO_ID_ELECTRONS, SEL_TWO_ID_MUONS, SEL_TWO_ID_EM,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_HEM_VETO, SEL_MLL_GT200, SEL_MLLJJ_GT800, SEL_MLLJJ_LT800, SEL_MLL_GT400,
    SEL_TWO_TIGHT_LEPTONS, SEL_TWO_TIGHT_ELECTRONS, SEL_TWO_TIGHT_MUONS, SEL_TWO_TIGHT_EM,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53, SEL_MIN_TWO_AK4_JETS,
    SEL_60_MLL_150, DYCR_MLL_SPLIT, SEL_MLL_60_100, SEL_MLL_100_150,
    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
    SEL_ATLEAST1AK8_DPHI_GT2, SEL_AK8JETS_WITH_LSF,
    SEL_MUMU_DYCR, SEL_EE_DYCR, SEL_MUMU_SR, SEL_EE_SR,
    SEL_EMU_CR, SEL_MUE_CR, #SEL_JET_VETO_MAP,
    SEL_LEAD_IS_ELECTRON, SEL_LEAD_IS_MUON,
    SEL_NO_DY_PAIR, SEL_NO_EXTRA_TIGHT_SR, SEL_NO_EXTRA_TIGHT_CR,
    SEL_SF_LEPTON_IN_AK8, SEL_NO_OF_LEPTON_IN_AK8,
    SEL_OF_LEPTON_IN_AK8, SEL_NO_SF_LEPTON_IN_AK8,
    SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED,
)
from wrcoffea.dy_reshape import dy_reshape_weight
from wrcoffea.dy_reweight import DY_PROCESS_GROUP, dy_zpt_weight
from wrcoffea.signal_kfactor import SIGNAL_PROCESS_GROUP, mass_point_key, signal_kfactor_weight
from wrcoffea.pdf_weights import (
    ALPHAS_SOURCE, CHUNKS_KEY, MEMBERS_KEY, NO_PDF, PDF_SOURCE, PdfMembers, pdf_set, sample_lhaid,
)
from wrcoffea.era_utils import ERA_MAPPING
from wrcoffea.regions import (
    RESOLVED_REGIONS, BOOSTED_REGIONS, added_selections, build_regions, check_unblind_allowed, validate_variants,
)
from wrcoffea.systematics import (
    DIRECTIONS, MEMBERS, NOMINAL, SHAPE, WEIGHT,
    normalize_families, normalize_omitted, producible, production_config,
    resolve_syst_observables,
)
from wrcoffea.histograms import all_histogram_names
from wrcoffea.scale_factors import muon_sf, muon_sf_loose, muon_trigger_sf, electron_reco_sf, electron_id_sf, electron_trigger_sf, lsf_sf, pileup_weight, jet_veto_event_mask, jet_veto_jet_mask, jet_veto_mode, apply_jet_corrections, apply_electron_scale_smearing, verify_scale_smearing, verify_muon_scale_smearing, apply_muon_scale_smearing
from wrcoffea.histograms import (
    RESOLVED_HIST_SPECS, BOOSTED_HIST_SPECS, RESOLVED_2D_HIST_SPECS, BOOSTED_2D_HIST_SPECS,
    _booking_specs, create_hist, create_hist2D,
    fill_resolved_histograms, fill_boosted_histograms, fill_cutflows, fill_boosted_cutflows,
    fill_event_weights, GENWEIGHT_COMPONENT, NORMALIZATION_COMPONENT,
)

ak.behavior.update(vector.behavior)
logger = logging.getLogger(__name__)
_WARN_ONCE_ANALYZER: set = set()

# Cache correctionlib payloads per worker process (avoid re-reading JSON every chunk).
_CORRECTIONSET_CACHE = {}

# Warn-once cache (per worker process) to avoid log spam.
_WARN_ONCE: set[str] = set()


# Run 2 single-lepton trigger ORs per year (AN-19-083 Table 1; the UL16 eras share 2016).  Branch names
# without the "HLT_" prefix, as NanoAOD stores them under events.HLT.
# Run 2 single-lepton paths per year (AN-19-083 Table 1; SKFlat HNWRAnalyzer.C initializeAnalyzer).  2017 has no
# Ele115: it was off for the first ~5 fb-1 and Run 2 left it out.  The Run 2 data were the SingleElectron and
# SingleMuon primary datasets only (no SinglePhoton), which the UL16/UL16APV/UL17 data configs follow; the Photon
# paths still count on those streams, as in Run 2.
RUN2_TRIGGERS = {
    "RunIISummer20UL16APV": {"e": ("Ele27_WPTight_Gsf", "Photon175", "Ele115_CaloIdVT_GsfTrkIdT"),
                             "mu": ("Mu50", "TkMu50")},
    "RunIISummer20UL16": {"e": ("Ele27_WPTight_Gsf", "Photon175", "Ele115_CaloIdVT_GsfTrkIdT"),
                          "mu": ("Mu50", "TkMu50")},
    "RunIISummer20UL17": {"e": ("Ele35_WPTight_Gsf", "Photon200"),
                          "mu": ("Mu50", "OldMu100", "TkMu100")},
    "RunIISummer20UL18": {"e": ("Ele32_WPTight_Gsf", "Photon200", "Ele115_CaloIdVT_GsfTrkIdT"),
                          "mu": ("Mu50", "OldMu100", "TkMu100")},
}


def _l1_prefiring_weights(events, era):
    """(nominal, up, down) L1 prefiring weights of the events, from the NanoAOD ``L1PreFiringWeight`` record
    (``Nom``/``Up``/``Dn``; the flat ``L1PreFiringWeight_*`` branches when the schema did not group them)."""
    import numpy as np

    rec = getattr(events, "L1PreFiringWeight", None)
    if rec is not None and hasattr(rec, "Nom"):
        nom, up, dn = rec.Nom, rec.Up, rec.Dn
    elif hasattr(events, "L1PreFiringWeight_Nom"):
        nom, up, dn = events.L1PreFiringWeight_Nom, events.L1PreFiringWeight_Up, events.L1PreFiringWeight_Dn
    else:
        raise RuntimeError(f"era '{era}' is listed in config.yaml prefiring_eras but the events carry no "
                           "L1PreFiringWeight branches")
    return (np.asarray(nom, dtype=np.float64), np.asarray(up, dtype=np.float64), np.asarray(dn, dtype=np.float64))


class WrAnalysis(processor.ProcessorABC):
    """Main Coffea processor for WR analysis.

    Expected `events.metadata` keys (typical):
      - `era`: campaign key (e.g. Run3Summer22EE, RunIII2024Summer24)
      - `datatype`: "mc" or "data"
      - `physics_group`: high-level sample name (e.g. DYJets, EGamma, Signal)
      - `sample`: dataset identifier string
      - `xsec`, `genEventSumw` (MC only)

    Parameters
    - `mass_point`: optional signal mass label (carried through for naming).
    - `enabled_systs`: list of enabled systematic families (see
      ``wrcoffea.systematics.FAMILIES``).  Weight families add entries to the
      histogram ``syst`` axis from one selection pass; shape families re-run
      the selection on shifted kinematics; normalization families (``lumi``)
      are not filled at event level at all, the writer scales the nominal.
    - `syst_observables`: which histograms receive the non-Nominal entries.
      ``None``, empty, or ``["all"]`` selects every histogram (default);
      ``["fit"]`` selects the fit observables
      (``wrcoffea.systematics.FIT_OBSERVABLES``); otherwise an explicit list
      of histogram names.
    - `omit_systs`: source names (e.g. ``ElectronTrigSf``) deliberately left
      out although their family is enabled.  Recorded in the output so the
      omission is never mistaken for coverage.
    - `region`: which analysis region to run ("resolved", "boosted", or "both").
    - `compute_sumw`: if True, compute sum-of-genWeights over all input events
      before filtering instead of using the pre-computed ``genEventSumw`` from
      metadata. Histograms are filled without the ``/sumw`` normalization; the
      caller must divide by the accumulated ``_sumw`` per dataset after processing.
    - `dy_zpt_reweight`: apply the DY Z-pT (QCD x EW) reweighting to ``DYJets``
      for eras with a payload in ``config.yaml`` (default True).  The
      ``dyzpt`` systematic family needs it enabled.
    - `signal_kfactor`: apply the k(m_lN)/<k> shape reweighting to ``Signal``
      for eras with a payload in ``config.yaml`` (default True).
    - `dy_reshape`: apply the data-driven DY reshape vs jet pT to ``DYJets``
      for eras with a payload in ``config.yaml`` (default True).  It is
      derived on top of the Z-pT reweighting, so it is only applied when
      that is on too.
    - `unblind`: also build the signal regions for data and keep its full
      cutflows (default False: data is blinded, see ``wrcoffea.regions``).
      Only Run 2 data may be unblinded; MC is never blinded.  Recorded in
      the production configuration of every chunk.
    """
    def __init__(self, mass_point, enabled_systs=None, region="both", compute_sumw=False,
                 region_variants=None, syst_observables=None, omit_systs=None,
                 dy_zpt_reweight=True, signal_kfactor=True, dy_reshape=True,
                 save_events=False, unblind=False):
        self._signal_sample = mass_point
        self._compute_sumw = compute_sumw
        # --save-events: also keep the per-event (mass, nominal weight) of the
        # signal regions (resolved: mass_fourobject, boosted: mass_twoobject),
        # written as TTrees next to the histograms (save_hists).
        self._save_events = bool(save_events)
        # Blinding: data never builds the signal regions (or their variants)
        # and its ee/mumu cutflows stop before the mll cut, unless unblinded.
        self._blind = not bool(unblind)
        self._dy_zpt_reweight = bool(dy_zpt_reweight)
        self._dy_reshape = bool(dy_reshape) and self._dy_zpt_reweight
        self._signal_kfactor = bool(signal_kfactor)
        # Extra regions derived from nominal ones by dropping cuts (see
        # wrcoffea.regions).  Validated up front so bad names fail before processing.
        self._region_variants = validate_variants(region_variants or ())
        self._enabled_systs = normalize_families(enabled_systs)
        # None -> every histogram gets every variation; otherwise a frozenset
        # of histogram names (Nominal is always filled everywhere).
        self._syst_observables = resolve_syst_observables(syst_observables, all_histogram_names())
        self._omit_systs = normalize_omitted(omit_systs)
        self._sources_by_era = {}
        self._region = region.lower() if region else "both"
        if self._region not in ("resolved", "boosted", "both"):
            raise ValueError(f"Invalid region '{region}'. Must be 'resolved', 'boosted', or 'both'.")
        # What this run was asked to produce, recorded in every chunk so the
        # writer can demand every region and variation and refuse to merge
        # payloads made with different settings.
        self._syst_config = production_config(
            self._enabled_systs, self._omit_systs, self._syst_observables,
            region=self._region, variants=self._region_variants, blind=self._blind,
        )
        booking = _booking_specs()
        def _make_output():
            out = {}
            
            # --- 1D hists
            for name, (bins, label) in booking.items():
                out[name] = create_hist(name, bins, label)
                
            # --- 2D resolved hists
            for hist_key, xinfo, yinfo, _ in RESOLVED_2D_HIST_SPECS:
                xname, xbins, xlabel = xinfo
                yname, ybins, ylabel = yinfo
                
                out[hist_key] = create_hist2D(
                    xname, xbins, xlabel,
                    yname, ybins, ylabel
                )
                
            return out

        self.make_output = _make_output
        # self.make_output = lambda: {
        #     name: create_hist(name, bins, label)
        #     for name, (bins, label) in booking.items()
        # }
        
    def apply_lumi_mask(self, events, mc_campaign, is_data):
        """Apply golden-JSON lumi mask for data.

        Returns filtered events. For MC (or if no JSON is configured), returns
        the input events unchanged.
        """
        if not is_data:
            return events

        json_path = os.environ.get("LUMI_JSON") or LUMI_JSONS.get(mc_campaign)
        if json_path:
            try:
                mask = LumiMask(json_path)
                events = events[mask(events.run, events.luminosityBlock)]
                if len(events) == 0:
                    key = f"lumi_mask_empty::{mc_campaign}"
                    if key not in _WARN_ONCE:
                        _WARN_ONCE.add(key)
                        logger.warning(f"All events removed by lumi mask for era '{mc_campaign}'.")
            except OSError as e:
                logger.warning(f"Failed to load lumi JSON '{json_path}': {e}")
        else:
            logger.warning(f"No lumi JSON found for era '{mc_campaign}'. Data left unmasked.")

        return events
    def ecal_bad_calib_filter(self, events):
        run_mask = (events.run >= 362433) & (events.run <= 367144)
        dphi = np.abs(events.Jet.phi - events.MET.phi) # to be replaced with Puppi MET in the next iteration of skims for 2022 and 2023
        dphi = ak.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        bad_jet = (
            (events.Jet.pt > 50) &
            (events.Jet.eta > -0.5) & (events.Jet.eta < -0.1) &
            (events.Jet.phi > -2.1) & (events.Jet.phi < -1.8) &
            ((events.Jet.neEmEF > 0.9) | (events.Jet.chEmEF > 0.9)) &
            (dphi > 2.9)
        )

        reject = (events.MET.pt > 100) & ak.any(bad_jet, axis=1) & run_mask
        return reject

    # MET-filter flags required for every era (data, background MC, and signal).
    _COMMON_NOISE_FLAGS = (
        "goodVertices",
        "globalSuperTightHalo2016Filter",
        "EcalDeadCellTriggerPrimitiveFilter",
        "BadPFMuonFilter",
        "eeBadScFilter",
    )
    # Only present in newer NanoAOD; applied whenever available.
    _OPTIONAL_NOISE_FLAGS = ("BadPFMuonDzFilter", "hfNoisyHitsFilter")

    def _noise_flag(self, events, name, mc_campaign):
        """Return ``events.Flag.<name>``, or an all-True mask (warn once) if absent."""
        flags = getattr(events, "Flag", None)
        if flags is not None and hasattr(flags, name):
            return getattr(flags, name)
        key = f"noise_flag_missing::{mc_campaign}::{name}"
        if key not in _WARN_ONCE:
            _WARN_ONCE.add(key)
            logger.warning(
                "MET filter flag '%s' missing for era '%s'; treating it as passed.",
                name, mc_campaign,
            )
        return ak.ones_like(events.event, dtype=bool)

    def apply_noise_filter(self, events, mc_campaign):
        """Apply the recommended MET noise filters for the era.

        Applied to data, background MC, and signal alike (AN-19-083 Sec. 4
        requires all events to pass the MET filters).  Flags that only exist
        in newer NanoAOD are applied when present; a missing required flag is
        treated as passed and reported once per worker.
        """
        mask = ak.ones_like(events.event, dtype=bool)
        for name in self._COMMON_NOISE_FLAGS:
            mask = mask & self._noise_flag(events, name, mc_campaign)
        flags = getattr(events, "Flag", None)
        for name in self._OPTIONAL_NOISE_FLAGS:
            if flags is not None and hasattr(flags, name):
                mask = mask & getattr(flags, name)

        if mc_campaign in ("RunIISummer20UL17", "RunIISummer20UL18", "Run2Autumn18"):
            for name in ("HBHENoiseFilter", "HBHENoiseIsoFilter", "ecalBadCalibFilter"):
                mask = mask & self._noise_flag(events, name, mc_campaign)
        elif mc_campaign in ("RunIISummer20UL16APV", "RunIISummer20UL16"):
            # JME Run 2 UL recommendation: the HBHE filters in every year, ecalBadCalibFilter only for 2017 and 2018
            for name in ("HBHENoiseFilter", "HBHENoiseIsoFilter"):
                mask = mask & self._noise_flag(events, name, mc_campaign)
        elif mc_campaign in ("Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix"):
            mask = mask & ~self.ecal_bad_calib_filter(events)
        elif mc_campaign == "RunIII2024Summer24":
            mask = mask & self._noise_flag(events, "ecalBadCalibFilter", mc_campaign)
        else:
            logger.warning(f"No known Era {mc_campaign}. Noise filters are not applied.")
        return events[mask]

    def select_leptons(self, events):
        """Build tight/loose lepton collections and bookkeeping masks.

        Returns
        - `tight_leps`: concatenated (e+μ) tight leptons sorted by pT
        - `loose_leps`: concatenated (e+μ) loose leptons (tight excluded) sorted by pT
        - `masks`: per-object boolean masks used later for cutflows

        Notes
        - Tight = (pT/eta) AND (ID).
        - Loose = (pT/eta) AND (loose-ID), with tight leptons excluded.
        """
        # --- Supercluster Eta & Crack Veto ---
        if "deltaEtaSC" in events.Electron.fields:
            ele_sc_eta = events.Electron.eta + events.Electron.deltaEtaSC
        else:
            ele_sc_eta = events.Electron.eta
            
        abs_ele_sc_eta = np.abs(ele_sc_eta)
        ele_not_in_crack = (abs_ele_sc_eta <= 1.4442) | (abs_ele_sc_eta >= 1.566)
        
        # Split pT/eta (kinematics) and ID components.
        ele_pteta_mask = (events.Electron.pt > CUTS["lepton_pt_min"]) & (np.abs(events.Electron.eta) < CUTS["electron_eta_max"]) & ele_not_in_crack
        mu_pteta_mask  = (events.Muon.pt > CUTS["lepton_pt_min"])     & (np.abs(events.Muon.eta) < CUTS["muon_eta_max"])

        ele_id_mask = events.Electron.cutBased_HEEP
        mu_id_mask  = (events.Muon.highPtId == CUTS["muon_highPtId"]) & (events.Muon.tkRelIso < CUTS["muon_iso_max"])

        # --- TIGHT masks (unchanged behavior: pT/eta AND ID) ---
        electron_tight_mask = ele_pteta_mask & ele_id_mask
        muon_tight_mask     = mu_pteta_mask  & mu_id_mask

        # --- LOOSE base masks (as you had) ---
        electron_loose_base = ele_pteta_mask & (events.Electron.cutBased == 2)
        muon_loose_base     = mu_pteta_mask  & (events.Muon.highPtId == CUTS["muon_highPtId"])

        # --- LOOSE-only masks (exclude tights) ---
        electron_loose_mask = electron_loose_base & ~electron_tight_mask
        muon_loose_mask     = muon_loose_base     & ~muon_tight_mask

        # --- Filter collections ---
        tight_electrons = events.Electron[electron_tight_mask]
        tight_muons     = events.Muon[muon_tight_mask]
        loose_electrons = events.Electron[electron_loose_mask]
        loose_muons     = events.Muon[muon_loose_mask]

        # --- Add flavor tags (on original collections, before slimming) ---
        tight_electrons = ak.with_field(tight_electrons, "electron", "flavor")
        tight_muons     = ak.with_field(tight_muons,     "muon",     "flavor")
        loose_electrons = ak.with_field(loose_electrons, "electron", "flavor")
        loose_muons     = ak.with_field(loose_muons,     "muon",     "flavor")

        # --- Combine and sort by pT ---
        # Slim to common fields before concatenating so awkward creates a
        # regular record (not a union), avoiding field-access issues.
        def _to_candidate(col):
            # If the collection has deltaEtaSC (Electrons), use it. Otherwise (Muons), fill with 0s.
            d_eta_sc = col.deltaEtaSC if hasattr(col, "deltaEtaSC") else ak.zeros_like(col.eta)           
            return ak.zip(
                {
                    "pt": col.pt, 
                    "eta": col.eta, 
                    "phi": col.phi,
                    "mass": col.mass, 
                    "charge": col.charge, 
                    "flavor": col.flavor,
                    "pdgId" : col.pdgId,
                    "deltaEtaSC": d_eta_sc
                },
                with_name="PtEtaPhiMCandidate",
            )
        tight_leptons = ak.concatenate(
            [_to_candidate(tight_electrons), _to_candidate(tight_muons)], axis=1,
        )
        loose_leptons = ak.concatenate(
            [_to_candidate(loose_electrons), _to_candidate(loose_muons)], axis=1,
        )

        tight_leps = tight_leptons[ak.argsort(tight_leptons.pt, axis=1, ascending=False)]
        loose_leps = loose_leptons[ak.argsort(loose_leptons.pt, axis=1, ascending=False)]

        masks = {
            "ele_pteta": ele_pteta_mask,
            "mu_pteta": mu_pteta_mask,
            "ele_id": ele_id_mask,
            "mu_id": mu_id_mask,
            "mu_loose": muon_loose_mask,
            "ele_loose": electron_loose_mask,
        }

        return tight_leps, loose_leps, masks

    def select_jets(self, events, era):
        """Select AK4 and AK8 jets and return (collections, per-object masks).

        For eras with a JME JSON configured, AK4 PUPPI TightLeptonVeto is
        evaluated via correctionlib.  If the required NanoAOD fields are
        missing, fall back to ``Jet.jetId >= 6`` and warn once per worker.
        """
        # ---------- AK4 ----------
        ak4_pteta_mask = (events.Jet.pt > CUTS["ak4_pt_min"]) & (np.abs(events.Jet.eta) < CUTS["ak4_eta_max"])
        if "passJetVeto" in events.Jet.fields:
            # Run 2 jet veto (`jetveto_mode: jet`): a jet in a vetoed cell is out of acceptance.
            ak4_pteta_mask = ak4_pteta_mask & events.Jet.passJetVeto
        if era in JME_JSONS:
            try:
                ak4_id_mask = self.jetid_mask_ak4puppi(events.Jet, era, jetid_name="AK4PUPPI_TightLeptonVeto")
            except AttributeError as e:
                # Fallback for missing fields (shouldn't happen after config fix)
                key = f"jetid_fallback::{era}"
                if key not in _WARN_ONCE:
                    _WARN_ONCE.add(key)
                    logger.warning(
                        "correctionlib puppi JetID requested for era '%s' but required jet fields "
                        "are missing; falling back to Jet.jetId>=6 for this worker process. "
                        "(example error: %s)", era, e
                    )
                ak4_id_mask = events.Jet.jetId >= 6
        else:
            # For eras without jetid.json (the NanoAODv12 eras)
            # Use isTightLeptonVeto if available, otherwise use jetId >= 6
            if hasattr(events.Jet, 'isTightLeptonVeto'):
                ak4_id_mask = events.Jet.isTightLeptonVeto
            else:
                # NanoAODv12 and earlier: use jetId integer flag (6 = TightLeptonVeto)
                ak4_id_mask = events.Jet.jetId >= 6

        ak4_mask = ak4_pteta_mask & ak4_id_mask
        ak4_jets = events.Jet[ak4_mask]

        # ---------- AK8 ----------
        ak8_pteta_mask = (events.FatJet.pt > CUTS["ak8_pt_min"]) & (np.abs(events.FatJet.eta) < CUTS["ak8_eta_max"])
        ak8_extra_sel = (events.FatJet.msoftdrop > CUTS["ak8_msoftdrop_min"])
        ak8_mask = ak8_pteta_mask & ak8_extra_sel
        ak8_jets = events.FatJet[ak8_mask]

        masks = {
            "ak4_pteta": ak4_pteta_mask,
            "ak4_id": ak4_id_mask,
            "ak8_pteta": ak8_pteta_mask,
        }

        return ak4_jets, ak8_jets, masks

    def jetid_mask_ak4puppi(self, jets, era,jetid_name="AK4PUPPI_TightLeptonVeto"):
        """
        Returns a jagged boolean mask aligned with `jets` for AK4 PUPPI TightLeptonVeto.
        """
        import correctionlib

        json_path = JME_JSONS[era]
        ceval = _CORRECTIONSET_CACHE.get(json_path)
        if ceval is None:
            ceval = correctionlib.CorrectionSet.from_file(json_path)
            _CORRECTIONSET_CACHE[json_path] = ceval

        out = ceval[jetid_name].evaluate(
                jets.eta,
                jets.chHEF,
                jets.neHEF,
                jets.chEmEF,
                jets.neEmEF,
                jets.muEF,
                jets.chMultiplicity,
                jets.neMultiplicity,
                jets.chMultiplicity + jets.neMultiplicity
        )
        return ak.values_astype(out, np.bool_)

    def build_trigger_masks(self, events, mc_campaign):
        """
        Returns (e_trig, mu_trig, emu_trig) per-event booleans.
        Handles missing HLT paths gracefully by defaulting to False.
        """
        import awkward as ak
        import numpy as np

        n = len(events)
        HLT = getattr(events, "HLT", None)
        run = ERA_MAPPING.get(mc_campaign, {}).get("run")

        def hlt(name):
            if HLT is not None and hasattr(HLT, name):
                return getattr(HLT, name)
            return ak.Array(np.zeros(n, dtype=bool))  # fallback

        def any_of(names):
            out = hlt(names[0])
            for name in names[1:]:
                out = out | hlt(name)
            return out

        # Electrons by run period (Run 2: the per-year paths of AN-19-083 Table 1)
        if run == "RunII":
            e_trig = any_of(RUN2_TRIGGERS.get(mc_campaign, RUN2_TRIGGERS["RunIISummer20UL18"])["e"])
        else:  # Run3 or unknown
            e_trig = (hlt("Ele30_WPTight_Gsf") | hlt("Photon200")
                      | hlt("Ele115_CaloIdVT_GsfTrkIdT") | hlt("Ele50_CaloIdVT_GsfTrkIdT_PFJet165"))

        # Muons by run period
        if run == "RunII":
            mu_trig = any_of(RUN2_TRIGGERS.get(mc_campaign, RUN2_TRIGGERS["RunIISummer20UL18"])["mu"])
        elif run == "Run3":
            mu_trig = hlt("Mu50") | hlt("HighPtTkMu100") | hlt("CascadeMu100")
        else:  # unknown era — OR all known muon triggers as safe fallback
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100") | hlt("HighPtTkMu100") | hlt("CascadeMu100")

        emu_trig = mu_trig

        return e_trig, mu_trig, emu_trig

    @staticmethod
    def _hem_veto_ok(era, *leptons):
        """Per-event True where none of *leptons* is an electron in the HEM 15/16 window.

        Ported from the Run 2 ``HNWRAnalyzer::isHEMVetoEvent``: only the
        leptons a region is built from are inspected, on electron eta and phi.
        *leptons* are per-event optional candidates (None where absent); a
        missing candidate never vetoes.  Eras without a ``hem_veto`` entry in
        ``config.yaml`` return all True.
        """
        n = len(leptons[0])
        window = HEM_VETO.get(era)
        if window is None:
            return np.ones(n, dtype=bool)
        bad = np.zeros(n, dtype=bool)
        for lep in leptons:
            is_electron = ak.fill_none(abs(lep.pdgId) == 11, False)
            in_window = ak.fill_none(
                (lep.eta < window["eta_max"]) & (lep.phi > window["phi_min"]) & (lep.phi < window["phi_max"]),
                False,
            )
            bad |= np.asarray(is_electron & in_window, dtype=bool)
        return ~bad

    def _add_dycr_split_selections(self, selections, mll):
        """DY CR1 / CR2 halves for the ``dy-cr-split`` study, on *mll* (the tight pair in
        resolved, the DY candidate pair in boosted).  Filled only when a requested region
        variant uses them, so other runs keep their selections unchanged."""
        added = added_selections(getattr(self, "_region_variants", ()))
        if SEL_MLL_60_100 in added:
            selections.add(SEL_MLL_60_100,
                           ak.fill_none((mll > CUTS["mll_dy_low"]) & (mll < DYCR_MLL_SPLIT), False))
        if SEL_MLL_100_150 in added:
            selections.add(SEL_MLL_100_150,
                           ak.fill_none((mll >= DYCR_MLL_SPLIT) & (mll < CUTS["mll_dy_high"]), False))

    def resolved_selections(self, tight_leptons, ak4_jets, *, lepton_masks=None, jet_masks=None, triggers=None,
                            era=None):
        """
        Build PackedSelection for the resolved region.

        Baseline Criteria:
          - exactly 2 tight leptons
          - leading lepton pT > 60 GeV
          - subleading lepton pT > 53 GeV
          - at least two AK4 jets passing selection
          - ΔR > 0.4 between all pairs among {l1, l2, j1, j2}

        Resolved DY CR Criteria:
          - 60 < mll < 150 GeV
          - mlljj > 800 GeV  (built from l1, l2, j1, j2)

        Resolved SR Criteria:
          - mll > 400 GeV
          - mlljj > 800 GeV  (built from l1, l2, j1, j2)

                Plus flavor and trigger requirements.

                Implementation detail:
                    - Selections are built on padded leading objects (`ak.pad_none`) so
                        mass/ΔR expressions are defined even when objects are missing.
        """
        selections = PackedSelection()

        n_tight = ak.num(tight_leptons)
        n_ak4   = ak.num(ak4_jets)

        # flavor counts (per-event)
        n_tight_e  = ak.sum(tight_leptons.flavor == "electron", axis=1)
        n_tight_mu = ak.sum(tight_leptons.flavor == "muon",     axis=1)

        # flavor selections
        selections.add("two_tight_electrons", n_tight_e == 2)
        selections.add("two_tight_muons",     n_tight_mu == 2)
        selections.add("two_tight_em",        (n_tight_e == 1) & (n_tight_mu == 1))

        # Pad to always have 2 leptons/jets (None if missing)
        lpad = ak.pad_none(tight_leptons, 2)
        l1, l2 = lpad[:, 0], lpad[:, 1]

        jpad = ak.pad_none(ak4_jets, 2)
        j1, j2 = jpad[:, 0], jpad[:, 1]

        # Add criteria
        selections.add(SEL_TWO_TIGHT_LEPTONS,    n_tight == 2)
        selections.add("lead_tight_lepton_pt60", ak.fill_none(l1.pt > CUTS["lead_lepton_pt_min"], False))
        selections.add("sublead_tight_pt53",     ak.fill_none(l2.pt > CUTS["sublead_lepton_pt_min"], False))
        selections.add("min_two_ak4_jets",       n_ak4 >= 2)

        # Invariant masses built from the cleaned leading pair
        mll   = (l1 + l2).mass
        mlljj = (l1 + l2 + j1 + j2).mass

        # mll selections
        selections.add("60_mll_150",   ak.fill_none((mll > CUTS["mll_dy_low"]) & (mll < CUTS["mll_dy_high"]), False))
        selections.add("mll_gt200",    ak.fill_none(mll > CUTS["mll_sr_min"], False))
        selections.add("mll_gt400",    ak.fill_none(mll > CUTS["mll_sr_high_min"], False))
        selections.add("mlljj_gt800",  ak.fill_none(mlljj > CUTS["mlljj_min"], False))
        selections.add("mlljj_lt800",  ak.fill_none(mlljj < CUTS["mlljj_min"], False))
        self._add_dycr_split_selections(selections, mll)

        # ΔR > 0.4 requirements among {l1, l2, j1, j2}.
        dr_ll   = ak.fill_none(l1.deltaR(l2) > CUTS["dr_min"], False)
        dr_l1j1 = ak.fill_none(l1.deltaR(j1) > CUTS["dr_min"], False)
        dr_l1j2 = ak.fill_none(l1.deltaR(j2) > CUTS["dr_min"], False)
        dr_l2j1 = ak.fill_none(l2.deltaR(j1) > CUTS["dr_min"], False)
        dr_l2j2 = ak.fill_none(l2.deltaR(j2) > CUTS["dr_min"], False)
        dr_jj   = ak.fill_none(j1.deltaR(j2) > CUTS["dr_min"], False)

        selections.add("dr_all_pairs_gt0p4", dr_ll & dr_l1j1 & dr_l1j2 & dr_l2j1 & dr_l2j2 & dr_jj)

        # HEM 15/16 veto on the two tight leptons (2018 only; all True elsewhere).
        selections.add(SEL_HEM_VETO, self._hem_veto_ok(era, l1, l2))

        # Triggers
        if triggers is not None:
            e_trig, mu_trig, emu_trig = triggers
            selections.add("e_trigger",   e_trig)
            selections.add("mu_trigger",  mu_trig)
            selections.add("emu_trigger", emu_trig)

        if lepton_masks is None or jet_masks is None:
            raise ValueError("resolved_selections requires lepton_masks and jet_masks")

        # Jet-level counts per event
        n_ak4_pteta = ak.sum(jet_masks["ak4_pteta"], axis=1)
        n_ak4_id    = ak.sum(jet_masks["ak4_id"],    axis=1)
        selections.add("min_two_ak4_jets_pteta", n_ak4_pteta >= 2)
        selections.add("min_two_ak4_jets_id",    n_ak4_id    >= 2)

        # Lepton-level counts per event (separate by flavor)
        n_ele_pteta = ak.sum(lepton_masks["ele_pteta"], axis=1)
        n_mu_pteta  = ak.sum(lepton_masks["mu_pteta"],  axis=1)
        n_ele_id    = ak.sum(lepton_masks["ele_id"],    axis=1)
        n_mu_id     = ak.sum(lepton_masks["mu_id"],     axis=1)

        # Object-level cutflow rows.  These use "at least" counts so that an
        # extra lepton passing only one of (pT/eta, ID) does not reject an
        # event that the region (exactly two tight leptons) keeps; the tight
        # rows that follow in the cutflow pin the count to exactly two.
        # pT/eta-only
        selections.add("two_pteta_electrons", n_ele_pteta >= 2)
        selections.add("two_pteta_muons",     n_mu_pteta  >= 2)
        selections.add("two_pteta_em",        (n_ele_pteta >= 1) & (n_mu_pteta >= 1))

        # ID-only
        selections.add("two_id_electrons", n_ele_id >= 2)
        selections.add("two_id_muons",     n_mu_id  >= 2)
        selections.add("two_id_em",        (n_ele_id >= 1) & (n_mu_id >= 1))

        return selections
    
    # --- Boosted helper functions -------------------------------------------------
    def remove_lepton(self,loose, tight):
        """Remove objects within ΔR<0.01 of the provided tight lepton."""
        match = (loose.deltaR(tight))
        keep_mask = match >= CUTS["dr_loose_veto"]
        return loose[keep_mask]
    
    def ElectronCut(self, cut_bitmap, id_level=1):
        """Evaluate NanoAOD electron vid bitmap with an ID threshold.

        Fully vectorized version that:
          - ignores the isolation flag (cut index 7)
          - requires each considered cut to be >= `id_level`
        """
        n_flags = 10         # total number of bits (cuts)
        cut_size = 3         # 3 bits per cut
        ignore_flag = 7      # ignore the isolation flag
        mask_per_cut = (1 << cut_size) - 1  # 0b111 = 7

        passes = True
        for cut_nr in range(n_flags):
            if cut_nr == ignore_flag:
                continue
            value = (cut_bitmap >> (cut_nr * cut_size)) & mask_per_cut
            passes = passes & (value >= id_level)

        return passes
    
    def selectLooseElectrons(self, events):
        loose_noIso_mask = self.ElectronCut(events.Electron.vidNestedWPBitmap, id_level=2)
        loose_noIso_mask = ak.fill_none(loose_noIso_mask, False)

        # Ensure the HEEP flag is a true boolean array before bitwise ops.
        heep_flag = ak.fill_none(events.Electron.cutBased_HEEP, 0)
        heep_flag = heep_flag != 0
        # --- Supercluster Eta & Crack Veto ---
        if "deltaEtaSC" in events.Electron.fields:
            ele_sc_eta = events.Electron.eta + events.Electron.deltaEtaSC
        else:
            ele_sc_eta = events.Electron.eta
        abs_ele_sc_eta = np.abs(ele_sc_eta)
        ele_not_in_crack = (abs_ele_sc_eta <= 1.4442) | (abs_ele_sc_eta >= 1.566)
        
        loose_electrons = (
            (events.Electron.pt > CUTS["lepton_pt_min"])
            & (np.abs(events.Electron.eta) < CUTS["electron_eta_max"])
            & ele_not_in_crack
            & (heep_flag | loose_noIso_mask)
        )
        return events.Electron[loose_electrons]

    def selectLooseMuons(self, events):
        loose_muons = (events.Muon.pt > CUTS["lepton_pt_min"]) & (np.abs(events.Muon.eta) < CUTS["muon_eta_max"]) & (events.Muon.highPtId == CUTS["muon_highPtId"])
        return events.Muon[loose_muons]
    
    def selectAK8Jets(self,events,era):
        """Baseline AK8 selection used in boosted categories."""
        if era in JME_JSONS:
            try:
                ak8_id_mask = self.jetid_mask_ak4puppi(events.FatJet, era, jetid_name="AK8PUPPI_Tight")
            except AttributeError as e:
                # Fallback for missing fields (shouldn't happen after config fix)
                key = f"jetid_fallback::{era}"
                if key not in _WARN_ONCE:
                    _WARN_ONCE.add(key)
                    logger.warning(
                        "correctionlib puppi JetID requested for era '%s' but required jet fields "
                        "are missing; falling back to Jet.jetId>=6 for this worker process. "
                        "(example error: %s)", era, e
                    )
                ak8_id_mask = events.FatJet.isTight #jetId >= 6
        else:
            if hasattr(events.FatJet, 'isTight'):
                ak8_id_mask = events.FatJet.isTight
            else:
                ak8_id_mask = events.FatJet.jetId >= 6

        ak8_jets = (events.FatJet.pt > CUTS["ak8_pt_min"]) & (np.abs(events.FatJet.eta) < CUTS["ak8_eta_max"])  & (events.FatJet.msoftdrop > CUTS["ak8_msoftdrop_min"]) & ak8_id_mask
        return events.FatJet[ak8_jets]

    def selectAK8Jets_withLSF(self,events,era):
        """AK8 selection with LSF requirement used for boosted SR/CR definitions."""
        if era in JME_JSONS:
            try:
                ak8_id_mask = self.jetid_mask_ak4puppi(events.FatJet, era, jetid_name="AK8PUPPI_Tight")
            except AttributeError as e:
                # Fallback for missing fields (shouldn't happen after config fix)
                key = f"jetid_fallback::{era}"
                if key not in _WARN_ONCE:
                    _WARN_ONCE.add(key)
                    logger.warning(
                        "correctionlib puppi JetID requested for era '%s' but required jet fields "
                        "are missing; falling back to Jet.jetId>=6 for this worker process. "
                        "(example error: %s)", era, e
                    )
                ak8_id_mask = events.FatJet.isTight #jetId >= 6
        else:
            if hasattr(events.FatJet, 'isTight'):
                ak8_id_mask = events.FatJet.isTight
            else:
                ak8_id_mask = events.FatJet.jetId >= 6

        ak8_jets = (events.FatJet.pt > CUTS["ak8_pt_min"]) & (np.abs(events.FatJet.eta) < CUTS["ak8_eta_max"])  & (events.FatJet.msoftdrop > CUTS["ak8_msoftdrop_min"]) & (events.FatJet.lsf3 > CUTS["ak8_lsf3_min"]) & ak8_id_mask
        return events.FatJet[ak8_jets]

    def _no_extra_tight_leptons(self, looseMuons, looseElectrons, tight_lep, sublead_lep):
        """Veto events with extra tight leptons beyond tight_lep and sublead_lep.

        Counts loose muons (passing tight ISO) and loose electrons (passing HEEP)
        that are not ΔR-matched to the two candidate leptons.  Returns a per-event
        boolean that is True when no such extras exist.
        """
        dr_cut = CUTS["dr_loose_veto"]
        extra_tight_mu = ak.sum(
            (looseMuons.tkRelIso < CUTS["muon_iso_max"]) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > dr_cut, True)) &
            (ak.fill_none(looseMuons.deltaR(sublead_lep) > dr_cut, True)),
            axis=1
        )
        extra_tight_el = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > dr_cut, True)) &
            (ak.fill_none(looseElectrons.deltaR(sublead_lep) > dr_cut, True)),
            axis=1
        )
        return (extra_tight_mu == 0) & (extra_tight_el == 0)

    def boosted_selections(self, events, era, triggers=None):
        """Build boosted PackedSelection and the objects needed for boosted histogram filling.

        Returns a tuple:
          (selections,
           tight_lep,
           ak8_cand_dy, dy_loose_lep,
           ak8_cand, of_candidate, sf_candidate)

        Notes
        - Boosted selections are skipped entirely upstream if `FatJet/lsf3` is missing.
        - The returned per-event objects are used directly by `fill_boosted_histograms()`.
        """

        selections = PackedSelection()

        # Object selections.
        looseElectrons = self.selectLooseElectrons(events)
        looseMuons = self.selectLooseMuons(events)
        looseElectrons = ak.with_field(looseElectrons, "electron", "flavor")
        looseMuons = ak.with_field(looseMuons, "muon", "flavor")
        AK8Jets = self.selectAK8Jets(events,era)
        AK8Jets_withLSF = self.selectAK8Jets_withLSF(events,era)
        AK4Jets_inc, _, _ = self.select_jets(events, era)
        # define tight by querying loose
        tight_mask_e = (looseElectrons.cutBased_HEEP)
        tightElectrons_inc = looseElectrons[tight_mask_e]
        tightElectrons_inc = tightElectrons_inc[ak.argsort(tightElectrons_inc.pt, axis=1, ascending=False)]
        ## define loose 
        cut_bitmap = looseElectrons.vidNestedWPBitmap
        loose_e_mask = self.ElectronCut(cut_bitmap, id_level=2)
        looseElectrons = looseElectrons[loose_e_mask]
        ## -- tight muons --- ##
        tight_mask_mu = (looseMuons.tkRelIso < CUTS["muon_iso_max"])
        tightMuons_inc = looseMuons[tight_mask_mu]
        tightMuons_inc = tightMuons_inc[ak.argsort(tightMuons_inc.pt, axis=1, ascending=False)]

        # Slim to common fields before concatenating to avoid UnionArrays.
        # Keep deltaEtaSC so downstream electron SFs can use supercluster eta.
        def _to_boosted_candidate(col):
            d_eta_sc = col.deltaEtaSC if hasattr(col, "deltaEtaSC") else ak.zeros_like(col.eta)
            return ak.zip(
                {"pt": col.pt, "eta": col.eta, "phi": col.phi,
                 "mass": col.mass, "charge": col.charge, "pdgId": col.pdgId,
                 "flavor": col.flavor, "deltaEtaSC": d_eta_sc},
                with_name="PtEtaPhiMCandidate",
            )

        looseLeptons = ak.concatenate([_to_boosted_candidate(looseElectrons), _to_boosted_candidate(looseMuons)], axis=1)
        looseLeptons = looseLeptons[ak.argsort(looseLeptons.pt, axis=1, ascending=False)]
        tightLeptons_inc = ak.concatenate([_to_boosted_candidate(tightElectrons_inc), _to_boosted_candidate(tightMuons_inc)], axis=1)
        tightLeptons_inc = tightLeptons_inc[ak.argsort(tightLeptons_inc.pt, axis=1, ascending=False)]
        # Define a resolved-like tag and take boosted as the complement.
        has_two_leptons = ak.num(tightLeptons_inc) >= 2
        muons_padded = ak.pad_none(tightLeptons_inc, 2, axis=1)
        # Compute dr only for events with >=2 leptons.
        dr_l1l2 = ak.where(
            ak.num(tightLeptons_inc) >= 2,
            muons_padded[:,0].deltaR(muons_padded[:,1]),
            np.nan,
        )
        # For jets
        jpad = ak.pad_none(AK4Jets_inc, 2)
        j1, j2 = jpad[:, 0], jpad[:, 1]
        dr_j1j2   = ak.fill_none(j1.deltaR(j2) > 0.4, False)

        has_two_jets = ak.num(AK4Jets_inc) >= 2
        # dr_jl_min: compute only if both jets and leptons exist.
        has_j_and_l = (ak.num(AK4Jets_inc) >= 1) & (ak.num(tightLeptons_inc) >= 1)
        dr_jl_min = ak.where( has_j_and_l, ak.min(AK4Jets_inc[:, :2].nearest(tightLeptons_inc).deltaR(AK4Jets_inc[:, :2]), axis=1), ak.full_like(has_j_and_l, np.nan))
        # Build all 2 leptons × 2 jets pairs.                                                                                                                               
        dr_lj = ak.cartesian({"lep": tightLeptons_inc[:,:2], "jet": AK4Jets_inc[:,:2]}, axis=1)
        dr_lj_vals = dr_lj["lep"].deltaR(dr_lj["jet"])
        # Condition: all l-j separations > 0.4.                                                                                                                             
        dr_lj_ok = ak.all(dr_lj_vals > 0.4, axis=1)
        resolved = (((ak.num(tightElectrons_inc)  + (ak.num(tightMuons_inc))) == 2) & (ak.num(AK4Jets_inc) >= 2) & (dr_l1l2 > 0.4) & (dr_j1j2 > 0.4) & (dr_jl_min >0.4))
        boosted  = ~resolved
        #selections.add("boostedtag",boosted)

        # jpad = ak.pad_none(AK4Jets_inc, 2)
        # j1, j2 = jpad[:, 0], jpad[:, 1] 
        # dr_j1j2   = ak.fill_none(j1.deltaR(j2) > CUTS["dr_min"], False)

        # has_two_jets = ak.num(AK4Jets_inc) >= 2

        # # dr_jl_min: compute only if both jets and leptons exist.
        # has_j_and_l = (ak.num(AK4Jets_inc) >= 1) & (ak.num(tightLeptons_inc) >= 1)
        # dr_jl_min = ak.where(has_j_and_l, ak.min(AK4Jets_inc[:, :2].nearest(tightLeptons_inc).deltaR(AK4Jets_inc[:, :2]), axis=1), np.nan)
        # # Build all 2 leptons × 2 jets pairs.
        # dr_lj = ak.cartesian({"lep": tightLeptons_inc[:,:2], "jet": AK4Jets_inc[:,:2]}, axis=1)
        # dr_lj_vals = dr_lj["lep"].deltaR(dr_lj["jet"])
        # # Condition: all l-j separations > 0.4.
        # dr_lj_ok = ak.all(dr_lj_vals > CUTS["dr_min"], axis=1)
        # resolved = (((ak.num(tightElectrons_inc)  + (ak.num(tightMuons_inc))) == 2) & (ak.num(AK4Jets_inc) >= 2) & (dr_l1l2 > CUTS["dr_min"]) & (dr_j1j2 > CUTS["dr_min"]) & (dr_jl_min > CUTS["dr_min"])) #(dr_lj_ok))                                                                                                                                                                   
        # boosted  = ~resolved
        selections.add(SEL_BOOSTEDTAG, boosted)

        # Leading tight lepton pT > 60.
        tightLepton_padded = ak.pad_none(tightLeptons_inc,1,axis=1)
        tight_lep   = tightLepton_padded[:, 0]
        lead_pdgid  = ak.fill_none(abs(tight_lep.pdgId), 0)
        is_lead_mu  = lead_pdgid == 13
        is_lead_e   = lead_pdgid == 11
        is_tight_pt = tight_lep.pt > CUTS["lead_lepton_pt_min"]
        selections.add(SEL_LEAD_TIGHT_PT60_BOOSTED, is_tight_pt)
        # Remove this tight lepton from loose lepton selection.
        looseLeptons = self.remove_lepton(looseLeptons, tight_lep)

        # Same-flavor and other-flavor loose collections.
        sf_loose = looseLeptons[abs(looseLeptons.pdgId) == abs(tight_lep.pdgId)]
        of_loose = looseLeptons[abs(looseLeptons.pdgId) != abs(tight_lep.pdgId)]
        sf_loose = sf_loose[ak.argsort(sf_loose.pt, axis=1, ascending=False)]
        of_loose = of_loose[ak.argsort(of_loose.pt, axis=1, ascending=False)]

        # DY pair check.
        mll_pairs  = (tight_lep + sf_loose).mass
        mask_mll   = (mll_pairs > CUTS["mll_dy_low"]) & (mll_pairs < CUTS["mll_dy_high"])
        has_dy_pair = ak.any(mask_mll, axis=1)
        # Pick the loose SF lepton candidate (first passing the DY window).
        DY_loose_lep  = ak.firsts(sf_loose[mask_mll])
        # AK8 jet candidate.
        AK8Jets = AK8Jets[ak.argsort(AK8Jets.pt, axis=1, ascending=False)]
        flag_ak8Jet = ak.num(AK8Jets)>=1
        AK8Jets = ak.pad_none(AK8Jets, 1, axis=1)
        dphi       = ak.fill_none(abs(AK8Jets.delta_phi(tight_lep)),0.0)
        has_ak8_dphi_gt2 = ak.any(dphi > CUTS["dphi_boosted_min"], axis=1)
        ak8_mask   = dphi > CUTS["dphi_boosted_min"]
        AK8_cand_dy   = ak.firsts(AK8Jets[ak8_mask])
        # Require at least one AK8 jet with Δφ(jet, tight lepton) > 2.
        selections.add(SEL_ATLEAST1AK8_DPHI_GT2, flag_ak8Jet & has_ak8_dphi_gt2)
        
        # Case 1: DY CR.
        dr_dy  = AK8_cand_dy.deltaR(DY_loose_lep)
        mlj_dy = ak.where(dr_dy < CUTS["dr_ak8_loose"],
                          (tight_lep + AK8_cand_dy).mass,
                          (tight_lep + DY_loose_lep + AK8_cand_dy).mass)
        mll_dy = (tight_lep + DY_loose_lep).mass
        pt_dilept_dy = (tight_lep + DY_loose_lep).pt
        pt_lj_dy = ak.where(dr_dy < CUTS["dr_ak8_loose"],
                          (tight_lep + AK8_cand_dy).pt,
                          (tight_lep + DY_loose_lep + AK8_cand_dy).pt)
        sublead_pdgID = abs(DY_loose_lep.pdgId)
        is_sublead_mu = sublead_pdgID == 13
        is_sublead_e = sublead_pdgID ==11
        DYCR_mask = has_dy_pair & (mlj_dy > CUTS["mlljj_min"])
        selections.add(SEL_DYCR_MASK, DYCR_mask)
        self._add_dycr_split_selections(selections, mll_dy)
        # Veto extra tight leptons for DY CR.
        no_extra_tight_dyCR = self._no_extra_tight_leptons(
            looseMuons, looseElectrons, tight_lep, DY_loose_lep)

        # Case 2: SR (no DY, SF near AK8, no OF near AK8).
        flag_ak8Jet_lsf = ak.num(AK8Jets_withLSF)>=1
        AK8Jets_withLSF = ak.pad_none(AK8Jets_withLSF, 1, axis=1)
        dphi_lsf       = ak.fill_none(abs(AK8Jets_withLSF.delta_phi(tight_lep)),0.0)
        has_ak8_dphi_gt2_lsf = ak.any(dphi_lsf > CUTS["dphi_boosted_min"], axis=1)
        ak8_mask_lsf   = dphi_lsf > CUTS["dphi_boosted_min"]
        AK8_cand   = ak.firsts(AK8Jets_withLSF[ak8_mask_lsf])
        selections.add(SEL_AK8JETS_WITH_LSF, flag_ak8Jet_lsf & has_ak8_dphi_gt2_lsf)
        dr_sf = AK8_cand.deltaR(sf_loose)
        mask_sf = dr_sf < CUTS["dr_ak8_loose"]
        # SF lepton candidate passing dR condition.
        sf_candidate = ak.firsts(sf_loose[mask_sf])
        sf_exist = ak.num(sf_loose[mask_sf])>=1
        dr_of = AK8_cand.deltaR(of_loose)
        mask_of = dr_of < CUTS["dr_ak8_loose"]
        # OF lepton candidate passing dR condition.
        of_candidate = ak.firsts(of_loose[mask_of])
        of_exist = ak.num(of_loose[mask_of])>=1

        is_sr = (~has_dy_pair) & (~ak.is_none(sf_candidate)) & ak.is_none(of_candidate) #sf_exist & (~of_exist) #(~ak.is_none(sf_candidate)) & ak.is_none(of_candidate)    
        sublead_pdgID = abs(sf_candidate.pdgId)
        is_sublead_mu_sr = sublead_pdgID == 13
        is_sublead_e_sr = sublead_pdgID ==11
        mlj_sr   = (tight_lep + AK8_cand).mass
        mll_sr = (tight_lep + sf_candidate).mass
        pt_dilept_sr  = (tight_lep + AK8_cand).pt
        pt_lj_sr = (tight_lep + sf_candidate).pt

        # Mass requirements are separate selections (SEL_MLL_GT200_BOOSTED,
        # SEL_MLJ_GT800_BOOSTED) so regions can be composed with or without them.
        SR_mask = is_sr
        # Veto extra tight leptons for SR.
        no_extra_tight_sr = self._no_extra_tight_leptons(
            looseMuons, looseElectrons, tight_lep, sf_candidate)

        # Case 3: Flavor CR (no DY, OF near AK8).
        is_cr = (~has_dy_pair) & (~ak.is_none(of_candidate))  & ak.is_none(sf_candidate) #(~sf_exist) & of_exist #(~ak.is_none(of_candidate))  & ak.is_none(sf_candidate)  
        sublead_pdgID = abs(of_candidate.pdgId)
        is_sublead_mu_cr = sublead_pdgID == 13
        is_sublead_e_cr = sublead_pdgID ==11

        mlj_cr  = (tight_lep + AK8_cand).mass
        mll_cr = (tight_lep + of_candidate).mass
        pt_dilept_cr  = (tight_lep + AK8_cand).pt
        pt_lj_cr = (tight_lep + of_candidate).pt

        CR_mask = is_cr
        # Veto extra tight leptons for flavor CR.
        no_extra_tight_flav_cr = self._no_extra_tight_leptons(
            looseMuons, looseElectrons, tight_lep, of_candidate)

        # Expanded boosted cutflow selections.
        selections.add(SEL_LEAD_IS_ELECTRON, is_lead_e)
        selections.add(SEL_LEAD_IS_MUON, is_lead_mu)
        selections.add(SEL_NO_DY_PAIR, ~has_dy_pair)
        selections.add(SEL_SF_LEPTON_IN_AK8, ~ak.is_none(sf_candidate))
        selections.add(SEL_NO_OF_LEPTON_IN_AK8, ak.is_none(of_candidate))
        selections.add(SEL_OF_LEPTON_IN_AK8, ~ak.is_none(of_candidate))
        selections.add(SEL_NO_SF_LEPTON_IN_AK8, ak.is_none(sf_candidate))
        selections.add(SEL_NO_EXTRA_TIGHT_SR, no_extra_tight_sr)
        selections.add(SEL_NO_EXTRA_TIGHT_CR, no_extra_tight_flav_cr)
        mll_boosted = ak.where(~ak.is_none(sf_candidate), mll_sr, mll_cr)
        selections.add(SEL_MLL_GT200_BOOSTED, ak.fill_none(mll_boosted > CUTS["mll_sr_min"], False))
        selections.add(SEL_MLJ_GT800_BOOSTED, ak.fill_none(mlj_sr > CUTS["mlljj_min"], False))

        # Triggers.
        if triggers is not None:
            eTrig, muTrig, emu_trig = triggers
            selections.add(SEL_E_TRIGGER, eTrig)
            selections.add(SEL_MU_TRIGGER, muTrig)

        # Region assignments.
        mumu_dy_cr = muTrig & DYCR_mask & is_lead_mu & no_extra_tight_dyCR
        ee_dy_cr   = eTrig & DYCR_mask & is_lead_e & no_extra_tight_dyCR
        emu_cr = eTrig  & CR_mask & is_lead_e & no_extra_tight_flav_cr # lead e, loose mu                                                                        
        mue_cr = muTrig & CR_mask & is_lead_mu & no_extra_tight_flav_cr # lead mu, loose e                                                                        
        ee_sr = eTrig & SR_mask & is_lead_e & no_extra_tight_sr
        mumu_sr = muTrig & SR_mask &  is_lead_mu & no_extra_tight_sr
        # HEM 15/16 veto (2018 only) on the leading tight lepton and the
        # sub-leading candidate each region is built from.
        hem_ok_dy = ak.Array(self._hem_veto_ok(era, tight_lep, DY_loose_lep))
        hem_ok_sr = ak.Array(self._hem_veto_ok(era, tight_lep, sf_candidate))
        hem_ok_cr = ak.Array(self._hem_veto_ok(era, tight_lep, of_candidate))
        selections.add(SEL_MUMU_DYCR, mumu_dy_cr & is_sublead_mu & hem_ok_dy)
        selections.add(SEL_EE_DYCR,  ee_dy_cr & is_sublead_e & hem_ok_dy)
        selections.add(SEL_MUMU_SR,  mumu_sr & is_sublead_mu_sr & hem_ok_sr)
        selections.add(SEL_EE_SR,    ee_sr & is_sublead_e_sr & hem_ok_sr)
        selections.add(SEL_EMU_CR,   emu_cr & is_sublead_mu_cr & hem_ok_cr)
        selections.add(SEL_MUE_CR,   mue_cr & is_sublead_e_cr & hem_ok_cr)

        return selections, tight_lep, AK8_cand_dy,DY_loose_lep, AK8_cand,of_candidate, sf_candidate

    def build_event_weights(self, events, metadata, is_mc):
        """
        Base event weights (no lepton SFs — those are applied per-region):
          - MC: genWeight × xsec × lumi / sumw + pileup + DY Z-pT reweight
          - Data: unit weights
        """
        n = len(events)
        # Individual factors are kept for the event-weight component histograms.
        weights = Weights(n, storeIndividual=True)

        if is_mc:
            # --- Base Weights ---
            lumi = float(LUMIS[metadata.get("era")])
            xsec = float(metadata.get("xsec"))

            if self._compute_sumw:
                event_weight = events.genWeight * xsec * lumi * 1000.0
            else:
                sumw = float(metadata.get("genEventSumw"))
                if sumw == 0.0:
                    raise ZeroDivisionError(
                        f"genEventSumw is zero for dataset '{metadata.get('sample')}'."
                    )
                event_weight = events.genWeight * xsec * lumi * 1000.0 / sumw

            weights.add("event_weight", event_weight)

            # Pileup reweighting.
            era = metadata.get("era")
            if era in PILEUP_JSONS:
                pu_nom, pu_up, pu_down = pileup_weight(events, era)
                weights.add("pileup", pu_nom, weightUp=pu_up, weightDown=pu_down)

            # L1 prefiring (Run 2 UL): the NanoAOD per-event weight, ECAL + muon
            # in 2016/2017, muon only in 2018 (config.yaml prefiring_eras).
            if era in PREFIRING_ERAS:
                pf_nom, pf_up, pf_down = _l1_prefiring_weights(events, era)
                weights.add("prefiring", pf_nom, weightUp=pf_up, weightDown=pf_down)

            # DY Z-pT reweighting (QCD NLO/LO x EW, ported from the Run 2
            # HNWRAnalyzer).  Non-trivial only for the DY sample the payload was
            # derived for; other MC (and other DY samples, e.g. NLO) gets the
            # identity so the variation labels still exist for coverage.
            dy = None
            if self._dy_zpt_reweight and era in DY_ZPT_REWEIGHT_JSONS:
                dy = dy_zpt_weight(events, era, is_dy=self._dy_reweight_target(metadata))
                weights.add("dy_zpt", dy["nominal"])

            # Signal NLO/LO k-factor vs gen m(lN), divided by the per-mass-point
            # average (shape only; the configured xsec already carries <k>).
            if self._signal_kfactor and era in SIGNAL_KFACTOR_JSONS \
                    and metadata.get("physics_group") == SIGNAL_PROCESS_GROUP:
                kf = signal_kfactor_weight(
                    events, era, mass_point=self._mass_point_key(metadata), is_signal=True,
                )
                weights.add("signal_kfactor", kf["nominal"])

            syst_weights = {NOMINAL: weights.weight()}
            # "lumi" is a normalization systematic: it is never filled at
            # event level.  save_hists scales the nominal histogram instead.

            # Optional pileup uncertainty, gated like every other source:
            # family enabled, inputs available for the era, not omitted.
            if "Pileup" in self._producible_names(era, WEIGHT):
                syst_weights["PileupUp"] = weights.weight(modifier="pileupUp")
                syst_weights["PileupDown"] = weights.weight(modifier="pileupDown")
            if "L1Prefiring" in self._producible_names(era, WEIGHT):
                syst_weights["L1PrefiringUp"] = weights.weight(modifier="prefiringUp")
                syst_weights["L1PrefiringDown"] = weights.weight(modifier="prefiringDown")

            # Theory weights (QCD scales, parton shower): normalized per-event
            # ratios, relative to the nominal like the DY variations below.
            for name, (w_up, w_dn) in self._theory_variations(events, metadata, era).items():
                syst_weights[f"{name}Up"] = syst_weights[NOMINAL] * w_up
                syst_weights[f"{name}Down"] = syst_weights[NOMINAL] * w_dn

            # DY Z-pT variations: swap the nominal reweight for the varied one.
            if dy is not None:
                allowed = self._producible_names(era, WEIGHT)
                safe_nom = np.where(dy["nominal"] > 0, dy["nominal"], 1.0)
                for name, (dy_up, dy_down) in dy["variations"].items():
                    if name in allowed:
                        syst_weights[f"{name}Up"] = syst_weights[NOMINAL] * (dy_up / safe_nom)
                        syst_weights[f"{name}Down"] = syst_weights[NOMINAL] * (dy_down / safe_nom)

        else:  # is_data
            weights.add("data", np.ones(n, dtype=np.float32))
            syst_weights = {
                "Nominal":  weights.weight(),
            }

        return weights, syst_weights

    @staticmethod
    def _event_weight_factors(events, metadata, weights):
        """{name: per-event array} of the factors of the base MC weight.

        The ``event_weight`` term is split into the generator weight and the
        xsec * lumi / sumw normalization; every other entry of *weights*
        (pileup, DY Z-pT, signal k-factor) is its own factor.  The product of
        the factors is ``weights.weight()``.
        """
        lumi = float(LUMIS[metadata.get("era")])
        normalization = float(metadata.get("xsec")) * lumi * 1000.0 / float(metadata.get("genEventSumw"))
        factors = {
            GENWEIGHT_COMPONENT: np.asarray(events.genWeight, dtype=np.float64),
            NORMALIZATION_COMPONENT: np.full(len(events), normalization),
        }
        for name in weights.weightStatistics:
            if name != "event_weight":
                factors[name] = np.asarray(weights.partial_weight(include=[name]), dtype=np.float64)
        return factors

    # (up, down) indices in the NanoAOD 9-weight LHEScaleWeight order
    # [muR,muF] = [0.5,0.5] [0.5,1] [0.5,2] [1,0.5] [1,1] [1,2] [2,0.5] [2,1] [2,2]
    _SCALE_INDEX = {"ScaleMuR": (7, 1), "ScaleMuF": (5, 3), "ScaleMuRF": (8, 0)}
    # Samples that store the variations without the nominal have 8 weights
    # (e.g. TTLNu-1Jets; branch title: [0] MUF=0.5 MUR=0.5, [1] MUF=1 MUR=0.5,
    # [2] MUF=2 MUR=0.5, [3] MUF=0.5 MUR=1, [4] MUF=2 MUR=1, [5] MUF=0.5 MUR=2,
    # [6] MUF=1 MUR=2, [7] MUF=2 MUR=2), i.e. the 9-weight order without [1,1].
    _SCALE_INDEX_8 = {"ScaleMuR": (6, 1), "ScaleMuF": (4, 3), "ScaleMuRF": (7, 0)}
    # PSWeight order: [ISR x2, FSR x2, ISR x0.5, FSR x0.5]
    _PS_INDEX = {"PSISR": (0, 2), "PSFSR": (1, 3)}

    def _theory_variations(self, events, metadata, era):
        """{name: (w_up, w_down)} per-event multiplicative variations for the
        enabled theory sources.  Each weight is the generator ratio divided by
        the dataset-level ratio from the config metadata (``LHEScaleSumw`` /
        ``PSSumw`` written by scripts/gen_event_sumw.py), so the dataset's
        total yield is unchanged.  Missing config ratios are errors.  A dataset
        without the branch (pure-Pythia 8 WW/WZ/ZZ have no LHEScaleWeight) gets
        the nominal weight with one warning per dataset.  Within a dataset, an
        event with NO weights (one file each in 2022 ZZZ and the 22EE WR
        MWR6000_N1500 point, see docs/skimming.md) also gets the nominal weight,
        with a warning; any other count mismatch is an error.  Samples that
        store 8 scale weights (no nominal, e.g. TTLNu-1Jets) use the 8-weight
        order; the dataset ratio then has 8 entries as well."""
        allowed = self._producible_names(era, WEIGHT)
        out = {}
        sample = metadata.get("sample") or metadata.get("dataset")
        for family, index, branch, key in (("scale", self._SCALE_INDEX, "LHEScaleWeight", "LHEScaleSumw"),
                                           ("ps", self._PS_INDEX, "PSWeight", "PSSumw")):
            names = [n for n in index if n in allowed]
            if not names:
                continue
            if branch not in events.fields:
                # A sample generated without that step (pure Pythia 8 dibosons
                # have no LHE scale weights) carries no such variation: fill it
                # with the nominal weight so the group's varied histogram keeps
                # the sample, and say so loudly, once per dataset.
                key_w = f"theory_nominal::{family}::{sample}"
                if key_w not in _WARN_ONCE_ANALYZER:
                    _WARN_ONCE_ANALYZER.add(key_w)
                    logger.warning("--systs %s: dataset '%s' has no %s branch; its %s variations "
                                   "are filled with the NOMINAL weight (no theory uncertainty "
                                   "defined for this sample).", family, sample, branch, "/".join(names))
                ones = np.ones(len(events), dtype=np.float64)
                for name in names:
                    out[name] = (ones, ones)
                continue
            ratios = metadata.get(key)
            n_allowed = (9, 8) if family == "scale" else (4,)
            if not ratios or len(ratios) not in n_allowed:
                raise ValueError(
                    f"--systs {family}: metadata of '{sample}' has no "
                    f"{' or '.join(map(str, n_allowed))}-entry '{key}' (found "
                    f"{len(ratios) if ratios else 0}). Regenerate the fileset from skims that carry "
                    "the per-dataset weighted means, or run scripts/gen_event_sumw.py on the config.")
            n_exp = len(ratios)
            if family == "scale" and n_exp == 8:
                index = self._SCALE_INDEX_8
            counts = ak.to_numpy(ak.num(events[branch]))
            if len(counts) and not np.all((counts == n_exp) | (counts == 0)):
                raise ValueError(
                    f"--systs {family}: '{sample}' has {sorted(set(counts.tolist()))} {branch} "
                    f"entries per event, expected {n_exp}.")
            # Events with NO weights at all come from a production file written without
            # them (one file each in 2022 ZZZ and the 22EE WR MWR6000_N1500 point; see
            # docs/skimming.md "Known input defects"). They get the nominal weight (factor
            # exactly 1): the dataset-level ratio in the metadata is computed from the
            # files that do have the weights, so the total yield stays unchanged.
            missing = counts == 0
            if missing.any():
                key_w = f"theory_partial::{family}::{sample}"
                if key_w not in _WARN_ONCE_ANALYZER:
                    _WARN_ONCE_ANALYZER.add(key_w)
                    logger.warning("--systs %s: %d of %d events of '%s' in this chunk have no %s; "
                                   "their %s variations use the NOMINAL weight (see docs/skimming.md, "
                                   "\"Known input defects\").", family, int(missing.sum()), len(counts),
                                   sample, branch, "/".join(names))
            w = ak.to_numpy(ak.fill_none(ak.pad_none(events[branch], n_exp, clip=True), 1.0))
            r = np.asarray(ratios, dtype=np.float64)
            for name in names:
                iu, idn = index[name]
                up, down = w[:, iu] / r[iu], w[:, idn] / r[idn]
                if missing.any():
                    up = np.where(missing, 1.0, up)
                    down = np.where(missing, 1.0, down)
                out[name] = (up, down)
        return out

    def _pdf_members(self, events, metadata, era, is_mc):
        """``(lhaid, PdfMembers)`` of this chunk for the ``pdf`` family.

        ``(None, None)`` when the family is off or for data; ``(NO_PDF, None)``
        for a sample generated without LHEPdfWeight (pure Pythia 8 WW/WZ/ZZ),
        which then contributes no members (with a warning).  The sample's set
        comes from the era's pdf_sets table.
        """
        names = [n for n in (PDF_SOURCE, ALPHAS_SOURCE) if n in self._producible_names(era, MEMBERS)]
        if not is_mc or not names:
            return None, None
        sample = metadata.get("sample") or metadata.get("dataset")
        lhaid = sample_lhaid(era, sample)
        has_branch = "LHEPdfWeight" in events.fields
        if lhaid is None:
            if has_branch and len(events) and ak.any(ak.num(events.LHEPdfWeight) > 0):
                raise ValueError(f"--systs pdf: '{sample}' has LHEPdfWeight but the {era} pdf_sets "
                                 "table lists no PDF set for it; rerun scripts/pdf_sets.py")
            key = f"pdf_none::{sample}"
            if key not in _WARN_ONCE_ANALYZER:
                _WARN_ONCE_ANALYZER.add(key)
                logger.warning("--systs pdf: dataset '%s' has no LHEPdfWeight; it contributes no PDF "
                               "members (no PDF uncertainty defined for this sample).", sample)
            return NO_PDF, None
        if not has_branch:
            raise ValueError(f"--systs pdf: '{sample}' (LHA {lhaid}) has no LHEPdfWeight branch")
        ratios = metadata.get("LHEPdfSumw")
        if not ratios:
            raise ValueError(
                f"--systs pdf: metadata of '{sample}' has no LHEPdfSumw. Regenerate the fileset from "
                "skims that carry the per-dataset weighted means, or run scripts/gen_event_sumw.py.")
        pdfset = pdf_set(lhaid)
        members = PdfMembers(events.LHEPdfWeight, ratios, pdfset)
        if members.n_missing:
            key = f"pdf_partial::{sample}"
            if key not in _WARN_ONCE_ANALYZER:
                _WARN_ONCE_ANALYZER.add(key)
                logger.warning("--systs pdf: %d of %d events of '%s' in this chunk have no LHEPdfWeight; "
                               "they enter every member with the nominal weight.",
                               members.n_missing, len(events), sample)
        if ALPHAS_SOURCE in names and not pdfset.alphas_members:
            key = f"pdf_no_alphas::{sample}"
            if key not in _WARN_ONCE_ANALYZER:
                _WARN_ONCE_ANALYZER.add(key)
                logger.warning("--systs pdf: set %s of '%s' has no alpha_s members; no alpha_s "
                               "shift for this sample.", pdfset.name, sample)
        return str(lhaid), members

    def _dy_reshape_for_region(self, syst_weights, kind, jet_pt, era, is_dy, is_mc,
                               *, factors=None):
        """Multiply the DY reshape ratio (and its variations) into region weights.

        *kind* is ``"resolved"`` or ``"boosted"``, *jet_pt* the per-event
        leading AK4 pT or AK8-candidate pT the ratio is binned in.  *is_dy*
        is ``_dy_reweight_target(metadata)``: only the DY sample the payload
        was derived for is reshaped; other MC receives the identity so the
        variation labels still exist for coverage.  Returns *syst_weights*
        unchanged when the era has no payload or the reshape is disabled.
        *factors*, when given, receives the nominal ratio as ``"dy_reshape"``.
        """
        if not isinstance(is_dy, (bool, np.bool_)):
            raise TypeError(f"is_dy must be a bool (from _dy_reweight_target), got {is_dy!r}")
        if not is_mc or not self._dy_reshape:
            return syst_weights
        rw = dy_reshape_weight(jet_pt, era, kind, is_dy=bool(is_dy))
        if rw is None:
            return syst_weights
        nom = rw["nominal"]
        if factors is not None:
            factors["dy_reshape"] = np.asarray(nom, dtype=np.float64)
        result = {k: v * nom for k, v in syst_weights.items()}
        if NOMINAL in syst_weights:
            allowed = self._producible_names(era, WEIGHT)
            safe_nom = np.where(nom > 0, nom, 1.0)
            for name, (up, down) in rw["variations"].items():
                if name in allowed:
                    result[f"{name}Up"] = result[NOMINAL] * (up / safe_nom)
                    result[f"{name}Down"] = result[NOMINAL] * (down / safe_nom)
        return result

    @staticmethod
    def _dy_reweight_target(metadata):
        """Whether this dataset is the DY sample its era's Z-pT reweight and reshape
        were derived for (config.yaml ``dy_reweight_datasets``, matched against the
        dataset name).  Any other dataset, including other DY samples of the era
        (NLO, LO inclusive, ...), gets the identity: an NLO/LO correction must
        never be applied to an NLO sample.
        """
        if metadata.get("physics_group") != DY_PROCESS_GROUP:
            return False
        era = metadata.get("era")
        pattern = DY_REWEIGHT_DATASETS.get(era)
        dataset = str(metadata.get("sample") or metadata.get("dataset") or "")
        if pattern is not None and re.search(pattern, dataset):
            return True
        if era in DY_ZPT_REWEIGHT_JSONS or era in DY_RESHAPE_JSONS:
            key = f"dy_reweight_not_for_sample::{era}::{dataset}"
            if key not in _WARN_ONCE_ANALYZER:
                _WARN_ONCE_ANALYZER.add(key)
                logger.warning("DY dataset %s is not the sample the %s DY Z-pT reweight / reshape "
                               "were derived for (%r): not reweighted.", dataset, era, pattern)
        return False

    def _mass_point_key(self, metadata):
        """``WR<m>_N<m>`` for a signal dataset, from its name or ``--mass``."""
        for name in (metadata.get("dataset"), metadata.get("sample"), self._signal_sample):
            key = mass_point_key(name)
            if key is not None:
                return key
        return None

    @staticmethod
    def _region_trigger(region):
        """Return which single-lepton trigger selects *region*: ``"e"`` or ``"mu"``.

        Mirrors the region definitions in ``_fill_resolved`` / ``_fill_boosted``:
        the ee regions and the boosted e+mu-jet flavor CR (leading electron)
        are selected by the electron triggers; the mumu regions, the resolved
        flavor CR, and the boosted mu+e-jet flavor CR by the muon triggers.
        """
        if "ee" in region or "emu" in region:
            return "e"
        return "mu"

    @staticmethod
    def _jet_lepton_flavor(region):
        """Flavour of the lepton inside the AK8 jet for a boosted SR / flavour-CR region.

        Same-flavour regions carry the leading lepton's flavour in the jet;
        the flavour CRs are named lead-then-jet (``emu``: leading electron,
        muon in the jet).  Returns ``None`` for regions without an LSF jet.
        """
        if "boosted" not in region or "dy_cr" in region:
            return None
        if "emu" in region:
            return "muon"
        if "mue" in region:
            return "electron"
        if "mumu" in region:
            return "muon"
        if "ee" in region:
            return "electron"
        return None

    def _lepton_sfs_for_region(self, region, syst_weights, era, tight_muons, tight_electrons,
                              loose_muons=None, loose_electrons=None, *, is_mc, factors=None):
        """Multiply per-region lepton SFs into MC base syst_weights.

        Data keeps its base weights, without efficiency SFs or SF variations.

        Computes only the SFs relevant for each region:
          - 'ee' regions  → electron RECO + HEEP ID + electron trigger SFs
          - 'mumu' regions → muon RECO + ID + ISO + muon trigger SFs
          - flavor CRs → electron RECO + ID and muon RECO + ID + ISO SFs, plus
            the trigger SF of whichever single-lepton trigger selects the
            region (see ``_region_trigger``)
          - boosted regions additionally get loose muon RECO + ID SFs (no ISO)
            and loose electron RECO SFs
          - boosted SR / flavour-CR regions get the flat LSF SF of the
            lepton-in-jet flavour when the era has one configured

        Per-component up/down variations are added to the returned dict for
        every component whose systematic source is enabled and producible
        (the ``sf`` family for the lepton components, ``lsf`` for the LSF SF).

        *factors*, when given, receives each component's nominal SF under its
        label (MC only).

        Returns a new syst_weights dict, with lepton SFs multiplied in for MC.
        """
        if not is_mc:
            return dict(syst_weights)

        first_key = next(iter(syst_weights))
        n = len(syst_weights[first_key])

        # Determine which lepton SFs to apply based on region name.
        apply_muon = "mumu" in region or "flavor_cr" in region or "mue" in region or "emu" in region
        apply_electron = "ee" in region or "flavor_cr" in region or "mue" in region or "emu" in region
        trigger = self._region_trigger(region)
        apply_muon_trig = apply_muon and trigger == "mu"
        apply_electron_trig = apply_electron and trigger == "e"

        # Collect all SF components as {label: (nom, up, down)}.
        components = {}

        # Tight muon RECO, ID, ISO SFs.
        if apply_muon and tight_muons is not None and era in MUON_JSONS:
            muon_sfs = muon_sf(tight_muons, era)
            for comp, vals in muon_sfs.items():
                components[f"Muon{comp.capitalize()}Sf"] = vals

            if apply_muon_trig:
                components["MuonTrigSf"] = muon_trigger_sf(tight_muons, era)

        # Loose muon RECO + ID SFs (no ISO, no trigger).
        if apply_muon and loose_muons is not None and era in MUON_JSONS:
            loose_sfs = muon_sf_loose(loose_muons, era)
            for comp, vals in loose_sfs.items():
                components[f"LooseMuon{comp.capitalize()}Sf"] = vals

        # Loose electron RECO SF.
        if apply_electron and loose_electrons is not None and era in ELECTRON_JSONS:
            components["LooseElectronRecoSf"] = electron_reco_sf(loose_electrons, era)

        # Electron RECO + HEEP ID SFs, plus trigger SF where the electron
        # triggers select the region.
        if apply_electron and tight_electrons is not None and era in ELECTRON_JSONS:
            components["ElectronRecoSf"] = electron_reco_sf(tight_electrons, era)
            components["ElectronIdSf"] = electron_id_sf(tight_electrons, era)

            if apply_electron_trig:
                components["ElectronTrigSf"] = electron_trigger_sf(tight_electrons, era)

        # LSF scale factor (boosted SR / flavour CR only; era-gated).
        jet_flavor = self._jet_lepton_flavor(region)
        if jet_flavor is not None:
            lsf = lsf_sf(era, jet_flavor, n)
            if lsf is not None:
                components["LsfSf"] = lsf

        # Build combined nominal SF.
        sf_nom = np.ones(n, dtype=np.float64)
        for label, (comp_nom, _comp_up, _comp_down) in components.items():
            sf_nom *= comp_nom
            if factors is not None:
                factors[label] = np.asarray(comp_nom, dtype=np.float64)

        result = {k: v * sf_nom for k, v in syst_weights.items()}

        # Add per-component up/down variations (only for nominal pass), and
        # only for components whose inputs exist for this era and that were
        # not explicitly omitted.  An identity SF from a missing payload must
        # not turn into a "produced" variation.
        allowed = self._producible_names(era, WEIGHT)
        if "Nominal" in syst_weights and len(components) > 0:
            for vary_label, (vary_nom, vary_up, vary_down) in components.items():
                if vary_label not in allowed:
                    continue
                # Ratio to swap nominal → varied for this component.
                safe_nom = np.where(vary_nom > 0, vary_nom, 1.0)
                ratio_up = vary_up / safe_nom
                ratio_down = vary_down / safe_nom
                result[f"{vary_label}Up"] = result["Nominal"] * ratio_up
                result[f"{vary_label}Down"] = result["Nominal"] * ratio_down

        return result

    @staticmethod
    def _require_boosted_inputs(events):
        """Raise unless the branches the boosted selection needs are present."""
        fatjet = getattr(events, "FatJet", None)
        if fatjet is None:
            raise RuntimeError(
                "Boosted regions were requested but the input has no FatJet collection. "
                "Run with --region resolved or use an input that keeps FatJet."
            )
        if "lsf3" not in getattr(fatjet, "fields", []):
            raise RuntimeError(
                "Boosted regions were requested but FatJet has no 'lsf3' field. "
                "Run with --region resolved or use an input that keeps FatJet_lsf3."
            )

    def _producible_sources(self, era, kind):
        """Enabled sources of *kind* with inputs for *era*, minus omissions (cached)."""
        key = (era, kind)
        if key not in self._sources_by_era:
            self._sources_by_era[key] = producible(
                era, self._enabled_systs, omitted=self._omit_systs, kinds=(kind,),
            )
        return self._sources_by_era[key]

    def _producible_names(self, era, kind):
        return {src.name for src in self._producible_sources(era, kind)}

    def _plan_shape_variations(self, events, era):
        """Decide, per enabled shape label, how this chunk covers it.

        Returns ``(passes, nominal_aliases)``:
          - ``passes``: ``(label, source, direction)`` triples whose shifted
            fields exist on the affected collections; each gets its own
            selection + fill pass.
          - ``nominal_aliases``: labels whose affected collections are empty
            in this chunk.  The variation cannot change anything here, so the
            chunk's nominal is filled under the label and counts as evaluated.

        A label whose fields are missing while affected objects *are* present
        means the correction inputs were unavailable or the correction
        failed.  That is an error, never a silent skip.
        """
        passes, nominal_aliases = [], []
        for src in self._producible_sources(era, SHAPE):
            for direction in DIRECTIONS:
                label = src.label(direction)
                required = src.required_fields(direction)
                # Every non-empty affected collection must carry the shifted
                # fields; an empty collection has nothing to shift.
                populated, incomplete = [], []
                for name in src.objects:
                    coll = getattr(events, name, None)
                    if coll is None or int(ak.sum(ak.num(coll))) == 0:
                        continue
                    populated.append(name)
                    missing = [f for f in required if f not in coll.fields]
                    if missing:
                        incomplete.append(f"{name} lacks {missing}")
                if incomplete:
                    raise RuntimeError(
                        f"Systematic '{label}' was requested for era '{era}' but this chunk has "
                        f"affected objects without shifted kinematics ({'; '.join(incomplete)}): "
                        "correction inputs are unavailable or the correction failed. Refusing to "
                        "produce an incomplete variation."
                    )
                if populated:
                    passes.append((label, src, direction))
                else:
                    nominal_aliases.append(label)
        return passes, nominal_aliases

    @staticmethod
    def _apply_shape_variation(events, source, direction):
        """Return events with shifted kinematics for shape *source*.

        Jet variations swap pt and mass; lepton variations swap pt only.
        Uses ``ak.with_field`` so the original events are not mutated.
        """
        pt_field = source.variation_field(direction)
        mass_field = pt_field.replace("pt_", "mass_", 1)
        for coll_name in source.objects:
            coll = getattr(events, coll_name, None)
            if coll is None or pt_field not in coll.fields:
                continue
            coll = ak.with_field(coll, coll[pt_field], "pt")
            if mass_field in coll.fields:
                coll = ak.with_field(coll, coll[mass_field], "mass")
            if isinstance(events, ak.Array):
                events = ak.with_field(events, coll, coll_name)
            else:
                # Plain container of collections (test doubles): shallow copy.
                events = copy.copy(events)
                setattr(events, coll_name, coll)
        return events

    def _run_shape_systematic(self, events, output, metadata, is_mc, syst_label, triggers):
        """Re-run selection and histogram filling for a single shape variation.

        ``events`` should already have modified kinematics via
        ``_apply_shape_variation``.  Only the ``syst_label`` entry is produced
        (no SF up/down crossing).  Every region is filled: a lepton shift can
        change the extra-lepton veto of the other flavor's regions, so no
        region is assumed unaffected here.

        Returns ``{region: {syst_label}}`` for the regions that were filled.
        """
        mc_campaign = metadata.get("era")
        process_name = metadata.get("physics_group")

        tight_leptons, loose_leptons, lepton_masks = self.select_leptons(events)
        ak4_jets, _ak8_jets, jet_masks = self.select_jets(events, mc_campaign)

        resolved_selections = None
        if self._region in ("resolved", "both"):
            resolved_selections = self.resolved_selections(
                tight_leptons, ak4_jets,
                lepton_masks=lepton_masks, jet_masks=jet_masks, triggers=triggers,
                era=mc_campaign,
            )

        boosted_payload = None
        if self._region in ("boosted", "both"):
            # A failure here would silently drop the boosted regions from
            # this variation only; let it propagate.
            self._require_boosted_inputs(events)
            boosted_payload = self.boosted_selections(events, mc_campaign, triggers=triggers)

        tight_electrons = events.Electron[lepton_masks["ele_pteta"] & lepton_masks["ele_id"]]
        tight_muons = events.Muon[lepton_masks["mu_pteta"] & lepton_masks["mu_id"]]
        weights, _ = self.build_event_weights(events, metadata, is_mc)
        era = mc_campaign

        # Single-entry syst_weights — lepton SFs are multiplied in by
        # _lepton_sfs_for_region but SF up/down variants are skipped
        # because "Nominal" is absent from the dict.
        syst_weights = {syst_label: weights.weight()}

        coverage = {}
        if resolved_selections is not None:
            coverage.update(self._fill_resolved(
                output, resolved_selections, process_name, ak4_jets, tight_leptons,
                weights, syst_weights, era, tight_muons, tight_electrons,
                is_mc=is_mc, skip_cutflows=True, dy_target=self._dy_reweight_target(metadata),
            ))

        if boosted_payload is not None:
            coverage.update(self._fill_boosted(
                output, boosted_payload, process_name, weights, syst_weights,
                era, is_mc=is_mc, skip_cutflows=True, dy_target=self._dy_reweight_target(metadata),
            ))
        return coverage

    def _fill_resolved(self, output, resolved_selections, process_name, ak4_jets, tight_leptons,
                       weights, syst_weights, era, tight_muons, tight_electrons,
                       *, is_mc, skip_cutflows=False, event_lists=None, event_weights=None,
                       weight_factors=None, dy_target=False, pdf=None):
        """Build resolved region masks and fill histograms + cutflows.

        Regions (and any enabled variants) come from ``wrcoffea.regions``;
        blinded data skips the signal regions and truncates its cutflows.
        *event_weights*, when given, receives each region's nominal weight
        distribution and, with *weight_factors* (the base-weight factors from
        ``_event_weight_factors``), its per-factor distributions.
        Returns ``{region: set(labels filled)}``.
        """
        blind = self._blind and not is_mc
        resolved_regions = build_regions(RESOLVED_REGIONS, resolved_selections, self._region_variants,
                                         blind=blind)
        # Leading AK4 jet pT for the DY reshape (0 where the event has no jet;
        # such events never enter a resolved region).
        lead_ak4_pt = np.asarray(ak.fill_none(ak.pad_none(ak4_jets, 1)[:, 0].pt, 0.0), dtype=np.float64)

        coverage = {}
        for region, cuts in resolved_regions.items():
            factors = dict(weight_factors) if weight_factors is not None else None
            region_syst = self._lepton_sfs_for_region(
                region, syst_weights, era, tight_muons, tight_electrons, is_mc=is_mc,
                factors=factors,
            )
            region_syst = self._dy_reshape_for_region(
                region_syst, "resolved", lead_ak4_pt, era, dy_target, is_mc, factors=factors,
            )
            coverage[region] = fill_resolved_histograms(
                output, region, cuts, process_name, ak4_jets, tight_leptons, weights, region_syst,
                syst_observables=self._syst_observables, event_lists=event_lists, pdf=pdf,
            )
            if event_weights is not None:
                fill_event_weights(event_weights, region, cuts, region_syst[NOMINAL], factors)

        if not skip_cutflows:
            fill_cutflows(output, resolved_selections, weights, blind=blind)
        return coverage

    @staticmethod
    def _candidates_by_flavor(candidate):
        """Split a per-event optional lepton candidate into (muons, electrons).

        Each output is a jagged collection with 0 or 1 entries per event, so
        it can be fed to the per-object SF functions exactly like a filtered
        NanoAOD collection.  Events where the candidate is None contribute an
        empty list to both outputs.
        """
        col = ak.singletons(candidate)
        pdg = abs(col.pdgId)
        return col[pdg == 13], col[pdg == 11]

    def _fill_boosted(self, output, boosted_payload, process_name, weights, syst_weights,
                      era, *, is_mc, skip_cutflows=False, event_lists=None, event_weights=None,
                      weight_factors=None, dy_target=False, pdf=None):
        """Unpack boosted payload, build region masks, and fill histograms.

        Returns ``{region: set(labels filled)}``.  As in ``_fill_resolved``,
        blinded data skips the signal regions and truncates its cutflows, and
        *event_weights*, when given, receives the nominal weight distributions
        (per factor too, with *weight_factors*).

        Lepton SFs are evaluated on the two objects each region is built from:
        the leading tight lepton gets the tight SFs (and the trigger SF), and
        the region's sub-leading candidate (DY partner, same-flavor lepton in
        the AK8 jet, or opposite-flavor lepton in the AK8 jet) gets the loose
        SFs.  This matches the objects used to fill the histograms.
        """
        boosted_sel, tight_lep, AK8_cand_dy, DY_loose_lep, AK8_cand, of_candidate, sf_candidate = boosted_payload
        blind = self._blind and not is_mc

        # Fill boosted cutflows
        if not skip_cutflows:
            fill_boosted_cutflows(output, boosted_sel, weights, blind=blind)

        lead_mu, lead_e = self._candidates_by_flavor(tight_lep)
        sublead_by_kind = {
            "dy_cr": self._candidates_by_flavor(DY_loose_lep),
            "flavor_cr": self._candidates_by_flavor(of_candidate),
            "sr": self._candidates_by_flavor(sf_candidate),
        }

        def _region_sfs(region, factors):
            kind = "dy_cr" if "dy_cr" in region else ("flavor_cr" if "flavor_cr" in region else "sr")
            sub_mu, sub_e = sublead_by_kind[kind]
            return self._lepton_sfs_for_region(
                region, syst_weights, era, lead_mu, lead_e,
                loose_muons=sub_mu, loose_electrons=sub_e, is_mc=is_mc, factors=factors,
            )

        boosted_regions = build_regions(BOOSTED_REGIONS, boosted_sel, self._region_variants, blind=blind)
        coverage = {}
        for region, cuts in boosted_regions.items():
            factors = dict(weight_factors) if weight_factors is not None else None
            region_syst = _region_sfs(region, factors)
            if "dy_cr" in region:
                sublead, ak8 = DY_loose_lep, AK8_cand_dy
            elif "flavor_cr" in region:
                sublead, ak8 = of_candidate, AK8_cand
            else:
                sublead, ak8 = sf_candidate, AK8_cand
            ak8_pt = np.asarray(ak.fill_none(ak8.pt, 0.0), dtype=np.float64)
            region_syst = self._dy_reshape_for_region(
                region_syst, "boosted", ak8_pt, era, dy_target, is_mc, factors=factors,
            )
            coverage[region] = fill_boosted_histograms(
                output, region, cuts, process_name, tight_lep, ak8, sublead, weights, region_syst,
                syst_observables=self._syst_observables, event_lists=event_lists, pdf=pdf,
            )
            if event_weights is not None:
                fill_event_weights(event_weights, region, cuts, region_syst[NOMINAL], factors)
        return coverage

    def process(self, events):
        """Run analysis for one NanoEvents chunk and return a dataset-nested output dict."""
        output = self.make_output()
        metadata = events.metadata
        mc_campaign = metadata.get("era")
        process_name = metadata.get("physics_group")
        dataset = metadata.get("sample")

        datatype = (metadata.get("datatype") or "").strip().lower()
        is_mc = datatype == "mc"
        is_data = not is_mc
        if is_data and not self._blind:
            # The CLI refuses --unblind outside Run 2; direct use must not bypass that.
            check_unblind_allowed(mc_campaign)

        # Normalization includes all input MC events, including those rejected
        # by the filters below. Preserve signed generator weights per chunk.
        chunk_sumw = float(ak.sum(events.genWeight)) if self._compute_sumw and is_mc else None

        # Apply lumi mask (data only).
        events = self.apply_lumi_mask(events, mc_campaign, is_data)
        # Apply MET noise filters (data, MC, and signal).
        events = self.apply_noise_filter(events, mc_campaign)
        raw_pt = events.Jet.pt * (1 - events.Jet.rawFactor)
        
        # JEC/JER always; JER and JES up/down kinematics only when requested.
        events = apply_jet_corrections(
            events, mc_campaign, is_mc, variations=self._enabled_systs & {"jer", "jes"},
        )

        # print("--- JEC VERIFICATION (First 10 events) ---")
        #corrected_pt = events.Jet.pt
        # jec_factor = corrected_pt / raw_pt[events.Jet.rawFactor<=0.9]

        # print(f"Raw pT:       {raw_pt[0:10].to_list()}")
        # print(f"Corrected pT: {corrected_pt[0:10].to_list()}")
        # print(f"JEC Factor:   {jec_factor[0:10].to_list()}")
        # print("-----------------------------------------")
        # flat_factors = ak.flatten(jec_factor)

        # print(f"Mean JEC factor: {np.mean(flat_factors):.3f}")
        # print(f"Max JEC factor:  {np.max(flat_factors):.3f}")
        # print(f"Min JEC factor:  {np.min(flat_factors):.3f}")


        # apply jet veto
        if mc_campaign in JME_JSONS:
            try:
                ak4_id_mask = self.jetid_mask_ak4puppi(events.Jet, mc_campaign, jetid_name="AK4PUPPI_TightLeptonVeto")
            except AttributeError as e:
                key = f"jetid_fallback::{mc_campaign}"
                if key not in _WARN_ONCE:
                    _WARN_ONCE.add(key)
                    logger.warning(
                        "correctionlib puppi JetID requested for era '%s' but required jet fields "
                        "are missing; falling back to Jet.jetId>=6 for this worker process. "
                        "(example error: %s)", mc_campaign, e
                    )

                ak4_id_mask = events.Jet.jetId >= 6
        else:
            if hasattr(events.Jet, 'isTightLeptonVeto'):
                ak4_id_mask = events.Jet.isTightLeptonVeto
            else:
                ak4_id_mask = events.Jet.jetId >= 6
        # Jet veto maps (JME recommendation): Run 3 rejects the whole event, Run 2
        # drops only the vetoed jets and keeps the event.  The two runs differ on
        # purpose; `jetveto_mode` in config.yaml picks the rule per era.
        if jet_veto_mode(mc_campaign) == "jet":
            keep = jet_veto_jet_mask(events, ak4_id_mask, mc_campaign)
            jets = ak.with_field(events.Jet, keep, "passJetVeto")
            if isinstance(events, ak.Array):
                events = ak.with_field(events, jets, "Jet")
            else:
                # Plain container of collections (test doubles): shallow copy.
                events = copy.copy(events)
                events.Jet = jets
        else:
            veto_mask = jet_veto_event_mask(events, ak4_id_mask, mc_campaign)
            events = events[veto_mask]

        corrected_electrons = apply_electron_scale_smearing(events, mc_campaign, is_mc)
        #verify_scale_smearing(events.Electron, corrected_electrons, is_mc)
        if corrected_electrons:
            # Overwrite nominal kinematics
            events["Electron", "pt"] = corrected_electrons["pt"]
            events["Electron", "energyErr"] = corrected_electrons["energyErr"]

            # Attach systematic variations if they exist (MC only)
            for syst in ["pt_smear_up", "pt_smear_down", "pt_scale_up", "pt_scale_down"]:
                if syst in corrected_electrons:
                    events["Electron", syst] = corrected_electrons[syst]

        corrected_muons = apply_muon_scale_smearing(events, mc_campaign, is_mc)
        if corrected_muons:
            events["Muon", "pt"] = corrected_muons["pt"]
            # Attach systematic variations if they exist (MC only)                  
            for syst in ["pt_smear_up", "pt_smear_down", "pt_scale_up", "pt_scale_down"]:
                if syst in corrected_muons:
                    events["Muon", syst] = corrected_muons[syst]
        
        tight_leptons, loose_leptons, lepton_masks = self.select_leptons(events)
        ak4_jets, ak8_jets, jet_masks = self.select_jets(events, mc_campaign)
        triggers = self.build_trigger_masks(events, mc_campaign)
        
        # Resolved selections (only if requested).
        resolved_selections = None
        if self._region in ("resolved", "both"):
            resolved_selections = self.resolved_selections(
                tight_leptons,
                ak4_jets,
                lepton_masks=lepton_masks,
                jet_masks=jet_masks,
                triggers=triggers,
                era=mc_campaign,
            )

        # Boosted selections (skip gracefully if FatJet/lsf3 is absent).
        boosted_payload = None
        if self._region in ("boosted", "both"):
            # The boosted regions are part of the requested output; a chunk
            # that cannot build them must fail rather than silently drop them.
            self._require_boosted_inputs(events)
            boosted_payload = self.boosted_selections(events, mc_campaign, triggers=triggers)

        # Reconstruct tight lepton collections from events + masks for SF
        # evaluation.  Using short-lived NanoAOD slices (not carried from
        # select_leptons) so the full column cache is freed promptly.
        tight_electrons = events.Electron[lepton_masks["ele_pteta"] & lepton_masks["ele_id"]]
        tight_muons = events.Muon[lepton_masks["mu_pteta"] & lepton_masks["mu_id"]]
        weights, syst_weights = self.build_event_weights(events, metadata, is_mc)
        era = metadata.get("era")

        # Shape systematics whose affected collections are empty in this
        # chunk cannot differ from nominal: fill nominal under their label so
        # the chunk still counts as covered.  Missing fields with objects
        # present is an error (raised inside _plan_shape_variations).
        shape_passes, nominal_aliases = ([], [])
        if is_mc:
            shape_passes, nominal_aliases = self._plan_shape_variations(events, era)
            for label in nominal_aliases:
                syst_weights[label] = weights.weight()

        # Fill histograms (and, with --save-events, the per-event SR lists).
        event_lists = output.setdefault("event_lists", {}) if self._save_events else None
        # Per-region distribution of the nominal MC event weight, filled in
        # this nominal pass only (shape passes never get it), never for data.
        # Not in compute_sumw mode: there the weights lack the 1/sumw factor
        # that normalize_by_sumw divides out of the histogram contents later,
        # which cannot repair a log10|w| axis.
        event_weights = (output.setdefault("event_weights", {})
                         if is_mc and not self._compute_sumw else None)
        weight_factors = (self._event_weight_factors(events, metadata, weights)
                          if event_weights is not None else None)
        dy_target = self._dy_reweight_target(metadata)
        # PDF members (nominal pass only): per-dataset member histograms of the
        # fit masses, combined into PDF/AlphaS at save time.
        pdf_lhaid, pdf_members = self._pdf_members(events, metadata, era, is_mc)
        pdf = None
        if pdf_members is not None:
            pdf = (output.setdefault(MEMBERS_KEY, {}), pdf_members)
            output[CHUNKS_KEY] = 1          # accumulates; save_hists requires == n_chunks
        coverage = {}
        if resolved_selections is not None:
            coverage.update(self._fill_resolved(
                output, resolved_selections, process_name, ak4_jets, tight_leptons,
                weights, syst_weights, era, tight_muons, tight_electrons, is_mc=is_mc,
                event_lists=event_lists, event_weights=event_weights, weight_factors=weight_factors,
                dy_target=dy_target, pdf=pdf,
            ))

        if boosted_payload is not None:
            coverage.update(self._fill_boosted(
                output, boosted_payload, process_name, weights, syst_weights, era, is_mc=is_mc,
                event_lists=event_lists, event_weights=event_weights, weight_factors=weight_factors,
                dy_target=dy_target, pdf=pdf,
            ))

        # Shape systematics: re-run selections with shifted kinematics.
        for syst_label, source, direction in shape_passes:
            varied_events = self._apply_shape_variation(events, source, direction)
            for region, labels in self._run_shape_systematic(
                varied_events, output, metadata, is_mc, syst_label, triggers,
            ).items():
                coverage.setdefault(region, set()).update(labels)

        # Coverage bookkeeping: counts accumulate across chunks by addition,
        # so save_hists can require count == n_chunks per (region, label).
        output["syst_coverage"] = {
            region: {label: 1 for label in sorted(labels)}
            for region, labels in coverage.items()
        }
        output["n_chunks"] = 1
        # A DY dataset other than the one the DY payloads were derived for is "not_for_sample".
        dy_identity = "not_for_sample" if process_name == DY_PROCESS_GROUP else "identity"
        if not is_mc or era not in DY_ZPT_REWEIGHT_JSONS:
            dy_zpt_status = "unconfigured"
        elif not self._dy_zpt_reweight:
            dy_zpt_status = "off"
        else:
            dy_zpt_status = "applied" if dy_target else dy_identity
        if not is_mc or era not in DY_RESHAPE_JSONS:
            dy_reshape_status = "unconfigured"
        elif not self._dy_reshape:
            dy_reshape_status = "off"
        else:
            dy_reshape_status = "applied" if dy_target else dy_identity
        if not is_mc or era not in SIGNAL_KFACTOR_JSONS or process_name != SIGNAL_PROCESS_GROUP:
            kfactor_status = "unconfigured"
        else:
            kfactor_status = "applied" if self._signal_kfactor else "off"
        output["dataset_meta"] = {
            "era": {str(era)},
            "datatype": {"mc" if is_mc else "data"},
            "physics_group": {str(process_name)},
            "dy_zpt_reweight": {dy_zpt_status},
            "signal_kfactor": {kfactor_status},
            "dy_reshape": {dy_reshape_status},
            # What this run was asked to produce; chunks and datasets with
            # different settings must never be merged.
            "syst_config": {self._syst_config},
        }
        if pdf_lhaid is not None:
            # The PDF set whose members were filled (or "none"); save_hists
            # sums the members per set.
            output["dataset_meta"]["pdf_lhaid"] = {pdf_lhaid}

        nested_output = {dataset: {**output}}

        # Accumulate the unfiltered sum even when every event was rejected,
        # so the caller can normalize after all chunks are processed.
        if chunk_sumw is not None:
            nested_output[dataset]["_sumw"] = chunk_sumw

        return nested_output

    def postprocess(self, accumulator):
        return accumulator
