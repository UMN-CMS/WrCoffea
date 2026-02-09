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
    - `fill_cutflows()` writes a nested `output["cutflow"]` structure with both
        weighted and unweighted one-bin hists plus PackedSelection cutflow hists.

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
import os
import logging
from coffea.nanoevents.methods import vector

from wrcoffea.analysis_config import (
    CUTS, ELECTRON_JSONS, JME_JSONS, LUMI_JSONS, LUMI_UNC, LUMIS, MUON_JSONS, PILEUP_JSONS,
    SEL_MIN_TWO_AK4_JETS_PTETA, SEL_MIN_TWO_AK4_JETS_ID,
    SEL_TWO_PTETA_ELECTRONS, SEL_TWO_PTETA_MUONS, SEL_TWO_PTETA_EM,
    SEL_TWO_ID_ELECTRONS, SEL_TWO_ID_MUONS, SEL_TWO_ID_EM,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_MLL_GT200, SEL_MLLJJ_GT800, SEL_MLL_GT400,
    SEL_TWO_TIGHT_ELECTRONS, SEL_TWO_TIGHT_MUONS, SEL_TWO_TIGHT_EM,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53, SEL_MIN_TWO_AK4_JETS,
    SEL_60_MLL_150,
    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
    SEL_ATLEAST1AK8_DPHI_GT2, SEL_AK8JETS_WITH_LSF,
    SEL_MUMU_DYCR, SEL_EE_DYCR, SEL_MUMU_SR, SEL_EE_SR,
    SEL_EMU_CR, SEL_MUE_CR, SEL_JET_VETO_MAP,
)
from wrcoffea.era_utils import ERA_MAPPING
from wrcoffea.scale_factors import muon_sf, muon_trigger_sf, electron_trigger_sf, electron_reco_sf, electron_id_sf, pileup_weight, jet_veto_event_mask
from wrcoffea.histograms import (
    RESOLVED_HIST_SPECS, BOOSTED_HIST_SPECS,
    _booking_specs, create_hist,
    fill_resolved_histograms, fill_boosted_histograms, fill_cutflows,
)

ak.behavior.update(vector.behavior)
logger = logging.getLogger(__name__)

# Cache correctionlib payloads per worker process (avoid re-reading JSON every chunk).
_CORRECTIONSET_CACHE = {}

# Warn-once cache (per worker process) to avoid log spam.
_WARN_ONCE: set[str] = set()


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
    - `enabled_systs`: list of enabled systematic families. Currently supported:
      - `lumi`: add `LumiUp`/`LumiDown` variations as histogram `syst` axis values.
    - `region`: which analysis region to run ("resolved", "boosted", or "both").
    - `compute_sumw`: if True, compute sum-of-genWeights on-the-fly instead of
      using the pre-computed ``genEventSumw`` from metadata. Histograms are filled
      without the ``/sumw`` normalization; the caller must divide by the accumulated
      ``_sumw`` per dataset after processing.
    """
    def __init__(self, mass_point, enabled_systs=None, region="both", compute_sumw=False):
        self._signal_sample = mass_point
        self._compute_sumw = compute_sumw
        enabled = enabled_systs or []
        self._enabled_systs = {str(s).strip().lower() for s in enabled if str(s).strip()}
        self._region = region.lower() if region else "both"
        if self._region not in ("resolved", "boosted", "both"):
            raise ValueError(f"Invalid region '{region}'. Must be 'resolved', 'boosted', or 'both'.")
        booking = _booking_specs()
        self.make_output = lambda: {
            name: create_hist(name, bins, label)
            for name, (bins, label) in booking.items()
        }

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
        # Split pT/eta (kinematics) and ID components.
        ele_pteta_mask = (events.Electron.pt > CUTS["lepton_pt_min"]) & (np.abs(events.Electron.eta) < CUTS["lepton_eta_max"])
        mu_pteta_mask  = (events.Muon.pt > CUTS["lepton_pt_min"])     & (np.abs(events.Muon.eta) < CUTS["lepton_eta_max"])

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

        # --- Add flavor tags ---
        tight_electrons = ak.with_field(tight_electrons, "electron", "flavor")
        tight_muons     = ak.with_field(tight_muons,     "muon",     "flavor")
        loose_electrons = ak.with_field(loose_electrons, "electron", "flavor")
        loose_muons     = ak.with_field(loose_muons,     "muon",     "flavor")

        # --- Combine and sort by pT ---
        tight_leptons = ak.with_name(
            ak.concatenate([tight_electrons, tight_muons], axis=1),
            "PtEtaPhiMCandidate",
        )
        loose_leptons = ak.with_name(
            ak.concatenate([loose_electrons, loose_muons], axis=1),
            "PtEtaPhiMCandidate",
        )

        tight_leps = tight_leptons[ak.argsort(tight_leptons.pt, axis=1, ascending=False)]
        loose_leps = loose_leptons[ak.argsort(loose_leptons.pt, axis=1, ascending=False)]

        masks = {
            "ele_pteta": ele_pteta_mask,
            "mu_pteta": mu_pteta_mask,
            "ele_id": ele_id_mask,
            "mu_id": mu_id_mask,
        }

        return tight_leps, loose_leps, masks

    def select_jets(self, events, era, *, is_signal: bool = False):
        """Select AK4 and AK8 jets and return (collections, per-object masks).

        For eras with a JME JSON configured (backgrounds only), AK4 PUPPI
        TightLeptonVeto is evaluated via correctionlib.  If the required
        NanoAOD fields are missing, fall back to ``Jet.isTightLeptonVeto``
        and warn once per worker.
        """
        # ---------- AK4 ----------
        ak4_pteta_mask = (events.Jet.pt > CUTS["ak4_pt_min"]) & (np.abs(events.Jet.eta) < CUTS["ak4_eta_max"])
        if era in JME_JSONS and (not is_signal):
            try:
                ak4_id_mask = self.jetid_mask_ak4puppi_tlv(events.Jet, era)
            except AttributeError as e:
                key = f"jetid_fallback::{era}"
                if key not in _WARN_ONCE:
                    _WARN_ONCE.add(key)
                    logger.warning(
                        "correctionlib puppi JetID requested for era '%s' but required jet fields "
                        "are missing; falling back to isTightLeptonVeto for this worker process. "
                        "(example error: %s)", era, e
                    )
                ak4_id_mask = events.Jet.isTightLeptonVeto
        else:
            ak4_id_mask = events.Jet.isTightLeptonVeto

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

    def jetid_mask_ak4puppi_tlv(self, jets, era):
        """
        Returns a jagged boolean mask aligned with `jets` for AK4 PUPPI TightLeptonVeto.
        """
        import correctionlib

        json_path = JME_JSONS[era]
        ceval = _CORRECTIONSET_CACHE.get(json_path)
        if ceval is None:
            ceval = correctionlib.CorrectionSet.from_file(json_path)
            _CORRECTIONSET_CACHE[json_path] = ceval

        out = ceval["AK4PUPPI_TightLeptonVeto"].evaluate(
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

        # Electrons by run period
        if run == "RunII":
            e_trig = hlt("Ele32_WPTight_Gsf") | hlt("Photon200") | hlt("Ele115_CaloIdVT_GsfTrkIdT")
        else:  # Run3 or unknown
            e_trig = (hlt("Ele30_WPTight_Gsf") | hlt("Photon200")
                      | hlt("Ele115_CaloIdVT_GsfTrkIdT") | hlt("Ele50_CaloIdVT_GsfTrkIdT_PFJet165"))

        # Muons by run period
        if run == "RunII":
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100")
        elif run == "Run3":
            mu_trig = hlt("Mu50") | hlt("HighPtTkMu100") | hlt("CascadeMu100")
        else:  # unknown era — OR all known muon triggers as safe fallback
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100") | hlt("HighPtTkMu100") | hlt("CascadeMu100")

        emu_trig = mu_trig

        return e_trig, mu_trig, emu_trig

    def resolved_selections(self, tight_leptons, ak4_jets, *, lepton_masks=None, jet_masks=None, triggers=None):
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
        selections.add("two_tight_leptons",      n_tight == 2)
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

        # ΔR > 0.4 requirements among {l1, l2, j1, j2}.
        dr_ll   = ak.fill_none(l1.deltaR(l2) > CUTS["dr_min"], False)
        dr_l1j1 = ak.fill_none(l1.deltaR(j1) > CUTS["dr_min"], False)
        dr_l1j2 = ak.fill_none(l1.deltaR(j2) > CUTS["dr_min"], False)
        dr_l2j1 = ak.fill_none(l2.deltaR(j1) > CUTS["dr_min"], False)
        dr_l2j2 = ak.fill_none(l2.deltaR(j2) > CUTS["dr_min"], False)
        dr_jj   = ak.fill_none(j1.deltaR(j2) > CUTS["dr_min"], False)

        selections.add("dr_all_pairs_gt0p4", dr_ll & dr_l1j1 & dr_l1j2 & dr_l2j1 & dr_l2j2 & dr_jj)

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

        # pT/eta-only
        selections.add("two_pteta_electrons", n_ele_pteta == 2)
        selections.add("two_pteta_muons",     n_mu_pteta  == 2)
        selections.add("two_pteta_em",        (n_ele_pteta == 1) & (n_mu_pteta == 1))

        # ID-only
        selections.add("two_id_electrons", n_ele_id == 2)
        selections.add("two_id_muons",     n_mu_id  == 2)
        selections.add("two_id_em",        (n_ele_id == 1) & (n_mu_id == 1))

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

        loose_electrons = (
            (events.Electron.pt > CUTS["lepton_pt_min"])
            & (np.abs(events.Electron.eta) < CUTS["lepton_eta_max"])
            & (heep_flag | loose_noIso_mask)
        )
        return events.Electron[loose_electrons]

    def selectLooseMuons(self, events):
        loose_muons = (events.Muon.pt > CUTS["lepton_pt_min"]) & (np.abs(events.Muon.eta) < CUTS["lepton_eta_max"]) & (events.Muon.highPtId == CUTS["muon_highPtId"])
        return events.Muon[loose_muons]
    
    def selectAK8Jets(self,events):
        """Baseline AK8 selection used in boosted categories."""
        ak8_jets = (events.FatJet.pt > CUTS["ak8_pt_min"]) & (np.abs(events.FatJet.eta) < CUTS["ak8_eta_max"])  & (events.FatJet.msoftdrop > CUTS["ak8_msoftdrop_min"])
        return events.FatJet[ak8_jets]

    def selectAK8Jets_withLSF(self,events):
        """AK8 selection with LSF requirement used for boosted SR/CR definitions."""
        ak8_jets = (events.FatJet.pt > CUTS["ak8_pt_min"]) & (np.abs(events.FatJet.eta) < CUTS["ak8_eta_max"])  & (events.FatJet.msoftdrop > CUTS["ak8_msoftdrop_min"]) & (events.FatJet.lsf3 > CUTS["ak8_lsf3_min"])
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
        AK8Jets = self.selectAK8Jets(events)
        AK8Jets_withLSF = self.selectAK8Jets_withLSF(events)
        is_signal = (getattr(events, "metadata", {}) or {}).get("physics_group") == "Signal"
        AK4Jets_inc, _, _ = self.select_jets(events, era, is_signal=is_signal)
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

        looseLeptons = ak.with_name(ak.concatenate((looseElectrons, looseMuons), axis=1), 'PtEtaPhiMCandidate')
        looseLeptons = looseLeptons[ak.argsort(looseLeptons.pt, axis=1, ascending=False)]
        tightLeptons_inc = ak.with_name(ak.concatenate((tightElectrons_inc, tightMuons_inc), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons_inc = tightLeptons_inc[ak.argsort(tightLeptons_inc.pt, axis=1, ascending=False)]

        # Define a resolved-like tag and take boosted as the complement.
        has_two_leptons = ak.num(tightLeptons_inc) >= 2
        muons_padded = ak.pad_none(tightLeptons_inc, 2, axis=1)
        # Compute dr only for events with >=2 leptons.
        dr_l1l2 = ak.where(
            ak.num(tightLeptons_inc) >= 2,
            muons_padded[:,0].deltaR(muons_padded[:,1]),
            ak.full_like(ak.num(tightLeptons_inc), np.nan)
        )
        # For jets
        jpad = ak.pad_none(AK4Jets_inc, 2)
        j1, j2 = jpad[:, 0], jpad[:, 1] 
        dr_j1j2   = ak.fill_none(j1.deltaR(j2) > CUTS["dr_min"], False)

        has_two_jets = ak.num(AK4Jets_inc) >= 2

        # dr_jl_min: compute only if both jets and leptons exist.
        has_j_and_l = (ak.num(AK4Jets_inc) >= 1) & (ak.num(tightLeptons_inc) >= 1)
        dr_jl_min = ak.where( has_j_and_l, ak.min(AK4Jets_inc[:, :2].nearest(tightLeptons_inc).deltaR(AK4Jets_inc[:, :2]), axis=1), ak.full_like(has_j_and_l, np.nan))
        # Build all 2 leptons × 2 jets pairs.
        dr_lj = ak.cartesian({"lep": tightLeptons_inc[:,:2], "jet": AK4Jets_inc[:,:2]}, axis=1)
        dr_lj_vals = dr_lj["lep"].deltaR(dr_lj["jet"])
        # Condition: all l-j separations > 0.4.
        dr_lj_ok = ak.all(dr_lj_vals > CUTS["dr_min"], axis=1)
        resolved = (((ak.num(tightElectrons_inc)  + (ak.num(tightMuons_inc))) == 2) & (ak.num(AK4Jets_inc) >= 2) & (dr_l1l2 > CUTS["dr_min"]) & (dr_j1j2 > CUTS["dr_min"]) & (dr_jl_min > CUTS["dr_min"])) #(dr_lj_ok))                                                                                                                                                                   
        boosted  = ~resolved
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

        SR_mask = is_sr & (mlj_sr > CUTS["mlljj_min"]) & (mll_sr > CUTS["mll_sr_min"])
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

        CR_mask = is_cr & (mlj_cr > CUTS["mlljj_min"]) & (mll_cr > CUTS["mll_sr_min"])
        # Veto extra tight leptons for flavor CR.
        no_extra_tight_flav_cr = self._no_extra_tight_leptons(
            looseMuons, looseElectrons, tight_lep, of_candidate)

        # Triggers.
        if triggers is not None:
            eTrig, muTrig, emu_trig = triggers
        
        # Region assignments.
        mumu_dy_cr = muTrig & DYCR_mask & is_lead_mu & no_extra_tight_dyCR
        ee_dy_cr   = eTrig & DYCR_mask & is_lead_e & no_extra_tight_dyCR
        emu_cr = eTrig  & CR_mask & is_lead_e & no_extra_tight_flav_cr # lead e, loose mu                                                                        
        mue_cr = muTrig & CR_mask & is_lead_mu & no_extra_tight_flav_cr # lead mu, loose e                                                                        
        ee_sr = eTrig & SR_mask & is_lead_e & no_extra_tight_sr
        mumu_sr = muTrig & SR_mask &  is_lead_mu & no_extra_tight_sr
        selections.add(SEL_MUMU_DYCR, mumu_dy_cr & is_sublead_mu)
        selections.add(SEL_EE_DYCR,  ee_dy_cr & is_sublead_e)
        selections.add(SEL_MUMU_SR,  mumu_sr & is_sublead_mu_sr)
        selections.add(SEL_EE_SR,    ee_sr & is_sublead_e_sr)
        selections.add(SEL_EMU_CR,   emu_cr & is_sublead_mu_cr)
        selections.add(SEL_MUE_CR,   mue_cr & is_sublead_e_cr)
        return selections, tight_lep, AK8_cand_dy,DY_loose_lep, AK8_cand,of_candidate, sf_candidate 

    def build_event_weights(self, events, metadata, is_mc, tight_muons=None, tight_electrons=None):
        """
        Minimal weights:
          - MC: xsec/nevts normalization (+ optional DY UL18 scale) + lumi Up/Down
                + muon RECO×ID×ISO SF + muon trigger SF
                + electron Reco SF
          - Data: unit weights
        NO genWeight, NO L1 prefire, NO pileup.
        """
        n = len(events)
        weights = Weights(n)

        if is_mc:
            # Cross-section normalization
            lumi = float(LUMIS[metadata.get("era")])
            xsec = float(metadata.get("xsec"))

            if self._compute_sumw:
                # Defer /sumw normalization — caller will divide by accumulated _sumw.
                event_weight = events.genWeight * xsec * lumi * 1000.0
            else:
                # IMPORTANT: use signed genEventSumw (do NOT abs) for NLO samples.
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

            # Muon scale factors (RECO, ID, ISO as independent weights + trigger).
            if tight_muons is not None and era in MUON_JSONS:
                muon_sfs = muon_sf(tight_muons, era)
                for component, (sf_nom, sf_up, sf_down) in muon_sfs.items():
                    weights.add(f"muon_{component}_sf", sf_nom, weightUp=sf_up, weightDown=sf_down)

                trig_nom, trig_up, trig_down = muon_trigger_sf(tight_muons, era)
                weights.add("muon_trig_sf", trig_nom, weightUp=trig_up, weightDown=trig_down)

            # Electron scale factors (Reco + ID + trigger).
            if tight_electrons is not None and era in ELECTRON_JSONS:
                ele_nom, ele_up, ele_down = electron_reco_sf(tight_electrons, era)
                weights.add("electron_reco_sf", ele_nom, weightUp=ele_up, weightDown=ele_down)

                ele_id_nom, ele_id_up, ele_id_down = electron_id_sf(tight_electrons, era)
                weights.add("electron_id_sf", ele_id_nom, weightUp=ele_id_up, weightDown=ele_id_down)

                ele_trig_nom, ele_trig_up, ele_trig_down = electron_trigger_sf(tight_electrons, era)
                weights.add("electron_trig_sf", ele_trig_nom, weightUp=ele_trig_up, weightDown=ele_trig_down)

            syst_weights = {"Nominal": weights.weight()}

            # Optional lumi uncertainty (produces syst histograms only if enabled).
            if "lumi" in self._enabled_systs:
                era_key = metadata.get("era")
                delta = LUMI_UNC.get(era_key)
                if delta is None:
                    logger.warning(
                        f"No luminosity uncertainty defined for era '{era_key}'. "
                        "Skipping lumiUp/lumiDown systematics."
                    )
                else:
                    ones = np.ones(n, dtype=np.float32)
                    weights.add(
                        "lumi",
                        ones,
                        weightUp=ones * (1.0 + float(delta)),
                        weightDown=ones * (1.0 - float(delta)),
                    )
                    syst_weights["Nominal"] = weights.weight()
                    syst_weights["LumiUp"] = weights.weight(modifier="lumiUp")
                    syst_weights["LumiDown"] = weights.weight(modifier="lumiDown")

            # Optional pileup uncertainty.
            if "pileup" in self._enabled_systs:
                era_key = metadata.get("era")
                if era_key in PILEUP_JSONS:
                    syst_weights["PileupUp"] = weights.weight(modifier="pileupUp")
                    syst_weights["PileupDown"] = weights.weight(modifier="pileupDown")

            # Optional scale-factor uncertainties (muon + electron SFs).
            if "sf" in self._enabled_systs:
                era_key = metadata.get("era")
                if tight_muons is not None and era_key in MUON_JSONS:
                    for comp in ["reco", "id", "iso"]:
                        camel = f"Muon{comp.capitalize()}Sf"
                        syst_weights[f"{camel}Up"] = weights.weight(modifier=f"muon_{comp}_sfUp")
                        syst_weights[f"{camel}Down"] = weights.weight(modifier=f"muon_{comp}_sfDown")
                    syst_weights["MuonTrigSfUp"] = weights.weight(modifier="muon_trig_sfUp")
                    syst_weights["MuonTrigSfDown"] = weights.weight(modifier="muon_trig_sfDown")
                if tight_electrons is not None and era_key in ELECTRON_JSONS:
                    for comp in ["reco", "id", "trig"]:
                        camel = f"Electron{comp.capitalize()}Sf"
                        syst_weights[f"{camel}Up"] = weights.weight(modifier=f"electron_{comp}_sfUp")
                        syst_weights[f"{camel}Down"] = weights.weight(modifier=f"electron_{comp}_sfDown")
        
        else:  # is_data
            weights.add("data", np.ones(n, dtype=np.float32))
            syst_weights = { 
                "Nominal":  weights.weight(),
            }

        return weights, syst_weights

    def _fill_resolved(self, output, resolved_selections, process_name, ak4_jets, tight_leptons, weights, syst_weights):
        """Build resolved region masks and fill histograms + cutflows."""
        resolved_regions = {
            'wr_ee_resolved_dy_cr': resolved_selections.all(
                SEL_TWO_TIGHT_ELECTRONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_E_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_60_MLL_150, SEL_MLLJJ_GT800, SEL_JET_VETO_MAP,
            ),
            'wr_mumu_resolved_dy_cr': resolved_selections.all(
                SEL_TWO_TIGHT_MUONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_MU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_60_MLL_150, SEL_MLLJJ_GT800, SEL_JET_VETO_MAP,
            ),
            'wr_resolved_flavor_cr': resolved_selections.all(
                SEL_TWO_TIGHT_EM, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_EMU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLL_GT400, SEL_MLLJJ_GT800, SEL_JET_VETO_MAP,
            ),
            'wr_ee_resolved_sr': resolved_selections.all(
                SEL_TWO_TIGHT_ELECTRONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_E_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLL_GT400, SEL_MLLJJ_GT800, SEL_JET_VETO_MAP,
            ),
            'wr_mumu_resolved_sr': resolved_selections.all(
                SEL_TWO_TIGHT_MUONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_MU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLL_GT400, SEL_MLLJJ_GT800, SEL_JET_VETO_MAP,
            ),
        }
        for region, cuts in resolved_regions.items():
            fill_resolved_histograms(output, region, cuts, process_name, ak4_jets, tight_leptons, weights, syst_weights)

        fill_cutflows(output, resolved_selections, weights)

    def _fill_boosted(self, output, boosted_payload, process_name, weights, syst_weights, jet_veto_pass):
        """Unpack boosted payload, build region masks, and fill histograms."""
        boosted_sel, tight_lep, AK8_cand_dy, DY_loose_lep, AK8_cand, of_candidate, sf_candidate = boosted_payload
        boosted_sel.add(SEL_JET_VETO_MAP, jet_veto_pass)

        boosted_regions = {
            'wr_mumu_boosted_dy_cr': boosted_sel.all(
                SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
                SEL_ATLEAST1AK8_DPHI_GT2, SEL_MUMU_DYCR, SEL_JET_VETO_MAP,
            ),
            'wr_mumu_boosted_sr': boosted_sel.all(
                SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_MUMU_SR, SEL_AK8JETS_WITH_LSF,
                SEL_JET_VETO_MAP,
            ),
            'wr_ee_boosted_dy_cr': boosted_sel.all(
                SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
                SEL_ATLEAST1AK8_DPHI_GT2, SEL_EE_DYCR, SEL_JET_VETO_MAP,
            ),
            'wr_ee_boosted_sr': boosted_sel.all(
                SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_EE_SR, SEL_AK8JETS_WITH_LSF,
                SEL_JET_VETO_MAP,
            ),
            'wr_emu_boosted_flavor_cr': boosted_sel.all(
                SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_EMU_CR, SEL_AK8JETS_WITH_LSF,
                SEL_JET_VETO_MAP,
            ),
            'wr_mue_boosted_flavor_cr': boosted_sel.all(
                SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_MUE_CR, SEL_AK8JETS_WITH_LSF,
                SEL_JET_VETO_MAP,
            ),
        }
        for region, cuts in boosted_regions.items():
            if "dy_cr" in region:
                fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand_dy, DY_loose_lep, weights, syst_weights)
            elif "flavor_cr" in region:
                fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand, of_candidate, weights, syst_weights)
            else:
                fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand, sf_candidate, weights, syst_weights)

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

        # Apply lumi mask (data only).
        events = self.apply_lumi_mask(events, mc_campaign, is_data)

        # Build physics objects.
        tight_leptons, loose_leptons, lepton_masks = self.select_leptons(events)
        ak4_jets, ak8_jets, jet_masks = self.select_jets(events, mc_campaign, is_signal=(process_name == "Signal"))
        triggers = self.build_trigger_masks(events, mc_campaign)
        jet_veto_pass = jet_veto_event_mask(events, mc_campaign)

        # Resolved selections (only if requested).
        resolved_selections = None
        if self._region in ("resolved", "both"):
            resolved_selections = self.resolved_selections(
                tight_leptons,
                ak4_jets,
                lepton_masks=lepton_masks,
                jet_masks=jet_masks,
                triggers=triggers,
            )
            resolved_selections.add(SEL_JET_VETO_MAP, jet_veto_pass)

        # Boosted selections (skip gracefully if FatJet/lsf3 is absent).
        boosted_payload = None
        if self._region in ("boosted", "both"):
            try:
                has_fatjet = hasattr(events, "FatJet")
                has_lsf3 = has_fatjet and ("lsf3" in getattr(events.FatJet, "fields", []))
                if has_fatjet and has_lsf3:
                    boosted_payload = self.boosted_selections(events, mc_campaign, triggers=triggers)
                else:
                    logger.info("Skipping boosted selections: missing FatJet/lsf3 branches")
            except Exception as e:
                logger.warning(f"Boosted selections failed; skipping boosted histograms: {e}")

        # Event weights (needs tight leptons split by flavor for SF evaluation).
        tight_muons = tight_leptons[tight_leptons.flavor == "muon"]
        tight_electrons = tight_leptons[tight_leptons.flavor == "electron"]
        weights, syst_weights = self.build_event_weights(
            events, metadata, is_mc,
            tight_muons=tight_muons, tight_electrons=tight_electrons,
        )

        # Fill histograms.
        if resolved_selections is not None:
            self._fill_resolved(output, resolved_selections, process_name, ak4_jets, tight_leptons, weights, syst_weights)

        if boosted_payload is not None:
            self._fill_boosted(output, boosted_payload, process_name, weights, syst_weights, jet_veto_pass)

        nested_output = {dataset: {**output}}

        # When computing sumw on-the-fly, accumulate sum(genWeight) per dataset
        # so the caller can normalize after all chunks are processed.
        if self._compute_sumw and is_mc:
            nested_output[dataset]["_sumw"] = float(ak.sum(events.genWeight))

        return nested_output

    def postprocess(self, accumulator):
        return accumulator
