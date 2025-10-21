from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from coffea.lookup_tools.dense_lookup import dense_lookup
import awkward as ak
import hist
import numpy as np
import os
import re
import logging
import warnings
import json
from coffea.nanoevents.methods import vector
ak.behavior.update(vector.behavior)
warnings.filterwarnings("ignore", module="coffea.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Move this to a file that gets imported.
LUMI_JSONS = {
    "RunIISummer20UL18": "data/lumis/RunII/2018/RunIISummer20UL18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
    "Run3Summer22":      "data/lumis/Run3/2022/Run3Summer22/Cert_Collisions2022_355100_362760_Golden.txt",
    "Run3Summer22EE":    "data/lumis/Run3/2022/Run3Summer22/Cert_Collisions2022_355100_362760_Golden.txt",
}

# Systematic Uncertainties: Integrated Luminosity
LUMI_UNC = {
    "RunIISummer20UL18": 0.025,  # 2.5% (UL2018) https://cds.cern.ch/record/2676164/files/LUM-18-002-pas.pdf
    "Run3Summer22":      0.014,  # 1.4% (2022) https://cds.cern.ch/record/2890833/files/LUM-22-001-pas.pdf
    "Run3Summer22EE":    0.014,  # 1.4% (2022EE) https://cds.cern.ch/record/2890833/files/LUM-22-001-pas.pdf
}

class WrAnalysis(processor.ProcessorABC):
    def __init__(self, mass_point, sf_file=None):
        self._signal_sample = mass_point

        self.make_output = lambda: {
            # Kinematic histograms
            'pt_leading_lepton':           self.create_hist('pt_leadlep',                  (200,   0, 2000), r'$p_{T}$ of the leading lepton [GeV]'),
            'eta_leading_lepton':          self.create_hist('eta_leadlep',                 (60,   -3,    3), r'$\eta$ of the leading lepton'),
            'phi_leading_lepton':          self.create_hist('phi_leadlep',                 (80,   -4,    4), r'$\phi$ of the leading lepton'),
            'pt_subleading_lepton':        self.create_hist('pt_subleadlep',               (200,   0, 2000), r'$p_{T}$ of the subleading lepton [GeV]'),
            'eta_subleading_lepton':       self.create_hist('eta_subleadlep',              (60,   -3,    3), r'$\eta$ of the subleading lepton'),
            'phi_subleading_lepton':       self.create_hist('phi_subleadlep',              (80,   -4,    4), r'$\phi$ of the subleading lepton'),
            'pt_leading_jet':              self.create_hist('pt_leadjet',                  (200,   0, 2000), r'$p_{T}$ of the leading jet [GeV]'),
            'eta_leading_jet':             self.create_hist('eta_leadjet',                 (60,   -3,    3), r'$\eta$ of the leading jet'),
            'phi_leading_jet':             self.create_hist('phi_leadjet',                 (80,   -4,    4), r'$\phi$ of the leading jet'),
            'pt_subleading_jet':           self.create_hist('pt_subleadjet',               (200,   0, 2000), r'$p_{T}$ of the subleading jet [GeV]'),
            'eta_subleading_jet':          self.create_hist('eta_subleadjet',              (60,   -3,    3), r'$\eta$ of the subleading jet'),
            'phi_subleading_jet':          self.create_hist('phi_subleadjet',              (80,   -4,    4), r'$\phi$ of the subleading jet'),
            'mass_dilepton':               self.create_hist('mass_dilepton',               (5000,  0, 5000), r'$m_{\ell\ell}$ [GeV]'),
            'pt_dilepton':                 self.create_hist('pt_dilepton',                 (200,   0, 2000), r'$p_{T,\ell\ell}$ [GeV]'),
            'mass_dijet':                  self.create_hist('mass_dijet',                  (500,   0, 5000), r'$m_{jj}$ [GeV]'),
            'pt_dijet':                    self.create_hist('pt_dijet',                    (500,   0, 5000), r'$p_{T,jj}$ [GeV]'),
            'mass_threeobject_leadlep':    self.create_hist('mass_threeobject_leadlep',    (800,   0, 8000), r'$m_{\ell jj}$ [GeV]'),
            'pt_threeobject_leadlep':      self.create_hist('pt_threeobject_leadlep',      (800,   0, 8000), r'$p_{T,\ell jj}$ [GeV]'),
            'mass_threeobject_subleadlep': self.create_hist('mass_threeobject_subleadlep', (800,   0, 8000), r'$m_{\ell jj}$ [GeV]'),
            'pt_threeobject_subleadlep':   self.create_hist('pt_threeobject_subleadlep',   (800,   0, 8000), r'$p_{T,\ell jj}$ [GeV]'),
            'mass_fourobject':             self.create_hist('mass_fourobject',             (800,   0, 8000), r'$m_{\ell\ell jj}$ [GeV]'),
            'pt_fourobject':               self.create_hist('pt_fourobject',               (800,   0, 8000), r'$p_{T,\ell\ell jj}$ [GeV]'),

            'pt_leading_AK8Jets':        self.create_hist('pt_leadAK8Jets',         (200,   0, 2000), r'$p_{T}$ of the leading  AK8Jets [GeV]'),
            'eta_leading_AK8Jets':       self.create_hist('eta_leadAK8Jets',        (60,   -3,    3), r'$\eta$ of theleading  AK8Jets'),
            'phi_leading_AK8Jets':       self.create_hist('phi_leadAK8Jets',        (80,   -4,    4), r'$\phi$ of theleading  AK8Jets'),
            'LSF_leading_AK8Jets':        self.create_hist('LSF_leadingAK8Jets',         (200,   0, 1.1), r'LSF of leading AK8Jets'),
            'mass_twoobject':        self.create_hist('mass_twoobject',         (800,   0, 8000), r'$m_{\ell\ell jj}$ [GeV]'),
            'pt_twoobject':          self.create_hist('pt_twoobject',           (800,   0, 8000), r'$p_{T,\ell\ell jj}$ [GeV]'),
            'count' : self.create_hist('count', (100,0,100), r'count'),
            'dPhi_leading_tightlepton_AK8Jet':       self.create_hist('dPhi_leadTightlep_AK8Jets',        (80,   -4,    4), r'$d\phi$ (leading Tight lepton, AK8 Jet)'),
        }

    def create_hist(self, name, bins, label):
        return (
            hist.Hist.new
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="region",  label="Analysis Region", growth=True)
            .StrCat([], name="syst",    label="Systematic", growth=True)
            .Reg(*bins, name=name, label=label)
            .Weight()
        )

    def apply_lumi_mask(self, events, mc_campaign, is_data):
        """
        Apply golden JSON luminosity mask to data.
        Returns filtered events (unchanged if MC or no mask).
        """
        if not is_data:
            return events

        json_path = os.environ.get("LUMI_JSON") or LUMI_JSONS.get(mc_campaign)
        if json_path:
            try:
                mask = LumiMask(json_path)
                events = events[mask(events.run, events.luminosityBlock)]
                if len(events) == 0:
                    logger.warning(f"All events removed by lumi mask for era '{mc_campaign}'.")
            except OSError as e:
                logger.warning(f"Failed to load lumi JSON '{json_path}': {e}")
        else:
            logger.warning(f"No lumi JSON found for era '{mc_campaign}'. Data left unmasked.")

        return events

    def select_leptons(self, events):
        # --- Split pT/eta (kinematics) and ID components ---
        ele_pteta_mask = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4)
        mu_pteta_mask  = (events.Muon.pt > 53)     & (np.abs(events.Muon.eta) < 2.4)

        ele_id_mask = events.Electron.cutBased_HEEP
        mu_id_mask  = (events.Muon.highPtId == 2) & (events.Muon.tkRelIso < 0.1)

        # --- TIGHT masks (unchanged behavior: pT/eta AND ID) ---
        electron_tight_mask = ele_pteta_mask & ele_id_mask
        muon_tight_mask     = mu_pteta_mask  & mu_id_mask

        # --- LOOSE base masks (as you had) ---
        electron_loose_base = ele_pteta_mask & (events.Electron.cutBased == 2)
        muon_loose_base     = mu_pteta_mask  & (events.Muon.highPtId == 2)

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

        # --- Expose masks for later (pt/eta-only vs ID-only cutflows); does not change returns ---
        self._ele_pteta_mask = ele_pteta_mask
        self._mu_pteta_mask  = mu_pteta_mask
        self._ele_id_mask    = ele_id_mask
        self._mu_id_mask     = mu_id_mask

        return tight_leps, loose_leps

    def select_jets(self, events):
        # ---------------------------
        # AK4 jets
        # ---------------------------
        ak4_pteta_mask = (events.Jet.pt > 40) & (np.abs(events.Jet.eta) < 2.4)
        ak4_id_mask    = events.Jet.isTightLeptonVeto  # quality/ID-only

        ak4_mask = ak4_pteta_mask & ak4_id_mask
        ak4_jets = events.Jet[ak4_mask]

        # ---------------------------
        # AK8 jets
        # ---------------------------
        ak8_pteta_mask = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4)
        ak8_id_mask    = (events.FatJet.jetId == 2)  # keep ID-only separate from kinematics

        ak8_extra_sel  = (events.FatJet.msoftdrop > 40)
        # ak8_extra_sel = ak8_extra_sel & (events.FatJet.lsf3 > 0.75)

        # Full (unchanged overall selection): pT/eta AND ID AND extra selection(s)
        ak8_mask = ak8_pteta_mask & ak8_id_mask & ak8_extra_sel
        ak8_jets = events.FatJet[ak8_mask]

        # Expose split masks for cutflow bookkeeping (does not change returns)
        self._ak4_pteta_mask = ak4_pteta_mask
        self._ak4_id_mask    = ak4_id_mask

        self._ak8_pteta_mask = ak8_pteta_mask
        self._ak8_id_mask    = ak8_id_mask

        return ak4_jets, ak8_jets

    def build_trigger_masks(self, events, mc_campaign):
        """
        Returns (e_trig, mu_trig, emu_trig) per-event booleans.
        Handles missing HLT paths gracefully by defaulting to False.
        """
        import awkward as ak
        import numpy as np

        n = len(events)
        HLT = getattr(events, "HLT", None)

        def hlt(name):
            if HLT is not None and hasattr(HLT, name):
                return getattr(HLT, name)
            return ak.Array(np.zeros(n, dtype=bool))  # fallback

        # Electrons (common)
        e_trig = hlt("Ele32_WPTight_Gsf") | hlt("Photon200") | hlt("Ele115_CaloIdVT_GsfTrkIdT")

        # Muons by era
        if mc_campaign in ("RunIISummer20UL18", "Run2Autumn18"):
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100")
        elif mc_campaign in ("Run3Summer22", "Run3Summer23BPix", "Run3Summer22EE", "Run3Summer23"):
            mu_trig = hlt("Mu50") | hlt("HighPtTkMu100")
        else:
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100") | hlt("HighPtTkMu100")

        emu_trig = e_trig | mu_trig
        return e_trig, mu_trig, emu_trig

    def resolved_selections(self, tight_leptons, ak4_jets, triggers=None):
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

        Plus flavor and trigger requirements
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
        selections.add("lead_tight_lepton_pt60", ak.fill_none(l1.pt > 60, False))
        selections.add("sublead_tight_pt53",     ak.fill_none(l2.pt > 53, False))
        selections.add("min_two_ak4_jets",       n_ak4 >= 2)

        # Invariant masses
        mll   = (l1 + l2).mass
        mlljj = (l1 + l2 + j1 + j2).mass

        # mll selections
        selections.add("60_mll_150",   ak.fill_none((mll > 60) & (mll < 150), False))
        selections.add("mll_gt200",    ak.fill_none(mll > 200, False))
        selections.add("mll_gt400",    ak.fill_none(mll > 400, False))
        selections.add("mlljj_gt800",  ak.fill_none(mlljj > 800, False))

        # ΔR > 0.4 between all pairs
        dr_ll   = ak.fill_none(l1.deltaR(l2) > 0.4, False)
        dr_l1j1 = ak.fill_none(l1.deltaR(j1) > 0.4, False)
        dr_l1j2 = ak.fill_none(l1.deltaR(j2) > 0.4, False)
        dr_l2j1 = ak.fill_none(l2.deltaR(j1) > 0.4, False)
        dr_l2j2 = ak.fill_none(l2.deltaR(j2) > 0.4, False)
        dr_jj   = ak.fill_none(j1.deltaR(j2) > 0.4, False)

        selections.add("dr_all_pairs_gt0p4", dr_ll & dr_l1j1 & dr_l1j2 & dr_l2j1 & dr_l2j2 & dr_jj)

        # Triggers
        if triggers is not None:
            e_trig, mu_trig, emu_trig = triggers
            selections.add("e_trigger",   e_trig)
            selections.add("mu_trigger",  mu_trig)
            selections.add("emu_trigger", emu_trig)

        # Jet-level counts per event
        n_ak4_pteta = ak.sum(self._ak4_pteta_mask, axis=1)
        n_ak4_id    = ak.sum(self._ak4_id_mask,    axis=1)
        selections.add("min_two_ak4_jets_pteta", n_ak4_pteta >= 2)
        selections.add("min_two_ak4_jets_id",    n_ak4_id    >= 2)

        # Lepton-level counts per event (separate by flavor)
        n_ele_pteta = ak.sum(self._ele_pteta_mask, axis=1)
        n_mu_pteta  = ak.sum(self._mu_pteta_mask,  axis=1)
        n_ele_id    = ak.sum(self._ele_id_mask,    axis=1)
        n_mu_id     = ak.sum(self._mu_id_mask,     axis=1)

        # pT/eta-only
        selections.add("two_pteta_electrons", n_ele_pteta == 2)
        selections.add("two_pteta_muons",     n_mu_pteta  == 2)
        selections.add("two_pteta_em",        (n_ele_pteta == 1) & (n_mu_pteta == 1))

        # ID-only
        selections.add("two_id_electrons", n_ele_id == 2)
        selections.add("two_id_muons",     n_mu_id  == 2)
        selections.add("two_id_em",        (n_ele_id == 1) & (n_mu_id == 1))

        return selections
    
    ### ----- Boosted Helper functions ----------- ###
    def remove_lepton(self,loose, tight):
        match = (loose.deltaR(tight))
        keep_mask = match >= 0.01
        return loose[keep_mask]
    
    def ElectronCut(self, cut_bitmap, id_level=1):
        """                                                                                                                                                                 
        Awkward-array-safe version of the ElectronCut function.                                                                                                             
        Ignores isolation bit (flag index 7) and applies ID threshold.                                                                                                      
        """
        nFlags = 10          # total number of bits (cuts)                                                                                                                  
        cut_size = 3         # 3 bits per cut                                                                                                                               
        ignore_flag = 7      # ignore the isolation flag                                                                                                                    
        mask_per_cut = (1 << cut_size) - 1  # 0b111 = 7                                                                                                                     
        # Define a vectorized helper                                                                                                                                        
        def passes(bitmap):
            for cut_nr in range(nFlags):
                if cut_nr == ignore_flag:
                    continue
                value = (bitmap >> (cut_nr * cut_size)) & mask_per_cut
                if value < id_level : 
                    return False
            return True
        # Apply per-electron check                                                                                                                                          
        mask = ak.Array([[passes(b) for b in event] for event in cut_bitmap])
        return mask
    
    def selectLooseElectrons(self, events):
        loose_noIso_mask = self.ElectronCut(events.Electron.vidNestedWPBitmap, id_level=2)
        loose_electrons = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4)  & ((events.Electron.cutBased_HEEP) | loose_noIso_mask)
        return events.Electron[loose_electrons]

    def selectLooseMuons(self, events):
        loose_muons = (events.Muon.pt > 53) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2)
        return events.Muon[loose_muons]
    
    def selectAK8Jets(self,events):
        ak8_jets = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4)  & (events.FatJet.msoftdrop > 40) & (events.FatJet.isTight)
        return events.FatJet[ak8_jets]

    def selectAK8Jets_withLSF(self,events):
        ak8_jets = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4)  & (events.FatJet.msoftdrop > 40) & (events.FatJet.isTight) & (events.FatJet.lsf3 > 0.75)
        return events.FatJet[ak8_jets]
    
    def boosted_selections(self, events, triggers=None):
        #tight_leptons, resolved_selections, loose_leptons, ak8_jets, triggers=None):
        selections = PackedSelection()
        ####  ---- boosted category of events ----- #####                                                                                                                   
        #### ---- object selections ---  ###
        looseElectrons = self.selectLooseElectrons(events)
        looseMuons = self.selectLooseMuons(events)
        AK8Jets = self.selectAK8Jets(events)
        AK8Jets_withLSF = self.selectAK8Jets_withLSF(events)
        AK4Jets_inc, _ = self.select_jets(events)
        # define tight by querying loose
        tight_mask_e = (looseElectrons.cutBased_HEEP)
        tightElectrons_inc = looseElectrons[tight_mask_e]
        tightElectrons_inc = tightElectrons_inc[ak.argsort(tightElectrons_inc.pt, axis=1, ascending=False)]
        ## define loose 
        cut_bitmap = looseElectrons.vidNestedWPBitmap
        loose_e_mask = self.ElectronCut(cut_bitmap, id_level=2)
        looseElectrons = looseElectrons[loose_e_mask]
        ## -- tight muons --- ##
        tight_mask_mu = (looseMuons.tkRelIso < 0.1)
        tightMuons_inc = looseMuons[tight_mask_mu]
        tightMuons_inc = tightMuons_inc[ak.argsort(tightMuons_inc.pt, axis=1, ascending=False)]

        looseLeptons = ak.with_name(ak.concatenate((looseElectrons, looseMuons), axis=1), 'PtEtaPhiMCandidate')
        looseLeptons = looseLeptons[ak.argsort(looseLeptons.pt, axis=1, ascending=False)]
        tightLeptons_inc = ak.with_name(ak.concatenate((tightElectrons_inc, tightMuons_inc), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons_inc = tightLeptons_inc[ak.argsort(tightLeptons_inc.pt, axis=1, ascending=False)]

        ## ------ not resolved events ----- #
        has_two_leptons = ak.num(tightLeptons_inc) >= 2
        muons_padded = ak.pad_none(tightLeptons_inc, 2, axis=1)
        # compute dr only for events with >=2 muons                                                                                                                         
        dr_l1l2 = ak.where(
            ak.num(tightLeptons_inc) >= 2,
            muons_padded[:,0].deltaR(muons_padded[:,1]),
            ak.full_like(ak.num(tightLeptons_inc), np.nan)
        )
        # For jets
        jpad = ak.pad_none(AK4Jets_inc, 2)
        j1, j2 = jpad[:, 0], jpad[:, 1] 
        dr_j1j2   = ak.fill_none(j1.deltaR(j2) > 0.4, False)

        has_two_jets = ak.num(AK4Jets_inc) >= 2

        # dr_jl_min: compute only if both jets and leptons exist                                                                                                            
        has_j_and_l = (ak.num(AK4Jets_inc) >= 1) & (ak.num(tightLeptons_inc) >= 1)
        dr_jl_min = ak.where( has_j_and_l, ak.min(AK4Jets_inc[:, :2].nearest(tightLeptons_inc).deltaR(AK4Jets_inc[:, :2]), axis=1), ak.full_like(has_j_and_l, np.nan))
        # Build all 2 leptons × 2 jets pairs                                                                                                                                
        dr_lj = ak.cartesian({"lep": tightLeptons_inc[:,:2], "jet": AK4Jets_inc[:,:2]}, axis=1)
        dr_lj_vals = dr_lj["lep"].deltaR(dr_lj["jet"])
        # Condition: all l-j separations > 0.4                                                                                                                              
        dr_lj_ok = ak.all(dr_lj_vals > 0.4, axis=1)
        resolved = (((ak.num(tightElectrons_inc)  + (ak.num(tightMuons_inc))) == 2) & (ak.num(AK4Jets_inc) >= 2) & (dr_l1l2 > 0.4) & (dr_j1j2 > 0.4) & (dr_jl_min >0.4)) #(dr_lj_ok))                                                                                                                                                                   
        boosted  = ~resolved
        selections.add("boostedtag",boosted)

        # ---------------------- Leading tight lepton pT>60 -------------------- #
        tightLepton_padded = ak.pad_none(tightLeptons_inc,1,axis=1)
        tight_lep   = tightLepton_padded[:, 0]
        lead_pdgid  = ak.fill_none(abs(tight_lep.pdgId), 0)
        is_lead_mu  = lead_pdgid == 13
        is_lead_e   = lead_pdgid == 11
        is_tight_pt = tight_lep.pt > 60
        selections.add("leadTightwithPt60",is_tight_pt)
        ## -- remove this tight lepton from loose lepton selection ---
        looseLeptons = self.remove_lepton(looseLeptons, tight_lep)

        # -- same-flavor and other-flavor loose collections --                                                                                                      
        sf_loose = looseLeptons[abs(looseLeptons.pdgId) == abs(tight_lep.pdgId)]
        of_loose = looseLeptons[abs(looseLeptons.pdgId) != abs(tight_lep.pdgId)]
        sf_loose = sf_loose[ak.argsort(sf_loose.pt, axis=1, ascending=False)]
        of_loose = of_loose[ak.argsort(of_loose.pt, axis=1, ascending=False)]

        # ---------------- DY pair check ----------------                                                                                                                  
        mll_pairs  = (tight_lep + sf_loose).mass
        mask_mll   = (mll_pairs > 60) & (mll_pairs < 150)
        has_dy_pair = ak.any(mask_mll, axis=1)
        # -------- picking loose SF lepton --- #
        DY_loose_lep  = ak.firsts(sf_loose[mask_mll])
        # ---------------- AK8 jet candidate ----------------                                                                                                               
        AK8Jets = AK8Jets[ak.argsort(AK8Jets.pt, axis=1, ascending=False)]
        flag_ak8Jet = ak.num(AK8Jets)>=1
        AK8Jets = ak.pad_none(AK8Jets, 1, axis=1)
        dphi       = ak.fill_none(abs(AK8Jets.delta_phi(tight_lep)),0.0)
        has_ak8_dphi_gt2 = ak.any(dphi > 2, axis=1)
        ak8_mask   = dphi > 2.0
        AK8_cand_dy   = ak.firsts(AK8Jets[ak8_mask])
        # -------------- asking for atleast 1 AK8 jet ---------
        selections.add("Atleast1AK8Jets & dPhi(J,tightLept)>2", flag_ak8Jet & has_ak8_dphi_gt2 )
        
        # ---------------- Case 1: DY CR ----------------                                                                                                                   
        dr_dy  = AK8_cand_dy.deltaR(DY_loose_lep)
        mlj_dy = ak.where(dr_dy < 0.8,
                          (tight_lep + AK8_cand_dy).mass,
                          (tight_lep + DY_loose_lep + AK8_cand_dy).mass)
        mll_dy = (tight_lep + DY_loose_lep).mass
        pt_dilept_dy = (tight_lep + DY_loose_lep).pt
        pt_lj_dy = ak.where(dr_dy < 0.8,
                          (tight_lep + AK8_cand_dy).pt,
                          (tight_lep + DY_loose_lep + AK8_cand_dy).pt)
        sublead_pdgID = abs(DY_loose_lep.pdgId)
        is_sublead_mu = sublead_pdgID == 13
        is_sublead_e = sublead_pdgID ==11
        DYCR_mask = has_dy_pair & (mlj_dy > 800)
        selections.add("DYCR_mask", DYCR_mask)
        # -------- veto extra tight leptons for DY CR --------                                                                                                              
        # ---- remove tight_lep and selected loose candidate from looseLepton collection                                                                                  
        extra_tight_mu = ak.sum(
            (looseMuons.tkRelIso < 0.1) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseMuons.deltaR(DY_loose_lep) > 0.01, True)),
            axis=1
        )
        extra_tight_el = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseElectrons.deltaR(DY_loose_lep) > 0.01, True)),
            axis=1
        )
        no_extra_tight_dyCR = (extra_tight_mu == 0) & (extra_tight_el == 0)

        # # ---------------- Case 2: SR (no DY, SF near AK8, no OF near AK8) ----------------                                                                              
        flag_ak8Jet_lsf = ak.num(AK8Jets_withLSF)>=1
        AK8Jets_withLSF = ak.pad_none(AK8Jets_withLSF, 1, axis=1)
        dphi_lsf       = ak.fill_none(abs(AK8Jets_withLSF.delta_phi(tight_lep)),0.0)
        has_ak8_dphi_gt2_lsf = ak.any(dphi_lsf > 2, axis=1)
        ak8_mask_lsf   = dphi_lsf > 2.0
        AK8_cand   = ak.firsts(AK8Jets_withLSF[ak8_mask_lsf])
        selections.add("AK8JetswithLSF", flag_ak8Jet_lsf & has_ak8_dphi_gt2_lsf )
        dr_sf = AK8_cand.deltaR(sf_loose)
        mask_sf = dr_sf < 0.8
        sf_candidate = ak.firsts(sf_loose[mask_sf]) # ---- SF lepton candidate passing dR condition                                                                         
        sf_exist = ak.num(sf_loose[mask_sf])>=1
        dr_of = AK8_cand.deltaR(of_loose)
        mask_of = dr_of < 0.8
        of_candidate = ak.firsts(of_loose[mask_of]) #  -  ----- OF lepton candidate passing dR condition                                                                    
        of_exist = ak.num(of_loose[mask_of])>=1

        is_sr = (~has_dy_pair) & (~ak.is_none(sf_candidate)) & ak.is_none(of_candidate) #sf_exist & (~of_exist) #(~ak.is_none(sf_candidate)) & ak.is_none(of_candidate)    
        sublead_pdgID = abs(sf_candidate.pdgId)
        is_sublead_mu_sr = sublead_pdgID == 13
        is_sublead_e_sr = sublead_pdgID ==11
        mlj_sr   = (tight_lep + AK8_cand).mass
        mll_sr = (tight_lep + sf_candidate).mass
        pt_dilept_sr  = (tight_lep + AK8_cand).pt
        pt_lj_sr = (tight_lep + sf_candidate).pt

        SR_mask = is_sr & (mlj_sr > 800) & (mll_sr > 200)
        selections.add("ee(mumu)SR", SR_mask)
        # -------- veto extra tight leptons for SR --------                                                                                                                 
        extra_tight_mu_sr = ak.sum(
            (looseMuons.tkRelIso < 0.1) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseMuons.deltaR(sf_candidate) > 0.01, True)),
            axis=1
        )
        extra_tight_el_sr = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseElectrons.deltaR(sf_candidate) > 0.01, True)),
            axis=1
        )
        no_extra_tight_sr = (extra_tight_mu_sr == 0) & (extra_tight_el_sr == 0)

        # ---------------- Case 3: Flavor CR (no DY, OF near AK8) ----------------                                                                                          
        is_cr = (~has_dy_pair) & (~ak.is_none(of_candidate))  & ak.is_none(sf_candidate) #(~sf_exist) & of_exist #(~ak.is_none(of_candidate))  & ak.is_none(sf_candidate)  
        sublead_pdgID = abs(of_candidate.pdgId)
        is_sublead_mu_cr = sublead_pdgID == 13
        is_sublead_e_cr = sublead_pdgID ==11

        mlj_cr  = (tight_lep + AK8_cand).mass
        mll_cr = (tight_lep + of_candidate).mass
        pt_dilept_cr  = (tight_lep + AK8_cand).pt
        pt_lj_cr = (tight_lep + of_candidate).pt

        CR_mask = is_cr & (mlj_cr > 800) & (mll_cr > 200 )
        selections.add("e(mu) or mu(e)CR", CR_mask)
        # -------- veto extra tight leptons for flavor CR --------                                                                                                          
        extra_tight_mu_cr = ak.sum(
            (looseMuons.tkRelIso < 0.1) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseMuons.deltaR(of_candidate) > 0.01, True)),
            axis=1
        )
        extra_tight_el_cr = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseElectrons.deltaR(of_candidate) > 0.01, True)),
            axis=1
        )
        no_extra_tight_flav_cr = (extra_tight_mu_cr == 0) & (extra_tight_el_cr == 0)

        # ----------------- Triggers ----- #
        if triggers is not None:
            eTrig, muTrig, emu_trig = triggers
        
        # ---------------- Region assignments ---------------- #                                                                                                            
        mumu_dy_cr = muTrig & DYCR_mask & is_lead_mu & no_extra_tight_dyCR
        ee_dy_cr   = eTrig & DYCR_mask & is_lead_e & no_extra_tight_dyCR
        emu_cr = eTrig  & CR_mask & is_lead_e & no_extra_tight_flav_cr # lead e, loose mu                                                                        
        mue_cr = muTrig & CR_mask & is_lead_mu & no_extra_tight_flav_cr # lead mu, loose e                                                                        
        ee_sr = eTrig & SR_mask & is_lead_e & no_extra_tight_sr
        mumu_sr = muTrig & SR_mask &  is_lead_mu & no_extra_tight_sr
        selections.add("mumu-dy_cr", mumu_dy_cr & is_sublead_mu)
        selections.add("ee-dy_cr",   ee_dy_cr & is_sublead_e)
        selections.add("mumu_sr", mumu_sr & is_sublead_mu_sr)
        selections.add("ee_sr",   ee_sr & is_sublead_e_sr)
        selections.add("emu-cr",     emu_cr & is_sublead_mu_cr)
        selections.add("mue-cr",     mue_cr & is_sublead_e_cr)        
        return selections, tight_lep, AK8_cand_dy,DY_loose_lep, AK8_cand,of_candidate, sf_candidate 

    def build_event_weights(self, events, metadata, is_mc, is_data, mc_campaign, process_name):
        """
        Minimal weights:
          - MC: xsec/nevts normalization (+ optional DY UL18 scale) + lumi Up/Down
          - Data: unit weights
        NO genWeight, NO L1 prefire, NO pileup.
        """
        n = len(events)
        weights = Weights(n)

        if is_mc:
            # Cross-section normalization
            xsec     = float(metadata.get("xsec", 1.0))
            n_evts   = float(metadata.get("nevts", 1.0))
            norm     = (xsec / n_evts) if n_evts > 0 else 0.0

            # Base unit weights (no genWeight)
            event_weight = np.ones(n, dtype=np.float32)

            # Manually scaling lumi for cutflow purposes
            #            if mc_campaign == "RunIISummer20UL18" and process_name == "DYJets":
            #                event_weight *= 59.83 * 1000

            event_weight *= norm
            weights.add("event_weight", event_weight)

            # Lumi uncertainty (weight-only)
            if mc_campaign not in LUMI_UNC:
                raise KeyError(f"No luminosity uncertainty defined for era '{mc_campaign}'. Add it to LUMI_UNC.")
            delta = float(LUMI_UNC[mc_campaign])

            ones = np.ones(n, dtype=np.float32)
            weights.add(
                "lumi",
                ones,
                weightUp  = ones * (1.0 + delta),
                weightDown= ones * (1.0 - delta),
            )

            syst_weights = {
                "Nominal":  weights.weight(),
                "LumiUp":   weights.weight(modifier="lumiUp"),
                "LumiDown": weights.weight(modifier="lumiDown"),
            }
        
        else:  # is_data
            weights.add("data", np.ones(n, dtype=np.float32))
            syst_weights = { 
                "Nominal":  weights.weight(),
            }

        return weights, syst_weights

    def fill_resolved_histograms(self, output, region, cut, process_name, jets, leptons, weights, syst_weights):
        leptons_cut = leptons[cut]
        jets_cut    = jets[cut]
        w_cut       = weights.weight()[cut]
        syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}
        
        resolved_specs = [
            ('pt_leading_lepton',           lambda L,J: L[:, 0].pt,                                   'pt_leadlep'),
            ('eta_leading_lepton',          lambda L,J: L[:, 0].eta,                                  'eta_leadlep'),
            ('phi_leading_lepton',          lambda L,J: L[:, 0].phi,                                  'phi_leadlep'),
            ('pt_subleading_lepton',        lambda L,J: L[:, 1].pt,                                   'pt_subleadlep'),
            ('eta_subleading_lepton',       lambda L,J: L[:, 1].eta,                                  'eta_subleadlep'),
            ('phi_subleading_lepton',       lambda L,J: L[:, 1].phi,                                  'phi_subleadlep'),
            ('pt_leading_jet',              lambda L,J: J[:, 0].pt,                                   'pt_leadjet'),
            ('eta_leading_jet',             lambda L,J: J[:, 0].eta,                                  'eta_leadjet'),
            ('phi_leading_jet',             lambda L,J: J[:, 0].phi,                                  'phi_leadjet'),
            ('pt_subleading_jet',           lambda L,J: J[:, 1].pt,                                   'pt_subleadjet'),
            ('eta_subleading_jet',          lambda L,J: J[:, 1].eta,                                  'eta_subleadjet'),
            ('phi_subleading_jet',          lambda L,J: J[:, 1].phi,                                  'phi_subleadjet'),
            ('mass_dilepton',               lambda L,J: (L[:, 0] + L[:, 1]).mass,                     'mass_dilepton'),
            ('pt_dilepton',                 lambda L,J: (L[:, 0] + L[:, 1]).pt,                       'pt_dilepton'),
            ('mass_dijet',                  lambda L,J: (J[:, 0] + J[:, 1]).mass,                     'mass_dijet'),
            ('pt_dijet',                    lambda L,J: (J[:, 0] + J[:, 1]).pt,                       'pt_dijet'),
            ('mass_threeobject_leadlep',    lambda L,J: (L[:, 0] + J[:, 0] + J[:, 1]).mass,           'mass_threeobject_leadlep'),
            ('pt_threeobject_leadlep',      lambda L,J: (L[:, 0] + J[:, 0] + J[:, 1]).pt,             'pt_threeobject_leadlep'),
            ('mass_threeobject_subleadlep', lambda L,J: (L[:, 1] + J[:, 0] + J[:, 1]).mass,           'mass_threeobject_subleadlep'),
            ('pt_threeobject_subleadlep',   lambda L,J: (L[:, 1] + J[:, 0] + J[:, 1]).pt,             'pt_threeobject_subleadlep'),
            ('mass_fourobject',             lambda L,J: (L[:, 0] + L[:, 1] + J[:, 0] + J[:, 1]).mass, 'mass_fourobject'),
            ('pt_fourobject',               lambda L,J: (L[:, 0] + L[:, 1] + J[:, 0] + J[:, 1]).pt,   'pt_fourobject'),
        ]
        
        for hist_name, expr, axis_name in resolved_specs:
            vals = expr(leptons_cut, jets_cut)
            for syst_label, sw in syst_weights_cut.items():
                output[hist_name].fill(
                    process=process_name,
                    region=region,
                    syst=syst_label,
                    **{axis_name: vals},
                    weight=sw,
                )

    def fill_boosted_histograms(self, output, region, cut, process_name, leptons, ak8jets, looseleptons, weights, syst_weights):
        leptons_cut = leptons[cut]
        ak8jets_cut    = ak8jets[cut]
        w_cut       = weights.weight()[cut]
        syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}
        variables = [
            ('pt_leading_lepton',         leptons.pt,    'pt_leadlep'),
	    ('eta_leading_lepton',        leptons.eta,   'eta_leadlep'),
            ('phi_leading_lepton',        leptons.phi,   'phi_leadlep'),
            ('pt_subleading_lepton',      looseleptons.pt,    'pt_subleadlep'),
            ('eta_subleading_lepton',     looseleptons.eta,   'eta_subleadlep'),
            ('phi_subleading_lepton',     looseleptons.phi,   'phi_subleadlep'),
            ('pt_leading_AK8Jets',            ak8jets.pt,       'pt_leadAK8Jets'),
            ('eta_leading_AK8Jets',           ak8jets.eta,      'eta_leadAK8Jets'),
            ('phi_leading_AK8Jets',           ak8jets.phi,      'phi_leadAK8Jets'),            
            ('mass_dilepton',          (leptons + looseleptons).mass , 'mass_dilepton'),
            ('pt_dilepton',             (leptons + looseleptons).pt ,   'pt_dilepton'),
            ('mass_twoobject',           (leptons + ak8jets).mass , 'mass_twoobject'),
            ('pt_twoobject',             (leptons + ak8jets).pt ,   'pt_twoobject'),
            ('LSF_leading_AK8Jets', ak8jets.lsf3,'LSF_leadingAK8Jets'),
            ('dPhi_leading_tightlepton_AK8Jet',  abs(ak8jets.delta_phi(leptons)),'dPhi_leadTightlep_AK8Jets')            
	    ]
        if "boosted_dy_cr" in region:
            dr_dy  = ak8jets.deltaR(looseleptons)
            mlj_dy = ak.where(dr_dy < 0.8,
                          (leptons + ak8jets).mass,
                          (leptons + ak8jets + looseleptons).mass)
            pt_lj = ak.where(dr_dy < 0.8,
                          (leptons + ak8jets).pt,
                             (leptons + ak8jets + looseleptons).pt)
            variables = [
                ('pt_leading_lepton',         leptons.pt,    'pt_leadlep'),
                ('eta_leading_lepton',        leptons.eta,   'eta_leadlep'),
                ('phi_leading_lepton',        leptons.phi,   'phi_leadlep'),
                ('pt_subleading_lepton',      looseleptons.pt,    'pt_subleadlep'),
                ('eta_subleading_lepton',     looseleptons.eta,   'eta_subleadlep'),
                ('phi_subleading_lepton',     looseleptons.phi,   'phi_subleadlep'),
                ('pt_leading_AK8Jets',            ak8jets.pt,       'pt_leadAK8Jets'),
                ('eta_leading_AK8Jets',           ak8jets.eta,      'eta_leadAK8Jets'),
                ('phi_leading_AK8Jets',           ak8jets.phi,      'phi_leadAK8Jets'),
                ('mass_dilepton',          (leptons + looseleptons).mass , 'mass_dilepton'),
                ('pt_dilepton',             (leptons + looseleptons).pt ,   'pt_dilepton'),
                ('mass_twoobject',           mlj_dy , 'mass_twoobject'),
                ('pt_twoobject',             pt_lj ,   'pt_twoobject'),
                ('LSF_leading_AK8Jets', ak8jets.lsf3,'LSF_leadingAK8Jets'),
                ('dPhi_leading_tightlepton_AK8Jet',  abs(ak8jets.delta_phi(leptons)),'dPhi_leadTightlep_AK8Jets')
                ]
        for hist_name, expr, axis_name in variables:
            #vals = expr(leptons_cut, jets_cut)
            vals = expr[cut]
            for syst_label, sw in syst_weights_cut.items():
                output[hist_name].fill(
                    process=process_name,
                    region=region,
                    syst=syst_label,
                    **{axis_name: vals},
                    weight=sw,
                )

    def fill_cutflows(self, output, selections, weights):
        """
        Build flat and cumulative cutflows for ee, mumu, and em channels.
        Also store one-bin histograms for pT/eta-only vs ID-only (jets & leptons).
        """
        output.setdefault("cutflow", {})

        # --- convenience: get masks by name once
        def M(name): return selections.all(name)
        

        m_j2_id    = M("min_two_ak4_jets_id")
        m_j2_pteta = M("min_two_ak4_jets_pteta")

        m_two_pteta_e  = M("two_pteta_electrons")
        m_two_pteta_mu = M("two_pteta_muons")
        m_two_pteta_em = M("two_pteta_em")

        m_two_id_e  = M("two_id_electrons")
        m_two_id_mu = M("two_id_muons")
        m_two_id_em = M("two_id_em")

        m_e_trig  = M("e_trigger")
        m_mu_trig = M("mu_trigger")
        m_em_trig = M("emu_trigger")

        m_dr      = M("dr_all_pairs_gt0p4")
        m_mll200  = M("mll_gt200")
        m_mlljj8  = M("mlljj_gt800")
        m_mll400  = M("mll_gt400")

        w = weights.weight()

        def _onebin_hist(lastcut_name: str, mask, use_weights=True):
            """1 bin in [0,1), named by the last cut."""
            h = hist.Hist(
                hist.axis.Regular(1, 0, 1, name=lastcut_name, label=lastcut_name),
                storage=hist.storage.Weight(),
            )
            if use_weights:
                w_pass = ak.to_numpy(w[mask])
            else:
                # unweighted: count passing events
                w_pass = np.ones(np.count_nonzero(mask), dtype="f8")
            if w_pass.size:
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
        _store_both_in(output["cutflow"], "min_two_ak4_jets_pteta", m_j2_pteta)

        # --- Prepare flavor folders (+ jets folder for pt/eta vs ID)
        output["cutflow"].setdefault("ee", {})
        output["cutflow"].setdefault("mumu", {})
        output["cutflow"].setdefault("em", {})

        # --- Define cumulative chains per flavor (kept as-is)
        chains = {
            "ee":   ["min_two_ak4_jets_pteta", "min_two_ak4_jets_id", 
                      "e_trigger", "two_pteta_electrons", "two_id_electrons",  
                     "dr_all_pairs_gt0p4", "mlljj_gt800","mll_gt200", "mll_gt400"],
            "mumu": ["min_two_ak4_jets_pteta", "min_two_ak4_jets_id",
                     "mu_trigger", "two_pteta_muons", "two_id_muons", 
                     "dr_all_pairs_gt0p4",  "mlljj_gt800", "mll_gt200", "mll_gt400"],
            "em":   ["min_two_ak4_jets_pteta", "min_two_ak4_jets_id",
                     "emu_trigger", "two_pteta_em", "two_id_em", 
                     "dr_all_pairs_gt0p4", "mlljj_gt800", "mll_gt200",  "mll_gt400"],
        }

        name2mask = {
            "min_two_ak4_jets_id":    m_j2_id,
            "min_two_ak4_jets_pteta": m_j2_pteta,

            "two_pteta_electrons": m_two_pteta_e,
            "two_pteta_muons":     m_two_pteta_mu,
            "two_pteta_em":        m_two_pteta_em,

            "two_id_electrons":    m_two_id_e,
            "two_id_muons":        m_two_id_mu,
            "two_id_em":           m_two_id_em,

            "e_trigger":           m_e_trig,
            "mu_trigger":          m_mu_trig,
            "emu_trigger":         m_em_trig,

            "dr_all_pairs_gt0p4":  m_dr,
            "mll_gt200":           m_mll200,
            "mlljj_gt800":         m_mlljj8,
            "mll_gt400":           m_mll400,
        }

        # --- Step-by-step cumulative counters per flavor
        for flavor, steps in chains.items():

            cumu = name2mask[steps[0]].copy()
            bucket = output["cutflow"][flavor]
            for step in steps:
                if step != steps[0]:
                    cumu = cumu & name2mask[step]
                bucket[step] = _onebin_hist(step, cumu, use_weights=True)
                bucket[f"{step}_unweighted"] = _onebin_hist(step, cumu, use_weights=False)

        # --- Region cutflows: onecut + cumulative into flavor folders
        flavor_regions = {
            "ee":   chains["ee"],
            "mumu": chains["mumu"],
            "em":   chains["em"],
        }

        for flavor, order in flavor_regions.items():
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

    def process(self, events):
        output = self.make_output()
        metadata = events.metadata

        mc_campaign = metadata.get("era", "")
        process_name = metadata.get("physics_group", "")
        dataset = metadata.get("sample", "")

        is_mc   = hasattr(events, "genWeight")
        is_data = not is_mc

        # Apply lumi mask via helper
        events = self.apply_lumi_mask(events, mc_campaign, is_data)

        # Get leptons and jets
        tight_leptons, loose_leptons = self.select_leptons(events)
        ak4_jets, ak8_jets = self.select_jets(events)

        # Construct the electron and muon triggers
        triggers = self.build_trigger_masks(events, mc_campaign)

        # Resolved selections
        resolved_selections = self.resolved_selections(tight_leptons, ak4_jets, triggers=triggers)

        
        # TO-COMPLETE
        boosted_selection_list, tight_lep, AK8_cand_dy,DY_loose_lep, AK8_cand,of_candidate, sf_candidate = self.boosted_selections(events, triggers=triggers) #tight_leptons, resolved_selections, loose_leptons, ak8_jets, triggers=triggers)

        weights, syst_weights = self.build_event_weights(events, metadata, is_mc, is_data, mc_campaign, process_name)

        # Define the resolved regions
        resolved_regions = {
            'wr_ee_resolved_dy_cr': resolved_selections.all('two_tight_electrons', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'e_trigger', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', '60_mll_150', 'mlljj_gt800'),
            'wr_mumu_resolved_dy_cr': resolved_selections.all('two_tight_muons', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'mu_trigger', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', '60_mll_150', 'mlljj_gt800'),
            'wr_resolved_flavor_cr': resolved_selections.all('two_tight_em', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'emu_trigger', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', 'mll_gt400', 'mlljj_gt800'),
            'wr_ee_resolved_sr': resolved_selections.all('two_tight_electrons', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'e_trigger', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', 'mll_gt400', 'mlljj_gt800'),
            'wr_mumu_resolved_sr': resolved_selections.all('two_tight_muons', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'mu_trigger', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', 'mll_gt400', 'mlljj_gt800'),
        }
        # Fill the resolved histograms
        for region, cuts in resolved_regions.items():
            self.fill_resolved_histograms(output, region, cuts, process_name, ak4_jets, tight_leptons, weights, syst_weights)

        # Fill the resolved cutflow histograms
        self.fill_cutflows(output, resolved_selections, weights)

        boosted_regions = {
            'wr_mumu_boosted_dy_cr': boosted_selection_list.all('boostedtag', 'leadTightwithPt60','DYCR_mask','Atleast1AK8Jets & dPhi(J,tightLept)>2','mumu-dy_cr'),
	    'wr_mumu_boosted_sr': boosted_selection_list.all('boostedtag', 'leadTightwithPt60','mumu_sr','AK8JetswithLSF'),
	    'wr_ee_boosted_dy_cr': boosted_selection_list.all('boostedtag', 'leadTightwithPt60','DYCR_mask','Atleast1AK8Jets & dPhi(J,tightLept)>2','ee-dy_cr'),
            'wr_ee_boosted_sr': boosted_selection_list.all('boostedtag', 'leadTightwithPt60','ee_sr','AK8JetswithLSF'),
	    'wr_emu_boosted_flavor_cr': boosted_selection_list.all('boostedtag', 'leadTightwithPt60','emu-cr','AK8JetswithLSF'),
            'wr_mue_boosted_flavor_cr': boosted_selection_list.all('boostedtag', 'leadTightwithPt60','mue-cr','AK8JetswithLSF'),
        }
        for region, cuts in boosted_regions.items():
	    #cut = selections.all(*cuts)
            if "dy_cr" in region:
                self.fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand_dy,DY_loose_lep, weights,syst_weights)

            elif "flavor_cr" in region :
                self.fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand,of_candidate, weights,syst_weights)
            else :
                self.fill_boosted_histograms(output, region, cuts, process_name, tight_lep,AK8_cand, sf_candidate, weights,syst_weights)
                
        nested_output = {dataset: {**output}}
        
        return nested_output

    def postprocess(self, accumulator):
        return accumulator
