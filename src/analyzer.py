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
        # --- TIGHT masks ---
        electron_tight_mask = (
            (events.Electron.pt > 53)
            & (np.abs(events.Electron.eta) < 2.4)
            & (events.Electron.cutBased_HEEP)
        )
        muon_tight_mask = (
            (events.Muon.pt > 53)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.highPtId == 2)     # high-pT tight
            & (events.Muon.tkRelIso < 0.1)
        )

        # --- LOOSE base masks ---
        electron_loose_base = (
            (events.Electron.pt > 53)
            & (np.abs(events.Electron.eta) < 2.4)
            & (events.Electron.cutBased == 2)
        )
        muon_loose_base = (
            (events.Muon.pt > 53)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.highPtId == 2)
        )

        # --- LOOSE-only masks (exclude tights) ---
        electron_loose_mask = electron_loose_base & ~electron_tight_mask
        muon_loose_mask = muon_loose_base & ~muon_tight_mask

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

        return tight_leps, loose_leps

    def select_jets(self, events):
        # --- AK4 jet selection ---
        ak4_mask = (
            (events.Jet.pt > 40)
            & (np.abs(events.Jet.eta) < 2.4)
            & (events.Jet.isTightLeptonVeto)
        )
        ak4_jets = events.Jet[ak4_mask]

        # --- AK8 jet selection ---
        ak8_mask = (
            (events.FatJet.pt > 200)
            & (np.abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.jetId == 2)
            & (events.FatJet.msoftdrop > 40)
            # & (events.FatJet.lsf3 > 0.75) 
        )
        ak8_jets = events.FatJet[ak8_mask]

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
        selections.add("mll_gt200",  ak.fill_none(mll > 200, False))
        selections.add("mll_gt400",  ak.fill_none(mll > 400, False))
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

        return selections

    def boosted_selections(self, tight_leptons, resolved_selections, loose_leptons, ak8_jets, triggers=None):
        """
        Unified boosted selection builder.

        Always builds:
          - Baseline:     ~resolved AND lead tight lepton pT > 60 (and triggers if provided)
          - DY control:   exactly-one-tight, ΔR(loose,AK8)<0.8 (same flavor), and
                          [ (60<mll<150 AND Δφ(l_tight,AK8)>2.0) OR (no 60–150 AND LSF>0.75 jet with Δφ>2.0) ],
                          with flavor exclusivity (exactly one flavor passes)
          - Signal region: exactly-one-tight, ΔR(loose,AK8)<0.8 (same flavor), m_ll>200, m(l_tight,AK8)>800,
                          and at least one LSF>0.75 AK8 with Δφ(l_tight,AK8)>2.0; flavor exclusive

        """
        selections = PackedSelection()

        resolved_mask = resolved_selections.all(
                "two_tight_leptons", "lead_tight_lepton_pt60", "sublead_tight_pt53", "min_two_ak4_jets", "dr_all_pairs_gt0p4"
        )

        # --- Leading tight lepton & baseline  ---------------------------------------------------------
        lt = ak.pad_none(tight_leptons, 1)[:, 0]

        selections.add("not_resolved", ~resolved_mask)
        selections.add("lead_tight_lepton_pt60", ak.fill_none(lt.pt > 60, False))

        if triggers is not None:
            e_trig, mu_trig, emu_trig = triggers
            selections.add("e_trigger",   e_trig)
            selections.add("mu_trigger",  mu_trig)
            selections.add("emu_trigger", emu_trig)

        boosted_mask = selections.all("not_resolved", "lead_tight_lepton_pt60")

        # --- Common per-event building blocks (used by DY-CR and SR) ---------------------------------
        # Exactly one tight lepton
        exactly_one_tight = ak.num(tight_leptons) == 1
        selections.add("exactly_one_tight", ak.fill_none(exactly_one_tight, False))

        # Split loose leptons by flavor
        loose_e = loose_leptons[loose_leptons.flavor == "electron"]
        loose_m = loose_leptons[loose_leptons.flavor == "muon"]

        # ΔR(loose, AK8) < 0.8 (per flavor)
        pairs_e = ak.cartesian({"l": loose_e, "j": ak8_jets}, axis=1, nested=False)
        pairs_m = ak.cartesian({"l": loose_m, "j": ak8_jets}, axis=1, nested=False)
        dr_e = pairs_e["l"].deltaR(pairs_e["j"])
        dr_m = pairs_m["l"].deltaR(pairs_m["j"])
        has_e_dr08 = ak.fill_none(ak.any(dr_e < 0.8, axis=-1), False)
        has_m_dr08 = ak.fill_none(ak.any(dr_m < 0.8, axis=-1), False)
        selections.add("has_e_dr08", has_e_dr08)
        selections.add("has_m_dr08", has_m_dr08)

        # m(l_tight, l_loose) per flavor
        m_ll_e = (lt[:, None] + loose_e).mass
        m_ll_m = (lt[:, None] + loose_m).mass

        # For DY-CR: 60 < mll < 150
        has_e_mll_60_150 = ak.fill_none(ak.any((m_ll_e > 60) & (m_ll_e < 150), axis=-1), False)
        has_m_mll_60_150 = ak.fill_none(ak.any((m_ll_m > 60) & (m_ll_m < 150), axis=-1), False)
        selections.add("has_e_mll_60_150", has_e_mll_60_150)
        selections.add("has_m_mll_60_150", has_m_mll_60_150)

        # For SR: mll > 200
        has_e_mll_gt200 = ak.fill_none(ak.any(m_ll_e > 200, axis=-1), False)
        has_m_mll_gt200 = ak.fill_none(ak.any(m_ll_m > 200, axis=-1), False)
        selections.add("has_e_mll_gt200", has_e_mll_gt200)
        selections.add("has_m_mll_gt200", has_m_mll_gt200)

        # Common m(l_tight, AK8) > 800 (SR)
        m_ltj = (lt[:, None] + ak8_jets).mass
        has_m_ltj_gt800 = ak.fill_none(ak.any(m_ltj > 800, axis=-1), False)
        selections.add("has_m_ltj_gt800", has_m_ltj_gt800)

        # Δφ(l_tight, AK8) conditions
        dphi_any_ak8 = lt.deltaphi(ak8_jets)
        any_dphi_gt2 = ak.fill_none(ak.any(dphi_any_ak8 > 2.0, axis=-1), False)  # used in DY-CR branch A
        selections.add("ak8_dphi_gt2_no_lsf", any_dphi_gt2)

        ak8_with_lsf = ak8_jets[ak8_jets.lsf3 > 0.75]
        dphi_lsf = lt.deltaphi(ak8_with_lsf)
        any_dphi_lsf_gt2 = ak.fill_none(ak.any(dphi_lsf > 2.0, axis=-1), False)  # DY-CR branch B and SR req
        selections.add("ak8_lsf_dphi_gt2", any_dphi_lsf_gt2)

        # --- DY control region (flavor-exclusive) -----------------------------------------------------
        e_branch_a = has_e_mll_60_150 & any_dphi_gt2
        e_branch_b = (~has_e_mll_60_150) & any_dphi_lsf_gt2
        m_branch_a = has_m_mll_60_150 & any_dphi_gt2
        m_branch_b = (~has_m_mll_60_150) & any_dphi_lsf_gt2

        e_pass_dycr = exactly_one_tight & has_e_dr08 & (e_branch_a | e_branch_b)
        m_pass_dycr = exactly_one_tight & has_m_dr08 & (m_branch_a | m_branch_b)

        selections.add("electron_pass_dycr", ak.fill_none(e_pass_dycr, False))
        selections.add("muon_pass_dycr",     ak.fill_none(m_pass_dycr, False))

        exclusive_one_flavor_dycr = e_pass_dycr ^ m_pass_dycr
        selections.add("exclusive_one_flavor_dycr", ak.fill_none(exclusive_one_flavor_dycr, False))

        dy_cr_mask = boosted_mask & ak.fill_none(exclusive_one_flavor_dycr, False)

        # --- Signal region (flavor-exclusive) ---------------------------------------------------------
        e_pass_sr = exactly_one_tight & has_e_dr08 & has_e_mll_gt200 & has_m_ltj_gt800
        m_pass_sr = exactly_one_tight & has_m_dr08 & has_m_mll_gt200 & has_m_ltj_gt800

        selections.add("electron_pass_bsr", ak.fill_none(e_pass_sr, False))
        selections.add("muon_pass_bsr",     ak.fill_none(m_pass_sr, False))

        exclusive_one_flavor_bsr = e_pass_sr ^ m_pass_sr
        selections.add("exclusive_one_flavor_bsr", ak.fill_none(exclusive_one_flavor_bsr, False))

        sr_mask = boosted_mask & ak.fill_none(exclusive_one_flavor_bsr, False) & any_dphi_lsf_gt2

        return selections

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

            # Your special-case DY scaling (kept as requested)
            if mc_campaign == "RunIISummer20UL18" and process_name == "DYJets":
                event_weight *= (59.84 * 1000.0)

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

        else:  # is_data
            weights.add("data", np.ones(n, dtype=np.float32))

        return weights

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
                    weight=w_cut * sw,
                )

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
        print(f"\n\nresolved_selections: {resolved_selections}\n\n")

        # TO-DO
        # Boosted selections
        boosted_selections = self.boosted_selections(tight_leptons, resolved_selections, loose_leptons, ak8_jets, triggers=triggers)
        print(f"\n\nboosted_selections: {boosted_selections}\n\n")        

        weights = self.build_event_weights(events, metadata, is_mc, is_data, mc_campaign, process_name)

        syst_weights = {
            "Nominal":  weights.weight(),
            "LumiUp":   weights.weight(modifier="lumiUp"),  
            "LumiDown": weights.weight(modifier="lumiDown"),
        }

        # Define the resolved regions
        resolved_regions = {
            'wr_ee_resolved_dy_cr': resolved_selections.all('lead_tight_lepton_pt60', 'sublead_tight_pt53', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', '60_mll_150', 'mlljj_gt800', 'two_tight_electrons'),
            'wr_mumu_resolved_dy_cr': resolved_selections.all('lead_tight_lepton_pt60', 'sublead_tight_pt53', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', '60_mll_150', 'mlljj_gt800', 'two_tight_muons'),
            'wr_resolved_flavor_cr': resolved_selections.all('two_tight_em', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', 'mll_gt400', 'mlljj_gt800'),
            'wr_ee_resolved_sr': resolved_selections.all('two_tight_electrons', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', 'mll_gt400', 'mlljj_gt800'),
            'wr_mumu_resolved_sr': resolved_selections.all('two_tight_muons', 'lead_tight_lepton_pt60', 'sublead_tight_pt53', 'min_two_ak4_jets', 'dr_all_pairs_gt0p4', 'mll_gt400', 'mlljj_gt800'),
        }

        # Fill the resolved histograms
        for region, cuts in resolved_regions.items():
            self.fill_resolved_histograms(output, region, cuts, process_name, ak4_jets, tight_leptons, weights, syst_weights)

        
        # Fill the cutflow histograms
        cutflow_regions = {
            "wr_ee_resolved": {
                "cutflow_order": ["two_tight_electrons","e_trigger", "min_two_ak4_jets", "dr_all_pairs_gt0p4", "mll_gt200", "mlljj_gt800", "mll_gt400"],
            },
            "wr_mumu_resolved": {
                "cutflow_order": ["two_tight_muons", "mu_trigger", "min_two_ak4_jets", "dr_all_pairs_gt0p4", "mll_gt200", "mlljj_gt800", "mll_gt400"],
            },
        }

        for region, info in cutflow_regions.items():
            order = info["cutflow_order"]

            # Weighted cutflow
            cf = resolved_selections.cutflow(*order, weights=weights)

            res = cf.yieldhist(weighted=True)
            h_onecut, h_cum = res[0], res[1]
            output.setdefault("cutflow", {})
            output["cutflow"].setdefault(region, {})
            output["cutflow"][region]["onecut"] = h_onecut
            output["cutflow"][region]["cumulative"] = h_cum

            # Unweighted cutflow
            cf_unw = resolved_selections.cutflow(*order, weights=None)
            res_unw = cf_unw.yieldhist(weighted=False)
            h_onecut_unw, h_cum_unw = res_unw[0], res_unw[1]
            output["cutflow"][region]["onecut_unweighted"] = h_onecut_unw
            output["cutflow"][region]["cumulative_unweighted"] = h_cum_unw

        nested_output = {
            dataset: {
                **output,
            }
        }
        
        return nested_output

    def postprocess(self, accumulator):
        return accumulator
