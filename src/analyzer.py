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


def some_event_weight(ones):
    return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

def _minimal_repeated_unit(s: str) -> str:
    """
    Heuristic: find the smallest substring which, when repeated, can reconstruct s.
    If none obvious, return s itself.
    """
    if not isinstance(s, str) or len(s) == 0:
        return s
    for size in range(1, len(s) // 2 + 1):
        if len(s) % size != 0:
            continue
        unit = s[:size]
        if unit * (len(s) // size) == s:
            return unit
    return s  # fallback


class WrAnalysis(processor.ProcessorABC):
    def __init__(self, mass_point, sf_file=None):
        self._signal_sample = mass_point

        self.make_output = lambda: {
            'pt_leading_lepton':        self.create_hist('pt_leadlep',        'process', 'region', (200,   0, 2000), r'$p_{T}$ of the leading lepton [GeV]'),
            'eta_leading_lepton':       self.create_hist('eta_leadlep',       'process', 'region', (60,   -3,    3), r'$\eta$ of the leading lepton'),
            'phi_leading_lepton':       self.create_hist('phi_leadlep',       'process', 'region', (80,   -4,    4), r'$\phi$ of the leading lepton'),

            'pt_subleading_lepton':     self.create_hist('pt_subleadlep',     'process', 'region', (200,   0, 2000), r'$p_{T}$ of the subleading lepton [GeV]'),
            'eta_subleading_lepton':    self.create_hist('eta_subleadlep',    'process', 'region', (60,   -3,    3), r'$\eta$ of the subleading lepton'),
            'phi_subleading_lepton':    self.create_hist('phi_subleadlep',    'process', 'region', (80,   -4,    4), r'$\phi$ of the subleading lepton'),

            'pt_leading_jet':           self.create_hist('pt_leadjet',           'process', 'region', (200,   0, 2000), r'$p_{T}$ of the leading jet [GeV]'),
            'eta_leading_jet':          self.create_hist('eta_leadjet',          'process', 'region', (60,   -3,    3), r'$\eta$ of the leading jet'),
            'phi_leading_jet':          self.create_hist('phi_leadjet',          'process', 'region', (80,   -4,    4), r'$\phi$ of the leading jet'),

            'pt_subleading_jet':        self.create_hist('pt_subleadjet',        'process', 'region', (200,   0, 2000), r'$p_{T}$ of the subleading jet [GeV]'),
            'eta_subleading_jet':       self.create_hist('eta_subleadjet',       'process', 'region', (60,   -3,    3), r'$\eta$ of the subleading jet'),
            'phi_subleading_jet':       self.create_hist('phi_subleadjet',       'process', 'region', (80,   -4,    4), r'$\phi$ of the subleading jet'),

            'mass_dilepton':            self.create_hist('mass_dilepton',            'process', 'region', (5000,  0, 5000), r'$m_{\ell\ell}$ [GeV]'),
            'pt_dilepton':              self.create_hist('pt_dilepton',              'process', 'region', (200,   0, 2000), r'$p_{T,\ell\ell}$ [GeV]'),

            'mass_dijet':               self.create_hist('mass_dijet',               'process', 'region', (500,   0, 5000), r'$m_{jj}$ [GeV]'),
            'pt_dijet':                 self.create_hist('pt_dijet',                 'process', 'region', (500,   0, 5000), r'$p_{T,jj}$ [GeV]'),

            'mass_threeobject_leadlep':  self.create_hist('mass_threeobject_leadlep',  'process', 'region', (800,   0, 8000), r'$m_{\ell jj}$ [GeV]'),
            'pt_threeobject_leadlep':    self.create_hist('pt_threeobject_leadlep',    'process', 'region', (800,   0, 8000), r'$p_{T,\ell jj}$ [GeV]'),

            'mass_threeobject_subleadlep': self.create_hist('mass_threeobject_subleadlep', 'process', 'region', (800,   0, 8000), r'$m_{\ell jj}$ [GeV]'),
            'pt_threeobject_subleadlep':   self.create_hist('pt_threeobject_subleadlep',   'process', 'region', (800,   0, 8000), r'$p_{T,\ell jj}$ [GeV]'),

            'mass_fourobject':        self.create_hist('mass_fourobject',        'process', 'region', (800,   0, 8000), r'$m_{\ell\ell jj}$ [GeV]'),
            'pt_fourobject':          self.create_hist('pt_fourobject',          'process', 'region', (800,   0, 8000), r'$p_{T,\ell\ell jj}$ [GeV]'),
        }

        # ——— Load SF lookup if provided ———
        if sf_file:
            fname = os.path.basename(sf_file)
            self.variable = fname.replace("_sf.json", "")
            with open(sf_file) as jf:
                data = json.load(jf)
            edges = np.array(data["edges"], dtype=float)
            sf_EE = np.array(data["sf_ee_resolved_dy_cr"], dtype=float)
            sf_MM = np.array(data["sf_mumu_resolved_dy_cr"], dtype=float)

            self.lookup_EE = dense_lookup(sf_EE, [edges])
            self.lookup_MM = dense_lookup(sf_MM, [edges])
            logger.info(f"Loaded {self.variable} SF lookup from {sf_file}")
        else:
            self.variable = None
            self.lookup_EE = None
            self.lookup_MM = None

    def create_hist(self, name, process, region, bins, label):
        """Helper function to create histograms."""
        return (
            hist.Hist.new
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="region",  label="Analysis Region", growth=True)
            .StrCat([], name="syst",    label="Systematic", growth=True)  # <— NEW
            .Reg(*bins, name=name, label=label)
            .Weight()
        )
#        return (
#            hist.Hist.new.StrCat([], name="process", label="Process", growth=True)
#            .StrCat([], name="region", label="Analysis Region", growth=True)
#            .Reg(*bins, name=name, label=label)
#            .Weight()
#        )

    def selectElectrons(self, events):
        tight_electrons = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased_HEEP)
        loose_electrons = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased == 2)
        return events.Electron[tight_electrons], events.Electron[loose_electrons]

    def selectMuons(self, events):
        tight_muons = (events.Muon.pt > 53) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2) & (events.Muon.tkRelIso < 0.1)
        loose_muons = (events.Muon.pt > 53) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2)
        return events.Muon[tight_muons], events.Muon[loose_muons]

    def selectJets(self, events):
        ak4_jets = (events.Jet.pt > 40) & (np.abs(events.Jet.eta) < 2.4) & (events.Jet.isTightLeptonVeto)
        return events.Jet[ak4_jets]

    def check_mass_point_resolved(self):
        match = re.match(r"WR(\d+)_N(\d+)", self._signal_sample)
        if match:
            mwr, mn = int(match.group(1)), int(match.group(2))
            ratio = mn / mwr
            if ratio < 0.1:
                raise NotImplementedError(
                    f"Choose a resolved sample (MN/MWR > 0.1). For this sample, MN/MWR = {ratio:.2f}."
                )
        else:
            raise ValueError(f"Invalid mass point format: {self._signal_sample}")

    def add_resolved_selections(self, selections, tightElectrons, tightMuons, AK4Jets, mlljj, dr_jl_min, dr_j1j2, dr_l1l2):
        selections.add("twoTightLeptons", (ak.num(tightElectrons) + ak.num(tightMuons)) == 2)
        selections.add("minTwoAK4Jets", ak.num(AK4Jets) >= 2)
        selections.add("leadTightLeptonPt60", (ak.any(tightElectrons.pt > 60, axis=1) | ak.any(tightMuons.pt > 60, axis=1)))
        selections.add("mlljj>800", mlljj > 800)
        selections.add("dr>0.4", (dr_jl_min > 0.4) & (dr_j1j2 > 0.4) & (dr_l1l2 > 0.4))

    def fill_basic_histograms(self, output, region, cut, process_name, jets, leptons, weights, syst_weights):
        variables = [
            ('pt_leading_lepton',         leptons[:, 0].pt,    'pt_leadlep'),
            ('eta_leading_lepton',        leptons[:, 0].eta,   'eta_leadlep'),
            ('phi_leading_lepton',        leptons[:, 0].phi,   'phi_leadlep'),
            ('pt_subleading_lepton',      leptons[:, 1].pt,    'pt_subleadlep'),
            ('eta_subleading_lepton',     leptons[:, 1].eta,   'eta_subleadlep'),
            ('phi_subleading_lepton',     leptons[:, 1].phi,   'phi_subleadlep'),
            ('pt_leading_jet',            jets[:, 0].pt,       'pt_leadjet'),
            ('eta_leading_jet',           jets[:, 0].eta,      'eta_leadjet'),
            ('phi_leading_jet',           jets[:, 0].phi,      'phi_leadjet'),
            ('pt_subleading_jet',         jets[:, 1].pt,       'pt_subleadjet'),
            ('eta_subleading_jet',        jets[:, 1].eta,      'eta_subleadjet'),
            ('phi_subleading_jet',        jets[:, 1].phi,      'phi_subleadjet'),
            ('mass_dilepton',             (leptons[:, 0] + leptons[:, 1]).mass, 'mass_dilepton'),
            ('pt_dilepton',               (leptons[:, 0] + leptons[:, 1]).pt,   'pt_dilepton'),
            ('mass_dijet',                (jets[:, 0] + jets[:, 1]).mass,       'mass_dijet'),
            ('pt_dijet',                  (jets[:, 0] + jets[:, 1]).pt,         'pt_dijet'),
            ('mass_threeobject_leadlep',  (leptons[:, 0] + jets[:, 0] + jets[:, 1]).mass, 'mass_threeobject_leadlep'),
            ('pt_threeobject_leadlep',    (leptons[:, 0] + jets[:, 0] + jets[:, 1]).pt,   'pt_threeobject_leadlep'),
            ('mass_threeobject_subleadlep', (leptons[:, 1] + jets[:, 0] + jets[:, 1]).mass, 'mass_threeobject_subleadlep'),
            ('pt_threeobject_subleadlep',  (leptons[:, 1] + jets[:, 0] + jets[:, 1]).pt,   'pt_threeobject_subleadlep'),
            ('mass_fourobject',           (leptons[:, 0] + leptons[:, 1] + jets[:, 0] + jets[:, 1]).mass, 'mass_fourobject'),
            ('pt_fourobject',             (leptons[:, 0] + leptons[:, 1] + jets[:, 0] + jets[:, 1]).pt,   'pt_fourobject'),
        ]

        if self.variable is not None:
            for _, vals_array, axis_name in variables:
                if axis_name == self.variable:
                    vals_all = vals_array
                    break

        for hist_name, values, axis_name in variables:
            vals = values[cut]
            w = weights.weight()[cut]

            if process_name == "DYJets" and self.lookup_EE is not None:
                if region.startswith("wr_ee_resolved_dy_cr") or region.startswith("wr_ee_resolved_sr"):
                    corr = self.lookup_EE(vals_all[cut])
                elif region.startswith("wr_mumu_resolved_dy_cr") or region.startswith("wr_mumu_resolved_sr"):
                    corr = self.lookup_MM(vals_all[cut])
                else:
                    corr = 1.0
                w = w * corr

            # Fill once per systematic label
            for syst_label, syst_w_full in syst_weights.items():
                syst_w = syst_w_full[cut]
                output[hist_name].fill(
                    process=process_name,
                    region=region,
                    syst=syst_label,           # <— NEW axis
                    **{axis_name: vals},
                    weight=w * syst_w
                )

#            output[hist_name].fill(
#                process=process_name,
#                region=region,
#                **{axis_name: vals},
#                weight=w
#            )

    def some_event_weight(ones):
        return (1.0 + np.array([0.05, -0.05], dtype=np.float32)) * ones[:, None]

    def process(self, events):
        output = self.make_output()
        metadata = events.metadata

        mc_campaign = metadata.get("era", "")
        process_name = metadata.get("physics_group", "")
        dataset = metadata.get("sample", "")
        isRealData = not hasattr(events, "genWeight")

        events.add_systematic("Toy", "UpDownSystematic", "weight", some_event_weight)
        ones = np.ones(len(events), dtype=np.float32)
        toy_up   = events.systematics.Toy.up.weight_Toy
        toy_down = events.systematics.Toy.down.weight_Toy

        syst_weights = {
            "Nominal":      ones,
            "ToyUp":        toy_up,
            "ToyDown":      toy_down,
        }

#        logger.info(f"\n\nAnalyzing {len(events)} {dataset} events.\n\n")
#        if process_name == "DYJets":
#            events.add_systematic("scale_var", "UpDownSystematic", "weight", flat_variation)

        if isRealData:
            syst_weights = {"Nominal": ones}
            if mc_campaign == "RunIISummer20UL18":
                lumi_mask = LumiMask("data/lumis/RunII/2018/RunIISummer20UL18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
            elif mc_campaign in ("Run3Summer22", "Run3Summer22EE"):
                lumi_mask = LumiMask("data/lumis/Run3/2022/Run3Summer22/Cert_Collisions2022_355100_362760_Golden.txt")
            events = events[lumi_mask(events.run, events.luminosityBlock)]

        if process_name == "Signal":
            self.check_mass_point_resolved()

        # Object selection
        tightElectrons, _ = self.selectElectrons(events)
        nTightElectrons = ak.num(tightElectrons)

        tightMuons, _ = self.selectMuons(events)
        nTightMuons = ak.num(tightMuons)

        AK4Jets = self.selectJets(events)
        nAK4Jets = ak.num(AK4Jets)

        # Event variables
        tightLeptons = ak.with_name(ak.concatenate((tightElectrons, tightMuons), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons = ak.pad_none(tightLeptons[ak.argsort(tightLeptons.pt, axis=1, ascending=False)], 2, axis=1)

        AK4Jets = ak.pad_none(AK4Jets, 2, axis=1)
        mjj = ak.fill_none((AK4Jets[:, 0] + AK4Jets[:, 1]).mass, False)

        mll = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1]).mass, False)
        mlljj = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1] + AK4Jets[:, 0] + AK4Jets[:, 1]).mass, False)

        dr_jl_min = ak.fill_none(ak.min(AK4Jets[:, :2].nearest(tightLeptons).delta_r(AK4Jets[:, :2]), axis=1), False)
        dr_j1j2 = ak.fill_none(AK4Jets[:, 0].delta_r(AK4Jets[:, 1]), False)
        dr_l1l2 = ak.fill_none(tightLeptons[:, 0].delta_r(tightLeptons[:, 1]), False)

        # Event selections
        selections = PackedSelection()
        self.add_resolved_selections(selections, tightElectrons, tightMuons, AK4Jets, mlljj, dr_jl_min, dr_j1j2, dr_l1l2)

        # Trigger selections
        if mc_campaign in ("RunIISummer20UL18", "Run2Autumn18"):
            eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            muTrig = events.HLT.Mu50 | events.HLT.OldMu100 | events.HLT.TkMu100
            selections.add("eeTrigger", (eTrig & (nTightElectrons == 2) & (nTightMuons == 0)))
            selections.add("mumuTrigger", (muTrig & (nTightElectrons == 0) & (nTightMuons == 2)))
            selections.add("emuTrigger", (eTrig & muTrig & (nTightElectrons == 1) & (nTightMuons == 1)))
        elif mc_campaign in ("Run3Summer22", "Run3Summer23BPix", "Run3Summer22EE", "Run3Summer23"):
            eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            muTrig = events.HLT.Mu50 | events.HLT.HighPtTkMu100
            selections.add("eeTrigger", (eTrig & (nTightElectrons == 2) & (nTightMuons == 0)))
            selections.add("mumuTrigger", (muTrig & (nTightElectrons == 0) & (nTightMuons == 2)))
            selections.add("emuTrigger", ((eTrig | muTrig) & (nTightElectrons == 1) & (nTightMuons == 1)))

        # Event Weights
        weights = Weights(len(events))
        if not (not hasattr(events, "genWeight")):  # is MC
            eventWeight = abs(np.sign(events.event))
            if mc_campaign == "RunIISummer20UL18" and process_name == "DYJets":
                eventWeight = eventWeight * 1.35

            if process_name != "Signal":
                sf = metadata['xsec'] / metadata['nevts']
                eventWeight = eventWeight * sf
            else:
                sf = metadata['xsec'] / metadata['nevts']
                eventWeight = eventWeight * sf
        else:
            eventWeight = abs(np.sign(events.event))

        weights.add("event_weight", weight=eventWeight)

        selections.add("eejj", ((ak.num(tightElectrons) == 2) & (ak.num(tightMuons) == 0)))
        selections.add("mumujj", ((ak.num(tightElectrons) == 0) & (ak.num(tightMuons) == 2)))
        selections.add("emujj", ((ak.num(tightElectrons) == 1) & (ak.num(tightMuons) == 1)))

        # mll selections
        selections.add("60mll150", ((mll > 60) & (mll < 150)))
        selections.add("400mll", (mll > 400))

        # Define regions
        regions = {
            'wr_ee_resolved_dy_cr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'eeTrigger', 'mlljj>800', 'dr>0.4', '60mll150', 'eejj'],
            'wr_mumu_resolved_dy_cr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mumuTrigger', 'mlljj>800', 'dr>0.4', '60mll150', 'mumujj'],
            'wr_resolved_flavor_cr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'emuTrigger', 'mlljj>800', 'dr>0.4', '400mll', 'emujj'],
            'wr_ee_resolved_sr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'eeTrigger', 'mlljj>800', 'dr>0.4', '400mll', 'eejj'],
            'wr_mumu_resolved_sr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mumuTrigger', 'mlljj>800', 'dr>0.4', '400mll', 'mumujj'],
        }

        for region, cuts in regions.items():
            cut = selections.all(*cuts)
            self.fill_basic_histograms(output, region, cut, process_name, AK4Jets, tightLeptons, weights, syst_weights)

        nested_output = {
            dataset: {
                **output,
            }
        }
        
        return nested_output

    def postprocess(self, accumulator):
        return accumulator
