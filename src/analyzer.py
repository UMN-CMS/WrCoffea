from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
#from coffea.lumi_tools import LumiData, LumiMask, LumiList
import awkward as ak
import hist.dask as dah
import hist
import numpy as np
import re
import time
import logging
import warnings
import dask_awkward as dak
warnings.filterwarnings("ignore",module="coffea.*")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WrAnalysis(processor.ProcessorABC):
    def __init__(self, mass_point=None,exclusive=False):
        self._signal_sample = mass_point
        self.exc=exclusive
        
        self.make_output = lambda: {
            'Lepton_0_Pt': self.create_hist('pt_leadlep', 'process', 'region', (200, 0, 2000), r'p_{T} of the leading lepton [GeV]'),
            'Lepton_0_Eta': self.create_hist('eta_leadlep', 'process', 'region', (60, -3, 3), r'#eta of the leading lepton'),
            'Lepton_0_Phi': self.create_hist('phi_leadlep', 'process', 'region', (80, -4, 4), r'#phi of the leading lepton'),
            'Lepton_1_Pt': self.create_hist('pt_subleadlep', 'process', 'region', (200, 0, 2000), r'p_{T} of the subleading lepton [GeV]'),
            'Lepton_1_Eta': self.create_hist('eta_subleadlep', 'process', 'region', (60, -3, 3), r'#eta of the subleading lepton'),
            'Lepton_1_Phi': self.create_hist('phi_subleadlep', 'process', 'region', (80, -4, 4), r'#phi of the subleading lepton'),
            'Jet_0_Pt': self.create_hist('pt_leadjet', 'process', 'region', (200, 0, 2000), r'p_{T} of the leading jet [GeV]'),
            'Jet_0_Eta': self.create_hist('eta_leadjet', 'process', 'region', (60, -3, 3), r'#eta of the leading jet [GeV]'),
            'Jet_0_Phi': self.create_hist('phi_leadjet', 'process', 'region', (80, -4, 4), r'#phi of the leading jet'),
            'Jet_1_Pt': self.create_hist('pt_subleadjet', 'process', 'region', (200, 0, 2000), r'p_{T} of the subleading jet [GeV]'),
            'Jet_1_Eta': self.create_hist('eta_subleadjet', 'process', 'region', (60, -3, 3), r'#eta of the subleading jet [GeV]'),
            'Jet_1_Phi': self.create_hist('phi_subleadjet', 'process', 'region', (80, -4, 4), r'#phi of the subleading jet'),
            'ZCand_Mass': self.create_hist('mass_dileptons', 'process', 'region', (5000, 0, 5000), r'm_{ll} [GeV]'),
            'ZCand_Pt': self.create_hist('pt_dileptons', 'process', 'region', (200, 0, 2000), r'p^{T}_{ll} [GeV]'),
            'Dijet_Mass': self.create_hist('mass_dijet', 'process', 'region', (500, 0, 5000), r'm_{jj} [GeV]'),
            'Dijet_Pt': self.create_hist('pt_dijet', 'process', 'region', (500, 0, 5000), r'p^{T}_{jj} [GeV]'),
            'NCand_Lepton_0_Mass': self.create_hist('mass_threeobject_leadlep', 'process', 'region', (800, 0, 8000), r'm_{ljj} [GeV]'),
            'NCand_Lepton_0_Pt': self.create_hist('pt_threeobject_leadlep', 'process', 'region', (800, 0, 8000), r'p^{T}_{ljj} [GeV]'),
            'NCand_Lepton_1_Mass': self.create_hist('mass_threeobject_subleadlep', 'process', 'region', (800, 0, 8000), r'm_{ljj} [GeV]'),
            'NCand_Lepton_1_Pt': self.create_hist('pt_threeobject_subleadlep', 'process', 'region', (800, 0, 8000), r'p^{T}_{ljj} [GeV]'),
            'WRCand_Mass': self.create_hist('mass_fourobject', 'process', 'region', (800, 0, 8000), r'm_{lljj} [GeV]'),
            'WRCand_Pt': self.create_hist('pt_fourobject', 'process', 'region', (800, 0, 8000), r'p^{T}_{lljj} [GeV]'),
            'mass_WR': self.create_hist('mass_WR','process','region', (100, 0,8000), r'$m_{W_R}$ [GeV]'),
        }

    def create_hist(self, name, process, region, bins, label):
        """Helper function to create histograms."""
        return dah.hist.Hist(
            hist.axis.StrCategory([], name="process", label="Process", growth=True),
            hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
            hist.axis.Regular(*bins, name=name, label=label),
            hist.storage.Weight(),
        )

    def selectElectrons(self, events):
        """Select tight and loose electrons."""
        tight_electrons = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased_HEEP)
        loose_electrons = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4) & (events.Electron.cutBased == 2)
        return events.Electron[tight_electrons], events.Electron[loose_electrons]

    def selectMuons(self, events):
        """Select tight and loose muons."""
        tight_muons = (events.Muon.pt > 53) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2) & (events.Muon.tkRelIso < 0.1)
        loose_muons= (events.Muon.pt > 53) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2)
        return events.Muon[tight_muons], events.Muon[loose_muons]

    def selectJets(self, events):
        """Select AK4 and AK8 jets."""
#        ak4_jets = (np.abs(events.Jet.eta) < 2.4) & (events.Jet.isTightLeptonVeto)

        # Usual Requirement
        ak4_jets = (events.Jet.pt > 40) & (np.abs(events.Jet.eta) < 2.4) & (events.Jet.isTightLeptonVeto)
        return events.Jet[ak4_jets]

    def check_mass_point_resolved(self):
        """Check if the specified mass point is a resolved sample.

        Raises:
            NotImplementedError: If MN/MWR is less than 0.2, indicating an unresolved sample.
            ValueError: If the mass point format in _signal_sample is invalid.
        """
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
        if self.exc:
            selections.add("minTwoAK4Jets", ak.num(AK4Jets) == 2)
        else:
            selections.add("minTwoAK4Jets", ak.num(AK4Jets) >= 2)
        selections.add("leadTightLeptonPt60", (ak.any(tightElectrons.pt > 60, axis=1) | ak.any(tightMuons.pt > 60, axis=1)))
        selections.add("mlljj>800", mlljj > 800)
        selections.add("dr>0.4", (dr_jl_min > 0.4) & (dr_j1j2 > 0.4) & (dr_l1l2 > 0.4))

    def fill_basic_histograms(self, output, region, cut,  process, jets, leptons, weights):
        """Helper function to fill histograms dynamically."""
        # Define a list of variables and their corresponding histograms
        variables = [
            ('Lepton_0_Pt', leptons[:, 0].pt, 'pt_leadlep'),
            ('Lepton_0_Eta', leptons[:, 0].eta, 'eta_leadlep'),
            ('Lepton_0_Phi', leptons[:, 0].phi, 'phi_leadlep'),
            ('Lepton_1_Pt', leptons[:, 1].pt, 'pt_subleadlep'),
            ('Lepton_1_Eta', leptons[:, 1].eta, 'eta_subleadlep'),
            ('Lepton_1_Phi', leptons[:, 1].phi, 'phi_subleadlep'),
            ('Jet_0_Pt', jets[:, 0].pt, 'pt_leadjet'),
            ('Jet_0_Eta', jets[:, 0].eta, 'eta_leadjet'),
            ('Jet_0_Phi', jets[:, 0].phi, 'phi_leadjet'),
            ('Jet_1_Pt', jets[:, 1].pt, 'pt_subleadjet'),
            ('Jet_1_Eta', jets[:, 1].eta, 'eta_subleadjet'),
            ('Jet_1_Phi', jets[:, 1].phi, 'phi_subleadjet'),
            ('ZCand_Mass', (leptons[:, 0] + leptons[:, 1]).mass, 'mass_dileptons'),
            ('ZCand_Pt', (leptons[:, 0] + leptons[:, 1]).pt, 'pt_dileptons'),
            ('Dijet_Mass', (jets[:, 0] + jets[:, 1]).mass, 'mass_dijet'),
            ('Dijet_Pt', (jets[:, 0] + jets[:, 1]).pt, 'pt_dijet'),
            ('NCand_Lepton_0_Mass', (leptons[:, 0] + jets[:, 0] + jets[:, 1]).mass, 'mass_threeobject_leadlep'),
            ('NCand_Lepton_0_Pt', (leptons[:, 0] + jets[:, 0] + jets[:, 1]).pt, 'pt_threeobject_leadlep'),
            ('NCand_Lepton_1_Mass', (leptons[:, 1] + jets[:, 0] + jets[:, 1]).mass, 'mass_threeobject_subleadlep'),
            ('NCand_Lepton_1_Pt', (leptons[:, 1] + jets[:, 0] + jets[:, 1]).pt, 'pt_threeobject_subleadlep'),
            ('WRCand_Mass', (leptons[:, 0] + leptons[:, 1] + jets[:, 0] + jets[:, 1]).mass, 'mass_fourobject'),
            ('WRCand_Pt', (leptons[:, 0] + leptons[:, 1] + jets[:, 0] + jets[:, 1]).pt, 'pt_fourobject'),
        ]

        # Loop over variables and fill corresponding histograms
        for hist_name, values, axis_name in variables:
            output[hist_name].fill(
                process=process,
                region=region,
                **{axis_name: values[cut]},
                weight=weights.weight()[cut]
            )

    def process(self, events): 
        output = self.make_output()
        
        metadata = events.metadata
        mc_campaign = metadata["era"]
        process = metadata["physics_group"]
        dataset = metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        isMC = hasattr(events, "genWeight")

        logger.info(f"Analyzing {len(events)} {dataset} events.")

        if isRealData:
            if mc_campaign == "RunIISummer20UL18":
                lumi_mask = LumiMask("data/lumis/RunII/2018/RunIISummer20UL18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
            elif mc_campaign == "Run3Summer22":
                lumi_mask = LumiMask("data/lumis/Run3/2022/Run3Summer22/Cert_Collisions2022_355100_362760_Golden.txt")
            events = events[lumi_mask(events.run, events.luminosityBlock)]
#            num_events_after_mask = len(events["run"].compute())  # Compute using a lightweight branch
#            lumi_list = LumiList(events.run, events.luminosityBlock)
#            lumi_data = LumiData(f"lumi2018.csv", is_inst_lumi=False)
#            lumi = lumi_data.get_lumi(lumi_list)
#            print(lumi.compute())

        output['mc_campaign'] = mc_campaign
        output['process'] = process
        output['dataset'] = dataset
        if not isRealData:
            output['x_sec'] = events.metadata["xsec"] 

        # Process signal samples
        if process == "Signal": self.check_mass_point_resolved()

        # Object selection
        tightElectrons, _  = self.selectElectrons(events)
        nTightElectrons = ak.num(tightElectrons)

        tightMuons, _ = self.selectMuons(events)
        nTightMuons = ak.num(tightMuons)

        AK4Jets = self.selectJets(events)
        nAK4Jets = ak.num(AK4Jets)

        # Event variables
        tightLeptons = ak.with_name(ak.concatenate((tightElectrons, tightMuons), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons = ak.pad_none(tightLeptons[ak.argsort(tightLeptons.pt, axis=1, ascending=False)], 2, axis=1)
        AK4Jets = ak.pad_none(AK4Jets, 2, axis=1)

        mll = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1]).mass, False)
        mlljj = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1] + AK4Jets[:, 0] + AK4Jets[:, 1]).mass, False)

        dr_jl_min = ak.fill_none(ak.min(AK4Jets[:,:2].nearest(tightLeptons).delta_r(AK4Jets[:,:2]), axis=1), False)
        dr_j1j2 = ak.fill_none(AK4Jets[:,0].delta_r(AK4Jets[:,1]), False)
        dr_l1l2 = ak.fill_none(tightLeptons[:,0].delta_r(tightLeptons[:,1]), False)

        # Event selections
        selections = PackedSelection()
        self.add_resolved_selections(selections, tightElectrons, tightMuons, AK4Jets, mlljj, dr_jl_min, dr_j1j2, dr_l1l2)

        # Trigger selections
        if mc_campaign == "RunIISummer20UL18" or mc_campaign == "Run2Autumn18":
            eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            muTrig = events.HLT.Mu50 | events.HLT.OldMu100 | events.HLT.TkMu100
            selections.add("eeTrigger", (eTrig & (nTightElectrons == 2) & (nTightMuons == 0)))
            selections.add("mumuTrigger", (muTrig & (nTightElectrons == 0) & (nTightMuons == 2)))
            selections.add("emuTrigger", (eTrig & muTrig & (nTightElectrons == 1) & (nTightMuons == 1)))
        elif mc_campaign == "Run3Summer22" or mc_campaign == "Run3Summer23BPix" or mc_campaign == "Run3Summer22EE" or mc_campaign == "Run3Summer23":
            eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            muTrig = events.HLT.Mu50 | events.HLT.HighPtTkMu100
            selections.add("eeTrigger", (eTrig & (nTightElectrons == 2) & (nTightMuons == 0)))
            selections.add("mumuTrigger", (muTrig & (nTightElectrons == 0) & (nTightMuons == 2)))
            selections.add("emuTrigger", (eTrig & muTrig & (nTightElectrons == 1) & (nTightMuons == 1)))

        # Event Weights
        weights = Weights(size=None, storeIndividual=True)
        if isMC:
            eventWeight = events.genWeight
            if not process == "Signal":
                unqiue_gensumws = np.unique(events.genEventSumw.compute())
            output['sumw'] = ak.sum(eventWeight) if process == "Signal" else np.sum(unqiue_gensumws)
        elif isRealData:
            eventWeight = abs(np.sign(events.event))
        weights.add("event_weight", weight=eventWeight)

        # Channel selections
        selections.add("eejj", ((nTightElectrons == 2) & (nTightMuons == 0)))
        selections.add("mumujj", ((nTightElectrons == 0) & (nTightMuons == 2)))
        selections.add("emujj", ((nTightElectrons == 1) & (nTightMuons == 1)))

        # mll selections
        selections.add("60mll150", ((mll > 60) & (mll < 150)))
        selections.add("400mll", (mll > 400))
        selections.add("150mll", (mll > 150))

        # Define analysis regions
        regions = {
            'WR_EE_Resolved_DYCR': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'eeTrigger', 'mlljj>800', 'dr>0.4', '60mll150', 'eejj'],
            'WR_MuMu_Resolved_DYCR': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mumuTrigger', 'mlljj>800', 'dr>0.4', '60mll150', 'mumujj'],
            'WR_EMu_Resolved_CR': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'emuTrigger', 'mlljj>800', 'dr>0.4', '60mll150', 'emujj'],
            'WR_EMu_Resolved_Sideband': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'emuTrigger', 'mlljj>800', 'dr>0.4', '400mll', 'emujj'],
            'WR_EE_Resolved_SR': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'eeTrigger', 'mlljj>800', 'dr>0.4', '400mll', 'eejj'],
            'WR_MuMu_Resolved_SR': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mumuTrigger', 'mlljj>800', 'dr>0.4', '400mll', 'mumujj'],
        }

        # Fill histogram
        for region, cuts in regions.items():
            cut = selections.all(*cuts)
            self.fill_basic_histograms(output, region, cut, process, AK4Jets, tightLeptons, weights)
            if process == "Signal":
                wr_pdg_id = 34
                gen = events.GenPart

                # boolean jagged mask: [nEvents][nGenPart] 
                is_wr = abs(gen.pdgId) == wr_pdg_id

                # 1) broadcast your 1D event-weights to the same shape as gen.mass
                weights_arr       = weights.weight()                               # [nEvents]
                weights_per_part, _ = ak.broadcast_arrays(weights_arr, gen.mass)   # both now [nEvents][nGenPart]

                # 2) mask & flatten masses and weights
                wr_masses  = ak.flatten(gen.mass[is_wr])         
                wr_weights = ak.flatten(weights_per_part[is_wr])

                output['mass_WR'].fill(
                    process = process,
                    region  = "gen",         # or whatever label you like
                    mass_WR = wr_masses,
                    weight  = wr_weights
                )

        output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
