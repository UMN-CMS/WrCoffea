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
# import csv
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
            'Jet_2_Pt': self.create_hist('pt_thirdjet', 'process', 'region', (200, 0, 2000), r'p_{T} of the third jet [GeV]'),
            'Jet_2_Eta': self.create_hist('eta_thirdjet', 'process', 'region', (60, -3, 3), r'#eta of the third jet [GeV]'),
            'Jet_2_Phi': self.create_hist('phi_thirdjet', 'process', 'region', (80, -4, 4), r'#phi of the third jet'),
            'Delta_r20': self.create_hist('del_r20', 'process', 'region', (80, 0, 6), 'Delta R Third Jet and Lead Jet'),
            'Delta_r21': self.create_hist('del_r21', 'process', 'region', (80, 0, 6), 'Delta R Third Jet and SubLead Jet'),
            'ZCand_Mass': self.create_hist('mass_dileptons', 'process', 'region', (5000, 0, 5000), r'm_{ll} [GeV]'),
            'ZCand_Pt': self.create_hist('pt_dileptons', 'process', 'region', (200, 0, 2000), r'p^{T}_{ll} [GeV]'),
            'Dijet_Mass': self.create_hist('mass_dijet', 'process', 'region', (500, 0, 5000), r'm_{jj} [GeV]'),
            'Dijet_Pt': self.create_hist('pt_dijet', 'process', 'region', (500, 0, 5000), r'p^{T}_{jj} [GeV]'),
            'NCand_Lepton_0_Mass': self.create_hist('mass_threeobject_leadlep', 'process', 'region', (800, 0, 8000), r'm_{ljj} [GeV]'),
            'NCand_Lepton_0_Pt': self.create_hist('pt_threeobject_leadlep', 'process', 'region', (800, 0, 8000), r'p^{T}_{ljj} [GeV]'),
            'NCand_Lepton_1_Mass': self.create_hist('mass_threeobject_subleadlep', 'process', 'region', (800, 0, 8000), r'm_{ljj} [GeV]'),
            'NCand_Lepton_1_Pt': self.create_hist('pt_threeobject_subleadlep', 'process', 'region', (800, 0, 8000), r'p^{T}_{ljj} [GeV]'),
            'WRCand_Mass': self.create_hist('mass_fiveobject', 'process', 'region', (800, 0, 8000), r'm_{lljjj} [GeV]'),
            'WRCand_Pt': self.create_hist('pt_fiveobject', 'process', 'region', (800, 0, 8000), r'p^{T}_{lljjj} [GeV]'),
            'WRMass4_DeltaR':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(20, 0, 6, name='del_r', label=r'\Delta R_{min}'),
                hist.storage.Weight(),
            ),
            'WRMass5_DeltaR':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(20, 0, 6, name='del_r', label=r'\Delta R_{min}'),
                hist.storage.Weight(),
            ),
            'WRMass4_pT':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(50, 0, 600, name='pTrel', label=r'pT_{min}^{rel} [GeV]'),
                hist.storage.Weight(),
            ),
            'WRMass5_pT':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(50, 0, 600, name='pTrel', label=r'pT_{min}^{rel} [GeV]'),
                hist.storage.Weight(),
            ),
            'WRMass4_sin':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(20, 0, 1, name='sin', label=r'sin(\theta)_{min}'),
                hist.storage.Weight(),
            ),
            'WRMass5_sin':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(20, 0, 1, name='sin', label=r'sin(\theta)_{min}'),
                hist.storage.Weight(),
            ),
            'WRMass4_pTnorm':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(30, 0, 5, name='pTnorm', label=r'pT_rel/pT'),
                hist.storage.Weight(),
            ),
            'WRMass5_pTnorm':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(30, 0, 5, name='pTnorm', label=r'pT_rel/pT'),
                hist.storage.Weight(),
            ),
            'WRMass4_magic3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='magic', label=r'pT_rel/m_{jjjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass5_magic3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='magic', label=r'pT_rel/m_{jjjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass4_magic':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='magic', label=r'pT_rel/m_{jjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass5_magic':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='magic', label=r'pT_rel/m_{jjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass4_pseudomagic3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='pseudomagic', label=r'pT_rel/m_{jjj}'),
                hist.storage.Weight(),
            ),
            'WRMass5_pseudomagic3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='pseudomagic', label=r'pT_rel/m_{jjj}'),
                hist.storage.Weight(),
            ),
            'WRMass4_pseudomagic':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='pseudomagic', label=r'pT_rel/m_{jj}'),
                hist.storage.Weight(),
            ),
            'WRMass5_pseudomagic':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(50, 0, 1, name='pseudomagic', label=r'pT_rel/m_{jj}'),
                hist.storage.Weight(),
            ),
            'WRMass4_neutrino3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='mass_neutrino', label=r'm_{jjjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass5_neutrino3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='mass_neutrino', label=r'm_{jjjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass4_neutrino':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='mass_neutrino', label=r'm_{jjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass5_neutrino':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='mass_neutrino', label=r'm_{jjl_x}'),
                hist.storage.Weight(),
            ),
            'WRMass4_pseudo3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='pseudomass', label=r'm_{jjj}'),
                hist.storage.Weight(),
            ),
            'WRMass5_pseudo3':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='pseudomass', label=r'm_{jjj}'),
                hist.storage.Weight(),
            ),
            'WRMass4_pseudo':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='pseudomass', label=r'm_{jj}'),
                hist.storage.Weight(),
            ),
            'WRMass5_pseudo':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(100, 0, 8000, name='pseudomass', label=r'm_{jj}'),
                hist.storage.Weight(),
            ),
            'WRMass4_restDeltaR':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.axis.Regular(20, 0, 6, name='del_r', label=r'\Delta R_{min}'),
                hist.storage.Weight(),
            ),
            'WRMass5_restDeltaR':dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(100, 0, 8000, name='mass_fiveobject', label=r'm_{lljjj} [GeV]'),
                hist.axis.Regular(20, 0, 6, name='del_r', label=r'\Delta R_{min}'),
                hist.storage.Weight(),
            ),
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
        ak4_jets = (np.abs(events.Jet.eta) < 2.4) & (events.Jet.isTightLeptonVeto)

        # Usual Requirement
#        ak4_jets = (events.Jet.pt > 40) & (np.abs(events.Jet.eta) < 2.4) & (events.Jet.isTightLeptonVeto)
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

    def add_resolved_selections(self, selections, tightElectrons, tightMuons, AK4Jets, mlljj, dr_jl_min, dr_j1j2, dr_j1j3, dr_j2j3, dr_l1l2):
        selections.add("twoTightLeptons", (ak.num(tightElectrons) + ak.num(tightMuons)) == 2)
        
        if self.exc:
            selections.add("minTwoAK4Jets", ak.num(AK4Jets) == 3)
        else:
            selections.add("minTwoAK4Jets", ak.num(AK4Jets) >= 3)
        
        selections.add("leadTightLeptonPt60", (ak.any(tightElectrons.pt > 60, axis=1) | ak.any(tightMuons.pt > 60, axis=1)))
        selections.add("mlljj>800", mlljj > 800)
        selections.add("dr>0.4", (dr_jl_min > 0.4) & (dr_j1j2 > 0.4) & (dr_j1j3 > 0.4) & (dr_j2j3 > 0.4) & (dr_l1l2 > 0.4))

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
            ('Jet_2_Pt', jets[:, 2].pt, 'pt_thirdjet'),
            ('Jet_2_Eta', jets[:, 2].eta, 'eta_thirdjet'),
            ('Jet_2_Phi', jets[:, 2].phi, 'phi_thirdjet'),
            ('Delta_r20',jets[:,2].delta_r(jets[:,0]),'del_r20'),
            ('Delta_r21',jets[:,2].delta_r(jets[:,1]),'del_r21'),
            ('ZCand_Mass', (leptons[:, 0] + leptons[:, 1]).mass, 'mass_dileptons'),
            ('ZCand_Pt', (leptons[:, 0] + leptons[:, 1]).pt, 'pt_dileptons'),
            ('Dijet_Mass', (jets[:, 0] + jets[:, 1]).mass, 'mass_dijet'),
            ('Dijet_Pt', (jets[:, 0] + jets[:, 1]).pt, 'pt_dijet'),
            ('NCand_Lepton_0_Mass', (leptons[:, 0] + jets[:, 0] + jets[:, 1]).mass, 'mass_threeobject_leadlep'),
            ('NCand_Lepton_0_Pt', (leptons[:, 0] + jets[:, 0] + jets[:, 1]).pt, 'pt_threeobject_leadlep'),
            ('NCand_Lepton_1_Mass', (leptons[:, 1] + jets[:, 0] + jets[:, 1]).mass, 'mass_threeobject_subleadlep'),
            ('NCand_Lepton_1_Pt', (leptons[:, 1] + jets[:, 0] + jets[:, 1]).pt, 'pt_threeobject_subleadlep'),
            ('WRCand_Mass', (leptons[:, 0] + leptons[:, 1] + jets[:, 0] + jets[:, 1] + jets[:, 2]).mass, 'mass_fiveobject'),
            ('WRCand_Pt', (leptons[:, 0] + leptons[:, 1] + jets[:, 0] + jets[:, 1] + jets[:, 2]).pt, 'pt_fiveobject'),
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
        AK4Jets = ak.pad_none(AK4Jets, 3, axis=1)

        mll = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1]).mass, False)
        mlljj = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1] + AK4Jets[:, 0] + AK4Jets[:, 1]+ AK4Jets[:, 2]).mass, False)

        dr_jl_min = ak.fill_none(ak.min(AK4Jets[:,:3].nearest(tightLeptons).delta_r(AK4Jets[:,:3]), axis=1), False)
        dr_j1j2 = ak.fill_none(AK4Jets[:,0].delta_r(AK4Jets[:,1]), False)
        dr_j1j3 = ak.fill_none(AK4Jets[:,0].delta_r(AK4Jets[:,2]), False)
        dr_j2j3 = ak.fill_none(AK4Jets[:,1].delta_r(AK4Jets[:,2]), False)
        dr_l1l2 = ak.fill_none(tightLeptons[:,0].delta_r(tightLeptons[:,1]), False)

        # Event selections
        selections = PackedSelection()
        self.add_resolved_selections(selections, tightElectrons, tightMuons, AK4Jets, mlljj, dr_jl_min, dr_j1j2, dr_j1j3,dr_j2j3, dr_l1l2)

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

        writestring=[]
        for region, cuts in regions.items():
            cut = selections.all(*cuts)
            self.fill_basic_histograms(output, region, cut, process, AK4Jets, tightLeptons, weights)
            mlljj1= (tightLeptons[cut][:, 0] + tightLeptons[cut][:, 1] + AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1]).mass
            mlljjj= (tightLeptons[cut][:, 0] + tightLeptons[cut][:, 1] + AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1] + AK4Jets[cut][:, 2]).mass
            mjjjl1= (AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1] + AK4Jets[cut][:, 2] + tightLeptons[cut][:, 0]).mass
            mjjjl2= (AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1] + AK4Jets[cut][:, 2] + tightLeptons[cut][:, 1]).mass
            mjjl1= (AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1] + tightLeptons[cut][:, 0]).mass
            mjjl2= (AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1] + tightLeptons[cut][:, 1]).mass
            mjjj= (AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1] + AK4Jets[cut][:, 2]).mass
            mjj= (AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1]).mass
            dr_j3_min = ak.min(AK4Jets[cut][:,2].delta_r(AK4Jets[cut][:,:2]),axis=1)
            
            x0,y0,z0=AK4Jets[cut][:,0].x,AK4Jets[cut][:,0].y,AK4Jets[cut][:,0].z
            x1,y1,z1=AK4Jets[cut][:,1].x,AK4Jets[cut][:,1].y,AK4Jets[cut][:,1].z
            x2,y2,z2=AK4Jets[cut][:,2].x,AK4Jets[cut][:,2].y,AK4Jets[cut][:,2].z
            
            jet3mag=np.sqrt(x2*x2+y2*y2+z2*z2)
            cosine20=(x0*x2+y0*y2+z0*z2)/np.sqrt((x0*x0+y0*y0+z0*z0))/jet3mag
            cosine21=(x1*x2+y1*y2+z1*z2)/np.sqrt((x1*x1+y1*y1+z1*z1))/jet3mag
            sine20=ak.where(cosine20>0,np.sqrt(1-cosine20**2),1)
            sine21=ak.where(cosine21>0,np.sqrt(1-cosine21**2),1)
            
            sine_min=ak.min(ak.concatenate([sine20[:,np.newaxis],sine21[:,np.newaxis]],axis=1),axis=1)
            mjjjl=ak.min(ak.concatenate([mjjjl1[:,np.newaxis],mjjjl2[:,np.newaxis]],axis=1),axis=1)
            mjjl=ak.min(ak.concatenate([mjjl1[:,np.newaxis],mjjl2[:,np.newaxis]],axis=1),axis=1)

            pt_min=jet3mag*sine_min
            pt_norm=pt_min/AK4Jets[cut][:, 0].pt
            magicparameter3=pt_min/mjjjl
            magicparameter=pt_min/mjjl
            pseudomagic3=pt_min/mjjj
            pseudomagic=pt_min/mjj
            
            """ count= ak.num(mlljj1, axis=0).compute()
            j1tb=ak.where(AK4Jets[cut][:, 0].partonFlavour==5,1,0)
            j2tb=ak.where(AK4Jets[cut][:, 1].partonFlavour==5,1,0)
            j3tb=ak.where(AK4Jets[cut][:, 2].partonFlavour==5,1,0)
            tbcount = ak.sum(ak.where((j1tb + j2tb + j3tb)>0,1,0)).compute()
            
            if region.endswith("SR"):
                if region.startswith("WR_EE"):
                    writestring.append(self._signal_sample)
                writestring.append(tbcount)
                writestring.append(count) """


            output['WRMass4_DeltaR'].fill(process=process,region=region,mass_fourobject=mlljj1,del_r=dr_j3_min,weight=weights.weight()[cut])
            output['WRMass5_DeltaR'].fill(process=process,region=region,mass_fiveobject=mlljjj,del_r=dr_j3_min,weight=weights.weight()[cut])
            output['WRMass4_pT'].fill(process=process,region=region,mass_fourobject=mlljj1,pTrel=pt_min,weight=weights.weight()[cut])
            output['WRMass5_pT'].fill(process=process,region=region,mass_fiveobject=mlljjj,pTrel=pt_min,weight=weights.weight()[cut])
            output['WRMass4_sin'].fill(process=process,region=region,mass_fourobject=mlljj1,sin=sine_min,weight=weights.weight()[cut])
            output['WRMass5_sin'].fill(process=process,region=region,mass_fiveobject=mlljjj,sin=sine_min,weight=weights.weight()[cut])
            output['WRMass4_pTnorm'].fill(process=process,region=region,mass_fourobject=mlljj1,pTnorm=pt_norm,weight=weights.weight()[cut])
            output['WRMass5_pTnorm'].fill(process=process,region=region,mass_fiveobject=mlljjj,pTnorm=pt_norm,weight=weights.weight()[cut])
            output['WRMass4_magic3'].fill(process=process,region=region,mass_fourobject=mlljj1,magic=magicparameter3,weight=weights.weight()[cut])
            output['WRMass5_magic3'].fill(process=process,region=region,mass_fiveobject=mlljjj,magic=magicparameter3,weight=weights.weight()[cut])
            output['WRMass4_magic'].fill(process=process,region=region,mass_fourobject=mlljj1,magic=magicparameter,weight=weights.weight()[cut])
            output['WRMass5_magic'].fill(process=process,region=region,mass_fiveobject=mlljjj,magic=magicparameter,weight=weights.weight()[cut])
            output['WRMass4_pseudomagic3'].fill(process=process,region=region,mass_fourobject=mlljj1,pseudomagic=pseudomagic3,weight=weights.weight()[cut])
            output['WRMass5_pseudomagic3'].fill(process=process,region=region,mass_fiveobject=mlljjj,pseudomagic=pseudomagic3,weight=weights.weight()[cut])
            output['WRMass4_pseudomagic'].fill(process=process,region=region,mass_fourobject=mlljj1,pseudomagic=pseudomagic,weight=weights.weight()[cut])
            output['WRMass5_pseudomagic'].fill(process=process,region=region,mass_fiveobject=mlljjj,pseudomagic=pseudomagic,weight=weights.weight()[cut])
            output['WRMass4_neutrino3'].fill(process=process,region=region,mass_fourobject=mlljj1,mass_neutrino=mjjjl,weight=weights.weight()[cut])
            output['WRMass5_neutrino3'].fill(process=process,region=region,mass_fiveobject=mlljjj,mass_neutrino=mjjjl,weight=weights.weight()[cut])
            output['WRMass4_neutrino'].fill(process=process,region=region,mass_fourobject=mlljj1,mass_neutrino=mjjl,weight=weights.weight()[cut])
            output['WRMass5_neutrino'].fill(process=process,region=region,mass_fiveobject=mlljjj,mass_neutrino=mjjl,weight=weights.weight()[cut])
            output['WRMass4_pseudo3'].fill(process=process,region=region,mass_fourobject=mlljj1,pseudomass=mjjj,weight=weights.weight()[cut])
            output['WRMass5_pseudo3'].fill(process=process,region=region,mass_fiveobject=mlljjj,pseudomass=mjjj,weight=weights.weight()[cut])
            output['WRMass4_pseudo'].fill(process=process,region=region,mass_fourobject=mlljj1,pseudomass=mjj,weight=weights.weight()[cut])
            output['WRMass5_pseudo'].fill(process=process,region=region,mass_fiveobject=mlljjj,pseudomass=mjj,weight=weights.weight()[cut])
        
            wrcand=(tightLeptons[cut][:, 0] + tightLeptons[cut][:, 1] + AK4Jets[cut][:, 0] + AK4Jets[cut][:, 1]).boostvec
            restdr_j3_min = ak.min(AK4Jets[cut][:,2].boost(-wrcand).delta_r(AK4Jets[cut][:,:2].boost(-wrcand)),axis=1)
            output['WRMass4_restDeltaR'].fill(process=process,region=region,mass_fourobject=mlljj1,del_r=restdr_j3_min,weight=weights.weight()[cut])
            output['WRMass5_restDeltaR'].fill(process=process,region=region,mass_fiveobject=mlljjj,del_r=restdr_j3_min,weight=weights.weight()[cut])

        output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
