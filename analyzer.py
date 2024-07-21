from coffea import processor
import warnings
from coffea.analysis_tools import Weights
warnings.filterwarnings("ignore",module="coffea.*")
import numpy as np
import awkward as ak
import hist.dask as dah
import hist
import dask.array as da
import dask_awkward as dak
from coffea.analysis_tools import PackedSelection
from CollectObject import CollectObject
import time
import re

class WrAnalysis(processor.ProcessorABC):
    def __init__(self, year='2018', mass_point=None):
        self._year = year
        self._signal_sample = mass_point

        self._triggers = {
            '2018': [
                'HLT_Mu50',
                'HLT_OldMu100',
                'HLT_TkMu100',
                'HLT_Ele32_WPTight_Gsf',
                'HLT_Photon200',
            ],
        }

        self.make_output = lambda: {
            'pt_leadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(400, 0, 2000, name='pt_leadlep', label=r'p_{T} of the leading lepton [GeV]'),
                hist.storage.Weight(),
            ),
            'pt_subleadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(400, 0, 2000, name='pt_subleadlep', label=r'p_{T} of the subleading lepton [GeV]'),
                hist.storage.Weight(),
            ),
            'pt_leadjet': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(400, 0, 2000, name='pt_leadjet', label=r'p_{T} of the leading jet [GeV]'),
                hist.storage.Weight(),
            ),
            'pt_subleadjet': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(400, 0, 2000, name='pt_subleadjet', label=r'p_{T} of the subleading jet [GeV]'),
                hist.storage.Weight(),
            ),
            'pt_dileptons': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(200, 0, 1000, name='pt_dileptons', label=r'p^{T}_{ll} [GeV]'),
                hist.storage.Weight(),
            ),
            'pt_dijets': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(200, 0, 1000, name='pt_dijets', label=r'p^{T}_{jj} [GeV]'),
                hist.storage.Weight(),
            ),
            'eta_leadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(600, -3, 3, name='eta_leadlep', label=r'#eta of the leading lepton [GeV]'),
                hist.storage.Weight(),
            ),
            'eta_subleadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(600, -3, 3, name='eta_subleadlep', label=r'#eta of the subleading lepton [GeV]'),
                hist.storage.Weight(),
            ),
            'eta_leadjet': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(600, -3, 3, name='eta_leadjet', label=r'#eta of the leading jet [GeV]'),
                hist.storage.Weight(),
            ),
            'eta_subleadjet': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(600, -3, 3, name='eta_subleadjet', label=r'#eta of the subleading jet [GeV]'),
                hist.storage.Weight(),
            ),

            'phi_leadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(800, -4, 4, name='phi_leadlep', label=r'#phi of the leading lepton [GeV]'),
                hist.storage.Weight(),
            ),
            'phi_subleadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(800, -4, 4, name='phi_subleadlep', label=r'#phi of the subleading lepton [GeV]'),
                hist.storage.Weight(),
            ),
            'phi_leadjet': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(800, -4, 4, name='phi_leadjet', label=r'#phi of the leading jet [GeV]'),
                hist.storage.Weight(),
            ),
            'phi_subleadjet': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(800, -4, 4, name='phi_subleadjet', label=r'#phi of the subleading jet [GeV]'),
                hist.storage.Weight(),
            ),
            'mass_dileptons': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(1000, 0, 5000, name='mass_dileptons', label=r'm_{ll} [GeV]'),
                hist.storage.Weight(),
            ),
            'mass_dijets': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(1000, 0, 5000, name='mass_dijets', label=r'm_{jj} [GeV]'),
                hist.storage.Weight(),
            ),
            'mass_threeobject_leadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(1600, 0, 8000, name='mass_threeobject_leadlep', label=r'm_{ljj} [GeV]'),
                hist.storage.Weight(),
            ),
            'mass_threeobject_subleadlep': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(1600, 0, 8000, name='mass_threeobject_subleadlep', label=r'm_{ljj} [GeV]'),
                hist.storage.Weight(),
            ),
            'mass_fourobject': dah.hist.Hist(
                hist.axis.StrCategory([], name="process", label="Process", growth=True),
                hist.axis.StrCategory([], name="region", label="Analysis Region", growth=True),
                hist.axis.Regular(1600, 0, 8000, name='mass_fourobject', label=r'm_{lljj} [GeV]'),
                hist.storage.Weight(),
            ),
        }

    def process(self, events): 
        output = self.make_output()

        isRealData = not hasattr(events, "genWeight")

        output['mc_campaign'] = events.metadata["mc_campaign"]

        if not isRealData:
            output['x_sec'] = events.metadata["xsec"]

        process = events.metadata["process"]
        output['process'] = process

        dataset = events.metadata["dataset"]
        output['dataset'] = dataset

        isMC = hasattr(events, "genWeight")
        isRealData = not hasattr(events, "genWeight")

        if process == "Signal":
            print(f"Analyzing {dataset} events.")
        elif process in {"SingleMuon", "EGamma"}:
            print(f"Analyzing {len(events)} {process} {dataset} events.")
        else:
            print(f"Analyzing {len(events)} {dataset} events.")

        ####################
        # OBJECT SELECTION #
        ####################

        coll = CollectObject(events)
        tightElectrons = coll.Electron({
            "pt" : (">", 53),
            "eta" : ("<", 2.4),
        })
        looseElectrons = coll.Electron({
            "pt" : (">", 53),
            "eta" : ("<", 2.4),
        })
        nTightElectrons = ak.num(tightElectrons)

        tightMuons, looseMuons = selectMuons(events)
        nTightMuons = ak.num(tightMuons)

        AK4Jets, AK8Jets = selectJets(events)
        nAK4Jets = ak.num(AK4Jets)

        ###########
        # WEIGHTS #
        ###########

        weights = Weights(size=None, storeIndividual=True)
        if not isRealData:
            eventWeight = np.sign(events.genWeight)
        else:
            eventWeight = abs(np.sign(events.event))

        #Only fill histogram with event specific weights
        weights.add("event_weight", weight=eventWeight)

        if not isRealData:
            output['sumw'] = ak.sum(eventWeight)

        ###################
        # EVENT VARIABLES #
        ###################

        tightLeptons = ak.with_name(ak.concatenate((tightElectrons, tightMuons), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons = ak.pad_none(tightLeptons[ak.argsort(tightLeptons.pt, axis=1, ascending=False)], 2, axis=1)
        AK4Jets = ak.pad_none(AK4Jets, 2, axis=1)

        mll = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1]).mass, False)
        mlljj = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1] + AK4Jets[:, 0] + AK4Jets[:, 1]).mass, False)

        dr_jl_min = ak.fill_none(ak.min(AK4Jets[:,:2].nearest(tightLeptons).delta_r(AK4Jets[:,:2]), axis=1), False)
        dr_j1j2 = ak.fill_none(AK4Jets[:,0].delta_r(AK4Jets[:,1]), False)
        dr_l1l2 = ak.fill_none(tightLeptons[:,0].delta_r(tightLeptons[:,1]), False)

        ###################
        # EVENT SELECTION #
        ###################

          
        selections = PackedSelection()

        # Resolved Selections
        selections.add("twoTightLeptons", (nTightElectrons + nTightMuons) == 2)
        selections.add("minTwoAK4Jets", (nAK4Jets >= 2))
        selections.add("leadTightLeptonPt60", ((ak.any(tightElectrons.pt > 60, axis=1)) | (ak.any(tightMuons.pt > 60, axis=1))))
        selections.add("mlljj>800", (mlljj > 800))
        selections.add("dr>0.4", (dr_jl_min > 0.4) & (dr_j1j2 > 0.4) & (dr_l1l2 > 0.4))

        # Trigger Selections
        eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
        muTrig = events.HLT.Mu50 | events.HLT.OldMu100 | events.HLT.TkMu100

        selections.add("eeTrigger", (eTrig & (nTightElectrons == 2) & (nTightMuons == 0)))
        selections.add("mumuTrigger", (muTrig & (nTightElectrons == 0) & (nTightMuons == 2)))
        selections.add("emuTrigger", (eTrig & muTrig & (nTightElectrons == 1) & (nTightMuons == 1)))

        # Flavor Selections
        selections.add("eejj", ((nTightElectrons == 2) & (nTightMuons == 0)))
        selections.add("mumujj", ((nTightElectrons == 0) & (nTightMuons == 2)))
        selections.add("emujj", ((nTightElectrons == 1) & (nTightMuons == 1)))

        # mll Selections
        selections.add("60mll150", ((mll > 60) & (mll < 150)))
        selections.add("150mll400", ((mll > 150) & (mll < 400)))
        selections.add("400mll", (mll > 400))

        regions = {
            'eejj_60mll150': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '60mll150', 'eejj'],
            'mumujj_60mll150': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '60mll150', 'mumujj'],
            'emujj_60mll150': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '60mll150', 'emujj'],
            'eejj_150mll400': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '150mll400', 'eejj'],
            'mumujj_150mll400': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '150mll400', 'mumujj'],
            'emujj_150mll400': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '150mll400', 'emujj'],
            'eejj_400mll': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '400mll', 'eejj'],
            'mumujj_400mll': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '400mll', 'mumujj'],
            'emujj_400mll': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'mlljj>800', 'dr>0.4', '400mll', 'emujj'],
        }

        #######################
        # BLIND SIGNAL REGION #
        #######################

        if isRealData:
            for region in ['eejj_400mll', 'mumujj_400mll']:
                if region in regions:
                    del regions[region]

        ##################
        # SIGNAL SAMPLES #
        ##################

        if process == "Signal":
            # Check if the specified mass point is resolved.
            match = re.search(r'MWR(\d+)_MN(\d+)', self._signal_sample)
            if match:
                mwr = int(match.group(1))
                mn = int(match.group(2))
                ratio = mn / mwr
                if ratio < 0.2:
                    raise NotImplementedError(f"Choose a resolved sample (MN/MWR > 0.2). MN/MWR = {ratio:.2f} for this sample.")

            # Add the cut to the specified mass point.
            for mass_point in events.GenModel.fields:
                if self._signal_sample in mass_point:
                    selections.add(f"{self._signal_sample}", eval(f"events.GenModel.WRtoNLtoLLJJ_{self._signal_sample}_TuneCP5_13TeV_madgraph_pythia8==1"))
                    break

            for region in regions:
                if 'mlljj>800' in regions[region]:
                    regions[region].remove('mlljj>800')
                regions[region].append(self._signal_sample)

            mass_cut =  selections.all('twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'dr>0.4', f"{self._signal_sample}")
            output['mlljj'] = (tightLeptons[mass_cut][:, 0] + tightLeptons[mass_cut][:, 1] + AK4Jets[mass_cut][:, 0] + AK4Jets[mass_cut][:, 1]).mass
            output['mljj_leadlep'] = (tightLeptons[mass_cut][:, 0] + AK4Jets[mass_cut][:, 0] + AK4Jets[mass_cut][:, 1]).mass
            output['mljj_subleadlep'] = (tightLeptons[mass_cut][:, 1] + AK4Jets[mass_cut][:, 0] + AK4Jets[mass_cut][:, 1]).mass

            process = dataset 

        ###################
        # FILL HISTOGRAMS #
        ###################

        for region, cuts in regions.items():
            cut = selections.all(*cuts)
            output['pt_leadlep'].fill(
                process=process,
                region=region,
                pt_leadlep=tightLeptons[cut][:, 0].pt,
                weight=weights.weight()[cut],
            )
            output['pt_subleadlep'].fill(
                process=process,
                region=region,
                pt_subleadlep=tightLeptons[cut][:, 1].pt,
                weight=weights.weight()[cut],
            )
            output['pt_leadjet'].fill(
                process=process,
                region=region,
                pt_leadjet=AK4Jets[cut][:, 0].pt,
                weight=weights.weight()[cut],
            )
            output['pt_subleadjet'].fill(
                process=process,
                region=region,
                pt_subleadjet=AK4Jets[cut][:, 1].pt,
                weight=weights.weight()[cut],
            )
            output['pt_dileptons'].fill(
                process=process,
                region=region,
                pt_dileptons=(tightLeptons[cut][:, 0]+tightLeptons[cut][:, 1]).pt,
                weight=weights.weight()[cut],
            )
            output['pt_dijets'].fill(
                process=process,
                region=region,
                pt_dijets=(AK4Jets[cut][:, 0]+AK4Jets[cut][:, 1]).pt,
                weight=weights.weight()[cut],
            )
            output['eta_leadlep'].fill(
                process=process,
                region=region,
                eta_leadlep=tightLeptons[cut][:, 0].eta,
                weight=weights.weight()[cut],
            )
            output['eta_subleadlep'].fill(
                process=process,
                region=region,
                eta_subleadlep=tightLeptons[cut][:, 1].eta,
                weight=weights.weight()[cut],
            )
            output['eta_leadjet'].fill(
                process=process,
                region=region,
                eta_leadjet=AK4Jets[cut][:, 0].eta,
                weight=weights.weight()[cut],
            )
            output['eta_subleadjet'].fill(
                process=process,
                region=region,
                eta_subleadjet=AK4Jets[cut][:, 1].eta,
                weight=weights.weight()[cut],
            )
            output['phi_leadlep'].fill(
                process=process,
                region=region,
                phi_leadlep=tightLeptons[cut][:, 0].phi,
                weight=weights.weight()[cut],
            )
            output['phi_subleadlep'].fill(
                process=process,
                region=region,
                phi_subleadlep=tightLeptons[cut][:, 1].phi,
                weight=weights.weight()[cut],
            )
            output['phi_leadjet'].fill(
                process=process,
                region=region,
                phi_leadjet=AK4Jets[cut][:, 0].phi,
                weight=weights.weight()[cut],
            )
            output['phi_subleadjet'].fill(
                process=process,
                region=region,
                phi_subleadjet=AK4Jets[cut][:, 1].phi,
                weight=weights.weight()[cut],
            )
            output['mass_dileptons'].fill(
                process=process,
                region=region,
                mass_dileptons=(tightLeptons[cut][:, 0]+tightLeptons[cut][:, 1]).mass,
                weight=weights.weight()[cut],
            )
            output['mass_dijets'].fill(
                process=process,
                region=region,
                mass_dijets=(AK4Jets[cut][:, 0]+AK4Jets[cut][:, 1]).mass,
                weight=weights.weight()[cut],
            )
            output['mass_threeobject_leadlep'].fill(
                process=process,
                region=region,
                mass_threeobject_leadlep=(tightLeptons[cut][:, 0]+AK4Jets[cut][:, 0]+AK4Jets[cut][:, 1]).mass,
                weight=weights.weight()[cut],
            )
            output['mass_threeobject_subleadlep'].fill(
                process=process,
                region=region,
                mass_threeobject_subleadlep=(tightLeptons[cut][:, 1]+AK4Jets[cut][:, 0]+AK4Jets[cut][:, 1]).mass,
                weight=weights.weight()[cut],
            )
            output['mass_fourobject'].fill(
                process=process,
                region=region,
                mass_fourobject=(tightLeptons[cut][:, 0]+tightLeptons[cut][:, 1]+AK4Jets[cut][:, 0]+AK4Jets[cut][:, 1]).mass,
                weight=weights.weight()[cut],
            )

        output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator

def selectElectrons(events):
    # select tight electrons
    electronSelectTight = (
            (events.Electron.pt > 53)
            & (np.abs(events.Electron.eta) < 2.4)
            & (events.Electron.cutBased_HEEP)
    )

    # select loose electrons
    electronSelectLoose = (
            (events.Electron.pt > 53)
            & (np.abs(events.Electron.eta) < 2.4)
            & (events.Electron.cutBased == 2)

    )
    return events.Electron[electronSelectTight], events.Electron[electronSelectLoose]

def selectMuons(events):
    # select tight muons
    muonSelectTight = (
            (events.Muon.pt > 53)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.highPtId == 2)
            & (events.Muon.tkRelIso < 0.1)
    )

    # select loose muons
    muonSelectLoose = (
            (events.Muon.pt > 53) 
            & (np.abs(events.Muon.eta) < 2.4) 
            & (events.Muon.highPtId == 2)
    )

    return events.Muon[muonSelectTight], events.Muon[muonSelectLoose] 

def selectJets(events):
    # select AK4 jets
    hem_issue = (-3.0 < np.abs(events.Jet.eta) < -1.3) & (-1.57 < np.abs(events.Jet.phi) < -0.87)

    jetSelectAK4 = (
            (events.Jet.pt > 40)
             & (np.abs(events.Jet.eta) < 2.4)
            & (events.Jet.isTightLeptonVeto)
            & hem_issue
    )

    # select AK8 jets (need to add LSF cut)
    jetSelectAK8 = (
            (events.FatJet.pt > 200)
            & (np.abs(events.FatJet.eta) < 2.4)
            & (events.FatJet.jetId == 2)
            & (events.FatJet.msoftdrop > 40)
            & hem_issue
    )

    return events.Jet[jetSelectAK4], events.FatJet[jetSelectAK8]
