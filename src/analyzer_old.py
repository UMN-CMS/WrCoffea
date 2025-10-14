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

def make_lumi_updown(delta):
    """
    Returns a function compatible with events.add_systematic(..., 'weight', func)
    It will be called with a length-N array (e.g. ones) and must return shape (N, 2):
    [:,0] = Up, [:,1] = Down multiplicative factors.
    """
    def _lumi_ud(ones):
        # ones: shape (N,)
        # return: shape (N, 2) with [Up, Down]
        updown = np.array([1.0 + delta, 1.0 - delta], dtype=np.float32)
        return ones[:, None] * updown[None, :]
    return _lumi_ud

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
        ak8_jets = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4) & (events.FatJet.jetId == 2) & (events.FatJet.msoftdrop > 40) #& (events.FatJet.lsf3 > 0.75)
        return events.Jet[ak4_jets], events.FatJet[ak8_jets]

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

    def add_resolved_selections(self, selections, tightElectrons, tightMuons, ak4Jets, mlljj, dr_jl_min, dr_j1j2, dr_l1l2):
        selections.add("twoTightLeptons", (ak.num(tightElectrons) + ak.num(tightMuons)) == 2)
        selections.add("minTwoAK4Jets", ak.num(ak4Jets) >= 2)
        selections.add("leadTightLeptonPt60", (ak.any(tightElectrons.pt > 60, axis=1) | ak.any(tightMuons.pt > 60, axis=1)))
        selections.add("mlljj>800", mlljj > 800)
        selections.add("dr>0.4", (dr_jl_min > 0.4) & (dr_j1j2 > 0.4) & (dr_l1l2 > 0.4))

    def add_boosted_selections(self, selections, ak8Jets):
        # See L388-389 in AN-19-083_temp.pdf
        resolved_mask = selections.all("twoTightLeptons","minTwoAK4Jets", "leadTightLeptonPt60","mlljj>800","dr>0.4")
        leadpt_mask = selections.all("leadTightLeptonPt60")
        boosted_baseline = ~resolved_mask & leadpt_mask
        selections.add("boostedBaseline", boosted_baseline)
        selections.add("minOneAK8Jet", ak.num(ak8Jets) >= 1)

    def fill_resolved_histograms(self, output, region, cut, process_name, jets, leptons, weights, syst_weights):
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
                    syst=syst_label,
                    **{axis_name: vals},
                    weight=w * syst_w
                )

    def fill_boosted_histograms(self, output, region, cut, process_name, jets, weights, syst_weights):
        variables = [('pt_leading_jet', jets[:, 0].pt, 'pt_leadjet')]

        for hist_name, values, axis_name in variables:
            vals = values[cut]
            w = weights.weight()[cut]

            # Fill once per systematic label
            for syst_label, syst_w_full in syst_weights.items():
                syst_w = syst_w_full[cut]
                output[hist_name].fill(
                    process=process_name,
                    region=region,
                    syst=syst_label,
                    **{axis_name: vals},
                    weight=w * syst_w
                )

    def process(self, events):
        output = self.make_output()
        metadata = events.metadata

        mc_campaign = metadata.get("era", "")
        process_name = metadata.get("physics_group", "")
        dataset = metadata.get("sample", "")

        is_mc = hasattr(events, "genWeight")
        is_data = not hasattr(events, "genWeight")

        if process_name == "Signal":
            self.check_mass_point_resolved()

        if is_data:
            json_path = os.environ.get("LUMI_JSON", LUMI_JSONS.get(mc_campaign))
            if json_path:
                lumi_mask = LumiMask(json_path)
                events = events[lumi_mask(events.run, events.luminosityBlock)]

        # Object selection
        tightElectrons, looseElectrons = self.selectElectrons(events)
        nTightElectrons = ak.num(tightElectrons)

        tightMuons, looseMuons = self.selectMuons(events)
        nTightMuons = ak.num(tightMuons)

        AK4Jets, AK8Jets = self.selectJets(events)
        nAK4Jets = ak.num(AK4Jets)
        nAK8Jets = ak.num(AK8Jets)

        # Event variables
        tightLeptons = ak.with_name(ak.concatenate((tightElectrons, tightMuons), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons = ak.pad_none(tightLeptons[ak.argsort(tightLeptons.pt, axis=1, ascending=False)], 2, axis=1)

        AK4Jets = ak.pad_none(AK4Jets, 2, axis=1)

        AK8Jets = ak.pad_none(AK8Jets, 1, axis=1) # Needed to fix 'IndexError(\'cannot slice ListArray (of length 198209) with array(0): index out of range while attempting to get index 0'

        mlljj = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1] + AK4Jets[:, 0] + AK4Jets[:, 1]).mass, False)

        dr_jl_min = ak.fill_none(ak.min(AK4Jets[:, :2].nearest(tightLeptons).delta_r(AK4Jets[:, :2]), axis=1), False)
        dr_j1j2 = ak.fill_none(AK4Jets[:, 0].delta_r(AK4Jets[:, 1]), False)
        dr_l1l2 = ak.fill_none(tightLeptons[:, 0].delta_r(tightLeptons[:, 1]), False)

        # Event selections
        selections = PackedSelection()
        self.add_resolved_selections(selections, tightElectrons, tightMuons, AK4Jets, mlljj, dr_jl_min, dr_j1j2, dr_l1l2)
        self.add_boosted_selections(selections, AK8Jets)


        # Trigger selections
        eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
        if mc_campaign in ("RunIISummer20UL18", "Run2Autumn18"):
            muTrig = events.HLT.Mu50 | events.HLT.OldMu100 | events.HLT.TkMu100
        elif mc_campaign in ("Run3Summer22", "Run3Summer23BPix", "Run3Summer22EE", "Run3Summer23"):
            eTrig = events.HLT.Ele32_WPTight_Gsf | events.HLT.Photon200 | events.HLT.Ele115_CaloIdVT_GsfTrkIdT
            muTrig = events.HLT.Mu50 | events.HLT.HighPtTkMu100


        selections.add("eTrigger", eTrig)
        selections.add("muTrigger", muTrig)
        selections.add("emuTrigger", (eTrig | muTrig))

        selections.add("two_tight_electrons", ((ak.num(tightElectrons) == 2) & (ak.num(tightMuons) == 0)))
        selections.add("two_tight_muons", ((ak.num(tightElectrons) == 0) & (ak.num(tightMuons) == 2)))
        selections.add("one_tight_electron", ((ak.num(tightElectrons) == 1) & (ak.num(tightMuons) == 0)))
        selections.add("one_tight_muon", ((ak.num(tightElectrons) == 0) & (ak.num(tightMuons) == 1)))
        selections.add("emujj", ((ak.num(tightElectrons) == 1) & (ak.num(tightMuons) == 1)))

        # mll selections
        # If there are not two tight leptons in an event, then mll = 0 
        mll_resolved = ak.fill_none((tightLeptons[:, 0] + tightLeptons[:, 1]).mass, 0.0)
        selections.add("60<mll<150_resolved", ((60 < mll_resolved) & (mll_resolved < 150)))
        selections.add("mll>400_resolved", (mll_resolved > 400))
        selections.add("mll>200_resolved", (mll_resolved > 200))

        # If there are not two tight leptons in an event, then mll = 0 

                # Event variables
        looseLeptons = ak.with_name(ak.concatenate((looseElectrons, looseMuons), axis=1), 'PtEtaPhiMCandidate')

        looseLeptons = ak.pad_none(looseLeptons[ak.argsort(looseLeptons.pt, axis=1, ascending=False)], 1, axis=1)
#        print(looseLeptons.pt)
        mll_boosted = ak.fill_none((tightLeptons[:, 0] + looseLeptons[:, 0]).mass, 0.0)
#        selections.add("60<mll<150_boosted", ((60 < mll_boosted) & (mll_boosted < 150)))
        print("mll_boosted", mll_boosted)
        # Event Weights
        weights = Weights(len(events))

        if is_mc:
            # get the cross section and number of events from metadata
            xsec = metadata.get("xsec", 1.0)
            n_events = metadata.get("nevts", 1.0)

            # fill each histogram with weight of 1
            eventWeight = abs(np.sign(events.event))

            norm = xsec / n_events

            if mc_campaign == "RunIISummer20UL18" and process_name == "DYJets":
                eventWeight = eventWeight * 59.84 * 1000

            eventWeight = eventWeight * norm
            weights.add("event_weight", eventWeight)
        elif is_data:
            weights.add("data", np.ones(len(events), dtype=np.float32))


        # --- Systematics via events.add_systematic ---
        ones = np.ones(len(events), dtype=np.float32)
        if is_data:
            # Data: no MC theory/systematic weights
            ones = np.ones(len(events), dtype=np.float32)
            syst_weights = {"Nominal": ones}
        elif is_mc:
            if mc_campaign not in LUMI_UNC:
                raise KeyError(f"No luminosity uncertainty defined for era '{mc_campaign}'. Please add it to LUMI_UNC.")

            lumi_unc = LUMI_UNC[mc_campaign]

            # MC: add integrated luminosity uncertainty as an Up/Down weight systematic
            events.add_systematic("Lumi", "UpDownSystematic", "weight",make_lumi_updown(lumi_unc))

            # Grab the Up/Down factors produced by the systematic
            lumi_up   = events.systematics.Lumi.up.weight_Lumi
            lumi_down = events.systematics.Lumi.down.weight_Lumi

            syst_weights = {
                "Nominal":  ones,
                "LumiUp":   lumi_up,    # multiplicative factors, length-N
                "LumiDown": lumi_down,
            }


        # DEFINE RESOLVED REGIONS
        resolved_regions = {
            'wr_ee_resolved_dy_cr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'eTrigger', 'mlljj>800', 'dr>0.4', '60<mll<150_resolved', 'two_tight_electrons'],
            'wr_mumu_resolved_dy_cr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'muTrigger', 'mlljj>800', 'dr>0.4', '60<mll<150_resolved', 'two_tight_muons'],
            'wr_resolved_flavor_cr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'emuTrigger', 'mlljj>800', 'dr>0.4', 'mll>400_resolved', 'emujj'],
            'wr_ee_resolved_sr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'eTrigger', 'mlljj>800', 'dr>0.4', 'mll>400_resolved', 'two_tight_electrons'],
            'wr_mumu_resolved_sr': ['twoTightLeptons', 'minTwoAK4Jets', 'leadTightLeptonPt60', 'muTrigger', 'mlljj>800', 'dr>0.4', 'mll>400_resolved', 'two_tight_muons'],
        }

        # FILL RESOLVED HISTOGRAMS
        for region, cuts in resolved_regions.items():
            cut = selections.all(*cuts)
            self.fill_resolved_histograms(output, region, cut, process_name, AK4Jets, tightLeptons, weights, syst_weights)

        # DEFINE BOOSTED REGIONS
        boosted_regions = {
            'wr_boosted': ["boostedBaseline", "minOneAK8Jet"]
        }

        # FILL BOOSTED HISTOGRAMS
#        for region, cuts in boosted_regions.items():
#            cut = selections.all(*cuts)
#            self.fill_boosted_histograms(output, region, cut, process_name, AK8Jets, weights, syst_weights)

        cutflow_regions = {
            "wr_ee_resolved": {
                "cutflow_order": ["two_tight_electrons","eTrigger", "minTwoAK4Jets", "dr>0.4", "mll>200_resolved", "mlljj>800", "mll>400_resolved"],
            },
            "wr_mumu_resolved": {
                "cutflow_order": ["two_tight_muons", "muTrigger", "minTwoAK4Jets", "dr>0.4", "mll>200_resolved", "mlljj>800", "mll>400_resolved"],
            },
            "wr_e_boosted": {
                "cutflow_order": ["boostedBaseline", "one_tight_electron", "eTrigger"],
            },
            "wr_mu_boosted": {
                "cutflow_order": ["boostedBaseline", "one_tight_muon", "muTrigger"],
            },
        }

        for region, info in cutflow_regions.items():
            order = info["cutflow_order"]

            # Weighted cutflow
            cf = selections.cutflow(*order, weights=weights)

            res = cf.yieldhist(weighted=True)
            h_onecut, h_cum = res[0], res[1]
            output.setdefault("cutflow", {})
            output["cutflow"].setdefault(region, {})
            output["cutflow"][region]["onecut"] = h_onecut
            output["cutflow"][region]["cumulative"] = h_cum

            # Unweighted cutflow
            cf_unw = selections.cutflow(*order, weights=None)
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
