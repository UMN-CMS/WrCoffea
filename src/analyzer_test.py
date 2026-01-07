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
        self.make_output = lambda: {}
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

            # Base unit weights (no genWeight)
            event_weight = events.genWeight
            weights.add("event_weight", event_weight)

            # Lumi uncertainty (weight-only)
            if mc_campaign not in LUMI_UNC:
                raise KeyError(f"No luminosity uncertainty defined for era '{mc_campaign}'. Add it to LUMI_UNC.")
        
        else:  # is_data
            weights.add("data", np.ones(n, dtype=np.float32))
            syst_weights = { 
                "Nominal":  weights.weight(),
            }

        return weights

    def process(self, events):
        output = self.make_output()
        metadata = events.metadata

        mc_campaign = metadata.get("era", "")
        process_name = metadata.get("physics_group", "")
        dataset = metadata.get("sample", "")

        is_mc   = hasattr(events, "genWeight")
        is_data = not is_mc

        # Compute genEventSumw for MC
        if is_mc:
#            gen_event_sumw = ak.sum(events.genWeight)
            gen_event_sumw = ak.sum(ak.values_astype(events.genWeight, "float64"), axis=0)
            output["genEventSumw"] = gen_event_sumw
        else:
            output["genEventSumw"] = None

        # Apply lumi mask via helper

#        weights = self.build_event_weights(events, metadata, is_mc, is_data, mc_campaign, process_name)

        nested_output = {dataset: {**output}}

        return nested_output

    def postprocess(self, accumulator):
        return accumulator
