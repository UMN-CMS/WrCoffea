import logging
import time
import warnings
import argparse
import uproot
import awkward as ak
import yaml
from typing import List, Dict
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from coffea.dataset_tools import apply_to_fileset, max_chunks, preprocess
from coffea.dataset_tools.preprocess import UprootFileSpec
from coffea.dataset_tools.preprocess import DatasetSpec
from coffea.dataset_tools.preprocess import _normalize_file_info

from analyzer.modules.objects import createObjects
from analyzer.modules.baseline import createSelection
import utils
warnings.filterwarnings("ignore", module="coffea.*")
NanoAODSchema.warn_missing_crossrefs = False  # silences warnings about branches we will not use here

class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self):
        self.eejj_60mll150 = utils.makeHistograms.eventHistos(['eejj', '60mll150'])
        self.mumujj_60mll150 = utils.makeHistograms.eventHistos(['mumujj', '60mll150'])
        self.emujj_60mll150 = utils.makeHistograms.eventHistos(['emujj', '60mll150'])

        self.eejj_150mll400 = utils.makeHistograms.eventHistos(['eejj','150mll400'])
        self.mumujj_150mll400 = utils.makeHistograms.eventHistos(['mumujj','150mll400'])
        self.emujj_150mll400 = utils.makeHistograms.eventHistos(['emujj','150mll400'])

        self.eejj_400mll = utils.makeHistograms.eventHistos(['eejj','400mll'])
        self.mumujj_400mll = utils.makeHistograms.eventHistos(['mumujj','400mll'])
        self.emujj_400mll = utils.makeHistograms.eventHistos(['emujj','400mll'])

        self.hists = {
            "eejj_60mll150": self.eejj_60mll150,
            "mumujj_60mll150": self.mumujj_60mll150,
            "emujj_60mll150": self.emujj_60mll150,
            "eejj_150mll400": self.eejj_150mll400,
            "mumujj_150mll400": self.mumujj_150mll400,
            "emujj_150mll400": self.emujj_150mll400,
            "eejj_400mll": self.eejj_400mll,
            "mumujj_400mll": self.mumujj_400mll,
            "emujj_400mll": self.emujj_400mll
        }

    def process(self, events): #Processes a single NanoEvents chunk

        events = createObjects(events)
        selections = createSelection(events)

        resolved_selections = selections.all('exactly2l', 'atleast2j', 'leadleppt60', "mlljj>800", "dr>0.4")

        for hist_name, hist_obj in self.hists.items():
            hist_obj.FillHists(events[resolved_selections & selections.all(*hist_obj.cuts)])

        return self.hists

    def postprocess(self, accumulator):
        return accumulator

print("\nStarting analyzer...\n")

with open('datasets/TTTo2L2Nu2018.yaml') as file:
    original_fileset = yaml.safe_load(file)

print(f"original fileset: {original_fileset}\n")

file_spec = UprootFileSpec(object_path="Events", steps = None)

print(file_spec.object_path)

test1 = _normalize_file_info(original_fileset)
print(test1)

filesets = {}
for sample_dict in original_fileset:
    sample_name = sample_dict['name'].split('.')[0]  # Extracting the sample name from the file name
    file_paths = [file for file in sample_dict['files']]
    filesets[sample_name] = file_paths


#test, _ = preprocess(filesets)
