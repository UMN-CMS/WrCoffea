import logging
import analyzer.utils as utils
import coffea.dataset_tools.preprocess as preprocess_tools
from coffea.dataset_tools.preprocess import DatasetSpec
from analyzer.datasets import SampleCollection, SampleSet
import coffea.dataset_tools as dataset_tools
from dataclasses import dataclass
from typing import Any, Iterable, List
from analyzer.datasets import AnalyzerInput

logger = logging.getLogger(__name__)

class DatasetPreprocessed:
    def __init__(self, dataset_input, coffea_dataset_split):
        self.dataset_input = dataset_input
        self.coffea_dataset_split = coffea_dataset_split

    def fromDatasetInput(dataset_input, **kwargs):
        out, x = dataset_tools.preprocess(dataset_input.coffea_dataset, save_form=False, **kwargs)
        return DatasetPreprocessed(dataset_input, out[dataset_input.dataset_name])

    def getCoffeaDataset(self):
        return self.coffea_dataset_split

