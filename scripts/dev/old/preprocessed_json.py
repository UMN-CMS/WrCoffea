#!/usr/bin/env python3
#
# -----------------------------------------------------------------------------
# Example usage:
#   # Process the TTbar dataset in the Run3Summer22 config:
#   python3 scripts/preprocessed_json.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_lo_dy.json --dataset TTbar
#
# This will produce:
#   data/jsons/Run3/2022/Run3Summer22/Run3Summer22_TTbar_preprocessed.json
#   data/filepaths/Run3/2022/Run3Summer22/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.txt
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

import json
import sys
import argparse
import logging
import subprocess
import re
import uproot
import numpy as np
import awkward as ak
import multiprocessing
from pathlib import Path
import os

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.dataset_tools import rucio_utils, preprocess, max_files, max_chunks
from coffea.dataset_tools.dataset_query import print_dataset_query, DataDiscoveryCLI
from rich.console import Console
from rich.table import Table
from dask.diagnostics import ProgressBar
from dask.distributed import Client, progress

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, repo_root)

from python.preprocess_utils import load_json, save_json

NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.error_missing_event_ids = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def filter_json_by_primary_ds_name(json_data, primary_ds_name):
    return {
        key: value
        for key, value in json_data.items()
        if value.get("physics_group") == primary_ds_name
    }


def query_datasets(data):
    print(f"\nQuerying replica sites")
    logging.info("Querying replica sites")
    ddc = DataDiscoveryCLI()
    ddc.do_blocklist_sites(["T2_US_MIT"])
    return ddc.load_dataset_definition(
        dataset_definition=data,
        query_results_strategy="all",
        replicas_strategy="choose"
    )


def preprocess_json(fileset):
    chunks = 100_000
    logging.info("Preprocessing files")
    with ProgressBar():
        runnable, updated = preprocess(
            fileset=max_files(fileset, 2500),
            step_size=chunks,
            skip_bad_files=False,
            uproot_options={
                "handler": uproot.MultithreadedXRootDSource,
                "timeout": 3600
            }
        )
    logging.info("Preprocessing completed.")
    return runnable, updated


def save_dataset_txt(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, ds_info in dataset.items():
        metadata = ds_info.get("metadata", {})
        dataset_filename = metadata.get("dataset", ds_name)
        sanitized = dataset_filename.replace(" ", "_")
        txt_path = output_dir / f"{sanitized}.txt"

        with open(txt_path, 'w') as f:
            for file_path in ds_info.get("files", {}):
                f.write(f"{file_path}\n")

        logging.info(f"Saved dataset file list to {txt_path}")


def main():
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Process a NanoAOD JSON configuration through coffea + Dask."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the input JSON configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Primary dataset name to filter the config (e.g. TTTo2L2Nu)"
    )
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_file():
        logging.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logging.info(f"Loading input file {config_path}")
    config = load_json(str(config_path))
    if config is None:
        logging.error("No valid JSON found in config file.")
        sys.exit(1)

    # filter to only the specified primary dataset
    config = filter_json_by_primary_ds_name(config, args.dataset)

    client = Client(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
        memory_limit='2GB',
        nanny=False
    )
    logging.info("Dask client started.")

    dataset = query_datasets(config)

    # locate 'data' root and compute the relative sub-dir under 'configs'
    cfg_dir = config_path.parent  # e.g. data/configs/Run3/2022/Run3Summer22
    cur = cfg_dir
    while cur.name != "configs":
        cur = cur.parent
    data_root = cur.parent
    rel_sub = cfg_dir.relative_to(data_root / "configs")

    # write .txt lists
    fp_dir = data_root / "filepaths" / rel_sub
    save_dataset_txt(dataset, fp_dir)

    # preprocess JSON
    runnable, updated = preprocess_json(dataset)

    # write preprocessed JSON
    json_dir = data_root / "jsons" / rel_sub
    json_dir.mkdir(parents=True, exist_ok=True)
    base = cfg_dir.name
    out_json = json_dir / f"{base}_{args.dataset}_preprocessed.json"
    save_json(out_json, runnable, updated)


if __name__ == "__main__":
    main()
