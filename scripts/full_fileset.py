#!/usr/bin/env python3
#
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Example usage:
#   # Takes in config JSON, uses rucio to query DAS and populate with ROOT filepaths:
#    python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_lo_dy.json --dataset TTbar
#    python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json --dataset Muon
#    python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json --dataset Signal

# This will produce:
#   data/jsons/Run3/2022/Run3Summer22/unskimmed/Run3Summer22_TTbar_fileset.json
#
# Options for --dataset: DYJets, TTbar, TW, WJets, TTbarSemileptonic, SingleTop, Diboson, Triboson, TTV, EGamma (data), Muon (data), Signal (signal)
# ----------------------------------------------------------------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

import json
import argparse
import logging
from pathlib import Path
import os
import sys

from coffea.dataset_tools.dataset_query import DataDiscoveryCLI

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, repo_root)

from python.preprocess_utils import load_json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def filter_json_by_primary_ds_name(json_data, primary_ds_name):
    return {
        key: value
        for key, value in json_data.items()
        if value.get("physics_group") == primary_ds_name
    }

def query_datasets(data, sample):
    strat = "first" if sample == "Signal" else "choose"

    logging.info("Querying replica sites")
    ddc = DataDiscoveryCLI()
    ddc.do_blocklist_sites(["T2_US_MIT"])
    return ddc.load_dataset_definition(
        dataset_definition=data,
        query_results_strategy="all",
        replicas_strategy=strat
    )

def rename_dataset_key_to_sample(data: dict) -> dict:
    for entry in data.values():
        md = entry.get("metadata", {})
        if "dataset" in md:
            md["sample"] = md.pop("dataset")
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Replace file lists in a JSON config and run Coffea preprocessing."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the input JSON configuration (e.g. data/configs/.../era_sample.json)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Primary dataset name to filter the config (e.g. TTTo2L2Nu)"
    )
    args = parser.parse_args()

    input_path = args.config
    if not input_path.is_file():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # derive run/year/era from the input path under data/configs
    cfg_dir = input_path.parent
    cur = cfg_dir
    while cur.name != "configs":
        cur = cur.parent
    data_root = cur.parent  # data/
    rel = cfg_dir.relative_to(data_root / "configs")
    run, year, era = rel.parts

    # derive sample from filename: should be era_sample.json
    stem = input_path.stem  # e.g. "Run3Summer22_mc"
    prefix = f"{era}_"
    if not stem.startswith(prefix):
        logging.warning(f"Filename '{stem}' doesn't start with '{prefix}'")
    sample = stem[len(prefix):]

    # load, replace file lists, preprocess, and save
    config_file = load_json(str(input_path))

    config = filter_json_by_primary_ds_name(config_file, args.dataset)
    
    dataset = query_datasets(config, args.dataset)

    fileset = rename_dataset_key_to_sample(dataset)

    # construct & create output directory
    out_dir = data_root / "jsons" / run / year / era / "unskimmed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save the final JSON
    out_file = out_dir / f"{era}_{args.dataset}_fileset.json"
    out_path = Path(out_file)               # out_file can already be a Path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        json.dump(fileset, f, indent=4, ensure_ascii=False, default=str)
        logging.info(f"Saved JSON to {out_file}")

if __name__ == "__main__":
    main()
