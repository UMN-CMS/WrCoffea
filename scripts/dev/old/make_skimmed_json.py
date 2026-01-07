#!/usr/bin/env python3
#
# -----------------------------------------------------------------------------
# Example usage:
#   # Replace files & preprocess skims for a given config JSON:
#    python3 scripts/make_skimmed_json.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_lo_dy.json
#       
#
#
# This will produce:
#   data/jsons/Run3/2022/Run3Summer22/skimmed/Run3Summer22_mc_preprocessed_skims.json
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

import json
import argparse
import subprocess
import logging
from coffea.dataset_tools import preprocess
from dask.diagnostics import ProgressBar
from pathlib import Path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, repo_root)

from python.preprocess_utils import get_era_details, load_json, save_json

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def replace_files_in_json(data: dict, run: str, year: str, era: str, umn: bool, sample: str) -> dict:
    metadata_keys = (["das_name", "run", "year", "era", "dataset", "physics_group", "datatype"]
                     if sample == "data"
                     else ["das_name", "run", "year", "era", "dataset", "physics_group", "xsec", "datatype", "nevts"])

    for key, entry in data.items():
        metadata = {k: entry.pop(k) for k in metadata_keys if k in entry}
        files = entry.pop("files", {})
        data[key] = {"files": files, "metadata": metadata}

    for ds_name, ds_info in data.items():
        dataset = ds_info["metadata"].get("dataset")
        if not dataset:
            logging.warning(f"Dataset not found in metadata for {ds_name}")
            continue

        root_files = (get_root_files_from_umn(dataset, era) if umn
                      else get_root_files_from_eos(dataset, run, year, era))

        if root_files:
            for fp in root_files:
                ds_info["files"][fp] = "Events"
        else:
            logging.warning(f"No ROOT files found for dataset {ds_name}")
    return data


def get_root_files_from_umn(dataset: str, mc_campaign: str) -> list[str]:
    # mc_campaign eg. "Run3Summer22"
    run, year, era = get_era_details(mc_campaign)
    base = Path(f"/local/cms/user/jack1851/skims/2025/{run}/{year}/{mc_campaign}/{dataset}/")
    files = []
    if base.exists():
        for p in base.rglob("*.root"):
            files.append(str(p))
        logging.info(f"Found {len(files)} ROOT files for {dataset} in UMN skims")
    else:
        logging.error(f"UMN base path '{base}' not found")
    return files


def get_root_files_from_eos(dataset: str, run: str, year: str, era: str) -> list[str]:
    base_path = f"/store/user/wijackso/WRAnalyzer/skims/2025/{run}/{year}/{era}/{dataset}/"
    cmd = ["xrdfs", "root://cmseos.fnal.gov", "ls", base_path]
    try:
        out = subprocess.check_output(cmd, text=True)
        files = [
            f"root://cmseos.fnal.gov/{line.strip()}"
            for line in out.splitlines() if line.endswith(".root")
        ]
        logging.info(f"Found {len(files)} ROOT files for {dataset} on EOS")
        return files
    except subprocess.CalledProcessError as e:
        logging.error(f"EOS listing failed for {dataset}: {e}")
        return []


def preprocess_json(fileset: dict, chunks: int = 100_000, timeout: int = 3600):
    logging.info(f"Starting preprocessing ({chunks=} , timeout={timeout}s)")
    with ProgressBar():
        runnable, updated = preprocess(
            fileset=fileset,
            step_size=chunks,
            skip_bad_files=True
        )
    logging.info("Preprocessing done.")
    return runnable, updated


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
        "--umn",
        action="store_true",
        help="Fetch ROOT files from UMN skims instead of EOS"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=100_000,
        help="Chunk size for Coffea preprocessing"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout (s) for file I/O operations"
    )
    args = parser.parse_args()

    input_path = args.config
    if not input_path.is_file():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # derive run/year/era from the input path under data/configs
    cfg_dir = input_path.parent  # e.g. data/configs/Run3/2022/Run3Summer22
    cur = cfg_dir
    while cur.name != "configs":
        cur = cur.parent
    data_root = cur.parent  # data/
    rel = cfg_dir.relative_to(data_root / "configs")  # Run3/2022/Run3Summer22
    run, year, era = rel.parts

    # derive sample from filename: should be era_sample.json
    stem = input_path.stem  # e.g. "Run3Summer22_mc"
    prefix = f"{era}_"
    if not stem.startswith(prefix):
        logging.warning(f"Filename '{stem}' doesn't start with '{prefix}'")
    sample = stem[len(prefix):]

    # load, replace file lists, preprocess, and save
    fileset = load_json(str(input_path))
    fileset = replace_files_in_json(fileset, run, year, era, args.umn, sample)
    runnable, updated = preprocess_json(fileset, chunks=args.chunks, timeout=args.timeout)

    # construct & create output directory
    out_dir = data_root / "jsons" / run / year / era / "skimmed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save the final JSON
    out_file = out_dir / f"{era}_{sample}_preprocessed_skims.json"
    save_json(str(out_file), runnable, updated)
    logging.info(f"Saved preprocessed skims JSON to {out_file}")


if __name__ == "__main__":
    main()
