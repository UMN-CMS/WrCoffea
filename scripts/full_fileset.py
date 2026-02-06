#!/usr/bin/env python3
#
# -----------------------------------------------------------------------------------
# Build unskimmed filesets by querying DAS for file lists via dasgoclient
# and using the FNAL XRootD redirector for file access.
#
# Example usage:
#   python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_data.json
#   python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_signal.json --dataset Signal
#   python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_mc_dy_lo_inc.json
#
# Output goes to:
#   data/filesets/<Run>/<Year>/<Era>/unskimmed/<Era>_<sample_tag>_fileset.json
# -----------------------------------------------------------------------------------

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from wrcoffea.era_utils import load_json
from wrcoffea.fileset_utils import (
    output_dir,
    parse_config_path,
    sample_from_config_filename,
    write_fileset_json,
)

REDIRECTOR = "root://cmsxrootd.fnal.gov/"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def filter_json_by_primary_ds_name(json_data, primary_ds_name):
    return {
        key: value
        for key, value in json_data.items()
        if value.get("physics_group") == primary_ds_name
    }


def query_das_files(das_name: str) -> list[str]:
    """Query DAS for the logical file names of a dataset using dasgoclient."""

    cmd = ["dasgoclient", "-query", f"file dataset={das_name}"]
    logging.info("  dasgoclient: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logging.error("dasgoclient failed for %s: %s", das_name, result.stderr.strip())
        raise RuntimeError(f"dasgoclient query failed for {das_name}")
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not files:
        logging.warning("No files returned by DAS for %s", das_name)
    return files


def build_fileset(config: dict) -> dict:
    """Build a coffea-compatible fileset dict from config entries via DAS queries."""

    fileset = {}
    for das_name, entry in config.items():
        logging.info("Querying DAS for %s", das_name)
        lfns = query_das_files(das_name)
        logging.info("  Found %d files", len(lfns))

        files = {f"{REDIRECTOR}{lfn}": "Events" for lfn in lfns}

        metadata = {k: v for k, v in entry.items()}
        # Rename "dataset" -> "sample" to match the convention used at analysis time
        if "dataset" in metadata:
            metadata["sample"] = metadata.pop("dataset")

        fileset[das_name] = {
            "files": files,
            "metadata": metadata,
        }
    return fileset


def main():
    parser = argparse.ArgumentParser(
        description="Build unskimmed fileset JSONs by querying DAS for file lists."
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
        default=None,
        help="Optional: filter config to a single physics_group (e.g. Muon, Signal)"
    )
    args = parser.parse_args()

    input_path = args.config
    if not input_path.is_file():
        logging.error("Input file not found: %s", input_path)
        sys.exit(1)

    data_root, run, year, era = parse_config_path(input_path)

    config = load_json(str(input_path))

    if args.dataset:
        config = filter_json_by_primary_ds_name(config, args.dataset)
        if not config:
            logging.error(
                "No entries found with physics_group='%s' in %s",
                args.dataset, input_path,
            )
            sys.exit(1)

    logging.info("Processing %d dataset(s) from %s", len(config), input_path.name)

    fileset = build_fileset(config)

    sample_tag = sample_from_config_filename(input_path, era=era)
    out_dir_path = output_dir(data_root=data_root, run=run, year=year, era=era) / "unskimmed"
    out_file = out_dir_path / f"{era}_{sample_tag}_fileset.json"
    write_fileset_json(out_file, fileset)
    logging.info("Saved fileset JSON to %s", out_file)


if __name__ == "__main__":
    main()
