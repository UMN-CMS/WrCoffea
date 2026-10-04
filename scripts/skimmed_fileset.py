#!/usr/bin/env python3
from __future__ import annotations

#
# -----------------------------------------------------------------------------
# Example usage:
#   # Takes in config JSON, and populates it with a list of skimmed ROOT files from EOS:
#   python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_lo_dy.json
#   python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
#   python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json 
#
# These commands will produce:
#   data/jsons/Run3/2022/Run3Summer22/skimmed/Run3Summer22_mc_lo_dy_skimmed_fileset.json
#   data/jsons/Run3/2022/Run3Summer22/skimmed/Run3Summer22_data_skimmed_fileset.json
#   data/jsons/Run3/2022/Run3Summer22/skimmed/Run3Summer22_signal_skimmed_fileset.json
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")

import argparse
import json
import subprocess
import logging
from pathlib import Path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.insert(0, repo_root)

from wrcoffea.era_utils import load_json
from wrcoffea.das_utils import base_dir_for_era
from wrcoffea.fileset_utils import (
    output_dir,
    parse_config_path,
    rename_dataset_key_to_sample,
    sample_from_config_filename,
    write_fileset_json,
)

EOS_HOST = "cmseos.fnal.gov"
EOS_SKIMS_ROOT = "/store/user/spatra1/skims"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def deduplicate_by_dataset(fileset: dict) -> dict:
    """Collapse fileset entries that share the same dataset name.

    After skimming, multiple DAS path versions (e.g. v1, v2, v3, v4 of the
    same run period) are merged into one set of part files.  The fileset
    builder assigns those same files to every DAS path entry, which causes
    the analyzer to process identical files multiple times.

    This function keeps one entry per unique ``metadata.dataset`` value,
    merging file lists (union) in case they somehow differ.
    """
    seen: dict[str, str] = {}  # dataset -> first DAS path key
    deduped: dict[str, dict] = {}

    for das_path, entry in fileset.items():
        dataset = entry.get("metadata", {}).get("dataset", das_path)

        if dataset not in seen:
            seen[dataset] = das_path
            deduped[das_path] = entry
        else:
            # Merge any files from this duplicate into the first entry
            first_key = seen[dataset]
            for fp, tree in entry.get("files", {}).items():
                deduped[first_key]["files"].setdefault(fp, tree)

    n_removed = len(fileset) - len(deduped)
    if n_removed:
        logging.info(
            "Deduplicated fileset: %d entries -> %d (removed %d duplicates)",
            len(fileset), len(deduped), n_removed,
        )

    return deduped


def _merged_root_files(files: list[str]) -> list[str]:
    """Keep merged part files and exclude raw per-job skim ROOT files."""
    return sorted(fp for fp in files if Path(fp).stem.rsplit("_part", 1)[-1].isdigit())


def _local_merge_status(dataset: str, mc_campaign: str) -> dict | None:
    """Read a local merge summary when building from UMN storage."""
    summary_path = (
        base_dir_for_era(mc_campaign, scratch=True)
        / "files"
        / dataset
        / "merge_summary.json"
    )
    if not summary_path.exists():
        return None
    try:
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("Could not read merge summary %s: %s", summary_path, e)
        return None
    return summary if isinstance(summary, dict) else None


def replace_files_in_json(
    data: dict, run: str, year: str, era: str, umn: bool, sample: str,
    allow_incomplete: bool = False,
) -> dict:
    metadata_keys = (["das_name", "run", "year", "era", "dataset", "physics_group", "datatype"]
                     if sample == "data"
                     else ["das_name", "run", "year", "era", "dataset", "physics_group", "xsec", "total_uncertainty", "genEventSumw" ,"datatype", "nevts"])

    for key, entry in data.items():
        metadata = {k: entry.pop(k) for k in metadata_keys if k in entry}
        if "dataset" in entry:
            metadata["sample"] = entry.pop("dataset")

        raw_files = entry.pop("files", {})

        if isinstance(raw_files, dict):
            files_dict = raw_files.copy()
        elif isinstance(raw_files, list):
            files_dict = {fp: "Events" for fp in raw_files}
        else:
            files_dict = {}

        data[key] = {"files": files_dict, "metadata": metadata}

    for ds_name, ds_info in data.items():
        dataset = ds_info["metadata"].get("dataset")
        if not dataset:
            logging.warning(f"Dataset not found in metadata for {ds_name}")
            continue

        if umn:
            summary = _local_merge_status(dataset, era)
            if summary and not summary.get("complete", True) and not allow_incomplete:
                raise RuntimeError(
                    f"Dataset {dataset} has an incomplete merge; rerun with "
                    "--allow-incomplete to use its available merged files"
                )
            root_files = get_root_files_from_umn(dataset, era)
        else:
            root_files = get_root_files_from_eos(dataset, run, year, era)

        root_files = _merged_root_files(root_files)
        if root_files:
            for fp in root_files:
                ds_info["files"].setdefault(fp, "Events")
            if umn and summary and not summary.get("complete", True):
                ds_info["metadata"]["skim_complete"] = False
                ds_info["metadata"]["skim_skipped_files"] = summary.get(
                    "skipped_files", []
                )
        else:
            logging.warning(f"No ROOT files found for dataset {ds_name}")

    return data


def get_root_files_from_umn(dataset: str, era: str) -> list[str]:
    """List merged ROOT files from the user's 3DayLifetime scratch area."""
    base = base_dir_for_era(era, scratch=True) / "files" / dataset
    files = []
    if base.exists():
        for p in base.rglob("*.root"):
            files.append(str(p))
        logging.info(f"Found {len(files)} ROOT files for {dataset} in scratch")
    else:
        logging.error(f"Scratch base path '{base}' not found")
    return files


def get_root_files_from_eos(
    dataset: str, run: str, year: str, era: str
) -> list[str]:
    """List merged ROOT files from the FNAL EOS skim area via XRootD."""
    if "WR" in dataset:
        category = "signals"
    elif "EGamma" in dataset or "Muon" in dataset:
        category = "files"
    else:
        category = "backgrounds"

    base = (
        f"{EOS_SKIMS_ROOT}/{run}/{year}/{era}/{category}/{dataset}"
    )
    files = xrdfs_list_root_files(EOS_HOST, base)
    logging.info(f"Found {len(files)} ROOT files for {dataset} on EOS")
    return files

def xrdfs_list_root_files(host: str, base_path: str) -> list[str]:
    """
    List .root files under base_path on an xrootd host.

    This is intentionally non-recursive (plus at most one level of descent)
    because recursive `xrdfs ls -R` over user areas can be extremely slow
    and spam `[!] Some of the requests failed` when any sub-paths are flaky.

    Returns fully qualified root:// URLs.
    """
    def _run(cmd):
        # Silence stderr to avoid noisy xrootd warnings unless we actually error.
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()

    root_urls: list[str] = []

    # Top-level listing: /store/.../<era>/<dataset>/
    try:
        top = _run(["xrdfs", host, "ls", base_path])
    except subprocess.CalledProcessError as e:
        logging.error(f"xrootd listing failed for {host}:{base_path}: {e}")
        return []

    subdirs: list[str] = []
    for line in top:
        line = line.strip()
        if not line:
            continue

        if line.endswith(".root"):
            # Direct ROOT files under base_path
            root_urls.append(f"root://{host}/{line}")
        else:
            # Treat everything else as a possible directory
            subdirs.append(line)

    # One-level descent: /store/.../<era>/<dataset>/<subdir>/*.root
    for d in subdirs:
        try:
            children = _run(["xrdfs", host, "ls", d])
        except subprocess.CalledProcessError:
            # If a subdir is flaky / gone, just skip it
            logging.warning(f"xrootd listing failed for subdir {host}:{d}")
            continue

        for c in children:
            c = c.strip()
            if c.endswith(".root"):
                root_urls.append(f"root://{host}/{c}")

    return root_urls

def get_root_files_from_wisc(dataset: str, run: str, year: str, era: str) -> list[str]:
    """
    Wisconsin storage for 2024, using gfal-ls over davs for *listing*,
    but still returning root:// URLs for Coffea / uproot.

    Example listing command:
      gfal-ls -l davs://cmsxrootd.hep.wisc.edu:1094/store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24/TTLL_Bin-MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8
    """
    host = "cmsxrootd.hep.wisc.edu"

    # POSIX-style path under /store
    if "WR" in dataset:
        rel_dir = f"/store/user/wijackso/WRAnalyzer/skims/{run}/{year}/{era}/signals/{dataset}/"
    elif "EGamma" in dataset or "Muon" in dataset:
        rel_dir = f"/store/user/wijackso/WRAnalyzer/skims/{run}/{year}/{era}/data/{dataset}/"
    else:
        rel_dir = f"/store/user/wijackso/WRAnalyzer/skims/{run}/{year}/{era}/backgrounds/{dataset}/"

    # davs URL for gfal
    davs_url = f"davs://{host}:1094{rel_dir}"

    cmd = ["gfal-ls", "-l", davs_url]

    try:
        out = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"gfal-ls failed for {davs_url}: {e}")
        return []

    files: list[str] = []

    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip summary lines like "total 12345" if present
        if line.lower().startswith("total"):
            continue

        parts = line.split()
        # gfal-ls -l output is usually like: "-rw-r--r-- ... filename.root"
        # so the last column is the filename or full path
        name = parts[-1] if parts else ""
        if not name.endswith(".root"):
            continue

        # If gfal prints just the filename, prepend rel_dir.
        # If it prints a full /store/... path, keep that.
        if name.startswith("/"):
            rel_path = name
        else:
            rel_path = rel_dir + name

        # Return root:// URLs for analysis
        files.append(f"root://{host}/{rel_path}")

    logging.info(f"Found {len(files)} ROOT files for {dataset} on WISC via gfal-ls")
    return files

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
        "--allow-incomplete",
        action="store_true",
        help="Use merged files from an incomplete merge (records skipped inputs)",
    )
    args = parser.parse_args()

    input_path = args.config
    if not input_path.is_file():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    data_root, run, year, era = parse_config_path(input_path)

    print(data_root)
    sample = sample_from_config_filename(input_path, era=era)
#    sample = normalize_skimmed_sample(sample)


    fileset = load_json(str(input_path))
    try:
        fileset = replace_files_in_json(
            fileset, run, year, era, args.umn, sample,
            allow_incomplete=args.allow_incomplete,
        )
    except RuntimeError as e:
        logging.error("%s", e)
        sys.exit(1)
    fileset = deduplicate_by_dataset(fileset)
    fileset = rename_dataset_key_to_sample(fileset)

    out_dir_path = output_dir(data_root=data_root, run=run, year=year, era=era) / "skimmed"
    out_file = out_dir_path / f"{era}_{sample}_fileset.json"
    write_fileset_json(out_file, fileset, indent=2, sort_keys=True)
    logging.info(f"Saved JSON to {out_file}")


if __name__ == "__main__":
    main()
