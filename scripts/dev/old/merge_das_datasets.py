#!/usr/bin/env python3
"""
merge_das_datasets.py

For each dataset listed in the JSON config:
  1. Query DAS for the list of files via `dasgoclient -json`.
  2. For the special point (WR8000_N2500), download all .root files via xrdcp into the directory.
  3. Otherwise, merge all files into a single ROOT file with hadd -fk into <dataset_name>/<dataset_name>.root
  4. Sum `genEventSumw` in the `Runs` tree and add it as a new branch `genEventSumw` in the `Events` tree.
"""

import os
import subprocess
import json
import argparse
from array import array

import ROOT

# Use the global redirector on port 1094
#XROOTD_PREFIX = "root://cms-xrd-global.cern.ch:1094//"
XROOTD_PREFIX = "root://cms-xrd-global.cern.ch//"
# The special mass point we want to download instead of hadd
SPECIAL_POINT = (
    "WRtoNEltoElElJJ_MWR8000_N2500_TuneCP5_13p6TeV_madgraph-pythia8"
)


def query_das_files(dataset):
    """
    Return a list of filenames for a dataset via DAS JSON.
    """
    cmd = [
        "dasgoclient",
        "--query", f"file dataset={dataset}",
        "-json"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print(f"[error] DAS query failed for {dataset}: {p.stderr.strip()}")
        return []
    try:
        entries = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        print(f"[error] failed to parse DAS JSON for {dataset}: {e}")
        return []

    files = []
    for entry in entries:
        for f in entry.get("file", []):
            name = f.get("name")
            if name:
                files.append(name)
    return files


def download_files(dataset_name, filenames):
    """
    Download each file via xrdcp into the dataset directory.
    """
    outdir = dataset_name
    os.makedirs(outdir, exist_ok=True)
    for fn in filenames:
        src = XROOTD_PREFIX + fn.lstrip("/")
        dst = os.path.join(outdir, os.path.basename(fn))
        print(f"Downloading: {src} -> {dst}")
        subprocess.run(["xrdcp", "-f", src, dst], check=True)


def hadd_all(dataset_dir, dataset_name, filenames):
    """
    Run hadd -fk on all files, writing the output under dataset_dir.
    Returns the path to the merged file.
    """
    out_path = os.path.join(dataset_dir, f"{dataset_name}.root")
    inputs = [XROOTD_PREFIX + fn.lstrip("/") for fn in filenames]

    print("[debug] first 3 input URLs for hadd:")
    for url in inputs[:3]:
        print("   ", url)

    print(f" => merging {len(inputs)} files into {out_path}")
    subprocess.run(["hadd", "-fk", out_path] + inputs, check=True)
    return out_path


def add_genEventSumw_branch(file_path):
    """
    Open the merged ROOT file in UPDATE mode, sum genEventSumw over Runs,
    and add a new branch 'genEventSumw' to the Events tree.
    """
    f = ROOT.TFile.Open(file_path, "UPDATE")
    runs = f.Get("Runs")
    total_sumw = 0.0
    for i in range(runs.GetEntries()):
        runs.GetEntry(i)
        total_sumw += getattr(runs, "genEventSumw", 0.0)

    events = f.Get("Events")
    buf = array('f', [0.0])
    branch = events.Branch("genEventSumw", buf, "genEventSumw/F")

    n_events = events.GetEntries()
    print(f"[info] adding genEventSumw={total_sumw} into {n_events} Events")
    for _ in range(n_events):
        buf[0] = total_sumw
        branch.Fill()

    events.Write("", ROOT.TObject.kOverwrite)
    f.Close()


def process_dataset(dataset):
    """
    Process a single dataset:
      - For SPECIAL_POINT: download files
      - Otherwise: hadd then add genEventSumw branch
    """
    dataset_name = dataset.strip("/").split("/")[0]
    files = query_das_files(dataset)
    if not files:
        print(f"[warn] no files found for {dataset}")
        return

    if dataset_name == SPECIAL_POINT:
        print(f"[info] SPECIAL_POINT detected ({dataset_name}) – downloading files only.")
        download_files(dataset_name, files)
        return

    # normal flow: hadd + genEventSumw
    os.makedirs(dataset_name, exist_ok=True)
    merged_path = hadd_all(dataset_name, dataset_name, files)
    add_genEventSumw_branch(merged_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge or download CMS DAS datasets as appropriate."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config listing datasets"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    for ds in cfg.keys():
        print(f"\n=== Processing {ds} ===")
        process_dataset(ds)


if __name__ == "__main__":
    main()

