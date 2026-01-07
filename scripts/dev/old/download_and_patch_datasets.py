#!/usr/bin/env python3
"""
download_patch_and_merge.py

For each dataset listed in the JSON config:
  1. Query DAS for the list of files via `dasgoclient -json`.
  2. Download each .root file via xrdcp into <dataset_name>/ directory.
     Try both global and :1094 redirectors if needed.
  3. For each downloaded ROOT, open in UPDATE mode, sum `genEventSumw` in
     the `Runs` tree, and add that sum as a new branch `genEventSumw` (float)
     into the `Events` tree.
  4. Once all files are patched, hadd them into a single file
     <dataset_name>/<dataset_name>.root, then delete the original files.
"""

import os
import subprocess
import json
import argparse
from array import array

import ROOT

# Try both the plain and the :1094 XRootD redirectors
XROOTD_PREFIXES = [
    "root://cms-xrd-global.cern.ch//",
    "root://cms-xrd-global.cern.ch:1094//",
]

# DAS client and copy commands
DAS_CMD = "dasgoclient"
COPY_CMD = "xrdcp"
HADD_CMD = "hadd"


def query_das_files(dataset):
    cmd = [DAS_CMD, "--query", f"file dataset={dataset}", "-json"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print(f"[error] DAS query failed for {dataset}: {p.stderr.strip()}")
        return []
    try:
        entries = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        print(f"[error] parsing DAS JSON for {dataset}: {e}")
        return []
    files = []
    for entry in entries:
        for f in entry.get("file", []):
            name = f.get("name")
            if name:
                files.append(name)
    return files


def download_and_patch(outdir, filenames):
    os.makedirs(outdir, exist_ok=True)
    local_files = []
    for fn in filenames:
        dst = os.path.join(outdir, os.path.basename(fn))
        # try each redirector prefix until one works
        for prefix in XROOTD_PREFIXES:
            src = prefix + fn.lstrip("/")
            print(f"Downloading: {src} -> {dst}")
            p = subprocess.run(
                [COPY_CMD, "-f", src, dst],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if p.returncode == 0:
                print("  success")
                local_files.append(dst)
                break
            else:
                err = p.stderr.strip().splitlines()[-1]
                print(f"  failed ({p.returncode}): {err}")
        else:
            print(f"  [ERROR] could not download {fn}, skipping.")
            continue

        # patch the downloaded file
        f = ROOT.TFile.Open(dst, "UPDATE")
        if not f or f.IsZombie():
            print(f"  [ERROR] cannot open {dst} in UPDATE mode")
            continue
        runs = f.Get("Runs")
        total_sumw = 0.0
        if runs:
            for i in range(runs.GetEntries()):
                runs.GetEntry(i)
                total_sumw += getattr(runs, "genEventSumw", 0.0)
        events = f.Get("Events")
        if events:
            buf = array('f', [0.0])
            branch = events.Branch("genEventSumw", buf, "genEventSumw/F")
            n_events = events.GetEntries()
            print(f"  [info] embedding genEventSumw={total_sumw} into {n_events} events")
            for _ in range(n_events):
                buf[0] = total_sumw
                branch.Fill()
            events.Write("", ROOT.TObject.kOverwrite)
        f.Close()
    return local_files


def hadd_all(outdir, local_files):
    if not local_files:
        print(f"[warn] No local files to hadd in {outdir}")
        return
    target = os.path.join(outdir, f"{outdir}.root")
    cmd = [HADD_CMD, "-f", target] + local_files
    print(f"Merging {len(local_files)} files into {target}")
    subprocess.run(cmd, check=True)
    # delete the originals
    for lf in local_files:
        try:
            os.remove(lf)
            print(f"  [info] deleted {lf}")
        except OSError as e:
            print(f"  [warn] could not delete {lf}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download, patch, and hadd CMS DAS ROOT files"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config listing datasets"
    )
    args = parser.parse_args()

    cfg = json.load(open(args.config))
    for ds in cfg.keys():
        dataset_name = ds.strip("/").split("/")[0]
        print(f"\n=== Processing {dataset_name} ===")
        files = query_das_files(ds)
        if not files:
            print(f"[warn] no files found for {ds}")
            continue
        local_files = download_and_patch(dataset_name, files)
        hadd_all(dataset_name, local_files)


if __name__ == "__main__":
    main()
