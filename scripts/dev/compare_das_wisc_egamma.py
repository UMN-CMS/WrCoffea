#!/usr/bin/env python3
"""
compare_das_wisc_egamma.py

Compares DAS file lists for each EGamma Run2024 dataset
against your Wisconsin Tier-2 skims, ignoring the "_skim" suffix.

Usage:
  ./compare_das_wisc_egamma.py
"""

import os
import subprocess
import sys
from typing import List, Set


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
XRD_HOST = "cmsxrootd.hep.wisc.edu"
BASE_DIR = "/store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24"

EGAMMA_MAP = {
    "EGamma0_Run2024C": "/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD",
    "EGamma0_Run2024D": "/EGamma0/Run2024D-MINIv6NANOv15-v1/NANOAOD",
    "EGamma0_Run2024E": "/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD",
    "EGamma0_Run2024F": "/EGamma0/Run2024F-MINIv6NANOv15-v1/NANOAOD",
    "EGamma0_Run2024G": "/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD",
    "EGamma0_Run2024H": "/EGamma0/Run2024H-MINIv6NANOv15-v2/NANOAOD",
    "EGamma0_Run2024I": "/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
    "EGamma1_Run2024C": "/EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD",
    "EGamma1_Run2024D": "/EGamma1/Run2024D-MINIv6NANOv15-v1/NANOAOD",
    "EGamma1_Run2024E": "/EGamma1/Run2024E-MINIv6NANOv15-v1/NANOAOD",
    "EGamma1_Run2024F": "/EGamma1/Run2024F-MINIv6NANOv15-v1/NANOAOD",
    "EGamma1_Run2024G": "/EGamma1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
    "EGamma1_Run2024H": "/EGamma1/Run2024H-MINIv6NANOv15-v1/NANOAOD",
    "EGamma1_Run2024I": "/EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD",
}


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def run_cmd(cmd: list[str]) -> list[str]:
    """Run a shell command and return stripped non-empty stdout lines."""
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(e.output, file=sys.stderr)
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def get_das_files(dataset: str) -> list[str]:
    """Return list of file basenames from DAS for a dataset."""
    cmd = ["dasgoclient", "-query", f"file dataset={dataset}", "-limit", "0"]
    files = run_cmd(cmd)
    return [os.path.basename(f) for f in files]


def get_wisc_files(egamma_dir: str) -> list[str]:
    """Return list of skim file basenames from Wisconsin Tier-2."""
    full_dir = os.path.join(BASE_DIR.rstrip("/"), egamma_dir)
    cmd = ["xrdfs", XRD_HOST, "ls", full_dir]
    files = run_cmd(cmd)
    return [os.path.basename(f) for f in files if f.endswith(".root")]


def normalize_skim_name(name: str) -> str:
    """Remove '_skim' before '.root' to match DAS filenames."""
    return name.replace("_skim.root", ".root") if name.endswith("_skim.root") else name


def compare(dataset: str, egamma_dir: str) -> None:
    """Compare DAS and Wisconsin files for one EGamma dataset."""
    print("=" * 90)
    print(f"Dataset    : {dataset}")
    print(f"Wisc folder: {egamma_dir}")

    das_files = get_das_files(dataset)
    wisc_files = get_wisc_files(egamma_dir)

    das_set: Set[str] = set(das_files)
    wisc_norm_set: Set[str] = {normalize_skim_name(f) for f in wisc_files}

    print(f"  DAS files  : {len(das_set)}")
    print(f"  Wisc files : {len(wisc_files)}")

    missing_on_wisc = sorted(das_set - wisc_norm_set)
    extra_on_wisc = sorted(wisc_norm_set - das_set)

    if not missing_on_wisc and not extra_on_wisc:
        print("  ✅ Match OK (counts and filenames align modulo _skim)")
        return

    if missing_on_wisc:
        print("\n  ❌ Files missing on Wisc (present in DAS but not skimmed):")
        for f in missing_on_wisc:
            print(f"    - {f}")

    if extra_on_wisc:
        print("\n  ❌ Extra files on Wisc (not found in DAS):")
        for f in extra_on_wisc:
            print(f"    - {f}")

    print()  # spacing


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------
def main():
    print(f"Comparing EGamma datasets between DAS and Wisconsin storage...\n")
    for egamma_dir, dataset in EGAMMA_MAP.items():
        compare(dataset, egamma_dir)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
