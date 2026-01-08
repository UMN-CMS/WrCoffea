#!/usr/bin/env python

import uproot
import re
import argparse

tree_name = "Events"

mass_pattern = re.compile(
    r"GenModel_WRtoNLtoLLJJ_MWR(\d+)_MN(\d+)_TuneCP5_13TeV_madgraph_pythia8"
)

parser = argparse.ArgumentParser(
    description="Scan NanoAOD files and list all (MWR, MN) mass points"
)
parser.add_argument(
    "--file-list",
    required=True,
    help="Text file with one ROOT file path per line (local paths)",
)
parser.add_argument(
    "--out",
    default="mass_points.txt",
    help="Output text file with 'MWR MN' per line (default: mass_points.txt)",
)
args = parser.parse_args()

# Read file list
with open(args.file_list) as f:
    files = [line.strip() for line in f if line.strip()]

if not files:
    raise RuntimeError(f"No files found in {args.file_list}")

print(f"Scanning {len(files)} files from {args.file_list}")
print("Example:", files[0])

all_mass_points = set()

for fname in files:
    print("  Inspecting:", fname)
    with uproot.open(fname) as f:
        tree = f[tree_name]
        for bname in tree.keys():
            m = mass_pattern.fullmatch(bname)
            if m:
                mwr, mn = map(int, m.groups())
                all_mass_points.add((mwr, mn))

if not all_mass_points:
    raise RuntimeError("No GenModel_WRtoNLtoLLJJ_MWR*_MN*_... branches found in any file")

print("\nFound mass points:")
for (mwr, mn) in sorted(all_mass_points):
    print(f"  MWR={mwr:4d}  MN={mn:4d}")

# Write to output file
with open(args.out, "w") as out:
    for (mwr, mn) in sorted(all_mass_points):
        out.write(f"{mwr} {mn}\n")

print(f"\nWrote {len(all_mass_points)} mass points to {args.out}")
