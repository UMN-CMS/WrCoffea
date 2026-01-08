#!/usr/bin/env python

import uproot
import awkward as ak
import re
import time
import argparse
import os

tree_name = "Events"
step_size = 100_000
xrootd_prefix = "root://cmsxrootd.fnal.gov"

mass_pattern = re.compile(
    r"GenModel_WRtoNLtoLLJJ_MWR(\d+)_MN(\d+)_TuneCP5_13TeV_madgraph_pythia8"
)


def genmodel_true(arr):
    """Treat GenModel flag as true if != 0."""
    return arr != 0


def is_genmodel_field(name: str) -> bool:
    return bool(mass_pattern.fullmatch(name))


# ------------------- CLI -------------------

parser = argparse.ArgumentParser(
    description="Split RPscan NanoAOD for ONE (MWR, MN) mass point"
)
parser.add_argument(
    "--file-list",
    required=True,
    help="Text file with one ROOT file path per line (local or /store/... if using XRootD)",
)
parser.add_argument(
    "--out-file",
    required=True,
    help="Output ROOT file for this mass point",
)
parser.add_argument(
    "--mwr",
    type=int,
    required=True,
    help="WR mass (e.g. 2200)",
)
parser.add_argument(
    "--mn",
    type=int,
    required=True,
    help="N mass (e.g. 1600)",
)
parser.add_argument(
    "--use-xrootd",
    action="store_true",
    help="Treat file_list entries as /store/... and prepend root://cmsxrootd.fnal.gov//",
)

args = parser.parse_args()

target_mwr = args.mwr
target_mn = args.mn
branch_name = (
    f"GenModel_WRtoNLtoLLJJ_MWR{target_mwr}_MN{target_mn}_TuneCP5_13TeV_madgraph_pythia8"
)

print(f"Target mass point: MWR={target_mwr}, MN={target_mn}")
print(f"Expected branch:   {branch_name}")

# ------------------- Read file list -------------------

with open(args.file_list) as f:
    raw_files = [line.strip() for line in f if line.strip()]

if not raw_files:
    raise RuntimeError(f"No input files found in {args.file_list}")

if args.use_xrootd:
    in_files = [f"{xrootd_prefix}//{path.lstrip('/')}" for path in raw_files]
else:
    in_files = raw_files

# ------------------- Discover files that contain this branch ---------------

files_with_branch = []
fields_to_keep = None  # Will be filled from first file that has the branch

print("Scanning files to find the target GenModel branch...")
for fname in in_files:
    print("  Inspecting:", fname)
    with uproot.open(fname) as f:
        tree = f[tree_name]
        if branch_name in tree.keys():
            files_with_branch.append(fname)
            if fields_to_keep is None:
                # Decide which branches to keep in the output: all non-GenModel
                fields_to_keep = [
                    b for b in tree.keys() if not is_genmodel_field(b)
                ]

# IMPORTANT CHANGE: if this file (or these files) don't have the branch, just exit quietly
if not files_with_branch:
    print(f"Branch {branch_name} not found in any file listed in {args.file_list}. Nothing to do.")
    raise SystemExit(0)

print("Files that contain the branch:")
for f_ in files_with_branch:
    print("  ", f_)

print("Number of branches kept in output:", len(fields_to_keep))

# ------------------- Create or open output file/tree -------------------

out_dir = os.path.dirname(os.path.abspath(args.out_file))
if out_dir and not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# NEW: append if file already exists, otherwise create
if os.path.exists(args.out_file):
    print(f"Output file {args.out_file} exists, opening in update mode.")
    out_file = uproot.update(args.out_file)
    try:
        out_tree = out_file["Events"]
        # If we somehow didn't get fields_to_keep yet, derive from existing tree
        if fields_to_keep is None:
            fields_to_keep = list(out_tree.keys())
    except KeyError:
        print("Existing file has no 'Events' tree, will create it.")
        out_tree = None
else:
    print(f"Output file {args.out_file} does not exist, creating new file.")
    out_file = uproot.recreate(args.out_file)
    out_tree = None

# ------------------- Loop over files & fill output -------------------

n_files = len(files_with_branch)
t_start = time.time()

for i_file, fname in enumerate(files_with_branch, start=1):
    print(f"\n=== Processing file {i_file}/{n_files}: {fname} ===")
    t0 = time.time()

    # Read only the branches we need: physics branches + this GenModel flag
    branches_to_read = fields_to_keep + [branch_name]

    for batch in uproot.iterate(
        fname + ":" + tree_name,
        filter_name=branches_to_read,
        step_size=step_size,
        library="ak",
    ):
        # batch is an ak.Array with only the requested branches
        flag_arr = batch[branch_name]
        mask = genmodel_true(flag_arr)

        if ak.sum(mask) == 0:
            continue

        sel = batch[mask]

        # Build dict for writing, dropping the GenModel flag itself
        sel_dict = {field: sel[field] for field in fields_to_keep}

        if out_tree is None:
            out_file["Events"] = sel_dict
            out_tree = out_file["Events"]
        else:
            out_tree.extend(sel_dict)

    dt = time.time() - t0
    print(f"=== Finished file {i_file}/{n_files} in {dt:.1f} s ({dt/60:.1f} min) ===")

t_total = time.time() - t_start
n_entries = out_tree.num_entries if out_tree is not None else 0
print(f"\nDone. Wrote {n_entries} events")
print(f"Output file: {args.out_file}")
print(f"Total time: {t_total:.1f} s ({t_total/60:.1f} min)")
