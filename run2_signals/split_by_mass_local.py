#!/usr/bin/env python

import uproot
import awkward as ak
import re
import time
import os

# ----------------- Config -----------------

FILE_LIST = "local_wr_files.txt"   # list of local ROOT files (relative or absolute)
TREE_NAME = "Events"
STEP_SIZE = 100_000               # events per chunk
OUTPUT_DIR = "WRsplits_local"     # where to put the per-mass ROOT files

# Treat GenModel branch as TRUE if != 0
def genmodel_true(arr):
    return arr != 0

mass_pattern = re.compile(
    r"GenModel_WRtoNLtoLLJJ_MWR(\d+)_MN(\d+)_TuneCP5_13TeV_madgraph_pythia8"
)

def is_genmodel_field(name: str) -> bool:
    return bool(mass_pattern.fullmatch(name))

# ----------------- Read file list -----------------

with open(FILE_LIST) as f:
    in_files = [line.strip() for line in f if line.strip()]

if not in_files:
    raise RuntimeError(f"No input files found in {FILE_LIST}")

print(f"Found {len(in_files)} input files")
print("Example input file:", in_files[0])

# ----------------- PASS 1: discover all mass points -----------------

all_mass_points = set()        # global set of (MWR, MN)
file_mass_branches = {}        # per-file: fname -> {(MWR, MN): branch_name}

print("\n=== PASS 1: scanning files to find GenModel branches ===")
for fname in in_files:
    t0 = time.time()
    print(f"  Inspecting file: {fname}")
    with uproot.open(fname) as f:
        tree = f[TREE_NAME]
        this_file = {}
        for bname in tree.keys():
            m = mass_pattern.fullmatch(bname)
            if m:
                mwr, mn = map(int, m.groups())
                this_file[(mwr, mn)] = bname
                all_mass_points.add((mwr, mn))
        file_mass_branches[fname] = this_file
    dt = time.time() - t0
    print(f"    -> found {len(this_file)} mass branches in {dt:.1f} s")

if not all_mass_points:
    raise RuntimeError("No GenModel_WRtoNLtoLLJJ_MWR*_MN*_... branches found in any file")

print("\nGlobal mass points discovered:")
for (mwr, mn) in sorted(all_mass_points):
    print(f"  MWR={mwr:4d}  MN={mn:4d}")
print(f"Total distinct mass points: {len(all_mass_points)}")

# ----------------- Prepare output files -----------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

out_files = {}
out_trees = {}

print(f"\nCreating output ROOT files in: {OUTPUT_DIR}")
for (mwr, mn) in sorted(all_mass_points):
    out_name = os.path.join(
        OUTPUT_DIR,
        f"WRtoNLtoLLJJ_MWR{mwr}_MN{mn}.root"
    )
    print(f"  Will write MWR={mwr}, MN={mn} -> {out_name}")
    out_files[(mwr, mn)] = uproot.recreate(out_name)
    out_trees[(mwr, mn)] = None

# ----------------- PASS 2: loop over input files & fill outputs -----------------

print("\n=== PASS 2: splitting events by mass point ===")
t_global_start = time.time()

n_files = len(in_files)
for i_file, fname in enumerate(in_files, start=1):
    t_file_start = time.time()
    this_file_mass = file_mass_branches[fname]

    print(f"\n>>> [{i_file}/{n_files}] Processing file: {fname}")
    if not this_file_mass:
        print("    No GenModel branches in this file; skipping.")
        continue

    # Iterate over events in chunks
    chunk_idx = 0
    for batch in uproot.iterate(
        fname + ":" + TREE_NAME,
        step_size=STEP_SIZE,
        library="ak",
    ):
        chunk_idx += 1
        n_batch = len(batch)
        print(f"    Chunk {chunk_idx}, {n_batch} events")

        # All non-GenModel branches we keep in outputs
        fields_to_keep = [f for f in batch.fields if not is_genmodel_field(f)]

        # For each mass point present in THIS file, fill its output
        mass_items = list(this_file_mass.items())
        n_mass = len(mass_items)
        for j, ((mwr, mn), brname) in enumerate(mass_items, start=1):
            # Progress within this file
            print(f"      [{j}/{n_mass}] MWR={mwr}, MN={mn}, branch={brname}")
            flag_arr = batch[brname]
            mask = genmodel_true(flag_arr)
            n_sel = int(ak.sum(mask))
            if n_sel == 0:
                # Nothing in this chunk for this mass
                continue

            sel = batch[mask]
            sel_dict = {field: sel[field] for field in fields_to_keep}

            if out_trees[(mwr, mn)] is None:
                # First time we see this mass point: create tree schema
                out_files[(mwr, mn)]["Events"] = sel_dict
                out_trees[(mwr, mn)] = out_files[(mwr, mn)]["Events"]
            else:
                # Append to existing tree
                out_trees[(mwr, mn)].extend(sel_dict)

    dt_file = time.time() - t_file_start
    print(f"<<< Finished file {fname} in {dt_file/60:.1f} min")

t_global = time.time() - t_global_start
print(f"\nAll done. Total time: {t_global/60:.1f} min")
print("Output files:")
for (mwr, mn) in sorted(all_mass_points):
    out_name = os.path.join(
        OUTPUT_DIR,
        f"WRtoNLtoLLJJ_MWR{mwr}_MN{mn}.root"
    )
    print(f"  {out_name}")
