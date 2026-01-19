#!/bin/bash
# Merge *_skim.root files, grouping by HLT_* branch content and splitting outputs
# so each merged file contains at most MAX_EVENTS events.
# Uses uproot to inspect branches and events, then ROOT's hadd to merge.
# Verifies that total event counts before and after merging are consistent.

# --- Environment -------------------------------------------------------------
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# ---------------------------------------------------------------------------

set -euo pipefail

MAX_EVENTS=1000000          # target max events per output file
OUT_SUFFIX="_part"          # output naming pattern (DATASET${OUT_SUFFIX}<N>.root)
TREE_NAME="Events"          # default tree name (override with --tree)

usage() {
  echo "Usage: $0 [--tree TREE_NAME] <dataset_dir>"
  exit 1
}

# --- Parse args --------------------------------------------------------------
if [[ $# -lt 1 ]]; then usage; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree) TREE_NAME="${2:-Events}"; shift 2;;
    -h|--help) usage;;
    *) DATASET_DIR="$1"; shift;;
  esac
done

[[ -d "${DATASET_DIR:-}" ]] || { echo "❌ Directory not found: ${DATASET_DIR:-<nil>}"; exit 1; }
cd "$DATASET_DIR"

DATASET_NAME="$(basename "$PWD")"

# --- Ensure uproot is available ---------------------------------------------
has_uproot() {
  python3 - <<'PY' >/dev/null 2>&1
import importlib.util
assert importlib.util.find_spec("uproot") is not None
PY
}

if ! has_uproot; then
  echo "❌ uproot is required for HLT-based grouping but is not available in this environment."
  exit 1
fi

# --- HLT-aware merging implemented in Python --------------------------------
export DATASET_NAME MAX_EVENTS OUT_SUFFIX TREE_NAME

python3 <<'PY'
import os
import sys
import glob
import subprocess
import hashlib
from collections import defaultdict

import uproot
from uproot.behaviors.TTree import TTree

DATASET_NAME = os.environ["DATASET_NAME"]
MAX_EVENTS = int(os.environ.get("MAX_EVENTS", "1000000"))
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "_part")
TREE_NAME = os.environ.get("TREE_NAME", "Events")

# ---------------------------------------------------------------------------
# Discover input *_skim.root files
# ---------------------------------------------------------------------------
files = sorted(f for f in glob.glob("*_skim.root") if os.path.isfile(f))
if not files:
    print("❌ No *_skim.root files in current directory")
    sys.exit(1)

print(f"🔍 Found {len(files)} input ROOT files.")
print(f"🌲 Using tree name: '{TREE_NAME}' (auto-detect fallback enabled)")
print(f"📦 Splitting outputs at ≤ {MAX_EVENTS} events per file")
print("🧪 Grouping files by HLT_* branch content...\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_tree(path, tname):
    """Return a TTree object from the file, preferring tname but falling back to first TTree."""
    with uproot.open(path) as f:
        tree = None
        if tname in f:
            obj = f[tname]
            if isinstance(obj, TTree):
                tree = obj
        if tree is None:
            # Fallback: first TTree in the file
            for k, cls in f.classnames().items():
                if cls.startswith("TTree"):
                    tree = f[k]
                    break
        return tree

def get_hlt_signature_and_events(path, tname):
    """Return (HLT_signature_tuple, n_events) for a file."""
    tree = get_tree(path, tname)
    if tree is None:
        raise RuntimeError("No TTree found in file")
    hlt_branches = sorted(tree.keys(filter_name="HLT_*"))
    signature = tuple(hlt_branches)
    n_events = int(tree.num_entries)
    return signature, n_events

def get_nevents(path, tname):
    """Return number of entries in the output tree (for verification)."""
    tree = get_tree(path, tname)
    if tree is None:
        raise RuntimeError("No TTree found in output file")
    return int(tree.num_entries)

def short_hash(signature):
    sig_str = "|".join(signature)
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:8]

# ---------------------------------------------------------------------------
# Group files by HLT signature
# ---------------------------------------------------------------------------
groups = defaultdict(list)
total_in_events = 0
skipped_files = []

for i, fname in enumerate(files, 1):
    try:
        sig, nev = get_hlt_signature_and_events(fname, TREE_NAME)
    except Exception as e:
        print(f"⚠️  Skipping {fname}: failed to inspect HLT branches / events ({e})")
        skipped_files.append(fname)
        continue

    groups[sig].append((fname, nev))
    total_in_events += nev
    print(f"📄 [{i}/{len(files)}] {fname:<60} : {nev} events, {len(sig)} HLT_* branches")

print(f"\n[INFO] Found {len(groups)} distinct HLT branch sets")
print(f"[INFO] Total events across all usable input files: {total_in_events}\n")

if not groups:
    print("❌ No usable files after HLT inspection. Aborting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Merge within each HLT group, respecting MAX_EVENTS
# ---------------------------------------------------------------------------
output_files = []
global_part = 1

def flush_chunk(chunk_files, chunk_events, part_index):
    if not chunk_files:
        return part_index
    outfile = f"{DATASET_NAME}{OUT_SUFFIX}{part_index}.root"
    print(f"🧩 Merging {len(chunk_files)} files into {outfile} (≈ {chunk_events} events)")
    cmd = ["hadd", "-f", outfile] + chunk_files
    print("  ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(outfile) / (1024.0 * 1024.0)
    print(f"✅ Wrote {outfile} ({size_mb:.1f} MB)")
    output_files.append(outfile)
    return part_index + 1

for gidx, (signature, file_info_list) in enumerate(groups.items()):
    sig_hash = short_hash(signature)
    print("=" * 80)
    print(f"[GROUP {gidx}] HLT signature hash: {sig_hash}")
    print(f"[GROUP {gidx}] Number of HLT_* branches: {len(signature)}")
    print(f"[GROUP {gidx}] Number of files: {len(file_info_list)}")
    print(f"[GROUP {gidx}] Total events in this group: {sum(nev for _, nev in file_info_list)}")

    # Sort files within group for reproducibility
    file_info_list = sorted(file_info_list, key=lambda x: x[0])

    cur_files = []
    cur_events = 0

    for fname, nev in file_info_list:
        # If adding this file would exceed the limit, flush current chunk
        if cur_files and (cur_events + nev > MAX_EVENTS):
            global_part = flush_chunk(cur_files, cur_events, global_part)
            cur_files = []
            cur_events = 0

        cur_files.append(fname)
        cur_events += nev

    # Flush remaining files in this group
    if cur_files:
        global_part = flush_chunk(cur_files, cur_events, global_part)

print("\n🎉 Merge complete.")

# ---------------------------------------------------------------------------
# Post-merge verification: recount events in merged outputs
# ---------------------------------------------------------------------------
print("\n🔍 Verifying merged outputs...")
total_out_events = 0
for outfile in sorted(output_files):
    try:
        nev = get_nevents(outfile, TREE_NAME)
    except Exception as e:
        print(f"⚠️  Could not read events from {outfile}: {e}")
        continue
    total_out_events += nev
    print(f"📦 {outfile:<60} : {nev} events")

print()
print(f"📊 Input total (usable files) : {total_in_events}")
print(f"📊 Output total (merged files): {total_out_events}")

if total_in_events == total_out_events:
    print("✅ Event counts match perfectly!")
    # Remove original skims that were actually used
    print()
    print("🧹 Removing original *_skim.root files...")
    for fname in files:
        if fname in skipped_files:
            continue
        try:
            os.remove(fname)
        except Exception as e:
            print(f"⚠️  Failed to delete {fname}: {e}")
    print("✅ Cleanup complete.")
else:
    diff = total_out_events - total_in_events
    print("⚠️  Mismatch detected! Original *_skim.root files have been preserved.")
    print(f"    Difference: {diff} events (output - input)")

# Final summary of merged files
print()
print(f"📂 Generated {len(output_files)} merged file(s):")
for outfile in sorted(output_files):
    size_mb = os.path.getsize(outfile) / (1024.0 * 1024.0)
    print(f"  {outfile:<60} {size_mb:.1f} MB")

if skipped_files:
    print()
    print("⚠️  The following files were skipped due to inspection errors:")
    for s in skipped_files:
        print(f"  - {s}")

# Explicit final total of events in hadded files (as requested)
print()
print(f"🔢 Total events in all hadded files: {total_out_events}")

PY
