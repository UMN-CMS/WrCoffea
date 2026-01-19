#!/bin/bash
# Merge (hadd) all *_skim.root files in a directory into one combined ROOT file.
# Usage:
#   ./hadd_dataset.sh EGamma0_Run2024C
# or:
#   ./hadd_dataset.sh /full/path/to/EGamma0_Run2024C

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

set -e

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# Output name template
OUT_SUFFIX="_merged.root"
NTHREADS=4  # number of threads for hadd (adjust based on CPU cores)
# --------------------------------------------------------------------------

# Argument check
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_directory>"
    exit 1
fi

DATASET_DIR="$1"

# Allow both relative and absolute paths
cd "$DATASET_DIR" || { echo "❌ Directory not found: $DATASET_DIR"; exit 1; }

# Get dataset name (leaf)
DATASET_NAME=$(basename "$PWD")

# Construct output name
OUTFILE="${DATASET_NAME}${OUT_SUFFIX}"

# Check existing output file
if [ -f "$OUTFILE" ]; then
    echo "⚠️  Output file already exists: $OUTFILE"
    echo "    Overwrite? (y/n)"
    read -r ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -f "$OUTFILE"
fi

# Count input files
FILES=(*_skim.root)
NFILES=${#FILES[@]}
if [ "$NFILES" -eq 0 ]; then
    echo "❌ No *_skim.root files found in $PWD"
    exit 1
fi

echo "🔍 Found $NFILES input ROOT files."
echo "🧩 Merging into: $OUTFILE"
echo

# Run hadd
hadd -j "$NTHREADS" -f "$OUTFILE" "${FILES[@]}"

# Final check
if [ -f "$OUTFILE" ]; then
    echo
    echo "✅ Merge complete!"
    echo "📦 Output: $(realpath "$OUTFILE")"
    ls -lh "$OUTFILE"
else
    echo "❌ hadd failed — output file not created."
fi
