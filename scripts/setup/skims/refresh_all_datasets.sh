#!/bin/bash
# Refresh a selected list of datasets on Wisc Tier-2.
# Downloads → merges locally → deletes old remote files → uploads new → cleans local.
# --- XRootD timeout settings to avoid hangs ---
export XRD_CONNECTIONWINDOW=15
export XRD_REQUESTTIMEOUT=20
export XRD_STREAMTIMEOUT=60
export XRD_REDIRECTLIMIT=2

set -euo pipefail

XROOTD="root://cmsxrootd.hep.wisc.edu"
REMOTE_BASE="/store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24"

# Only these datasets:
DATASETS=(
  "EGamma0_Run2024H"
  "EGamma1_Run2024C"
  "EGamma1_Run2024G"
  "EGamma1_Run2024H"
  "Muon0_Run2024C"
  "Muon0_Run2024G"
  "Muon0_Run2024H"
  "Muon1_Run2024C"
  "Muon1_Run2024H"
)

for DATASET in "${DATASETS[@]}"; do
  REMOTE_PATH="${REMOTE_BASE}/${DATASET}"

  echo
  echo "=============================================="
  echo "🚀 Processing ${DATASET}"
  echo "=============================================="

  # 1. Download from Wisconsin
  echo "🔽 Downloading from Wisconsin..."
  xrdcp -rP ${XROOTD}/${REMOTE_PATH} .

  # 2. Merge locally (now with HLT-aware merging)
  echo
  echo "⚙️  Merging locally..."
  ./hadd_dataset2.sh "${DATASET}/"

  # 3. Remove old version on Wisconsin
  echo
  echo "🧹 Removing old version on Wisconsin..."
  xrdfs ${XROOTD} ls ${REMOTE_PATH} | while read -r file; do
    echo "  Deleting $file"
    xrdfs ${XROOTD} rm "$file" || echo "⚠️  Failed to delete $file"
  done
  xrdfs ${XROOTD} rmdir ${REMOTE_PATH} || echo "⚠️  Failed to remove directory ${REMOTE_PATH}"

  # 4. Upload new merged dataset
  echo
  echo "☁️  Uploading new merged dataset..."
  xrdcp -rP "${DATASET}" ${XROOTD}//${REMOTE_BASE}/

  # 5. Cleanup local copy
  echo
  echo "🧽 Cleaning local copy..."
  rm -rf "${DATASET}"

  echo
  echo "✅ Finished refreshing ${DATASET}"
  echo "----------------------------------------------"
done

echo
echo "🎉 All selected datasets refreshed successfully!"
