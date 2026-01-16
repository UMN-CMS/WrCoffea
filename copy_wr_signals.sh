#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
QUERY='dataset=/WRtoN*to*JJ_MWR*_N*_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer23BPixNanoAODv12*/NANOAODSIM'

DEST_BASE='davs://cmsxrootd.hep.wisc.edu:1094/store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24/Signals'
REDIRECTOR='root://cms-xrd-global.cern.ch'

# gfal-copy timeout (seconds)
COPY_TIMEOUT=3600

# -----------------------------
# Checks
# -----------------------------
command -v dasgoclient >/dev/null || { echo "ERROR: dasgoclient not found in PATH"; exit 1; }
command -v gfal-copy   >/dev/null || { echo "ERROR: gfal-copy not found in PATH"; exit 1; }
command -v gfal-mkdir  >/dev/null || { echo "ERROR: gfal-mkdir not found in PATH"; exit 1; }
command -v gfal-stat   >/dev/null || { echo "ERROR: gfal-stat not found in PATH"; exit 1; }

if ! command -v voms-proxy-info >/dev/null; then
  echo "WARNING: voms-proxy-info not found; assuming proxy is handled elsewhere."
else
  if ! voms-proxy-info -timeleft >/dev/null 2>&1; then
    echo "ERROR: No valid proxy found. Run: voms-proxy-init -voms cms"
    exit 1
  fi
  TL=$(voms-proxy-info -timeleft)
  if [[ "$TL" -lt 600 ]]; then
    echo "ERROR: Proxy has <10 minutes left ($TL sec). Renew it."
    exit 1
  fi
fi

# -----------------------------
# Make destination directory
# -----------------------------
echo "[i] Ensuring destination exists:"
echo "    $DEST_BASE"
gfal-mkdir -p "$DEST_BASE"

# -----------------------------
# Get dataset list
# -----------------------------
echo "[i] Querying DAS for datasets..."
dasgoclient -query="$QUERY" -limit=0 | sort > datasets.txt
echo "[i] Found $(wc -l < datasets.txt) datasets -> datasets.txt"

# -----------------------------
# Loop datasets and copy files
# -----------------------------
LOG="copy_wr_signals_$(date +%Y%m%d_%H%M%S).log"
echo "[i] Logging to $LOG"
echo "[i] Starting copy..." | tee -a "$LOG"

while read -r ds; do
  [[ -z "$ds" ]] && continue

  # safe folder name: strip leading '/', replace '/' with '__'
  tag="$(echo "$ds" | sed 's|^/||; s|/|__|g')"
  dest_dir="${DEST_BASE}/${tag}"

  echo "------------------------------------------------------------" | tee -a "$LOG"
  echo "[i] Dataset: $ds" | tee -a "$LOG"
  echo "[i] Dest   : $dest_dir" | tee -a "$LOG"

  gfal-mkdir -p "$dest_dir"

  # list files in this dataset
  dasgoclient -query="file dataset=${ds}" -limit=0 | sort > files.txt
  nfiles="$(wc -l < files.txt)"
  echo "[i] Files: $nfiles" | tee -a "$LOG"

  # copy each file (skip if already exists at destination)
  while read -r lfn; do
    [[ -z "$lfn" ]] && continue
    src="${REDIRECTOR}/${lfn}"
    dst="${dest_dir}/$(basename "$lfn")"

    if gfal-stat "$dst" >/dev/null 2>&1; then
      echo "[skip] exists: $dst" | tee -a "$LOG"
      continue
    fi

    echo "[copy] $src -> $dst" | tee -a "$LOG"
    if ! gfal-copy -p -t "$COPY_TIMEOUT" "$src" "$dst" >>"$LOG" 2>&1; then
      echo "[ERROR] copy failed: $src" | tee -a "$LOG"
      echo "        continuing (see log: $LOG)" | tee -a "$LOG"
      continue
    fi
  done < files.txt

done < datasets.txt

echo "------------------------------------------------------------" | tee -a "$LOG"
echo "[i] Done." | tee -a "$LOG"
echo "[i] Destination listing:" | tee -a "$LOG"
gfal-ls -l "$DEST_BASE" | tee -a "$LOG"
