#!/usr/bin/env bash
# copy_dy4j_skims.sh
# Copy a fixed list of DYto2L-4Jets samples from EOS to local

set -euo pipefail

# ——— CONFIGURATION ———
EOS_HOST="root://cmseos.fnal.gov"
REMOTE_DIR="/store/user/wijackso/WRAnalyzer/skims/2025/Run3/2022/Run3Summer22EE"

# the exact samples you want to copy
SAMPLES=(
  "DYto2L-4Jets_MLL-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8"
  "DYto2L-4Jets_MLL-50to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8"
)

echo "Copying ${#SAMPLES[@]} DYto2L-4Jets samples from $REMOTE_DIR …"

for sample in "${SAMPLES[@]}"; do
  echo "→ Copying $sample"
  xrdcp -r "${EOS_HOST}//${REMOTE_DIR}/${sample}" .
done

echo "✅ Done! All specified DYto2L-4Jets folders are now in $(pwd)"
