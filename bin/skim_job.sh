#!/bin/bash
# Condor worker script for WrCoffea skimming.
#
# Called by HTCondor with arguments: JOB_IDX DAS_PATH LFN [DS_NAME]
# Untars the WrCoffea repo, runs the skimmer on one file, tars the output.
set -e

JOB_IDX=$1
DAS_PATH=$2
LFN=$3
# Dataset name for output file naming. Falls back to the DAS primary
# dataset when not provided (backward compat with single-dataset runs).
DS_NAME=${4:-$(echo "$DAS_PATH" | cut -d'/' -f2)}

echo "-------------------------------------------"
echo "Skim job: das_path=$DAS_PATH job_idx=$JOB_IDX lfn=$LFN ds_name=$DS_NAME"
echo "-------------------------------------------"

# Unpack the repo and activate the .env venv (Python 3.12, matches container)
tar -xzf WrCoffea.tar.gz
cd WrCoffea
# The activate script has a hardcoded VIRTUAL_ENV from the build host.
# Override it with the correct path on the worker before sourcing.
export VIRTUAL_ENV="$(pwd)/.env"
export PATH="$(pwd)/.env/bin:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the skimmer for this single file (--lfn skips DAS query on worker)
STATUS_JSON_LOCAL="/tmp/${DS_NAME}_skim${JOB_IDX}.status.json"
python3 bin/skim.py run "$DAS_PATH" --local --lfn "$LFN" --status-json "$STATUS_JSON_LOCAL"

# Tar the output for Condor transfer
cd data/skims
OUTPUT_TAR="${DS_NAME}_skim${JOB_IDX}.tar.gz"
mapfile -t ROOT_FILES < <(find . -name "*_skim.root" -type f)
if [ "${#ROOT_FILES[@]}" -gt 0 ]; then
  printf '%s\n' "${ROOT_FILES[@]}" | tar -czf "$OUTPUT_TAR" -T -
else
  # Create a valid empty tarball so Condor output transfer still succeeds.
  tar -czf "$OUTPUT_TAR" --files-from /dev/null
fi
mv "$OUTPUT_TAR" /srv/
mv "$STATUS_JSON_LOCAL" "/srv/${DS_NAME}_skim${JOB_IDX}.status.json"

echo "-------------------------------------------"
echo "Job completed successfully"
echo "-------------------------------------------"
