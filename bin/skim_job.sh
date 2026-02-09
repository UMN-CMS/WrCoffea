#!/bin/bash
# Condor worker script for WrCoffea skimming.
#
# Called by HTCondor with arguments: FILE_NUM DAS_PATH
# Untars the WrCoffea repo, runs the skimmer on one file, tars the output.
set -e

FILE_NUM=$1
DAS_PATH=$2

# Extract primary dataset name from DAS path (first component)
PRIMARY_DS=$(echo "$DAS_PATH" | cut -d'/' -f2)

echo "-------------------------------------------"
echo "Skim job: das_path=$DAS_PATH file=$FILE_NUM"
echo "-------------------------------------------"

# Unpack the repo and activate the .env venv (Python 3.10, matches container)
tar -xzf WrCoffea.tar.gz
cd WrCoffea
# The activate script has a hardcoded VIRTUAL_ENV from the build host.
# Override it with the correct path on the worker before sourcing.
export VIRTUAL_ENV="$(pwd)/.env"
export PATH="$(pwd)/.env/bin:$PATH"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Run the skimmer for this single file
python3 bin/skim.py run "$DAS_PATH" --start "$FILE_NUM" --end "$FILE_NUM" --local

# Tar the output for Condor transfer
cd data/skims
OUTPUT_TAR="${PRIMARY_DS}_skim$((FILE_NUM - 1)).tar.gz"
find . -name "*_skim.root" -type f | tar -czf "$OUTPUT_TAR" -T -
mv "$OUTPUT_TAR" /srv/

echo "-------------------------------------------"
echo "Job completed successfully"
echo "-------------------------------------------"
