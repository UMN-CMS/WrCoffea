#!/bin/bash

# Ensure at least two arguments are provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <campaign> <process> [dataset]"
    exit 1
fi

CAMPAIGN=$1
PROCESS=$2
DATASET_FILTER=${3:-""}  # Optional third argument

# Determine RUN based on CAMPAIGN content
if [[ "$CAMPAIGN" == *"RunII"* ]] || [[ "$CAMPAIGN" == "Run2018A" ]] || [[ "$CAMPAIGN" == "Run2018B" ]] || [[ "$CAMPAIGN" == "Run2018C" ]] || [[ "$CAMPAIGN" == "Run2018D" ]]; then
    RUN="RunII"
elif [[ "$CAMPAIGN" == *"Run3"* ]]; then
    RUN="Run3"
else
    echo "Error: Could not determine RUN from CAMPAIGN ($CAMPAIGN)"
    exit 1
fi

echo "Using RUN: $RUN"

# Define base directory paths
BASE_PATH="/uscms_data/d1/bjackson/WrCoffea/scripts/skims/${RUN}/$CAMPAIGN"
JSON_DIR="/uscms_data/d1/bjackson/WrCoffea/data/jsons/${RUN}/2022/$CAMPAIGN"
JSON_FILE="${JSON_DIR}/${CAMPAIGN}_${PROCESS}_preprocessed.json"

# Check if JSON file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: JSON file not found: $JSON_FILE"
    exit 1
fi

echo "Successfully loaded JSON file $JSON_FILE."

# Extract dataset names from JSON file
DATASETS=($(jq -r 'to_entries[] | select(.value.metadata.dataset) | .value.metadata.dataset' "$JSON_FILE"))

if [ "${#DATASETS[@]}" -eq 0 ]; then
    echo "Error: No datasets found in $JSON_FILE"
    exit 1
fi

# Filter dataset if optional argument is provided
if [ -n "$DATASET_FILTER" ]; then
    if [[ ! " ${DATASETS[@]} " =~ " ${DATASET_FILTER} " ]]; then
        echo "Error: Specified dataset $DATASET_FILTER not found in JSON file"
        exit 1
    fi
    DATASETS=("$DATASET_FILTER")
fi

mkdir -p $BASE_PATH

# Copy the tarball
cd /uscms/home/bjackson/nobackup
echo "Creating tarball of working directory. Wait approx 30 seconds..."
tar  --exclude=WrCoffea/.git --exclude=WrCoffea/.env --exclude=WrCoffea/WR_Plotter --exclude=WrCoffea/scripts/skims/Run3 --exclude=WrCoffea/test -czf WrCoffea.tar.gz WrCoffea
echo "Tarball created. Submitting scripts."

for DATASET in "${DATASETS[@]}"
do
    mkdir -p WrCoffea/scripts/skims/$RUN/$CAMPAIGN/$DATASET
    cp WrCoffea.tar.gz WrCoffea/scripts/skims/$RUN/$CAMPAIGN/$DATASET
done
rm WrCoffea.tar.gz

# Create JDL files and job directories
cd WrCoffea/scripts/skims
for DATASET in "${DATASETS[@]}"
do
    python3 create_job.py $CAMPAIGN $PROCESS $DATASET
done

# Submit jobs
THIS_PWD=$PWD
for DATASET in "${DATASETS[@]}"
do
    cd $BASE_PATH/$DATASET
    condor_submit job.jdl
    cd $THIS_PWD
done
