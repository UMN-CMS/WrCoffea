#!/bin/bash

# Exit on error
set -e

FILE_NUM=$1
CAMPAIGN=$2
PROCESS=$3
DATASET=$4

# Determine RUN based on CAMPAIGN content
if [[ "$CAMPAIGN" == *"2018"* ]]; then
    RUN="RunII"
    YEAR="2018"
elif [[ "$CAMPAIGN" == *"Run3Summer22"* ]]; then
    RUN="Run3"
    YEAR="2022"
elif [[ "$CAMPAIGN" == *"Run3Summer23"* ]]; then
    RUN="Run3"
    YEAR="2023"
elif [[ "$CAMPAIGN" == *"RunIII2024Summer24"* ]]; then
    RUN="Run3"
    YEAR="2024"
else
    echo "Error: Could not determine RUN from CAMPAIGN ($CAMPAIGN)"
    exit 1
fi

echo "-------------------------------------------"
tar -xzf WrCoffea.tar.gz
cd WrCoffea
source .venv/bin/activate

echo "-------------------------------------------"
echo "### **Start of Job**"
echo "**MC Campaign:** $CAMPAIGN"
echo "**Process:** $PROCESS"
echo "**Dataset:** $DATASET"
echo "-------------------------------------------"

export PATH="/srv/WrCoffea/.venv/bin:$PATH"
export PYTHONPATH="/srv/WrCoffea/venv/lib/python3.9/site-packages:$PYTHONPATH"
python3 scripts/setup/skims/skim_files.py $CAMPAIGN $PROCESS $DATASET --start $FILE_NUM

cd scripts/setup/skims/tmp/$RUN/$YEAR/$CAMPAIGN
cd $DATASET
echo "### **Output Files Generated**"
echo "-------------------------------------------"
echo "**Directory:** $(pwd)"
echo "**Files**"
echo "|------------| User |--------| Size |----------------------------------| Filename |-----------------------------------|"
echo "$(ls -lrth)"

cd ..
tar -czf "${DATASET}_skim$((FILE_NUM - 1)).tar.gz" "$DATASET"

# Move the tarball to `/srv/` for Condor to detect it
mv "${DATASET}_skim$((FILE_NUM - 1)).tar.gz" /srv/

echo "### **End of Job**"
echo "-------------------------------------------"
echo "✅ Job completed successfully! 🚀"

# Debugging
#echo "Using Python: $(which python3)"
#echo "Python version: $(python3 --version)"
#echo "Python Path: $PYTHONPATH"

#echo "Now the contents of the directory is:"
#echo "pwd: $(pwd)"
#echo "ls: $(ls -lrt)"
