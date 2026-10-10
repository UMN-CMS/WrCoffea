#!/bin/bash
# Condor wrapper (runs inside the cc7 image on an LPC worker): kfactor_job.sh <13|13p6> <ncores> MWR_MN+MWR_MN+...
# (the points come '+'-joined: a Condor "queue ... from" item may not contain spaces)
set -e
E=$1; NCORE=$2; PTS=${3//+/ }; TAG=$(echo $3 | tr '+' '_' | cut -c1-80)
echo "host $(hostname) start $(date) energy $E points $PTS"
ls -la
bash run_points.sh $E $NCORE $PTS
echo "end $(date)"
cat out/timing.txt
# keep only what the analysis needs (the MG directory stays on the worker)
tar -czf out_${E}_${TAG}.tgz out
