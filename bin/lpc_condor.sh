#!/bin/bash
# Host-side wrapper for `skim.py condor` at the LPC: condor_q/condor_submit only exist on the host,
# the skim tooling only runs inside the coffea container.  Usage (on a cmslpc login node):
#   bin/lpc_condor.sh --jobs-dir ~/nobackup/skimjobs/v2 [skim.py condor options...] [--submit]
# Environment: SKIM_IMAGE (container), SKIM_PLAN (plan dir), SKIM_OUT (campaign root with statuses),
#              SKIM_PROXY (x509 proxy file).  Defaults below match Billy's setup.
set -euo pipefail
IMG=${SKIM_IMAGE:-/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-almalinux9-noml:2026.9.0-py3.12}
PLAN=${SKIM_PLAN:-$HOME/nobackup/skims_v2/plans/v2}
OUT=${SKIM_OUT:-$HOME/nobackup/skims_v2}
PROXY=${SKIM_PROXY:-/uscms/homes/b/bjackson/x509up_u$(id -u)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
SUBMIT=0; ARGS=(); JOBS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1;;
    --jobs-dir) JOBS="$2"; ARGS+=("$1" "$2"); shift;;
    *) ARGS+=("$1");;
  esac
  shift
done
[ -n "$JOBS" ] || { echo "need --jobs-dir"; exit 2; }
mkdir -p "$JOBS"
# the queue as the container cannot see it
condor_q -af SkimGroup JobStatus Iwd > "$JOBS/inflight.txt" 2>/dev/null || : > "$JOBS/inflight.txt"
export X509_USER_PROXY=$PROXY
apptainer exec -B /cvmfs -B /uscms_data -B /uscms/home -B /uscms/homes "$IMG" bash -c \
  "cd '$REPO' && python3 bin/skim.py condor --plan-dir '$PLAN' --out '$OUT' --proxy '$PROXY' --inflight-file '$JOBS/inflight.txt' ${ARGS[*]}"
if [ $SUBMIT -eq 1 ] && [ -s "$JOBS/groups.txt" ]; then
  (cd "$JOBS" && condor_submit submit.jdl)
fi
