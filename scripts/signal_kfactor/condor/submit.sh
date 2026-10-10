#!/bin/bash
# Write and submit the Condor campaign for one energy on a cmslpc login node.
#   submit.sh <13|13p6> <jobs dir> <env tarball> <points per job> MWR_MN [MWR_MN ...]
# The jobs dir receives jobs.txt, submit.jdl and the logs; the outputs come back as out_<energy>_<points>.tgz.
set -e
E=$1; J=$2; ENV=$(readlink -f $3); PER=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $J/logs && cd $J
cp $HERE/kfactor_job.sh $HERE/../run_points.sh $HERE/../analysis_HwU_mlN.f $HERE/../lhe_mln.py .
# one line per job: the points joined by '+' (Condor splits "queue ... from" lines on whitespace as well as commas,
# so no item may contain a space) and the output tag (kfactor_job.sh derives the same tag)
: > jobs.txt
pts=("$@"); i=0
while [ $i -lt ${#pts[@]} ]; do
  p=$(echo "${pts[@]:$i:$PER}" | tr ' ' '+'); echo "$p, $(echo $p | tr '+' '_' | cut -c1-80)" >> jobs.txt; i=$((i + PER))
done
# the LPC condor_submit wrapper insists on a proxy even though these jobs never use the grid
export X509_USER_PROXY=${X509_USER_PROXY:-/uscms/homes/b/bjackson/x509up_u$(id -u)}
cat > submit.jdl <<EOF
universe                = vanilla
executable              = kfactor_job.sh
use_x509userproxy       = true
x509userproxy           = $X509_USER_PROXY
arguments               = "$E 4 \$(points)"
+SingularityImage       = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64"
+SingularityBindCVMFS   = True
request_cpus            = 4
request_memory          = 6000
request_disk            = 15000000
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = run_points.sh, analysis_HwU_mlN.f, lhe_mln.py, $ENV
transfer_output_files   = out_${E}_\$(tag).tgz
output                  = logs/job_\$(Cluster)_\$(Process).out
error                   = logs/job_\$(Cluster)_\$(Process).err
log                     = logs/condor.log
periodic_hold           = (JobStatus == 2) && ((time() - EnteredCurrentStatus) > 36000)
notification            = never
EOF
# KF_JOB_ENV="REQ_ACC=0.001 LO_EVENTS=400000": run_points.sh precision knobs for the jobs
[ -n "$KF_JOB_ENV" ] && echo "environment             = \"$KF_JOB_ENV\"" >> submit.jdl
echo "queue points, tag from jobs.txt" >> submit.jdl
condor_submit submit.jdl
