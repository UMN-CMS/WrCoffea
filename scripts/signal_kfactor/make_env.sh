#!/bin/bash
# Build MG5_aMC_env_<version>.tgz for one energy on a cmslpc login node: the MadGraph directory with the
# EffLRSM_NLO model and Ninja installed, so Condor jobs skip the (network-dependent) tool installation.
#   make_env.sh <13|13p6> <build dir>     (needs MG5_aMC_v<version>.tar.gz, run_points.sh, analysis_HwU_mlN.f there)
set -e
E=$1; B=$(readlink -f $2); cd $B
cat > make_env_inner.sh <<EOF
cd $B
bash run_points.sh $E 2            # no points: sets MG up, outputs the processes (installs Ninja on the way)
EOF
/cvmfs/cms.cern.ch/common/cmssw-cc7 --bind /uscms_data --bind /uscms/homes -- bash $B/make_env_inner.sh
MGV=$([ "$E" = 13 ] && echo 2.6.5 || echo 2.9.18)
MGDIR=MG5_aMC_v${MGV//./_}
[ -f $MGDIR/HEPTools/lib/libninja.a ] || { echo "no Ninja in $MGDIR"; exit 1; }
tar -czf MG5_aMC_env_${MGV}.tgz --exclude="$MGDIR/HEPTools/HEPToolsInstallers" $MGDIR
ls -la MG5_aMC_env_${MGV}.tgz
