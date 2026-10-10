#!/bin/bash
# LO-only variants of one point on a prepared directory (LOPROC exists): which run_card setting shapes the m(lN) tails?
#   lo_diag.sh <13|13p6> MWR_MN "<label> <run_card settings...>" ...     e.g. "bw1000 bwcutoff 1000"
set -e
E=$1; PT=$2; shift 2; M=${PT%_*}; N=${PT#*_}
if [ "$E" = 13 ]; then REL=CMSSW_10_6_19; ARCH=slc7_amd64_gcc700; MGV=2.6.5; PY=python2; else REL=CMSSW_12_4_8; ARCH=slc7_amd64_gcc10; MGV=2.9.18; PY=python3; fi
source /cvmfs/cms.cern.ch/cmsset_default.sh; export SCRAM_ARCH=$ARCH; W=$PWD
cd $REL/src && eval "$(scram runtime -sh)" && cd $W
MG=$W/MG5_aMC_v${MGV//./_}
mkdir -p diag
for spec in "$@"; do
  set -- $spec; L=$1; shift
  SETS=""; while [ $# -gt 1 ]; do SETS="$SETS
set run_card $1 $2"; shift 2; done
  cat > diag/lo_$L.mg <<EOF
launch LOPROC -n diag_$L
done
set param_card mass 34 $M
set param_card mass 9900014 $N
set param_card decay 34 auto
set run_card gridpack False
set run_card nevents ${LO_EVENTS:-20000}
set run_card iseed 12345$SETS
done
EOF
  $PY $MG/bin/mg5_aMC diag/lo_$L.mg > diag/lo_$L.log 2>&1 || true
  $(which python3 || echo $PY) lhe_mln.py diag/lo_$L.json LOPROC/Events/diag_$L
  grep "Cross-section :" diag/lo_$L.log | tail -1
  rm -rf LOPROC/Events/diag_$L
done
echo DIAG_DONE
