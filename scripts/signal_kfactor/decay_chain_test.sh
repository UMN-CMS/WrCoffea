#!/bin/bash
# As run on the LPC login node on 2026-10-10 from ~/nobackup/kfactor/test_job136 (paths hard-coded; the MG 2.9.18
# directory and the extracted gridpack cards `gp/` there come from an earlier run_points.sh invocation).
# Result: 0.1673 +- 0.0005 pb (bwcutoff 15, the sample's value) vs 0.1708 +- 0.0006 pb (bwcutoff 1000).
# LO cross section of the official gridpack's own process (p p > nn ll, nn > ll jj jj) with the official cards,
# bwcutoff 15 (the gridpack value) and 1000, MG 2.9.18 at 13.6 TeV, WR2000_N100: is the 1.8 % between our
# production-only LO and the sample's LO the Breit-Wigner truncation of the decay chain?
set -x
source /cvmfs/cms.cern.ch/cmsset_default.sh; export SCRAM_ARCH=slc7_amd64_gcc10
cd ~/nobackup/kfactor/test_job136/CMSSW_12_4_8/src && eval "$(scram runtime -sh)"; cd ~/nobackup/kfactor/test_job136
export LHAPDF_DATA_PATH=$(lhapdf-config --datadir)
MG=MG5_aMC_v2_9_18
grep -E "^(lhapdf|run_mode|nb_core) " $MG/input/mg5_configuration.txt
if [ ! -d DECPROC ]; then
cat > dec.mg <<MGEOF
import model EffLRSM_NLO
define p = 21 2 4 1 3 -2 -4 -1 -3 5 -5
define j = p
define ll = mu+ mu-
define nn = n2
define jj = u d c s b u~ d~ c~ s~ b~
generate p p > nn ll, nn > ll jj jj
output DECPROC
MGEOF
python3 $MG/bin/mg5_aMC dec.mg > DECPROC_output.log 2>&1
fi
[ -d DECPROC ] || { echo "output failed"; tail -20 DECPROC_output.log; exit 1; }
cp gp/process/madevent/Cards/run_card.dat DECPROC/Cards/run_card.dat
cp gp/process/madevent/Cards/param_card.dat DECPROC/Cards/param_card.dat      # the WR2000_N100 card itself, width untouched
for BW in 15 1000; do
cat > dec_bw$BW.mg <<MGEOF
launch DECPROC -n bw$BW
done
set run_card gridpack False
set run_card nevents 10000
set run_card bwcutoff $BW
set run_card iseed 2100
done
MGEOF
python3 $MG/bin/mg5_aMC dec_bw$BW.mg > dec_bw$BW.log 2>&1
echo "bwcutoff $BW: $(grep -i -E 'Cross-section' dec_bw$BW.log | tail -1)"
done
echo DEC_DONE
