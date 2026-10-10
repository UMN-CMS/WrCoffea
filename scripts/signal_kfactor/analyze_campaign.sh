#!/bin/bash
# After the Condor campaign: fetch the job outputs from the LPC, run the analysis for both energies, write the
# k(m_lN) payloads.   analyze_campaign.sh <local work dir>
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
W=$(readlink -f ${1:-$HOME/work/skims_v2/_kfactor}); mkdir -p $W; cd $W
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=30"; SSH="ssh $SSH_OPTS"
# every campaign of an energy (jobs_<E>, jobs_<E>_hi, jobs_<E>_grid, ...): later tarballs overwrite earlier points
for E in 13 13p6; do
  for J in $(ssh $SSH_OPTS cmslpc "ls -d nobackup/kfactor/jobs_${E} nobackup/kfactor/jobs_${E}_* 2>/dev/null"); do
    D=$(basename $J); mkdir -p $D
    rsync -a -e "$SSH" "cmslpc:$J/out_${E}_*.tgz" $D/ 2>/dev/null || true
    rsync -a -e "$SSH" "cmslpc:$J/logs/" $D/logs/ 2>/dev/null || true
  done
  mkdir -p out_$E
  for t in $(ls -tr jobs_${E}/out_${E}_*.tgz jobs_${E}_*/out_${E}_*.tgz 2>/dev/null); do tar -xzf $t -C out_$E --strip-components=1; done
  echo "$E: $(ls -d out_$E/${E}_WR* | wc -l) points"
done
rsync -a -e "$SSH" cmslpc:nobackup/kfactor/gp_integrals.json . 2>/dev/null || true
sed -i.bak '/GP_DONE/d' gp_integrals.json && rm -f gp_integrals.json.bak     # gp_integrals.sh ends the file with a marker
# one out/ tree per energy for the analysis (it globs <energy>_WR*_N*)
# 13 TeV: the 490 private UL18 points (no runs) get interpolated direct k and scaled shapes
for E in 13 13p6; do
  EXTRA=""; [ $E = 13 ] && [ -f ul18_private_points.txt ] && EXTRA="--interpolate-to ul18_private_points.txt"
  $REPO/.venv/bin/python $REPO/scripts/signal_kfactor/kfactor_analysis.py --out out_$E --energy $E --plots plots \
      --run2-inclusive $HOME/work/wr-references/HNWR_Plotter/script/PrintKFactor/result.txt \
      --gridpack-integrals gp_integrals.json --sample-lo sample_lo.json --lo-norm auto --payload payload_$E.json $EXTRA | tee analysis_$E.txt
done
