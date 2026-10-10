#!/bin/bash
# WR signal NLO/LO k-factor (AN-19-083 App. A method) for a list of (mWR, mN) points at one energy.
#
#   run_points.sh <13|13p6> <ncores> MWR_MN [MWR_MN ...]
#
# Runs inside the cc7 container (cmssw-cc7) with, in the working directory, MG5_aMC_v<version>.tar.gz and
# analysis_HwU_mlN.f.  The model (EffLRSM_NLO), the param_card and the LO run_card come from the official CMS
# gridpack of that energy (the samples we normalise), so only the masses and the W_R width change per point.
# Per point it writes out/<energy>_WR<m>_N<n>/:
#   nlo.HwU     fixed-order NLO: total rate and m(lN), for NLO and its Born part (9 scale weights)
#   nlo.txt     the NLO run summary (cross section and its uncertainty)
#   lo.json     the LO cross section with the official LO run_card (madevent) and its m(lN) histogram (LHE)
set -e
E=$1; NCORE=$2; shift 2
WORK=$PWD
G=/cvmfs/cms.cern.ch/phys_generator/gridpacks
GP13=$G/RunII/13TeV/MadGraph5_aMCatNLO/v265/WRtoNLtoLLJJ_emu/WRtoNMutoMuMuJJ_MWR-2000_MN-100_13TeV_slc7_amd64_gcc700_CMSSW_10_6_19_tarball.tar.xz
GP136=$G/RunIII/13p6TeV/MadGraph5_aMCatNLO/v2918/WRtoNLtoLLJJ_mu/WRtoNMutoMuMuJJ_MWR-2000_MN-100_13p6TeV_slc7_amd64_gcc10_CMSSW_12_4_8_tarball.tar.xz
# Both energies run MG 2.9.18 (CMSSW_12_4_8, python3).  The 13 TeV samples were made with 2.6.5, whose NLO code
# cannot find pdfsets.index with the CMSSW_10_6 LHAPDF whatever path it is given; MG13=2.6.5 keeps that route.
# The LO run with the official card reproduces the 13 TeV gridpack integral (results.dat) at the per-mille level,
# which is the check that the LO denominator is the samples' own normalisation.
if [ "$E" = 13 ]; then
  EBEAM=6500; GP=$GP13
  if [ "${MG13:-2.9.18}" = 2.6.5 ]; then REL=CMSSW_10_6_19; ARCH=slc7_amd64_gcc700; MGV=2.6.5; PY=python2
  else REL=CMSSW_12_4_8; ARCH=slc7_amd64_gcc10; MGV=2.9.18; PY=python3; fi
else
  REL=CMSSW_12_4_8; ARCH=slc7_amd64_gcc10; MGV=2.9.18; EBEAM=6800; PY=python3; GP=$GP136
fi
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=$ARCH
[ -d $REL ] || scram project $REL > /dev/null
cd $REL/src && eval "$(scram runtime -sh)" && cd $WORK
# aMC@NLO looks PDF ids up in pdfsets.index, which the CMSSW_10_6 LHAPDF data directory lacks, and MG sets
# LHAPDF_DATA_PATH itself from `lhapdf-config --datadir`: give MG a wrapper whose --datadir holds the sets and an index.
# KF_LHAPDF_CONFIG=<lhapdf-config>: use that LHAPDF build instead of CMSSW's (MG 2.6.5's compiled PDF code needs an
# LHAPDF whose own data directory holds pdfsets.index; a local 6.2.3 build does, docs/signal_kfactor.md)
LHAPDF_REAL=${KF_LHAPDF_CONFIG:-$(which lhapdf-config)}
if [ -n "$KF_LHAPDF_CONFIG" ]; then
  export LD_LIBRARY_PATH=$($LHAPDF_REAL --libdir):$LD_LIBRARY_PATH PATH=$(dirname $LHAPDF_REAL):$PATH
fi
LHAPDF_CONFIG=$LHAPDF_REAL
DATADIR=$($LHAPDF_REAL --datadir)
if [ ! -f $DATADIR/pdfsets.index ]; then
  mkdir -p $WORK/lhapdf_data $WORK/bin
  for f in $DATADIR/*; do ln -sfn $f $WORK/lhapdf_data/; done
  IDX=$(ls /cvmfs/cms.cern.ch/lhapdf/pdfsets/current/pdfsets.index \
           /cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/lhapdf/*/share/LHAPDF/pdfsets.index 2>/dev/null | head -1)
  [ -n "$IDX" ] || { echo "no pdfsets.index found on cvmfs"; exit 1; }
  cp $IDX $WORK/lhapdf_data/pdfsets.index
  printf '#!/bin/bash\nif [ "$1" = "--datadir" ]; then echo %s; else exec %s "$@"; fi\n' $WORK/lhapdf_data $LHAPDF_REAL > $WORK/bin/lhapdf-config
  chmod +x $WORK/bin/lhapdf-config
  LHAPDF_CONFIG=$WORK/bin/lhapdf-config
fi
export LHAPDF_DATA_PATH=$($LHAPDF_CONFIG --datadir)
MG=$WORK/MG5_aMC_v${MGV//./_}
# MG5_aMC_env_<version>.tgz (make_env.sh): the MG directory with the model and Ninja already installed, for Condor
if [ ! -d $MG ]; then
  if [ -f MG5_aMC_env_${MGV}.tgz ]; then tar -xzf MG5_aMC_env_${MGV}.tgz; else tar -xzf MG5_aMC_v${MGV}.tar.gz; fi
fi
mkdir -p gp && tar -xJf $GP -C gp mgbasedir/models/EffLRSM_NLO process/madevent/Cards/param_card.dat \
    process/madevent/Cards/run_card.dat
[ -d $MG/models/EffLRSM_NLO ] || cp -r gp/mgbasedir/models/EffLRSM_NLO $MG/models/
mgcfg() {   # <key> <value>: one definition per key in mg5_configuration.txt (a duplicate makes MG read a string)
  sed -i "/^#\{0,1\} *$1 *=/d" $MG/input/mg5_configuration.txt
  echo "$1 = $2" >> $MG/input/mg5_configuration.txt
}
mgcfg lhapdf $LHAPDF_CONFIG
mgcfg run_mode 2
mgcfg nb_core $NCORE
mgcfg automatic_html_opening False
mgcfg auto_update 0
[ -f $MG/HEPTools/lib/libninja.a ] && mgcfg ninja $MG/HEPTools/lib
# MG 2.6.5's generated PDF wrapper replaces LHAPDF's search path by the process-local lib/PDFsets, which MG recreates
# at every launch with the set but no pdfsets.index (the id lookup fails with the CMSSW_10_6 LHAPDF): keep the default
# paths behind the local one.  2.9.18 no longer calls setPaths; the sed is a no-op there.
for f in $MG/Template/NLO/Source/PDF/pdf_lhapdf62.cc $MG/Template/LO/Source/PDF/pdf_lhapdf62.cc \
         $WORK/NLOPROC/Source/PDF/pdf_lhapdf62.cc $WORK/LOPROC/Source/PDF/pdf_lhapdf62.cc; do
  [ -f $f ] && sed -i 's/LHAPDF::setPaths(/LHAPDF::pathsPrepend(/' $f
done
# ... and MG wipes lib/PDFsets at every launch and copies only the set: make copy_lhapdf_set copy pdfsets.index too
$(which python3 || echo $PY) - $MG/madgraph/interface/common_run_interface.py <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "                    files.cp(pjoin(pdfsets_dir, pdfset), pjoin(self.me_dir, 'lib', 'PDFsets'))\n"
add = "                    files.cp(pjoin(pdfsets_dir, 'pdfsets.index'), pjoin(self.me_dir, 'lib', 'PDFsets'))\n"
if old in s and add not in s:
    open(p, "w").write(s.replace(old, old + add))
EOF
# the MG 2.6.5 LO PDF wrapper includes Boost headers: CMSSW_10_6 has them as an external (make_opts honours CXXFLAGS)
BOOSTINC=$(cd $REL/src && scram tool tag boost INCLUDE 2>/dev/null || true)    # scram needs the project area as cwd
[ -n "$BOOSTINC" ] && [ -f $BOOSTINC/boost/shared_ptr.hpp ] && export CXXFLAGS="-O -I$BOOSTINC"

# MG5_aMC's "output" of an NLO process installs the one-loop reduction library Ninja through the HEPToolsInstaller
# it downloads; the current installer (V168) crashes on MG 2.6.5 (an unguarded version comparison on None).  Patch
# the guard, install Ninja by hand into $MG/HEPTools and point the configuration at it, then output again.
mg_output() {   # <mg script> <output dir>
  $PY $MG/bin/mg5_aMC $1 > ${2}_output.log 2>&1 || true
  if [ ! -d $2 ] && grep -q "Installation of ninja failed" ${2}_output.log; then
    I=$MG/HEPTools/HEPToolsInstallers/HEPToolInstaller.py
    sed -i 's/LooseVersion("5.3.5.13") <= _mg5_version < LooseVersion("5.3.6.0")/(_mg5_version is not None and LooseVersion("5.3.5.13") <= _mg5_version < LooseVersion("5.3.6.0"))/' $I
    # the installer itself needs python3 (urllib.request), also when MG runs under python2 (CMSSW_10_6_X has 3.6)
    (cd $(dirname $I) && $(which python3 || echo $PY) HEPToolInstaller.py ninja --prefix=$MG/HEPTools --mg5_path=$MG \
        --fortran_compiler=gfortran --cpp_compiler=g++ > $MG/HEPTools/ninja_manual.log 2>&1)
    [ -f $MG/HEPTools/lib/libninja.a ] || { echo "ninja installation failed (see $MG/HEPTools/ninja_manual.log)"; exit 1; }
    sed -i '/^ninja /d' $MG/input/mg5_configuration.txt
    echo "ninja = $MG/HEPTools/lib" >> $MG/input/mg5_configuration.txt
    rm -rf $2
    $PY $MG/bin/mg5_aMC $1 > ${2}_output2.log 2>&1 || true
  fi
  [ -d $2 ] || { echo "MG5_aMC output of $2 failed"; exit 1; }
  # MG copies the PDF set, but not pdfsets.index, into the process directory's lib/PDFsets, the only path the
  # compiled code searches with the CMSSW_10_6 LHAPDF: the id -> set lookup needs the index there
  mkdir -p $2/lib/PDFsets && cp $DATADIR/pdfsets.index $2/lib/PDFsets/
}

# Processes: N_mu + mu at NLO QCD and at LO (the k-factor is a QCD correction to W_R production; the N decay and
# the lepton flavour do not enter).  5-flavour proton as in the official samples.
if [ ! -d NLOPROC ]; then
  cat > nlo.mg <<EOF
import model EffLRSM_NLO
define p = 21 2 4 1 3 -2 -4 -1 -3 5 -5
define j = p
generate p p > n2 mu+ [QCD]
add process p p > n2 mu- [QCD]
output NLOPROC
EOF
  mg_output nlo.mg NLOPROC
  cp analysis_HwU_mlN.f NLOPROC/FixedOrderAnalysis/
  sed -i 's/^ *FO_ANALYSE *=.*/FO_ANALYSE = analysis_HwU_mlN.o/' NLOPROC/Cards/FO_analyse_card.dat
  cp gp/process/madevent/Cards/param_card.dat NLOPROC/Cards/param_card.dat
fi
if [ ! -d LOPROC ]; then
  cat > lo.mg <<EOF
import model EffLRSM_NLO
define p = 21 2 4 1 3 -2 -4 -1 -3 5 -5
define j = p
generate p p > n2 mu+
add process p p > n2 mu-
output LOPROC
EOF
  mg_output lo.mg LOPROC
  cp gp/process/madevent/Cards/run_card.dat LOPROC/Cards/run_card.dat
  cp gp/process/madevent/Cards/param_card.dat LOPROC/Cards/param_card.dat
fi

mkdir -p out
for pt in "$@"; do
  M=${pt%_*}; N=${pt#*_}; TAG=WR${M}_N${N}; OUT=out/${E}_${TAG}; mkdir -p $OUT
  # --- NLO, fixed order, the default NLO scale (H_T/2) with the 9-point scale variation
  cat > nlo_$TAG.mg <<EOF
launch NLOPROC -n $TAG
fixed_order = ON
done
set param_card mass 34 $M
set param_card mass 9900014 $N
set param_card decay 34 auto
set run_card lhaid 325300
set run_card pdlabel lhapdf
set run_card ebeam1 $EBEAM
set run_card ebeam2 $EBEAM
set run_card req_acc_FO ${REQ_ACC:-0.002}
set run_card reweight_scale True
done
EOF
  t0=$(date +%s)
  $PY $MG/bin/mg5_aMC nlo_$TAG.mg > $OUT/nlo.log 2>&1 || echo "NLO $TAG failed" >> $OUT/errors.txt
  cp NLOPROC/Events/$TAG/MADatNLO.HwU $OUT/nlo.HwU 2>/dev/null || true
  cp NLOPROC/Events/$TAG/summary.txt $OUT/nlo.txt 2>/dev/null || grep -i -A3 "Total cross section" $OUT/nlo.log > $OUT/nlo.txt || true
  t1=$(date +%s)
  # --- LO with the official run_card (madevent default scale), the cross section our samples carry
  cat > lo_$TAG.mg <<EOF
launch LOPROC -n $TAG
done
set param_card mass 34 $M
set param_card mass 9900014 $N
set param_card decay 34 auto
set run_card gridpack False
set run_card nevents ${LO_EVENTS:-100000}
set run_card iseed $((M + N))
done
EOF
  $PY $MG/bin/mg5_aMC lo_$TAG.mg > $OUT/lo.log 2>&1 || echo "LO $TAG failed" >> $OUT/errors.txt
  $(which python3 || echo $PY) lhe_mln.py $OUT/lo.json LOPROC/Events/$TAG
  t2=$(date +%s)
  echo "$TAG nlo $((t1 - t0)) s lo $((t2 - t1)) s" >> out/timing.txt
  rm -rf NLOPROC/Events/$TAG LOPROC/Events/$TAG
done
