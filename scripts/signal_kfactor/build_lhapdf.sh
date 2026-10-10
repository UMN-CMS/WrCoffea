#!/bin/bash
# As run on the LPC login node on 2026-10-10 from ~/nobackup/kfactor/lhapdf_build (paths hard-coded): the
# CMSSW_10_6_19 project area gives gcc 7; the PDF sets are symlinked from cvmfs into the install's share dir.
# Build LHAPDF 6.2.3 with the CMSSW_10_6_19 (gcc 7) toolchain for MG5_aMC 2.6.5's NLO mode; runs inside cmssw-cc7.
set -x
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
cd ~/nobackup/kfactor/test_job13/CMSSW_10_6_19/src && eval "$(scram runtime -sh)" && cd ~/nobackup/kfactor/lhapdf_build
which gcc g++ gfortran; gcc --version | head -1
PREFIX=$HOME/nobackup/kfactor/lhapdf-6.2.3
[ -f LHAPDF-6.2.3.tar.gz ] || curl -L --max-time 300 -o LHAPDF-6.2.3.tar.gz "https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.2.3.tar.gz"
ls -la LHAPDF-6.2.3.tar.gz && tar -xzf LHAPDF-6.2.3.tar.gz && cd LHAPDF-6.2.3
./configure --prefix=$PREFIX --disable-python > configure.log 2>&1 || { tail -20 configure.log; exit 1; }
make -j4 > make.log 2>&1 || { tail -30 make.log; exit 1; }
make install > install.log 2>&1 || { tail -20 install.log; exit 1; }
$PREFIX/bin/lhapdf-config --version --datadir --libdir
ls $PREFIX/share/LHAPDF/
for s in NNPDF31_nnlo_as_0118_mc_hessian_pdfas NNPDF31_nnlo_as_0118 NNPDF23_nlo_as_0119 NNPDF31_nlo_as_0118; do
  [ -d /cvmfs/cms.cern.ch/lhapdf/pdfsets/current/$s ] && ln -sfn /cvmfs/cms.cern.ch/lhapdf/pdfsets/current/$s $PREFIX/share/LHAPDF/$s
done
ls -la $PREFIX/share/LHAPDF/ | head; grep -c . $PREFIX/share/LHAPDF/pdfsets.index
echo BUILD_DONE
