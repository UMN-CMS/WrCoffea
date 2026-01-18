# WrCoffea Documentation

Welcome to the WR analyzer! This repository provides tools for analyzing and processing WR background, data, and signal events. Below, you’ll find instructions on setting up the environment, running the analysis, and extending the framework.

---

## Quick Start
To make a standard set of histograms, run the analyzer like
```bash
python3 bin/run_analysis.py era sample
```
where `era` is either `RunIISummer20UL18` or `RunIII2024Summer24` and `sample` is one of [`DYJets`, `tt_tW`, `Nonprompt`, `Other`] for backgrounds, [`EGamma`, `Muon`] for data or `Signal` for signal.

---
### Backgrounds
Examples:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIISummer20UL18 Nonprompt
```
To run over all backgrounds, one can use
```bash
bash bin/analyze_all.sh bkg RunIII2024Summer24
```

---
### Data
Similarly to run over data,
```bash
python3 bin/run_analysis.py RunIII2024Summer24 EGamma
python3 bin/run_analysis.py RunIISummer20UL18 Muon
```
Or to run over both `EGamma` and `Muon` in one command
```bash
bash bin/analyze_all.sh data RunIII2024Summer24
```

---
### Signal
To analyze signal files, include the `--mass` flag
```bash
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass 
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass 
```
To see the possible signal points for a given era, run
```bash
python3 bin/run_analysis.py RunIII2024Summer24 --list-masses
```

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIII2024Summer24 tt_tW 
python3 bin/run_analysis.py RunIII2024Summer24 Nonprompt
python3 bin/run_analysis.py RunIII2024Summer24 Other
python3 bin/run_analysis.py RunIII2024Summer24 EGamma
python3 bin/run_analysis.py RunIII2024Summer24 Muon
```
To run over 2024 signals,
```bash
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass 33322?
```
To see avaliable mass options, choose 
```bash
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass 33322?
```
Note that 
## Table of Contents
- [Creating Filesets](docs/filesets.md) – Instructions for creating both skimmed and unskimmed filesets.
- [Running the Analyzer](docs/run_analysis.md) – How to execute `run_analysis.py` to perform a basic analysis.
- [Condor](docs/condor.md) – How to run the analyzer on Condor at the LPC.
- [Plotting](docs/plotting.md) – Instructions for generating plots from histogram ROOT files.
- [Repository Structure](README.md#repository-structure) – Overview of how the repository is organized.
- [Getting Started](README.md#getting-started) – Instructions for installing and setting up the analyzer.
---

## 📂 Repository Structure
This repository is structured to separate executable scripts, core analysis logic, and documentation.

```
WR_Plotter/ # Submodule where ROOT histograms are saved and plotted.
bin/        # Holds the main script for running the analysis.
data/       # Configuration files, all json files, and important logging info are stored here.
docs/       # Contains documentation markdown.
python/     # Includes reusable Python modules.
scripts/    # Contains helper scripts for setup and post-processing.
src/        # Includes the core analysis code.
test/       # Holds test and development scripts.
```

---

## Quick Start
To run over 2024 samples,
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dy NLO_mll_binned --dir output_dir
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dy LO_inclusive --dir output_dir
python3 bin/run_analysis.py RunIII2024Summer24 tt_tW --dir output_dir
python3 bin/run_analysis.py RunIII2024Summer24 Nonprompt --dir output_dir
python3 bin/run_analysis.py RunIII2024Summer24 Other --dir output_dir
python3 bin/run_analysis.py RunIII2024Summer24 EGamma --dir output_dir
python3 bin/run_analysis.py RunIII2024Summer24 Muon --dir output_dir
```
To view 2024 skims
```bash
gfal-ls -l davs://cmsxrootd.hep.wisc.edu:1094/store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24
```
To add skims to Wisconsin storage area
```bash
xrdcp -r some_dataset root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims/RunII/2018/RunIISummer20UL18
```

## Getting Started
Begin by cloning the repository:
```bash
git clone git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
```
Create and source a virtual Python environment:
```bash
python3 -m venv wr-env
source wr-env/bin/activate
```
Install the required packages:
```bash
python3 -m pip install -r requirements.txt
```

### Grid UI
To authenticate for accessing grid resources, use:
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

### ROOT
To enable ROOT functionality, source the appropriate LCG release:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```
If using UMN’s setup, use:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
```

## Update Plotter Submodule
To update the commit of the WR_Plotter submodule, use the following commands
```
cd WR_Plotter
git switch main
git pull
cd ..
git commit -am "Pulled down update to WR_Plotter"
git push
```
