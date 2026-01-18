# WrCoffea Documentation

Welcome to the WR analyzer! This repository provides tools for analyzing and processing WR background, data, and signal events. Below, you’ll find instructions on setting up the environment, running the analysis, and extending the framework.

---

## Table of Contents
- [Quickstart](README.md#quick-start) – Basic commands for running the analyzer.
- [Repository Structure](README.md#repository-structure) – Overview of how the repository is organized.
- [Getting Started](README.md#getting-started) – Instructions for installing and setting up the analyzer.
- [Plotting](docs/plotting.md) – Make stack plots from the analyzer's output `ROOT` files.
- [Running the Analyzer](docs/run_analysis.md) – Useful options to include when running the analyzer.
- [Creating Filesets](docs/filesets.md) – Instructions for creating filesets.
- [Condor](docs/condor.md) – How to run the analyzer on Condor at the LPC. [IN PROGRESS]

---

## Quick Start
To make a standard set of histograms, run the analyzer with a command of the form
```bash
python3 bin/run_analysis.py era sample
```
To see the list of eras, run
```bash
python3 bin/run_analysis.py --list-eras
```
To see the list of samples, run
```bash
python3 bin/run_analysis.py --list-samples
```

---
### Backgrounds
Examples:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIISummer20UL18 Nonprompt
```
To run over all backgrounds for a given era,
```bash
bash bin/analyze_all.sh bkg RunIII2024Summer24
```

---
### Data
Analyzing data is similar,
```bash
python3 bin/run_analysis.py RunIII2024Summer24 EGamma
python3 bin/run_analysis.py RunIISummer20UL18 Muon
```
Or to run over both `EGamma` and `Muon` in one command,
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
To see the possible signal points, run either
```bash
python3 bin/run_analysis.py --list-masses
python3 bin/run_analysis.py RunIII2024Summer24 --list-masses
```

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
