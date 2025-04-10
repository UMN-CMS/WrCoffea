# WrCoffea Documentation

Welcome to the WR analyzer! This repository provides tools for analyzing and processing WR background, data, and signal events. Below, you’ll find instructions on setting up the environment, running the analysis, and extending the framework.

## Table of Contents
- [Running the Analyzer](docs/run_analysis.md) – How to execute `run_analysis.py` to perform the analysis.
- [Workflow Overview](docs/workflow.md) – A detailed guide on the full data processing pipeline.
- [Plotting](docs/plotting.md) – Instructions for generating plots from histogram ROOT files.
- [Repository Structure](README.md#repository-structure) – Overview of how the repository is organized.
- [Getting Started](README.md#getting-started) – Instructions for installing and setting up the analyzer.
- [Extending the analyzer](README.md#extending-the-analyzer) – Guidelines for contributing to the analyzer
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

### Adding a New Study
1. Copy an existing script from `src/`:
   ```bash
   cp src/analyzer.py test/dev_analyzer.py
   ```
2. Modify the script to include your custom selections and histograms.
3. Once finalized, integrate your study into the main pipeline via `bin/`, `python/`, or `src/`.

---
