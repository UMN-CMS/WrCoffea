# WrCoffea Documentation

This repository provides the main analysis framework for processing WR→Nℓ→ℓℓjj events using the Coffea columnar analysis toolkit. It handles background, data, and signal samples to produce histograms for downstream limit-setting and plotting.

## Table of Contents
- [Quick Start](#quick-start) – Run the analyzer
- [Skimming](#skimming) – Skim NanoAOD files for faster analysis
- [Running on Condor](#running-on-condor) – Scale out with HTCondor at LPC
- [Command Reference](#command-reference) – Complete flag reference and examples
- [Repository Structure](#-repository-structure) – Overview of how the codebase is organized
- [Getting Started](#getting-started) – Installation and environment setup
- [Testing](#testing) – Running the automated test suite
- [Additional Documentation](#additional-documentation) – Links to detailed guides

---

## Quick Start

First, create filesets (see [filesets.md](docs/filesets.md)). Then run the analyzer by specifying an era and sample:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets                          # background
python3 bin/run_analysis.py RunIII2024Summer24 EGamma                          # data
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100      # signal
bash bin/analyze_all.sh all RunIII2024Summer24                                 # everything
```

Output ROOT histograms are saved to `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/`.

See **[Running the Analyzer](docs/run_analysis.md)** for full details: all samples, output customization, region selection, systematics, and batch processing.

---

### Skimming

The skimmer applies a loose event preselection to NanoAOD files, reducing file sizes for faster analysis iteration. It uses `bin/skim.py` with subcommands for the full workflow: skim locally or on Condor, check for failures, and merge outputs.

```bash
python3 bin/skim.py --cuts                                           # show skim cuts
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM                    # submit all to Condor
python3 bin/skim.py check /TTto2L2Nu_.../NANOAODSIM                  # check for failures
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM                  # extract + hadd + validate
```

See **[Skimming](docs/skimming.md)** for full documentation: selection cuts, all subcommand flags, output layout, and architecture.

---

### Running on Condor

Scale out processing across many workers at FNAL LPC using HTCondor with the Dask executor. Requires the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue) Apptainer environment.

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12           # enter container
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor        # run with Condor
bash bin/analyze_all.sh all RunIII2024Summer24 --condor              # run everything
```

See **[Running on Condor](docs/condor.md)** for full documentation: setup, worker/chunksize defaults, tmux tips, and log locations.

---

## Command Reference

### run_analysis.py Flags

| Flag | Arguments | Description |
|------|-----------|-------------|
| `era` | `<era_name>` | **Required positional.** Campaign to analyze (e.g., RunIII2024Summer24) |
| `sample` | `<sample_name>` | **Required positional.** Sample to analyze (e.g., DYJets, Signal, EGamma) |
| `--mass` | `<mass_point>` | Signal mass point (e.g., WR4000_N2100). **Required for Signal sample** |
| `--region` | `resolved\|boosted\|both` | Analysis region to run (default: both) |
| `--dy` | `LO_inclusive\|NLO_mll_binned\|LO_HT` | Specific DY sample variant (only valid for DYJets) |
| `--dir` | `<directory>` | Create output subdirectory under rootfiles path |
| `--name` | `<suffix>` | Append suffix to output ROOT filename |
| `--debug` | | Run without saving histograms (for testing) |
| `--reweight` | `<json_file>` | Path to DY reweight JSON file (DYJets only) |
| `--unskimmed` | | Use unskimmed filesets instead of default skimmed files |
| `--condor` | | Submit jobs to HTCondor at LPC (requires Apptainer shell, see [Running on Condor](#running-on-condor)) |
| `--max-workers` | `<int>` | Number of Dask workers (local default: 6, condor default: 50, analyze_all.sh: 50 skimmed / 400 unskimmed for data/bkg, 10 for signal) |
| `--chunksize` | `<int>` | Number of events per processing chunk (default: 250000, analyze_all.sh: 50000 for data/signal) |
| `--threads-per-worker` | `<int>` | Threads per Dask worker for local runs |
| `--systs` | `lumi` `pileup` `sf` | Enable systematic variations (see [Systematics](#systematics)) |
| `--list-eras` | | Print available eras and exit |
| `--list-samples` | | Print available samples and exit |
| `--list-masses` | | Print available signal mass points and exit |
| `--preflight-only` | | Validate fileset and exit without processing |

### Examples

```bash
# Basic background processing
python3 bin/run_analysis.py RunIII2024Summer24 DYJets

# Signal with specific mass point
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100

# Only process resolved region (faster)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved

# Custom output directory and filename
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir my_study --name test

# Debug mode (no histogram output)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --debug

# Process with systematics (all supported)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf

# Only pileup uncertainty
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs pileup

# Validate fileset without processing
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only

# Process all backgrounds
bash bin/analyze_all.sh bkg RunIII2024Summer24

# Process all data
bash bin/analyze_all.sh data RunIII2024Summer24

# Process signal mass points
bash bin/analyze_all.sh signal RunIII2024Summer24

# analyze_all.sh with custom directory
bash bin/analyze_all.sh bkg RunIII2024Summer24 --dir my_study --name test

# Run on Condor (must be inside Apptainer shell)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 400

# Run all samples on Condor (uses optimized worker/chunksize defaults)
bash bin/analyze_all.sh all RunIII2024Summer24 --condor

# Run with systematics (applied to MC only, filtered for data)
bash bin/analyze_all.sh all RunIII2024Summer24 --condor --systs lumi pileup sf
```

---

## 📂 Repository Structure

The repository follows a clean architecture separating executable scripts, core analysis logic, configuration, and documentation.

### Directory Overview

```
WR_Plotter/  # Submodule for plotting ROOT histograms
bin/         # User-facing CLI scripts (production workflows)
wrcoffea/    # Installable Python package (analysis code, utilities, config)
data/        # Configuration files (JSON, CSV) and metadata
docs/        # Documentation (markdown guides)
scripts/     # Helper scripts for setup and post-processing
tests/       # Automated test suite (pytest)
test/        # Development and validation scripts
```

### Key Directories

**`bin/`** - Production Scripts
- [`run_analysis.py`](bin/run_analysis.py) - Main analysis driver script
- [`analyze_all.sh`](bin/analyze_all.sh) - Batch processing wrapper for multiple samples
- [`skim.py`](bin/skim.py) - Skimming pipeline (`run`, `check`, `merge` subcommands)
- [`skim_job.sh`](bin/skim_job.sh) - Condor worker shell script for skimming

**`wrcoffea/`** - Installable Python Package
- [`analyzer.py`](wrcoffea/analyzer.py) - Main Coffea processor implementing WR→Nℓ→ℓℓjj analysis (object selection, resolved/boosted regions, histogram filling, cutflows)
- [`histograms.py`](wrcoffea/histograms.py) - Histogram specification, creation, and filling
- [`scale_factors.py`](wrcoffea/scale_factors.py) - Lepton scale factor evaluation (correctionlib)
- [`analysis_config.py`](wrcoffea/analysis_config.py) - Centralized configuration (luminosities, correction paths, selection names, cuts)
- [`cli_utils.py`](wrcoffea/cli_utils.py) - CLI plumbing: fileset loading, sample validation, mass point handling
- [`era_utils.py`](wrcoffea/era_utils.py) - Era/year/run mapping and JSON I/O
- [`fileset_utils.py`](wrcoffea/fileset_utils.py) - Fileset path construction, config parsing, JSON writing
- [`fileset_validation.py`](wrcoffea/fileset_validation.py) - Schema and selection validation for filesets
- [`save_hists.py`](wrcoffea/save_hists.py) - ROOT histogram serialization
- [`skimmer.py`](wrcoffea/skimmer.py) - Skim selection, Runs tree handling, single-file skimming
- [`skim_merge.py`](wrcoffea/skim_merge.py) - Post-skim merging, HLT grouping, hadd, validation
- [`das_utils.py`](wrcoffea/das_utils.py) - DAS dataset path validation, dasgoclient queries, XRootD URL construction

**`data/`** - Configuration and Metadata
- `configs/` - Per-era dataset configurations (JSON format, input to fileset scripts)
- `filesets/` - Per-era NanoAOD file lists (JSON format, output of fileset scripts)
- `signal_points/` - Available signal mass points per era (CSV format)
- `lumis/` - Golden JSON lumi masks for data
- `jsonpog/` - Correction payloads for scale factors (correctionlib)

**`WR_Plotter/`** - Plotting Submodule
- Separate repository for ROOT histogram plotting
- Output histograms from this analyzer are saved here
- See [`WR_Plotter/README.md`](WR_Plotter/README.md) for plotting documentation

**`scripts/`** - Helper Scripts
- Fileset creation, preprocessing, and validation tools
- Post-processing and analysis utilities

**`docs/`** - Documentation
- [`run_analysis.md`](docs/run_analysis.md) - Detailed analysis options and workflows
- [`filesets.md`](docs/filesets.md) - Fileset creation instructions
- [`skimming.md`](docs/skimming.md) - Skimming pipeline: cuts, subcommands, Condor jobs, output layout
- [`condor.md`](docs/condor.md) - HTCondor setup, worker defaults, tmux, logs
- [`run_combine.md`](docs/run_combine.md) - Limit-setting with Combine framework

**`tests/`** - Automated Test Suite
- Unit tests for utilities, config consistency, and validation logic
- Run with pytest (see [Testing](#testing))

**`test/`** - Development Scripts
- Analysis validation and optimization studies
- Debugging and testing utilities

---

## Getting Started

### Clone the Repository

Clone with submodules to include the WR_Plotter:
```bash
git clone --recursive git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
```

If you already cloned without `--recursive`, initialize the submodule:
```bash
git submodule update --init --recursive
```

### Environment Setup

There are two ways to set up your environment depending on whether you need Condor submission. Both use **Python 3.12** and **coffea 2025.12.0** to ensure local and Condor environments match.

#### Option A: Local runs (no Condor)

Requires Python 3.12 (available on FNAL LPC via CVMFS, since the system Python is 3.9). Create and activate a virtual environment, then install the package:
```bash
/cvmfs/sft.cern.ch/lcg/releases/Python/3.12.11-531c6/x86_64-el9-gcc13-opt/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -e .
```

> **Note:** The `--system-site-packages` flag is required so the venv can access XRootD Python bindings from CVMFS. The venv must be named `.venv` so that `analyze_all.sh` can auto-detect it. If you already have a `.venv` without `wrcoffea` installed, activate it and run `pip install -e .`.

#### Option B: Condor runs at FNAL LPC (recommended for production)

Set up the lpcjobqueue Apptainer environment (one-time):
```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

Enter the container using a **pinned tag** (required before each Condor session):
```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
```

> **Important:** Always use a pinned container tag instead of `:latest`. The `:latest` tag may lag behind and ship older coffea versions, causing version mismatches between the container's system packages and `pip install -e .` dependencies.

On first launch, the `.env` virtual environment is created automatically. Then install the analysis package:
```bash
pip install -e .
```

To leave the container, type `exit`.

#### Verifying your environment

After setup, confirm that versions match between local and container environments:
```bash
python -c "import coffea; print(coffea.__version__)"   # should print 2025.12.0
python -c "import sys; print(sys.version)"              # should print 3.12.x
```

### Grid Proxy

Authenticate for accessing grid resources (required for both local and Condor runs):
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

### ROOT

Source the appropriate LCG release for ROOT functionality (only needed outside the Apptainer container):

**For FNAL LPC (el9 nodes):**
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

**For UMN (centos8 nodes):**
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
```

---

## Testing

Run the automated test suite with pytest:
```bash
python -m pytest tests/ -v
```

The tests cover utility functions, configuration consistency, fileset validation, histogram creation/filling, and processor selection logic. They run quickly (no data or correctionlib files needed) and are useful for catching regressions when modifying analysis configuration or utility code.

### Quick Local Validation

To quickly test the full analysis chain on a small slice of data:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --maxchunks 1 --maxfiles 1 --chunksize 1000
```

This processes a single file with one small chunk, which is useful for verifying that code changes don't break the processing pipeline before submitting large Condor jobs.

To install the test dependency:
```bash
pip install -e ".[test]"
```

---

## Additional Documentation

- **[Running the Analyzer](docs/run_analysis.md)** - Detailed analysis options and workflows
- **[Creating Filesets](docs/filesets.md)** - Instructions for generating NanoAOD file lists
- **[Skimming](docs/skimming.md)** - Skimming pipeline: selection cuts, CLI reference, Condor job details
- **[Running on Condor](docs/condor.md)** - HTCondor setup, worker defaults, tmux tips, log locations
- **[Expected Limits](docs/run_combine.md)** - Limit-setting with Higgs Combine framework
- **[WR Plotter](WR_Plotter/README.md)** - Plotting ROOT histograms and making stackplots

---

## Update Plotter Submodule

To update the WR_Plotter submodule to the latest commit:
```bash
cd WR_Plotter
git switch main
git pull
cd ..
git commit -am "Update WR_Plotter submodule"
git push
```

To work on a new feature branch in the submodule:
```bash
cd WR_Plotter
git checkout -b my_feature_branch
git push -u origin my_feature_branch
cd ..
```
