# WrCoffea Documentation

This repository provides the main analysis framework for processing WR→Nℓ→ℓℓjj events using the Coffea columnar analysis toolkit. It handles background, data, and signal samples to produce histograms for downstream limit-setting and plotting.

## Table of Contents
- [Quick Start](#quick-start) – Get started running the analyzer
  - [Prerequisites](#prerequisites) – Required filesets
  - [Backgrounds](#backgrounds) – Process background samples
  - [Data](#data) – Process data samples
  - [Signal](#signal) – Process signal samples
  - [Output Locations](#output-locations) – Where histograms are saved
  - [Region Selection](#region-selection) – Run resolved/boosted separately
- [Command Reference](#command-reference) – Complete flag reference and examples
- [📂 Repository Structure](#-repository-structure) – Overview of how the codebase is organized
- [Getting Started](#getting-started) – Installation and environment setup
- [Additional Documentation](#additional-documentation) – Links to detailed guides

---

## Quick Start

### Prerequisites

Before running the analyzer, you must create filesets (see [`docs/filesets.md`](docs/filesets.md) for instructions). Check that filesets exist for your era:
```bash
ls data/filesets/<era>/
```

To see available eras:
```bash
python3 bin/run_analysis.py --list-eras
```

To see available samples:
```bash
python3 bin/run_analysis.py --list-samples
```

---

### Backgrounds

Run the analyzer on background samples:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIISummer20UL18 Nonprompt
python3 bin/run_analysis.py Run3Summer22EE tt_tW
```

To run over all backgrounds for a given era:
```bash
bash bin/analyze_all.sh bkg RunIII2024Summer24
```

---

### Data

Process data samples (EGamma or Muon):
```bash
python3 bin/run_analysis.py RunIII2024Summer24 EGamma
python3 bin/run_analysis.py RunIISummer20UL18 Muon
```

To run over both EGamma and Muon:
```bash
bash bin/analyze_all.sh data RunIII2024Summer24
```

---

### Signal

Signal samples require the `--mass` flag:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3200_N3000
```

To see available signal mass points:
```bash
python3 bin/run_analysis.py --list-masses
python3 bin/run_analysis.py RunIII2024Summer24 --list-masses
```

To run over a set of signal mass points:
```bash
bash bin/analyze_all.sh signal RunIII2024Summer24
```

---

### Output Locations

By default, ROOT histograms are saved to:
```
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/WRAnalyzer_<Sample>.root
```

Examples:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_DYJets.root
WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/WRAnalyzer_signal_WR4000_N2100.root
```

Using `--dir` creates a subdirectory:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/my_study/WRAnalyzer_DYJets.root
```

Using `--name` modifies the filename:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_test_DYJets.root
```

---

### Region Selection

By default, both resolved and boosted histograms are filled. Use `--region` to process only specific regions:

```bash
# Only fill resolved histograms (faster)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved

# Only fill boosted histograms (faster)
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --region boosted

# Fill both (default behavior)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region both
```

This is useful for:
- **Performance**: Processing only the region you need reduces runtime
- **Testing**: Debugging specific region selections independently
- **Studies**: Focused analysis on resolved or boosted topologies

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
| `--condor` | | Submit jobs to HTCondor (not yet implemented) |
| `--max-workers` | `<int>` | Cap number of Dask workers (local: adaptive max, condor: scale count) |
| `--threads-per-worker` | `<int>` | Threads per Dask worker for local runs |
| `--systs` | `lumi` | Enable systematic variations (currently: lumi) |
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

# Process with systematics
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi

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
```

---

## 📂 Repository Structure

The repository follows a clean architecture separating executable scripts, core analysis logic, configuration, and documentation.

### Directory Overview

```
WR_Plotter/  # Submodule for plotting ROOT histograms
bin/         # User-facing CLI scripts (production workflows)
src/         # Core analysis code (Coffea processor)
python/      # Reusable analysis utilities and helpers
data/        # Configuration files (JSON, CSV) and metadata
docs/        # Documentation (markdown guides)
scripts/     # Helper scripts for setup and post-processing
test/        # Development and validation scripts
```

### Key Directories

**`bin/`** - Production Scripts
- [`run_analysis.py`](bin/run_analysis.py) - Main analysis driver script
- [`analyze_all.sh`](bin/analyze_all.sh) - Batch processing wrapper for multiple samples
- Thin wrappers around the core analysis processor

**`src/`** - Core Analysis Code
- [`analyzer.py`](src/analyzer.py) - Main Coffea processor implementing WR→Nℓ→ℓℓjj analysis
  - Object selection (electrons, muons, AK4/AK8 jets)
  - Resolved and boosted region definitions
  - Histogram filling with systematic variations
  - Cutflow bookkeeping

**`python/`** - Analysis Utilities
- [`run_utils.py`](python/run_utils.py) - Fileset loading, sample validation, mass point handling
- [`preprocess_utils.py`](python/preprocess_utils.py) - Era/year/run parsing utilities
- [`save_hists.py`](python/save_hists.py) - ROOT histogram serialization
- [`analysis_config.py`](python/analysis_config.py) - Centralized configuration (lumi, JME JSONs, etc.)

**`data/`** - Configuration and Metadata
- `filesets/` - Per-era NanoAOD file lists (JSON format)
  - Organized by era (e.g., `RunIII2024Summer24/`, `RunIISummer20UL18/`)
  - Background, data, and signal filesets
- `signal_points/` - Available signal mass points per era (CSV format)
- `lumis/` - Golden JSON lumi masks for data
- `analysis_config.py` - Luminosity values, JME correction paths, uncertainties

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
- [`run_combine.md`](docs/run_combine.md) - Limit-setting with Combine framework

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

Create and activate a Python virtual environment:
```bash
python3 -m venv wr-env
source wr-env/bin/activate
```

Install required packages:
```bash
python3 -m pip install -r requirements.txt
```

### Grid UI

Authenticate for accessing grid resources:
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

### ROOT

Source the appropriate LCG release for ROOT functionality:

**For FNAL LPC (el9 nodes):**
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

**For UMN (centos8 nodes):**
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
```

---

## Additional Documentation

- **[Running the Analyzer](docs/run_analysis.md)** - Detailed analysis options and workflows
- **[Creating Filesets](docs/filesets.md)** - Instructions for generating NanoAOD file lists
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
