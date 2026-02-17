<div align="center">
  <img src="docs/coffea_logo.svg" alt="Coffea Logo" width="250">
</div>

# WrCoffea Documentation

This repository provides the main analysis framework for processing WR→Nℓ→ℓℓjj events using the Coffea columnar analysis toolkit. It handles background, data, and signal samples to produce histograms for downstream limit-setting and plotting.

## Getting Started

Activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

> **Tip:** Add this line to your `~/.bashrc` to activate automatically on login:
> ```bash
> cd /path/to/WrCoffea && source .venv/bin/activate && cd -
> ```

For first-time setup (cloning, creating the venv, Condor environment), see **[Getting Started](docs/getting_started.md)**.

## Table of Contents
- [Quick Start](#quick-start) – Run the analyzer
- [Running on Condor](#running-on-condor) – Scale out with HTCondor at LPC
- [Skimming](#skimming) – Skim NanoAOD files for faster analysis
- [Command Reference](#command-reference) – Complete flag reference and examples
- [Repository Structure](#repository-structure) – Overview of how the codebase is organized
- [Testing](#testing) – Running the automated test suite
- [Additional Documentation](#additional-documentation) – Links to detailed guides

---

## Quick Start

Run the analyzer by specifying an era and sample:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets                          # background
python3 bin/run_analysis.py RunIII2024Summer24 EGamma                          # data
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100      # signal
python3 bin/run_analysis.py RunIII2024Summer24 all                             # everything
```

Output ROOT histograms are saved to `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/`.

> **Note:** Filesets for existing eras are already included in the repository. To create filesets for a new era, see [filesets.md](docs/filesets.md).

See **[Running the Analyzer](docs/run_analysis.md)** for full details: all samples, output customization, region selection, systematics, and batch processing.

---

## Running on Condor

Scale out processing across many workers at FNAL LPC using HTCondor with the Dask executor. Requires the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue) Apptainer environment.

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12           # enter container
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor        # single sample on Condor
python bin/run_analysis.py RunIII2024Summer24 all                    # everything (auto-Condor)
```

See **[Running on Condor](docs/condor.md)** for full documentation: setup, worker/chunksize defaults, log locations, and using tmux.

---

## Skimming

The skimmer applies a loose event preselection to NanoAOD files, reducing file sizes for faster analysis iteration. It uses `bin/skim.py` with subcommands for the full workflow: skim locally or on Condor, check for failures, and merge outputs.

```bash
python3 bin/skim.py --cuts                                           # show skim cuts
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM                    # submit all to Condor
python3 bin/skim.py check /TTto2L2Nu_.../NANOAODSIM                  # check for failures
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM                  # extract + hadd + validate
```

See **[Skimming](docs/skimming.md)** for full documentation: selection cuts, all subcommand flags, output layout, and architecture.

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
| `--max-workers` | `<int>` | Number of Dask workers (local default: 3, single-sample condor: 50, composite condor: 3000) |
| `--chunksize` | `<int>` | Number of events per processing chunk (default: 250000) |
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

# Composite modes (auto-submit to Condor)
python3 bin/run_analysis.py RunIII2024Summer24 all                   # everything
python3 bin/run_analysis.py RunIII2024Summer24 bkg                   # all backgrounds
python3 bin/run_analysis.py RunIII2024Summer24 data                  # all data
python3 bin/run_analysis.py RunIII2024Summer24 mc                    # backgrounds + signal
python3 bin/run_analysis.py RunIII2024Summer24 signal                # signal only

# Composite mode with custom directory
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir my_study --name test

# Run on Condor (must be inside Apptainer shell)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100

# Run with systematics (applied to MC only, ignored for data)
python3 bin/run_analysis.py RunIII2024Summer24 all --systs lumi pileup sf
```

---

## Repository Structure

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
- [`run_analysis.py`](bin/run_analysis.py) - Main analysis driver script (single samples and composite modes)
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
- [`getting_started.md`](docs/getting_started.md) - Installation, environment setup, grid proxy
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

- **[Getting Started](docs/getting_started.md)** - Installation, environment setup, and grid proxy
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
