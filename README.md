<div align="center">
  <img src="docs/coffea_logo.svg" alt="Coffea Logo" width="250">
</div>

# WrCoffea Documentation

This repository provides the main analysis framework for processing WR→Nℓ→ℓℓjj events using the Coffea columnar analysis toolkit. It handles background, data, and signal samples to produce histograms for downstream limit-setting and plotting.

## Getting Started

For first-time setup (cloning, creating the venv, Condor environment), see **[Getting Started](docs/getting_started.md)**.

Activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

> **Tip:** Add this line to your `~/.bashrc` to activate automatically on login:
> ```bash
> cd /path/to/WrCoffea && source .venv/bin/activate && cd -
> ```


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

Run the analyzer by specifying an era and composite mode:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 all                             # everything
python3 bin/run_analysis.py RunIII2024Summer24 mc                              # backgrounds + signal
python3 bin/run_analysis.py RunIII2024Summer24 bkg                             # backgrounds only
python3 bin/run_analysis.py RunIII2024Summer24 data                            # data only
python3 bin/run_analysis.py RunIII2024Summer24 signal                          # signal only
```

Composite modes process multiple samples sequentially (locally) or in parallel (on Condor with `--condor`). You can also run individual samples directly:

| Mode | Samples |
|------|---------|
| `all` | EGamma, Muon, DYJets, tt_tW, Nonprompt, Other, Signal |
| `data` | EGamma, Muon |
| `bkg` | DYJets, tt_tW, Nonprompt, Other |
| `signal` | Signal (default subset of mass points) |
| `mc` | DYJets, tt_tW, Nonprompt, Other, Signal |
| Single sample | `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal` (with `--mass`) |

Output ROOT histograms are saved to `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/`.

> **Note:** Filesets for existing eras are already included in the repository. To create filesets for a new era, see [filesets.md](docs/filesets.md).

See **[Running the Analyzer](docs/run_analysis.md)** for full details: all samples, output customization, region selection, systematics, and batch processing.

### tmux

Analysis jobs can run for a long time. Use `tmux` to keep your session alive after disconnecting from the LPC node. Note which node you are on (`hostname`), since tmux sessions are local to that node — you must SSH back to the same node to reattach.

```bash
# Check and note your hostname (e.g., cmslpc320.fnal.gov)
hostname

# Start a new named session
tmux new -s analysis

# Run your jobs as usual
python bin/run_analysis.py RunIII2024Summer24 all --dir 20260217_skimmed
```

You can then detach from the session with `Ctrl-b` then `d` (press `Ctrl-b`, release, then press `d`) and safely log out. To reattach later, SSH to the **same node**:
```bash
ssh cmslpc320.fnal.gov   # replace with your node
tmux attach -t analysis
```

Other useful tmux commands:
- `tmux ls` — list active sessions
- `Ctrl-b` then `d` — detach from current session
- `tmux kill-session -t analysis` — kill a session

---

## Running on Condor

Scale out processing across many workers at FNAL LPC using HTCondor with the Dask executor. Requires the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue) Apptainer environment.

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12           # enter container
python bin/run_analysis.py RunIII2024Summer24 all --condor           # everything on Condor
```

See **[Running on Condor](docs/condor.md)** for full documentation: setup, worker/chunksize defaults, and log locations.

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
| `--dy` | `lo_inc\|nlo_inc` | Specific DY sample variant (only valid for DYJets) |
| `--dir` | `<directory>` | Create output subdirectory under rootfiles path |
| `--name` | `<suffix>` | Append suffix to output ROOT filename |
| `--debug` | | Run without saving histograms (for testing) |
| `--reweight` | `<json_file>` | Path to DY reweight JSON file (DYJets only) |
| `--unskimmed` | | Use unskimmed filesets instead of default skimmed files |
| `--condor` | | Submit jobs to HTCondor at LPC (requires Apptainer shell, see [Running on Condor](#running-on-condor)) |
| `--fileset` | `<path>` | Override automatic fileset path with a custom fileset JSON |
| `--max-workers` | `<int>` | Number of Dask workers (local default: 3, condor single-sample: 50, condor composite skimmed: 200, condor composite unskimmed: 2000) |
| `--worker-wait-timeout` | `<int>` | Seconds to wait for first Condor worker before failing (default: 1200) |
| `--chunksize` | `<int>` | Number of events per processing chunk (default: 250000) |
| `--maxchunks` | `<int>` | Max chunks per file (default: all). Use `1` for quick testing |
| `--maxfiles` | `<int>` | Max files per dataset (default: all). Use `1` for quick testing |
| `--threads-per-worker` | `<int>` | Threads per Dask worker for local runs |
| `--systs` | `lumi` `pileup` `sf` | Enable systematic variations (see [Systematics](#systematics)) |
| `--tf-study` | | Add transfer factor study regions (no mass cut) to the output |
| `--xrd-fallback` | | Enable XRootD redirector fallback during unskimmed preprocess |
| `--xrd-fallback-timeout` | `<int>` | Seconds per fallback probe (default: 10) |
| `--xrd-fallback-retries-per-redirector` | `<int>` | Probe attempts per redirector during fallback (default: 10) |
| `--xrd-fallback-sleep` | `<float>` | Seconds between fallback retries (default: 10.0) |
| `--list-eras` | | Print available eras and exit |
| `--list-samples` | | Print available samples and exit |
| `--list-masses` | | Print available signal mass points and exit |
| `--preflight-only` | | Validate fileset and exit without processing |

### Examples

```bash
# Composite modes (run locally by default, sequential)
python3 bin/run_analysis.py RunIII2024Summer24 all                   # everything
python3 bin/run_analysis.py RunIII2024Summer24 bkg                   # all backgrounds
python3 bin/run_analysis.py RunIII2024Summer24 data                  # all data
python3 bin/run_analysis.py RunIII2024Summer24 mc                    # backgrounds + signal
python3 bin/run_analysis.py RunIII2024Summer24 signal                # signal only

# Composite mode with custom directory and systematics
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir my_study --name test
python3 bin/run_analysis.py RunIII2024Summer24 all --systs lumi pileup sf

# Composite modes on Condor (parallel, must be inside Apptainer shell)
python3 bin/run_analysis.py RunIII2024Summer24 all --condor --systs lumi pileup sf
python3 bin/run_analysis.py RunIII2024Summer24 bkg --condor

# Single sample
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100

# Single sample on Condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100

# Custom output directory and filename
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir my_study --name test

# Only process resolved region
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved

# Validate fileset without processing
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
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
- [`xrootd_fallback.py`](wrcoffea/xrootd_fallback.py) - XRootD redirector fallback for unskimmed file preprocessing

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
