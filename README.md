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
python3 bin/run_analysis.py RunIII2024Summer24 all                             # everything, LO HT-binned DY
python3 bin/run_analysis.py RunIII2024Summer24 all --dy nlo_inc                # everything, NLO inclusive DY
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
| `signal` | Signal: the 9-point default grid, or the points chosen with `--signal-points` / `--mwr-range` |
| `mc` | DYJets, tt_tW, Nonprompt, Other, Signal |
| Single sample | `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal` (with `--mass`) |

Output ROOT histograms are saved to `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/`, each file with a JSON manifest `WRAnalyzer_<sample>_systematics.json` next to it.

- **Data is blinded by default.** Data files have no signal-region folders (nor variants such as `_sr_nomass`), and their `ee`/`mumu` cutflows stop before the m(ll) > 200 GeV cut (`em` is complete). The manifest records this under its top-level `blinding` key. `--unblind` lifts the blinding for Run 2 eras only; Run 3 eras are refused. Run 2 workflows that need observed signal-region data must be rerun with `--unblind`.
- **MC event weights.** MC files also carry `event_weights/<region>/positive` and `/negative` (event counts vs log10|w| of the region's nominal weight, split by sign) and `event_weights/<region>/components/<name>` (one distribution per weight factor: generator weight, normalization, pileup, lepton SFs, ...). Nominal only, and not with `--unskimmed`.

See [Running the Analyzer](docs/run_analysis.md#output) for the full file layout and [Blinding](docs/run_analysis.md#blinding) for the details.

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

The skims keep every event that could enter any analysis region (two leptons above 45 GeV with the leading one above 52 GeV, plus two AK4 jets or one AK8 jet) and every branch the analysis can use. The campaign tooling in `wrcoffea/skim/` plans a campaign from the config JSONs through DBS, downloads each input with checksum verification, skims it with two independent selection implementations that must agree, writes size-bounded parts with one schema per dataset, and validates every part against the inputs.

```bash
python3 bin/skim.py cuts                                                     # selection and branch policy
python3 bin/skim.py plan --campaign v2 --plan-dir plans/v2 --era RunIII2024Summer24
python3 bin/skim.py run --plan-dir plans/v2 --dataset TTto2L2Nu --workers 3  # one dataset at a time
python3 bin/skim.py status --plan-dir plans/v2
python3 bin/skim.py validate --plan-dir plans/v2 --report plans/v2/validation.json
python3 bin/skim.py upload --plan-dir plans/v2 && python3 bin/skim.py fileset --plan-dir plans/v2
```

See **[Skimming](docs/skimming.md)** for the selection, the branch policy, the output layout, the verification steps and the full workflow.

---

## Command Reference

### run_analysis.py Flags

| Flag | Arguments | Description |
|------|-----------|-------------|
| `era` | `<era_name>` | **Required positional.** Campaign to analyze (e.g., RunIII2024Summer24) |
| `sample` | `<sample_name>` | **Required positional.** Sample to analyze (e.g., DYJets, Signal, EGamma) |
| `--mass` | `<mass_point>` | Signal mass point (e.g., WR4000_N2100). **Required for Signal sample** |
| `--signal-points` | `default` \| `all` \| `<mass_point> ...` | Signal points of a composite mode with Signal (`signal`, `mc`, `all`): the 9-point default grid, every point of `data/signal_points/<era>_mass_points.csv`, or a list |
| `--mwr-range` | `<min> <max>` | Keep only the signal points with min ≤ M_WR ≤ max (composite modes with Signal) |
| `--region` | `resolved\|boosted\|both` | Analysis region to run (default: both) |
| `--unblind` | | Fill the data signal regions and the full data cutflows (default: data is blinded; MC never is). Run 2 eras only; refused for Run 3 |
| `--dy` | `VARIANT` | DY sample variant (only valid for DYJets). Variants are per-era; see `config.yaml` |
| `--dir` | `<directory>` | Create output subdirectory under rootfiles path |
| `--name` | `<suffix>` | Append suffix to output ROOT filename |
| `--debug` | | Run without saving histograms (for testing) |
| `--unskimmed` | | Use unskimmed filesets instead of default skimmed files |
| `--condor` | | Submit jobs to HTCondor at LPC (requires Apptainer shell, see [Running on Condor](#running-on-condor)) |
| `--fileset` | `<path>` | Override automatic fileset path with a custom fileset JSON |
| `--max-workers` | `<int>` | Number of Dask workers (local default: 3, condor single-sample: 50, condor composite skimmed: 200, condor composite unskimmed: 500) |
| `--worker-wait-timeout` | `<int>` | Seconds to wait for first Condor worker before failing (default: 1200) |
| `--chunksize` | `<int>` | Number of events per processing chunk (default: 250000) |
| `--maxchunks` | `<int>` | Max chunks per file (default: all). Use `1` for quick testing |
| `--maxfiles` | `<int>` | Max files per dataset (default: all). Use `1` for quick testing |
| `--threads-per-worker` | `<int>` | Threads per Dask worker for local runs |
| `--systs` | `lumi` `pileup` `sf` `jer` `jes` `elescale` `elesmear` `muscale` `musmear` `dyzpt` `lsf` `dyreshape` `scale` `ps` `pdf` | Enable systematic variations (see [Systematics](#systematics) and [docs/run_analysis.md](docs/run_analysis.md)) |
| `--syst-observables` | `all` `fit` or `<hist names>` | Histograms that receive the variations (default: all) |
| `--omit-systs` | `<source names>` | Record a deliberate omission of named sources (needed when an era lacks the inputs for a source) |
| `--study` | `transfer-factor`, `dy-cr-split` | Add a study's extra regions to the output. `transfer-factor` adds SR / flavor-CR copies without the m > 800 GeV cut (suffix `_nomass`) for the eμ transfer-factor closure; `dy-cr-split` adds the DY CRs split at m(ll) = 100 GeV (suffixes `_mll60to100`, `_mll100to150`) for the DY-reshape closure |
| `--split-groups` | `tt_tW`, `Nonprompt`, `Other` | Composite runs: save these groups as one ROOT file per subgroup (`tt_tW` → `TTbar`/`tW`; `Nonprompt` → `WJets`/`TTSemiLep`/`SingleTop`; `Other` → `Diboson`/`Triboson`/`ttV`) |
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
python3 bin/run_analysis.py RunIISummer20UL18 signal --signal-points all --mwr-range 1000 6000 --condor   # every signal point, one cluster

# Single sample
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100

# Use NLO inclusive DY for all samples
python3 bin/run_analysis.py RunIII2024Summer24 all --dy nlo_inc

# Single sample on Condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100

# Custom output directory and filename
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir my_study --name test

# Only process resolved region
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved

# Unblinded data with the signal regions (Run 2 eras only)
python3 bin/run_analysis.py RunIISummer20UL18 data --unblind

# Validate fileset without processing
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
```

---

## Repository Structure

The repository follows a clean architecture separating executable scripts, core analysis logic, configuration, and documentation.

### Directory Overview

```
WR_Plotter/         # Submodule: plots, cutflows, transfer factors, signal fitting and limits
bin/                # User-facing CLI scripts (production workflows)
wrcoffea/           # Installable Python package (analysis code, utilities, config)
data/               # Dataset configs, filesets, correction payloads, golden JSONs, signal mass points
docs/               # Documentation (markdown guides)
scripts/            # Helper scripts: filesets, cross sections, reweight derivations, payload conversion
tests/              # Automated test suite (pytest)
studies/            # One-off studies (m(ll) cut optimization)
```

### Key Directories

**`bin/`** - Production Scripts
- [`run_analysis.py`](bin/run_analysis.py) - Main analysis driver script (single samples and composite modes)
- [`skim.py`](bin/skim.py) - Skim campaign driver (`plan`, `run`, `group`, `refresh-metadata`, `condor`, `ship`, `mirror`, `bad-files`, `status`, `validate`, `upload`, `split-scan`, `fileset`, `cuts`; see [docs/skimming.md](docs/skimming.md))
- [`lpc_condor.sh`](bin/lpc_condor.sh) - Host-side wrapper for `skim.py condor` at the LPC

**`wrcoffea/`** - Installable Python Package
- [`analyzer.py`](wrcoffea/analyzer.py) - Main Coffea processor implementing WR→Nℓ→ℓℓjj analysis (object selection, resolved/boosted regions, histogram filling, cutflows)
- [`histograms.py`](wrcoffea/histograms.py) - Histogram specification, creation, and filling
- [`scale_factors.py`](wrcoffea/scale_factors.py) - Lepton scale factor evaluation (correctionlib)
- [`analysis_config.py`](wrcoffea/analysis_config.py) - Centralized configuration (luminosities, correction paths, selection names, cuts)
- [`cli_utils.py`](wrcoffea/cli_utils.py) - CLI plumbing: fileset loading, sample validation, mass point handling
- [`era_utils.py`](wrcoffea/era_utils.py) - Era/year/run mapping and JSON I/O
- [`fileset_utils.py`](wrcoffea/fileset_utils.py) - Fileset path construction, config parsing, JSON writing
- [`fileset_validation.py`](wrcoffea/fileset_validation.py) - Schema and selection validation for filesets
- [`save_hists.py`](wrcoffea/save_hists.py) - Merges datasets, checks systematic coverage, writes the ROOT files and the `*_systematics.json` manifest
- [`regions.py`](wrcoffea/regions.py) - Region registry, study variants (`--study`) and the data blinding policy
- [`systematics.py`](wrcoffea/systematics.py) - Systematic registry (families, sources, coverage)
- [`dy_reweight.py`](wrcoffea/dy_reweight.py), [`dy_reshape.py`](wrcoffea/dy_reshape.py), [`signal_kfactor.py`](wrcoffea/signal_kfactor.py) - DY Z-pT reweight, DY reshape vs jet pT, signal NLO k-factor
- [`MuonScaRe.py`](wrcoffea/MuonScaRe.py) - Muon POG momentum scale and resolution corrections
- [`skim/`](wrcoffea/skim/) - Skim campaign toolkit: `catalog` (DBS/Rucio), `plan`, `fetch`, `selection` + `recount`, `schema` + `branches.yaml`, `writer` + `runs` (Runs/LuminosityBlocks trees), `worker`, `validate`, `upload` (filesets too), `condor` (LPC jobs), `scan` (multi-model scan samples)
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
- [`dataset_configs.md`](docs/dataset_configs.md) - Dataset config keys, how each cross section is built, its perturbative order
- [`skimming.md`](docs/skimming.md) - Skim campaign: selection, branch policy, workflow, verification, output layout
- [`condor.md`](docs/condor.md) - HTCondor setup, worker defaults, tmux, logs
- [`adding_a_new_era.md`](docs/adding_a_new_era.md) - Checklist for adding an era
- [`run2_hnwr_comparison.md`](docs/run2_hnwr_comparison.md) - Comparison with the Run 2 HNWRAnalyzer; DY corrections and closure
- [`repo_review.md`](docs/repo_review.md) - Open review findings
- `configs/`, `pog_recommendations/`, `pdmv/` - Config rebuild notes, POG and PdmV reference material

**`tests/`** - Automated Test Suite
- Unit tests for utilities, config consistency, and validation logic
- Run with pytest (see [Testing](#testing))

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

- **[Repository Review](docs/repo_review.md)** - Findings, resolved issues, and validation recorded on 2026-09-07
- **[Getting Started](docs/getting_started.md)** - Installation, environment setup, and grid proxy
- **[Running the Analyzer](docs/run_analysis.md)** - Detailed analysis options and workflows
- **[Creating Filesets](docs/filesets.md)** - Instructions for generating NanoAOD file lists
- **[Dataset Configs](docs/dataset_configs.md)** - Every config key, cross-section provenance and the `order` of each value
- **[Skimming](docs/skimming.md)** - Skim campaign: selection, branch policy, workflow, verification
- **[Running on Condor](docs/condor.md)** - HTCondor setup, worker defaults, tmux tips, log locations
- **[Limits](WR_Plotter/signal_fitting/LIMITS.md)** - Signal and background models and the Combine limits (in WR_Plotter)
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
