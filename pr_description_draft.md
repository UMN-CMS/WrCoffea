## Summary

Full rewrite of the WR analysis framework for Run III 2024 data-taking. Replaces the old single-file analyzer with an installable Python package, adds NanoAOD skimming pipeline, Condor integration, scale factors, systematic uncertainties, and comprehensive documentation.

### Architecture
- **Package restructure**: The old `src/analyzer.py` and `python/` directory are replaced with an installable `wrcoffea/` package (with `pyproject.toml`). All analysis code now lives under `wrcoffea/`.
- **Python 3.12 + coffea 2025.12.0**: Pinned to match the `coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12` Apptainer container. `requires-python = ">=3.10"`.
- **Old test files removed** (`test/analyzer_test.py`, etc.) and replaced with proper `tests/` directory with unit tests.

### Core Analysis Code
- **`wrcoffea/analyzer.py`** (new) — Complete rewrite of the processor. Adds boosted selections, lepton flavor tagging, trigger masks, jet veto maps, correctionlib-based jet ID, and systematic weight infrastructure (pileup, muon/electron SFs, lumi uncertainty).
- **`wrcoffea/histograms.py`** (new) — Separated histogram specs and filling from the analyzer. Includes both resolved and boosted histogram definitions, cutflow filling.
- **`wrcoffea/scale_factors.py`** (new) — Muon (RECO/ID/ISO/trigger), electron (RECO/ID/trigger), pileup, and jet veto map scale factors using correctionlib.
- **`wrcoffea/analysis_config.py`** (new) — Centralized cuts, selection labels, luminosities, cross-sections, and JSON paths. Includes corrected Run3 2024 LO DY cross section.
- **`wrcoffea/cli_utils.py`** (new) — Fileset loading, mass point parsing, validation, `max_files` support.
- **`wrcoffea/era_utils.py`** (new) — Era/campaign metadata helpers.

### NanoAOD Skimming Pipeline
- **`bin/skim.py`** (new) — CLI with `run`, `check`, `merge` subcommands. Resolves files directly from DAS, submits to Condor, validates outputs.
- **`bin/skim_job.sh`** (new) — Condor worker script for single-file skimming inside Apptainer container.
- **`wrcoffea/skimmer.py`** (new) — Core skim selection logic (loose lepton + jet preselection), streaming I/O with uproot.
- **`wrcoffea/skim_merge.py`** (new) — Post-skim tarball extraction, HLT-aware grouping, uproot-based merge (replaces `hadd` to handle RNTuple input files), and event count/genEventSumw validation.
- **`wrcoffea/das_utils.py`** (new) — DAS dataset path validation, `dasgoclient` queries, XRootD URL construction with redirector fallback.

### CLI & Workflow
- **`bin/run_analysis.py`** — Major expansion: added `--condor`, `--unskimmed`, `--maxchunks`, `--maxfiles`, `--systs`, `--region`, `--preflight-only`, `--list-*` flags. On-the-fly sumw normalization for unskimmed files.
- **`bin/analyze_all.sh`** — Rewritten to support `data`, `bkg`, `signal`, and `all` modes. Dynamically selects signal mass points from CSV. Optimized worker/chunksize defaults for skimmed vs. unskimmed.

### Running on Condor
- **`LPCCondorCluster` integration** — `--condor` flag in `run_analysis.py` submits Dask workers to HTCondor at FNAL LPC via `lpcjobqueue`. Configurable `--max-workers` and `--chunksize`.
- **`bin/analyze_all.sh`** — Optimized Condor defaults: 50 workers for skimmed background/data, 400 for unskimmed, 10 for signal. Chunksize tuned per sample type.
- **`bin/benchmark_condor.sh`** (new) — Worker/chunksize sweep tool for performance tuning.
- **Apptainer container** — Pinned to `coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12` to ensure version consistency between local and worker environments.

### tmux for Long-Running Jobs
- **README and docs** include tmux workflow for keeping Condor sessions alive after SSH disconnects: `tmux new -s analysis`, detach with `Ctrl-b d`, reattach by SSHing back to the same LPC node.
- Important note: tmux sessions are node-local — must reconnect to the same `cmslpcXXX.fnal.gov` host to reattach.

### Bug Fixes
- **`wrcoffea/histograms.py`** — Fixed `yieldhist()` unpacking: Coffea returns 3 values (honecut, hcutflow, labels), not 2.
- **RuntimeWarnings** — Fixed numpy warnings from division and Dask memory settings tuned for Condor stability.

### Data & Filesets
- **Run III 2024** (`RunIII2024Summer24`) fully added: configs, filesets (skimmed + unskimmed), data, signal, DY (LO inclusive + NLO mll-binned).
- **Fileset reorganization**: `data/filesets/` split into `skimmed/` and `unskimmed/` subdirectories.
- **Signal mass points**: moved to `data/signal_points/` CSVs per era.
- **Scale factor JSONs** added under `data/jsonpog/` (EGM, MUO, JME, LUM, PU).
- **Lumi CSVs removed** — golden JSON certs added instead.

### Testing & CI
- **`.github/workflows/ci.yml`** (new) — GitHub Actions CI running pytest on Python 3.9/3.11/3.12 for every push and PR.
- **`tests/test_analyzer.py`** — 22 unit tests for `WrAnalysis` processor: init/region validation, lepton selection, jet selection, trigger mask construction, electron bitmap decoding.
- **`tests/test_histograms.py`** — 12 unit tests for histogram utilities: booking specs, `create_hist` axes/weights, resolved/boosted filling with mock arrays.
- **`tests/test_skimmer.py`**, **`tests/test_skim_merge.py`**, **`tests/test_das_utils.py`**, **`tests/test_cli_utils.py`**, **`tests/test_fileset_validation.py`**, **`tests/test_era_utils.py`**, **`tests/test_save_hists.py`**, **`tests/test_analysis_config.py`** — Additional unit tests covering the full utility layer.

### Dependencies
- **`pyproject.toml`** — Added upper-bound pins to all dependencies (e.g. `coffea>=2025.7,<2026`, `awkward>=2.8,<3`). Added `dask[distributed]` and `fsspec-xrootd`. Requires Python >= 3.10.

### Documentation
- **`docs/getting_started.md`** (new) — Clone, environment setup (CVMFS Python 3.12 venv + Apptainer container), grid proxy.
- **`docs/skimming.md`** (new) — Skim selection cuts, CLI reference, Condor job details, merge pipeline, T2 upload, output layout.
- **`docs/condor.md`** — Complete rewrite: Apptainer setup, pinned container tags, worker/chunksize defaults, tmux tips, log locations, processing times.
- **`docs/adding_a_new_era.md`** (new) — Step-by-step guide for adding new analysis eras with quick-reference checklist.
- **README.md** — Reorganized: Getting Started section with venv activation at top, Quick Start with examples, bashrc tip, command reference table, repository structure.

### Removed
- Old `src/analyzer.py`, `python/preprocess_utils.py`, `docs/plotting.md`, various old scripts in `scripts/old/`, large lumi CSVs.
