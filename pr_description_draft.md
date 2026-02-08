## Summary

### Architecture
- **Package restructure**: The old `src/analyzer.py` and `python/` directory are replaced with an installable `wrcoffea/` package (with `pyproject.toml`). All analysis code now lives under `wrcoffea/`.
- **Old test files removed** (`test/analyzer_test.py`, etc.) and replaced with proper `tests/` directory with unit tests.

### Core Analysis Code
- **`wrcoffea/analyzer.py`** (new) — Complete rewrite of the processor. Adds boosted selections, lepton flavor tagging, trigger masks, jet veto maps, correctionlib-based jet ID, and systematic weight infrastructure (pileup, muon/electron SFs, lumi uncertainty).
- **`wrcoffea/histograms.py`** (new) — Separated histogram specs and filling from the analyzer. Includes both resolved and boosted histogram definitions, cutflow filling.
- **`wrcoffea/scale_factors.py`** (new) — Muon (RECO/ID/ISO/trigger), electron (RECO/ID/trigger), pileup, and jet veto map scale factors using correctionlib.
- **`wrcoffea/analysis_config.py`** (new) — Centralized cuts, selection labels, luminosities, cross-sections, and JSON paths. Includes corrected Run3 2024 LO DY cross section.
- **`wrcoffea/cli_utils.py`** (new) — Fileset loading, mass point parsing, validation, `max_files` support.
- **`wrcoffea/era_utils.py`** (new) — Era/campaign metadata helpers.

### CLI & Workflow
- **`bin/run_analysis.py`** — Major expansion: added `--condor`, `--unskimmed`, `--maxchunks`, `--maxfiles`, `--systs`, `--region`, `--preflight-only`, `--list-*` flags. Condor support with `LPCCondorCluster`. On-the-fly sumw normalization for unskimmed files.
- **`bin/analyze_all.sh`** — Rewritten to support `data`, `bkg`, `signal`, and `all` modes. Dynamically selects signal mass points from CSV.
- **`bin/benchmark_condor.sh`** (new) — Worker/chunksize sweep tool.

### Bug Fixes
- **`wrcoffea/histograms.py`** — Fixed `yieldhist()` unpacking: Coffea returns 3 values (honecut, hcutflow, labels), not 2.

### Data & Filesets
- **Run III 2024** (`RunIII2024Summer24`) fully added: configs, filesets (skimmed + unskimmed), data, signal, DY (LO inclusive + NLO mll-binned).
- **Fileset reorganization**: `data/filesets/` split into `skimmed/` and `unskimmed/` subdirectories.
- **Signal mass points**: moved to `data/signal_points/` CSVs per era.
- **Scale factor JSONs** added under `data/jsonpog/` (EGM, MUO, JME, LUM, PU).
- **Lumi CSVs removed** — golden JSON certs added instead.

### Testing & CI
- **`.github/workflows/ci.yml`** (new) — GitHub Actions CI running pytest on Python 3.9/3.11/3.12 for every push and PR.
- **`tests/test_analyzer.py`** (new) — 22 unit tests for `WrAnalysis` processor: init/region validation, lepton selection (muon/electron/mixed/pT-sorted), jet selection (pT/eta cuts, AK8), trigger mask construction, electron bitmap decoding.
- **`tests/test_histograms.py`** (new) — 12 unit tests for histogram utilities: booking specs completeness, `create_hist` axes/weights, `fill_resolved_histograms` and `fill_boosted_histograms` with mock vector arrays.

### Dependencies
- **`pyproject.toml`** — Added upper-bound pins to all dependencies (e.g. `coffea>=2025.7,<2026`, `awkward>=2.8,<3`) to prevent unexpected breakage from major version bumps.

### README
- Complete rewrite: Quick Start guide, Condor instructions with tmux/hostname tips, condor log inspection, full command reference table.
- Added quick local validation section documenting `--maxchunks 1 --maxfiles 1 --chunksize 1000` pattern for fast smoke tests.

### Removed
- Old `src/analyzer.py`, `python/preprocess_utils.py`, `docs/condor.md`, `docs/plotting.md`, various old scripts in `scripts/old/`.
