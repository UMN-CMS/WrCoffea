# Copilot instructions (WrCoffea)

## Big picture
- This repo is a Coffea/Dask NanoAOD analysis pipeline for WR searches.
- Primary flow: **build filesets** (`scripts/*fileset.py`) → **run processor** (`bin/run_analysis.py` → `src/analyzer.py`) → **write ROOT histograms** (`python/save_hists.py`) → **plot in submodule** (`WR_Plotter/`).

## Key entrypoints
- Local/Condor runner: `bin/run_analysis.py` (argparse defines the canonical CLI + valid eras/samples).
- Processor implementation: `src/analyzer.py` (`WrAnalysis`), uses Coffea `ProcessorABC`, Awkward, `hist`.
- Histogram writing/layout: `python/save_hists.py` (sums datasets, splits by `region`/`syst`, writes via uproot).

## Filesets, configs, and data layout
- Config templates live under `data/configs/<Run>/<Year>/<Era>/*.json`.
- Generated filesets are written under `data/jsons/<Run>/<Year>/<Era>/{skimmed,unskimmed}/..._fileset.json`.
- Skimmed filesets (xrootd listing of user skims): `scripts/skimmed_fileset.py` (supports `--umn`; may use EOS or WISC depending on year/sample).
- Unskimmed filesets (Rucio/DAS discovery): `scripts/full_fileset.py --dataset <PrimaryDataset>`.
- Era→(Run,Year) mapping is centralized in `python/preprocess_utils.py`.

## Running
- Install python deps: `python3 -m pip install -r requirements.txt` (preferred).
- Typical local run (once filesets exist): `python3 bin/run_analysis.py Run3Summer22 DYJets`.
- Data samples are `EGamma`/`Muon`; signal uses `Signal --mass ...`.

## Signal mass-point convention (important)
- `bin/run_analysis.py` derives valid mass strings from `data/<ERA>_mass_points.csv`.
- Canonical format is `WR<wr>_N<n>` (e.g. `--mass WR2000_N1900`).
- Legacy `MWR<wr>_MN<n>` inputs are auto-normalized to `WR<wr>_N<n>`.

## Outputs
- ROOT outputs go under `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/[--dir]/`.
- Naming:
  - bkg/data: `WRAnalyzer[_<--name>]_SAMPLE.root`
  - signal: `WRAnalyzer[_<--name>]_signal_<--mass>.root`
- ROOT key layout (see `python/save_hists.py`):
  - Nominal: `/<region>/<var>_<region>`
  - Systematics: `/syst_<syst>_<region>/<var>_syst_<syst>_<region>` (syst normalized to lowercase alnum)
  - Cutflows: `/cutflow/...` (recursively written)

## Condor @ LPC
- Setup (once): `bash bootstrap.sh` (creates `./shell` + `.bashrc` for an apptainer Coffea environment).
- Start image: `./shell coffeateam/coffea-base-almalinux9:0.7.29-py3.10`.
- Run: `python3 bin/run_analysis.py Run3Summer22EE DYJets --condor` (uses `lpcjobqueue.LPCCondorCluster`).

## Repo-specific gotchas
- Several scripts/docs use older sample/mass naming; always trust the argparse choices in `bin/run_analysis.py`.
- Imports commonly rely on manual `sys.path` adjustments (not an installed package); preserve that pattern when adding new modules.
