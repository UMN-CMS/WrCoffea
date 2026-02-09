# Skimming

The skimmer applies a loose event preselection (looser than the analysis cuts) to NanoAOD files, reducing file sizes for faster analysis iteration. The core logic lives in `wrcoffea/skimmer.py` (importable/testable), with a single CLI entry point at `bin/skim.py` using subcommands. Files are resolved directly from DAS via `dasgoclient` — no pre-built filesets required.

## Skim Selection

The skim keeps events passing a loose lepton + jet preselection. These cuts are intentionally wider than the analysis cuts so the skims remain usable if analysis thresholds change.

| Cut | Value |
|-----|-------|
| Lepton pT min | 45 GeV |
| Lepton eta max | 2.5 |
| Lead lepton pT min | 52 GeV |
| Sublead lepton pT min | 45 GeV |
| AK4 jet pT min | 32 GeV |
| AK4 jet eta max | 2.5 |
| AK8 jet pT min | 180 GeV |
| AK8 jet eta max | 2.5 |

**Selection logic:** (>= 2 leptons passing pT/eta, lead > 52, sublead > 45) AND (>= 2 AK4 jets OR >= 1 AK8 jet)

To print these from the command line:
```bash
python3 bin/skim.py --cuts
```

## Quick Start

All subcommands take a **DAS dataset path** as the first argument — the same `/<primary_dataset>/<campaign>/<datatier>` string you would pass to `dasgoclient`. File lists are resolved directly from DAS, so no pre-built filesets are needed.

### Skim all files via Condor

```bash
python3 bin/skim.py run /TTto2L2Nu_.../RunIII2024Summer24NanoAODv15-.../NANOAODSIM
```

By default, all files are submitted to HTCondor. Use `--dry-run` to generate the job files without submitting, or `--local` to run directly on the current machine.

### Skim a single file locally

```bash
python3 bin/skim.py run /TTto2L2Nu_.../RunIII2024Summer24NanoAODv15-.../NANOAODSIM --start 1 --end 1 --local
```

### Check for failures

```bash
python3 bin/skim.py check /TTto2L2Nu_.../RunIII2024Summer24NanoAODv15-.../NANOAODSIM
```

### Merge outputs

```bash
python3 bin/skim.py merge /TTto2L2Nu_.../RunIII2024Summer24NanoAODv15-.../NANOAODSIM
```

## Subcommands

### `run` — Skim NanoAOD files

By default, submits to Condor. Use `--local` for direct execution on the current machine.

| Flag | Description |
|------|-------------|
| `das_path` | **Required positional.** DAS dataset path |
| `--start N` | 1-indexed start file number (default: first) |
| `--end N` | 1-indexed end file number (default: last) |
| `--local` | Run locally instead of submitting to Condor |
| `--dry-run` | Generate Condor job files without submitting (no effect with `--local`) |

By default, all files in the dataset are processed. Use `--start`/`--end` to select a subset.

**Examples:**
```bash
# Submit all files to Condor (default)
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM

# Skim file 1 locally
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --start 1 --end 1 --local

# Skim files 1-10 locally
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --start 1 --end 10 --local

# Submit a file range to Condor
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --start 1 --end 100

# Dry run (generate JDL + arguments without submitting)
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --dry-run
```

### `check` — Detect missing/failed jobs

Compares the expected number of output tarballs against what was produced. Reports any missing or failed jobs.

| Flag | Description |
|------|-------------|
| `das_path` | **Required positional.** DAS dataset path |
| `--resubmit FILE` | Write a resubmit `arguments.txt` for failed jobs |

**Example:**
```bash
# Check completion
python3 bin/skim.py check /TTto2L2Nu_.../NANOAODSIM

# Write resubmit file for failed jobs
python3 bin/skim.py check /TTto2L2Nu_.../NANOAODSIM --resubmit resubmit_args.txt
```

### `merge` — Extract, hadd, and validate

Each Condor skim job produces one small ROOT file per input file. Coffea and XRootD are inefficient when opening many small files, so `merge` combines them into larger files of ~1M events each, which the analyzer can process much more efficiently. Tarballs are extracted incrementally one at a time and `hadd`'d in batches, with individual skim files deleted after each successful merge to keep disk usage bounded. Files are grouped by HLT path to avoid schema mismatches. After all files are merged, event counts and `genEventSumw` totals are validated.

| Flag | Description |
|------|-------------|
| `das_path` | **Required positional.** DAS dataset path |
| `--max-events N` | Max events per merged file (default: 1,000,000) |
| `--validate-only` | Only validate existing merged files, don't merge |
| `--config JSON` | Cross-check merged `genEventSumw` against an analysis config JSON |

**Examples:**
```bash
# Full merge pipeline
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM

# Merge with config cross-check
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM \
  --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_mc_dy_lo_inc.json

# Validate only (no merging)
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM --validate-only
```

## Output Layout

Output directories are derived automatically from the DAS path. The campaign string is parsed to determine the run period, year, and era, producing a hierarchical layout under `data/skims/`:

```
data/skims/Run3/2024/RunIII2024Summer24/
  TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/
    TTto2L2Nu_..._skim0.tar.gz        # Condor output tarballs (before merge)
    TTto2L2Nu_..._part1.root           # Merged output files (after merge)
    TTto2L2Nu_..._part2.root
    ...
  jobs/
    TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/
      job.jdl                          # Condor job description
      arguments.txt                    # File numbers + DAS paths
      skim_job.sh                      # Worker script
      WrCoffea.tar.gz                  # Repo tarball sent to workers
  logs/
    TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/
      TTto2L2Nu_..._0.out             # Condor stdout
      TTto2L2Nu_..._0.err             # Condor stderr
      TTto2L2Nu_..._0.log             # Condor log
```

This mirrors the directory convention used for configs (`data/configs/Run3/2024/RunIII2024Summer24/`) and other data files. The mapping from DAS campaign to subdirectory is defined in `wrcoffea/das_utils.py:ERA_SUBDIRS`.

## Architecture

- **`wrcoffea/skimmer.py`** — Core skim selection logic (`apply_skim_selection`), streaming I/O with uproot (`_skim_impl`), and the single-file entry point (`skim_single_file`). Uses coffea/dask to compute the selection mask, then uproot for branch-filtered streaming writes.
- **`wrcoffea/das_utils.py`** — DAS dataset path validation, `dasgoclient` queries, XRootD URL construction with redirector fallback.
- **`wrcoffea/skim_merge.py`** — Post-skim tarball extraction, HLT-aware grouping, `hadd`, and validation of event counts and Runs tree `genEventSumw`.
- **`bin/skim.py`** — CLI entry point with `run`, `check`, `merge` subcommands. Handles Condor job generation (JDL, arguments, tarball creation).
- **`bin/skim_job.sh`** — Condor worker script. Extracts the repo tarball, activates the `.env` venv, runs the skimmer on one file, and tars the output for transfer back.

## Condor Job Details

Each skim job processes a single NanoAOD file. The JDL uses:
- **Container:** `coffeateam/coffea-dask-almalinux8:latest` via Apptainer
- **Memory:** 4 GB
- **Input:** Repo tarball (`WrCoffea.tar.gz`, ~24 MB)
- **Output:** Per-file skim tarball (`<dataset>_skim<N>.tar.gz`)

The worker script (`skim_job.sh`) activates the `.env` Python 3.10 venv that was built inside the coffea container, ensuring package versions match the container runtime.
