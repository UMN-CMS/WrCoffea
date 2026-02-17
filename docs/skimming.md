# Skimming

The skimmer applies a loose event preselection (looser than the analysis cuts) to NanoAOD files, reducing file sizes for faster analysis iteration. The core logic lives in `wrcoffea/skimmer.py` (importable/testable), with a single CLI entry point at `bin/skim.py` using subcommands. Files are resolved directly from DAS via `dasgoclient` — no pre-built filesets required.

## Full Pipeline

The typical workflow to skim an entire era end-to-end:

```bash
# 1. Submit all datasets for an era to Condor
python3 bin/skim.py run-era --era Run3Summer22

# 2. Check all datasets for completion
python3 bin/skim.py check-era --era Run3Summer22

# 2b. If step 2 reports INCOMPLETE, resubmit failed file-jobs
python3 bin/skim.py resubmit-failures --era Run3Summer22

# 3. Merge all datasets (parallel across datasets)
python3 bin/skim.py merge-era --era Run3Summer22 --workers 4

# 4. Upload merged files to Wisconsin
python3 bin/skim.py upload --era Run3Summer22

# 4b. If step 4 had upload failures, retry only the failed datasets
python3 bin/skim.py upload --era Run3Summer22 --retry-failed

# 5. Regenerate skimmed filesets (run once per config file)
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json
```

Each step is explained in detail below. All era-level subcommands accept either `--era ERA` (auto-discovers all config JSONs in `data/configs/`) or `--config JSON [JSON ...]` (mutually exclusive). When multiple configs share the same DAS paths, duplicates are skipped automatically.

---

## Step 1: Submit skim jobs (`run-era`)

Submits one Condor job per NanoAOD file for every dataset in the era. Creates the repo tarball once and reuses it across all datasets.

```bash
python3 bin/skim.py run-era --era Run3Summer22
```

You can also pass individual config files:
```bash
python3 bin/skim.py run-era \
  --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json \
           data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
```

Datasets with `"note"` containing "no longer available" are automatically skipped.

## Step 2: Check for failures (`check-era`)

Compares the expected number of output tarballs against what was produced. Reports any missing or failed jobs, and auto-writes a resubmit state file.

```bash
python3 bin/skim.py check-era --era Run3Summer22
```

If failures are found, a state file (`scripts/.check_era_state_<era>.json`) and resubmit script (`scripts/resubmit_failed_<era>.sh`) are written automatically. Use `--resubmit-script <path>` to override the script output path.

### Resubmitting failures

```bash
# Resubmit only failed file-jobs from the latest check-era state
python3 bin/skim.py resubmit-failures --era Run3Summer22

# Validate what would run without submitting
python3 bin/skim.py resubmit-failures --era Run3Summer22 --dry-run
```

`resubmit-failures` errors if no state exists (check-era not run yet) or if the latest check-era found zero failed jobs.

## Step 3: Merge outputs (`merge-era`)

Each Condor skim job produces one small ROOT file per input file. Coffea and XRootD are inefficient when opening many small files, so `merge` combines them into larger files of ~1M events each. Tarballs are extracted incrementally one at a time and merged in batches using uproot, with individual skim files deleted after each successful merge to keep disk usage bounded. Files are grouped by HLT path to avoid schema mismatches. After all files are merged, event counts and `genEventSumw` totals are validated.

```bash
python3 bin/skim.py merge-era --era Run3Summer22 --workers 4
```

| Flag | Description |
|------|-------------|
| `--workers N` | Parallelize merging across datasets |
| `--max-events N` | Max events per merged file (default: 1,000,000) |
| `--skip-check` | Skip the pre-merge completion check |

The merge auto cross-checks `genEventSumw` from the config.

## Step 4: Upload to Wisconsin (`upload`)

Copies merged ROOT files to Wisconsin via `xrdcp`.

```bash
python3 bin/skim.py upload --era Run3Summer22
```

| Flag | Description |
|------|-------------|
| `--retry-failed` | Only retry datasets that failed in the previous upload |
| `--remote-user USER` | Wisconsin storage user (default: `wijackso`) |
| `--dry-run` | Print xrdcp commands without executing |

Datasets are uploaded to `root://cmsxrootd.hep.wisc.edu/store/user/<remote-user>/WRAnalyzer/skims/<run>/<year>/<era>/<category>/<dataset>/`, where category is `signals`, `data`, or `backgrounds` based on the dataset name.

If some uploads fail, re-run with `--retry-failed` to retry only the failed datasets:
```bash
python3 bin/skim.py upload --era Run3Summer22 --retry-failed
```

## Step 5: Regenerate skimmed filesets

After uploading, regenerate the skimmed fileset JSONs that the analyzer reads:

```bash
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json
```

This queries Wisconsin via `gfal-ls`, rebuilds the file list for all datasets in the config, and writes the output to `data/filesets/Run3/2022/Run3Summer22/skimmed/`.

---

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

---

## Single-Dataset Subcommands

For skimming individual datasets (output goes to local `data/skims/` rather than scratch space):

### `run` — Skim NanoAOD files

All subcommands take a **DAS dataset path** as the first argument — the same `/<primary_dataset>/<campaign>/<datatier>` string you would pass to `dasgoclient`.

```bash
# Submit all files to Condor (default)
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM

# Skim file 1 locally
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --start 1 --end 1 --local

# Skim files 1-10 locally
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --start 1 --end 10 --local

# Dry run (generate JDL + arguments without submitting)
python3 bin/skim.py run /TTto2L2Nu_.../NANOAODSIM --dry-run
```

| Flag | Description |
|------|-------------|
| `das_path` | **Required positional.** DAS dataset path |
| `--start N` | 1-indexed start file number (default: first) |
| `--end N` | 1-indexed end file number (default: last) |
| `--local` | Run locally instead of submitting to Condor |
| `--dry-run` | Generate Condor job files without submitting (no effect with `--local`) |

### `check` — Detect missing/failed jobs

```bash
python3 bin/skim.py check /TTto2L2Nu_.../NANOAODSIM

# Write resubmit file for failed jobs
python3 bin/skim.py check /TTto2L2Nu_.../NANOAODSIM --resubmit resubmit_args.txt
```

| Flag | Description |
|------|-------------|
| `das_path` | **Required positional.** DAS dataset path |
| `--resubmit FILE` | Write a resubmit `arguments.txt` for failed jobs |

### `merge` — Extract, merge, and validate

```bash
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM

# Validate only (no merging)
python3 bin/skim.py merge /TTto2L2Nu_.../NANOAODSIM --validate-only
```

| Flag | Description |
|------|-------------|
| `das_path` | **Required positional.** DAS dataset path |
| `--max-events N` | Max events per merged file (default: 1,000,000) |
| `--validate-only` | Only validate existing merged files, don't merge |
| `--config JSON` | Cross-check merged `genEventSumw` against an analysis config JSON |

### `upload` — Upload a single dataset

```bash
python3 bin/skim.py upload --das-path /TTto2L2Nu_.../NANOAODSIM --scratch
python3 bin/skim.py upload --das-path /TTto2L2Nu_.../NANOAODSIM --scratch --dry-run
```

| Flag | Description |
|------|-------------|
| `--das-path` | DAS dataset path |
| `--scratch` | Read from scratch space (implied for `--config`/`--era` mode) |
| `--remote-user USER` | Wisconsin storage user (default: `wijackso`) |
| `--dry-run` | Print xrdcp commands without executing |
| `--retry-failed` | Only retry datasets that failed in the previous upload |

---

## Scratch Space Notes

Era-level subcommands write to the LPC 3DayLifetime scratch area instead of local `data/skims/`.

- **Location:** `/uscmst1b_scratch/lpc1/3DayLifetime/$USER/skims/<run>/<year>/<era>/`
- **Size:** 141 TB shared, no per-user quota
- **Purge policy:** Files are deleted after 3 days of no access
- **Contents:** Output tarballs, merged ROOT files, Condor job files, and logs all live here

---

## Output Layout

Output directories are derived automatically from the DAS path. The campaign string is parsed to determine the run period, year, and era, producing a hierarchical layout under `data/skims/`:

```
data/skims/Run3/2024/RunIII2024Summer24/
  files/
    TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/
      TTto2L2Nu_..._skim0.tar.gz      # Condor output tarballs (before merge)
      TTto2L2Nu_..._skim0.status.json # Per-file skim status sidecar
      TTto2L2Nu_..._part1.root         # Merged output files (after merge)
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

---

## Architecture

- **`wrcoffea/skimmer.py`** — Core skim selection logic (`apply_skim_selection`), streaming I/O with uproot (`_skim_impl`), and the single-file entry point (`skim_single_file`). Uses coffea/dask to compute the selection mask, then uproot for branch-filtered streaming writes.
- **`wrcoffea/das_utils.py`** — DAS dataset path validation, `dasgoclient` queries, XRootD URL construction with redirector fallback.
- **`wrcoffea/skim_merge.py`** — Post-skim tarball extraction, HLT-aware grouping, uproot-based merge, and validation of event counts and Runs tree `genEventSumw`.
- **`bin/skim.py`** — CLI entry point with `run`, `check`, `merge` (single-dataset) and `run-era`, `check-era`, `resubmit-failures`, `merge-era`, `upload` (era-level) subcommands. Handles Condor job generation (JDL, arguments, tarball creation), era failure-state tracking, and targeted failed-job resubmission.
- **`bin/skim_job.sh`** — Condor worker script. Extracts the repo tarball, activates the `.env` venv, runs the skimmer on one file, and tars the output for transfer back.

## Condor Job Details

Each skim job processes a single NanoAOD file. The JDL uses:
- **Container:** `coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12` via Apptainer
- **Memory:** 4 GB
- **Input:** Repo tarball (`WrCoffea.tar.gz`, ~24 MB)
- **Output:** Per-file skim tarball (`<dataset>_skim<N>.tar.gz`) plus status sidecar (`<dataset>_skim<N>.status.json`)

The worker script (`skim_job.sh`) activates the `.env` Python 3.12 venv that was built inside the coffea container, ensuring package versions match the container runtime.
