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

### `merge` — Extract, merge, and validate

Each Condor skim job produces one small ROOT file per input file. Coffea and XRootD are inefficient when opening many small files, so `merge` combines them into larger files of ~1M events each, which the analyzer can process much more efficiently. Tarballs are extracted incrementally one at a time and merged in batches using uproot, with individual skim files deleted after each successful merge to keep disk usage bounded. Files are grouped by HLT path to avoid schema mismatches. After all files are merged, event counts and `genEventSumw` totals are validated.

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

## Era-Level Skimming (Scratch Space)

To skim an entire era/campaign at once, use the `run-era`, `check-era`, `merge-era`, and `upload` subcommands. These write to the LPC 3DayLifetime scratch area (`/uscmst1b_scratch/lpc1/3DayLifetime/$USER/skims/`) instead of local `data/skims/`, providing effectively unlimited storage. Files are auto-purged after 3 days of no access.

### Full pipeline

The simplest way to skim an entire era is with `--era`, which auto-discovers all config JSONs in the corresponding `data/configs/` directory and deduplicates shared datasets across configs:

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

# 5. Regenerate skimmed filesets (run once per config file)
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json
```

You can also pass individual config files with `--config`:

```bash
python3 bin/skim.py run-era \
  --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json

# Multiple configs at once:
python3 bin/skim.py run-era \
  --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json \
           data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
```

### Subcommand details

All era-level subcommands accept either `--era ERA` or `--config JSON [JSON ...]` (mutually exclusive).

| Subcommand | Description |
|------------|-------------|
| `run-era (--era ERA \| --config JSON [...])` | Submit Condor jobs for all datasets. Creates the repo tarball once. |
| `check-era (--era ERA \| --config JSON [...]) [--resubmit-script PATH]` | Check all datasets for missing/failed jobs. If failures are found, auto-writes a resubmit shell script (default: `scripts/resubmit_failed_<era>.sh`) and state file (`scripts/.check_era_state_<era>.json`). |
| `resubmit-failures --era ERA [--dry-run]` | Re-run the failed file-jobs from the latest `check-era` state. Errors if no prior check-era state exists or if there are no failed jobs. |
| `merge-era (--era ERA \| --config JSON [...]) [--max-events N] [--skip-check] [--workers N]` | Merge all datasets. `--workers N` parallelizes across datasets. Auto cross-checks `genEventSumw` from config. |
| `upload (--era ERA \| --config JSON [...]) [--remote-user USER] [--dry-run]` | Upload merged files to Wisconsin via `xrdcp`. Default user: `wijackso`. |

Known era names: `RunIISummer20UL18`, `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`.

Datasets with `"note"` containing "no longer available" are automatically skipped. When multiple configs share the same DAS paths (e.g. shared backgrounds), duplicates are skipped automatically. All era-level commands use continue-on-failure: failed datasets are logged and skipped, with a summary printed at the end.

### Handling failures

`check-era` now supports a built-in failed-job workflow:

```bash
# 1) Run era check (writes state + resubmit script if failures are found)
python3 bin/skim.py check-era --era Run3Summer23

# 2) Resubmit only failed file-jobs from that state
python3 bin/skim.py resubmit-failures --era Run3Summer23

# Optional: validate what would run without submitting
python3 bin/skim.py resubmit-failures --era Run3Summer23 --dry-run
```

Notes:
- Default artifacts are `scripts/resubmit_failed_<era>.sh` and `scripts/.check_era_state_<era>.json`.
- `resubmit-failures` errors if no state exists (check-era not run yet) or if latest check-era found zero failed jobs.
- Use `check-era --resubmit-script <path>` to override the script output path.

### Scratch space notes

- **Location:** `/uscmst1b_scratch/lpc1/3DayLifetime/$USER/skims/<run>/<year>/<era>/`
- **Size:** 141 TB shared, no per-user quota
- **Purge policy:** Files are deleted after 3 days of no access
- **Contents:** Output tarballs, merged ROOT files, Condor job files, and logs all live here

## Uploading to Remote Storage

The `upload` subcommand automates copying merged files to Wisconsin:

```bash
# Upload all datasets from a config (reads from scratch space)
python3 bin/skim.py upload --config data/configs/.../Run3Summer22_mc_dy_lo_ht.json

# Upload a single dataset (from local or scratch)
python3 bin/skim.py upload --das-path /TTto2L2Nu_.../NANOAODSIM --scratch

# Dry run (print xrdcp commands without executing)
python3 bin/skim.py upload --config data/configs/.../Run3Summer22_mc_dy_lo_ht.json --dry-run

# Use a different remote user
python3 bin/skim.py upload --config data/configs/.../Run3Summer22_mc_dy_lo_ht.json --remote-user myuser
```

Datasets are uploaded to `root://cmsxrootd.hep.wisc.edu/store/user/<remote-user>/WRAnalyzer/skims/<run>/<year>/<era>/<category>/<dataset>/`, where category is `signals`, `data`, or `backgrounds` based on the dataset name.

After uploading, regenerate the skimmed fileset:
```bash
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_mc_dy_lo_inc.json
```

This queries Wisconsin via `gfal-ls`, rebuilds the file list for all datasets in the config, and writes the output to `data/filesets/Run3/2024/RunIII2024Summer24/skimmed/`.

### Manual upload

If you prefer to upload manually (e.g., to a non-Wisconsin endpoint), see the `xrdcp` examples below. If using a different storage endpoint, you will need to modify `scripts/skimmed_fileset.py` to query the new location when regenerating filesets.

```bash
REMOTE_HOST=cmsxrootd.hep.wisc.edu
REMOTE_PATH=/store/user/<username>/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24/backgrounds/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8
LOCAL_DIR=data/skims/Run3/2024/RunIII2024Summer24/files/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8

for f in $LOCAL_DIR/*.root; do
  echo "Copying $(basename $f) ..."
  xrdcp "$f" "root://$REMOTE_HOST/$REMOTE_PATH/$(basename $f)"
done

# Verify
xrdfs $REMOTE_HOST ls -l $REMOTE_PATH
```

The upload is ~3-4 GB per file, so run this in a tmux session. Your grid proxy must have write access to the destination directory.

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
