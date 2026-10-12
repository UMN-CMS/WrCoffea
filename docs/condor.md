# Running on Condor

What this page answers: how to run the analyzer on LPC Condor workers (Dask through
[lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue)), which defaults and limits apply there,
and where the logs of analysis and skim jobs go. Condor only works at the FNAL LPC, inside the
lpcjobqueue Apptainer environment below. Verified on 2026-10-12.

> **Config overlays do not reach Condor workers.** `WRCOFFEA_CONFIG_OVERLAY` is read where
> `wrcoffea.analysis_config` is imported; the workers do not inherit the variable, and the overlay
> file is not shipped to them (`wrcoffea/analysis_config.py`). A run with an overlay must be local.

## Setup

### First-time setup (run once from the repo root):
```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

### Enter the Apptainer shell (required before each Condor session):
```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
```

> **Important:** Always use a pinned container tag instead of `:latest`. The `:latest` tag may lag behind and ship older coffea versions, causing version mismatches between the container's system packages and `pip install -e .` dependencies. Available tags can be found under `/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:*`.

On first launch, the `.env` virtual environment is created automatically. Install the analysis package once:
```bash
pip install -e .
```

This is a one-time step — the `.env` venv persists between sessions, and editable mode picks up code changes automatically. You only need to re-run `pip install -e .` if:
- You switch to a different container tag (which recreates `.env`)
- You change dependencies in `pyproject.toml`

## Running the Analysis on Condor

### Composite modes on Condor

Use `--condor` to run composite modes on Condor (without it, they run locally in sequential order):
```bash
python bin/run_analysis.py RunIII2024Summer24 all --condor       # everything
python bin/run_analysis.py RunIII2024Summer24 bkg --condor       # all backgrounds
python bin/run_analysis.py RunIII2024Summer24 data --condor      # all data
python bin/run_analysis.py RunIII2024Summer24 mc --condor        # backgrounds + signal
python bin/run_analysis.py RunIII2024Summer24 signal --condor    # signal only
```

The composite modes that include signal (`signal`, `mc`, `all`) run the 9-point default grid. Add
`--signal-points all` for every point of `data/signal_points/<era>_mass_points.csv`, or
`--mwr-range MIN MAX` to keep a range of M(W_R).

### Unskimmed Samples on Condor

Use `--unskimmed` to run on the full NanoAOD files instead of the skimmed filesets:
```bash
python bin/run_analysis.py RunIII2024Summer24 all --condor --unskimmed
python bin/run_analysis.py RunIII2024Summer24 bkg --condor --unskimmed
python bin/run_analysis.py RunIII2024Summer24 data --condor --unskimmed
```

The default worker count for unskimmed composite runs is 500. Override with `--max-workers` if needed.

### Single samples on Condor

```bash
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --condor
```

Use `--max-workers` to override the default worker count:
```bash
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100
```

**Default Condor worker counts:**
- **Single-sample** (`DYJets`, `EGamma`, etc.): 50 workers
- **Composite skimmed** (`all`, `data`, `bkg`, `mc`, `signal`): 200 workers
- **Composite unskimmed** (composite + `--unskimmed`): 500 workers

These defaults can be overridden with `--max-workers`.

> **Tip:** More workers isn't always faster. Spinning up a large number of Condor jobs takes time, and for smaller samples a lower worker count may complete the analysis sooner overall. If a run feels slow to get started, try reducing `--max-workers`.

> **Tip:** You do not need to wait for all workers to shut down after the analysis finishes. Once the histograms have been saved (you'll see the output path printed to the terminal), it is safe to exit — the remaining Condor jobs will clean up on their own.

### Systematics on Condor

Systematics are automatically filtered by sample type:
```bash
# Systematics applied only to MC (backgrounds and signal), ignored for data
python bin/run_analysis.py RunIII2024Summer24 all --condor --systs lumi pileup sf --xrootd-timeout 120
```

The XRootD read timeout per chunk is 60 s under `--condor` (10 s locally). Runs with `--systs` have
failed with "Operation expired" at the defaults: raise it with `--xrootd-timeout` (120 to 300 s; see
[run_analysis.md](run_analysis.md)).

## Logs

### Dask worker logs

Written to:
```
/uscmst1b_scratch/lpc1/3DayLifetime/$USER/dask-logs/
```

Files in this scratch area are automatically deleted after 3 days. To inspect logs:
```bash
# List recent logs
ls -lt /uscmst1b_scratch/lpc1/3DayLifetime/$USER/dask-logs/ | head -20

# View a specific worker log
less /uscmst1b_scratch/lpc1/3DayLifetime/$USER/dask-logs/worker-<JOBID>.0.err
```

A small percentage of workers failing with "Nanny failed to start" is normal — Dask redistributes work to healthy workers automatically.

### Condor job status

```bash
condor_q              # currently queued/running jobs
condor_history        # recently completed jobs
```

### Skim job logs

Skim jobs are submitted with `bin/skim.py condor`, through `bin/lpc_condor.sh` at the LPC. They do
not use the lpcjobqueue environment above: each job runs in its own coffea image from cvmfs (the
image is named in [skimming.md](skimming.md#running-on-condor-at-the-lpc)).

- Each job writes `logs/<dataset>_g<gid>.out` and `.err` in the jobs directory given by `--jobs-dir`, plus a shared `logs/condor.log`.
- The group's status file (done or failed, bad inputs with their attempt logs) is shipped to `<campaign root>/<run>/<year>/<era>/<category>/<dataset>/_groups/`.

See [Running on Condor at the LPC](skimming.md#running-on-condor-at-the-lpc).
