# Running on Condor

Scale out processing across many workers at FNAL LPC using HTCondor with the Dask executor. This requires the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue) Apptainer environment.

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
- **Composite modes** (`all`, `data`, `bkg`, `mc`, `signal`): 3000 workers
- **Single-sample** (`DYJets`, `EGamma`, etc.): 50 workers

These defaults can be overridden with `--max-workers`.

### Systematics on Condor

Systematics are automatically filtered by sample type:
```bash
# Systematics applied only to MC (backgrounds and signal), ignored for data
python bin/run_analysis.py RunIII2024Summer24 all --systs lumi pileup sf
```

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

Skim Condor logs are written to `data/skims/logs/<primary_dataset>/`. See [Skimming](skimming.md) for details.

## Processing Times

Wall-clock times for `RunIII2024Summer24` on Condor (250k chunksize, 4 GB/worker).

| Sample    | Skimmed (50 workers) | Unskimmed (400 workers) |
|-----------|----------------------|-------------------------|
| Muon      |     17.49 minutes    |       81.24 minutes     |
| EGamma    |     9.91 minutes     |       80.12 minutes     |
| DYJets    |     8.05 minutes     |       55.87 minutes     |
| tt_tW     |     8.51 minutes     |       30.74 minutes     |
| Nonprompt |     10.47 minutes    |       71.67 minutes     |
| Other     |     7.05 minutes     |       64.50 minutes     |
| Signal    |     Run locally      |       Run locally       |
