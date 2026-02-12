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

On first launch, the `.env` virtual environment is created automatically. Then install the analysis package:
```bash
pip install -e .
```

## Running the Analysis on Condor

Use the `--condor` flag to submit jobs:
```bash
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --condor
```

By default, 50 Condor workers are launched. Use `--max-workers` to change this:
```bash
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100
```

### Batch processing with analyze_all.sh

```bash
bash bin/analyze_all.sh data RunIII2024Summer24 --condor
bash bin/analyze_all.sh bkg RunIII2024Summer24 --condor
bash bin/analyze_all.sh signal RunIII2024Summer24 --condor
bash bin/analyze_all.sh all RunIII2024Summer24 --condor
```

**Default `analyze_all.sh` Condor configuration:**
- **Data**: 50 workers, 250k events/chunk (skimmed); 400 workers with `--unskimmed`
- **Background**: 50 workers, 250k events/chunk (skimmed); 400 workers with `--unskimmed`
- **Signal**: 10 workers/mass point, 50k events/chunk (conservative)

These defaults can be overridden by passing `--max-workers` and `--chunksize` as extra arguments.

### Systematics on Condor

Systematics are automatically filtered by sample type:
```bash
# Systematics applied only to backgrounds and signal, not data
bash bin/analyze_all.sh all RunIII2024Summer24 --condor --systs lumi pileup sf
```

## tmux

Condor jobs can run for a long time. Use `tmux` to keep your session alive after disconnecting from the LPC node. Note which node you are on (`hostname`), since tmux sessions are local to that node — you must SSH back to the same node to reattach.

```bash
# Check and note your hostname (e.g., cmslpc320.fnal.gov)
hostname

# Start a new named session
tmux new -s analysis

# Enter the Apptainer shell and run your jobs as usual
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
bash bin/analyze_all.sh all RunIII2024Summer24 --condor
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
| Muon      |      21 minutes      |       81.24 minutes     |
| EGamma    |       9 minutes      |       80.12 minutes     |
| DYJets    |       7 minutes      |       55.87 minutes     |
| tt_tW     |                      |       30.74 minutes     |
| Nonprompt |                      |                         |
| Other     |                      |                         |
| Signal    |                      |                         |
