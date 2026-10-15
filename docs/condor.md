# Running on Condor

What this page answers: how to run the analyzer on LPC Condor workers (Dask through
[lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue)), which settings make it fast and
reliable and why (measured on 2026-10-14), what sets the time a run takes, and where the logs of
analysis and skim jobs go. Condor only works at the FNAL LPC, inside the lpcjobqueue Apptainer
environment below. Verified on 2026-10-14.

> **Config overlays do not reach Condor workers.** `WRCOFFEA_CONFIG_OVERLAY` is read where
> `wrcoffea.analysis_config` is imported; the workers do not inherit the variable, and the overlay
> file is not shipped to them (`wrcoffea/analysis_config.py`). A run with an overlay must be local.

## Setup

### Once per clone (from the repo root, on an LPC login node)

```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
pip install -e .
```

`./shell` creates the `.env` virtual environment on its first launch and installs lpcjobqueue into
it. The workers get the coffea image and `.env`, nothing else from your account. Rerun
`pip install -e .` only after switching the image tag (which recreates `.env`) or changing the
dependencies in `pyproject.toml`; the editable install picks up code changes by itself.

> **Pin the image tag.** `:latest` can lag behind and mismatch the versions `pyproject.toml` pins;
> the tag above ships exactly those. Tags: `/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:*`.

> **An lpcjobqueue in `~/.local` breaks every worker.** If an earlier `pip install` put lpcjobqueue
> in your user site (`~/.local/lib/python3.12/site-packages`), the bootstrap finds it there and
> leaves `.env` without it; every worker then dies at start (`No module named 'lpcjobqueue'`, in
> the worker logs). `run_analysis.py --condor` checks for this before submitting anything and names
> the directories to delete; delete them (with their `*.dist-info`) and start `./shell` again.

### Every session: one named node, inside tmux

```bash
ssh <user>@cmslpc3NN.fnal.gov    # one named node, not the cmslpc-el9 alias
tmux new -s condor               # later: tmux attach -t condor (on the same node)
cd ~/nobackup/WrCoffea && ./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
```

The Dask scheduler runs inside your `run_analysis.py` process on the login node, so the run ends
with the ssh session or the node (on 2026-10-14 the LPC rebooted its login nodes one after the
other). tmux keeps the run alive when you log out; the `cmslpc-el9` alias picks a node at random,
so you would not find the session again. Several runs may share a node, in separate tmux windows.

## Running the analysis on Condor

### Composite modes

Use `--condor` to run composite modes on Condor (without it, they run locally one sample after the
other):

```bash
python bin/run_analysis.py RunIII2024Summer24 all --condor       # everything
python bin/run_analysis.py RunIII2024Summer24 bkg --condor       # all backgrounds
python bin/run_analysis.py RunIII2024Summer24 data --condor      # all data
python bin/run_analysis.py RunIII2024Summer24 mc --condor        # backgrounds + signal
python bin/run_analysis.py RunIII2024Summer24 signal --condor    # signal only
```

The composite modes that include signal (`signal`, `mc`, `all`) run the 9-point default grid. Add
`--signal-points all` for every point of `data/signal_points/<era>_mass_points.csv`, or
`--mwr-range MIN MAX` to keep a range of M(W_R). One composite run is faster than the same samples
run one by one: every run pays the wait for its first workers and the slow end described below.

### Unskimmed samples

```bash
python bin/run_analysis.py RunIII2024Summer24 all --condor --unskimmed
```

`--unskimmed` reads the full NanoAOD files instead of the skims ([inputs.md](inputs.md)).

### Single samples

```bash
python bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --condor
python bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --condor
```

### Systematics

```bash
python bin/run_analysis.py RunIII2024Summer24 all --condor --systs lumi pileup sf --xrootd-timeout 120
```

Systematics apply to MC only. The XRootD read timeout per chunk is 60 s under `--condor` (10 s
locally). Runs with `--systs` have failed with "Operation expired" at the defaults: raise it with
`--xrootd-timeout` (120 to 300 s; see [run_analysis.md](run_analysis.md)).

## One cluster for several commands

Every `--condor` run waits for its own workers, from a minute to half an hour in a busy pool (see
[What sets the time of a run](#what-sets-the-time-of-a-run)). For several commands (a production,
the DY variants, reruns after a fix), start the workers once and attach the commands to them:

```bash
# tmux window 1: the workers, until Ctrl+C or 30 minutes without work
python bin/condor_cluster.py --workers 200          # --worker-memory 8GB for runs with --systs

# tmux window 2, same checkout: one command after another, or several at the same time
python bin/run_analysis.py RunIII2024Summer24 all --signal-points all --scheduler condor_scheduler.json --dir <tag>
python bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --scheduler condor_scheduler.json --dir <tag>
```

`--scheduler` takes the scheduler file that `condor_cluster.py` writes (or its address), implies
`--condor` and starts no workers, so `--max-workers` and `--worker-memory` belong to
`condor_cluster.py`. Workers that die are replaced. The cluster stops by itself after
`--idle-timeout` seconds (1800 by default) without a task and without an attached run, so that
idle workers do not hold slots of your share.

The workers keep the copy of `wrcoffea/` and of the data they read (`data/lumis`, `jsonpog`,
`reweights`, `pdf_sets`, `muon_highpt`) made when the cluster started. `condor_cluster.py` records
that copy (a hash of every file), the package versions and the image on the scheduler, and a run
whose checkout differs is refused with the list of files, before any work:

```text
RuntimeError: Cannot use the cluster started at 2026-10-14T22:43:44+00:00: 1 file(s) of wrcoffea/ or
data/ differ from the workers' copy: wrcoffea/histograms.py. ...
```

After changing the code, `config.yaml` or a payload, stop the cluster (Ctrl+C) and start it again.

Measured on 2026-10-14 with 60 workers: DY `lo_inc` and `nlo_inc` took 2.9 and 3.2 minutes attached,
against 8 and 9 minutes as separate `--condor` runs (which wait for their own workers); `Other` and a
repeat of `lo_inc` attached at the same time took 3.5 and 2.6 minutes. The attached outputs equal the
separate runs' (yields to 2e-16, unweighted cutflows exactly).

The same evening every era (UL16APV to 2024: data, the backgrounds, every DY variant except
`nlo_ptll`, `lo_ht` also with `--dy-reweight zpt` and `zpt+reshape`, all signal points; 79 commands,
2.3 B events, `--dir 20261015_skims_v2`) ran on one cluster of 250 workers, three commands at a time
from a list, in 84 minutes from the first command to the last, none failed. The 2024 files equal
the separate runs of the day before (`20261014_skims_v2`) to 2e-16.

## Settings

The defaults below come from measurements on 2026-10-14 (RunIII2024Summer24 skims read from
Wisconsin; the numbers are in the next section). The constants are at the top of
`bin/run_analysis.py`.

| Option | Default | Why |
|---|---|---|
| `--worker-memory` | `CONDOR_WORKER_MEMORY` (6GB); `CONDOR_WORKER_MEMORY_SYSTS` (8GB) with `--systs` | The job's memory request (a hard limit on the worker node) and the Dask worker's memory limit. It must cover the worker's peak, including the merges of the outputs. |
| `--chunksize` | `CONDOR_CHUNKSIZE` (100000) with `--condor`, 250000 locally | Lower worker memory than 250000 and a shorter end of the run, at the same throughput. |
| `--max-workers` | 50 single sample, 200 composite, 500 unskimmed composite | The pool, not this number, sets how fast workers come. |
| `--worker-wait-timeout` | 3600 s | How long to wait for the first worker while the jobs sit idle in a busy pool. Jobs that end or are held before any worker connected stop the run at once. |

The chunk outputs are merged in a tree on the workers (ten at a time), and the merged result of a
composite holds every dataset's histograms: about 4 MB per dataset (nominal), so about 1.2 GB for
`all --signal-points all` (about 300 datasets). The outputs travel and wait compressed (lz4, about 28
times smaller); the last merges still build the whole output on one worker, on top of the memory a
worker keeps after processing (1.5 to 2 GB).

Two Dask thresholds act on top of the hard limit: at 80 % of `--worker-memory` a worker stops taking
work, and at 95 % it is restarted. A paused worker keeps its one queued chunk (Dask does not move it),
and a restarted worker loses the merged results it held, which Dask recomputes from their chunks; a
merge whose worker dies four times ends the run (`KilledWorker`). Both happened with 4 GB workers.

## What sets the time of a run

Measured on 2026-10-14:

1. **Getting workers.** The first workers took 1 to 5 minutes to match. The pool is usually full:
   workers then come as other users' jobs end (about 7 per minute in the morning) or in bursts (50 in
   one minute), and between 10:06 and 10:41 no job of ours matched at all. Each user gets a share of
   the free slots, and recent use lowers your priority for hours: while our runs held about 500
   workers, newly submitted runs of ours waited over half an hour, and after that hour of use new
   jobs were passed over while slots that fit them stood free. Asking for more workers does not make
   them come faster, and a large memory request slows them down: of 50 workers of each size submitted
   at the same moment, the first 2 GB and 4 GB workers arrived after about a minute, the first 10 GB
   worker after five.
2. **Throughput.** A busy worker processes 2,700 to 3,600 events per second of nominal 2024 skims
   (31 to 37 s per 100000-event chunk), about a third of a laptop core, and about 320 per second with
   every systematic producible for 2024 (5 minutes per chunk). Reading the skims from FNAL EOS instead
   of Wisconsin gave the same time per chunk, and the job's inputs (25 MB) transfer in seconds.
3. **The end of a run.** The run ends when its slowest chunk and the last merges end; a paused or
   restarted worker there costs minutes (see Settings). Smaller chunks end sooner.
4. **After saving.** Closing the workers takes about 1.5 minutes. Once the log says
   `All output saved`, Ctrl+C is safe; the jobs end on their own.

Worker memory (peak resident memory, from the workers and from Condor), 100000-event chunks: 1.6 to
2.9 GB nominal (processing and merging), 2.8 to 3.9 GB with every 2024 systematic; with 250000-event
chunks, up to 3.5 GB nominal. With systematics, `--chunksize 50000` about halves the peak (2.1 GB),
so `--worker-memory 6GB` is enough there if 8 GB jobs wait long for slots.

The 2024 production of 2026-10-14 (`--dir 20261014_skims_v2`: data, every background, the six other
DY setups, all 137 signal points; about 770 M events) ran as seven commands, each in its own tmux
window. The six DY runs, at the same time, took 8 to 20 minutes each, including the wait for workers.
The composite `all --signal-points all` (368 M events) took 44 minutes: 10 waiting for its first
worker (our priority after a day of heavy use), 1.5 preprocessing, 26 processing on 60 to 90
workers, 1.5 saving and 3.5 closing the workers. The same set took 8.2 hours on a laptop the day
before, one command after the other.

Tested and not worth changing: skipping the preprocessing (it overlaps the wait for workers),
shipping only the era's payloads (the input transfer takes seconds), and workers with several cores
(they need several free cores in one slot of an almost full pool).

## Same results on the laptop and on Condor

Since 2026-10-14 the smearing random numbers (stochastic JER, Run 3 electron smearing, high-pT muon
smearing) are the same on macOS and on Linux ([corrections.md](corrections.md)), so a laptop run and a
Condor run of the same code and inputs give the same histograms (checked: 14 unweighted cutflows equal,
yields equal to 1e-16). Outputs made before that differ between the two platforms, and from newer
ones, at the level of the smearing's statistical noise.

## Logs

### Dask worker logs

```text
/uscmst1b_scratch/lpc1/3DayLifetime/$USER/dask-logs/worker-<JOBID>.0.{out,err}
```

Deleted after 3 days. A few workers failing with "Nanny failed to start" is harmless: Dask gives
their work to the others.

### Condor job status

lpcjobqueue submits to one of the LPC schedds, so ask all of them:

```bash
condor_q -global $USER -totals
```

A run that ends normally or with Ctrl+C removes its worker jobs. A run that is killed (its tmux
window closed, the node rebooted) leaves them in the queue, where they keep taking slots from your
share: find them with `condor_q -global $USER` and remove them with
`condor_rm -name <schedd> <ClusterId>` (or `condor_rm -name <schedd> $USER` for all your jobs there).

### Skim job logs

Skim jobs are submitted with `bin/skim.py condor`, through `bin/lpc_condor.sh` at the LPC. They do
not use the lpcjobqueue environment above: each job runs in its own coffea image from cvmfs (the
image is named in [skimming.md](skimming.md#running-on-condor-at-the-lpc)).

- Each job writes `logs/<dataset>_g<gid>.out` and `.err` in the jobs directory given by `--jobs-dir`, plus a shared `logs/condor.log`.
- The group's status file (done or failed, bad inputs with their attempt logs) is shipped to `<campaign root>/<run>/<year>/<era>/<category>/<dataset>/_groups/`.

See [Running on Condor at the LPC](skimming.md#running-on-condor-at-the-lpc).
