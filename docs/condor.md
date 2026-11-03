# Running on Condor

The analyzer runs on LPC Condor workers through Dask and
[lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue). Condor works at the FNAL LPC only, inside
the Apptainer environment below. `WRCOFFEA_CONFIG_OVERLAY` does not reach the workers (the variable
is not inherited and the file is not shipped): a run with an overlay must be local.

## Setup

Once per clone, from the repo root on an LPC login node:

```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
pip install -e .
```

Every session, on one named node inside tmux:

```bash
ssh <user>@cmslpc3NN.fnal.gov    # one named node, not the cmslpc-el9 alias
tmux new -s condor               # later: tmux attach -t condor, on the same node
cd ~/nobackup/WrCoffea && ./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
```

- `./shell` creates the `.env` virtual environment on its first launch and installs lpcjobqueue into
  it. The workers get the coffea image and `.env`, nothing else. Rerun `pip install -e .` only after
  changing the image tag or the dependencies in `pyproject.toml`.
- Pin the image tag. `:latest` can lag behind the versions `pyproject.toml` pins. Tags:
  `/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux8:*`.
- An lpcjobqueue in `~/.local/lib/python3.12/site-packages` leaves `.env` without it and every
  worker dies at start (`No module named 'lpcjobqueue'`). `run_analysis.py --condor` checks for this
  before submitting and names the directories to delete.
- The Dask scheduler runs inside the `run_analysis.py` process on the login node, so the run ends
  with the ssh session or the node. tmux keeps it alive; the `cmslpc-el9` alias picks a node at
  random, so you would not find the session again.

## Running

```bash
python bin/run_production.py RunIII2024Summer24 --condor         # data, backgrounds and every signal point: starts the workers, runs the three commands at once, stops them
python bin/run_production.py RunIII2024Summer24 --condor --all-dy  # also every other DY setup
python bin/run_production.py RunIII2024Summer24 --condor --only signal   # one command by name; also data, bkg, dy_nlo_inc, ...
python bin/run_production.py RunIII2024Summer24 --plan --all-dy  # the commands of the production, nothing runs
python bin/run_production.py all --condor                        # every era with a data fileset, one era's commands at a time
python bin/run_production.py RunIII2024Summer24 --condor -- --systs lumi pileup sf --xrootd-timeout 120
python bin/run_analysis.py RunIII2024Summer24 all --condor       # one command with workers of its own: everything
python bin/run_analysis.py RunIII2024Summer24 bkg --condor       # all backgrounds
python bin/run_analysis.py RunIII2024Summer24 data --condor      # all data
python bin/run_analysis.py RunIII2024Summer24 mc --condor        # backgrounds + signal
python bin/run_analysis.py RunIII2024Summer24 signal --condor    # signal only
python bin/run_analysis.py RunIII2024Summer24 all --condor --unskimmed
python bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --condor
python bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --condor
python bin/run_analysis.py RunIII2024Summer24 all --condor --systs lumi pileup sf --xrootd-timeout 120
```

- A production is the `data`, `bkg` and `signal` composites (every signal point) and, with
  `--all-dy`, every other DY setup from `config.yaml`: `DYJets --dy-reweight zpt` and `zpt+reshape`
  where the era has the payloads, and `DYJets --dy <variant>` for every `dy_variants` entry but the
  default and `nlo_ptll`. `--only NAME ...` picks any of them. Flags after `--` go to every command
  (`--signal-points` and `--mwr-range` to `signal` only); with `--systs` among them the two reweighted DY commands also
  get `dyzpt` and `dyreshape`, and the workers the memory of a `--systs` run. Every command runs
  `--preflight-only` first, so a missing fileset or a refused flag stops the production before a
  worker is requested. Each command writes `logs/<dir>/<era>_<name>.log`; the summary lists every
  command's outcome and the `--only` line that reruns the failed ones. `--workers` (250) sizes the
  workers, `--jobs N` runs N commands at a time (default: an era's commands at once on Condor, one
  locally), `--dir` defaults to today's date.
- `signal`, `mc` and `all` run every point of `data/signal_points/<era>_mass_points.csv`;
  `--signal-points grid` the 9-point grid (lowest, median and highest N at m_WR = 2, 4, 6 TeV),
  `--mwr-range MIN MAX` a range of m(W_R).
- One composite run is faster than the same samples run one by one: every run pays the wait for
  its first workers and the slow end.
- Systematics apply to MC only. The XRootD read timeout per chunk is 60 s under `--condor`; runs
  with `--systs` have failed with "Operation expired" at that value, so raise `--xrootd-timeout` to
  120 to 300 s.

## One cluster for several commands

```bash
# tmux window 1: the workers, until Ctrl+C or 30 minutes without work
python bin/condor_cluster.py --workers 200          # --worker-memory 8GB for runs with --systs

# tmux window 2, same checkout: one command after another, or several at the same time
python bin/run_production.py RunIII2024Summer24 --scheduler condor_scheduler.json --dir <tag>
python bin/run_analysis.py RunIII2024Summer24 all --scheduler condor_scheduler.json --dir <tag>
python bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --scheduler condor_scheduler.json --dir <tag>
```

- `--scheduler` takes the file `condor_cluster.py` writes (or its address), implies `--condor` and
  starts no workers; `--max-workers` and `--worker-memory` belong to `condor_cluster.py`.
  `run_production.py --condor` starts workers of its own the same way and stops them when its
  commands end; with `--scheduler` it uses this cluster instead, so a rerun of a failed command does
  not wait for new workers.
- Workers that die are replaced. The cluster stops after `--idle-timeout` seconds (1800) without a
  task and without an attached run.
- The workers keep the copy of `wrcoffea/` and of the data they read (`data/lumis`, `jsonpog`,
  `reweights`, `pdf_sets`, `muon_highpt`) made when the cluster started. `condor_cluster.py` records
  a hash of every file; a run whose checkout differs is refused with the list of files. After a
  change to the code, `config.yaml` or a payload, stop the cluster and start it again.
- Attached runs equal separate `--condor` runs (yields to 2e-16, unweighted cutflows exactly). The
  whole 2024 production (data, backgrounds, six DY setups, 137 signal points, 770 M events) ran as
  seven commands in 30 to 60 minutes on one cluster; every era together (79 commands, 2.3 B events)
  took 84 minutes on 250 workers. `run_production.py --all-dy` issues the same work, with `all`
  as its `data`, `bkg` and `signal` commands.

## Settings

The constants are at the top of `bin/run_analysis.py`.

| Option | Default | Reason |
|---|---|---|
| `--worker-memory` | `CONDOR_WORKER_MEMORY` (6GB); `CONDOR_WORKER_MEMORY_SYSTS` (8GB) with `--systs` | the job's memory request, a hard limit, must cover the worker's peak including the merges |
| `--chunksize` | `CONDOR_CHUNKSIZE` (100000) with `--condor`, 250000 locally | lower worker memory and a shorter end of the run at the same throughput |
| `--max-workers` | 50 single sample, 200 composite, 500 unskimmed composite | the pool, not this number, sets how fast workers come |
| `--worker-wait-timeout` | 3600 s | the wait for the first worker in a busy pool; jobs that end or are held before a worker connected stop the run at once |

- Chunk outputs are merged in a tree on the workers, ten at a time. A composite's merged result
  holds every dataset's histograms, about 4 MB per dataset nominal, 1.2 GB for
  `all` with every signal point; the last merges build it on one worker on top of the 1.5 to 2 GB a
  worker keeps after processing.
- Dask pauses a worker at 80 % of `--worker-memory` and restarts it at 95 %. A paused worker keeps
  its queued chunk; a restarted one loses its merged results, which are recomputed; a merge whose
  worker dies four times ends the run (`KilledWorker`). Both happened with 4 GB workers.
- Peak worker memory with 100000-event chunks: 1.6 to 2.9 GB nominal, 2.8 to 3.9 GB with every 2024
  systematic; 250000-event chunks up to 3.5 GB nominal. With systematics `--chunksize 50000` halves
  the peak (2.1 GB), so 6 GB workers do if 8 GB jobs wait long for slots.

## What sets the time of a run

- Getting workers: the first ones take 1 to 5 minutes. The pool is usually full; workers come as
  other jobs end (about 7 per minute) or in bursts, and recent heavy use lowers your priority for
  hours. Asking for more workers does not make them come faster; a large memory request slows them
  down (2 GB and 4 GB workers arrive in about a minute, 10 GB ones in five).
- Throughput: a worker processes 2,700 to 3,600 events per second of nominal 2024 skims (31 to 37 s
  per 100000-event chunk), about a third of a laptop core, and about 320 per second with every
  systematic (5 minutes per chunk). Reading from FNAL EOS instead of Wisconsin gives the same time.
- The end of a run: the slowest chunk and the last merges; a paused or restarted worker there costs
  minutes. Smaller chunks end sooner.
- After saving: closing the workers takes about 1.5 minutes. Once the log says `All output saved`,
  Ctrl+C is safe.
- Reference: `all` for 2024 (every signal point, 368 M events) took 44 minutes (10 waiting for the
  first worker, 1.5 preprocessing, 26 processing on 60 to 90 workers, 1.5 saving, 3.5 closing),
  against 8.2 hours on a laptop.
- Tested and not worth changing: skipping the preprocessing (it overlaps the wait for workers),
  shipping only the era's payloads (the inputs transfer in seconds), multi-core workers (they need
  several free cores in one slot of an almost full pool).

## Laptop and Condor agree

The smearing random numbers (stochastic JER, Run 3 electron smearing, high-pT muon smearing) are the
same on macOS and Linux, so a laptop run and a Condor run of the
same code and inputs give the same histograms (14 unweighted cutflows equal, yields to 1e-16).
Outputs made before the switch to correctionlib's `normal` distribution differ between the two
platforms at the level of the smearing's statistical noise.

## Logs

- Dask worker logs: `/uscmst1b_scratch/lpc1/3DayLifetime/$USER/dask-logs/worker-<JOBID>.0.{out,err}`,
  deleted after 3 days. A few workers failing with "Nanny failed to start" is harmless.
- Job status: lpcjobqueue submits to one of the LPC schedds, so `condor_q -global $USER -totals`.
  A run that ends normally or with Ctrl+C removes its jobs. A killed run (tmux window closed, node
  rebooted) leaves them in the queue: `condor_q -global $USER`, then
  `condor_rm -name <schedd> <ClusterId>` or `condor_rm -name <schedd> $USER`.
- Skim jobs (`bin/skim.py condor` through `bin/lpc_condor.sh`) do not use this environment: each
  runs in its own coffea image from cvmfs. Their logs are `logs/<dataset>_g<gid>.out|.err` and
  `logs/condor.log` in the jobs directory, and the group status is shipped to
  `<campaign root>/<run>/<year>/<era>/<category>/<dataset>/_groups/` ([skimming.md](skimming.md#condor-at-the-lpc)).
