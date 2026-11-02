# WrCoffea

## Run all of 2024 on LPC Condor

From an LPC login node, with a CMS grid certificate in `~/.globus`. The measurements behind the
settings are in [docs/condor.md](docs/condor.md).

**Once**, on the node you landed on (the run lives on this node, inside tmux):

```bash
hostname                                    # note it: tmux attach works only on this node
voms-proxy-init --voms cms --valid 192:00   # lets the workers read the skims (8 days)
cd ~/nobackup
git clone -b billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
tmux new -s wr
```

**Inside tmux**, the container and one command. It starts 250 Condor workers, runs the
`run_analysis.py` commands of the production on them at once and stops them:

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
pip install -e .                                                          # first time only
python bin/run_production.py RunIII2024Summer24 --condor                  # all of 2024: seven commands, 149 files
python bin/run_production.py RunIII2024Summer24 --condor --only nominal   # without the DY variants: one command, 143 files
```

Both write to `WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/<today's date>/`: data (blinded),
tt_tW, Nonprompt, Other, all 137 signal points and the default DY (LO HT-binned); the full
production adds six other DY setups. It takes 30 to 60 minutes, depending on how fast the pool
hands out workers, and ends with a summary of every command. Each command has its log in
`logs/<today's date>/`.

- `--plan` prints the seven commands without running them; `--dir TAG` names the output directory.
- A failed command: the summary names it and prints the line that reruns it alone (`--only NAME`).
- No worker after an hour (a busy pool, or low priority after heavy use): `--worker-wait-timeout 21600`.
- A refusal that names directories in `~/.local`: delete them and start `./shell` again.
- Progress: `condor_q -global $USER -totals` in a tmux window outside `./shell`.
- Systematics: flags after `--` go to every command,
  `-- --systs lumi pileup sf jer jes jms jmr elescale elesmear muscale musmear lsf scale ps pdf --xrootd-timeout 120`
  (the workers then get 8 GB).

## What this is

Search for a right-handed W boson and a heavy neutrino, W_R → ℓN → ℓℓqq' (ee and μμ), in CMS
Run 2 UL and Run 3 data. This repository is the coffea analyzer: NanoAOD (skimmed or not) in,
ROOT histograms with every systematic variation out. The plots, the fits and the limits live in
the [WR_Plotter](WR_Plotter/) submodule.

Two topologies: **resolved** (two tight leptons and two AK4 jets, observable m(ℓℓjj)) and
**boosted** (one tight lepton and one AK8 jet with a lepton inside, observable m(ℓJ)).

## Locally, in five commands

```bash
git clone --recursive -b billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea
python3.12 -m venv .venv && .venv/bin/pip install -e ".[xrootd,test]"     # details: docs/getting_started.md
voms-proxy-init --rfc --voms cms -valid 192:00                            # grid proxy for the skims
.venv/bin/python bin/run_analysis.py RunIII2024Summer24 DYJets --dir mytag  # one sample, one era
cd WR_Plotter && ../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir mytag --local-plots
```

The work happens on branch `billy_dev`; `main` is far behind and has the old documentation.
`.venv/bin/python -m pytest tests -q` runs the analyzer tests offline (about two minutes).
Composite modes (`all`, `bkg`, `data`, `mc`, `signal`), eras, systematics and every flag:
[docs/run_analysis.md](docs/run_analysis.md).

## Where things are

| | |
| --- | --- |
| `bin/` | `run_analysis.py` (the analyzer), `run_production.py` (every command of an era's production), `condor_cluster.py` (Condor workers shared by several runs), `skim.py` (skims and filesets), `lpc_condor.sh` |
| `wrcoffea/` | the package: `analyzer.py`, `regions.py`, `systematics.py`, `scale_factors.py`, `config.yaml`, `skim/` |
| `data/` | dataset configs, filesets, correction payloads (`jsonpog/`, `reweights/`), golden JSONs and luminosities, PDF sets, signal mass points |
| `WR_Plotter/` | submodule: stack plots, cutflows, systematics tables |
| `scripts/` | derivations and campaigns: cross sections, DY Z-pT reweight and jet-pT reshape, luminosity, PDF sets, payload conversions |
| `tests/` | pytest, fully offline |

Outputs go to `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>.root`, each with a
JSON manifest next to it. Data is blinded by default (no signal-region folders; `--unblind` is
Run 2 only).

## Documentation

| Page | Answers |
| --- | --- |
| [getting_started.md](docs/getting_started.md) | How do I set up the environment on a laptop or at the LPC, and check it works? |
| [run_analysis.md](docs/run_analysis.md) | How do I run the analyzer: modes, flags, outputs, systematics production? |
| [systematics_status.md](docs/systematics_status.md) | Which systematic uncertainties are finished for Run 2 and Run 3, and which are still placeholders or missing? |
| [skimming.md](docs/skimming.md) | The skim campaign: selection, branch policy, workflow, verification, Condor at the LPC. |
| [condor.md](docs/condor.md) | Running the analyzer on HTCondor at the LPC. |
| [WR_Plotter/docs/plotting.md](WR_Plotter/docs/plotting.md) | The WR_Plotter commands: stacks, cutflows, DY comparisons, systematics tables. |

Reference analysis: CMS-EXO-20-002 (Run 2, 138 fb⁻¹), whose analysis note AN-19-083 and code
(SKFlat HNWRAnalyzer) this analyzer follows.
