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
```

**Start a tmux session** and the container:
```bash
tmux new -s wr
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
pip install -e .                            # first time only
```

To run over everything for 2024 (all backgrounds including every DY variant, data and signal):
```bash
python bin/run_production.py RunIII2024Summer24 --condor --all-dy # Run background (including all DY variants), data, signal
```
This starts 250 Condor workers and runs nine `run_analysis.py` commands of the full 2024 production on them at once.
It writes one file per sample to `WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/<YYYYMMDD>/`, today's date
(`--dir <tag>` names the directory instead):
data (blinded), tt_tW, Nonprompt, Other, every signal point of the era's mass-point CSV, the default
DY (LO HT-binned) and six other DY setups. It takes 30 to 60 minutes, depending on how fast the pool
hands out workers, and ends with a summary of every command. Each command has its log in `logs/<today's date>/`.

When these have completed, the plotter reads the set back by its directory name:
```bash
cd WR_Plotter
../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261011 --local-plots               # the set rootfiles/Run3/2024/RunIII2024Summer24/20261011/
../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261011 --dy nlo_inc --local-plots  # one of its DY variants instead of the default
```
- `--dir` is the name of the directory under `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/`: the date the production
  ran, or its `--dir`.
- `--dy <variant>` picks a DY variant file of the set, `WRAnalyzer_DYJets_<variant>.root`: `lo_inc`, `nlo_inc`,
  `nlo_mll`, `nlo_njet`, `lo_ht_zpt`, `lo_ht_zpt_reshape` (the `--all-dy` commands). Without it the stack uses the
  default LO HT-binned DY, `WRAnalyzer_DYJets.root`, and stops when that file is missing.

The cutflow tables, signal plots, event-weight plots and DY comparisons take the same `--era`, `--dir` and `--dy`:
[WR_Plotter/docs/plotting.md](https://github.com/UMN-CMS/WR_Plotter/blob/billyDev/docs/plotting.md).

The other years run the same way: one era, several eras, or `all` for every era with data skims.
```bash
python bin/run_production.py Run3Summer22 --condor --all-dy
python bin/run_production.py RunIISummer20UL17 RunIISummer20UL18 --condor --all-dy
python bin/run_production.py all --condor --all-dy
```
Eras: `RunIISummer20UL16APV`, `RunIISummer20UL16`, `RunIISummer20UL17`, `RunIISummer20UL18`, `Run3Summer22`,
`Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`, `RunIII2025Summer25` (data; its MC and
signal are the 2024 samples, `mc_alias` in `wrcoffea/config.yaml`).
Each era writes under its own `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<YYYYMMDD>/` and runs its own DY
variants (`--plan --all-dy` lists an era's commands).

## Run all of 2024 except the extra DY variants
You may wish to run over everything except all the extra DY variants (i.e. LO HT DY only + remaining backgrounds + data + signal):
```bash
python bin/run_production.py RunIII2024Summer24 --condor
```

Or you can analyzer data, background and signal seperately:
```bash
python bin/run_production.py RunIII2024Summer24 --condor --only data      # data only
python bin/run_production.py RunIII2024Summer24 --condor --only bkg       # backgrounds only, LO HT DY
python bin/run_production.py RunIII2024Summer24 --condor --only signal    # every signal point of the mass-point CSV
```

- A refusal that names directories in `~/.local`: delete them and start `./shell` again.
- Progress: `condor_q -global $USER -totals` in a tmux window outside `./shell`.
- Systematics: flags after `--` go to every command,
  `-- --systs lumi pileup sf jer jes jms jmr elescale elesmear muscale musmear lsf scale ps pdf --xrootd-timeout 120`
  (the workers then get 8 GB).

## Locally, in five commands

```bash
git clone --recursive -b billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea
python3.12 -m venv .venv && .venv/bin/pip install -e ".[xrootd,test]"     # details: docs/getting_started.md
voms-proxy-init --rfc --voms cms -valid 192:00                            # grid proxy for the skims
.venv/bin/python bin/run_analysis.py RunIII2024Summer24 DYJets --dir mytag  # one sample, one era
```

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
| [WR_Plotter/docs/plotting.md](https://github.com/UMN-CMS/WR_Plotter/blob/billyDev/docs/plotting.md) | The WR_Plotter commands: stacks, cutflows, signal plots, DY comparisons, systematics tables. |
