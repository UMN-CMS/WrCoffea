# WrCoffea

## Run all of 2024 on LPC Condor

From an LPC login node, with a CMS grid certificate in `~/.globus`. Tested from a fresh clone on
2026-10-15; the measurements behind the settings are in [docs/condor.md](docs/condor.md).

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

**tmux window 1**, the Condor workers (they stop after 30 idle minutes, or on Ctrl+C):

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
pip install -e .                            # first time only
python bin/condor_cluster.py --workers 250
```

**tmux window 2** (Ctrl-b c), all of 2024 on those workers, seven commands at once:

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
TAG=$(date +%Y%m%d)_2024
mkdir -p logs
i=0
for args in "all --signal-points all" \
            "DYJets --dy-reweight zpt" "DYJets --dy-reweight zpt+reshape" \
            "DYJets --dy lo_inc" "DYJets --dy nlo_inc" "DYJets --dy nlo_mll" "DYJets --dy nlo_njet"; do
  i=$((i + 1))
  python bin/run_analysis.py RunIII2024Summer24 $args --scheduler condor_scheduler.json --dir $TAG > logs/$TAG.$i.log 2>&1 &
done
wait
grep -h "Execution took" logs/$TAG.*.log    # 7 lines when all succeeded
```

This writes 149 files to `WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/$TAG/`: data (blinded),
tt_tW, Nonprompt, Other, all 137 signal points, the default DY and six other DY setups. It takes 30 to
60 minutes, depending on how fast the pool hands out workers; then press Ctrl+C in window 1.

- No worker after an hour (a busy pool, or low priority after heavy use): start both the cluster and
  the runs with `--worker-wait-timeout 21600`.
- A refusal that names directories in `~/.local`: delete them and start `./shell` again.
- Progress: `condor_q -global $USER -totals` in a tmux window outside `./shell`.
- Data, backgrounds and signal only: the first command alone. Systematics: start the cluster with
  `--worker-memory 8GB`.

## What this is

Search for a right-handed W boson and a heavy neutrino, W_R → ℓN → ℓℓqq' (ee and μμ), in CMS
Run 2 UL and Run 3 data. This repository is the coffea analyzer: NanoAOD (skimmed or not) in,
ROOT histograms with every systematic variation out. The plots, the fits and the limits live in
the [WR_Plotter](WR_Plotter/) submodule.

Two topologies: **resolved** (two tight leptons and two AK4 jets, observable m(ℓℓjj)) and
**boosted** (one tight lepton and one AK8 jet with a lepton inside, observable m(ℓJ)). What the
code does, cut by cut, is in [docs/analysis.md](docs/analysis.md).

## Locally, in five commands

```bash
git clone --recursive -b billy_dev git@github.com:UMN-CMS/WrCoffea.git && cd WrCoffea
python3.12 -m venv .venv && .venv/bin/pip install -e ".[xrootd,test]"     # details: docs/getting_started.md
voms-proxy-init --rfc --voms cms -valid 192:00                            # grid proxy for the skims
.venv/bin/python bin/run_analysis.py RunIII2024Summer24 DYJets --dir mytag  # one sample, one era
cd WR_Plotter && ../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir mytag --local-plots
```

The work happens on branch `billy_dev`; `main` is far behind and has the old documentation.
`.venv/bin/python -m pytest tests -q` runs the analyzer tests offline (967 tests, about a minute).
Composite modes (`all`, `bkg`, `data`, `mc`, `signal`), eras, systematics and every flag:
[docs/run_analysis.md](docs/run_analysis.md).

## Where things are

| | |
| --- | --- |
| `bin/` | `run_analysis.py` (the analyzer), `condor_cluster.py` (Condor workers shared by several runs), `skim.py` (skims and filesets), `lpc_condor.sh` |
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
| [analysis.md](docs/analysis.md) | What does the analysis do: filters, objects, triggers, regions, weights, blinding, systematics, open items? |
| [run_analysis.md](docs/run_analysis.md) | How do I run the analyzer: modes, flags, outputs, systematics production? |
| [corrections.md](docs/corrections.md) | Which correction payload does each era use, where it comes from, what is a placeholder? |
| [systematics_status.md](docs/systematics_status.md) | Which systematic uncertainties are finished for Run 2 and Run 3, and which are still placeholders or missing? |
| [luminosity.md](docs/luminosity.md) | Golden JSONs, the brilcalc recipe, the processed luminosity of every era, the uncertainties. |
| [inputs.md](docs/inputs.md) | Dataset configs, how cross sections are built, skimmed and unskimmed filesets. |
| [skimming.md](docs/skimming.md) | The skim campaign: selection, branch policy, workflow, verification, Condor at the LPC. |
| [condor.md](docs/condor.md) | Running the analyzer on HTCondor at the LPC. |
| [WR_Plotter/docs/plotting.md](WR_Plotter/docs/plotting.md) | The WR_Plotter commands: stacks, cutflows, DY comparisons, systematics tables, limits. |
| [adding_a_new_era.md](docs/adding_a_new_era.md) | The checklist for a new era. |
| [studies/](docs/studies/) | Comparisons and derivations: the Run 2 analyzer comparison, the signal k-factor, muon momentum, the UL cross-section campaign, the skim format study. |

Reference analysis: CMS-EXO-20-002 (Run 2, 138 fb⁻¹), whose analysis note AN-19-083 and code
(SKFlat HNWRAnalyzer) this analyzer follows and is compared against in
[docs/studies/run2_hnwr_comparison.md](docs/studies/run2_hnwr_comparison.md).
