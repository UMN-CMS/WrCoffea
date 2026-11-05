<div align="center">
  <img src="docs/coffea_logo.svg" alt="Coffea Logo" width="250">
</div>

# WrCoffea Documentation

This repository is the analysis framework for the CMS search for a right-handed W boson and a heavy neutrino, W_R → Nℓ → ℓℓjj (ee and μμ), built on the Coffea columnar toolkit. It turns NanoAOD (skimmed or not) into ROOT histograms with every systematic variation, for data, backgrounds and signal. The plots, fits and limits live in the [WR_Plotter](WR_Plotter/) submodule.

## Getting Started

For first-time setup (cloning, the virtual environment, the grid proxy, the Condor container), see **[Getting Started](docs/getting_started.md)**. In short:

```bash
git clone --recursive -b billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea
python3.12 -m venv .venv && .venv/bin/pip install -e ".[xrootd,test]"
voms-proxy-init --rfc --voms cms -valid 192:00                            # grid proxy for the skims
```

Activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

> **Tip:** Add this line to your `~/.bashrc` to activate automatically on login:
> ```bash
> cd /path/to/WrCoffea && source .venv/bin/activate && cd -
> ```

## Table of Contents
- [Quick Start](#quick-start) – A whole production on LPC Condor
- [Plotting](#plotting) – Plots and tables from a production
- [Drell-Yan Samples](#drell-yan-samples) – The DY setups and how to run one
- [Running on Condor](#running-on-condor) – Single commands and a shared cluster
- [Skimming](#skimming) – The skim campaign
- [Command Reference](#command-reference) – Flags of `run_production.py` and `run_analysis.py`, examples
- [Repository Structure](#repository-structure) – How the codebase is organized
- [Testing](#testing) – The automated test suites
- [Additional Documentation](#additional-documentation) – Detailed guides
- [Update Plotter Submodule](#update-plotter-submodule)

---

## Quick Start

All of 2024 in one command, on LPC Condor. From an LPC login node, with a CMS grid certificate in `~/.globus`.

**Once**, on the node you landed on (the run lives on this node, inside tmux):

```bash
hostname                                    # note it: tmux attach works only on this node
voms-proxy-init --voms cms --valid 192:00   # lets the workers read the skims (8 days)
cd ~/nobackup
git clone -b billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

**Every session**, a tmux session and the container:
```bash
tmux new -s wr
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
pip install -e .                            # first time only
```

**Run** everything for 2024: data, every background with every DY variant, and every signal point:
```bash
python bin/run_production.py RunIII2024Summer24 --condor --all-dy
```

This starts 250 Condor workers, runs nine `run_analysis.py` commands on them at once and stops them. It writes one file per sample to `WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/<YYYYMMDD>/`, today's date (`--dir <tag>` names the directory instead): data (blinded), tt_tW, Nonprompt, Other, every signal point of the era's mass-point CSV, the default DY (LO HT-binned) and six other DY setups. It takes 30 to 60 minutes, depending on how fast the pool hands out workers, and ends with a summary of every command. Each command has its log in `logs/<YYYYMMDD>/`.

Without the extra DY variants (the LO HT DY only, the other backgrounds, data and signal):
```bash
python bin/run_production.py RunIII2024Summer24 --condor
```

Or data, backgrounds and signal separately:
```bash
python bin/run_production.py RunIII2024Summer24 --condor --only data      # data only
python bin/run_production.py RunIII2024Summer24 --condor --only bkg       # backgrounds only, LO HT DY
python bin/run_production.py RunIII2024Summer24 --condor --only signal    # every signal point of the mass-point CSV
```

The other years run the same way: one era, several eras, or `all` for every era with data skims.
```bash
python bin/run_production.py Run3Summer22 --condor --all-dy
python bin/run_production.py RunIISummer20UL17 RunIISummer20UL18 --condor --all-dy
python bin/run_production.py all --condor --all-dy
```

| Eras | Data | Simulation |
|------|------|------------|
| `RunIISummer20UL16APV`, `RunIISummer20UL16`, `RunIISummer20UL17`, `RunIISummer20UL18` | Run 2 UL | the era's own samples |
| `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24` | Run 3 | the era's own samples |
| `RunIII2025Summer25` | 2025 | the 2024 samples (`mc_alias` in `wrcoffea/config.yaml`) |

Each era writes under its own `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<YYYYMMDD>/` and runs its own DY variants (`--plan --all-dy` lists an era's commands).

> **Note:** A refusal that names directories in `~/.local`: delete them and start `./shell` again. Progress: `condor_q -global $USER -totals` in a tmux window outside `./shell`. Systematics: flags after `--` go to every command, e.g. `-- --systs lumi pileup sf jer jes jms jmr elescale elesmear muscale musmear lsf scale ps pdf --xrootd-timeout 120` (the workers then get 8 GB).

### tmux

A production runs for an hour or more, and the Dask scheduler lives in your login session, so the run ends with the session. `tmux` keeps the session alive after you disconnect. Note which node you are on (`hostname`): tmux sessions are local to that node, so you must SSH back to the same node to reattach.

```bash
hostname                                 # note it, e.g. cmslpc320.fnal.gov
tmux new -s wr                           # a new named session
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
python bin/run_production.py RunIII2024Summer24 --condor --all-dy
```

Detach with `Ctrl-b` then `d` (press `Ctrl-b`, release, then press `d`) and log out. To reattach later, SSH to the **same node**:
```bash
ssh cmslpc320.fnal.gov                   # replace with your node
tmux attach -t wr
```

Other useful tmux commands:
- `tmux ls` — list active sessions
- `Ctrl-b` then `d` — detach from the current session
- `tmux kill-session -t wr` — kill a session

---

## Plotting

The plotter reads a production back by its directory name and writes the plots next to it, under `WR_Plotter/plots/<Run>/<Year>/<Era>/<dir>/` (`--eos` sends them to EOS instead):

```bash
cd WR_Plotter
../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261011               # rootfiles/Run3/2024/RunIII2024Summer24/20261011/ -> plots/Run3/2024/RunIII2024Summer24/20261011/stack_dy_lo_ht/
../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261011 --dy nlo_inc  # one of its DY variants instead of the default -> stack_dy_nlo_inc/
```

- `--dir` is the name of the directory under `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/`: the date the production ran, or its `--dir`.
- `--dy <tag>` picks a DY variant file of the set, `WRAnalyzer_DYJets_<tag>.root` (the table below). Without it the stack uses the default LO HT-binned DY, `WRAnalyzer_DYJets.root`, and stops when that file is missing.

The cutflow tables, signal plots, event-weight plots and DY comparisons take the same `--era`, `--dir` and `--dy`. See **[WR Plotter](https://github.com/UMN-CMS/WR_Plotter/blob/billyDev/README.md)** for every script, its options and its outputs.

---

## Drell-Yan Samples

`--all-dy` runs every DY setup of the era; one alone is its command name with `--only`. Each writes its own file into the set, and the plotter reads it with `--dy <tag>`. Every sample is normalized to the inclusive NNLO cross section.

```bash
python bin/run_production.py RunIII2024Summer24 --condor --only dy_nlo_inc              # one DY setup into the set of the day
python bin/run_production.py RunIII2024Summer24 --condor --only dy_zpt dy_zpt_reshape   # the two reweighted ones
```

| `--only` | File in the set | Plotter `--dy` | Sample |
|----------|-----------------|----------------|--------|
| `bkg` | `WRAnalyzer_DYJets.root` | `lo_ht` (default) | LO MadGraph MLM, HT-binned (`DYto2L-4Jets_Bin-HT-*`), the DY of the nominal production |
| `dy_zpt` | `WRAnalyzer_DYJets_lo_ht_zpt.root` | `lo_ht_zpt` | the same with the Z-pT NLO/LO reweight (`--dy-reweight zpt`) |
| `dy_zpt_reshape` | `WRAnalyzer_DYJets_lo_ht_zpt_reshape.root` | `lo_ht_zpt_reshape` | the same with the Z-pT reweight and the data-driven jet-pT reshape (`--dy-reweight zpt+reshape`) |
| `dy_lo_inc` | `WRAnalyzer_DYJets_lo_inc.root` | `lo_inc` | LO MadGraph MLM, inclusive (`DYto2E/Mu-4Jets_Bin-MLL-50`) |
| `dy_nlo_inc` | `WRAnalyzer_DYJets_nlo_inc.root` | `nlo_inc` | NLO aMC@NLO FxFX, inclusive (`DYto2E/Mu-2Jets_Bin-MLL-50`) |
| `dy_nlo_mll` | `WRAnalyzer_DYJets_nlo_mll.root` | `nlo_mll` | NLO POWHEG, m(ll)-binned (`DYto2E/Mu_Bin-MLL-*`) |
| `dy_nlo_njet` | `WRAnalyzer_DYJets_nlo_njet.root` | `nlo_njet` | NLO aMC@NLO FxFX, jet-multiplicity-binned (`DYto2E/Mu-2Jets_Bin-0J/1J/2J`) |

2022 and 2023 also have `dy_lo_njet` and `dy_lo_ptll` (LO, jet- and pT(ll)-binned), the UL eras `dy_nlo_njet_mll` and `dy_lo_njet`; `--plan --all-dy` lists an era's. The pT(ll)-binned NLO sample (`nlo_ptll`) exists but is not run: its slices overlap, so it is not a prediction.

---

## Running on Condor

One `run_analysis.py` command on workers of its own, or a cluster that several commands share. Both need the [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue) Apptainer container of the Quick Start.

```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12                     # enter the container
python bin/run_analysis.py RunIII2024Summer24 all --condor                      # everything on Condor, one command
python bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --condor      # one sample

python bin/condor_cluster.py --workers 200                                      # tmux window 1: the workers
python bin/run_analysis.py RunIII2024Summer24 bkg --scheduler condor_scheduler.json --dir <tag>      # window 2: commands on them
python bin/run_production.py RunIII2024Summer24 --scheduler condor_scheduler.json --dir <tag>        # or a whole production
```

See **[Running on Condor](docs/condor.md)** for full documentation: setup, worker and chunk-size defaults, what sets the time of a run, and log locations.

---

## Skimming

The skim campaign applies a loose preselection and a branch policy to NanoAOD, writes the skims to a remote store, validates the campaign and writes the analyzer filesets. `bin/skim.py` has one subcommand per step; the Condor jobs go through `bin/lpc_condor.sh`.

```bash
python3 bin/skim.py cuts                                                              # the skim selection and branch policy
python3 bin/skim.py plan --campaign v2 --plan-dir <plan dir> --era RunIII2025Summer25  # the campaign plan from the configs and DBS
bin/lpc_condor.sh --jobs-dir <jobs dir> --era RunIII2025Summer25 --remote <root://...> --submit   # one Condor job per group
```

| Subcommand | Does |
|------------|------|
| `plan` | build the campaign plan from the config JSONs (DBS and branch sampling) |
| `run`, `condor` | process the plan groups locally, or write (and submit) one Condor job per pending group |
| `status`, `bad-files` | progress and problems from the group status files |
| `validate` | whole-campaign validation against the plan, DBS and the golden JSON |
| `upload` | copy finished parts to remote storage with checksum verification |
| `fileset`, `unskimmed-fileset`, `alias-filesets` | the analyzer fileset JSONs: skimmed, unskimmed, and the MC of a data-only era |

See **[Skimming](docs/skimming.md)** for full documentation: the 2025 data recipe, the selection, every subcommand, verification and the Condor jobs.

---

## Command Reference

### run_production.py Flags

| Flag | Arguments | Description |
|------|-----------|-------------|
| `eras` | `<era> ...` | **Required positional.** One or more eras, or `all` for every era with a data fileset |
| `--condor` | | Start Condor workers (inside `./shell`), run the commands on them at once, stop them |
| `--scheduler` | `<file>` | Run on the workers of a running `bin/condor_cluster.py` instead of starting workers |
| `--all-dy` | | Also run every other DY setup of the era (default: `data`, `bkg` and `signal`) |
| `--only` | `<name> ...` | Only these commands, from the whole plan (`--plan --all-dy` lists the names) |
| `--skip` | `<name> ...` | Not these commands |
| `--plan` | | Print the commands and exit |
| `--dir` | `<tag>` | Output directory under `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/` and log directory under `logs/` (default: today's date) |
| `--workers` | `<int>` | Condor workers to start (default: 250) |
| `--worker-memory` | `<size>` | Memory of each worker (default: 6GB; 8GB when the commands get `--systs`) |
| `--worker-wait-timeout` | `<int>` | Seconds to wait for the first worker (default: 3600; a busy pool may need 21600) |
| `--jobs` | `<int>` | Commands at a time (default: every command of an era at once on Condor, one locally) |
| `-- <flags>` | | Everything after `--` goes to every `run_analysis.py` command (`--signal-points` and `--mwr-range` to `signal` only) |

Without `--condor` or `--scheduler` the commands run one after the other on the local machine.

### run_analysis.py Flags

| Flag | Arguments | Description |
|------|-----------|-------------|
| `era` | `<era>` | **Required positional.** Campaign to analyze (`--list-eras`) |
| `sample` | `<sample>` | **Required positional.** A sample (`DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal`, or a `--split-groups` subgroup) or a composite mode (`all`, `data`, `bkg`, `mc`, `signal`) |
| `--mass` | `WR<wr>_N<n>` | Signal mass point. **Required for `Signal`** (`--list-masses`) |
| `--signal-points` | `all` \| `grid` \| `<point> ...` | Signal points of a composite mode: every point of `data/signal_points/<era>_mass_points.csv` (default), the 9-point grid, or a list |
| `--mwr-range` | `<min> <max>` | Keep only the signal points with min ≤ m(W_R) ≤ max |
| `--dy` | `<variant>` | DY variant (`dy_variants` per era in `config.yaml`); default `lo_ht` |
| `--dy-reweight` | `none` \| `zpt` \| `zpt+reshape` | Corrections of the LO HT-binned DY: none (default), the Z-pT reweight, or both |
| `--no-signal-kfactor` | | Switch the signal k(m(ℓN))/⟨k⟩ reweight off |
| `--region` | `resolved` \| `boosted` \| `both` | Topologies to fill (default: both) |
| `--unblind` | | Fill the data signal regions and full cutflows (Run 2 only) |
| `--study` | `<study> ...` | Extra regions: `transfer-factor`, `dy-cr-split`, `no-lsf` |
| `--split-groups` | `<group> ...` | Composite runs: `tt_tW`, `Nonprompt` or `Other` as one file per subgroup |
| `--save-events` | | Also write the signal-region (mass, weight) TTrees |
| `--systs` | `<family> ...` | Systematic variations: `lumi pileup prefire sf jer jes jms jmr elescale elesmear muscale musmear dyzpt lsf dyreshape scale ps pdf` |
| `--syst-observables` | `fit` \| `all` \| `<name> ...` | Histograms that receive the variations (default: the fit observables) |
| `--omit-systs` | `<source> ...` | Leave out named sources of an enabled family (recorded in the manifest) |
| `--dir` | `<tag>` | Output subdirectory under `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/` |
| `--name` | `<suffix>` | `WRAnalyzer_<suffix>_<sample>.root` |
| `--unskimmed` | | Read the unskimmed filesets (the original NanoAOD files) |
| `--fileset` | `<path>` | An explicit fileset JSON |
| `--debug` | | Run without saving histograms |
| `--condor` | | Run on Condor at the LPC (inside `./shell`) with workers of its own |
| `--scheduler` | `<file>` | Run on the workers of `bin/condor_cluster.py` (implies `--condor`) |
| `--max-workers` | `<int>` | Dask workers (local: 3; Condor: 50 single sample, 200 composite, 500 unskimmed composite) |
| `--worker-memory` | `<size>` | Memory of each Condor worker (default: 6GB; 8GB with `--systs`) |
| `--worker-wait-timeout` | `<int>` | Seconds to wait for the first Condor worker (default: 3600) |
| `--threads-per-worker` | `<int>` | Threads per Dask worker for local runs |
| `--chunksize` | `<int>` | Events per chunk (default: 100000 with `--condor`, 250000 locally) |
| `--maxchunks`, `--maxfiles` | `<int>` | Chunks per file, files per dataset (default: all). Use `1` for a quick test |
| `--xrootd-timeout` | `<int>` | XRootD read timeout per chunk (default: 60 with `--condor`, 10 locally); raise it for systematics runs |
| `--xrd-fallback`, `--xrd-fallback-timeout`, `--xrd-fallback-retries-per-redirector`, `--xrd-fallback-sleep` | | Redirector fallback for unskimmed preprocessing (on by default with `--unskimmed`) |
| `--list-eras`, `--list-samples`, `--list-masses` | | Print the choices and exit |
| `--preflight-only` | | Validate the fileset path, schema and selection, then exit |

### Examples

```bash
# Composite modes (sequential locally, parallel on Condor with --condor)
python3 bin/run_analysis.py RunIII2024Summer24 all --dir 20261011                     # everything, LO HT-binned DY
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir 20261011                     # the four background groups
python3 bin/run_analysis.py RunIII2024Summer24 data --dir 20261011                    # the two data streams
python3 bin/run_analysis.py RunIII2024Summer24 signal --dir 20261011                  # every signal point
python3 bin/run_analysis.py RunIII2024Summer24 signal --signal-points grid --dir 20261011   # the 9-point grid

# Single samples
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dir 20261011
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --dir 20261011           # WRAnalyzer_DYJets_nlo_inc.root
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dy-reweight zpt+reshape --dir 20261011   # WRAnalyzer_DYJets_lo_ht_zpt_reshape.root
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --dir 20261011

# Systematics
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf jer jes jms jmr elescale elesmear muscale musmear lsf scale ps pdf --xrootd-timeout 120 --dir 20261011_systs

# Condor (inside ./shell)
python3 bin/run_analysis.py RunIII2024Summer24 all --condor --dir 20261011
python3 bin/run_analysis.py RunIII2024Summer24 all --condor --systs lumi pileup sf --xrootd-timeout 120 --dir 20261011_systs

# Run 2 data unblinded; one topology; a quick test; validate without processing
python3 bin/run_analysis.py RunIISummer20UL18 data --unblind --dir 20261011
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved --dir 20261011
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --maxfiles 1 --maxchunks 1 --dir smoke
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
```

Output: `WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>.root`, each with a JSON manifest next to it; one file per signal point, `WRAnalyzer_signal_WR<wr>_N<n>.root`. Data is blinded by default (no signal-region folders; `--unblind` is Run 2 only). See **[Running the Analyzer](docs/run_analysis.md)** for the output layout, blinding and systematics production in full.

---

## Repository Structure

```
WR_Plotter/  # Submodule: plots, cutflows, DY comparisons, systematics tables, signal plots
bin/         # User-facing CLI scripts
wrcoffea/    # Installable Python package (analysis code, utilities, config)
data/        # Configuration files, filesets, correction payloads, luminosities
docs/        # Documentation (markdown guides)
scripts/     # Derivations and campaigns behind the inputs
tests/       # Automated test suite (pytest)
```

**`bin/`** - Production Scripts
- [`run_production.py`](bin/run_production.py) - A whole production of an era: starts the workers, runs every command at once, summarizes
- [`run_analysis.py`](bin/run_analysis.py) - The analyzer: one sample or a composite mode, locally or on Condor
- [`condor_cluster.py`](bin/condor_cluster.py) - Condor workers that several commands share
- [`skim.py`](bin/skim.py) - The skim campaign (`plan`, `run`, `condor`, `status`, `validate`, `upload`, `fileset`, ...)
- [`lpc_condor.sh`](bin/lpc_condor.sh) - The Condor jobs of the skim campaign at the LPC

**`wrcoffea/`** - Installable Python Package
- [`analyzer.py`](wrcoffea/analyzer.py) - The Coffea processor: object selection, the resolved and boosted regions, histograms, cutflows, systematic passes
- [`regions.py`](wrcoffea/regions.py) - The region registry, study variants and the blinding policy
- [`systematics.py`](wrcoffea/systematics.py) - The systematic families and sources, and the production configuration
- [`histograms.py`](wrcoffea/histograms.py) - Histogram specifications, cutflows, event-weight histograms
- [`save_hists.py`](wrcoffea/save_hists.py) - Merging, coverage checks, ROOT output and manifests
- [`scale_factors.py`](wrcoffea/scale_factors.py) - correctionlib scale factors, JEC/JER, scale and smearing
- [`muon_momentum.py`](wrcoffea/muon_momentum.py), [`MuonScaRe.py`](wrcoffea/MuonScaRe.py) - The muon momentum corrections
- [`dy_reweight.py`](wrcoffea/dy_reweight.py), [`dy_reshape.py`](wrcoffea/dy_reshape.py) - The DY Z-pT reweight and jet-pT reshape
- [`signal_kfactor.py`](wrcoffea/signal_kfactor.py) - The signal k(m(ℓN))/⟨k⟩ reweight
- [`pdf_weights.py`](wrcoffea/pdf_weights.py) - PDF member weights
- [`analysis_config.py`](wrcoffea/analysis_config.py), [`config.yaml`](wrcoffea/config.yaml) - Luminosities, cuts, payload maps, selection names
- [`cli_utils.py`](wrcoffea/cli_utils.py), [`era_utils.py`](wrcoffea/era_utils.py), [`fileset_validation.py`](wrcoffea/fileset_validation.py), [`xrootd_fallback.py`](wrcoffea/xrootd_fallback.py) - CLI plumbing, era mapping, fileset validation, XRootD fallback
- [`skim/`](wrcoffea/skim/) - The skim campaign: plan, fetch, selection, writer, validation, upload, filesets

**`data/`** - Configuration and Metadata
- `configs/` - Per-era dataset configurations with cross sections (JSON)
- `filesets/` - Per-era file lists, skimmed and unskimmed (JSON)
- `signal_points/` - The signal mass points per era (CSV)
- `lumis/` - Golden JSONs and the processed luminosities
- `jsonpog/`, `reweights/`, `pdf_sets/`, `muon_highpt/` - Correction payloads: POG JSONs, the DY reweights, the PDF sets, the high-pT muon corrections

**`WR_Plotter/`** - Plotting Submodule
- Separate repository, branch `billyDev`; the analyzer's histograms are written into it
- See [`WR_Plotter/README.md`](https://github.com/UMN-CMS/WR_Plotter/blob/billyDev/README.md) for every script

**`scripts/`** - Derivations and Campaigns
- Cross sections (`genxsec_campaign.py`), luminosities (`processed_lumi.py`), PDF sets, the DY Z-pT reweight and jet-pT reshape derivations, the signal k-factor campaign (`signal_kfactor/`), the Run 2 payload converters

**`docs/`** - Documentation
- [`getting_started.md`](docs/getting_started.md) - Environment on a laptop or at the LPC, grid proxy, a first run
- [`run_analysis.md`](docs/run_analysis.md) - Modes, flags, outputs, blinding, systematics production
- [`condor.md`](docs/condor.md) - HTCondor at the LPC: setup, the shared cluster, measured defaults, logs
- [`skimming.md`](docs/skimming.md) - The skim campaign: selection, branch policy, workflow, verification, Condor
- [`systematics_status.md`](docs/systematics_status.md) - Which systematic uncertainties are done per era

**`tests/`** - Automated Test Suite
- Fully offline; run with pytest (see [Testing](#testing))

---

## Testing

Both suites run offline:
```bash
.venv/bin/python -m pytest tests -q                       # the analyzer, about a minute
cd WR_Plotter && ../.venv/bin/python -m pytest tests -q   # the plotter, under a minute
```

### Quick Local Validation

The full chain on one file and one chunk of one sample, before a large Condor run:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --maxfiles 1 --maxchunks 1 --dir smoke
```

The data/MC ratio of such a run means nothing: one file carries the full sum of weights.

---

## Additional Documentation

- **[Getting Started](docs/getting_started.md)** - Environment on a laptop or at the LPC, grid proxy, a first run
- **[Running the Analyzer](docs/run_analysis.md)** - Modes, flags, outputs, blinding, systematics production
- **[Running on Condor](docs/condor.md)** - HTCondor at the LPC: setup, the shared cluster, measured defaults, logs
- **[Skimming](docs/skimming.md)** - The skim campaign: selection, branch policy, workflow, verification, Condor
- **[Systematics Status](docs/systematics_status.md)** - Which systematic uncertainties are done per era
- **[WR Plotter](https://github.com/UMN-CMS/WR_Plotter/blob/billyDev/README.md)** - Every plotting script, its options and outputs

---

## Update Plotter Submodule

To point WrCoffea at the latest plotter commit:
```bash
cd WR_Plotter
git switch billyDev
git pull
cd ..
git commit -am "Point WR_Plotter at $(git -C WR_Plotter rev-parse --short HEAD)"
git push
```

To work on a new feature branch in the submodule:
```bash
cd WR_Plotter
git checkout -b my_feature_branch
git push -u origin my_feature_branch
cd ..
```
