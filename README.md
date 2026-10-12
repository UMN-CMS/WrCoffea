# WrCoffea

Search for a right-handed W boson and a heavy neutrino, W_R → ℓN → ℓℓqq' (ee and μμ), in CMS
Run 2 UL and Run 3 data. This repository is the coffea analyzer: NanoAOD (skimmed or not) in,
ROOT histograms with every systematic variation out. The plots, the fits and the limits live in
the [WR_Plotter](WR_Plotter/) submodule.

Two topologies: **resolved** (two tight leptons and two AK4 jets, observable m(ℓℓjj)) and
**boosted** (one tight lepton and one AK8 jet with a lepton inside, observable m(ℓJ)). What the
code does, cut by cut, is in [docs/analysis.md](docs/analysis.md).

## In five commands

```bash
git clone --recursive git@github.com:UMN-CMS/WrCoffea.git && cd WrCoffea
python3.12 -m venv .venv && .venv/bin/pip install -e ".[xrootd,test]"     # details: docs/getting_started.md
voms-proxy-init --rfc --voms cms -valid 192:00                            # grid proxy for the skims
.venv/bin/python bin/run_analysis.py RunIII2024Summer24 DYJets --dir mytag  # one sample, one era
cd WR_Plotter && ../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir mytag --local-plots
```

`.venv/bin/python -m pytest tests -q` runs the analyzer tests offline (about a minute).
Composite modes (`all`, `bkg`, `data`, `mc`, `signal`), eras, systematics and every flag:
[docs/run_analysis.md](docs/run_analysis.md).

## Where things are

| | |
| --- | --- |
| `bin/` | `run_analysis.py` (the analyzer), `skim.py` (skims and filesets), `lpc_condor.sh` |
| `wrcoffea/` | the package: `analyzer.py`, `regions.py`, `systematics.py`, `scale_factors.py`, `config.yaml`, `skim/` |
| `data/` | dataset configs, filesets, correction payloads (`jsonpog/`, `reweights/`), golden JSONs and luminosities, PDF sets, signal mass points |
| `WR_Plotter/` | submodule: stack plots, cutflows, systematics tables, `signal_fitting/` (shapes, fits, Combine limits) |
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
| [plotting.md](docs/plotting.md) | The WR_Plotter commands: stacks, cutflows, DY comparisons, systematics tables, limits. |
| [adding_a_new_era.md](docs/adding_a_new_era.md) | The checklist for a new era. |
| [studies/](docs/studies/) | Comparisons and derivations: the Run 2 analyzer comparison, the signal k-factor, muon momentum, the dataset surveys, the skim format study. |
| [history.md](docs/history.md) | Choices that were retired, and when. |

Reference analysis: CMS-EXO-20-002 (Run 2, 138 fb⁻¹), whose analysis note AN-19-083 and code
(SKFlat HNWRAnalyzer) this analyzer follows and is compared against in
[docs/studies/run2_hnwr_comparison.md](docs/studies/run2_hnwr_comparison.md).
