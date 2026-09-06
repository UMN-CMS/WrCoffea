# Running the Analyzer

## Basic Usage

Run the analyzer by specifying an era and sample:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 all                             # everything
python3 bin/run_analysis.py RunIII2024Summer24 mc                              # backgrounds + signal
python3 bin/run_analysis.py RunIII2024Summer24 bkg                             # backgrounds only
python3 bin/run_analysis.py RunIII2024Summer24 data                            # data only
python3 bin/run_analysis.py RunIII2024Summer24 signal                          # signal only
```

By default, processing runs locally with 3 Dask workers. Use `--condor` for HTCondor at LPC (see [condor.md](condor.md)).

Available eras: `RunIISummer20UL18`, `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`.

### Composite Modes

Composite modes process multiple samples. Locally, they run sequentially (one sample at a time, reusing the same Dask cluster). On Condor (`--condor`), all samples are processed in parallel.

| Mode | Samples |
|------|---------|
| `all` | EGamma, Muon, DYJets, tt_tW, Nonprompt, Other, Signal |
| `data` | EGamma, Muon |
| `bkg` | DYJets, tt_tW, Nonprompt, Other |
| `signal` | Signal (default subset of mass points) |
| `mc` | DYJets, tt_tW, Nonprompt, Other, Signal |

Single samples: `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal`.

```bash
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir my_study
python3 bin/run_analysis.py RunIII2024Summer24 all --systs lumi pileup sf
python3 bin/run_analysis.py RunIII2024Summer24 all --condor              # parallel on Condor
```

When a sample fails during local composite processing, the error is logged and the next sample continues.

### Output

ROOT histograms are saved to:
```
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/WRAnalyzer_<Sample>.root
```

Examples:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_DYJets.root
WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/WRAnalyzer_signal_WR4000_N2100.root
```

Use `--dir` to create a subdirectory, `--name` to modify the filename:
```bash
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir my_study --name test
# -> WR_Plotter/rootfiles/Run3/2022/Run3Summer22EE/my_study/WRAnalyzer_test_DYJets.root
```

### Region Selection

By default, both resolved and boosted histograms are filled. Use `--region` to process only one:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --region boosted
```

### Systematics

Use `--systs` to produce systematic-varied histograms alongside the nominal. Systematics are only applied to MC samples; data always produces nominal-only histograms. The full list of sources lives in `wrcoffea/systematics.py` (the systematic registry); the CLI choices, the analyzer, and the ROOT writer all read from it.

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf jer jes elescale elesmear muscale musmear
```

| Family | Kind | Variations produced | Description |
|--------|------|-------------------|-------------|
| `lumi` | normalization | LumiUp, LumiDown | Luminosity uncertainty (flat ±1.4% for Run3, ±2.5% for UL18). Never filled at event level: the writer scales the nominal histogram at export |
| `pileup` | weight | PileupUp, PileupDown | Pileup reweighting up/down from correctionlib |
| `sf` | weight | MuonRecoSf, MuonIdSf, MuonIsoSf, MuonTrigSf, LooseMuonRecoSf, LooseMuonIdSf, ElectronRecoSf, ElectronIdSf, ElectronTrigSf, LooseElectronRecoSf (each Up/Down) | Lepton scale-factor uncertainties, one source per component. Only the components that enter a region are produced for it |
| `jer` | shape | JERUp, JERDown | Jet energy resolution; selection re-run on smeared jets (every era with a `JR_tag_MC` in `config.yaml`) |
| `jes` | shape | JESUp, JESDown (one pair per entry of `jes_unc_mapping` in `config.yaml`) | Jet energy scale; selection re-run on shifted jets. All eras currently map the `Total` source; UL18 also ships the `Regrouped_*` sources if per-source nuisances are wanted |
| `elescale`, `elesmear` | shape | EleScaleUp/Down, EleSmearUp/Down | Electron energy scale and resolution (Run 3 eras only) |
| `muscale`, `musmear` | shape | MuScaleUp/Down, MuSmearUp/Down | Muon momentum scale and resolution. **Placeholder**: the TuneP branch uses a flat 5%/TeV scale and smear equals nominal |
| `lsf` | weight | LsfSfUp, LsfSfDown | Flat LSF3 > 0.75 efficiency scale factor by lepton-in-jet flavour (AN-19-083 Table 60), boosted SR and flavour CR only. Only eras with an `lsf_sf` entry in `config.yaml` (UL18) |
| `dyreshape` | weight | DYReshapeStat, DYReshapeChannel (each Up/Down) | Statistical and ee-vs-mumu uncertainties of the data-driven DY reshape vs jet pT (`scripts/derive_dy_reshape.py`). Only eras with a `dy_reshape` payload in `config.yaml`; needs the Z-pT reweighting on |
| `dyzpt` | weight | DYZptQcdScale, DYZptQcdPdf, DYZptQcdAlphaS, DYZptEw1, DYZptEw2, DYZptEw3 (each Up/Down) | Uncertainties of the DY Z-pT reweighting (QCD NLO/LO ratio and EW correction, ported from the Run 2 HNWRAnalyzer, see `docs/run2_hnwr_comparison.md`). Only eras with a `dy_zpt_reweight` payload in `config.yaml` (UL18). Non-trivial for `DYJets`; other MC carries nominal copies |

A family the era cannot produce (for example `elescale` for UL18) is an error at start-up, not a silent skip. The same holds for a single source whose correction inputs are missing: a family whose payload is absent for the era refuses to run until the omission is recorded explicitly with `--omit-systs <Source>`. Omitted sources are listed in the manifest and never presented as coverage. (Every configured era currently has all `sf` payloads; the UL18 electron trigger one is the Run 2 HEEP trigger-OR measurement, see `docs/run2_hnwr_comparison.md`.) The registry keeps two separate questions apart: whether a source can physically change a region (`applies`, for example muon scale factors never touch an `ee` region) and whether its inputs exist for the era (`available`). Only the first makes a variation "unaffected"; the second, unresolved, is an error.

**Which histograms get variations.** By default every histogram carries the Up/Down entries. If memory or throughput becomes a problem, `--syst-observables fit` restricts the variations to the fit observables (`mass_dilepton`, `mass_fourobject`, `mass_twoobject`), and an explicit list of histogram names restricts them further; Nominal is always filled everywhere.

**Coverage bookkeeping.** Each chunk records which (region, variation) pairs it completed, together with the production configuration (families, omissions, observables); chunks or datasets produced with different configurations are never merged. The set of variations a dataset must provide comes from the registry, so a variation nobody produced is an error. A shape variation is only accepted for a chunk when every populated affected collection carries the shifted fields (AK4 and AK8 jets both, including the varied mass). The configured regions are part of the record too: a run with `--region both` or `--region boosted` fails immediately if the input has no `FatJet` collection or no `FatJet_lsf3` branch, instead of silently dropping the boosted regions, and the writer refuses a dataset that never filled a configured region. Use `--region resolved` for inputs without the boosted branches. At save time every dataset must have completed every applicable variation in every chunk, otherwise the writer raises instead of producing a partial template. A chunk with no affected objects (for example no electrons under an electron-scale shift) fills its nominal under the variation label and counts as complete. When datasets are merged into a group, a dataset for which the registry says a source does not apply contributes its nominal to that variation.

Each variation is written under `syst_<name>_<region>/`. Variations that do not apply to a region (muon scale factors in an `ee` region, say) are not written as empty templates; they are listed as `unaffected` in the manifest. A JSON manifest `WRAnalyzer_<sample>_systematics.json` next to each ROOT file records the requested families, the observables covered, the datasets and chunk counts, and per region the produced, normalization, and unaffected variations.

---

## Flag Reference

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `era` | Campaign to analyze (use `--list-eras` to see options) |
| `sample` | Sample or composite mode to analyze (use `--list-samples` to see options) |

### Discovery

| Flag | Description |
|------|-------------|
| `--list-eras` | Print available eras and exit |
| `--list-samples` | Print available samples and exit |
| `--list-masses` | Print available signal mass points for the given era (or all eras) and exit |
| `--preflight-only` | Validate fileset path/schema and selection, then exit without processing |

### Sample Selection

| Flag | Description |
|------|-------------|
| `--mass MASS` | Signal mass point (e.g., `WR4000_N2100`). **Required for Signal sample** |
| `--dy VARIANT` | DY sample variant (only valid for DYJets). Available variants are configured per era in `config.yaml`; use `lo_ht` for RunIISummer20UL18 HT-binned, `nlo_inc` for NLO inclusive |
| `--region {resolved,boosted,both}` | Analysis region to run (default: `both`) |
| `--no-dy-zpt-reweight` | Do not apply the DY Z-pT (QCD × EW) reweighting to `DYJets` for eras with a payload in `config.yaml` (currently UL18). Applied by default; incompatible with `--systs dyzpt` |
| `--xrootd-timeout SECONDS` | XRootD read timeout per chunk (default 60 with `--condor`, 10 locally). Raise it when streaming skims over a high-latency link and chunks fail with "Operation expired" |
| `--no-dy-reshape` | Do not apply the data-driven DY reshape vs jet pT for eras with a `dy_reshape` payload in `config.yaml`. Implied by `--no-dy-zpt-reweight` |
| `--no-signal-kfactor` | Do not apply the signal NLO/LO k(m(lN))/⟨k⟩ shape reweighting for eras with a payload in `config.yaml` (currently UL18). Applied by default; see `docs/run2_hnwr_comparison.md` |
| `--unskimmed` | Use unskimmed filesets instead of default skimmed files |
| `--fileset PATH` | Override automatic fileset path with a custom fileset JSON |

### Output

| Flag | Description |
|------|-------------|
| `--dir DIR` | Create output subdirectory under rootfiles path |
| `--name SUFFIX` | Append suffix to output ROOT filename |
| `--debug` | Run without saving histograms (for testing) |
| `--study STUDY [STUDY ...]` | Add a study's extra regions to the output. `transfer-factor` adds the SR / flavor-CR regions without the m(lljj) or m(lJ) > 800 GeV cut, named `<region>_nomass`; the mass histograms span 0–8000 GeV so the plotter slices them at 800 GeV. Registered in `wrcoffea/regions.py` |
| `--split-groups GROUP [GROUP ...]` | Composite runs (`bkg`, `mc`, `all`): write these groups as one ROOT file per subgroup, e.g. `tt_tW` → `WRAnalyzer_TTbar.root` + `WRAnalyzer_tW.root`. Applies to local and Condor runs alike |

#### Transfer-factor study

The eμ transfer factor R = SR / flavor CR is measured on tt̄ and its closure is
checked below 800 GeV (AN-19-083 Appendix J). Run the analyzer with the study
regions and tt̄ split out, then fit with the plotter:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 bkg --study transfer-factor --split-groups tt_tW --dir tf_study
cd WR_Plotter && python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir tf_study --tf-study --input WRAnalyzer_TTbar.root
```

### Systematics

| Flag | Description |
|------|-------------|
| `--systs FAMILY [FAMILY ...]` | Enable systematic variations (MC only; ignored for data). Choices: `lumi` `pileup` `sf` `jer` `jes` `elescale` `elesmear` `muscale` `musmear` `dyzpt` |
| `--syst-observables OBS [OBS ...]` | Histograms that receive the variations (default: `all`; `fit` for the fit masses only, or explicit names) |
| `--omit-systs SOURCE [SOURCE ...]` | Deliberately leave out named sources (e.g. `ElectronTrigSf`) although their family is enabled; recorded in the manifest |

### Processing

| Flag | Description |
|------|-------------|
| `--condor` | Submit to HTCondor at LPC (see [condor.md](condor.md)) |
| `--max-workers N` | Number of Dask workers (local default: 3, condor single-sample: 50, condor composite skimmed: 200, condor composite unskimmed: 500) |
| `--worker-wait-timeout N` | Seconds to wait for first Condor worker before failing (default: 1200) |
| `--threads-per-worker N` | Threads per Dask worker for local runs |
| `--chunksize N` | Events per processing chunk (default: 250000) |
| `--maxchunks N` | Max chunks per file (default: all). Use `1` for quick testing |
| `--maxfiles N` | Max files per dataset (default: all). Use `1` for quick testing |

### XRootD Fallback

These flags control redirector fallback during unskimmed preprocessing. Fallback is enabled by default when `--unskimmed` is used.

| Flag | Description |
|------|-------------|
| `--xrd-fallback` | Explicitly enable XRootD redirector fallback |
| `--xrd-fallback-timeout N` | Seconds per fallback probe (default: 10) |
| `--xrd-fallback-retries-per-redirector N` | Probe attempts per redirector (default: 10) |
| `--xrd-fallback-sleep N` | Seconds between retries (default: 10.0) |

---

## Examples

```bash
# Everything locally (sequential)
python3 bin/run_analysis.py RunIII2024Summer24 all --dir 20260217_skimmed

# All backgrounds locally (sequential)
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir 20260217_skimmed

# Everything on Condor (parallel)
python3 bin/run_analysis.py RunIII2024Summer24 all --condor --systs lumi pileup sf

# Single sample on Condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100

# Quick local test (1 file, 1 chunk, 1000 events)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --maxchunks 1 --maxfiles 1 --chunksize 1000

# Custom output directory and filename
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir my_study --name test

# Only resolved region
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved

# Full systematics
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf

# Unskimmed files
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy lo_inc

# Use NLO inclusive DY for all samples
python3 bin/run_analysis.py RunIII2024Summer24 all --dy nlo_inc

# Use HT-binned DY for UL18 (only available for this era)
python3 bin/run_analysis.py RunIISummer20UL18 DYJets --dy lo_ht

# Validate fileset without processing
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
```

---

## Troubleshooting

### Unskimmed runs crash mid-job

Unskimmed files are accessed via XRootD and are more prone to transient I/O errors. If a run crashes unexpectedly, try setting `skipbadfiles=True` in `bin/run_analysis.py` ([line 433](../bin/run_analysis.py#L433)):

```python
skipbadfiles=True,
```

This allows the runner to skip unreadable files and continue processing the rest, at the cost of silently dropping any events in bad files.
