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

Eras the CLI accepts (`--list-eras`): `RunIISummer20UL16APV`, `RunIISummer20UL16`, `RunIISummer20UL17`, `RunIISummer20UL18`, `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`, `RunIII2025Summer25`. Filesets exist for `RunIISummer20UL18` and the 2022, 2023 and 2024 eras; UL16APV, UL16 and UL17 have dataset configs only, and `RunIII2025Summer25` is data-only (its MC is the 2024 MC, `mc_alias` in `wrcoffea/config.yaml`).

### Composite Modes

Composite modes process multiple samples. Locally, they run sequentially (one sample at a time, reusing the same Dask cluster). On Condor (`--condor`), all samples are processed in parallel.

| Mode | Samples |
|------|---------|
| `all` | EGamma, Muon, DYJets, tt_tW, Nonprompt, Other, Signal |
| `data` | EGamma, Muon |
| `bkg` | DYJets, tt_tW, Nonprompt, Other |
| `signal` | Signal (the default grid of mass points, or `--signal-points`) |
| `mc` | DYJets, tt_tW, Nonprompt, Other, Signal |

Single samples: `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal`.

```bash
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir my_study
python3 bin/run_analysis.py RunIII2024Summer24 all --systs lumi pileup sf
python3 bin/run_analysis.py RunIII2024Summer24 all --condor              # parallel on Condor
python3 bin/run_analysis.py RunIISummer20UL18 signal --signal-points all --mwr-range 1000 6000 --condor --dir my_signals
```

Which signal points a composite mode with Signal (`signal`, `mc`, `all`) runs:

- **Default:** a 9-point grid, the lowest, median and highest N for WR = 2, 4 and 6 TeV (`SIGNAL_WR_GRID` in `wrcoffea/cli_utils.py`).
- **`--signal-points all`:** every point of `data/signal_points/<era>_mass_points.csv`.
- **`--signal-points WR2000_N100 WR4000_N2100 ...`:** an explicit list.
- **`--mwr-range MIN MAX`:** restricts any of these to MIN ≤ M_WR ≤ MAX.

Each point is written to its own ROOT file, as for the default grid.

When a sample fails during local composite processing, the error is logged and the next sample continues.

### UL18 signal samples

A UL18 mass point is the official UL pair, `WRtoNEltoElElJJ` and `WRtoNMutoMuMuJJ`, processed into one file (105 points,
M_WR = 1, 2, 4, 6 and 8 TeV). The private pre-legacy scan (`WRtoNLtoLLJJ`, 490 points, 102X Autumn18 NanoAODv7) has its
own fileset and runs only when that fileset is named, for checks:

```bash
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3800_N1600 --dir my_scan_check \
    --fileset data/filesets/RunII/2018/RunIISummer20UL18/skimmed/RunIISummer20UL18_signal_scan_fileset.json
```

With `--fileset`, `--mass` only has to be a point of that fileset. Give scan runs their own `--dir`: the file names do
not say which sample a point came from.

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

Each ROOT file contains:

- `<region>/<var>_<region>`: the nominal histograms (MC scaled to σ × L), plus `syst_<name>_<region>/` folders with `--systs` (see [Systematics](#systematics)).
- `cutflow_resolved/` and `cutflow_boosted/`: per channel (`ee`, `mumu`, `em`) the `cumulative` and `onecut` cutflows, weighted and `_unweighted`, with the cut names as bin labels. The weighted bins exclude the per-region lepton SFs.
- `event_weights/<region>/` (MC only): the distribution of the region's nominal event weight w (lepton SFs included).
  - `positive` and `negative`: event counts vs log10|w|, split by the sign of w. Each event counts once (unit weight); events with w = 0 are skipped.
  - `components/<name>`: one distribution per factor of w. `genweight_positive` / `genweight_negative` (the generator weight, split by sign) and `normalization` (σ × L / Σw) are histogrammed vs log10 of the value; every correction factor gets one histogram vs its value: `pileup`, `dy_zpt`, `signal_kfactor` and `dy_reshape` where they apply, and each lepton SF component (`MuonRecoSf`, `ElectronIdSf`, `LsfSf`, ...). `corrections` is the product of the correction factors, so genweight × normalization × corrections = w.

  The event-weight distributions are filled in the nominal pass only (no systematic variations), never for data, and not with `--unskimmed` (whose weights lack the 1/Σw factor until the end of the run).
- With `--save-events`, the `events_<region>` TTrees of the signal regions.

Data files are blinded by default, see [Blinding](#blinding).

### Blinding

Data is blinded by default; MC is never blinded.

- **Signal regions.** Data never builds the signal regions (`wr_ee_resolved_sr`, `wr_mumu_resolved_sr`, `wr_ee_boosted_sr`, `wr_mumu_boosted_sr`) or any variant of them (such as `wr_ee_resolved_sr_nomass` from `--study transfer-factor`). The data files (`WRAnalyzer_EGamma.root`, `WRAnalyzer_Muon.root`) therefore have no signal-region folders and no signal-region `--save-events` trees. The DY and flavor control regions, and their variants, are filled as usual.
- **Cutflows.** The data `ee` and `mumu` cutflows stop before the m(ll) > 200 GeV cut: the resolved chains end at `mlljj_gt800` (no `mll_gt200` or `mll_gt400` rows) and the boosted chains at `no_of_lepton_in_ak8` (no `mll_gt200_boosted` or `mlj_gt800_boosted` rows). The cut axis is shorter, not zero-filled. The `em` cutflows (flavor CR) are complete. This holds for `cumulative` and `onecut`, weighted and unweighted.
- **Manifest.** The top-level `blinding` entry of `WRAnalyzer_<sample>_systematics.json` records what the file withholds:

  ```json
  "blinding": {"applied": true,
               "signal_regions": ["wr_ee_boosted_sr", "wr_ee_resolved_sr", "wr_mumu_boosted_sr", "wr_mumu_resolved_sr"],
               "cutflow_truncated_before": {"cutflow_resolved": {"ee": "mll_gt200", "mumu": "mll_gt200"},
                                            "cutflow_boosted": {"ee": "mll_gt200_boosted", "mumu": "mll_gt200_boosted"}}}
  ```

  `applied` is true only for data of a blinded run. `signal_regions` also lists the withheld variants, and `cutflow_truncated_before` only the cutflows the run produced (`--region`). MC files and unblinded data report `{"applied": false, "signal_regions": [], "cutflow_truncated_before": {}}`. Files written before blinding existed have no `blinding` entry, and their data still carries the signal regions and the full cutflows.
- **Unblinding.** `--unblind` fills the data signal regions and the full data cutflows. It is allowed for Run 2 eras (`RunIISummer20UL*`) only: any Run 3 era is refused at start-up, and the processor refuses it too. Blinding is part of the recorded production configuration, so blinded and unblinded outputs are never merged.

> **Run 2 workflows that need observed signal-region data** (observed limits, signal-region data/MC comparisons) need data files produced with `--unblind`. Rerun the Run 2 data with it, for 2018 after the reskim:
>
> ```bash
> python3 bin/run_analysis.py RunIISummer20UL18 data --unblind --dir <dir>
> ```

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
| `scale` | weight | ScaleMuR, ScaleMuF, ScaleMuRF (each Up/Down) | QCD renormalization and factorization scales from `LHEScaleWeight` (NanoAOD 9-weight order; samples that store 8 weights without the nominal, such as TTLNu-1Jets, use the matching 8-weight order), each divided by the dataset-level ratio `LHEScaleSumw` from the fileset metadata, so the yield is unchanged and only acceptance and shape move. WR_Plotter combines the three into an envelope |
| `ps` | weight | PSISR, PSFSR (each Up/Down) | Parton-shower ISR and FSR from `PSWeight`, normalized the same way with `PSSumw` |
| `pdf` | members | none (members, see right) | PDF and alpha_s from `LHEPdfWeight`, each member divided by the dataset mean `LHEPdfSumw` (acceptance and shape only). No Up/Down histograms: the analyzer fills every member of `mass_fourobject` / `mass_twoobject`, the writer sums them over the datasets that share a PDF set and stores `pdfmembers_<region>/<var>_lha<id>` (TH2: variable x member), with the set layouts in the manifest (`pdf_sets`). WR_Plotter (`wrplotter/pdf.py`, sources `pdf` and `alphas`) and the MC datacards rebin every member first and then take the Hessian quadrature (replicas: standard deviation) and the alpha_s shift scaled to 0.0015: a quadrature formed on fine bins and summed afterwards overstates wide bins, because each eigenvector only moves events between bins. The set of each sample comes from `data/pdf_sets/<Run>/<Year>/<Era>.json` (`scripts/pdf_sets.py`, `config.yaml` `pdf_sets`); samples without weights (pure Pythia WW/WZ/ZZ) contribute no members, with a warning |
| `sf` | weight | MuonRecoSf, MuonIdSf, MuonIsoSf, MuonTrigSf, LooseMuonRecoSf, LooseMuonIdSf, ElectronRecoSf, ElectronIdSf, ElectronTrigSf, LooseElectronRecoSf (each Up/Down) | Lepton scale-factor uncertainties, one source per component. Only the components that enter a region are produced for it |
| `jer` | shape | JERUp, JERDown | Jet energy resolution; selection re-run on smeared jets (every era with a `JR_tag_MC` in `config.yaml`) |
| `jes` | shape | JESUp, JESDown (one pair per entry of `jes_unc_mapping` in `config.yaml`) | Jet energy scale; selection re-run on shifted jets. All eras currently map the `Total` source; UL18 also ships the `Regrouped_*` sources if per-source nuisances are wanted |
| `elescale`, `elesmear` | shape | EleScaleUp/Down, EleSmearUp/Down | Electron energy scale and resolution (Run 3 eras only) |
| `muscale`, `musmear` | shape | MuScaleUp/Down, MuSmearUp/Down | Muon momentum scale and resolution on the TuneP pT. Eras in `config.yaml` `muon_highpt` (`wrcoffea/muon_momentum.py`, Muon POG Run 3 recipe): below 200 GeV the MuonScaRe (`ScaleNSmear` payload) scale and smearing with its statistical variations; above 200 GeV the generalized-endpoint bias removed from data (q/pT -> q/pT - kappa), extra MC smearing p -> p(1 + N(0, sigma(p, eta) f)), MuScale = the kappa uncertainty on q/pT, MuSmear = the smearing fraction +- the POG resolution systematic. The payloads (`data/muon_highpt/`) hold numbers from CMS-internal documents; [docs/muon_momentum.md](muon_momentum.md) lists their sources. Other eras: **placeholder**, flat 5%/TeV scale and smear equals nominal |
| `lsf` | weight | LsfSfUp, LsfSfDown | Flat LSF3 > 0.75 efficiency scale factor by lepton-in-jet flavour (AN-19-083 Table 60), boosted SR and flavour CR only. Only eras with an `lsf_sf` entry in `config.yaml` (UL18) |
| `dyreshape` | weight | DYReshapeStat, DYReshapeChannel (each Up/Down) | Statistical and ee-vs-mumu uncertainties of the data-driven DY reshape vs jet pT (`scripts/derive_dy_reshape.py`). Only eras with a `dy_reshape` payload in `config.yaml`; needs the Z-pT reweighting on |
| `dyzpt` | weight | DYZptQcdScale, DYZptQcdPdf, DYZptQcdAlphaS, DYZptEw1, DYZptEw2, DYZptEw3 (each Up/Down) | Uncertainties of the DY Z-pT reweighting (QCD NLO/LO ratio and EW correction, ported from the Run 2 HNWRAnalyzer, see `docs/run2_hnwr_comparison.md`). Only eras with a `dy_zpt_reweight` payload in `config.yaml` (UL18). Non-trivial only for the DY sample the payload was derived for (`dy_reweight_datasets` in `config.yaml`: the LO HT-binned DY); other DY samples (NLO, LO inclusive) and other MC carry nominal copies |

**Samples without generator weights.** The pure-Pythia 8 dibosons (`WW`, `WZ`, `ZZ` `_TuneCP5_13p6TeV_pythia8`) have no `LHEScaleWeight` in any Run 3 era: their `scale` variations are filled with the nominal weight and the analyzer warns once per dataset (every sample has `PSWeight`). Two 2022 samples have a single file whose events lack the LHE weights: ZZZ in Run3Summer22 (12,342 of 1.97 M events) and the WR electron point MWR6000_N1500 in Run3Summer22EE (621 of 210,000). Those events also get the nominal weight for the `scale` variations (decided 2026-10-08), with one warning per dataset; since the dataset-level ratio comes from the files that have the weights, the total yield is unchanged. Any other mismatch in the number of weights is an error. Details and the full list of input defects: `docs/skimming.md`, "Known input defects".

A family the era cannot produce (for example `elescale` for UL18) is an error at start-up, not a silent skip. The same holds for a single source whose correction inputs are missing: a family whose payload is absent for the era refuses to run until the omission is recorded explicitly with `--omit-systs <Source>`. Omitted sources are listed in the manifest and never presented as coverage. (Every configured era currently has all `sf` payloads; the UL18 electron trigger one is the Run 2 HEEP trigger-OR measurement, see `docs/run2_hnwr_comparison.md`.) The registry keeps two separate questions apart: whether a source can physically change a region (`applies`, for example muon scale factors never touch an `ee` region) and whether its inputs exist for the era (`available`). Only the first makes a variation "unaffected"; the second, unresolved, is an error.

**Which histograms get variations.** By default every histogram carries the Up/Down entries. If memory or throughput becomes a problem, `--syst-observables fit` restricts the variations to the fit observables (`mass_dilepton`, `mass_fourobject`, `mass_twoobject`), and an explicit list of histogram names restricts them further; Nominal is always filled everywhere.

**Coverage bookkeeping.** Each chunk records which (region, variation) pairs it completed, together with the production configuration (families, omissions, observables, regions, blinding); chunks or datasets produced with different configurations are never merged. The set of variations a dataset must provide comes from the registry, so a variation nobody produced is an error. A shape variation is only accepted for a chunk when every populated affected collection carries the shifted fields (AK4 and AK8 jets both, including the varied mass). The configured regions are part of the record too: a run with `--region both` or `--region boosted` fails immediately if the input has no `FatJet` collection or no `FatJet_lsf3` branch, instead of silently dropping the boosted regions, and the writer refuses a dataset that never filled a configured region. Use `--region resolved` for inputs without the boosted branches. At save time every dataset must have completed every applicable variation in every chunk, otherwise the writer raises instead of producing a partial template. A chunk with no affected objects (for example no electrons under an electron-scale shift) fills its nominal under the variation label and counts as complete. When datasets are merged into a group, a dataset for which the registry says a source does not apply contributes its nominal to that variation.

Each variation is written under `syst_<name>_<region>/`. Variations that do not apply to a region (muon scale factors in an `ee` region, say) are not written as empty templates; they are listed as `unaffected` in the manifest. A JSON manifest `WRAnalyzer_<sample>_systematics.json` next to each ROOT file records the requested families, the observables covered, the datasets and chunk counts, per region the produced, normalization, and unaffected variations, and under `blinding` what a data file withholds (see [Blinding](#blinding)).

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
| `--signal-points POINT [POINT ...]` | Signal points of a composite mode with Signal (`signal`, `mc`, `all`): `default` (the 9-point grid), `all` (every point of the era's mass-point CSV) or a list of `WR<wr>_N<n>` tags |
| `--mwr-range MIN MAX` | Keep only the signal points with MIN ≤ M_WR ≤ MAX (composite modes with Signal) |
| `--dy VARIANT` | DY sample variant (only valid for DYJets). Available variants are configured per era in `config.yaml` (`dy_variants`). Without it every era uses the LO HT-binned set (`lo_ht`, `default_mc_tag`); `lo_inc` for LO inclusive, `nlo_inc` for NLO inclusive |
| `--region {resolved,boosted,both}` | Analysis region to run (default: `both`) |
| `--unblind` | Fill the data signal regions (and their variants) and the full data cutflows. Run 2 eras only; refused for every Run 3 era. Default: data is blinded, see [Blinding](#blinding). MC is never blinded |
| `--no-dy-zpt-reweight` | Do not apply the DY Z-pT (QCD × EW) reweighting to `DYJets` for eras with a payload in `config.yaml` (currently UL18). Applied by default, and only to the DY datasets the payload was derived for (`dy_reweight_datasets`); any other DY sample, e.g. `--dy nlo_inc`, keeps its weight and its manifest records `not_for_sample`. Incompatible with `--systs dyzpt` |
| `--xrootd-timeout SECONDS` | XRootD read timeout per chunk (default 60 with `--condor`, 10 locally). Raise it when streaming skims over a high-latency link and chunks fail with "Operation expired" |
| `--no-dy-reshape` | Do not apply the data-driven DY reshape vs jet pT for eras with a `dy_reshape` payload in `config.yaml`. Implied by `--no-dy-zpt-reweight` |
| `--no-signal-kfactor` | Do not apply the signal NLO/LO k(m(lN))/⟨k⟩ shape reweighting for eras with a payload in `config.yaml` (currently UL18). Applied by default; see `docs/run2_hnwr_comparison.md` |
| `--unskimmed` | Use unskimmed filesets instead of default skimmed files |
| `--fileset PATH` | Override automatic fileset path with a custom fileset JSON. With `Signal`, `--mass` then only has to be a point of that fileset (e.g. the [UL18 scan](#ul18-signal-samples)) |

### Output

| Flag | Description |
|------|-------------|
| `--dir DIR` | Create output subdirectory under rootfiles path |
| `--name SUFFIX` | Append suffix to output ROOT filename |
| `--debug` | Run without saving histograms (for testing) |
| `--study STUDY [STUDY ...]` | Add a study's extra regions to the output. `transfer-factor` adds the SR / flavor-CR regions without the m(lljj) or m(lJ) > 800 GeV cut, named `<region>_nomass`; the mass histograms span 0–8000 GeV so the plotter slices them at 800 GeV. `dy-cr-split` adds the DY CRs split into 60 < m(ll) < 100 GeV (`<region>_mll60to100`, AN-19-083 DY CR1) and 100 < m(ll) < 150 GeV (`<region>_mll100to150`, CR2), for the DY-reshape CR1 → CR2 closure (`scripts/derive_dy_reshape.py --region-suffix`). Registered in `wrcoffea/regions.py` |
| `--save-events` | Also write the per-event (mass, nominal weight) of the signal regions as TTrees `events_<region>` (branches `mass`, `weight`; resolved `mass_fourobject`, boosted `mass_twoobject`) into the output ROOT files, concatenated over the datasets of each file. For unbinned smoothing / KDE studies. Data gets them only with `--unblind` |
| `--split-groups GROUP [GROUP ...]` | Composite runs (`bkg`, `mc`, `all`): write these groups as one ROOT file per subgroup. `tt_tW` → `WRAnalyzer_TTbar.root` + `WRAnalyzer_tW.root`; `Nonprompt` → `WJets` + `TTSemiLep` + `SingleTop`; `Other` → `Diboson` + `Triboson` + `ttV` (name prefixes cover Run3 and RunII-UL datasets; see `_PHYSICS_SUBGROUPS` in `wrcoffea/cli_utils.py`). Applies to local and Condor runs alike |

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
| `--systs FAMILY [FAMILY ...]` | Enable systematic variations (MC only; ignored for data). Choices: `lumi` `pileup` `prefire` `sf` `jer` `jes` `elescale` `elesmear` `muscale` `musmear` `dyzpt` `lsf` `dyreshape` `scale` `ps` `pdf` (see [Systematics](#systematics)) |
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

# Use LO inclusive DY instead of the default HT-binned set
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dy lo_inc

# Unblinded data with the signal regions (Run 2 eras only)
python3 bin/run_analysis.py RunIISummer20UL18 data --unblind

# Validate fileset without processing
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
```

---

## Troubleshooting

### Unskimmed runs crash mid-job

Unskimmed files are accessed via XRootD and are more prone to transient I/O errors. If a run crashes unexpectedly, try setting `skipbadfiles=True` in the `Runner(...)` call of `bin/run_analysis.py` (search for `skipbadfiles`):

```python
skipbadfiles=True,
```

This allows the runner to skip unreadable files and continue processing the rest, at the cost of silently dropping any events in bad files.
