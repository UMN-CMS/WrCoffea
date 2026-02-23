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

Use `--systs` to produce systematic-varied histograms alongside the nominal. Systematics are only applied to MC samples; data always produces nominal-only histograms.

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf
```

| Option | Variations produced | Description |
|--------|-------------------|-------------|
| `lumi` | LumiUp, LumiDown | Luminosity uncertainty (flat ±1.4% for Run3, ±2.5% for UL18) |
| `pileup` | PileupUp, PileupDown | Pileup reweighting up/down from correctionlib |
| `sf` | MuonRecoSfUp/Down, MuonIdSfUp/Down, MuonIsoSfUp/Down, MuonTrigSfUp/Down, ElectronRecoSfUp/Down, ElectronIdSfUp/Down, ElectronTrigSfUp/Down | Lepton scale factor uncertainties (7 independent sources) |

Each variation produces a separate histogram under `syst_<name>_<region>/` directories. Without `--systs`, only nominal histograms are produced.

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
| `--unskimmed` | Use unskimmed filesets instead of default skimmed files |
| `--fileset PATH` | Override automatic fileset path with a custom fileset JSON |

### Output

| Flag | Description |
|------|-------------|
| `--dir DIR` | Create output subdirectory under rootfiles path |
| `--name SUFFIX` | Append suffix to output ROOT filename |
| `--debug` | Run without saving histograms (for testing) |
| `--tf-study` | Add transfer factor study regions (no mass cut) to the output |

### Systematics

| Flag | Description |
|------|-------------|
| `--systs [lumi] [pileup] [sf]` | Enable systematic variations (MC only; ignored for data) |

### Processing

| Flag | Description |
|------|-------------|
| `--condor` | Submit to HTCondor at LPC (see [condor.md](condor.md)) |
| `--max-workers N` | Number of Dask workers (local default: 3, condor single-sample: 50, condor composite skimmed: 200, condor composite unskimmed: 2000) |
| `--worker-wait-timeout N` | Seconds to wait for first Condor worker before failing (default: 1200) |
| `--threads-per-worker N` | Threads per Dask worker for local runs |
| `--chunksize N` | Events per processing chunk (default: 250000) |
| `--maxchunks N` | Max chunks per file (default: all). Use `1` for quick testing |
| `--maxfiles N` | Max files per dataset (default: all). Use `1` for quick testing |

### Reweighting

| Flag | Description |
|------|-------------|
| `--reweight PATH` | Path to DY reweight JSON file (DYJets only) |

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
