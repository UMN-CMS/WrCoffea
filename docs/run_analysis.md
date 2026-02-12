# Running the Analyzer

## Basic Usage

First, create filesets (see [filesets.md](filesets.md)). Then run the analyzer by specifying an era and sample:

### Backgrounds

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIISummer20UL18 tt_tW
python3 bin/run_analysis.py Run3Summer22EE Nonprompt
```

Background samples: `DYJets`, `tt_tW`, `Nonprompt`, `Other`.

### Signal

Signal samples require the `--mass` flag:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3200_N3000
```

### Data

```bash
python3 bin/run_analysis.py RunIII2024Summer24 EGamma
python3 bin/run_analysis.py RunIII2024Summer24 Muon
```

### Output

By default, ROOT histograms are saved to:
```
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/WRAnalyzer_<Sample>.root
```

Examples:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_DYJets.root
WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/WRAnalyzer_signal_WR4000_N2100.root
```

Using `--dir` creates a subdirectory:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/my_study/WRAnalyzer_DYJets.root
```

Using `--name` modifies the filename:
```
WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_test_DYJets.root
```

### Region Selection

By default, both resolved and boosted histograms are filled. Use `--region` to process only one:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --region boosted
```

### Systematics

Use `--systs` to produce systematic-varied histograms alongside the nominal. Systematics are only applied to MC samples; data samples always produce nominal-only histograms.

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --systs lumi pileup sf
```

| Option | Variations produced | Description |
|--------|-------------------|-------------|
| `lumi` | LumiUp, LumiDown | Luminosity uncertainty (flat ±1.4% for Run3, ±2.5% for UL18) |
| `pileup` | PileupUp, PileupDown | Pileup reweighting up/down from correctionlib |
| `sf` | MuonRecoSfUp/Down, MuonIdSfUp/Down, MuonIsoSfUp/Down, MuonTrigSfUp/Down, ElectronRecoSfUp/Down, ElectronIdSfUp/Down, ElectronTrigSfUp/Down | Lepton scale factor uncertainties (7 independent sources) |

Each enabled variation produces a separate histogram in the output ROOT file under `syst_<name>_<region>/` directories. Without `--systs`, only nominal histograms are produced.

When passing `--systs` to `analyze_all.sh`, systematics are automatically filtered — applied to MC backgrounds and signal, but removed for data.

## Batch Processing

Use `analyze_all.sh` to process multiple samples in one command:
```bash
bash bin/analyze_all.sh bkg RunIII2024Summer24
bash bin/analyze_all.sh data RunIII2024Summer24
bash bin/analyze_all.sh signal RunIII2024Summer24
bash bin/analyze_all.sh all RunIII2024Summer24
```

Extra flags are forwarded to `run_analysis.py`:
```bash
bash bin/analyze_all.sh bkg RunIII2024Summer24 --dir my_study --name test
bash bin/analyze_all.sh all RunIII2024Summer24 --condor --systs lumi pileup sf
```

## Flag Reference

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `era` | Campaign to analyze: `RunIISummer20UL18`, `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24` |
| `sample` | Sample to analyze: `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal` |

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
| `--dy {LO_inclusive,NLO_mll_binned,LO_HT}` | Specific DY sample variant (only valid for DYJets) |
| `--region {resolved,boosted,both}` | Analysis region to run (default: `both`) |
| `--unskimmed` | Use unskimmed filesets instead of default skimmed files |
| `--fileset PATH` | Override automatic fileset path with a custom fileset JSON |

### Output

| Flag | Description |
|------|-------------|
| `--dir DIR` | Create output subdirectory under rootfiles path |
| `--name SUFFIX` | Append suffix to output ROOT filename |
| `--debug` | Run without saving histograms (for testing) |

### Systematics

| Flag | Description |
|------|-------------|
| `--systs [lumi] [pileup] [sf]` | Enable systematic variations (MC only; ignored for data) |

| Option | Variations | Description |
|--------|-----------|-------------|
| `lumi` | LumiUp/Down | Luminosity uncertainty |
| `pileup` | PileupUp/Down | Pileup reweighting |
| `sf` | Muon{Reco,Id,Iso,Trig}SfUp/Down, Electron{Reco,Id,Trig}SfUp/Down | Lepton scale factors (7 sources) |

### Processing

| Flag | Description |
|------|-------------|
| `--condor` | Submit jobs to HTCondor at LPC (see [condor.md](condor.md)) |
| `--max-workers N` | Number of Dask workers (local default: 6, condor default: 50) |
| `--threads-per-worker N` | Threads per Dask worker for local runs |
| `--chunksize N` | Events per processing chunk (default: 250000) |
| `--maxchunks N` | Max chunks per file (default: all). Use `1` for quick testing |
| `--maxfiles N` | Max files per dataset (default: all). Use `1` for quick testing |

### Reweighting

| Flag | Description |
|------|-------------|
| `--reweight PATH` | Path to DY reweight JSON file (output of `scripts/derive_reweights.py`) |

## Examples

```bash
# Quick local test (1 file, 1 chunk, 1000 events)
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --maxchunks 1 --maxfiles 1 --chunksize 1000

# Custom output directory and filename
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir my_study --name test
# -> WR_Plotter/rootfiles/Run3/2022/Run3Summer22EE/my_study/WRAnalyzer_test_DYJets.root

# Only resolved region
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --region resolved

# Full systematics
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf

# Unskimmed files
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy LO_inclusive

# Condor submission
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --condor --max-workers 100

# Everything on Condor with systematics
bash bin/analyze_all.sh all RunIII2024Summer24 --condor --systs lumi pileup sf
```
