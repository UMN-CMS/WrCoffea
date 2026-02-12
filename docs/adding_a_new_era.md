# Adding a New Era

This guide walks through every step needed to add a new analysis era
(e.g. `Run3Summer23`) to the WrCoffea pipeline.  Steps are listed in
dependency order — complete them top-to-bottom.

---

## 1. Register the era

**File:** `wrcoffea/era_utils.py` — `ERA_MAPPING`

```python
ERA_MAPPING = {
    ...
    "Run3Summer23": {"run": "Run3", "year": "2023"},
}
```

This is the single source of truth for which eras exist.  The `run` key
(`"RunII"` or `"Run3"`) controls trigger path selection in
`build_trigger_masks()` and the `year` key is used for directory layout.

**File:** `wrcoffea/das_utils.py` — `ERA_SUBDIRS`

```python
ERA_SUBDIRS = {
    ...
    "Run3Summer23": "Run3/2023/Run3Summer23",
}
```

Maps the era name to the subdirectory path used under `data/configs/`,
`data/filesets/`, `data/skims/`, and `data/lumis/`.

---

## 2. Add analysis configuration

**File:** `wrcoffea/config.yaml`

Add the era to **every applicable section**.  Sections where the
correction is not yet available can be omitted — the code logs an info
message and falls back to identity (SF = 1 / weight = 1 / pass all).

### Required (analysis will not run without these)

```yaml
lumis:
  Run3Summer23: 17.794        # fb^-1, update when finalised

lumi_unc:
  Run3Summer23: 0.014         # fractional uncertainty

default_mc_tag:
  Run3Summer23: dy_lo_inc     # which DY fileset tag to use by default
```

### Data quality

```yaml
lumi_jsons:
  Run3Summer23: data/lumis/Run3/2023/Run3Summer23/Cert_Collisions2023_*.txt
```

### Scale factors / corrections (add as JSONs become available)

```yaml
muon_jsons:
  Run3Summer23: data/jsonpog/MUO/Run3/Run3Summer23/muon_HighPt.json.gz

pileup_jsons:
  Run3Summer23: data/jsonpog/LUM/Run3/Run3Summer23/puWeights.json.gz

pileup_correction_names:
  Run3Summer23: Collisions23_goldenJSON   # name inside the JSON

electron_jsons:
  Run3Summer23:
    RECO: data/jsonpog/EGM/Run3/Run3Summer23/electron.json.gz
    # TRIGGER: ...  (add when available)

electron_sf_era_keys:
  Run3Summer23: "2023"          # key used inside EGM correctionlib JSON

electron_reco_config:
  Run3Summer23:
    correction: Electron-ID-SF
    wp_low: Reco20to75
    wp_high: RecoAbove75
    pt_split: 75.0
```

### Optional jet corrections

```yaml
jme_jsons:
  Run3Summer23: data/jsonpog/JME/Run3/Run3Summer23/jetid.json.gz

jetveto_jsons:
  Run3Summer23: data/jsonpog/JME/Run3/Run3Summer23/jetvetomaps.json.gz

jetveto_correction_names:
  Run3Summer23: Summer23_V1
```

### Signal-only (if unskimmed signal filesets are unavailable on DAS)

```yaml
skimmed_only_signal:
  - RunIISummer20UL18
  # - Run3Summer23          # add here if needed
```

---

## 3. Obtain POG JSON payloads

Download the correctionlib JSON files from the CMS JSON POG repository
and place them under the paths declared in `config.yaml`:

```
data/jsonpog/
├── EGM/Run3/Run3Summer23/
│   ├── electron.json.gz
│   └── electronHlt.json.gz      # when available
├── JME/Run3/Run3Summer23/
│   ├── jetid.json.gz            # optional
│   └── jetvetomaps.json.gz      # optional
├── LUM/Run3/Run3Summer23/
│   └── puWeights.json.gz
└── MUO/Run3/Run3Summer23/
    └── muon_HighPt.json.gz
```

Also place the golden JSON for data lumi masking:

```
data/lumis/Run3/2023/Run3Summer23/
└── Cert_Collisions2023_*.txt
```

---

## 4. Build filesets

### 4a. Create DAS config JSONs

Create sample configuration files listing DAS dataset paths:

```
data/configs/Run3/2023/Run3Summer23/
├── Run3Summer23_data.json
├── Run3Summer23_mc_dy_lo_inc.json
├── Run3Summer23_signal.json
└── ...
```

See `docs/filesets.md` for the expected JSON schema.

### 4b. Generate fileset JSONs

Use the CLI to query DAS and produce the fileset JSONs that the analyzer
reads at runtime:

```bash
python bin/run_analysis.py Run3Summer23 DYJets --build-fileset
python bin/run_analysis.py Run3Summer23 Muon  --build-fileset
# ... repeat for each process
```

The resulting fileset JSONs land under:

```
data/filesets/Run3/2023/Run3Summer23/
├── skimmed/
│   ├── Run3Summer23_data_fileset.json
│   └── Run3Summer23_mc_dy_lo_inc_fileset.json
└── unskimmed/
    └── ...
```

### 4c. Signal mass points

Create a two-column CSV listing every (WR, N) mass point available for
the era:

```
data/signal_points/Run3Summer23_mass_points.csv
```

Format (with header):

```csv
WR,N
2000,100
2000,300
...
```

`analyze_all.sh signal` reads this file to choose which signal points
to process.

---

## 5. Skim (if using skimmed files)

Follow `docs/skimming.md` to submit Condor skimming jobs and merge
the output.  Skimmed NanoAOD files are stored under:

```
data/skims/Run3/2023/Run3Summer23/<sample>/
```

---

## 6. Validate

Run a quick local test to make sure the new era works end-to-end:

```bash
python bin/run_analysis.py Run3Summer23 DYJets \
    --maxchunks 1 --maxfiles 1 --chunksize 1000
```

Run the test suite to verify config consistency (e.g. lumi_unc covers
every lumi era, electron_sf_era_keys covers every electron_jsons era):

```bash
python -m pytest tests/ -v
```

---

## 7. WR_Plotter (for plotting)

If you plan to make plots with the new era, update these files:

**`WR_Plotter/data/lumi.json`** — add the era entry:

```json
"Run3Summer23": {
  "run": "Run3",
  "year": "2023",
  "lumi": 17.794,
  "com": 13.6
}
```

**`WR_Plotter/data/plot_settings/Run3Summer23.yaml`** — create by
copying an existing Run3 era file (e.g. `Run3Summer22.yaml`) and
adjusting labels.

**`WR_Plotter/data/sample_groups/Run3Summer23.yaml`** — create by
copying an existing file and updating the sample-to-group mapping
for the new era's datasets.

**`WR_Plotter/data/kfactors.yaml`** — add a block:

```yaml
Run3Summer23:
  DYJets: 1.0
  _default: 1.0
```

---

## 8. Combine (for limit setting)

Update the era path in `combine_framework/datacards.py` if you want
to produce datacards for the new era.

---

## Quick-reference checklist

| Step | File(s) | Required? |
|------|---------|-----------|
| ERA_MAPPING | `wrcoffea/era_utils.py` | Yes |
| ERA_SUBDIRS | `wrcoffea/das_utils.py` | Yes |
| Lumi + cuts | `wrcoffea/config.yaml` (lumis, lumi_unc, default_mc_tag) | Yes |
| Golden JSON | `config.yaml` → lumi_jsons + file on disk | Yes (data) |
| Muon SFs | `config.yaml` → muon_jsons + file on disk | When available |
| Electron SFs | `config.yaml` → electron_jsons, electron_sf_era_keys, electron_reco_config + files | When available |
| Pileup weights | `config.yaml` → pileup_jsons, pileup_correction_names + file | When available |
| Jet veto maps | `config.yaml` → jetveto_jsons, jetveto_correction_names + file | When available |
| JME JetID | `config.yaml` → jme_jsons + file | When available |
| Fileset JSONs | `data/configs/` + `data/filesets/` | Yes |
| Signal mass CSV | `data/signal_points/<era>_mass_points.csv` | Yes (signal) |
| WR_Plotter | `WR_Plotter/data/` (lumi.json, plot_settings, sample_groups, kfactors) | For plotting |
| Combine | `combine_framework/datacards.py` | For limits |
| Tests | `python -m pytest tests/ -v` | Recommended |
