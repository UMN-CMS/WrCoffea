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

**Files:** `wrcoffea/das_utils.py` and `wrcoffea/skim/plan.py` — `ERA_SUBDIRS`
(two copies of the same map; update both)

```python
ERA_SUBDIRS = {
    ...
    "Run3Summer23": "Run3/2023/Run3Summer23",
}
```

Maps the era name to the subdirectory path used under `data/configs/`,
`data/filesets/`, `data/skims/`, and `data/lumis/`.  The skim campaign
(`bin/skim.py`) reads the copy in `wrcoffea/skim/plan.py`; the fileset and DAS
tools read the one in `wrcoffea/das_utils.py`.  An era missing from the skim
copy silently falls back to `<run>/<year>/<era>`.

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
  Run3Summer23: dy_lo_ht      # which DY fileset tag to use by default
```

### Data quality

```yaml
lumi_jsons:
  Run3Summer23: data/lumis/Run3/2023/Run3Summer23/Cert_Collisions2023_*.txt
```

### Scale factors / corrections (add as JSONs become available)

```yaml
muon_jsons:
  Run3Summer23:
    RECO: data/jsonpog/MUO/Run3/Run3Summer23/muon_HighPt.json.gz
    # Per-era high-pT HLT payload. If it only carries a per-lepton SF (no
    # _DATAeff/_MCeff corrections) the leading-muon SF is applied; the POG
    # muon_HighPt.json.gz also works here as a fallback.
    TRIGGER: data/jsonpog/MUO/Run3/Run3Summer23/ScaleFactors_Muon_highPt_HLT_2023_schemaV2.json

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

See `docs/inputs.md` for the expected JSON schema.

### 4b. Fileset JSONs

The analyzer reads fileset JSONs at runtime (see `docs/inputs.md`):

```bash
# Skimmed filesets: written by the skim campaign once the era is skimmed (step 5)
python3 bin/skim.py fileset --plan-dir plans/v2_Run3Summer23 --era Run3Summer23
```

Unskimmed filesets (`unskimmed/`, for `--unskimmed` runs) come straight from the configs and DBS:

```bash
python3 bin/skim.py unskimmed-fileset --era Run3Summer23
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

`run_analysis.py` reads this file to validate `--mass` and to choose the
signal points of composite modes like `signal` or `all`. By default it picks
the 9-point grid (`select_default_signal_points()`); `--signal-points all`
and `--mwr-range` select from the whole list (`select_signal_points()`).

---

## 5. Skim (if using skimmed files)

Run a skim campaign for the era ([Skimming](skimming.md) has the details):

1. Build its plan: `bin/skim.py plan --campaign v2 --plan-dir plans/v2_Run3Summer23 --era Run3Summer23`.
2. Process the groups, locally with `run` or as LPC Condor jobs with
   `bin/lpc_condor.sh`.
3. `validate`, then `upload` (local runs; Condor jobs ship their parts
   themselves).
4. Write the filesets with `fileset`.
5. For data, compute the processed luminosity with
   `scripts/processed_lumi.py` and use it in `config.yaml` and
   `WR_Plotter/data/lumi.yaml`.

The parts live at Wisconsin under
`/store/user/wijackso/WRAnalyzer/skims_v2/<run>/<year>/<era>/<category>/<dataset>/`
(and in the local campaign root, `~/work/skims_v2` by default).

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

**`WR_Plotter/data/lumi.yaml`** — add the era entry (luminosity in fb⁻¹; the
`*run3_signals` anchor supplies the default signal points):

```yaml
Run3Summer23:
  run: Run3
  year: "2023"
  lumi: 18.59
  com: 13.6
  plot_settings: run3_subera
  <<: *run3_signals
```

If the era belongs to a year or to `Run3`, add it to that combination's
`sub_eras` list; the combination's luminosity is derived from its eras.

**Plot settings** — `plot_settings:` above names an existing file in
`WR_Plotter/data/plot_settings/` (without it the era uses `default.yaml`).
Only if the era needs its own ranges, create
`WR_Plotter/data/plot_settings/<era>.yaml` that starts with
`inherit: default` and lists only the values that differ.

Sample groups (`WR_Plotter/data/sample_groups.yaml`) are shared by all eras
and normally need no change.

---

## 8. Limits

Limit setting lives in `WR_Plotter/signal_fitting/`: its `README.md` maps the
stages, their inputs and environments, and `LIMITS.md` the limit configurations.

---

## Quick-reference checklist

| Step | File(s) | Required? |
| --- | --- | --- |
| ERA_MAPPING | `wrcoffea/era_utils.py` | Yes |
| ERA_SUBDIRS | `wrcoffea/das_utils.py` and `wrcoffea/skim/plan.py` | Yes |
| Lumi + cuts | `wrcoffea/config.yaml` (lumis, lumi_unc, default_mc_tag) | Yes |
| Golden JSON | `config.yaml` → lumi_jsons + file on disk | Yes (data) |
| Muon SFs | `config.yaml` → muon_jsons + file on disk | When available |
| Electron SFs | `config.yaml` → electron_jsons, electron_sf_era_keys, electron_reco_config + files | When available |
| Pileup weights | `config.yaml` → pileup_jsons, pileup_correction_names + file | When available |
| Jet veto maps | `config.yaml` → jetveto_jsons, jetveto_correction_names + file | When available |
| JME JetID | `config.yaml` → jme_jsons + file | When available |
| Fileset JSONs | `data/configs/` + `data/filesets/` | Yes |
| Signal mass CSV | `data/signal_points/<era>_mass_points.csv` | Yes (signal) |
| WR_Plotter | `WR_Plotter/data/lumi.yaml` (+ `plot_settings/` only for era-specific ranges) | For plotting |
| Limits | `WR_Plotter/signal_fitting/` (`README.md`, `LIMITS.md`) | For limits |
| Tests | `python -m pytest tests/ -v` | Recommended |
