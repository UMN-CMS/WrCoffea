# Adding a New Era

What this page answers: which code tables, config entries, payloads and files a new analysis era
needs, in the order to add them, from the era name to skims, filesets, plots and limits. The most
recent addition, `RunIII2025Summer25` (data only, on the 2024 MC), has a complete set of entries in
`wrcoffea/config.yaml` to copy from. Verified on 2026-10-12.

---

## 1. Register the era

**File:** `wrcoffea/era_utils.py`: `ERA_MAPPING`

```python
ERA_MAPPING = {
    ...
    "Run3Summer23": {"run": "Run3", "year": "2023"},
}
```

This is the single source of truth for which eras exist (`run_analysis.py` takes its era choices
from it). The `run` key (`"RunII"` or `"Run3"`) controls trigger path selection in
`build_trigger_masks()`; `run` and `year` give the directory layout `<run>/<year>/<era>` of
`data/configs/`, `data/filesets/`, the skims and the analyzer outputs. `ERA_SUBDIRS` is derived
from `ERA_MAPPING` in the same file (`wrcoffea/skim/plan.py` re-exports
it), so there is nothing else to register; `tests/test_era_utils.py` checks that every configured
era resolves.

Two per-era tables live in the analyzer code itself (`wrcoffea/analyzer.py`):

- `RUN2_TRIGGERS`: the single-lepton paths of each Run 2 era. A Run 2 era missing there silently
  gets the UL18 paths. The Run 3 eras share one set of paths in `build_trigger_masks()`.
- `apply_noise_filter()`: the era-specific MET filters (`ecalBadCalibFilter`, the HBHE filters,
  the 2022/2023 ECAL veto). An era not listed there gets only the common flags and a warning.

---

## 2. Add analysis configuration

**File:** `wrcoffea/config.yaml`. Add the era to every section that applies, copying an existing
era's entries as the template; the values themselves live there, not here.

### Required: the run stops without these

- `lumis`: the era's luminosity in fb⁻¹, which normalizes the MC (the processed luminosity of the
  skimmed data, step 5).
- `lumi_unc`: the fractional luminosity uncertainty; `--systs lumi` refuses an era without it, and
  a test requires it for every era in `lumis`.
- `default_mc_tag`: the DY fileset tag an MC run uses when `--dy` is not given.
- `dy_variants`: the `--dy` variants that have filesets for the era; `--dy` refuses the others.
- `lumi_jsons`: the golden JSON; a data run stops when it is missing or unreadable.
- `jerc_jsons`: the JEC and JER payloads and tags, AK4 and AK8; every run stops without it.
- `electron_hlt_paths`: the path evaluated in the electron TRIGGER payload; required once
  `electron_jsons` has a TRIGGER entry.
- `dy_reweight_datasets`: the regular expression of the DY datasets the Z-pT reweight and the
  reshape were derived for; required, when the config loads, for every era with `dy_zpt_reweight`
  or `dy_reshape`.
- `mc_alias` (data-only eras): the era whose MC the era uses; `skim.py fileset` writes the
  borrowed MC filesets from it, and without them an MC run of the era finds no fileset.

Every file a key maps must exist: a missing file is an error, never an identity.

### Corrections: identity when absent (info or warning in the log)

- `jetveto_jsons`, `jetveto_correction_names`: the jet veto map and the correction name inside it.
- `jetveto_mode`: `event` (the Run 3 default) or `jet` (the UL eras).
- `jme_jsons`: the correctionlib jet ID (`jetid.json.gz`), mapped for 2024 and 2025 (required for NanoAODv15, which has
  no `Jet_jetId`); other eras use the
  NanoAOD jet ID branches.
- `muon_jsons`: RECO (reco, ID and isolation SFs), TRIGGER and ScaleNSmear payloads.
- `muon_hlt_correction`: the correction name in the muon TRIGGER payload when it is not the
  default.
- `muon_highpt`: the high-pT muon momentum payload (`data/muon_highpt/`); eras not listed keep the
  TuneP placeholder.
- `prefiring_eras`: the eras whose MC gets the NanoAOD L1 prefiring weight (the UL eras).
- `pileup_jsons`, `pileup_correction_names`: the pileup payload and the correction name inside it.
- `electron_jsons`: RECO (reco and ID SFs), TRIGGER and ScaleNSmear payloads.
- `electron_sf_era_keys`: the era key inside the EGM payloads; a test requires it for every era
  in `electron_jsons`.
- `electron_reco_config`: the correction name, the reco working points and their pT split, and
  the last measured bins (`pt_max`, `id_pt_max`) the SFs are clipped at.
- `lsf_sf`: the LSF scale factor per flavour of the lepton in the jet.
- `hem_veto`: the HEM 15/16 veto (UL18 only).
- `dy_zpt_reweight`, `dy_reshape`: the DY Z-pT reweight and jet-pT reshape payloads
  (`data/reweights/`).
- `signal_kfactor`: the signal `k(m_lN) / <k>` payload (`data/reweights/`).
- `pdf_sets`: the PDF set of each MC sample's `LHEPdfWeight` (`data/pdf_sets/`, written by
  `scripts/pdf_sets.py`); without it the era cannot produce `--systs pdf`.

What each correction does and which payload every era uses is in
[corrections.md](corrections.md).

---

## 3. Obtain POG payloads

Take the correctionlib payloads from the CAT metadata area on cvmfs, the authority (the older
`jsonpog-integration` mirror is stale); [corrections.md](corrections.md) names the campaign of
each era and how to compare the copies. Place them under the paths `config.yaml` declares, for
example (Run3Summer23):

```
data/jsonpog/
├── EGM/Run3/Run3Summer23/
│   ├── electron.json.gz                # reco + ID SFs (RECO)
│   ├── electronHlt.json.gz             # trigger SF (TRIGGER)
│   └── electronSS_EtDependent.json.gz  # scale and smearing (ScaleNSmear)
├── JME/Run3/Run3Summer23/
│   ├── jet_jerc.json.gz                # AK4 JEC/JER (jerc_jsons)
│   ├── fatJet_jerc.json.gz             # AK8 JEC/JER (jerc_jsons)
│   ├── jetvetomaps.json.gz
│   └── jetid.json.gz                   # used only where jme_jsons maps it
├── LUM/Run3/Run3Summer23/
│   └── puWeights.json.gz
└── MUO/Run3/Run3Summer23/
    ├── muon_HighPt.json.gz             # reco/ID/iso SFs (RECO)
    ├── muon_Z.json.gz                  # trigger-OR efficiencies (TRIGGER)
    └── muon_scalesmearing.json.gz      # scale and smearing (ScaleNSmear)
```

Also place the golden JSON for data lumi masking under `data/lumis/<run>/<year>/` (eras of one
year can share it, as in 2022 and 2023) or `data/lumis/<run>/<year>/<era>/`, and map it in
`lumi_jsons`.

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
python3 bin/skim.py fileset --plan-dir <plan dir> --out <campaign root> --remote <remote> --era Run3Summer23
```

`<plan dir>` is the directory `skim.py plan --plan-dir` wrote (by convention
`<campaign root>/plans/v2_<era>`); plans and campaign roots are not in the repository.

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

1. Build its plan: `bin/skim.py plan --campaign v2 --plan-dir <campaign root>/plans/v2_Run3Summer23 --era Run3Summer23`.
2. Process the groups, locally with `run` or as LPC Condor jobs with
   `bin/lpc_condor.sh`.
3. `validate`, then `upload` (local runs; Condor jobs ship their parts
   themselves).
4. Write the filesets with `fileset`.
5. For data, compute the processed luminosity with
   `scripts/processed_lumi.py` and use it in `config.yaml` and
   `WR_Plotter/data/lumi.yaml`.

The parts land under `<remote>/<run>/<year>/<era>/<category>/<dataset>/`, and the statuses and
sidecars in the same layout under the local campaign root (`--out`). The skims of the existing eras
are in Billy's area at Wisconsin (`/store/user/wijackso/WRAnalyzer/skims_v2`); several `skim.py`
defaults point at his setup ([Skimming](skimming.md), "Billy's defaults").

---

## 6. Validate

Run a quick local test to make sure the new era works end-to-end:

```bash
python bin/run_analysis.py Run3Summer23 DYJets \
    --maxchunks 1 --maxfiles 1 --chunksize 1000 --dir smoke
```

Use a scratch `--dir` and delete its output (`WR_Plotter/rootfiles/<run>/<year>/<era>/smoke/`)
afterwards.

Run the test suite to verify config consistency (e.g. lumi_unc covers
every lumi era, electron_sf_era_keys covers every electron_jsons era):

```bash
python -m pytest tests/ -v
```

---

## 7. WR_Plotter (for plotting)

If you plan to make plots with the new era, update these files:

**`WR_Plotter/data/lumi.yaml`** — add the era entry (luminosity in fb⁻¹, the same value as
`lumis` in `wrcoffea/config.yaml`; the `*run3_signals` anchor supplies the default signal points).
The plotter has no entry for `RunIII2025Summer25` yet.

```yaml
Run3Summer23:
  run: Run3
  year: "2023"
  lumi: <lumi>
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

## Quick-reference checklist

| Step | File(s) | Required? |
| --- | --- | --- |
| ERA_MAPPING | `wrcoffea/era_utils.py` (ERA_SUBDIRS follows from it) | Yes |
| Run 2 triggers, MET filters | `wrcoffea/analyzer.py` → RUN2_TRIGGERS, `apply_noise_filter()` | Yes |
| Lumi, DY tags | `wrcoffea/config.yaml` → lumis, lumi_unc, default_mc_tag, dy_variants | Yes |
| Golden JSON | `config.yaml` → lumi_jsons + file on disk | Yes (data) |
| JEC/JER | `config.yaml` → jerc_jsons + files | Yes |
| Data-only era | `config.yaml` → mc_alias | Yes (data-only) |
| Muon SFs | `config.yaml` → muon_jsons (muon_hlt_correction, muon_highpt) + files | When available |
| Electron SFs | `config.yaml` → electron_jsons, electron_hlt_paths, electron_sf_era_keys, electron_reco_config + files | When available |
| Pileup weights | `config.yaml` → pileup_jsons, pileup_correction_names + file | When available |
| Jet veto maps | `config.yaml` → jetveto_jsons, jetveto_correction_names (jetveto_mode) + file | When available |
| JME JetID | `config.yaml` → jme_jsons + file | When available |
| Prefiring, HEM, LSF | `config.yaml` → prefiring_eras, hem_veto, lsf_sf | When they apply |
| DY, signal, PDF | `config.yaml` → dy_zpt_reweight, dy_reshape, dy_reweight_datasets, signal_kfactor, pdf_sets + files | MC |
| Fileset JSONs | `data/configs/` + `data/filesets/` | Yes |
| Signal mass CSV | `data/signal_points/<era>_mass_points.csv` | Yes (signal) |
| WR_Plotter | `WR_Plotter/data/lumi.yaml` (+ `plot_settings/` only for era-specific ranges) | For plotting |
| Tests | `python -m pytest tests/ -v` | Recommended |
