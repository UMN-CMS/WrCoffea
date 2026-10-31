# Adding a new era

`RunIII2025Summer25` (data only, on the 2024 MC) has a complete set of entries in
`wrcoffea/config.yaml` to copy from.

## 1. Register the era

`wrcoffea/era_utils.py`, `ERA_MAPPING`: `"Run3Summer23": {"run": "Run3", "year": "2023"}`.
`run` (`RunII` or `Run3`) selects the trigger paths; `run` and `year` give the directory layout
`<run>/<year>/<era>` of configs, filesets, skims and outputs. `ERA_SUBDIRS` follows from it;
`tests/test_era_utils.py` checks every configured era.

In `wrcoffea/analyzer.py`: `RUN2_TRIGGERS` (the single-lepton paths of each Run 2 era; a Run 2 era
missing there gets the UL18 paths; Run 3 eras share one set) and `apply_noise_filter()` (the
era-specific MET filters; an era not listed gets the common flags and a warning).

## 2. `wrcoffea/config.yaml`

Copy an existing era's entries. Every file a key maps must exist.

Required, the run stops without them:

- `lumis`: the era's luminosity in fb⁻¹ (the processed luminosity of the skims, step 5).
- `lumi_unc`: the fractional luminosity uncertainty.
- `default_mc_tag`: the DY fileset tag used without `--dy`.
- `dy_variants`: the `--dy` variants with filesets for the era.
- `lumi_jsons`: the golden JSON (data runs).
- `jerc_jsons`: the JEC and JER payloads and tags, AK4 and AK8.
- `electron_hlt_paths`: the path evaluated in the electron TRIGGER payload, once `electron_jsons`
  has a TRIGGER entry.
- `dy_reweight_datasets`: the regular expression of the DY datasets the Z-pT reweight and the
  reshape apply to, for every era with `dy_zpt_reweight` or `dy_reshape`.
- `mc_alias` (data-only eras): the era whose MC the era uses.

Corrections, identity when absent:

- `jetveto_jsons`, `jetveto_correction_names`, `jetveto_mode` (`event`, the Run 3 default, or `jet`).
- `jme_jsons`: the correctionlib jet ID (`jetid.json.gz`), needed for NanoAODv15 (no `Jet_jetId`).
- `muon_jsons`: RECO (reco, ID, isolation SFs), TRIGGER, ScaleNSmear; `muon_hlt_correction` when
  the TRIGGER correction name is not the default; `muon_highpt`: the high-pT momentum payload
  (`data/muon_highpt/`).
- `prefiring_eras`: the eras whose MC gets the NanoAOD L1 prefiring weight (UL).
- `pileup_jsons`, `pileup_correction_names`.
- `electron_jsons`: RECO (reco and ID SFs), TRIGGER, ScaleNSmear; `electron_sf_era_keys`: the era
  key inside the EGM payloads; `electron_reco_config`: the correction name, the reco working points
  and their pT split, the last measured bins (`pt_max`, `id_pt_max`).
- `lsf_sf`: the LSF scale factor per flavour. `hem_veto`: UL18 only.
- `dy_zpt_reweight`, `dy_reshape`, `signal_kfactor`: payloads under `data/reweights/`.
- `pdf_sets`: the PDF set of each MC sample's `LHEPdfWeight` (`data/pdf_sets/`, written by
  `scripts/pdf_sets.py`); without it no `--systs pdf`.

What each correction does and which payload every era uses: [corrections.md](corrections.md).

## 3. POG payloads

From the CAT metadata area on cvmfs ([corrections.md](corrections.md) names the campaigns), under the
paths `config.yaml` declares:

```
data/jsonpog/
├── EGM/Run3/Run3Summer23/
│   ├── electron.json.gz                # reco + ID SFs (RECO)
│   ├── electronHlt.json.gz             # trigger SF (TRIGGER)
│   └── electronSS_EtDependent.json.gz  # scale and smearing (ScaleNSmear)
├── JME/Run3/Run3Summer23/
│   ├── jet_jerc.json.gz                # AK4 JEC/JER
│   ├── fatJet_jerc.json.gz             # AK8 JEC/JER
│   ├── jetvetomaps.json.gz
│   └── jetid.json.gz                   # where jme_jsons maps it
├── LUM/Run3/Run3Summer23/
│   └── puWeights.json.gz
└── MUO/Run3/Run3Summer23/
    ├── muon_HighPt.json.gz             # reco/ID/iso SFs (RECO)
    ├── muon_Z.json.gz                  # trigger-OR efficiencies (TRIGGER)
    └── muon_scalesmearing.json.gz      # scale and smearing (ScaleNSmear)
```

The golden JSON goes under `data/lumis/<run>/<year>/` (eras of one year can share it) or
`data/lumis/<run>/<year>/<era>/`, mapped in `lumi_jsons`.

## 4. Configs, filesets, signal points

- Configs: `data/configs/Run3/2023/Run3Summer23/Run3Summer23_{data,mc_dy_<variant>,signal}.json`
  with the DAS paths ([inputs.md](inputs.md) has the schema).
- Skimmed filesets, once the era is skimmed:
  `python3 bin/skim.py fileset --plan-dir <campaign root>/plans/v2_Run3Summer23 --out <campaign root> --remote <remote> --era Run3Summer23`
  into `data/filesets/Run3/2023/Run3Summer23/skimmed/`.
- Unskimmed filesets, from the configs and DBS:
  `python3 bin/skim.py unskimmed-fileset --era Run3Summer23` into `.../unskimmed/`.
- Signal points: `data/signal_points/Run3Summer23_mass_points.csv`, header `WR,N`, one row per
  (m_WR, m_N). `run_analysis.py` validates `--mass` against it and picks the 9-point default grid
  from it (`select_default_signal_points()`); `--signal-points all` and `--mwr-range` select from
  the whole list.

## 5. Skim

[skimming.md](skimming.md): `plan`, then the groups with `bin/lpc_condor.sh` at the LPC or
`skim.py run` locally, `validate`, `upload` for local runs, `fileset`. For data, the processed
luminosity with `scripts/processed_lumi.py` goes into `config.yaml` and
`WR_Plotter/data/lumi.yaml` ([luminosity.md](luminosity.md)).

## 6. Check

```bash
python bin/run_analysis.py Run3Summer23 DYJets --maxchunks 1 --maxfiles 1 --chunksize 1000 --dir smoke   # delete the smoke output afterwards
python -m pytest tests -q      # config consistency: lumi_unc covers every lumi era, electron_sf_era_keys every electron_jsons era, ...
```

## 7. WR_Plotter

`WR_Plotter/data/lumi.yaml`: an entry per era, and the era added to its year's and `Run3`'s
`sub_eras` (a combination's luminosity is the sum of its eras):

```yaml
Run3Summer23:
  run: Run3
  year: "2023"
  lumi: <lumi>
  com: 13.6
  plot_settings: run3_subera
  <<: *run3_signals
```

`plot_settings` names a file in `WR_Plotter/data/plot_settings/` (`default.yaml` otherwise). An era
with its own ranges gets `<era>.yaml` starting with `inherit: default` and listing only the values
that differ. `WR_Plotter/data/sample_groups.yaml` is shared by all eras.

## Checklist

| Step | File(s) | Required |
| --- | --- | --- |
| ERA_MAPPING | `wrcoffea/era_utils.py` | yes |
| Run 2 triggers, MET filters | `wrcoffea/analyzer.py`: RUN2_TRIGGERS, `apply_noise_filter()` | yes |
| Lumi, DY tags | `config.yaml`: lumis, lumi_unc, default_mc_tag, dy_variants | yes |
| Golden JSON | `config.yaml`: lumi_jsons + file | data |
| JEC/JER | `config.yaml`: jerc_jsons + files | yes |
| Data-only era | `config.yaml`: mc_alias | data-only |
| Muon SFs | `config.yaml`: muon_jsons, muon_hlt_correction, muon_highpt + files | when available |
| Electron SFs | `config.yaml`: electron_jsons, electron_hlt_paths, electron_sf_era_keys, electron_reco_config + files | when available |
| Pileup | `config.yaml`: pileup_jsons, pileup_correction_names + file | when available |
| Jet veto maps | `config.yaml`: jetveto_jsons, jetveto_correction_names, jetveto_mode + file | when available |
| JME jet ID | `config.yaml`: jme_jsons + file | NanoAODv15 |
| Prefiring, HEM, LSF | `config.yaml`: prefiring_eras, hem_veto, lsf_sf | when they apply |
| DY, signal, PDF | `config.yaml`: dy_zpt_reweight, dy_reshape, dy_reweight_datasets, signal_kfactor, pdf_sets + files | MC |
| Configs and filesets | `data/configs/`, `data/filesets/` | yes |
| Signal mass CSV | `data/signal_points/<era>_mass_points.csv` | signal |
| WR_Plotter | `WR_Plotter/data/lumi.yaml`, `plot_settings/` for era-specific ranges | plotting |
| Tests | `python -m pytest tests -q` | yes |
