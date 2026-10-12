# Run 3 dataset configs for WrCoffea – DAS survey and draft configs

Generated 2026-10-04 from the CMS DBS REST API (`https://cmsweb.cern.ch/dbs/prod/global/DBSReader/`, endpoints `datasets`, `filesummaries`, `outputconfigs`, `datasetparents`, `runs`) with the user's grid proxy. Nothing under `/Users/billy/work/WrCoffea` was modified. All outputs are in `configs_run3/` next to this file:

* `configs_run3/<era>_{data,signal}.json` and `configs_run3/<era>_mc_dy_<family>.json` – the **current-recommendation set** (2022/2023 on NanoAODv12 with the `22Sep2023` data, 2024 on `RunIII2024Summer24NanoAODv15` with `MINIv6NANOv15` data, 2025 data on PromptReco NanoAODv15).
* `configs_run3/nanov15/<era>_*.json` – the **NanoAODv15 re-processing set** for the four 2022/2023 eras (data `NanoAODv15-v1`, MC `Run3Summer2{2,2EE,3,3BPix}NanoAODv15-150X_*`), only for the families/processes that exist there (see §7).
* `configs_run3/REPORT_signal.md` – the full per-mass-point signal tables (200 rows per era).
* The JSONs use exactly the key set of the current configs (`das_name, run, year, era, dataset, physics_group, xsec, total_uncertainty, genEventSumw, datatype`, data entries without the xsec/sumw keys). Entries whose DAS path already exists in a current config carry its `xsec`, `total_uncertainty` and `genEventSumw` verbatim; new entries have `genEventSumw: null`. A free-text `note` key (already used in the current configs) is added only where something needs attention (replaced INVALID version, ext entry, nested sample, value to verify).

## 1. Key findings

1. **Every DAS path in the current signal and data configs is VALID.** 65 entries in the current MC configs are now **INVALID** on DAS (all DY): the ten `DYto2L-2Jets_MLL-50_PTLL-<bin>_{1J,2J}` samples of every archived `nlo_ptll` config (2022, 2022EE, 2023, 2023BPix and the 2024 `Bin-` equivalents), several `lo_ht` HT bins (2022: `MLL-50to120_HT-70to100` v1; 2022EE: `MLL-120_HT-40to70` v3, `MLL-50to120_HT-100to400` v1, `MLL-50to120_HT-70to100` v3; 2023BPix: `MLL-50to120_HT-40to70` v3, `MLL-50to120_HT-70to100` v2) and three archived `nlo_njet` entries (2023 `1J` v3; 2023BPix `1J` v3, `2J` v3). The drafts replace each of them by the highest VALID version (§8 lists them; all need a re-skim). The note in the 2022 `lo_ht` config that "v1 is the only version with events" for `MLL-50to120_HT-70to100` is obsolete: v1 is INVALID and v2 is VALID with 29.55 M events.
2. **The pT(ll)×N-jet DY samples were invalidated centrally (McM, Aug 2025) and replaced by four *nested* samples** `DYto2L-2Jets_MLL-50_PTLL-{100,200,400,600}` (2024: `DYto2L-2Jets_Bin-MLL-50-PTLL-{100,...}`), each defined as pT(ll) > threshold and inclusive in jets, i.e. overlapping. The draft `nlo_ptll` configs contain these four samples with the cross sections used by cms-flaf/FLAF (sum of the former XSDB bins above the threshold). They cannot be added together; FLAF uses them with an `LHE_NpNLO >= 1` filter inside a stitching scheme. The 40–100 GeV bin no longer has a dedicated sample.
3. **DY variants missing from the current configs** (all VALID, all included in the drafts): low-mass companions `MLL-10to50` for the LO (4Jets), NLO (2Jets) and powheg (e/mu) families in every era; a complete **2024 HT-binned LO family** (14 `DYto2L-4Jets_Bin-HT-*-MLL-*` samples, no 2024 `lo_ht` config existed); LO jet-binned `DYto2L-4Jets_MLL-50_{1J..4J}` and LO pT(ll)-binned `DYto2L-4Jets_MLL-50_PTLL-*` (2022/2023 only – written as two new families `lo_njet`, `lo_ptll`); the nested NLO pT(ll) samples above; `ext1` extensions of the LO/NLO inclusive DY, `MLL-10to50`, `TTto2L2Nu`, `TTtoLNu2Q`, `TW`, `TTLL` samples in 2022/2022EE (and `TTLL`/`MLL-10to50` ext1 in 2023/2023BPix). In 2024 the combined-flavour `DYto2L-4Jets_Bin-MLL-50` does not exist; only the e/mu split samples do (as already used).
4. **Backgrounds**: the 2024 process list exists in every era, so the drafts use it everywhere: 4FS `TBbarQtoLNu/TbarBQtoLNu-t-channel` instead of the 5FS `TQbartoLNu/TbarQtoLNu` of the 2022/2023 configs, jet-binned `WtoLNu-4Jets_{1J..4J}` instead of the 12 HT×M(lν)-binned W samples, `TTLL`, `TTLNu-1Jets` (the `mg35x` re-generation, 20× the statistics of the old-MadGraph sample the 2022/2023 configs use) **plus** `TTLNu-EWK` (EW ttW; the 2024 config has only this and misses the QCD ttW), `WW/WZ/ZZ`, `WWW/WWZ/WZZ/ZZZ`. The HT-binned W samples remain VALID in all eras (they also exist for 2024, §10) and can be swapped back.
5. **Newer NanoAOD processings of 2022/2023 exist** (§7): NanoAODv15 (`CMSSW_15_0_15_patch4`, GT `150X_dataRun3_v5`) for all data eras incl. `SingleMuon` 2022C and 2023 B/C/D, and a NanoAODv15 MC campaign (`150X_mcRun3_202x_realistic_v1`) that is **complete for the inclusive LO/NLO DY, the low-mass companions, tt, tW, s-channel, inclusive 4FS t-channel, TTLL, TTLNu-1Jets, WW/WZ/ZZ and jet-binned W**, but has **no HT/MLL/NJ/pT(ll)-binned DY (except `MLL-50to120`), no tribosons, no `TTLNu-EWK`, no LNu-filtered t-channel and no WR signal**. NanoAODv13 (`ForNanov13`, data `16Dec2023`/`24Jan2024`) is also present but only for a handful of MC samples. The `nanov15/` set therefore contains only `lo_inc`/`nlo_inc` (17-sample background block) and the data.
6. **2024**: `MINIv6NANOv15` is the newest and correct choice for C–I (every version included; era I has `-v1` and `_v2-v1`). DBS parentage shows `/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD` ← `/Muon0/Run2024C-2024CDEReprocessing-v1/AOD` ← RAW, i.e. the C/D/E `MINIv6NANOv15` are made *from* the CDE re-reconstruction; the `Run2024C-2024CDEReprocessing-v1/NANOAOD` datasets are the older NanoAOD (`CMSSW_14_0_19_patch2`, `140X_dataRun3_v20`, Jan 2025) of that same re-reco and are superseded by `MINIv6NANOv15` (Mar 2025). The only 2024 MC string on DBS is `RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2` (v1 INVALID); no `realistic_v3/v4` and no NanoAODv16 exist. **No RunIII2024Summer24 WR signal exists** (0 datasets for `/WRto*/RunIII2024Summer24*/`); the 2024 signal config stays the Run3Summer23BPix placeholder.
7. **2025 is data-only**: eras B–G as `PromptReco-v1/-v2` NanoAODv15 (CMSSW_15_0_5…15_0_15_patch4, GT `150X_dataRun3_Prompt_v1`), 32 datasets, 8.16 G events, 12.4 TB; the latest golden JSON is `Cert_Collisions2025_391658_398903_Golden.json` (513 runs 391668–398860, all contained in the B–G PromptReco datasets). No `*Summer25*` MC campaign exists (only `Run3Winter25NanoAOD` pilot/validation samples, tt̄ INVALID), so no 2025 MC configs were written: 2025 reuses the RunIII2024Summer24 skims as PdmV recommends.
8. **Cross sections** (§9): the current values are consistent across eras except the NLO jet-binned DY (2022/2023: 5365/1015/382.8; 2024 flavour-split: 5382/972.3–975.5/372.4–374) – the 2024 numbers are *all-flavour-sized values on single-flavour samples* and are ~3× too large relative to the per-flavour 2113.4 used for the inclusive 2024 DY (needs verification). The 5FS t-channel values 130.0/78.56 of the 2022/2023 configs are consistent with *undecayed* inclusive cross sections (NNLO 145.0/87.2) and probably lack BR(W→ℓν)=0.326. Everything else agrees with public references to ≤1.2% (BR convention) or exactly. The inclusive DY 6340.2 pb (= 3 × 2113.4) is a theory-type number, −5.2% from the XSDB NLO value 6688 and +0.9% from the MATRIX NNLO-QCD+NLO-EW 3×2094.2 quoted by cmsdb.
9. Edge eras with golden-JSON runs but not in any config: 2022B (`SingleMuon`/`EGamma` `Run2022B-22Sep2023`, 19 golden runs 355374–355769), 2023B (`Run2023B-22Sep2023-v1`, 21 golden runs 366727–367079, pre-BPix), 2024B (`PromptReco` only, 16 golden runs, no `MINIv6NANOv15`). `Run2024J-PromptReco-v1` NanoAODs contain 1–2 events (dummy).

## 2. Files written

| file | content |
|---|---|
| `configs_run3/<era>_data.json` (6 files: Run3Summer22, Run3Summer22EE, Run3Summer23, Run3Summer23BPix, RunIII2024Summer24, RunIII2025Summer25) | 5 / 6 / 16 / 8 / 32 / 32 datasets |
| `configs_run3/<era>_mc_dy_lo_inc.json`, `_nlo_inc`, `_lo_ht`, `_nlo_mll`, `_nlo_njet`, `_nlo_ptll` (5 eras) | one DY family + the full background block (27 entries in 2022/2022EE incl. 6 ext1, 23 in 2023/2023BPix incl. 1 ext1, 22 in 2024) |
| `configs_run3/<era>_mc_dy_lo_njet.json`, `_lo_ptll` (2022/2022EE/2023/2023BPix only) | new families; no 2024 equivalents on DAS |
| `configs_run3/<era>_signal.json` (5 eras) | 200 datasets each; identical DAS sets to the current configs (all VALID); 2024 = 23BPix placeholder copied verbatim |
| `configs_run3/nanov15/<era>_data.json`, `_mc_dy_lo_inc.json`, `_mc_dy_nlo_inc.json` (4 eras) | NanoAODv15 set |
| `configs_run3/REPORT_signal.md` | full signal tables |

The 2025 era name `RunIII2025Summer25` is a placeholder (no MC campaign exists yet); `year` is the string `"2025"`, `run` is `"Run3"`, `dataset` follows `Muon0_Run2025C` (both PromptReco versions map to the same `dataset`, as `Muon0_Run2024I` does today). Note that `skimmed_fileset.py` collapses entries with the same `dataset` name, so nominal+ext1 pairs (same `dataset`) are skimmed into one directory; `genEventSumw` of an ext1 entry must be combined with its nominal.

## 3. Method

* Discovery: for each era campaign prefix (`Run3Summer22NanoAOD*`, `Run3Summer22EENanoAOD*`, `Run3Summer23NanoAOD*`, `Run3Summer23BPixNanoAOD*`, `RunIII2024Summer24NanoAOD*`, `*2025*`, `*Winter25*`, `*Summer25*`) the primary-dataset wildcards `DYto2*`, `DY*Jets*`, `TTto2L2Nu*`, `TTtoLNu2Q*`, `TTto4Q*`, `TWminus*`, `TbarWplus*`, `TBbar*`, `TbarB*`, `TQbar*`, `TbarQ*`, `WtoLNu*`, `WW*`, `WZ*`, `ZZ*`, `WWW*`, `WWZ*`, `WZZ*`, `ZZZ*`, `TTLL*`, `TTLNu*`, `TTW*`, `TTZ*`, `WRtoN*`, `WRto*` were queried with `dataset_access_type=*` and `detail=true` (so INVALID/PRODUCTION versions are seen too), e.g. `/DYto2*/Run3Summer22NanoAOD*/NANOAODSIM`. Data: `/SingleMuon/Run2022*/NANOAOD`, `/Muon/Run2022*/NANOAOD`, `/EGamma/Run2022*/NANOAOD`, `/{Muon0,Muon1,EGamma0,EGamma1}/Run2023*/NANOAOD`, `/…/Run2024*/NANOAOD`, `/…/Run2025*/NANOAOD`, `/Muon/Run2023*/NANOAOD`, `/Muon/Run2024*/NANOAOD`, `/Muon/Run2025*/NANOAOD`. 2025 MC: `/*/Run3Winter25NanoAOD*/NANOAODSIM` plus explicit probes `/TTto2L2Nu*/*/NANOAODSIM`, `/DYto2L-2Jets_MLL-50*/*/NANOAODSIM`, `/WW_TuneCP5_13p6TeV_pythia8/*/NANOAODSIM` (listing every campaign string). DBS wildcards are plain `*` (no character classes).
* Every DAS path of the 1230 distinct paths in the current configs was looked up individually (`datasets?dataset=<exact>&dataset_access_type=*`).
* Version rule: within the era campaign (`<Era>NanoAODv12-13?X_mcRun3_..._vN-vM` for 2022/2023, `RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_vN-vM` for 2024) the VALID dataset with the highest (global-tag vN, -vM) is chosen; `_extN` productions are separate entries with the same xsec; special-purpose strings (`JMENano`, `BTVNano`, `EGMNano`, `MUOPOGNano`, `FS`, `NoPU`, `KeepSi`, `Poisson60ForMUOVal`) are ignored; for `TTLNu-1Jets` the `mg35x` string is preferred. Other versions and their status/event counts are listed per era.
* `filesummaries` was queried for every chosen dataset (1700 MC + 110 data queries); `outputconfigs` (release, global tag) for every candidate data dataset; `runs` for the 2025 datasets and for the edge-era checks; `datasetparents` for the 2024 provenance.
* Cross sections: XSDB is unreachable from this machine and both twikis need CERN SSO (`MATRIXCrossSectionsat13p6TeV` redirects to `auth.cern.ch`; `StandardModelCrossSectionsat13p6TeV` returns 404 without login). Public references used instead: LHCPhysics `TtbarNNLO` and `SingleTopNNLORef` twikis (public), `cms-flaf/FLAF` `config/crossSections13p6TeV.yaml` (XSDB values with retrieval dates, downloaded), `uhh-cms/cmsdb` `processes/ewk.py`, `processes/top.py` (downloaded), FLAF PR #336 / H_mumu PR #74 (nested PTLL samples). Golden JSONs: `data/lumis/...` in the repo for 2022–2024 and `cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions25/` for 2025 (directory listing fetched; the latest file downloaded).

## 4. PdmV recommendations and luminosities (July 2026 PPD tables, PdmVRun3Analysis)

| era (config) | data | MC campaign | golden-JSON lumi [fb⁻¹] |
|---|---|---|---|
| Run3Summer22 (2022 C, D) | `22Sep2023` NanoAODv12 (current) / `NanoAODv15-v1` (new) | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5` / `Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1` | C 5.02 + D 2.97 = 7.99 |
| Run3Summer22EE (2022 E, F, G) | `22Sep2023` / `NanoAODv15-v1` | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6` / `Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1` | E 5.81 + F 17.78 + G 3.09 = 26.68 |
| Run3Summer23 (2023 C v1–v4) | `22Sep2023_vN` / `NanoAODv15[_vN]-v1` | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14/v15` / `Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1` | C 17.96 |
| Run3Summer23BPix (2023 D v1–v2) | `22Sep2023_vN` / `NanoAODv15[_v2]-v1` | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2/v6` / `Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1` | D 9.68 |
| RunIII2024Summer24 (2024 C–I) | `MINIv6NANOv15` (C, D, E from the CDE re-reco AOD; F–I from PromptReco AOD), every version | `RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2` (only GT string on DBS; the PPD table's `realistic_v4` does not exist as a NanoAOD campaign string) | C 7.26 + D 7.98 + E 11.42 + F 28.04 + G 38.07 + H 5.49 + I 11.56 = 109.82 |
| RunIII2025Summer25 (2025 B–G) | `PromptReco-vN` NanoAODv15, every version | none – reuse RunIII2024Summer24 MC/skims | B 0.25 + C 21.56 + D 25.82 + E 14.05 + F 26.69 + G 22.25 = 110.62 (preliminary, no normtag) |

The per-era sums reproduce the `lumis` block of `wrcoffea/config.yaml` (7.99, 26.68, 17.96, 9.68, 109.82). **Update 2026-10-06:** 2024 now uses 109.70 fb⁻¹, the luminosity of the data actually in the skims_v2 skims (B-I, processed lumisections intersected with the golden JSON, normtag_PHYSICS); see `docs/skimming.md`, section "Luminosity of the skimmed data". **Update 2026-10-07:** Run3Summer23 uses 18.59 fb⁻¹ (2023B+C processed) and Run3Summer23BPix 9.675; the 2022 eras use their processed values since 2026-10-08 (Run3Summer22 2022B+C+D 8.08, Run3Summer22EE 26.64). The 2022–2024 lumis are with normtag; the 2025 ones are preliminary.

## 5. Data

### 5.1 Chosen datasets (current-recommendation set)

All chosen data paths are VALID. Release/global tag come from DBS `outputconfigs`; `NanoAODv12` = `CMSSW_13_0_13`, `NanoAODv13` = `CMSSW_13_3_1_patch1` (`ForNanov13` GT), `NanoAODv15` = `CMSSW_15_0_x`.

### Run3Summer22_data.json (5 datasets)

| DAS path | physics_group | CMSSW release | global tag | files | events | GB | runs (DBS) |
|---|---|---|---|---|---|---|---|
| `/SingleMuon/Run2022C-22Sep2023-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_v2 | 35 | 20,162,441 | 17 |  |
| `/Muon/Run2022C-22Sep2023-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_v2 | 124 | 138,427,345 | 114 |  |
| `/Muon/Run2022D-22Sep2023-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_v2 | 82 | 75,468,381 | 62 |  |
| `/EGamma/Run2022C-22Sep2023-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_v2 | 313 | 263,689,151 | 250 |  |
| `/EGamma/Run2022D-22Sep2023-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_v2 | 109 | 89,134,996 | 88 |  |

Total: 586,882,314 events, 0.53 TB

### Run3Summer22EE_data.json (6 datasets)

| DAS path | physics_group | CMSSW release | global tag | files | events | GB | runs (DBS) |
|---|---|---|---|---|---|---|---|
| `/Muon/Run2022E-22Sep2023-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_v2 | 147 | 141,460,608 | 116 |  |
| `/Muon/Run2022F-22Sep2023-v2/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 359 | 449,887,248 | 376 |  |
| `/Muon/Run2022G-22Sep2023-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 88 | 76,689,396 | 64 |  |
| `/EGamma/Run2022E-22Sep2023-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_v2 | 188 | 148,654,795 | 146 |  |
| `/EGamma/Run2022F-22Sep2023-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 490 | 464,373,259 | 459 |  |
| `/EGamma/Run2022G-22Sep2023-v2/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 87 | 76,807,350 | 76 |  |

Total: 1,357,872,656 events, 1.24 TB

### Run3Summer23_data.json (16 datasets)

| DAS path | physics_group | CMSSW release | global tag | files | events | GB | runs (DBS) |
|---|---|---|---|---|---|---|---|
| `/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 90 | 54,715,896 | 44 |  |
| `/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 31 | 17,063,451 | 14 |  |
| `/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 42 | 20,015,377 | 16 |  |
| `/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 165 | 138,943,783 | 118 |  |
| `/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 73 | 54,698,315 | 44 |  |
| `/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 33 | 17,059,895 | 14 |  |
| `/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 30 | 20,010,429 | 16 |  |
| `/Muon1/Run2023C-22Sep2023_v4-v2/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 108 | 138,912,262 | 118 |  |
| `/EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 90 | 67,598,081 | 65 |  |
| `/EGamma0/Run2023C-22Sep2023_v2-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 22 | 17,233,307 | 17 |  |
| `/EGamma0/Run2023C-22Sep2023_v3-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 26 | 21,993,048 | 21 |  |
| `/EGamma0/Run2023C-22Sep2023_v4-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 168 | 160,108,119 | 161 |  |
| `/EGamma1/Run2023C-22Sep2023_v1-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 71 | 67,582,665 | 65 |  |
| `/EGamma1/Run2023C-22Sep2023_v2-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 24 | 17,230,822 | 17 |  |
| `/EGamma1/Run2023C-22Sep2023_v3-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 25 | 21,987,586 | 21 |  |
| `/EGamma1/Run2023C-22Sep2023_v4-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 177 | 160,049,621 | 161 |  |

Total: 995,202,657 events, 0.91 TB

### Run3Summer23BPix_data.json (8 datasets)

| DAS path | physics_group | CMSSW release | global tag | files | events | GB | runs (DBS) |
|---|---|---|---|---|---|---|---|
| `/Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 125 | 100,291,308 | 84 |  |
| `/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 32 | 21,462,916 | 18 |  |
| `/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 100 | 100,281,976 | 84 |  |
| `/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD` | Muon | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 30 | 21,463,645 | 18 |  |
| `/EGamma0/Run2023D-22Sep2023_v1-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 124 | 105,892,646 | 106 |  |
| `/EGamma0/Run2023D-22Sep2023_v2-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 33 | 22,657,211 | 23 |  |
| `/EGamma1/Run2023D-22Sep2023_v1-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 104 | 105,824,276 | 106 |  |
| `/EGamma1/Run2023D-22Sep2023_v2-v1/NANOAOD` | EGamma | CMSSW_13_0_13 | 130X_dataRun3_PromptAnalysis_v1 | 30 | 22,653,287 | 23 |  |

Total: 500,527,265 events, 0.46 TB

### RunIII2024Summer24_data.json (32 datasets)

| DAS path | physics_group | CMSSW release | global tag | files | events | GB | runs (DBS) |
|---|---|---|---|---|---|---|---|
| `/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 103 | 97,505,587 | 135 |  |
| `/Muon1/Run2024C-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 94 | 97,531,998 | 135 |  |
| `/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 167 | 157,351,226 | 251 |  |
| `/EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 247 | 157,860,731 | 252 |  |
| `/Muon0/Run2024D-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 180 | 120,787,065 | 164 |  |
| `/Muon1/Run2024D-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 168 | 120,467,371 | 164 |  |
| `/EGamma0/Run2024D-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 173 | 156,558,757 | 249 |  |
| `/EGamma1/Run2024D-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 279 | 156,275,778 | 249 |  |
| `/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 255 | 169,640,946 | 235 |  |
| `/Muon1/Run2024E-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 171 | 172,848,674 | 239 |  |
| `/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 267 | 249,417,634 | 393 |  |
| `/EGamma1/Run2024E-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 335 | 249,489,829 | 393 |  |
| `/Muon0/Run2024F-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 594 | 442,432,787 | 615 |  |
| `/Muon1/Run2024F-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 538 | 442,360,423 | 615 |  |
| `/EGamma0/Run2024F-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 719 | 638,196,622 | 1012 |  |
| `/EGamma1/Run2024F-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 747 | 631,079,889 | 1000 |  |
| `/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 948 | 642,028,803 | 888 |  |
| `/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 565 | 641,959,643 | 886 |  |
| `/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 850 | 903,520,258 | 1429 |  |
| `/EGamma1/Run2024G-MINIv6NANOv15-v2/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 888 | 903,441,926 | 1429 |  |
| `/Muon0/Run2024H-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 152 | 93,983,627 | 130 |  |
| `/Muon1/Run2024H-MINIv6NANOv15-v2/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 93 | 93,981,102 | 129 |  |
| `/EGamma0/Run2024H-MINIv6NANOv15-v2/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 138 | 134,680,448 | 213 |  |
| `/EGamma1/Run2024H-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 191 | 134,835,799 | 214 |  |
| `/Muon0/Run2024I-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 173 | 97,634,104 | 132 |  |
| `/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 174 | 105,194,627 | 139 |  |
| `/Muon1/Run2024I-MINIv6NANOv15-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 94 | 97,630,010 | 132 |  |
| `/Muon1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | Muon | CMSSW_15_0_2 | 150X_dataRun3_v2 | 133 | 105,189,040 | 139 |  |
| `/EGamma0/Run2024I-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 211 | 132,904,290 | 208 |  |
| `/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 198 | 150,687,674 | 229 |  |
| `/EGamma1/Run2024I-MINIv6NANOv15-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 176 | 132,903,874 | 208 |  |
| `/EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | EGamma | CMSSW_15_0_2 | 150X_dataRun3_v2 | 179 | 150,687,112 | 229 |  |

Total: 8,581,067,654 events, 12.84 TB

### RunIII2025Summer25_data.json (32 datasets)

| DAS path | physics_group | CMSSW release | global tag | files | events | GB | runs (DBS) |
|---|---|---|---|---|---|---|---|
| `/Muon0/Run2025B-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_5 | 150X_dataRun3_Prompt_v1 | 82 | 5,674,007 | 7 | 391548-392112 (68 runs) |
| `/Muon0/Run2025C-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_6, CMSSW_15_0_6_patch1 | 150X_dataRun3_Prompt_v1 | 677 | 261,592,099 | 372 | 392174-393087 (77 runs) |
| `/Muon0/Run2025C-PromptReco-v2/NANOAOD` | Muon | CMSSW_15_0_7, CMSSW_15_0_8 | 150X_dataRun3_Prompt_v1 | 356 | 148,575,745 | 213 | 393111-393516 (37 runs) |
| `/Muon0/Run2025D-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_10, CMSSW_15_0_10_patch1, CMSSW_15_0_12_patch1 | 150X_dataRun3_Prompt_v1 | 1,166 | 479,676,562 | 698 | 394393-395948 (187 runs) |
| `/Muon0/Run2025E-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_13, CMSSW_15_0_13_patch1 | 150X_dataRun3_Prompt_v1 | 606 | 265,645,863 | 388 | 395982-396422 (66 runs) |
| `/Muon0/Run2025F-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_13_patch2, CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 884 | 377,119,912 | 544 | 396629-397596 (143 runs) |
| `/Muon0/Run2025F-PromptReco-v2/NANOAOD` | Muon | CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 344 | 143,519,758 | 208 | 397619-397853 (40 runs) |
| `/Muon0/Run2025G-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch1, CMSSW_15_0_15_patch4 | 150X_dataRun3_Prompt_v1 | 1,009 | 442,329,683 | 629 | 397954-398903 (126 runs) |
| `/Muon1/Run2025B-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_5 | 150X_dataRun3_Prompt_v1 | 82 | 5,667,234 | 7 | 391548-392112 (68 runs) |
| `/Muon1/Run2025C-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_6, CMSSW_15_0_6_patch1 | 150X_dataRun3_Prompt_v1 | 683 | 261,572,965 | 372 | 392174-393087 (79 runs) |
| `/Muon1/Run2025C-PromptReco-v2/NANOAOD` | Muon | CMSSW_15_0_7, CMSSW_15_0_8 | 150X_dataRun3_Prompt_v1 | 350 | 148,565,659 | 213 | 393111-393516 (37 runs) |
| `/Muon1/Run2025D-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_10, CMSSW_15_0_10_patch1, CMSSW_15_0_12_patch1 | 150X_dataRun3_Prompt_v1 | 1,159 | 479,642,866 | 698 | 394393-395948 (186 runs) |
| `/Muon1/Run2025E-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_13, CMSSW_15_0_13_patch1 | 150X_dataRun3_Prompt_v1 | 611 | 265,630,211 | 388 | 395982-396422 (66 runs) |
| `/Muon1/Run2025F-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_13_patch2, CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 891 | 377,092,035 | 544 | 396629-397596 (143 runs) |
| `/Muon1/Run2025F-PromptReco-v2/NANOAOD` | Muon | CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 345 | 143,511,119 | 208 | 397619-397853 (40 runs) |
| `/Muon1/Run2025G-PromptReco-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch1, CMSSW_15_0_15_patch4 | 150X_dataRun3_Prompt_v1 | 1,003 | 442,232,031 | 630 | 397954-398903 (125 runs) |
| `/EGamma0/Run2025B-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_5 | 150X_dataRun3_Prompt_v1 | 80 | 5,281,596 | 8 | 391548-392112 (66 runs) |
| `/EGamma0/Run2025C-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_6, CMSSW_15_0_6_patch1 | 150X_dataRun3_Prompt_v1 | 705 | 256,242,724 | 406 | 392174-393087 (77 runs) |
| `/EGamma0/Run2025C-PromptReco-v2/NANOAOD` | EGamma | CMSSW_15_0_7, CMSSW_15_0_8 | 150X_dataRun3_Prompt_v1 | 342 | 133,952,379 | 213 | 393111-393516 (35 runs) |
| `/EGamma0/Run2025D-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_10, CMSSW_15_0_10_patch1, CMSSW_15_0_12_patch1 | 150X_dataRun3_Prompt_v1 | 1,105 | 414,017,171 | 674 | 394402-395948 (174 runs) |
| `/EGamma0/Run2025E-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_13, CMSSW_15_0_13_patch1 | 150X_dataRun3_Prompt_v1 | 595 | 240,927,385 | 393 | 395982-396422 (62 runs) |
| `/EGamma0/Run2025F-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_13_patch2, CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 888 | 346,019,505 | 558 | 396629-397596 (143 runs) |
| `/EGamma0/Run2025F-PromptReco-v2/NANOAOD` | EGamma | CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 351 | 139,317,024 | 225 | 397619-397837 (43 runs) |
| `/EGamma0/Run2025G-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_15, CMSSW_15_0_15_patch1, CMSSW_15_0_15_patch4 | 150X_dataRun3_Prompt_v1 | 1,012 | 421,398,444 | 672 | 397867-398903 (119 runs) |
| `/EGamma1/Run2025B-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_5 | 150X_dataRun3_Prompt_v1 | 80 | 5,277,455 | 8 | 391548-392112 (66 runs) |
| `/EGamma1/Run2025C-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_6, CMSSW_15_0_6_patch1 | 150X_dataRun3_Prompt_v1 | 702 | 256,233,285 | 406 | 392174-393087 (74 runs) |
| `/EGamma1/Run2025C-PromptReco-v2/NANOAOD` | EGamma | CMSSW_15_0_7, CMSSW_15_0_8 | 150X_dataRun3_Prompt_v1 | 347 | 133,951,067 | 213 | 393111-393516 (34 runs) |
| `/EGamma1/Run2025D-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_10, CMSSW_15_0_10_patch1, CMSSW_15_0_12_patch1 | 150X_dataRun3_Prompt_v1 | 1,096 | 413,998,146 | 674 | 394402-395948 (173 runs) |
| `/EGamma1/Run2025E-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_13, CMSSW_15_0_13_patch1 | 150X_dataRun3_Prompt_v1 | 596 | 240,921,314 | 393 | 395982-396422 (63 runs) |
| `/EGamma1/Run2025F-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_13_patch2, CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 875 | 346,000,262 | 558 | 396629-397596 (143 runs) |
| `/EGamma1/Run2025F-PromptReco-v2/NANOAOD` | EGamma | CMSSW_15_0_14 | 150X_dataRun3_Prompt_v1 | 350 | 139,310,909 | 225 | 397619-397837 (44 runs) |
| `/EGamma1/Run2025G-PromptReco-v1/NANOAOD` | EGamma | CMSSW_15_0_15, CMSSW_15_0_15_patch1, CMSSW_15_0_15_patch4 | 150X_dataRun3_Prompt_v1 | 1,010 | 421,102,377 | 672 | 397867-398903 (118 runs) |

Total: 8,161,998,792 events, 12.41 TB

### 5.2 Other NanoAOD processings of the same primary datasets (what else is on DAS)

For each chosen dataset the other VALID, non-special NanoAOD processings of the same PD/era are listed with their event counts and release. Special-purpose nanos (`BTVNano`, `JMENano`, `MuoPOGNano`/`MUOPOGNANO`, `EGMNano`, `ZMu_*`, `TrkRadDamage`, `MLPF*`, `PF*PU`, `ECAL_*`, `PromptNanoAODv1x`, `ReRecoNanoAODv11`, `18Sep2023`, `Backfill`) are excluded.

* **2022**: `22Sep2023` (NanoAODv12, current) → `16Dec2023` (NanoAODv13, GT `133X_dataRun3_ForNanov13_v1`, all eras), `19Dec2023` (NanoAODv12 re-nano with `130X_dataRun3_PromptAnalysis_HcalSiPM_v1`, only Muon/EGamma F and G) and `NanoAODv15-v1` (`CMSSW_15_0_15_patch4`, `150X_dataRun3_v5`, all eras A–G incl. `SingleMuon`). Event counts agree with `22Sep2023` to ≤0.2% (e.g. Muon 2022F 449.9 M vs 449.2 M; 2022G identical).
* **2023**: `22Sep2023_vN` (NanoAODv12, current) → `24Jan2024_vN` (NanoAODv13) and `NanoAODv15[_vN]-v1` for B, C v1–v4, D v1–v2 (naming: the v1 sub-era is `Run2023C-NanoAODv15-v1`, the others `Run2023C-NanoAODv15_v2-v1` etc.). `Muon1 Run2023C-24Jan2024_v4` exists as `-v1` (131.8 M, incomplete) and `-v2` (138.8 M).
* **2024**: `MINIv6NANOv15` (current, `CMSSW_15_0_2`, `150X_dataRun3_v2`, created Mar–Apr 2025) vs `PromptReco` (`CMSSW_14_0_x`, prompt NanoAOD) and `2024CDEReprocessing-v1/-v2` (`CMSSW_14_0_19_patch2`, `140X_dataRun3_v20`; C/D/E only; EGamma `-v1` INVALID, `-v2` created 2026-07-12). DBS parentage: `MINIv6NANOv15` C/D/E ← `2024CDEReprocessing/AOD` ← RAW; F–I ← `PromptReco/AOD`. **Verdict: the `2024CDEReprocessing` NANOAOD datasets do not supersede `MINIv6NANOv15`; they are the older-format NanoAOD of the same re-reconstruction, and `MINIv6NANOv15` is the PdmV-recommended, newest and only format consistent with the Summer24 NanoAODv15 MC.** Event counts agree to ≤0.2% (e.g. Muon0 2024E: 172.7 M re-reco NANO vs 169.6 M MINIv6NANOv15 vs 97.0+79.4 M prompt).
* **2025**: PromptReco only (no re-reco, no re-nano yet).

**Run3Summer22**

| current (in draft) | events | release | other NanoAOD processings of the same PD/era (events, release) |
|---|---|---|---|
| `/SingleMuon/Run2022C-22Sep2023-v1/NANOAOD` | 20,162,441 | CMSSW_13_0_13 | `/SingleMuon/Run2022C-16Dec2023-v1/NANOAOD` (20,122,806, CMSSW_13_3_1_patch1)<br>`/SingleMuon/Run2022C-NanoAODv15-v1/NANOAOD` (20,162,441, CMSSW_15_0_15_patch4) |
| `/Muon/Run2022C-22Sep2023-v1/NANOAOD` | 138,427,345 | CMSSW_13_0_13 | `/Muon/Run2022C-16Dec2023-v1/NANOAOD` (138,281,764, CMSSW_13_3_1_patch1)<br>`/Muon/Run2022C-NanoAODv15-v1/NANOAOD` (138,329,693, CMSSW_15_0_15_patch4) |
| `/Muon/Run2022D-22Sep2023-v1/NANOAOD` | 75,468,381 | CMSSW_13_0_13 | `/Muon/Run2022D-16Dec2023-v1/NANOAOD` (75,336,535, CMSSW_13_3_1_patch1)<br>`/Muon/Run2022D-NanoAODv15-v1/NANOAOD` (75,440,027, CMSSW_15_0_15_patch4) |
| `/EGamma/Run2022C-22Sep2023-v1/NANOAOD` | 263,689,151 | CMSSW_13_0_13 | `/EGamma/Run2022C-16Dec2023-v1/NANOAOD` (263,549,470, CMSSW_13_3_1_patch1)<br>`/EGamma/Run2022C-NanoAODv15-v1/NANOAOD` (263,501,653, CMSSW_15_0_15_patch4) |
| `/EGamma/Run2022D-22Sep2023-v1/NANOAOD` | 89,134,996 | CMSSW_13_0_13 | `/EGamma/Run2022D-16Dec2023-v1/NANOAOD` (89,103,530, CMSSW_13_3_1_patch1)<br>`/EGamma/Run2022D-NanoAODv15-v1/NANOAOD` (89,081,259, CMSSW_15_0_15_patch4) |

**Run3Summer22EE**

| current (in draft) | events | release | other NanoAOD processings of the same PD/era (events, release) |
|---|---|---|---|
| `/Muon/Run2022E-22Sep2023-v1/NANOAOD` | 141,460,608 | CMSSW_13_0_13 | `/Muon/Run2022E-16Dec2023-v1/NANOAOD` (141,480,973, CMSSW_13_3_1_patch1)<br>`/Muon/Run2022E-NanoAODv15-v1/NANOAOD` (141,480,973, CMSSW_15_0_15_patch4) |
| `/Muon/Run2022F-22Sep2023-v2/NANOAOD` | 449,887,248 | CMSSW_13_0_13 | `/Muon/Run2022F-16Dec2023-v1/NANOAOD` (449,356,981, CMSSW_13_3_1_patch1)<br>`/Muon/Run2022F-19Dec2023-v1/NANOAOD` (449,188,966, CMSSW_13_0_17)<br>`/Muon/Run2022F-NanoAODv15-v1/NANOAOD` (449,185,088, CMSSW_15_0_15_patch4) |
| `/Muon/Run2022G-22Sep2023-v1/NANOAOD` | 76,689,396 | CMSSW_13_0_13 | `/Muon/Run2022G-16Dec2023-v1/NANOAOD` (76,639,473, CMSSW_13_3_1_patch1)<br>`/Muon/Run2022G-19Dec2023-v2/NANOAOD` (76,479,102, CMSSW_13_0_17)<br>`/Muon/Run2022G-NanoAODv15-v1/NANOAOD` (76,689,396, CMSSW_15_0_15_patch4) |
| `/EGamma/Run2022E-22Sep2023-v1/NANOAOD` | 148,654,795 | CMSSW_13_0_13 | `/EGamma/Run2022E-16Dec2023-v1/NANOAOD` (148,601,587, CMSSW_13_3_1_patch1)<br>`/EGamma/Run2022E-NanoAODv15-v1/NANOAOD` (148,616,546, CMSSW_15_0_15_patch4) |
| `/EGamma/Run2022F-22Sep2023-v1/NANOAOD` | 464,373,259 | CMSSW_13_0_13 | `/EGamma/Run2022F-16Dec2023-v1/NANOAOD` (463,656,342, CMSSW_13_3_1_patch1)<br>`/EGamma/Run2022F-19Dec2023-v1/NANOAOD` (463,389,499, CMSSW_13_0_17)<br>`/EGamma/Run2022F-NanoAODv15-v1/NANOAOD` (463,684,806, CMSSW_15_0_15_patch4) |
| `/EGamma/Run2022G-22Sep2023-v2/NANOAOD` | 76,807,350 | CMSSW_13_0_13 | `/EGamma/Run2022G-16Dec2023-v1/NANOAOD` (76,724,231, CMSSW_13_3_1_patch1)<br>`/EGamma/Run2022G-19Dec2023-v1/NANOAOD` (76,638,647, CMSSW_13_0_17)<br>`/EGamma/Run2022G-NanoAODv15-v1/NANOAOD` (76,620,820, CMSSW_15_0_15_patch4) |

**Run3Summer23**

| current (in draft) | events | release | other NanoAOD processings of the same PD/era (events, release) |
|---|---|---|---|
| `/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD` | 54,715,896 | CMSSW_13_0_13 | `/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,063,451, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD` (20,015,377, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD` (138,943,783, CMSSW_13_0_13)<br>`/Muon0/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,715,896, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,039,517, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,015,377, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v4-v1/NANOAOD` (138,943,783, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-NanoAODv15-v1/NANOAOD` (54,715,896, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,063,451, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,015,377, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,943,783, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-PromptReco-v4/NANOAOD` (138,943,783, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD` | 17,063,451 | CMSSW_13_0_13 | `/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD` (54,715,896, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD` (20,015,377, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD` (138,943,783, CMSSW_13_0_13)<br>`/Muon0/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,715,896, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,039,517, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,015,377, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v4-v1/NANOAOD` (138,943,783, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-NanoAODv15-v1/NANOAOD` (54,715,896, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,063,451, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,015,377, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,943,783, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-PromptReco-v4/NANOAOD` (138,943,783, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD` | 20,015,377 | CMSSW_13_0_13 | `/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD` (54,715,896, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,063,451, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD` (138,943,783, CMSSW_13_0_13)<br>`/Muon0/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,715,896, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,039,517, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,015,377, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v4-v1/NANOAOD` (138,943,783, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-NanoAODv15-v1/NANOAOD` (54,715,896, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,063,451, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,015,377, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,943,783, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-PromptReco-v4/NANOAOD` (138,943,783, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon0/Run2023C-22Sep2023_v4-v1/NANOAOD` | 138,943,783 | CMSSW_13_0_13 | `/Muon0/Run2023C-22Sep2023_v1-v1/NANOAOD` (54,715,896, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,063,451, CMSSW_13_0_13)<br>`/Muon0/Run2023C-22Sep2023_v3-v1/NANOAOD` (20,015,377, CMSSW_13_0_13)<br>`/Muon0/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,715,896, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,039,517, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,015,377, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-24Jan2024_v4-v1/NANOAOD` (138,943,783, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023C-NanoAODv15-v1/NANOAOD` (54,715,896, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,063,451, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,015,377, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,943,783, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023C-PromptReco-v4/NANOAOD` (138,943,783, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD` | 54,698,315 | CMSSW_13_0_13 | `/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,059,895, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD` (20,010,429, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v4-v2/NANOAOD` (138,912,262, CMSSW_13_0_13)<br>`/Muon1/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,621,922, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,059,895, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,010,429, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v1/NANOAOD` (131,799,838, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v2/NANOAOD` (138,828,860, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-NanoAODv15-v1/NANOAOD` (54,621,922, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,059,895, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,010,429, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,886,941, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-PromptReco-v4/NANOAOD` (138,917,988, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD` | 17,059,895 | CMSSW_13_0_13 | `/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD` (54,698,315, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD` (20,010,429, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v4-v2/NANOAOD` (138,912,262, CMSSW_13_0_13)<br>`/Muon1/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,621,922, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,059,895, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,010,429, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v1/NANOAOD` (131,799,838, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v2/NANOAOD` (138,828,860, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-NanoAODv15-v1/NANOAOD` (54,621,922, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,059,895, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,010,429, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,886,941, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-PromptReco-v4/NANOAOD` (138,917,988, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD` | 20,010,429 | CMSSW_13_0_13 | `/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD` (54,698,315, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,059,895, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v4-v2/NANOAOD` (138,912,262, CMSSW_13_0_13)<br>`/Muon1/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,621,922, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,059,895, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,010,429, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v1/NANOAOD` (131,799,838, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v2/NANOAOD` (138,828,860, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-NanoAODv15-v1/NANOAOD` (54,621,922, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,059,895, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,010,429, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,886,941, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-PromptReco-v4/NANOAOD` (138,917,988, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/Muon1/Run2023C-22Sep2023_v4-v2/NANOAOD` | 138,912,262 | CMSSW_13_0_13 | `/Muon1/Run2023C-22Sep2023_v1-v1/NANOAOD` (54,698,315, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,059,895, CMSSW_13_0_13)<br>`/Muon1/Run2023C-22Sep2023_v3-v1/NANOAOD` (20,010,429, CMSSW_13_0_13)<br>`/Muon1/Run2023C-24Jan2024_v1-v1/NANOAOD` (54,621,922, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,059,895, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v3-v1/NANOAOD` (20,010,429, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v1/NANOAOD` (131,799,838, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-24Jan2024_v4-v2/NANOAOD` (138,828,860, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023C-NanoAODv15-v1/NANOAOD` (54,621,922, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,059,895, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (20,010,429, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (138,886,941, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023C-PromptReco-v4/NANOAOD` (138,917,988, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD` | 67,598,081 | CMSSW_13_0_13 | `/EGamma0/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,233,307, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v3-v1/NANOAOD` (21,993,048, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v4-v1/NANOAOD` (160,108,119, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,598,081, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,233,307, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,993,046, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v4-v1/NANOAOD` (160,108,092, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-NanoAODv15-v1/NANOAOD` (67,598,081, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,233,307, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,993,048, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (160,108,119, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-PromptReco-v4/NANOAOD` (160,108,119, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma0/Run2023C-22Sep2023_v2-v1/NANOAOD` | 17,233,307 | CMSSW_13_0_13 | `/EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD` (67,598,081, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v3-v1/NANOAOD` (21,993,048, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v4-v1/NANOAOD` (160,108,119, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,598,081, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,233,307, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,993,046, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v4-v1/NANOAOD` (160,108,092, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-NanoAODv15-v1/NANOAOD` (67,598,081, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,233,307, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,993,048, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (160,108,119, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-PromptReco-v4/NANOAOD` (160,108,119, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma0/Run2023C-22Sep2023_v3-v1/NANOAOD` | 21,993,048 | CMSSW_13_0_13 | `/EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD` (67,598,081, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,233,307, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v4-v1/NANOAOD` (160,108,119, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,598,081, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,233,307, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,993,046, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v4-v1/NANOAOD` (160,108,092, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-NanoAODv15-v1/NANOAOD` (67,598,081, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,233,307, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,993,048, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (160,108,119, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-PromptReco-v4/NANOAOD` (160,108,119, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma0/Run2023C-22Sep2023_v4-v1/NANOAOD` | 160,108,119 | CMSSW_13_0_13 | `/EGamma0/Run2023C-22Sep2023_v1-v1/NANOAOD` (67,598,081, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,233,307, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-22Sep2023_v3-v1/NANOAOD` (21,993,048, CMSSW_13_0_13)<br>`/EGamma0/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,598,081, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,233,307, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,993,046, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-24Jan2024_v4-v1/NANOAOD` (160,108,092, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023C-NanoAODv15-v1/NANOAOD` (67,598,081, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,233,307, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,993,048, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-NanoAODv15_v4-v1/NANOAOD` (160,108,119, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023C-PromptReco-v4/NANOAOD` (160,108,119, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma1/Run2023C-22Sep2023_v1-v1/NANOAOD` | 67,582,665 | CMSSW_13_0_13 | `/EGamma1/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,230,822, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v3-v1/NANOAOD` (21,987,586, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v4-v1/NANOAOD` (160,049,621, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,530,273, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,230,822, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,939,165, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v4-v1/NANOAOD` (159,997,157, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-NanoAODv15-v1/NANOAOD` (67,530,273, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,230,822, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,987,586, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (159,997,174, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-PromptReco-v4/NANOAOD` (160,074,872, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma1/Run2023C-22Sep2023_v2-v1/NANOAOD` | 17,230,822 | CMSSW_13_0_13 | `/EGamma1/Run2023C-22Sep2023_v1-v1/NANOAOD` (67,582,665, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v3-v1/NANOAOD` (21,987,586, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v4-v1/NANOAOD` (160,049,621, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,530,273, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,230,822, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,939,165, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v4-v1/NANOAOD` (159,997,157, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-NanoAODv15-v1/NANOAOD` (67,530,273, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,230,822, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,987,586, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (159,997,174, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-PromptReco-v4/NANOAOD` (160,074,872, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma1/Run2023C-22Sep2023_v3-v1/NANOAOD` | 21,987,586 | CMSSW_13_0_13 | `/EGamma1/Run2023C-22Sep2023_v1-v1/NANOAOD` (67,582,665, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,230,822, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v4-v1/NANOAOD` (160,049,621, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,530,273, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,230,822, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,939,165, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v4-v1/NANOAOD` (159,997,157, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-NanoAODv15-v1/NANOAOD` (67,530,273, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,230,822, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,987,586, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (159,997,174, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-PromptReco-v4/NANOAOD` (160,074,872, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |
| `/EGamma1/Run2023C-22Sep2023_v4-v1/NANOAOD` | 160,049,621 | CMSSW_13_0_13 | `/EGamma1/Run2023C-22Sep2023_v1-v1/NANOAOD` (67,582,665, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v2-v1/NANOAOD` (17,230,822, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-22Sep2023_v3-v1/NANOAOD` (21,987,586, CMSSW_13_0_13)<br>`/EGamma1/Run2023C-24Jan2024_v1-v1/NANOAOD` (67,530,273, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v2-v1/NANOAOD` (17,230,822, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v3-v1/NANOAOD` (21,939,165, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-24Jan2024_v4-v1/NANOAOD` (159,997,157, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023C-NanoAODv15-v1/NANOAOD` (67,530,273, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v2-v1/NANOAOD` (17,230,822, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v3-v1/NANOAOD` (21,987,586, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-NanoAODv15_v4-v1/NANOAOD` (159,997,174, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023C-PromptReco-v4/NANOAOD` (160,074,872, CMSSW_13_0_6,CMSSW_13_0_7,CMSSW_13_0_7_TOTEM) |

**Run3Summer23BPix**

| current (in draft) | events | release | other NanoAOD processings of the same PD/era (events, release) |
|---|---|---|---|
| `/Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD` | 100,291,308 | CMSSW_13_0_13 | `/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD` (21,462,916, CMSSW_13_0_13)<br>`/Muon0/Run2023D-24Jan2024_v1-v1/NANOAOD` (100,078,297, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023D-24Jan2024_v2-v1/NANOAOD` (21,413,464, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023D-NanoAODv15-v1/NANOAOD` (100,211,533, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023D-NanoAODv15_v2-v1/NANOAOD` (21,462,916, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023D-PromptReco-v1/NANOAOD` (100,291,308, CMSSW_13_0_9)<br>`/Muon0/Run2023D-PromptReco-v2/NANOAOD` (21,462,916, CMSSW_13_0_10) |
| `/Muon0/Run2023D-22Sep2023_v2-v1/NANOAOD` | 21,462,916 | CMSSW_13_0_13 | `/Muon0/Run2023D-22Sep2023_v1-v1/NANOAOD` (100,291,308, CMSSW_13_0_13)<br>`/Muon0/Run2023D-24Jan2024_v1-v1/NANOAOD` (100,078,297, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023D-24Jan2024_v2-v1/NANOAOD` (21,413,464, CMSSW_13_3_1_patch1)<br>`/Muon0/Run2023D-NanoAODv15-v1/NANOAOD` (100,211,533, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023D-NanoAODv15_v2-v1/NANOAOD` (21,462,916, CMSSW_15_0_15_patch4)<br>`/Muon0/Run2023D-PromptReco-v1/NANOAOD` (100,291,308, CMSSW_13_0_9)<br>`/Muon0/Run2023D-PromptReco-v2/NANOAOD` (21,462,916, CMSSW_13_0_10) |
| `/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD` | 100,281,976 | CMSSW_13_0_13 | `/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD` (21,463,645, CMSSW_13_0_13)<br>`/Muon1/Run2023D-24Jan2024_v1-v1/NANOAOD` (100,281,976, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023D-24Jan2024_v2-v1/NANOAOD` (21,463,645, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023D-NanoAODv15-v1/NANOAOD` (100,281,976, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023D-NanoAODv15_v2-v1/NANOAOD` (21,463,645, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023D-PromptReco-v1/NANOAOD` (100,281,976, CMSSW_13_0_9)<br>`/Muon1/Run2023D-PromptReco-v2/NANOAOD` (21,463,645, CMSSW_13_0_10) |
| `/Muon1/Run2023D-22Sep2023_v2-v1/NANOAOD` | 21,463,645 | CMSSW_13_0_13 | `/Muon1/Run2023D-22Sep2023_v1-v1/NANOAOD` (100,281,976, CMSSW_13_0_13)<br>`/Muon1/Run2023D-24Jan2024_v1-v1/NANOAOD` (100,281,976, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023D-24Jan2024_v2-v1/NANOAOD` (21,463,645, CMSSW_13_3_1_patch1)<br>`/Muon1/Run2023D-NanoAODv15-v1/NANOAOD` (100,281,976, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023D-NanoAODv15_v2-v1/NANOAOD` (21,463,645, CMSSW_15_0_15_patch4)<br>`/Muon1/Run2023D-PromptReco-v1/NANOAOD` (100,281,976, CMSSW_13_0_9)<br>`/Muon1/Run2023D-PromptReco-v2/NANOAOD` (21,463,645, CMSSW_13_0_10) |
| `/EGamma0/Run2023D-22Sep2023_v1-v1/NANOAOD` | 105,892,646 | CMSSW_13_0_13 | `/EGamma0/Run2023D-22Sep2023_v2-v1/NANOAOD` (22,657,211, CMSSW_13_0_13)<br>`/EGamma0/Run2023D-24Jan2024_v1-v1/NANOAOD` (105,892,646, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023D-24Jan2024_v2-v1/NANOAOD` (22,657,211, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023D-NanoAODv15-v1/NANOAOD` (105,892,646, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023D-NanoAODv15_v2-v1/NANOAOD` (22,657,211, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023D-PromptReco-v1/NANOAOD` (105,897,671, CMSSW_13_0_9)<br>`/EGamma0/Run2023D-PromptReco-v2/NANOAOD` (22,657,211, CMSSW_13_0_10) |
| `/EGamma0/Run2023D-22Sep2023_v2-v1/NANOAOD` | 22,657,211 | CMSSW_13_0_13 | `/EGamma0/Run2023D-22Sep2023_v1-v1/NANOAOD` (105,892,646, CMSSW_13_0_13)<br>`/EGamma0/Run2023D-24Jan2024_v1-v1/NANOAOD` (105,892,646, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023D-24Jan2024_v2-v1/NANOAOD` (22,657,211, CMSSW_13_3_1_patch1)<br>`/EGamma0/Run2023D-NanoAODv15-v1/NANOAOD` (105,892,646, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023D-NanoAODv15_v2-v1/NANOAOD` (22,657,211, CMSSW_15_0_15_patch4)<br>`/EGamma0/Run2023D-PromptReco-v1/NANOAOD` (105,897,671, CMSSW_13_0_9)<br>`/EGamma0/Run2023D-PromptReco-v2/NANOAOD` (22,657,211, CMSSW_13_0_10) |
| `/EGamma1/Run2023D-22Sep2023_v1-v1/NANOAOD` | 105,824,276 | CMSSW_13_0_13 | `/EGamma1/Run2023D-22Sep2023_v2-v1/NANOAOD` (22,653,287, CMSSW_13_0_13)<br>`/EGamma1/Run2023D-24Jan2024_v1-v1/NANOAOD` (105,850,540, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023D-24Jan2024_v2-v1/NANOAOD` (22,653,287, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023D-NanoAODv15-v1/NANOAOD` (105,850,543, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023D-NanoAODv15_v2-v1/NANOAOD` (22,653,287, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023D-PromptReco-v1/NANOAOD` (105,875,994, CMSSW_13_0_9)<br>`/EGamma1/Run2023D-PromptReco-v2/NANOAOD` (22,653,287, CMSSW_13_0_10) |
| `/EGamma1/Run2023D-22Sep2023_v2-v1/NANOAOD` | 22,653,287 | CMSSW_13_0_13 | `/EGamma1/Run2023D-22Sep2023_v1-v1/NANOAOD` (105,824,276, CMSSW_13_0_13)<br>`/EGamma1/Run2023D-24Jan2024_v1-v1/NANOAOD` (105,850,540, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023D-24Jan2024_v2-v1/NANOAOD` (22,653,287, CMSSW_13_3_1_patch1)<br>`/EGamma1/Run2023D-NanoAODv15-v1/NANOAOD` (105,850,543, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023D-NanoAODv15_v2-v1/NANOAOD` (22,653,287, CMSSW_15_0_15_patch4)<br>`/EGamma1/Run2023D-PromptReco-v1/NANOAOD` (105,875,994, CMSSW_13_0_9)<br>`/EGamma1/Run2023D-PromptReco-v2/NANOAOD` (22,653,287, CMSSW_13_0_10) |

**RunIII2024Summer24**

| current (in draft) | events | release | other NanoAOD processings of the same PD/era (events, release) |
|---|---|---|---|
| `/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD` | 97,505,587 | CMSSW_15_0_2 | `/Muon0/Run2024C-2024CDEReprocessing-v1/NANOAOD` (97,646,805, CMSSW_14_0_19_patch2)<br>`/Muon0/Run2024C-PromptReco-v1/NANOAOD` (99,560,627, CMSSW_14_0_5_patch1) |
| `/Muon1/Run2024C-MINIv6NANOv15-v1/NANOAOD` | 97,531,998 | CMSSW_15_0_2 | `/Muon1/Run2024C-2024CDEReprocessing-v1/NANOAOD` (97,559,135, CMSSW_14_0_19_patch2)<br>`/Muon1/Run2024C-PromptReco-v1/NANOAOD` (99,546,845, CMSSW_14_0_5_patch1) |
| `/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD` | 157,351,226 | CMSSW_15_0_2 | `/EGamma0/Run2024C-2024CDEReprocessing-v2/NANOAOD` (157,881,330, CMSSW_14_0_19_patch2)<br>`/EGamma0/Run2024C-PromptReco-v1/NANOAOD` (162,344,928, CMSSW_14_0_5_patch1) |
| `/EGamma1/Run2024C-MINIv6NANOv15-v1/NANOAOD` | 157,860,731 | CMSSW_15_0_2 | `/EGamma1/Run2024C-2024CDEReprocessing-v2/NANOAOD` (157,877,112, CMSSW_14_0_19_patch2)<br>`/EGamma1/Run2024C-PromptReco-v1/NANOAOD` (162,330,796, CMSSW_14_0_5_patch1) |
| `/Muon0/Run2024D-MINIv6NANOv15-v1/NANOAOD` | 120,787,065 | CMSSW_15_0_2 | `/Muon0/Run2024D-2024CDEReprocessing-v1/NANOAOD` (120,787,065, CMSSW_14_0_19_patch2)<br>`/Muon0/Run2024D-PromptReco-v1/NANOAOD` (122,737,270, CMSSW_14_0_6_patch1) |
| `/Muon1/Run2024D-MINIv6NANOv15-v1/NANOAOD` | 120,467,371 | CMSSW_15_0_2 | `/Muon1/Run2024D-2024CDEReprocessing-v1/NANOAOD` (120,467,371, CMSSW_14_0_19_patch2)<br>`/Muon1/Run2024D-PromptReco-v1/NANOAOD` (122,726,921, CMSSW_14_0_6_patch1) |
| `/EGamma0/Run2024D-MINIv6NANOv15-v1/NANOAOD` | 156,558,757 | CMSSW_15_0_2 | `/EGamma0/Run2024D-2024CDEReprocessing-v2/NANOAOD` (156,585,950, CMSSW_14_0_19_patch2)<br>`/EGamma0/Run2024D-PromptReco-v1/NANOAOD` (159,682,409, CMSSW_14_0_6_patch1) |
| `/EGamma1/Run2024D-MINIv6NANOv15-v1/NANOAOD` | 156,275,778 | CMSSW_15_0_2 | `/EGamma1/Run2024D-2024CDEReprocessing-v2/NANOAOD` (156,577,168, CMSSW_14_0_19_patch2)<br>`/EGamma1/Run2024D-PromptReco-v1/NANOAOD` (159,668,543, CMSSW_14_0_6_patch1) |
| `/Muon0/Run2024E-MINIv6NANOv15-v1/NANOAOD` | 169,640,946 | CMSSW_15_0_2 | `/Muon0/Run2024E-2024CDEReprocessing-v1/NANOAOD` (172,724,756, CMSSW_14_0_19_patch2)<br>`/Muon0/Run2024E-PromptReco-v1/NANOAOD` (97,002,595, CMSSW_14_0_7)<br>`/Muon0/Run2024E-PromptReco-v2/NANOAOD` (79,363,754, CMSSW_14_0_7,CMSSW_14_0_8) |
| `/Muon1/Run2024E-MINIv6NANOv15-v1/NANOAOD` | 172,848,674 | CMSSW_15_0_2 | `/Muon1/Run2024E-2024CDEReprocessing-v1/NANOAOD` (173,108,330, CMSSW_14_0_19_patch2)<br>`/Muon1/Run2024E-PromptReco-v1/NANOAOD` (96,996,813, CMSSW_14_0_7,CMSSW_14_0_8)<br>`/Muon1/Run2024E-PromptReco-v2/NANOAOD` (79,359,841, CMSSW_14_0_7) |
| `/EGamma0/Run2024E-MINIv6NANOv15-v1/NANOAOD` | 249,417,634 | CMSSW_15_0_2 | `/EGamma0/Run2024E-2024CDEReprocessing-v2/NANOAOD` (249,642,406, CMSSW_14_0_19_patch2)<br>`/EGamma0/Run2024E-PromptReco-v1/NANOAOD` (139,307,881, CMSSW_14_0_7)<br>`/EGamma0/Run2024E-PromptReco-v2/NANOAOD` (117,664,197, CMSSW_14_0_7) |
| `/EGamma1/Run2024E-MINIv6NANOv15-v1/NANOAOD` | 249,489,829 | CMSSW_15_0_2 | `/EGamma1/Run2024E-2024CDEReprocessing-v2/NANOAOD` (249,644,780, CMSSW_14_0_19_patch2)<br>`/EGamma1/Run2024E-PromptReco-v1/NANOAOD` (139,308,688, CMSSW_14_0_7)<br>`/EGamma1/Run2024E-PromptReco-v2/NANOAOD` (117,660,063, CMSSW_14_0_7) |
| `/Muon0/Run2024F-MINIv6NANOv15-v1/NANOAOD` | 442,432,787 | CMSSW_15_0_2 | `/Muon0/Run2024F-PromptReco-v1/NANOAOD` (446,085,730, CMSSW_14_0_11,CMSSW_14_0_9) |
| `/Muon1/Run2024F-MINIv6NANOv15-v1/NANOAOD` | 442,360,423 | CMSSW_15_0_2 | `/Muon1/Run2024F-PromptReco-v1/NANOAOD` (446,051,284, CMSSW_14_0_11,CMSSW_14_0_9) |
| `/EGamma0/Run2024F-MINIv6NANOv15-v1/NANOAOD` | 638,196,622 | CMSSW_15_0_2 | `/EGamma0/Run2024F-PromptReco-v1/NANOAOD` (638,990,898, CMSSW_14_0_11,CMSSW_14_0_9) |
| `/EGamma1/Run2024F-MINIv6NANOv15-v1/NANOAOD` | 631,079,889 | CMSSW_15_0_2 | `/EGamma1/Run2024F-PromptReco-v1/NANOAOD` (638,980,684, CMSSW_14_0_11,CMSSW_14_0_9) |
| `/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD` | 642,028,803 | CMSSW_15_0_2 | `/Muon0/Run2024G-PromptReco-v1/NANOAOD` (647,360,838, CMSSW_14_0_12,CMSSW_14_0_13,CMSSW_14_0_14,CMSSW_14_0_15) |
| `/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD` | 641,959,643 | CMSSW_15_0_2 | `/Muon1/Run2024G-PromptReco-v1/NANOAOD` (647,340,693, CMSSW_14_0_12,CMSSW_14_0_13,CMSSW_14_0_14,CMSSW_14_0_15) |
| `/EGamma0/Run2024G-MINIv6NANOv15-v2/NANOAOD` | 903,520,258 | CMSSW_15_0_2 | `/EGamma0/Run2024G-PromptReco-v1/NANOAOD` (914,645,988, CMSSW_14_0_12,CMSSW_14_0_13,CMSSW_14_0_14,CMSSW_14_0_15) |
| `/EGamma1/Run2024G-MINIv6NANOv15-v2/NANOAOD` | 903,441,926 | CMSSW_15_0_2 | `/EGamma1/Run2024G-PromptReco-v1/NANOAOD` (913,963,326, CMSSW_14_0_12,CMSSW_14_0_13,CMSSW_14_0_14,CMSSW_14_0_15) |
| `/Muon0/Run2024H-MINIv6NANOv15-v1/NANOAOD` | 93,983,627 | CMSSW_15_0_2 | `/Muon0/Run2024H-PromptReco-v1/NANOAOD` (94,449,581, CMSSW_14_0_15,CMSSW_14_0_16) |
| `/Muon1/Run2024H-MINIv6NANOv15-v2/NANOAOD` | 93,981,102 | CMSSW_15_0_2 | `/Muon1/Run2024H-PromptReco-v1/NANOAOD` (94,447,297, CMSSW_14_0_15,CMSSW_14_0_16) |
| `/EGamma0/Run2024H-MINIv6NANOv15-v2/NANOAOD` | 134,680,448 | CMSSW_15_0_2 | `/EGamma0/Run2024H-PromptReco-v1/NANOAOD` (135,902,643, CMSSW_14_0_15,CMSSW_14_0_16) |
| `/EGamma1/Run2024H-MINIv6NANOv15-v1/NANOAOD` | 134,835,799 | CMSSW_15_0_2 | `/EGamma1/Run2024H-PromptReco-v1/NANOAOD` (135,899,057, CMSSW_14_0_15,CMSSW_14_0_16) |
| `/Muon0/Run2024I-MINIv6NANOv15-v1/NANOAOD` | 97,634,104 | CMSSW_15_0_2 | `/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` (105,194,627, CMSSW_15_0_2)<br>`/Muon0/Run2024I-PromptReco-v1/NANOAOD` (100,329,358, CMSSW_14_0_16)<br>`/Muon0/Run2024I-PromptReco-v2/NANOAOD` (106,485,742, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | 105,194,627 | CMSSW_15_0_2 | `/Muon0/Run2024I-MINIv6NANOv15-v1/NANOAOD` (97,634,104, CMSSW_15_0_2)<br>`/Muon0/Run2024I-PromptReco-v1/NANOAOD` (100,329,358, CMSSW_14_0_16)<br>`/Muon0/Run2024I-PromptReco-v2/NANOAOD` (106,485,742, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/Muon1/Run2024I-MINIv6NANOv15-v1/NANOAOD` | 97,630,010 | CMSSW_15_0_2 | `/Muon1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` (105,189,040, CMSSW_15_0_2)<br>`/Muon1/Run2024I-PromptReco-v1/NANOAOD` (100,325,747, CMSSW_14_0_16)<br>`/Muon1/Run2024I-PromptReco-v2/NANOAOD` (106,479,336, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/Muon1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | 105,189,040 | CMSSW_15_0_2 | `/Muon1/Run2024I-MINIv6NANOv15-v1/NANOAOD` (97,630,010, CMSSW_15_0_2)<br>`/Muon1/Run2024I-PromptReco-v1/NANOAOD` (100,325,747, CMSSW_14_0_16)<br>`/Muon1/Run2024I-PromptReco-v2/NANOAOD` (106,479,336, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/EGamma0/Run2024I-MINIv6NANOv15-v1/NANOAOD` | 132,904,290 | CMSSW_15_0_2 | `/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` (150,687,674, CMSSW_15_0_2)<br>`/EGamma0/Run2024I-PromptReco-v1/NANOAOD` (137,599,442, CMSSW_14_0_16)<br>`/EGamma0/Run2024I-PromptReco-v2/NANOAOD` (153,078,631, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/EGamma0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | 150,687,674 | CMSSW_15_0_2 | `/EGamma0/Run2024I-MINIv6NANOv15-v1/NANOAOD` (132,904,290, CMSSW_15_0_2)<br>`/EGamma0/Run2024I-PromptReco-v1/NANOAOD` (137,599,442, CMSSW_14_0_16)<br>`/EGamma0/Run2024I-PromptReco-v2/NANOAOD` (153,078,631, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/EGamma1/Run2024I-MINIv6NANOv15-v1/NANOAOD` | 132,903,874 | CMSSW_15_0_2 | `/EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` (150,687,112, CMSSW_15_0_2)<br>`/EGamma1/Run2024I-PromptReco-v1/NANOAOD` (137,596,331, CMSSW_14_0_16)<br>`/EGamma1/Run2024I-PromptReco-v2/NANOAOD` (153,075,941, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |
| `/EGamma1/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | 150,687,112 | CMSSW_15_0_2 | `/EGamma1/Run2024I-MINIv6NANOv15-v1/NANOAOD` (132,903,874, CMSSW_15_0_2)<br>`/EGamma1/Run2024I-PromptReco-v1/NANOAOD` (137,596,331, CMSSW_14_0_16)<br>`/EGamma1/Run2024I-PromptReco-v2/NANOAOD` (153,075,941, CMSSW_14_0_18,CMSSW_14_0_18_patch1) |

### 5.3 Golden-JSON coverage of the edge eras (DBS `runs` ∩ golden JSON)

| dataset | runs on DAS | golden runs contained | comment |
|---|---|---|---|
| `/SingleMuon/Run2022B-22Sep2023-v1/NANOAOD` | 355100–355769 (74) | 19 (355374–355769) | 2022B is in `Cert_Collisions2022_355100_362760_Golden.json` but in no config (also `EGamma Run2022B-22Sep2023-v2`); Summer22 MC conditions are C/D |
| `/SingleMuon/Run2022C-22Sep2023-v1/NANOAOD` | 355862–356386 (60) | 35 | early-2022C `SingleMuon` PD (before the `Muon` PD started at run 356426) – kept |
| `/Muon0/Run2023B-22Sep2023-v1/NANOAOD` | 366403–367079 (56) | 21 (366727–367079) | 2023B (17.4 M events, pre-BPix) is golden but not in `Run3Summer23_data.json`; `NanoAODv15-v1` also exists |
| `/Muon0/Run2024B-PromptReco-v1/NANOAOD` | 378981–379391 (62) | 16 (378985–379355) | no `MINIv6NANOv15` for 2024B; PdmV lumi table starts at C |
| `/Muon0/Run2024I-MINIv6NANOv15_v2-v1/NANOAOD` | 386694–386951 (36) | 23 | last golden run 386951 is covered |
| `/Muon0/Run2025B-PromptReco-v1/NANOAOD` | 391548–392112 (68) | 24 (391668–392112) | |
| `/Muon0/Run2025C-PromptReco-v1/NANOAOD` | 392174–393087 (77) | 52 (392293–393087) | |
| `/Muon0/Run2025C-PromptReco-v2/NANOAOD` | 393111–393516 (37) | 24 (393111–393461) | |
| `/Muon0/Run2025D-PromptReco-v1/NANOAOD` | 394393–395948 (187) | 124 (394637–395948) | |
| `/Muon0/Run2025E-PromptReco-v1/NANOAOD` | 395982–396422 (66) | 54 (395982–396422) | |
| `/Muon0/Run2025F-PromptReco-v1/NANOAOD` | 396629–397596 (143) | 108 (396733–397596) | |
| `/Muon0/Run2025F-PromptReco-v2/NANOAOD` | 397619–397853 (40) | 32 (397619–397817) | |
| `/Muon0/Run2025G-PromptReco-v1/NANOAOD` | 397954–398903 (126) | 95 (398011–398860) | |

### 5.4 2025

* **Datasets**: `Muon0`, `Muon1`, `EGamma0`, `EGamma1` × `Run2025B-PromptReco-v1`, `Run2025C-PromptReco-v1`, `Run2025C-PromptReco-v2`, `Run2025D-PromptReco-v1`, `Run2025E-PromptReco-v1`, `Run2025F-PromptReco-v1`, `Run2025F-PromptReco-v2`, `Run2025G-PromptReco-v1` (32 datasets, all VALID). Releases `CMSSW_15_0_5` (B) … `CMSSW_15_0_15_patch4` (G), GT `150X_dataRun3_Prompt_v1` → prompt **NanoAODv15**. No era H or later exists on DBS; no ReReco/re-nano of 2025. Excluded special nanos: `PromptBTVNano`, `PromptJMENano`, `PromptMuoPOGNano`, `PromptEGMNano`, `ZMu_PromptMUODPGNano`, `TrkRadDamage`, `MLPFHighPU`/`MLPFNominalPU`/`PFHighPU`/`PFNominalPU` (2025G).
* **Golden JSON**: directory `https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions25/` contains 25 `Cert_Collisions2025_391658_<last>_Golden.json` files (first 391953 on 2025-05-23, latest **`Cert_Collisions2025_391658_398903_Golden.json`**, dated 2026-07-30), the matching `Muon` JSONs, `2025_lowPU.json` / `2025_lowPU_updated.json` (3 low-pile-up runs 398682–398803, not in the golden JSON) and `DCSOnly_JSONS/`, `PileUp/`. The latest golden JSON has 513 runs (391668–398860); every one of them is contained in the B–G PromptReco datasets above.
* **Luminosity (PdmV, preliminary, no normtag)**: B 0.25, C 21.56, D 25.82, E 14.05, F 26.69, G 22.25 fb⁻¹ (110.62 fb⁻¹ total).
* **MC**: no `*Summer25*` campaign exists on DBS. `Run3Winter25NanoAOD` (357 datasets, `142X_mcRun3_2025_realistic_v4/v7/v9` and `MiniAODv6_NanoAODv15_150X_mcRun3_2025_realistic_v6` strings) is a pilot/validation campaign: of our processes only `DYto2L-4Jets_Bin-MLL-50` (VALID in `v9-v3`, `KeepSi_v7-v2`, `MiniAODv6_NanoAODv15_...v6-v2`), inclusive `TT` and the INVALID `TTto2L2Nu`/`TTToLNu2Q` exist; no W+jets, diboson, single top or signal. Following PdmV, **2025 reuses the RunIII2024Summer24 MC (the existing 2024 skims)**; no 2025 MC configs were written. (For completeness: `Run3Winter26NanoAODv15` and `RunIII2026LowPUSummer26NanoAODv15` campaigns already exist on DBS.)

## 6. MC per era – current-recommendation set

Columns: `src` = cross-section source code (legend at the end of this section); `in current configs` = `current` (same DAS path in a non-archived config, `xsec`/`genEventSumw` copied), `archived` (same path only in an archived family config) or `NEW` (`genEventSumw: null`).

### Run3Summer22


**Run3Summer22_mc_dy_lo_inc.json** – DY block: DY LO inclusive (+ MLL-10to50 companion) (3 datasets, 293,704,970 events, 0.3 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 6340.2 | S1 | 389 | 72,999,340 | 94 | current |
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v3/NANOAODSIM` | DYJets | 6340.2 | S1 | 509 | 73,925,935 | 96 | NEW |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 17380.0 | S2 | 1,473 | 146,779,695 | 127 | NEW |

**Run3Summer22_mc_dy_nlo_inc.json** – DY block: DY NLO inclusive (+ MLL-10to50 companion) (4 datasets, 333,044,796 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v5/NANOAODSIM` | DYJets | 6340.2 | S1 | 525 | 93,826,598 | 130 | current |
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 1,000 | 92,738,407 | 130 | NEW |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v4/NANOAODSIM` | DYJets | 20950.0 | S3 | 448 | 97,571,898 | 88 | NEW |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM` | DYJets | 20950.0 | S3 | 734 | 48,907,893 | 46 | NEW |

**Run3Summer22_mc_dy_lo_ht.json** – DY block: DY LO HT-binned (14 datasets, 128,607,280 events, 0.2 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 316.7 | S4 | 693 | 41,806,316 | 64 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 140.1 | S4 | 335 | 29,552,642 | 48 | NEW |
| `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 179.6 | S4 | 613 | 34,765,767 | 67 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 6.742 | S4 | 48 | 1,246,830 | 4 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.693 | S4 | 61 | 2,340,651 | 7 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.05047 | S4 | 79 | 2,385,017 | 8 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v4/NANOAODSIM` | DYJets | 0.00346 | S4 | 98 | 1,630,460 | 6 | current |
| `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 4.6 | S4 | 128 | 2,707,578 | 5 | current |
| `/DYto2L-4Jets_MLL-120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 2.205 | S4 | 59 | 1,953,844 | 3 | current |
| `/DYto2L-4Jets_MLL-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 3.352 | S4 | 33 | 2,113,763 | 4 | current |
| `/DYto2L-4Jets_MLL-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.1757 | S4 | 62 | 1,856,726 | 5 | current |
| `/DYto2L-4Jets_MLL-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.02089 | S4 | 38 | 2,049,333 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.001697 | S4 | 61 | 1,935,494 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.0001247 | S4 | 108 | 2,262,859 | 9 | current |

**Run3Summer22_mc_dy_nlo_mll.json** – DY block: DY NLO (powheg) MLL-binned (+ MLL-10to50) (20 datasets, 19,088,171 events, 0.0 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 2219 | S5 | 61 | 2,892,300 | 4 | archived |
| `/DYto2E_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 30 | 1,497,870 | 2 | archived |
| `/DYto2E_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 49 | 874,584 | 2 | archived |
| `/DYto2E_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 39 | 1,074,165 | 2 | archived |
| `/DYto2E_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 24 | 610,464 | 1 | archived |
| `/DYto2E_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 3 | 586,690 | 1 | archived |
| `/DYto2E_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 2 | 290,480 | 1 | archived |
| `/DYto2E_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 19 | 298,948 | 1 | archived |
| `/DYto2E_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 18 | 145,094 | 0 | archived |
| `/DYto2E_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 6744.0 | S6 | 25 | 1,477,950 | 1 | NEW |
| `/DYto2Mu_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 2219 | S5 | 38 | 2,820,937 | 3 | archived |
| `/DYto2Mu_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 50 | 1,453,748 | 2 | archived |
| `/DYto2Mu_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 37 | 853,443 | 1 | archived |
| `/DYto2Mu_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 46 | 874,240 | 2 | archived |
| `/DYto2Mu_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 25 | 579,560 | 1 | archived |
| `/DYto2Mu_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 20 | 590,523 | 1 | archived |
| `/DYto2Mu_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 10 | 299,278 | 1 | archived |
| `/DYto2Mu_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 24 | 289,200 | 1 | archived |
| `/DYto2Mu_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 18 | 145,002 | 0 | archived |
| `/DYto2Mu_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 6744.0 | S7 | 30 | 1,433,695 | 1 | NEW |

**Run3Summer22_mc_dy_nlo_njet.json** – DY block: DY NLO jet-binned (3 datasets, 273,449,832 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 5365 | S8 | 643 | 98,597,821 | 122 | archived |
| `/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 1015 | S8 | 546 | 97,156,305 | 147 | archived |
| `/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 382.8 | S8 | 721 | 77,695,706 | 141 | archived |

**Run3Summer22_mc_dy_nlo_ptll.json** – DY block: DY NLO pT(ll) nested thresholds (4 datasets, 152,286,172 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_PTLL-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 108.257 | S9 | 418 | 56,429,873 | 115 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 11.157 | S9 | 335 | 44,727,479 | 110 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.61601 | S9 | 424 | 41,629,822 | 114 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | DYJets | 0.08411 | S9 | 185 | 9,498,998 | 27 | NEW |

**Run3Summer22_mc_dy_lo_njet.json** – DY block: DY LO jet-binned (new family) (4 datasets, 42,099,351 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 973.1 | S10 | 137 | 15,216,450 | 22 | NEW |
| `/DYto2L-4Jets_MLL-50_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 312.4 | S11 | 126 | 15,269,289 | 25 | NEW |
| `/DYto2L-4Jets_MLL-50_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 93.93 | S12 | 155 | 8,842,997 | 16 | NEW |
| `/DYto2L-4Jets_MLL-50_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 45.43 | S13 | 78 | 2,770,615 | 6 | NEW |

**Run3Summer22_mc_dy_lo_ptll.json** – DY block: DY LO pT(ll)-binned (new family) (5 datasets, 71,683,554 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_PTLL-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 403.7 | S14 | 470 | 48,615,009 | 79 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 58.46 | S15 | 485 | 19,261,827 | 39 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 6.678 | S16 | 53 | 1,857,550 | 5 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.3833 | S17 | 50 | 983,966 | 3 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | DYJets | 0.06843 | S18 | 52 | 965,202 | 3 | NEW |

**Non-DY backgrounds (identical block in every Run3Summer22_mc_dy_*.json)** (27 datasets, 295,479,908 events, 0.6 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | tt_tW | 96.978 | S19 | 67 | 23,778,148 | 54 | current |
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM` | tt_tW | 96.978 | S19 | 319 | 24,047,488 | 56 | NEW |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 21 | 2,369,680 | 5 | current |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 56 | 2,499,302 | 5 | NEW |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 26 | 2,465,972 | 5 | current |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 73 | 2,461,995 | 5 | NEW |
| `/TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 38.6 | S21 | 48 | 2,998,239 | 5 | NEW |
| `/TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 23.34 | S21 | 81 | 1,497,050 | 3 | NEW |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 2.278 | S22 | 26 | 1,271,280 | 2 | current |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 1.43 | S23 | 27 | 785,520 | 1 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 404.537 | S24 | 188 | 88,510,970 | 205 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2/NANOAODSIM` | Nonprompt | 404.537 | S24 | 718 | 77,291,760 | 181 | NEW |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` | Nonprompt | 9141 | S25 | 38 | 9,871,448 | 11 | NEW |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` | Nonprompt | 2931 | S26 | 37 | 9,184,457 | 12 | NEW |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 864.6 | S27 | 78 | 7,804,738 | 12 | NEW |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Nonprompt | 417.8 | S28 | 64 | 1,463,885 | 3 | NEW |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | Other | 0.08646 | S29 | 7 | 399,000 | 1 | current |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v3/NANOAODSIM` | Other | 0.08646 | S29 | 21 | 4,000,000 | 13 | NEW |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-mg35x_130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 0.2502 | S30 | 26 | 2,056,905 | 6 | NEW |
| `/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 0.01769 | S31 | 31 | 298,552 | 1 | NEW |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 80.23 | S29 | 63 | 15,357,390 | 19 | current |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 29.1 | S29 | 46 | 7,527,043 | 10 | current |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 12.75 | S29 | 21 | 1,181,750 | 2 | current |
| `/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 0.2328 | S29 | 24 | 450,000 | 1 | current |
| `/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 0.1851 | S29 | 40 | 1,950,044 | 4 | current |
| `/WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 0.06206 | S29 | 19 | 1,987,058 | 4 | current |
| `/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | Other | 0.01591 | S29 | 23 | 1,970,234 | 4 | current |

**Other versions of the chosen primaries in the same campaign (Run3Summer22)** – chosen version listed first, others with their DAS status and event count:

| primary dataset | chosen | other versions |
|---|---|---|
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v2 [INVALID] |
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2022_realistic_v5_ext1-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5_ext1-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v2 [INVALID] |
| DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v2 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID] |
| DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5-v5 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v1 [INVALID] |
| DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5_ext1-v2 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v1 [INVALID] |
| DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5-v4 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v1 [INVALID] |
| DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5_ext1-v2 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5_ext1-v1 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v2 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v2 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID, 29,650,452 ev] |
| DYto2L-4Jets_MLL-50to120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v2 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v2 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v4 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID]; 130X_mcRun3_2022_realistic_v5-v3 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v1 [INVALID]; 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_0J_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_1J_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_2J_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_PTLL-100_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_1J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_2J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_3J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_4J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-40to100_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-100to200_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-200to400_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-400to600_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-600_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_v5-v3 | 130X_mcRun3_2022_realistic_v5-v2 [INVALID] |

### Run3Summer22EE


**Run3Summer22EE_mc_dy_lo_inc.json** – DY block: DY LO inclusive (+ MLL-10to50 companion) (3 datasets, 1,005,881,963 events, 1.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 6340.2 | S1 | 1,353 | 263,385,429 | 334 | current |
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v3/NANOAODSIM` | DYJets | 6340.2 | S1 | 1,275 | 253,535,615 | 321 | NEW |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | DYJets | 17380.0 | S2 | 3,057 | 488,960,919 | 412 | NEW |

**Run3Summer22EE_mc_dy_nlo_inc.json** – DY block: DY NLO inclusive (+ MLL-10to50 companion) (4 datasets, 1,223,120,811 events, 1.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5/NANOAODSIM` | DYJets | 6340.2 | S1 | 1,682 | 335,560,148 | 457 | current |
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 1,533 | 353,554,978 | 480 | NEW |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5/NANOAODSIM` | DYJets | 20950.0 | S3 | 1,573 | 355,091,803 | 314 | NEW |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v4/NANOAODSIM` | DYJets | 20950.0 | S3 | 1,113 | 178,913,882 | 159 | NEW |

**Run3Summer22EE_mc_dy_lo_ht.json** – DY block: DY LO HT-binned (14 datasets, 419,059,892 events, 0.8 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 316.7 | S4 | 1,371 | 125,345,985 | 187 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | DYJets | 140.1 | S4 | 982 | 94,595,270 | 151 | NEW |
| `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 179.6 | S4 | 1,276 | 124,757,918 | 235 | NEW |
| `/DYto2L-4Jets_MLL-50to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 6.742 | S4 | 41 | 3,522,511 | 10 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | DYJets | 0.693 | S4 | 23 | 1,256,042 | 4 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 0.05047 | S4 | 67 | 6,936,428 | 23 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5/NANOAODSIM` | DYJets | 0.00346 | S4 | 193 | 7,048,263 | 26 | current |
| `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | DYJets | 4.6 | S4 | 120 | 9,904,309 | 16 | NEW |
| `/DYto2L-4Jets_MLL-120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6/NANOAODSIM` | DYJets | 2.205 | S4 | 69 | 8,071,088 | 14 | current |
| `/DYto2L-4Jets_MLL-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 3.352 | S4 | 148 | 7,987,714 | 16 | current |
| `/DYto2L-4Jets_MLL-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6/NANOAODSIM` | DYJets | 0.1757 | S4 | 169 | 7,121,044 | 20 | current |
| `/DYto2L-4Jets_MLL-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | DYJets | 0.02089 | S4 | 274 | 7,490,387 | 24 | current |
| `/DYto2L-4Jets_MLL-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6/NANOAODSIM` | DYJets | 0.001697 | S4 | 236 | 7,286,454 | 26 | current |
| `/DYto2L-4Jets_MLL-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | DYJets | 0.0001247 | S4 | 247 | 7,736,479 | 29 | current |

**Run3Summer22EE_mc_dy_nlo_mll.json** – DY block: DY NLO (powheg) MLL-binned (+ MLL-10to50) (20 datasets, 66,529,653 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 2219 | S5 | 98 | 10,403,118 | 15 | archived |
| `/DYto2E_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 46 | 5,238,528 | 8 | archived |
| `/DYto2E_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 33 | 3,147,891 | 5 | archived |
| `/DYto2E_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 33 | 3,044,700 | 6 | archived |
| `/DYto2E_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 34 | 2,004,300 | 4 | archived |
| `/DYto2E_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 72 | 2,053,944 | 5 | archived |
| `/DYto2E_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 30 | 986,496 | 2 | archived |
| `/DYto2E_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 49 | 1,027,656 | 2 | archived |
| `/DYto2E_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 34 | 550,816 | 1 | archived |
| `/DYto2E_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 6744.0 | S6 | 58 | 5,333,106 | 5 | NEW |
| `/DYto2Mu_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 2219 | S5 | 106 | 10,150,330 | 12 | archived |
| `/DYto2Mu_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 91 | 5,050,710 | 7 | archived |
| `/DYto2Mu_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 75 | 2,972,383 | 4 | archived |
| `/DYto2Mu_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 49 | 2,930,748 | 5 | archived |
| `/DYto2Mu_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 34 | 2,093,529 | 4 | archived |
| `/DYto2Mu_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 59 | 2,008,146 | 4 | archived |
| `/DYto2Mu_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 38 | 1,003,792 | 2 | archived |
| `/DYto2Mu_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 41 | 994,900 | 2 | archived |
| `/DYto2Mu_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 19 | 503,888 | 1 | archived |
| `/DYto2Mu_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 6744.0 | S7 | 108 | 5,030,672 | 5 | NEW |

**Run3Summer22EE_mc_dy_nlo_njet.json** – DY block: DY NLO jet-binned (3 datasets, 966,220,922 events, 1.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 5365 | S8 | 2,083 | 345,496,261 | 422 | archived |
| `/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 1015 | S8 | 2,350 | 336,400,299 | 501 | archived |
| `/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 382.8 | S8 | 1,411 | 284,324,362 | 506 | archived |

**Run3Summer22EE_mc_dy_nlo_ptll.json** – DY block: DY NLO pT(ll) nested thresholds (4 datasets, 472,265,880 events, 1.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_PTLL-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 108.257 | S9 | 1,451 | 155,924,254 | 314 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 11.157 | S9 | 833 | 142,202,452 | 345 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.61601 | S9 | 806 | 140,605,102 | 379 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | DYJets | 0.08411 | S9 | 248 | 33,534,072 | 93 | NEW |

**Run3Summer22EE_mc_dy_lo_njet.json** – DY block: DY LO jet-binned (new family) (4 datasets, 138,927,656 events, 0.2 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 973.1 | S10 | 234 | 48,075,885 | 67 | NEW |
| `/DYto2L-4Jets_MLL-50_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 312.4 | S11 | 317 | 52,871,956 | 84 | NEW |
| `/DYto2L-4Jets_MLL-50_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 93.93 | S12 | 276 | 28,134,050 | 51 | NEW |
| `/DYto2L-4Jets_MLL-50_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 45.43 | S13 | 205 | 9,845,765 | 22 | NEW |

**Run3Summer22EE_mc_dy_lo_ptll.json** – DY block: DY LO pT(ll)-binned (new family) (5 datasets, 255,499,878 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_PTLL-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6/NANOAODSIM` | DYJets | 403.7 | S14 | 852 | 174,980,872 | 277 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 58.46 | S15 | 783 | 67,196,347 | 132 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 6.678 | S16 | 103 | 6,422,177 | 16 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 0.3833 | S17 | 78 | 3,665,389 | 10 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | DYJets | 0.06843 | S18 | 87 | 3,235,093 | 9 | NEW |

**Non-DY backgrounds (identical block in every Run3Summer22EE_mc_dy_*.json)** (27 datasets, 997,910,097 events, 2.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | tt_tW | 96.978 | S19 | 137 | 84,809,345 | 190 | current |
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM` | tt_tW | 96.978 | S19 | 707 | 84,870,780 | 192 | NEW |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 51 | 8,129,084 | 16 | current |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 233 | 8,466,410 | 18 | NEW |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 47 | 8,183,339 | 16 | current |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 196 | 8,555,460 | 18 | NEW |
| `/TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 38.6 | S21 | 132 | 10,249,351 | 18 | NEW |
| `/TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 23.34 | S21 | 105 | 5,239,919 | 9 | NEW |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 2.278 | S22 | 28 | 4,363,850 | 7 | current |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 1.43 | S23 | 23 | 2,762,668 | 5 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 404.537 | S24 | 524 | 267,927,960 | 612 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM` | Nonprompt | 404.537 | S24 | 1,786 | 275,339,453 | 633 | NEW |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 9141 | S25 | 133 | 40,329,821 | 46 | NEW |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 2931 | S26 | 105 | 33,044,336 | 44 | NEW |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 864.6 | S27 | 104 | 26,832,656 | 41 | NEW |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Nonprompt | 417.8 | S28 | 123 | 4,908,700 | 9 | NEW |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | Other | 0.08646 | S29 | 50 | 1,400,000 | 5 | current |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2/NANOAODSIM` | Other | 0.08646 | S29 | 97 | 12,586,478 | 40 | NEW |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-mg35x_130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 0.2502 | S30 | 171 | 7,212,758 | 21 | NEW |
| `/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | Other | 0.01769 | S31 | 27 | 599,291 | 2 | NEW |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 80.23 | S29 | 117 | 53,190,560 | 65 | current |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 29.1 | S29 | 90 | 26,840,468 | 33 | current |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 12.75 | S29 | 38 | 3,890,400 | 5 | current |
| `/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 0.2328 | S29 | 45 | 1,482,480 | 3 | current |
| `/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 0.1851 | S29 | 45 | 5,601,076 | 12 | current |
| `/WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 0.06206 | S29 | 40 | 5,290,014 | 12 | current |
| `/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | Other | 0.01591 | S29 | 45 | 5,803,440 | 11 | current |

**Other versions of the chosen primaries in the same campaign (Run3Summer22EE)** – chosen version listed first, others with their DAS status and event count:

| primary dataset | chosen | other versions |
|---|---|---|
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6_ext1-v2 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| TTLNu-EWK_amcatnlo-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6_ext1-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v2 [INVALID] |
| DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v5 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v1 [INVALID] |
| DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6_ext1-v2 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v1 [INVALID] |
| DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v5 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v3 [INVALID] |
| DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6_ext1-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6_ext1-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v2 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID, 98,847,366 ev] |
| DYto2L-4Jets_MLL-50to120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v2 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID, 142,936,007 ev] |
| DYto2L-4Jets_MLL-50to120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v5 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v4 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID, 10,411,115 ev] |
| DYto2L-4Jets_MLL-120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v6 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v4 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v5 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v6 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v4 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v5 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v6 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v4 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v5 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v4 | 130X_mcRun3_2022_realistic_postEE_v6-v1 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_0J_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_1J_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_2J_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-2Jets_MLL-50_PTLL-200_amcatnloFXFX-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_1J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_2J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_3J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_4J_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-40to100_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v6 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v4 [INVALID]; 130X_mcRun3_2022_realistic_postEE_v6-v5 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-100to200_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-200to400_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-400to600_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-600_madgraphMLM-pythia8 | 130X_mcRun3_2022_realistic_postEE_v6-v3 | 130X_mcRun3_2022_realistic_postEE_v6-v2 [INVALID] |

### Run3Summer23


**Run3Summer23_mc_dy_lo_inc.json** – DY block: DY LO inclusive (+ MLL-10to50 companion) (2 datasets, 393,793,612 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 457 | 127,376,178 | 165 | current |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 17380.0 | S2 | 726 | 266,417,434 | 231 | NEW |

**Run3Summer23_mc_dy_nlo_inc.json** – DY block: DY NLO inclusive (+ MLL-10to50 companion) (2 datasets, 400,279,035 events, 0.5 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 506 | 200,038,891 | 278 | current |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14_ext1-v4/NANOAODSIM` | DYJets | 20950.0 | S3 | 657 | 200,240,144 | 184 | NEW |

**Run3Summer23_mc_dy_lo_ht.json** – DY block: DY LO HT-binned (14 datasets, 256,202,300 events, 0.5 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 316.7 | S4 | 529 | 76,207,428 | 115 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | DYJets | 140.1 | S4 | 537 | 57,235,953 | 93 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 179.6 | S4 | 726 | 77,606,305 | 150 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 6.742 | S4 | 47 | 2,153,656 | 6 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.693 | S4 | 140 | 4,523,217 | 15 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.05047 | S4 | 58 | 4,062,569 | 14 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.00346 | S4 | 111 | 4,090,115 | 15 | current |
| `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 4.6 | S4 | 60 | 6,356,065 | 10 | current |
| `/DYto2L-4Jets_MLL-120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 2.205 | S4 | 32 | 3,992,010 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 3.352 | S4 | 43 | 3,643,744 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.1757 | S4 | 103 | 3,691,611 | 11 | current |
| `/DYto2L-4Jets_MLL-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.02089 | S4 | 75 | 4,925,219 | 16 | current |
| `/DYto2L-4Jets_MLL-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.001697 | S4 | 149 | 3,782,702 | 14 | current |
| `/DYto2L-4Jets_MLL-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.0001247 | S4 | 110 | 3,931,706 | 15 | current |

**Run3Summer23_mc_dy_nlo_mll.json** – DY block: DY NLO (powheg) MLL-binned (+ MLL-10to50) (20 datasets, 38,393,000 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 2219 | S5 | 63 | 5,832,000 | 9 | archived |
| `/DYto2E_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | DYJets | 21.65 | S5 | 29 | 2,991,000 | 5 | archived |
| `/DYto2E_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 27 | 1,734,000 | 3 | archived |
| `/DYto2E_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 24 | 1,788,000 | 4 | archived |
| `/DYto2E_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | DYJets | 0.01915 | S5 | 20 | 1,200,000 | 3 | archived |
| `/DYto2E_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 26 | 1,154,000 | 3 | archived |
| `/DYto2E_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 33 | 590,000 | 1 | archived |
| `/DYto2E_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 28 | 572,000 | 1 | archived |
| `/DYto2E_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 22 | 298,000 | 1 | archived |
| `/DYto2E_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 6744.0 | S6 | 46 | 2,968,000 | 3 | NEW |
| `/DYto2Mu_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 2219 | S5 | 44 | 5,904,000 | 7 | archived |
| `/DYto2Mu_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 70 | 2,994,000 | 4 | archived |
| `/DYto2Mu_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 23 | 1,776,000 | 3 | archived |
| `/DYto2Mu_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 52 | 1,794,000 | 3 | archived |
| `/DYto2Mu_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 26 | 1,179,000 | 2 | archived |
| `/DYto2Mu_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 46 | 1,191,000 | 2 | archived |
| `/DYto2Mu_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 17 | 585,000 | 1 | archived |
| `/DYto2Mu_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 17 | 558,000 | 1 | archived |
| `/DYto2Mu_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 9 | 285,000 | 1 | archived |
| `/DYto2Mu_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | DYJets | 6744.0 | S7 | 34 | 3,000,000 | 3 | NEW |

**Run3Summer23_mc_dy_nlo_njet.json** – DY block: DY NLO jet-binned (3 datasets, 530,969,087 events, 0.8 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 5365 | S8 | 663 | 201,186,460 | 251 | archived |
| `/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 1015 | S8 | 588 | 175,272,543 | 266 | NEW |
| `/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 382.8 | S8 | 476 | 154,510,084 | 281 | archived |

**Run3Summer23_mc_dy_nlo_ptll.json** – DY block: DY NLO pT(ll) nested thresholds (4 datasets, 271,258,299 events, 0.7 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_PTLL-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | DYJets | 108.257 | S9 | 815 | 95,212,367 | 198 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | DYJets | 11.157 | S9 | 398 | 80,677,175 | 201 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.61601 | S9 | 257 | 75,275,774 | 206 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | DYJets | 0.08411 | S9 | 158 | 20,092,983 | 56 | NEW |

**Run3Summer23_mc_dy_lo_njet.json** – DY block: DY LO jet-binned (new family) (4 datasets, 80,035,240 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 973.1 | S10 | 127 | 29,630,439 | 42 | NEW |
| `/DYto2L-4Jets_MLL-50_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 312.4 | S11 | 110 | 29,725,448 | 48 | NEW |
| `/DYto2L-4Jets_MLL-50_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 93.93 | S12 | 134 | 15,297,076 | 28 | NEW |
| `/DYto2L-4Jets_MLL-50_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4/NANOAODSIM` | DYJets | 45.43 | S13 | 123 | 5,382,277 | 12 | NEW |

**Run3Summer23_mc_dy_lo_ptll.json** – DY block: DY LO pT(ll)-binned (new family) (5 datasets, 175,199,125 events, 0.3 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_PTLL-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 403.7 | S14 | 779 | 121,445,289 | 197 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 58.46 | S15 | 525 | 45,141,048 | 92 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 6.678 | S16 | 72 | 4,236,951 | 11 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.3833 | S17 | 57 | 2,211,418 | 6 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | DYJets | 0.06843 | S18 | 42 | 2,164,419 | 6 | NEW |

**Non-DY backgrounds (identical block in every Run3Summer23_mc_dy_*.json)** (23 datasets, 363,908,328 events, 0.7 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | tt_tW | 96.978 | S19 | 198 | 48,203,000 | 111 | current |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 34 | 4,985,000 | 10 | current |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM` | tt_tW | 4.61475 | S20 | 42 | 4,931,000 | 10 | current |
| `/TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v1/NANOAODSIM` | Nonprompt | 38.6 | S21 | 55 | 5,935,200 | 11 | NEW |
| `/TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v1/NANOAODSIM` | Nonprompt | 23.34 | S21 | 26 | 2,958,200 | 5 | NEW |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | Nonprompt | 2.278 | S22 | 49 | 2,596,000 | 4 | current |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | Nonprompt | 1.43 | S23 | 41 | 1,600,000 | 3 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Nonprompt | 404.537 | S24 | 798 | 152,797,000 | 359 | current |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | Nonprompt | 9141 | S25 | 168 | 23,088,467 | 27 | NEW |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | Nonprompt | 2931 | S26 | 137 | 20,264,539 | 28 | NEW |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | Nonprompt | 864.6 | S27 | 129 | 15,887,453 | 25 | NEW |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` | Nonprompt | 417.8 | S28 | 106 | 2,961,904 | 6 | NEW |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | Other | 0.08646 | S29 | 5 | 800,000 | 3 | current |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15_ext1-v2/NANOAODSIM` | Other | 0.08646 | S29 | 55 | 7,850,000 | 26 | NEW |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-mg35x_130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | Other | 0.2502 | S30 | 34 | 3,935,565 | 12 | NEW |
| `/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | Other | 0.01769 | S31 | 32 | 597,000 | 2 | NEW |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 80.23 | S29 | 125 | 33,504,000 | 42 | current |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 29.1 | S29 | 60 | 16,770,000 | 21 | current |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 12.75 | S29 | 11 | 2,517,000 | 3 | current |
| `/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 0.2328 | S29 | 8 | 936,000 | 2 | current |
| `/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 0.1851 | S29 | 35 | 3,600,000 | 8 | current |
| `/WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 0.06206 | S29 | 36 | 3,591,000 | 8 | current |
| `/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | Other | 0.01591 | S29 | 39 | 3,600,000 | 7 | current |

**Other versions of the chosen primaries in the same campaign (Run3Summer23)** – chosen version listed first, others with their DAS status and event count:

| primary dataset | chosen | other versions |
|---|---|---|
| TbarWplusto2L2Nu_powheg-pythia8 | 130X_mcRun3_2023_realistic_v15-v4 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2023_realistic_v15_ext1-v2 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v2 | 130X_mcRun3_2023_realistic_v14-v1 [INVALID] |
| DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_v14-v2 | 130X_mcRun3_2023_realistic_v14-v1 [INVALID] |
| DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_v14_ext1-v4 | 130X_mcRun3_2023_realistic_v14_ext1-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v3 | 130X_mcRun3_2023_realistic_v14-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v14-v3 [PRODUCTION]; 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2E_MLL-120to200_powheg-pythia8 | 130X_mcRun3_2023_realistic_v14-v3 | 130X_mcRun3_2023_realistic_v14-v2 [INVALID] |
| DYto2E_MLL-800to1500_powheg-pythia8 | 130X_mcRun3_2023_realistic_v14-v3 | 130X_mcRun3_2023_realistic_v14-v2 [INVALID] |
| DYto2Mu_MLL-4000to6000_powheg-pythia8 | 130X_mcRun3_2023_realistic_v14-v2 | 130X_mcRun3_2023_realistic_v14-v3 [PRODUCTION] |
| DYto2L-2Jets_MLL-50_0J_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_1J_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID, 192,409,561 ev] |
| DYto2L-2Jets_MLL-50_2J_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_PTLL-400_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_1J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_2J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_3J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_4J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v14-v4 | 130X_mcRun3_2023_realistic_v14-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-40to100_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-100to200_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-200to400_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-400to600_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-600_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_v15-v3 | 130X_mcRun3_2023_realistic_v15-v2 [INVALID] |

### Run3Summer23BPix


**Run3Summer23BPix_mc_dy_lo_inc.json** – DY block: DY LO inclusive (+ MLL-10to50 companion) (2 datasets, 211,679,573 events, 0.2 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 6340.2 | S1 | 301 | 70,167,286 | 91 | current |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 17380.0 | S2 | 495 | 141,512,287 | 123 | NEW |

**Run3Summer23BPix_mc_dy_nlo_inc.json** – DY block: DY NLO inclusive (+ MLL-10to50 companion) (2 datasets, 195,287,120 events, 0.2 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 6340.2 | S1 | 312 | 94,876,537 | 132 | current |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2_ext1-v4/NANOAODSIM` | DYJets | 20950.0 | S3 | 342 | 100,410,583 | 92 | NEW |

**Run3Summer23BPix_mc_dy_lo_ht.json** – DY block: DY LO HT-binned (14 datasets, 139,211,476 events, 0.3 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5/NANOAODSIM` | DYJets | 316.7 | S4 | 221 | 37,755,405 | 57 | NEW |
| `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5/NANOAODSIM` | DYJets | 140.1 | S4 | 202 | 34,689,209 | 56 | NEW |
| `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 179.6 | S4 | 357 | 45,276,159 | 87 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 6.742 | S4 | 9 | 1,162,372 | 3 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 0.693 | S4 | 54 | 1,832,189 | 6 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v5/NANOAODSIM` | DYJets | 0.05047 | S4 | 32 | 1,938,272 | 7 | current |
| `/DYto2L-4Jets_MLL-50to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 0.00346 | S4 | 77 | 1,882,022 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 4.6 | S4 | 25 | 2,842,602 | 5 | current |
| `/DYto2L-4Jets_MLL-120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 2.205 | S4 | 34 | 1,893,044 | 3 | current |
| `/DYto2L-4Jets_MLL-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 3.352 | S4 | 33 | 1,863,139 | 4 | current |
| `/DYto2L-4Jets_MLL-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v4/NANOAODSIM` | DYJets | 0.1757 | S4 | 58 | 1,905,893 | 6 | current |
| `/DYto2L-4Jets_MLL-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 0.02089 | S4 | 64 | 2,177,931 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 0.001697 | S4 | 60 | 1,877,696 | 7 | current |
| `/DYto2L-4Jets_MLL-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v4/NANOAODSIM` | DYJets | 0.0001247 | S4 | 57 | 2,115,543 | 8 | current |

**Run3Summer23BPix_mc_dy_nlo_mll.json** – DY block: DY NLO (powheg) MLL-binned (+ MLL-10to50) (20 datasets, 19,082,000 events, 0.0 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 2219 | S5 | 49 | 2,910,000 | 4 | archived |
| `/DYto2E_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 21.65 | S5 | 23 | 1,476,000 | 2 | archived |
| `/DYto2E_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 3.058 | S5 | 23 | 867,000 | 2 | archived |
| `/DYto2E_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 0.2691 | S5 | 44 | 894,000 | 2 | archived |
| `/DYto2E_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 0.01915 | S5 | 20 | 591,000 | 1 | archived |
| `/DYto2E_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 0.001111 | S5 | 19 | 580,000 | 1 | archived |
| `/DYto2E_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 17 | 300,000 | 1 | archived |
| `/DYto2E_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 20 | 300,000 | 1 | archived |
| `/DYto2E_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 9 | 150,000 | 0 | archived |
| `/DYto2E_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 6744.0 | S6 | 20 | 1,476,000 | 1 | NEW |
| `/DYto2Mu_MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 2219 | S5 | 30 | 2,872,000 | 4 | archived |
| `/DYto2Mu_MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 21.65 | S5 | 25 | 1,496,000 | 2 | archived |
| `/DYto2Mu_MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 3.058 | S5 | 20 | 891,000 | 1 | archived |
| `/DYto2Mu_MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 0.2691 | S5 | 27 | 892,000 | 2 | archived |
| `/DYto2Mu_MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 0.01915 | S5 | 19 | 584,000 | 1 | archived |
| `/DYto2Mu_MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 0.001111 | S5 | 11 | 600,000 | 1 | archived |
| `/DYto2Mu_MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 11 | 291,000 | 1 | archived |
| `/DYto2Mu_MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 13 | 294,000 | 1 | archived |
| `/DYto2Mu_MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 9 | 150,000 | 0 | archived |
| `/DYto2Mu_MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | DYJets | 6744.0 | S7 | 30 | 1,468,000 | 1 | NEW |

**Run3Summer23BPix_mc_dy_nlo_njet.json** – DY block: DY NLO jet-binned (3 datasets, 258,874,539 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_0J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 5365 | S8 | 316 | 99,748,250 | 124 | archived |
| `/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 1015 | S8 | 312 | 86,282,834 | 131 | NEW |
| `/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5/NANOAODSIM` | DYJets | 382.8 | S8 | 246 | 72,843,455 | 133 | NEW |

**Run3Summer23BPix_mc_dy_nlo_ptll.json** – DY block: DY NLO pT(ll) nested thresholds (4 datasets, 124,801,372 events, 0.3 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_PTLL-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | DYJets | 108.257 | S9 | 203 | 41,940,852 | 87 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | DYJets | 11.157 | S9 | 286 | 32,405,482 | 81 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | DYJets | 0.61601 | S9 | 244 | 40,272,138 | 111 | NEW |
| `/DYto2L-2Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | DYJets | 0.08411 | S9 | 80 | 10,182,900 | 29 | NEW |

**Run3Summer23BPix_mc_dy_lo_njet.json** – DY block: DY LO jet-binned (new family) (4 datasets, 41,313,936 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 973.1 | S10 | 67 | 15,289,548 | 22 | NEW |
| `/DYto2L-4Jets_MLL-50_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 312.4 | S11 | 76 | 14,740,579 | 24 | NEW |
| `/DYto2L-4Jets_MLL-50_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 93.93 | S12 | 71 | 8,393,375 | 16 | NEW |
| `/DYto2L-4Jets_MLL-50_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4/NANOAODSIM` | DYJets | 45.43 | S13 | 74 | 2,890,434 | 7 | NEW |

**Run3Summer23BPix_mc_dy_lo_ptll.json** – DY block: DY LO pT(ll)-binned (new family) (5 datasets, 72,425,913 events, 0.1 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_PTLL-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 403.7 | S14 | 345 | 46,919,628 | 76 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-100to200_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 58.46 | S15 | 271 | 21,393,032 | 44 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-200to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 6.678 | S16 | 31 | 2,135,466 | 5 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-400to600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 0.3833 | S17 | 54 | 978,850 | 3 | NEW |
| `/DYto2L-4Jets_MLL-50_PTLL-600_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | DYJets | 0.06843 | S18 | 31 | 998,937 | 3 | NEW |

**Non-DY backgrounds (identical block in every Run3Summer23BPix_mc_dy_*.json)** (23 datasets, 194,106,124 events, 0.4 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | tt_tW | 96.978 | S19 | 150 | 24,649,000 | 57 | current |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | tt_tW | 4.61475 | S20 | 30 | 2,479,000 | 5 | current |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 50 | 2,488,000 | 5 | current |
| `/TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v1/NANOAODSIM` | Nonprompt | 38.6 | S21 | 100 | 10,319,600 | 18 | NEW |
| `/TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v1/NANOAODSIM` | Nonprompt | 23.34 | S21 | 29 | 1,486,700 | 3 | NEW |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | Nonprompt | 2.278 | S22 | 25 | 1,276,000 | 2 | current |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | Nonprompt | 1.43 | S23 | 21 | 788,000 | 1 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | Nonprompt | 404.537 | S24 | 492 | 82,058,000 | 193 | current |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | Nonprompt | 9141 | S25 | 80 | 11,369,658 | 14 | NEW |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | Nonprompt | 2931 | S26 | 111 | 9,345,696 | 13 | NEW |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | Nonprompt | 864.6 | S27 | 76 | 7,718,351 | 12 | NEW |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | Nonprompt | 417.8 | S28 | 29 | 1,441,228 | 3 | NEW |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | Other | 0.08646 | S29 | 7 | 400,000 | 1 | current |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6_ext1-v2/NANOAODSIM` | Other | 0.08646 | S29 | 43 | 3,856,000 | 13 | NEW |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-mg35x_130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | Other | 0.2502 | S30 | 10 | 1,981,391 | 6 | NEW |
| `/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | Other | 0.01769 | S31 | 21 | 294,000 | 1 | NEW |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | Other | 80.23 | S29 | 54 | 16,734,000 | 21 | current |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | Other | 29.1 | S29 | 33 | 8,379,000 | 11 | current |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | Other | 12.75 | S29 | 5 | 1,254,000 | 2 | current |
| `/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | Other | 0.2328 | S29 | 2 | 466,500 | 1 | current |
| `/WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | Other | 0.1851 | S29 | 28 | 1,746,000 | 4 | current |
| `/WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | Other | 0.06206 | S29 | 9 | 1,788,000 | 4 | current |
| `/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | Other | 0.01591 | S29 | 8 | 1,788,000 | 4 | current |

**Other versions of the chosen primaries in the same campaign (Run3Summer23BPix)** – chosen version listed first, others with their DAS status and event count:

| primary dataset | chosen | other versions |
|---|---|---|
| TbarWplusto2L2Nu_powheg-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 | 130X_mcRun3_2023_realistic_postBPix_v5-v2 [INVALID] |
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| TTLL_MLL-50_amcatnlo-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6_ext1-v2 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| WWZ_4F_amcatnlo-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 | 130X_mcRun3_2023_realistic_postBPix_v2-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v1 [INVALID]; 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2_ext1-v4 | 130X_mcRun3_2023_realistic_postBPix_v2_ext1-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v5 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID, 36,265,983 ev] |
| DYto2L-4Jets_MLL-50to120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v5 | 130X_mcRun3_2023_realistic_postBPix_v2-v2 [INVALID, 29,460,714 ev] |
| DYto2L-4Jets_MLL-50to120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 | 130X_mcRun3_2023_realistic_postBPix_v2-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v5 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID]; 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID]; 130X_mcRun3_2023_realistic_postBPix_v6-v3 [INVALID] |
| DYto2L-4Jets_MLL-50to120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-40to70_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-70to100_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-100to400_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-400to800_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v4 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID]; 130X_mcRun3_2023_realistic_postBPix_v6-v3 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-800to1500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-1500to2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-120_HT-2500_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v4 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID]; 130X_mcRun3_2023_realistic_postBPix_v6-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_0J_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-2Jets_MLL-50_1J_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID, 91,293,455 ev] |
| DYto2L-2Jets_MLL-50_2J_amcatnloFXFX-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v5 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID, 68,804,529 ev]; 130X_mcRun3_2023_realistic_postBPix_v2-v4 [INVALID] |
| DYto2L-4Jets_MLL-50_1J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_2J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_3J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_4J_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v2-v4 | 130X_mcRun3_2023_realistic_postBPix_v2-v3 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-40to100_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-100to200_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-200to400_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-400to600_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |
| DYto2L-4Jets_MLL-50_PTLL-600_madgraphMLM-pythia8 | 130X_mcRun3_2023_realistic_postBPix_v6-v3 | 130X_mcRun3_2023_realistic_postBPix_v6-v2 [INVALID] |

### RunIII2024Summer24


**RunIII2024Summer24_mc_dy_lo_inc.json** – DY block: DY LO inclusive (+ MLL-10to50 companion) (4 datasets, 1,895,054,188 events, 2.8 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 2113.4 | S1 | 2,821 | 494,826,377 | 976 | current |
| `/DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 2113.4 | S1 | 3,194 | 530,955,854 | 866 | current |
| `/DYto2E-4Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 5793.333333333333 | S2 | 2,425 | 403,551,955 | 473 | NEW |
| `/DYto2Mu-4Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 5793.333333333333 | S2 | 3,080 | 465,720,002 | 531 | NEW |

**RunIII2024Summer24_mc_dy_nlo_inc.json** – DY block: DY NLO inclusive (+ MLL-10to50 companion) (4 datasets, 1,260,531,223 events, 2.2 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 2113.4 | S1 | 2,990 | 486,448,139 | 1017 | current |
| `/DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6/NANOAODSIM` | DYJets | 2113.4 | S1 | 2,975 | 490,076,405 | 861 | current |
| `/DYto2E-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 6983.333333333333 | S3 | 694 | 139,117,188 | 170 | NEW |
| `/DYto2Mu-2Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 6983.333333333333 | S3 | 1,015 | 144,889,491 | 174 | NEW |

**RunIII2024Summer24_mc_dy_lo_ht.json** – DY block: DY LO HT-binned (14 datasets, 1,169,701,714 events, 2.8 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_Bin-HT-40to70-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 316.7 | S4 | 2,547 | 410,493,396 | 846 | NEW |
| `/DYto2L-4Jets_Bin-HT-70to100-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 140.1 | S4 | 1,733 | 268,137,316 | 595 | NEW |
| `/DYto2L-4Jets_Bin-HT-100to400-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 179.6 | S4 | 2,526 | 396,793,361 | 1043 | NEW |
| `/DYto2L-4Jets_Bin-HT-400to800-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 6.742 | S4 | 249 | 16,223,572 | 67 | NEW |
| `/DYto2L-4Jets_Bin-HT-800to1500-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v5/NANOAODSIM` | DYJets | 0.693 | S4 | 43 | 2,659,044 | 13 | NEW |
| `/DYto2L-4Jets_Bin-HT-1500to2500-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 0.05047 | S4 | 38 | 1,889,567 | 10 | NEW |
| `/DYto2L-4Jets_Bin-HT-2500-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 0.00346 | S4 | 38 | 984,539 | 6 | NEW |
| `/DYto2L-4Jets_Bin-HT-40to70-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 4.6 | S4 | 257 | 30,009,269 | 66 | NEW |
| `/DYto2L-4Jets_Bin-HT-70to100-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 2.205 | S4 | 219 | 18,785,855 | 45 | NEW |
| `/DYto2L-4Jets_Bin-HT-100to400-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 3.352 | S4 | 216 | 18,974,607 | 54 | NEW |
| `/DYto2L-4Jets_Bin-HT-400to800-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 0.1757 | S4 | 41 | 1,775,774 | 8 | NEW |
| `/DYto2L-4Jets_Bin-HT-800to1500-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 0.02089 | S4 | 33 | 949,421 | 5 | NEW |
| `/DYto2L-4Jets_Bin-HT-1500to2500-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 0.001697 | S4 | 33 | 999,402 | 6 | NEW |
| `/DYto2L-4Jets_Bin-HT-2500-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v4/NANOAODSIM` | DYJets | 0.0001247 | S4 | 35 | 1,026,591 | 6 | NEW |

**RunIII2024Summer24_mc_dy_nlo_mll.json** – DY block: DY NLO (powheg) MLL-binned (+ MLL-10to50) (20 datasets, 1,782,337,868 events, 2.9 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E_Bin-MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 2219 | S5 | 657 | 478,240,880 | 951 | archived |
| `/DYto2E_Bin-MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 110 | 48,821,981 | 105 | archived |
| `/DYto2E_Bin-MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 33 | 6,823,360 | 17 | archived |
| `/DYto2E_Bin-MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 14 | 977,525 | 3 | archived |
| `/DYto2E_Bin-MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 25 | 992,750 | 3 | archived |
| `/DYto2E_Bin-MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 27 | 998,592 | 3 | archived |
| `/DYto2E_Bin-MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 6 | 996,630 | 3 | archived |
| `/DYto2E_Bin-MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 7 | 984,688 | 3 | archived |
| `/DYto2E_Bin-MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 21 | 998,101 | 4 | archived |
| `/DYto2E_Bin-MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 6744.0 | S6 | 395 | 348,121,585 | 436 | NEW |
| `/DYto2Mu_Bin-MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 2219 | S5 | 640 | 492,270,708 | 809 | archived |
| `/DYto2Mu_Bin-MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 21.65 | S5 | 110 | 48,037,000 | 84 | archived |
| `/DYto2Mu_Bin-MLL-200to400_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 3.058 | S5 | 49 | 6,588,316 | 13 | archived |
| `/DYto2Mu_Bin-MLL-400to800_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.2691 | S5 | 20 | 977,500 | 2 | archived |
| `/DYto2Mu_Bin-MLL-800to1500_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.01915 | S5 | 3 | 979,723 | 3 | archived |
| `/DYto2Mu_Bin-MLL-1500to2500_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.001111 | S5 | 26 | 962,340 | 3 | archived |
| `/DYto2Mu_Bin-MLL-2500to4000_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 5.949e-05 | S5 | 19 | 934,352 | 3 | archived |
| `/DYto2Mu_Bin-MLL-4000to6000_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 1.558e-06 | S5 | 9 | 999,261 | 3 | archived |
| `/DYto2Mu_Bin-MLL-6000_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 3.519e-08 | S5 | 29 | 998,508 | 3 | archived |
| `/DYto2Mu_Bin-MLL-10to50_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 6744.0 | S7 | 405 | 341,634,068 | 413 | NEW |

**RunIII2024Summer24_mc_dy_nlo_njet.json** – DY block: DY NLO jet-binned (6 datasets, 2,256,404,917 events, 4.6 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2E-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 5382 | S32 | 548 | 453,230,267 | 844 | archived |
| `/DYto2E-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 972.3 | S32 | 2,108 | 338,867,940 | 773 | archived |
| `/DYto2E-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 372.4 | S32 | 317 | 307,042,054 | 833 | archived |
| `/DYto2Mu-2Jets_Bin-0J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 5382 | S32 | 534 | 466,773,211 | 719 | archived |
| `/DYto2Mu-2Jets_Bin-1J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v1/NANOAODSIM` | DYJets | 975.5 | S32 | 327 | 404,580,799 | 776 | archived |
| `/DYto2Mu-2Jets_Bin-2J-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 374 | S32 | 1,739 | 285,910,646 | 676 | archived |

**RunIII2024Summer24_mc_dy_nlo_ptll.json** – DY block: DY NLO pT(ll) nested thresholds (4 datasets, 766,372,865 events, 2.8 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_Bin-MLL-50-PTLL-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 108.257 | S9 | 1,443 | 186,947,219 | 540 | NEW |
| `/DYto2L-2Jets_Bin-MLL-50-PTLL-200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v1/NANOAODSIM` | DYJets | 11.157 | S9 | 343 | 289,689,475 | 1062 | NEW |
| `/DYto2L-2Jets_Bin-MLL-50-PTLL-400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | DYJets | 0.61601 | S9 | 276 | 194,875,038 | 813 | NEW |
| `/DYto2L-2Jets_Bin-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | DYJets | 0.08411 | S9 | 657 | 94,861,133 | 407 | NEW |

**Non-DY backgrounds (identical block in every RunIII2024Summer24_mc_dy_*.json)** (22 datasets, 2,193,511,961 events, 5.5 TB)

| DAS path | physics_group | xsec [pb] | src | files | events | GB | in current configs |
|---|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` | tt_tW | 96.978 | S19 | 780 | 470,123,263 | 1492 | current |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 176 | 14,998,000 | 43 | current |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | tt_tW | 4.61475 | S20 | 90 | 15,000,000 | 43 | current |
| `/TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 38.6 | S21 | 391 | 44,258,039 | 106 | current |
| `/TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 23.34 | S21 | 249 | 22,102,160 | 53 | current |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 2.278 | S22 | 94 | 12,999,000 | 30 | current |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 1.43 | S23 | 68 | 7,980,999 | 18 | current |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 404.537 | S24 | 790 | 484,475,057 | 1551 | current |
| `/WtoLNu-4Jets_Bin-1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 9141 | S25 | 2,491 | 429,800,607 | 672 | current |
| `/WtoLNu-4Jets_Bin-2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 2931 | S26 | 1,632 | 286,526,689 | 522 | current |
| `/WtoLNu-4Jets_Bin-3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 864.6 | S27 | 915 | 139,773,986 | 294 | current |
| `/WtoLNu-4Jets_Bin-4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Nonprompt | 417.8 | S28 | 644 | 85,950,957 | 224 | current |
| `/TTLL_Bin-MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.08646 | S29 | 47 | 3,952,000 | 18 | current |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-mg35x_150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.2502 | S30 | 186 | 20,362,371 | 86 | NEW |
| `/TTLNu-EWK_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.01769 | S31 | 38 | 2,999,000 | 13 | current |
| `/WW_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 80.23 | S29 | 518 | 63,986,968 | 115 | current |
| `/WZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 29.1 | S29 | 307 | 30,547,825 | 55 | current |
| `/ZZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 12.75 | S29 | 76 | 4,800,000 | 9 | current |
| `/WWW-4F_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.2328 | S29 | 49 | 4,299,284 | 13 | current |
| `/WWZ-4F_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.1851 | S29 | 171 | 16,184,379 | 50 | current |
| `/WZZ-5F_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.06206 | S29 | 127 | 16,199,286 | 50 | current |
| `/ZZZ-5F_TuneCP5_13p6TeV_amcatnlo-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | Other | 0.01591 | S29 | 201 | 16,192,091 | 44 | current |

**Other versions of the chosen primaries in the same campaign (RunIII2024Summer24)** – chosen version listed first, others with their DAS status and event count:

| primary dataset | chosen | other versions |
|---|---|---|
| TTto2L2Nu_powheg-pythia8 | 150X_mcRun3_2024_realistic_v2-v3 | 150X_mcRun3_2024_realistic_v1-v2 [INVALID]; 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| TTtoLNu2Q_powheg-pythia8 | 150X_mcRun3_2024_realistic_v2-v2 | 150X_mcRun3_2024_realistic_v2_ext1-v2 [PRODUCTION] |
| DYto2E-2Jets_Bin-MLL-50_amcatnloFXFX-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v1-v2 [INVALID] |
| DYto2Mu-2Jets_Bin-MLL-50_amcatnloFXFX-pythia8 | 150X_mcRun3_2024_realistic_v2-v6 | 150X_mcRun3_2024_realistic_v1-v4 [INVALID]; 150X_mcRun3_2024_realistic_v1-v5 [INVALID] |
| DYto2L-4Jets_Bin-HT-800to1500-MLL-50to120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v5 | 150X_mcRun3_2024_realistic_v2-v3 [INVALID] |
| DYto2L-4Jets_Bin-HT-1500to2500-MLL-50to120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2L-4Jets_Bin-HT-2500-MLL-50to120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2L-4Jets_Bin-HT-400to800-MLL-120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2L-4Jets_Bin-HT-800to1500-MLL-120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2L-4Jets_Bin-HT-1500to2500-MLL-120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2L-4Jets_Bin-HT-2500-MLL-120_madgraphMLM-pythia8 | 150X_mcRun3_2024_realistic_v2-v4 | 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2E-2Jets_Bin-2J-MLL-50_amcatnloFXFX-pythia8 | 150X_mcRun3_2024_realistic_v2-v2 | 150X_mcRun3_2024_realistic_v2-v1 [PRODUCTION]; 150X_mcRun3_2024_realistic_v2_ext1-v2 [PRODUCTION] |
| DYto2Mu-2Jets_Bin-0J-MLL-50_amcatnloFXFX-pythia8 | 150X_mcRun3_2024_realistic_v2-v3 | 150X_mcRun3_2024_realistic_v1-v2 [INVALID]; 150X_mcRun3_2024_realistic_v2-v2 [INVALID] |
| DYto2Mu-2Jets_Bin-1J-MLL-50_amcatnloFXFX-pythia8 | 150X_mcRun3_2024_realistic_v2-v1 | 150X_mcRun3_2024_realistic_v2_ext1-v2 [PRODUCTION]; 150X_mcRun3_2024_realistic_v2_ext2-v2 [PRODUCTION] |

### Cross-section source legend

| code | source |
|---|---|
| S1 | existing WrCoffea configs (6340.2 = 3 x 2113.4/flavour, NNLO-type normalisation; XSDB GenXsecAnalyzer values: NLO 6688, LO 5467 [FLAF yaml]) |
| S2 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-4Jets_MLL-10to50, modifiedOn 2024-04-03; raw LO value, no k-factor) |
| S3 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-2Jets_MLL-10to50, modifiedOn 2023-03-26) |
| S4 | existing WrCoffea configs (archived lo_ht configs; no independent public reference found, XSDB unreachable) |
| S5 | existing WrCoffea configs (archived nlo_mll configs) = XSDB [FLAF yaml, modifiedOn 2024-04-03] |
| S6 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2E_MLL-10to50 powheg, modifiedOn 2024-04-03) |
| S7 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2Mu_MLL-10to50 powheg, modifiedOn 2024-04-03) |
| S8 | existing WrCoffea configs (2022/2023 archived nlo_njet) ; XSDB [FLAF yaml, 2024-04-03]: 0J 5378, 1J 1017, 2J 385.5 (all-flavour) |
| S9 | cms-flaf/FLAF crossSections13p6TeV.yaml (sum of the XSDB DYto2L-2Jets_MLL-50_PTLL-<bin>_<1J,2J> values above the threshold; FLAF PR #336) |
| S10 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-4Jets_MLL-50_1J, modifiedOn 2024-04-03; raw LO value) |
| S11 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-4Jets_MLL-50_2J, modifiedOn 2024-04-03; raw LO value) |
| S12 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-4Jets_MLL-50_3J, modifiedOn 2024-04-03; raw LO value) |
| S13 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-4Jets_MLL-50_4J, modifiedOn 2024-04-03; raw LO value) |
| S14 | XSDB value as recorded in uhh-cms/cmsdb cmsdb/processes/ewk.py (dy_m50toinf_pt40to100, raw LO value) |
| S15 | XSDB value as recorded in uhh-cms/cmsdb cmsdb/processes/ewk.py (dy_m50toinf_pt100to200, raw LO value) |
| S16 | XSDB value as recorded in uhh-cms/cmsdb cmsdb/processes/ewk.py (dy_m50toinf_pt200to400, raw LO value) |
| S17 | XSDB value as recorded in uhh-cms/cmsdb cmsdb/processes/ewk.py (dy_m50toinf_pt400to600, raw LO value) |
| S18 | XSDB value as recorded in uhh-cms/cmsdb cmsdb/processes/ewk.py (dy_m50toinf_pt600toinf, raw LO value) |
| S19 | existing WrCoffea configs (= 923.6 pb NNLO+NNLL [LHCPhysics/TtbarNNLO] x 0.105) |
| S20 | existing WrCoffea configs (= 87.9/2 pb [LHCPhysics/SingleTopNNLORef tW] x 0.105) |
| S21 | existing WrCoffea configs (2024 config) = XSDB [FLAF yaml, modifiedOn 2025-09-22] |
| S22 | existing WrCoffea configs (XSDB-style value; FLAF uses 7.244 x 0.3259 = 2.361) |
| S23 | existing WrCoffea configs (XSDB-style value; FLAF uses 4.534 x 0.3259 = 1.478) |
| S24 | existing WrCoffea configs (= 923.6 x 0.438) |
| S25 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 9084) |
| S26 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 2925) |
| S27 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 861.7) |
| S28 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 416.5) |
| S29 | existing WrCoffea configs = XSDB [FLAF yaml] |
| S30 | existing WrCoffea configs (2022/2023 configs; XSDB [FLAF yaml, retrieved 2026-09-24] = 0.2505 +- 0.001043) |
| S31 | existing WrCoffea configs (2024 config) = XSDB [FLAF yaml, modifiedOn 2024-12-20] |
| S32 | existing WrCoffea configs (2024 archived nlo_njet; VERIFY: value is at the all-flavour level although the sample is single-flavour, see report) |

## 7. NanoAODv15 set for 2022/2023 (`configs_run3/nanov15/`)

The NanoAODv15 campaigns `Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-vN`, `Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-vN`, `Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-vN`, `Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-vN` were searched with `/<primary>/<Era>NanoAODv15*/NANOAODSIM` for every primary dataset chosen in the v12 set (plus `/WRtoN*/*NanoAODv15*/`, `/TBbarQ_t-channel_4FS*/*NanoAODv15*/`, `/TbarBQ_t-channel_4FS*/*NanoAODv15*/`, `/TTLNu-EWK*/*NanoAODv15*/`, `/WWW*/*NanoAODv15*/`). Completeness per era (v15 event counts are typically within a few % of v12; 2022EE `TTto2L2Nu` v15 has 74.2 M vs 84.8 M):

### Run3Summer22: NanoAODv12 (current) vs NanoAODv15 availability

| family / block | v12 datasets (events) | v15 datasets (events) | v15 status |
|---|---|---|---|
| non-DY background | 22 processes / 27 datasets (295,479,908) | 17 processes / 17 datasets (158,094,554) | partial |
| DY LO inclusive | 2 processes / 3 datasets (293,704,970) | 2 processes / 2 datasets (224,375,311) | complete |
| DY NLO inclusive | 2 processes / 4 datasets (333,044,796) | 2 processes / 2 datasets (191,494,297) | complete |
| DY LO HT-binned | 14 processes / 14 datasets (128,607,280) | 0 processes / 0 datasets () | absent |
| DY powheg MLL-binned | 20 processes / 20 datasets (19,088,171) | 2 processes / 2 datasets (5,739,085) | partial (only MLL-50to120 e/mu) |
| DY NLO jet-binned | 3 processes / 3 datasets (273,449,832) | 0 processes / 0 datasets () | absent |
| DY NLO pT(ll) nested | 4 processes / 4 datasets (152,286,172) | 0 processes / 0 datasets () | absent |
| DY LO jet-binned | 4 processes / 4 datasets (42,099,351) | 0 processes / 0 datasets () | absent |
| DY LO pT(ll)-binned | 5 processes / 5 datasets (71,683,554) | 0 processes / 0 datasets () | absent |
| WR signal (WRtoNEl/WRtoNMu, 200 points) | 200 datasets | 0 | absent (no /WRto*/*NanoAODv15*/ datasets at all) |

Process-level detail (Run3Summer22):

| process | v12 dataset(s) | v12 events | v15 dataset(s) | v15 events |
|---|---|---|---|---|
| TTto2L2Nu | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2` | 23,778,148<br>24,047,488 | `TTto2L2Nu_powheg-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2` | 23,802,613 |
| TWminusto2L2Nu | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2` | 2,369,680<br>2,499,302 | `TWminusto2L2Nu_powheg-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 2,387,056 |
| TbarWplusto2L2Nu | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2` | 2,465,972<br>2,461,995 | `TbarWplusto2L2Nu_powheg-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 2,327,688 |
| TBbarQtoLNu-t-channel-4FS | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 2,998,239 | `TBbarQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 2,973,675 |
| TbarBQtoLNu-t-channel-4FS | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,497,050 | `TbarBQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 1,433,215 |
| TBbartoLplusNuBbar-s-channel-4FS | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,271,280 | `TBbartoLplusNuBbar-s-channel-4FS_amcatnlo-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 1,271,280 |
| TbarBtoLminusNuB-s-channel-4FS | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 785,520 | `TbarBtoLminusNuB-s-channel-4FS_amcatnlo-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 785,520 |
| TTtoLNu2Q | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2` | 88,510,970<br>77,291,760 | `TTtoLNu2Q_powheg-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2` | 65,851,885 |
| WtoLNu-4Jets_1J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1` | 9,871,448 | `WtoLNu-4Jets_1J_madgraphMLM-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 11,769,862 |
| WtoLNu-4Jets_2J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1` | 9,184,457 | `WtoLNu-4Jets_2J_madgraphMLM-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 9,283,334 |
| WtoLNu-4Jets_3J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 7,804,738 | `WtoLNu-4Jets_3J_madgraphMLM-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 8,221,862 |
| WtoLNu-4Jets_4J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,463,885 | `WtoLNu-4Jets_4J_madgraphMLM-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 1,463,885 |
| TTLL_MLL-50 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v3` | 399,000<br>4,000,000 | `TTLL_MLL-50_amcatnlo-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 399,000 |
| TTLNu-1Jets | `Run3Summer22NanoAODv12-mg35x_130X_mcRun3_2022_realistic_v5-v2` | 2,056,905 | `TTLNu-1Jets_amcatnloFXFX-pythia8 / Run3Summer22NanoAODv15-mg35x_150X_mcRun3_2022_realistic_v1-v1` | 2,056,905 |
| TTLNu-EWK | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 298,552 | — | — |
| WW | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 15,357,390 | `WW_pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 15,405,496 |
| WZ | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 7,527,043 | `WZ_pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 7,479,528 |
| ZZ | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,181,750 | `ZZ_pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 1,181,750 |
| WWW | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 450,000 | — | — |
| WWZ | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,950,044 | — | — |
| WZZ | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,987,058 | — | — |
| ZZZ | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,970,234 | — | — |
| DY_LO_inc | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v3` | 72,999,340<br>73,925,935 | `DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2` | 73,131,170 |
| DY_LO_10to50 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 146,779,695 | `DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 151,244,141 |
| DY_NLO_inc | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v5`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2` | 93,826,598<br>92,738,407 | `DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2` | 93,967,480 |
| DY_NLO_10to50 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v4`<br>`Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5_ext1-v2` | 97,571,898<br>48,907,893 | `DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1` | 97,526,817 |
| DY_LO_HT_50to120_40to70 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 41,806,316 | — | — |
| DY_LO_HT_50to120_70to100 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 29,552,642 | — | — |
| DY_LO_HT_50to120_100to400 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 34,765,767 | — | — |
| DY_LO_HT_50to120_400to800 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,246,830 | — | — |
| DY_LO_HT_50to120_800to1500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,340,651 | — | — |
| DY_LO_HT_50to120_1500to2500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,385,017 | — | — |
| DY_LO_HT_50to120_2500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v4` | 1,630,460 | — | — |
| DY_LO_HT_120_40to70 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,707,578 | — | — |
| DY_LO_HT_120_70to100 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 1,953,844 | — | — |
| DY_LO_HT_120_100to400 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,113,763 | — | — |
| DY_LO_HT_120_400to800 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 1,856,726 | — | — |
| DY_LO_HT_120_800to1500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,049,333 | — | — |
| DY_LO_HT_120_1500to2500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 1,935,494 | — | — |
| DY_LO_HT_120_2500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,262,859 | — | — |
| DY_NLO_MLL_E_50to120 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 2,892,300 | `DYto2E_MLL-50to120_powheg-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v3` | 2,918,148 |
| DY_NLO_MLL_E_120to200 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,497,870 | — | — |
| DY_NLO_MLL_E_200to400 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 874,584 | — | — |
| DY_NLO_MLL_E_400to800 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,074,165 | — | — |
| DY_NLO_MLL_E_800to1500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 610,464 | — | — |
| DY_NLO_MLL_E_1500to2500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 586,690 | — | — |
| DY_NLO_MLL_E_2500to4000 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 290,480 | — | — |
| DY_NLO_MLL_E_4000to6000 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 298,948 | — | — |
| DY_NLO_MLL_E_6000 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 145,094 | — | — |
| DY_NLO_MLL_E_10to50 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,477,950 | — | — |
| DY_NLO_MLL_Mu_50to120 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 2,820,937 | `DYto2Mu_MLL-50to120_powheg-pythia8 / Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v3` | 2,820,937 |
| DY_NLO_MLL_Mu_120to200 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,453,748 | — | — |
| DY_NLO_MLL_Mu_200to400 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 853,443 | — | — |
| DY_NLO_MLL_Mu_400to800 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 874,240 | — | — |
| DY_NLO_MLL_Mu_800to1500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 579,560 | — | — |
| DY_NLO_MLL_Mu_1500to2500 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 590,523 | — | — |
| DY_NLO_MLL_Mu_2500to4000 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 299,278 | — | — |
| DY_NLO_MLL_Mu_4000to6000 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 289,200 | — | — |
| DY_NLO_MLL_Mu_6000 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 145,002 | — | — |
| DY_NLO_MLL_Mu_10to50 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 1,433,695 | — | — |
| DY_NLO_NJ_0J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 98,597,821 | — | — |
| DY_NLO_NJ_1J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 97,156,305 | — | — |
| DY_NLO_NJ_2J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 77,695,706 | — | — |
| DY_NLO_PTLL_100 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 56,429,873 | — | — |
| DY_NLO_PTLL_200 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 44,727,479 | — | — |
| DY_NLO_PTLL_400 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 41,629,822 | — | — |
| DY_NLO_PTLL_600 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2` | 9,498,998 | — | — |
| DY_LO_NJ_1J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 15,216,450 | — | — |
| DY_LO_NJ_2J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 15,269,289 | — | — |
| DY_LO_NJ_3J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 8,842,997 | — | — |
| DY_LO_NJ_4J | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 2,770,615 | — | — |
| DY_LO_PTLL_40to100 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 48,615,009 | — | — |
| DY_LO_PTLL_100to200 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 19,261,827 | — | — |
| DY_LO_PTLL_200to400 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 1,857,550 | — | — |
| DY_LO_PTLL_400to600 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 983,966 | — | — |
| DY_LO_PTLL_600 | `Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3` | 965,202 | — | — |

### Run3Summer22EE: NanoAODv12 (current) vs NanoAODv15 availability

| family / block | v12 datasets (events) | v15 datasets (events) | v15 status |
|---|---|---|---|
| non-DY background | 22 processes / 27 datasets (997,910,097) | 17 processes / 17 datasets (582,472,371) | partial |
| DY LO inclusive | 2 processes / 3 datasets (1,005,881,963) | 2 processes / 2 datasets (767,513,561) | complete |
| DY NLO inclusive | 2 processes / 4 datasets (1,223,120,811) | 2 processes / 2 datasets (689,298,281) | complete |
| DY LO HT-binned | 14 processes / 14 datasets (419,059,892) | 0 processes / 0 datasets () | absent |
| DY powheg MLL-binned | 20 processes / 20 datasets (66,529,653) | 2 processes / 2 datasets (20,131,508) | partial (only MLL-50to120 e/mu) |
| DY NLO jet-binned | 3 processes / 3 datasets (966,220,922) | 0 processes / 0 datasets () | absent |
| DY NLO pT(ll) nested | 4 processes / 4 datasets (472,265,880) | 0 processes / 0 datasets () | absent |
| DY LO jet-binned | 4 processes / 4 datasets (138,927,656) | 0 processes / 0 datasets () | absent |
| DY LO pT(ll)-binned | 5 processes / 5 datasets (255,499,878) | 0 processes / 0 datasets () | absent |
| WR signal (WRtoNEl/WRtoNMu, 200 points) | 200 datasets | 0 | absent (no /WRto*/*NanoAODv15*/ datasets at all) |

Process-level detail (Run3Summer22EE):

| process | v12 dataset(s) | v12 events | v15 dataset(s) | v15 events |
|---|---|---|---|---|
| TTto2L2Nu | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2` | 84,809,345<br>84,870,780 | `TTto2L2Nu_powheg-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v3` | 74,184,271 |
| TWminusto2L2Nu | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2` | 8,129,084<br>8,466,410 | `TWminusto2L2Nu_powheg-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 8,065,364 |
| TbarWplusto2L2Nu | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2` | 8,183,339<br>8,555,460 | `TbarWplusto2L2Nu_powheg-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 8,260,009 |
| TBbarQtoLNu-t-channel-4FS | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 10,249,351 | `TBbarQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 10,178,237 |
| TbarBQtoLNu-t-channel-4FS | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,239,919 | `TbarBQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 5,185,208 |
| TBbartoLplusNuBbar-s-channel-4FS | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 4,363,850 | `TBbartoLplusNuBbar-s-channel-4FS_amcatnlo-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 4,363,850 |
| TbarBtoLminusNuB-s-channel-4FS | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,762,668 | `TbarBtoLminusNuB-s-channel-4FS_amcatnlo-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 2,762,668 |
| TTtoLNu2Q | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2` | 267,927,960<br>275,339,453 | `TTtoLNu2Q_powheg-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v2` | 265,200,599 |
| WtoLNu-4Jets_1J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 40,329,821 | `WtoLNu-4Jets_1J_madgraphMLM-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 42,695,566 |
| WtoLNu-4Jets_2J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 33,044,336 | `WtoLNu-4Jets_2J_madgraphMLM-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 36,349,344 |
| WtoLNu-4Jets_3J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 26,832,656 | `WtoLNu-4Jets_3J_madgraphMLM-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 27,828,446 |
| WtoLNu-4Jets_4J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 4,908,700 | `WtoLNu-4Jets_4J_madgraphMLM-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 4,906,634 |
| TTLL_MLL-50 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2` | 1,400,000<br>12,586,478 | `TTLL_MLL-50_amcatnlo-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 1,400,000 |
| TTLNu-1Jets | `Run3Summer22EENanoAODv12-mg35x_130X_mcRun3_2022_realistic_postEE_v6-v2` | 7,212,758 | `TTLNu-1Jets_amcatnloFXFX-pythia8 / Run3Summer22EENanoAODv15-mg35x_150X_mcRun3_2022_realistic_postEE_v1-v1` | 7,214,273 |
| TTLNu-EWK | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 599,291 | — | — |
| WW | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 53,190,560 | `WW_pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 53,112,080 |
| WZ | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 26,840,468 | `WZ_pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 26,722,782 |
| ZZ | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 3,890,400 | `ZZ_pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 4,043,040 |
| WWW | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 1,482,480 | — | — |
| WWZ | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,601,076 | — | — |
| WZZ | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,290,014 | — | — |
| ZZZ | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,803,440 | — | — |
| DY_LO_inc | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v3` | 263,385,429<br>253,535,615 | `DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v2` | 264,413,000 |
| DY_LO_10to50 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4` | 488,960,919 | `DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 503,100,561 |
| DY_NLO_inc | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v2` | 335,560,148<br>353,554,978 | `DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v2` | 335,241,916 |
| DY_NLO_10to50 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5`<br>`Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6_ext1-v4` | 355,091,803<br>178,913,882 | `DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1` | 354,056,365 |
| DY_LO_HT_50to120_40to70 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 125,345,985 | — | — |
| DY_LO_HT_50to120_70to100 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4` | 94,595,270 | — | — |
| DY_LO_HT_50to120_100to400 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 124,757,918 | — | — |
| DY_LO_HT_50to120_400to800 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 3,522,511 | — | — |
| DY_LO_HT_50to120_800to1500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4` | 1,256,042 | — | — |
| DY_LO_HT_50to120_1500to2500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 6,936,428 | — | — |
| DY_LO_HT_50to120_2500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v5` | 7,048,263 | — | — |
| DY_LO_HT_120_40to70 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4` | 9,904,309 | — | — |
| DY_LO_HT_120_70to100 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6` | 8,071,088 | — | — |
| DY_LO_HT_120_100to400 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 7,987,714 | — | — |
| DY_LO_HT_120_400to800 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6` | 7,121,044 | — | — |
| DY_LO_HT_120_800to1500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4` | 7,490,387 | — | — |
| DY_LO_HT_120_1500to2500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6` | 7,286,454 | — | — |
| DY_LO_HT_120_2500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4` | 7,736,479 | — | — |
| DY_NLO_MLL_E_50to120 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 10,403,118 | `DYto2E_MLL-50to120_powheg-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v3` | 10,403,118 |
| DY_NLO_MLL_E_120to200 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,238,528 | — | — |
| DY_NLO_MLL_E_200to400 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 3,147,891 | — | — |
| DY_NLO_MLL_E_400to800 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 3,044,700 | — | — |
| DY_NLO_MLL_E_800to1500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,004,300 | — | — |
| DY_NLO_MLL_E_1500to2500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,053,944 | — | — |
| DY_NLO_MLL_E_2500to4000 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 986,496 | — | — |
| DY_NLO_MLL_E_4000to6000 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 1,027,656 | — | — |
| DY_NLO_MLL_E_6000 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 550,816 | — | — |
| DY_NLO_MLL_E_10to50 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,333,106 | — | — |
| DY_NLO_MLL_Mu_50to120 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 10,150,330 | `DYto2Mu_MLL-50to120_powheg-pythia8 / Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v4` | 9,728,390 |
| DY_NLO_MLL_Mu_120to200 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,050,710 | — | — |
| DY_NLO_MLL_Mu_200to400 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,972,383 | — | — |
| DY_NLO_MLL_Mu_400to800 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,930,748 | — | — |
| DY_NLO_MLL_Mu_800to1500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,093,529 | — | — |
| DY_NLO_MLL_Mu_1500to2500 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 2,008,146 | — | — |
| DY_NLO_MLL_Mu_2500to4000 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 1,003,792 | — | — |
| DY_NLO_MLL_Mu_4000to6000 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 994,900 | — | — |
| DY_NLO_MLL_Mu_6000 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 503,888 | — | — |
| DY_NLO_MLL_Mu_10to50 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 5,030,672 | — | — |
| DY_NLO_NJ_0J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 345,496,261 | — | — |
| DY_NLO_NJ_1J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 336,400,299 | — | — |
| DY_NLO_NJ_2J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 284,324,362 | — | — |
| DY_NLO_PTLL_100 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 155,924,254 | — | — |
| DY_NLO_PTLL_200 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 142,202,452 | — | — |
| DY_NLO_PTLL_400 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 140,605,102 | — | — |
| DY_NLO_PTLL_600 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2` | 33,534,072 | — | — |
| DY_LO_NJ_1J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 48,075,885 | — | — |
| DY_LO_NJ_2J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 52,871,956 | — | — |
| DY_LO_NJ_3J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 28,134,050 | — | — |
| DY_LO_NJ_4J | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 9,845,765 | — | — |
| DY_LO_PTLL_40to100 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v6` | 174,980,872 | — | — |
| DY_LO_PTLL_100to200 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 67,196,347 | — | — |
| DY_LO_PTLL_200to400 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 6,422,177 | — | — |
| DY_LO_PTLL_400to600 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 3,665,389 | — | — |
| DY_LO_PTLL_600 | `Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3` | 3,235,093 | — | — |

### Run3Summer23: NanoAODv12 (current) vs NanoAODv15 availability

| family / block | v12 datasets (events) | v15 datasets (events) | v15 status |
|---|---|---|---|
| non-DY background | 22 processes / 23 datasets (363,908,328) | 17 processes / 17 datasets (340,205,876) | partial |
| DY LO inclusive | 2 processes / 2 datasets (393,793,612) | 2 processes / 2 datasets (419,713,927) | complete |
| DY NLO inclusive | 2 processes / 2 datasets (400,279,035) | 2 processes / 2 datasets (408,153,677) | complete |
| DY LO HT-binned | 14 processes / 14 datasets (256,202,300) | 0 processes / 0 datasets () | absent |
| DY powheg MLL-binned | 20 processes / 20 datasets (38,393,000) | 2 processes / 2 datasets (11,764,000) | partial (only MLL-50to120 e/mu) |
| DY NLO jet-binned | 3 processes / 3 datasets (530,969,087) | 0 processes / 0 datasets () | absent |
| DY NLO pT(ll) nested | 4 processes / 4 datasets (271,258,299) | 0 processes / 0 datasets () | absent |
| DY LO jet-binned | 4 processes / 4 datasets (80,035,240) | 0 processes / 0 datasets () | absent |
| DY LO pT(ll)-binned | 5 processes / 5 datasets (175,199,125) | 0 processes / 0 datasets () | absent |
| WR signal (WRtoNEl/WRtoNMu, 200 points) | 200 datasets | 0 | absent (no /WRto*/*NanoAODv15*/ datasets at all) |

Process-level detail (Run3Summer23):

| process | v12 dataset(s) | v12 events | v15 dataset(s) | v15 events |
|---|---|---|---|---|
| TTto2L2Nu | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 48,203,000 | `TTto2L2Nu_powheg-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v2` | 48,104,000 |
| TWminusto2L2Nu | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 4,985,000 | `TWminusto2L2Nu_powheg-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 4,985,000 |
| TbarWplusto2L2Nu | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4` | 4,931,000 | `TbarWplusto2L2Nu_powheg-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 4,907,000 |
| TBbarQtoLNu-t-channel-4FS | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v1` | 5,935,200 | `TBbarQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 5,908,000 |
| TbarBQtoLNu-t-channel-4FS | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v1` | 2,958,200 | `TbarBQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 2,878,000 |
| TBbartoLplusNuBbar-s-channel-4FS | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2` | 2,596,000 | `TBbartoLplusNuBbar-s-channel-4FS_amcatnlo-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 2,588,000 |
| TbarBtoLminusNuB-s-channel-4FS | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2` | 1,600,000 | `TbarBtoLminusNuB-s-channel-4FS_amcatnlo-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 1,600,000 |
| TTtoLNu2Q | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 152,797,000 | `TTtoLNu2Q_powheg-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v2` | 149,503,000 |
| WtoLNu-4Jets_1J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 23,088,467 | `WtoLNu-4Jets_1J_madgraphMLM-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 23,085,414 |
| WtoLNu-4Jets_2J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 20,264,539 | `WtoLNu-4Jets_2J_madgraphMLM-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 20,267,771 |
| WtoLNu-4Jets_3J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 15,887,453 | `WtoLNu-4Jets_3J_madgraphMLM-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 15,887,453 |
| WtoLNu-4Jets_4J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 2,961,904 | `WtoLNu-4Jets_4J_madgraphMLM-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 2,962,673 |
| TTLL_MLL-50 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3`<br>`Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15_ext1-v2` | 800,000<br>7,850,000 | `TTLL_MLL-50_amcatnlo-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 800,000 |
| TTLNu-1Jets | `Run3Summer23NanoAODv12-mg35x_130X_mcRun3_2023_realistic_v15-v2` | 3,935,565 | `TTLNu-1Jets_amcatnloFXFX-pythia8 / Run3Summer23NanoAODv15-mg35x_150X_mcRun3_2023_realistic_v1-v1` | 3,935,565 |
| TTLNu-EWK | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2` | 597,000 | — | — |
| WW | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 33,504,000 | `WW_pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 33,507,000 |
| WZ | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 16,770,000 | `WZ_pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 16,770,000 |
| ZZ | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 2,517,000 | `ZZ_pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 2,517,000 |
| WWW | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 936,000 | — | — |
| WWZ | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 3,600,000 | — | — |
| WZZ | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 3,591,000 | — | — |
| ZZZ | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 3,600,000 | — | — |
| DY_LO_inc | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 127,376,178 | `DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v3` | 127,692,949 |
| DY_LO_10to50 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 266,417,434 | `DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 292,020,978 |
| DY_NLO_inc | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 200,038,891 | `DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1` | 205,068,372 |
| DY_NLO_10to50 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14_ext1-v4` | 200,240,144 | `DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1_ext1-v1` | 203,085,305 |
| DY_LO_HT_50to120_40to70 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 76,207,428 | — | — |
| DY_LO_HT_50to120_70to100 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 57,235,953 | — | — |
| DY_LO_HT_50to120_100to400 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 77,606,305 | — | — |
| DY_LO_HT_50to120_400to800 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 2,153,656 | — | — |
| DY_LO_HT_50to120_800to1500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 4,523,217 | — | — |
| DY_LO_HT_50to120_1500to2500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 4,062,569 | — | — |
| DY_LO_HT_50to120_2500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 4,090,115 | — | — |
| DY_LO_HT_120_40to70 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 6,356,065 | — | — |
| DY_LO_HT_120_70to100 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 3,992,010 | — | — |
| DY_LO_HT_120_100to400 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 3,643,744 | — | — |
| DY_LO_HT_120_400to800 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 3,691,611 | — | — |
| DY_LO_HT_120_800to1500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 4,925,219 | — | — |
| DY_LO_HT_120_1500to2500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 3,782,702 | — | — |
| DY_LO_HT_120_2500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 3,931,706 | — | — |
| DY_NLO_MLL_E_50to120 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 5,832,000 | `DYto2E_MLL-50to120_powheg-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v3` | 5,904,000 |
| DY_NLO_MLL_E_120to200 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 2,991,000 | — | — |
| DY_NLO_MLL_E_200to400 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,734,000 | — | — |
| DY_NLO_MLL_E_400to800 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,788,000 | — | — |
| DY_NLO_MLL_E_800to1500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3` | 1,200,000 | — | — |
| DY_NLO_MLL_E_1500to2500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,154,000 | — | — |
| DY_NLO_MLL_E_2500to4000 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 590,000 | — | — |
| DY_NLO_MLL_E_4000to6000 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 572,000 | — | — |
| DY_NLO_MLL_E_6000 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 298,000 | — | — |
| DY_NLO_MLL_E_10to50 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 2,968,000 | — | — |
| DY_NLO_MLL_Mu_50to120 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 5,904,000 | `DYto2Mu_MLL-50to120_powheg-pythia8 / Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v3` | 5,860,000 |
| DY_NLO_MLL_Mu_120to200 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 2,994,000 | — | — |
| DY_NLO_MLL_Mu_200to400 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,776,000 | — | — |
| DY_NLO_MLL_Mu_400to800 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,794,000 | — | — |
| DY_NLO_MLL_Mu_800to1500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,179,000 | — | — |
| DY_NLO_MLL_Mu_1500to2500 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 1,191,000 | — | — |
| DY_NLO_MLL_Mu_2500to4000 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 585,000 | — | — |
| DY_NLO_MLL_Mu_4000to6000 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 558,000 | — | — |
| DY_NLO_MLL_Mu_6000 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 285,000 | — | — |
| DY_NLO_MLL_Mu_10to50 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2` | 3,000,000 | — | — |
| DY_NLO_NJ_0J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 201,186,460 | — | — |
| DY_NLO_NJ_1J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 175,272,543 | — | — |
| DY_NLO_NJ_2J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 154,510,084 | — | — |
| DY_NLO_PTLL_100 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2` | 95,212,367 | — | — |
| DY_NLO_PTLL_200 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2` | 80,677,175 | — | — |
| DY_NLO_PTLL_400 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 75,275,774 | — | — |
| DY_NLO_PTLL_600 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2` | 20,092,983 | — | — |
| DY_LO_NJ_1J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 29,630,439 | — | — |
| DY_LO_NJ_2J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 29,725,448 | — | — |
| DY_LO_NJ_3J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 15,297,076 | — | — |
| DY_LO_NJ_4J | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v4` | 5,382,277 | — | — |
| DY_LO_PTLL_40to100 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 121,445,289 | — | — |
| DY_LO_PTLL_100to200 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 45,141,048 | — | — |
| DY_LO_PTLL_200to400 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 4,236,951 | — | — |
| DY_LO_PTLL_400to600 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 2,211,418 | — | — |
| DY_LO_PTLL_600 | `Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3` | 2,164,419 | — | — |

### Run3Summer23BPix: NanoAODv12 (current) vs NanoAODv15 availability

| family / block | v12 datasets (events) | v15 datasets (events) | v15 status |
|---|---|---|---|
| non-DY background | 22 processes / 23 datasets (194,106,124) | 17 processes / 17 datasets (158,456,336) | partial |
| DY LO inclusive | 2 processes / 2 datasets (211,679,573) | 2 processes / 2 datasets (219,269,568) | complete |
| DY NLO inclusive | 2 processes / 2 datasets (195,287,120) | 2 processes / 2 datasets (196,535,352) | complete |
| DY LO HT-binned | 14 processes / 14 datasets (139,211,476) | 0 processes / 0 datasets () | absent |
| DY powheg MLL-binned | 20 processes / 20 datasets (19,082,000) | 2 processes / 2 datasets (5,762,000) | partial (only MLL-50to120 e/mu) |
| DY NLO jet-binned | 3 processes / 3 datasets (258,874,539) | 0 processes / 0 datasets () | absent |
| DY NLO pT(ll) nested | 4 processes / 4 datasets (124,801,372) | 0 processes / 0 datasets () | absent |
| DY LO jet-binned | 4 processes / 4 datasets (41,313,936) | 0 processes / 0 datasets () | absent |
| DY LO pT(ll)-binned | 5 processes / 5 datasets (72,425,913) | 0 processes / 0 datasets () | absent |
| WR signal (WRtoNEl/WRtoNMu, 200 points) | 200 datasets | 0 | absent (no /WRto*/*NanoAODv15*/ datasets at all) |

Process-level detail (Run3Summer23BPix):

| process | v12 dataset(s) | v12 events | v15 dataset(s) | v15 events |
|---|---|---|---|---|
| TTto2L2Nu | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 24,649,000 | `TTto2L2Nu_powheg-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v2` | 24,556,000 |
| TWminusto2L2Nu | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 2,479,000 | `TWminusto2L2Nu_powheg-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 2,479,000 |
| TbarWplusto2L2Nu | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 2,488,000 | `TbarWplusto2L2Nu_powheg-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 2,488,000 |
| TBbarQtoLNu-t-channel-4FS | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v1` | 10,319,600 | `TBbarQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 2,954,000 |
| TbarBQtoLNu-t-channel-4FS | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v1` | 1,486,700 | `TbarBQ_t-channel_4FS_powheg-madspin-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 1,488,000 |
| TBbartoLplusNuBbar-s-channel-4FS | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 1,276,000 | `TBbartoLplusNuBbar-s-channel-4FS_amcatnlo-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 1,288,000 |
| TbarBtoLminusNuB-s-channel-4FS | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 788,000 | `TbarBtoLminusNuB-s-channel-4FS_amcatnlo-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 784,000 |
| TTtoLNu2Q | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 82,058,000 | `TTtoLNu2Q_powheg-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v2` | 63,875,000 |
| WtoLNu-4Jets_1J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 11,369,658 | `WtoLNu-4Jets_1J_madgraphMLM-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 11,369,658 |
| WtoLNu-4Jets_2J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 9,345,696 | `WtoLNu-4Jets_2J_madgraphMLM-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 9,459,992 |
| WtoLNu-4Jets_3J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 7,718,351 | `WtoLNu-4Jets_3J_madgraphMLM-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 7,718,351 |
| WtoLNu-4Jets_4J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 1,441,228 | `WtoLNu-4Jets_4J_madgraphMLM-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 1,436,944 |
| TTLL_MLL-50 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3`<br>`Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6_ext1-v2` | 400,000<br>3,856,000 | `TTLL_MLL-50_amcatnlo-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 400,000 |
| TTLNu-1Jets | `Run3Summer23BPixNanoAODv12-mg35x_130X_mcRun3_2023_realistic_postBPix_v6-v2` | 1,981,391 | `TTLNu-1Jets_amcatnloFXFX-pythia8 / Run3Summer23BPixNanoAODv15-mg35x_150X_mcRun3_2023_realistic_postBPix_v1-v1` | 1,981,391 |
| TTLNu-EWK | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 294,000 | — | — |
| WW | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2` | 16,734,000 | `WW_pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 16,545,000 |
| WZ | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2` | 8,379,000 | `WZ_pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 8,379,000 |
| ZZ | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2` | 1,254,000 | `ZZ_pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 1,254,000 |
| WWW | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2` | 466,500 | — | — |
| WWZ | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 1,746,000 | — | — |
| WZZ | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2` | 1,788,000 | — | — |
| ZZZ | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2` | 1,788,000 | — | — |
| DY_LO_inc | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 70,167,286 | `DYto2L-4Jets_MLL-50_madgraphMLM-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v2` | 70,099,933 |
| DY_LO_10to50 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 141,512,287 | `DYto2L-4Jets_MLL-10to50_madgraphMLM-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 149,169,635 |
| DY_NLO_inc | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 94,876,537 | `DYto2L-2Jets_MLL-50_amcatnloFXFX-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1` | 95,217,757 |
| DY_NLO_10to50 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2_ext1-v4` | 100,410,583 | `DYto2L-2Jets_MLL-10to50_amcatnloFXFX-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1_ext1-v1` | 101,317,595 |
| DY_LO_HT_50to120_40to70 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5` | 37,755,405 | — | — |
| DY_LO_HT_50to120_70to100 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5` | 34,689,209 | — | — |
| DY_LO_HT_50to120_100to400 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 45,276,159 | — | — |
| DY_LO_HT_50to120_400to800 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 1,162,372 | — | — |
| DY_LO_HT_50to120_800to1500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 1,832,189 | — | — |
| DY_LO_HT_50to120_1500to2500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v5` | 1,938,272 | — | — |
| DY_LO_HT_50to120_2500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 1,882,022 | — | — |
| DY_LO_HT_120_40to70 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 2,842,602 | — | — |
| DY_LO_HT_120_70to100 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 1,893,044 | — | — |
| DY_LO_HT_120_100to400 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 1,863,139 | — | — |
| DY_LO_HT_120_400to800 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v4` | 1,905,893 | — | — |
| DY_LO_HT_120_800to1500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 2,177,931 | — | — |
| DY_LO_HT_120_1500to2500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 1,877,696 | — | — |
| DY_LO_HT_120_2500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v4` | 2,115,543 | — | — |
| DY_NLO_MLL_E_50to120 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 2,910,000 | `DYto2E_MLL-50to120_powheg-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v3` | 2,910,000 |
| DY_NLO_MLL_E_120to200 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 1,476,000 | — | — |
| DY_NLO_MLL_E_200to400 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 867,000 | — | — |
| DY_NLO_MLL_E_400to800 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 894,000 | — | — |
| DY_NLO_MLL_E_800to1500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 591,000 | — | — |
| DY_NLO_MLL_E_1500to2500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 580,000 | — | — |
| DY_NLO_MLL_E_2500to4000 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 300,000 | — | — |
| DY_NLO_MLL_E_4000to6000 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 300,000 | — | — |
| DY_NLO_MLL_E_6000 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 150,000 | — | — |
| DY_NLO_MLL_E_10to50 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 1,476,000 | — | — |
| DY_NLO_MLL_Mu_50to120 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 2,872,000 | `DYto2Mu_MLL-50to120_powheg-pythia8 / Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v3` | 2,852,000 |
| DY_NLO_MLL_Mu_120to200 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 1,496,000 | — | — |
| DY_NLO_MLL_Mu_200to400 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 891,000 | — | — |
| DY_NLO_MLL_Mu_400to800 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 892,000 | — | — |
| DY_NLO_MLL_Mu_800to1500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 584,000 | — | — |
| DY_NLO_MLL_Mu_1500to2500 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 600,000 | — | — |
| DY_NLO_MLL_Mu_2500to4000 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 291,000 | — | — |
| DY_NLO_MLL_Mu_4000to6000 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 294,000 | — | — |
| DY_NLO_MLL_Mu_6000 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 150,000 | — | — |
| DY_NLO_MLL_Mu_10to50 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3` | 1,468,000 | — | — |
| DY_NLO_NJ_0J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 99,748,250 | — | — |
| DY_NLO_NJ_1J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 86,282,834 | — | — |
| DY_NLO_NJ_2J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v5` | 72,843,455 | — | — |
| DY_NLO_PTLL_100 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 41,940,852 | — | — |
| DY_NLO_PTLL_200 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 32,405,482 | — | — |
| DY_NLO_PTLL_400 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 40,272,138 | — | — |
| DY_NLO_PTLL_600 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2` | 10,182,900 | — | — |
| DY_LO_NJ_1J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 15,289,548 | — | — |
| DY_LO_NJ_2J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 14,740,579 | — | — |
| DY_LO_NJ_3J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 8,393,375 | — | — |
| DY_LO_NJ_4J | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v4` | 2,890,434 | — | — |
| DY_LO_PTLL_40to100 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 46,919,628 | — | — |
| DY_LO_PTLL_100to200 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 21,393,032 | — | — |
| DY_LO_PTLL_200to400 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 2,135,466 | — | — |
| DY_LO_PTLL_400to600 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 978,850 | — | — |
| DY_LO_PTLL_600 | `Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3` | 998,937 | — | — |

Consequences for the `nanov15/` configs:

* Only `lo_inc` and `nlo_inc` families are written (with the `MLL-10to50` companions). The powheg MLL family has only the `50to120` bin in v15, HT/NJ/pT(ll) families and the LO jet/pT(ll) families do not exist in v15.
* The background block has 17 datasets: `TTLNu-EWK`, `WWW`, `WWZ`, `WZZ`, `ZZZ` do not exist in v15 (they could be mixed in from v12 at the cost of two NanoAOD formats in one job); the LNu-filtered 4FS t-channel samples are replaced by the inclusive `TBbarQ_t-channel_4FS` / `TbarBQ_t-channel_4FS` (W decays inclusive) with the NNLO cross sections 145.0 / 87.2 pb (LHCPhysics `SingleTopNNLORef`), which is a different normalisation convention from the XSDB values 38.6 / 23.34 pb of the LNu samples in the v12 set.
* No v15 `ext1` of `TTto2L2Nu`/`TW`/`TTLL` exist (v15 of the nominal only); `DYto2L-2Jets_MLL-10to50` in 2023/2023BPix exists only as `_ext1`.
* **No WR signal in NanoAODv15** for any era, so a v15 analysis would mix v15 backgrounds/data with v12 signal.
* All `nanov15/` entries are new (`genEventSumw: null`); xsecs are the same as in the v12 set.

### 7.1 nanov15 data configs

### nanov15/Run3Summer22_data.json (5 datasets)

| NanoAODv15 DAS path | physics_group | release | global tag | files | events | GB | v12 (22Sep2023) events |
|---|---|---|---|---|---|---|---|
| `/SingleMuon/Run2022C-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 19 | 20,162,441 | 24 | 20,162,441 (`Run2022C-22Sep2023-v1`) |
| `/Muon/Run2022C-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 112 | 138,329,693 | 166 | 138,427,345 (`Run2022C-22Sep2023-v1`) |
| `/Muon/Run2022D-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 66 | 75,440,027 | 90 | 75,468,381 (`Run2022D-22Sep2023-v1`) |
| `/EGamma/Run2022C-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 255 | 263,501,653 | 363 | 263,689,151 (`Run2022C-22Sep2023-v1`) |
| `/EGamma/Run2022D-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 80 | 89,081,259 | 129 | 89,134,996 (`Run2022D-22Sep2023-v1`) |

### nanov15/Run3Summer22EE_data.json (6 datasets)

| NanoAODv15 DAS path | physics_group | release | global tag | files | events | GB | v12 (22Sep2023) events |
|---|---|---|---|---|---|---|---|
| `/Muon/Run2022E-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 130 | 141,480,973 | 170 | 141,460,608 (`Run2022E-22Sep2023-v1`) |
| `/Muon/Run2022F-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 349 | 449,185,088 | 557 | 449,887,248 (`Run2022F-22Sep2023-v2`) |
| `/Muon/Run2022G-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 69 | 76,689,396 | 95 | 76,689,396 (`Run2022G-22Sep2023-v1`) |
| `/EGamma/Run2022E-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 152 | 148,616,546 | 217 | 148,654,795 (`Run2022E-22Sep2023-v1`) |
| `/EGamma/Run2022F-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 444 | 463,684,806 | 688 | 464,373,259 (`Run2022F-22Sep2023-v1`) |
| `/EGamma/Run2022G-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 60 | 76,620,820 | 114 | 76,807,350 (`Run2022G-22Sep2023-v2`) |

### nanov15/Run3Summer23_data.json (16 datasets)

| NanoAODv15 DAS path | physics_group | release | global tag | files | events | GB | v12 (22Sep2023) events |
|---|---|---|---|---|---|---|---|
| `/Muon0/Run2023C-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 66 | 54,715,896 | 68 | 54,715,896 (`Run2023C-22Sep2023_v1-v1`) |
| `/Muon0/Run2023C-NanoAODv15_v2-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 18 | 17,063,451 | 21 | 17,063,451 (`Run2023C-22Sep2023_v2-v1`) |
| `/Muon0/Run2023C-NanoAODv15_v3-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 29 | 20,015,377 | 25 | 20,015,377 (`Run2023C-22Sep2023_v3-v1`) |
| `/Muon0/Run2023C-NanoAODv15_v4-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 135 | 138,943,783 | 178 | 138,943,783 (`Run2023C-22Sep2023_v4-v1`) |
| `/Muon1/Run2023C-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 61 | 54,621,922 | 68 | 54,698,315 (`Run2023C-22Sep2023_v1-v1`) |
| `/Muon1/Run2023C-NanoAODv15_v2-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 23 | 17,059,895 | 21 | 17,059,895 (`Run2023C-22Sep2023_v2-v1`) |
| `/Muon1/Run2023C-NanoAODv15_v3-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 22 | 20,010,429 | 25 | 20,010,429 (`Run2023C-22Sep2023_v3-v1`) |
| `/Muon1/Run2023C-NanoAODv15_v4-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 116 | 138,886,941 | 178 | 138,912,262 (`Run2023C-22Sep2023_v4-v2`) |
| `/EGamma0/Run2023C-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 67 | 67,598,081 | 101 | 67,598,081 (`Run2023C-22Sep2023_v1-v1`) |
| `/EGamma0/Run2023C-NanoAODv15_v2-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 17 | 17,233,307 | 26 | 17,233,307 (`Run2023C-22Sep2023_v2-v1`) |
| `/EGamma0/Run2023C-NanoAODv15_v3-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 22 | 21,993,048 | 34 | 21,993,048 (`Run2023C-22Sep2023_v3-v1`) |
| `/EGamma0/Run2023C-NanoAODv15_v4-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 127 | 160,108,119 | 248 | 160,108,119 (`Run2023C-22Sep2023_v4-v1`) |
| `/EGamma1/Run2023C-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 67 | 67,530,273 | 101 | 67,582,665 (`Run2023C-22Sep2023_v1-v1`) |
| `/EGamma1/Run2023C-NanoAODv15_v2-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 19 | 17,230,822 | 26 | 17,230,822 (`Run2023C-22Sep2023_v2-v1`) |
| `/EGamma1/Run2023C-NanoAODv15_v3-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 24 | 21,987,586 | 34 | 21,987,586 (`Run2023C-22Sep2023_v3-v1`) |
| `/EGamma1/Run2023C-NanoAODv15_v4-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 161 | 159,997,174 | 248 | 160,049,621 (`Run2023C-22Sep2023_v4-v1`) |

### nanov15/Run3Summer23BPix_data.json (8 datasets)

| NanoAODv15 DAS path | physics_group | release | global tag | files | events | GB | v12 (22Sep2023) events |
|---|---|---|---|---|---|---|---|
| `/Muon0/Run2023D-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 97 | 100,211,533 | 126 | 100,291,308 (`Run2023D-22Sep2023_v1-v1`) |
| `/Muon0/Run2023D-NanoAODv15_v2-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 21 | 21,462,916 | 27 | 21,462,916 (`Run2023D-22Sep2023_v2-v1`) |
| `/Muon1/Run2023D-NanoAODv15-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 97 | 100,281,976 | 127 | 100,281,976 (`Run2023D-22Sep2023_v1-v1`) |
| `/Muon1/Run2023D-NanoAODv15_v2-v1/NANOAOD` | Muon | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 18 | 21,463,645 | 27 | 21,463,645 (`Run2023D-22Sep2023_v2-v1`) |
| `/EGamma0/Run2023D-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 101 | 105,892,646 | 165 | 105,892,646 (`Run2023D-22Sep2023_v1-v1`) |
| `/EGamma0/Run2023D-NanoAODv15_v2-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 25 | 22,657,211 | 36 | 22,657,211 (`Run2023D-22Sep2023_v2-v1`) |
| `/EGamma1/Run2023D-NanoAODv15-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 98 | 105,850,543 | 165 | 105,824,276 (`Run2023D-22Sep2023_v1-v1`) |
| `/EGamma1/Run2023D-NanoAODv15_v2-v1/NANOAOD` | EGamma | CMSSW_15_0_15_patch4 | 150X_dataRun3_v5 | 23 | 22,653,287 | 36 | 22,653,287 (`Run2023D-22Sep2023_v2-v1`) |

### 7.2 nanov15 MC configs

### nanov15/Run3Summer22


**nanov15/Run3Summer22_mc_dy_lo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 63 | 73,131,170 | 132 |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | DYJets | 17380.0 | S2 | 108 | 151,244,141 | 178 |

**nanov15/Run3Summer22_mc_dy_nlo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 90 | 93,967,480 | 181 |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | DYJets | 20950.0 | S3 | 75 | 97,526,817 | 120 |

**nanov15/Run3Summer22 non-DY background block** (17 datasets; missing in v15: TTLNu-EWK, WWW, WWZ, WZZ, ZZZ; t-channel = inclusive 4FS samples)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2/NANOAODSIM` | tt_tW | 96.978 | S4 | 32 | 23,802,613 | 77 |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 4 | 2,387,056 | 7 |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 4 | 2,327,688 | 7 |
| `/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 145.0 | S6 | 5 | 2,973,675 | 7 |
| `/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 87.2 | S7 | 2 | 1,433,215 | 3 |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 2.278 | S8 | 3 | 1,271,280 | 3 |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 1.43 | S9 | 2 | 785,520 | 2 |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v2/NANOAODSIM` | Nonprompt | 404.537 | S10 | 83 | 65,851,885 | 218 |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 9141 | S11 | 11 | 11,769,862 | 19 |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 2931 | S12 | 10 | 9,283,334 | 17 |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 864.6 | S13 | 12 | 8,221,862 | 18 |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Nonprompt | 417.8 | S14 | 2 | 1,463,885 | 4 |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Other | 0.08646 | S15 | 2 | 399,000 | 2 |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv15-mg35x_150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Other | 0.2502 | S16 | 6 | 2,056,905 | 9 |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Other | 80.23 | S15 | 16 | 15,405,496 | 28 |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Other | 29.1 | S15 | 9 | 7,479,528 | 14 |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22NanoAODv15-150X_mcRun3_2022_realistic_v1-v1/NANOAODSIM` | Other | 12.75 | S15 | 2 | 1,181,750 | 2 |

### nanov15/Run3Summer22EE


**nanov15/Run3Summer22EE_mc_dy_lo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 210 | 264,413,000 | 449 |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | DYJets | 17380.0 | S2 | 318 | 503,100,561 | 542 |

**nanov15/Run3Summer22EE_mc_dy_nlo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 284 | 335,241,916 | 610 |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | DYJets | 20950.0 | S3 | 250 | 354,056,365 | 402 |

**nanov15/Run3Summer22EE non-DY background block** (17 datasets; missing in v15: TTLNu-EWK, WWW, WWZ, WZZ, ZZZ; t-channel = inclusive 4FS samples)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v3/NANOAODSIM` | tt_tW | 96.978 | S4 | 108 | 74,184,271 | 231 |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 11 | 8,065,364 | 22 |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 12 | 8,260,009 | 23 |
| `/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 145.0 | S6 | 12 | 10,178,237 | 23 |
| `/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 87.2 | S7 | 7 | 5,185,208 | 12 |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 2.278 | S8 | 6 | 4,363,850 | 10 |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 1.43 | S9 | 4 | 2,762,668 | 6 |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v2/NANOAODSIM` | Nonprompt | 404.537 | S10 | 328 | 265,200,599 | 845 |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 9141 | S11 | 43 | 42,695,566 | 64 |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 2931 | S12 | 34 | 36,349,344 | 64 |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 864.6 | S13 | 29 | 27,828,446 | 57 |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Nonprompt | 417.8 | S14 | 7 | 4,906,634 | 13 |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Other | 0.08646 | S15 | 4 | 1,400,000 | 6 |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv15-mg35x_150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Other | 0.2502 | S16 | 13 | 7,214,273 | 30 |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Other | 80.23 | S15 | 51 | 53,112,080 | 92 |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Other | 29.1 | S15 | 28 | 26,722,782 | 47 |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv15-150X_mcRun3_2022_realistic_postEE_v1-v1/NANOAODSIM` | Other | 12.75 | S15 | 5 | 4,043,040 | 7 |

### nanov15/Run3Summer23


**nanov15/Run3Summer23_mc_dy_lo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v3/NANOAODSIM` | DYJets | 6340.2 | S1 | 124 | 127,692,949 | 223 |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | DYJets | 17380.0 | S2 | 190 | 292,020,978 | 326 |

**nanov15/Run3Summer23_mc_dy_nlo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | DYJets | 6340.2 | S1 | 192 | 205,068,372 | 383 |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1_ext1-v1/NANOAODSIM` | DYJets | 20950.0 | S3 | 137 | 203,085,305 | 238 |

**nanov15/Run3Summer23 non-DY background block** (17 datasets; missing in v15: TTLNu-EWK, WWW, WWZ, WZZ, ZZZ; t-channel = inclusive 4FS samples)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v2/NANOAODSIM` | tt_tW | 96.978 | S4 | 60 | 48,104,000 | 153 |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 6 | 4,985,000 | 14 |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 6 | 4,907,000 | 14 |
| `/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 145.0 | S6 | 7 | 5,908,000 | 14 |
| `/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 87.2 | S7 | 4 | 2,878,000 | 7 |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 2.278 | S8 | 3 | 2,588,000 | 6 |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 1.43 | S9 | 3 | 1,600,000 | 4 |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v2/NANOAODSIM` | Nonprompt | 404.537 | S10 | 199 | 149,503,000 | 481 |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 9141 | S11 | 22 | 23,085,414 | 35 |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 2931 | S12 | 21 | 20,267,771 | 36 |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 864.6 | S13 | 19 | 15,887,453 | 33 |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Nonprompt | 417.8 | S14 | 5 | 2,962,673 | 8 |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Other | 0.08646 | S15 | 3 | 800,000 | 4 |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv15-mg35x_150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Other | 0.2502 | S16 | 9 | 3,935,565 | 17 |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Other | 80.23 | S15 | 32 | 33,507,000 | 59 |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Other | 29.1 | S15 | 17 | 16,770,000 | 30 |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23NanoAODv15-150X_mcRun3_2023_realistic_v1-v1/NANOAODSIM` | Other | 12.75 | S15 | 4 | 2,517,000 | 4 |

### nanov15/Run3Summer23BPix


**nanov15/Run3Summer23BPix_mc_dy_lo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v2/NANOAODSIM` | DYJets | 6340.2 | S1 | 59 | 70,099,933 | 122 |
| `/DYto2L-4Jets_MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | DYJets | 17380.0 | S2 | 107 | 149,169,635 | 166 |

**nanov15/Run3Summer23BPix_mc_dy_nlo_inc.json** – DY block (2 datasets)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | DYJets | 6340.2 | S1 | 86 | 95,217,757 | 178 |
| `/DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1_ext1-v1/NANOAODSIM` | DYJets | 20950.0 | S3 | 71 | 101,317,595 | 119 |

**nanov15/Run3Summer23BPix non-DY background block** (17 datasets; missing in v15: TTLNu-EWK, WWW, WWZ, WZZ, ZZZ; t-channel = inclusive 4FS samples)

| DAS path | physics_group | xsec [pb] | src | files | events | GB |
|---|---|---|---|---|---|---|
| `/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v2/NANOAODSIM` | tt_tW | 96.978 | S4 | 36 | 24,556,000 | 78 |
| `/TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 3 | 2,479,000 | 7 |
| `/TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | tt_tW | 4.61475 | S5 | 4 | 2,488,000 | 7 |
| `/TBbarQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 145.0 | S6 | 6 | 2,954,000 | 7 |
| `/TbarBQ_t-channel_4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 87.2 | S7 | 3 | 1,488,000 | 3 |
| `/TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 2.278 | S8 | 3 | 1,288,000 | 3 |
| `/TbarBtoLminusNuB-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 1.43 | S9 | 2 | 784,000 | 2 |
| `/TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v2/NANOAODSIM` | Nonprompt | 404.537 | S10 | 81 | 63,875,000 | 205 |
| `/WtoLNu-4Jets_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 9141 | S11 | 11 | 11,369,658 | 17 |
| `/WtoLNu-4Jets_2J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 2931 | S12 | 9 | 9,459,992 | 17 |
| `/WtoLNu-4Jets_3J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 864.6 | S13 | 10 | 7,718,351 | 16 |
| `/WtoLNu-4Jets_4J_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Nonprompt | 417.8 | S14 | 4 | 1,436,944 | 4 |
| `/TTLL_MLL-50_TuneCP5_13p6TeV_amcatnlo-pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Other | 0.08646 | S15 | 1 | 400,000 | 2 |
| `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv15-mg35x_150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Other | 0.2502 | S16 | 4 | 1,981,391 | 8 |
| `/WW_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Other | 80.23 | S15 | 17 | 16,545,000 | 29 |
| `/WZ_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Other | 29.1 | S15 | 9 | 8,379,000 | 15 |
| `/ZZ_TuneCP5_13p6TeV_pythia8/Run3Summer23BPixNanoAODv15-150X_mcRun3_2023_realistic_postBPix_v1-v1/NANOAODSIM` | Other | 12.75 | S15 | 2 | 1,254,000 | 2 |

| code | source (nanov15 tables) |
|---|---|
| S1 | existing WrCoffea configs (6340.2 = 3 x 2113.4/flavour, NNLO-type normalisation; XSDB GenXsecAnalyzer values: NLO 6688, LO 5467 [FLAF yaml]) |
| S2 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-4Jets_MLL-10to50, modifiedOn 2024-04-03; raw LO value, no k-factor) |
| S3 | XSDB value as recorded in cms-flaf/FLAF config/crossSections13p6TeV.yaml (DYto2L-2Jets_MLL-10to50, modifiedOn 2023-03-26) |
| S4 | existing WrCoffea configs (= 923.6 pb NNLO+NNLL [LHCPhysics/TtbarNNLO] x 0.105) |
| S5 | existing WrCoffea configs (= 87.9/2 pb [LHCPhysics/SingleTopNNLORef tW] x 0.105) |
| S6 | LHCPhysics/SingleTopNNLORef NNLO t-channel (top), inclusive in W decay; used because no NanoAODv15 of the LNu-filtered sample exists (v12 drafts use TBbarQtoLNu-t-channel-4FS with XSDB 38.6) |
| S7 | LHCPhysics/SingleTopNNLORef NNLO t-channel (antitop), inclusive in W decay; used because no NanoAODv15 of the LNu-filtered sample exists (v12 drafts use TbarBQtoLNu-t-channel-4FS with XSDB 23.34) |
| S8 | existing WrCoffea configs (XSDB-style value; FLAF uses 7.244 x 0.3259 = 2.361) |
| S9 | existing WrCoffea configs (XSDB-style value; FLAF uses 4.534 x 0.3259 = 1.478) |
| S10 | existing WrCoffea configs (= 923.6 x 0.438) |
| S11 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 9084) |
| S12 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 2925) |
| S13 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 861.7) |
| S14 | existing WrCoffea configs (2024 config = XSDB Summer24 entry per cmsdb; XSDB Summer22 entry in FLAF: 416.5) |
| S15 | existing WrCoffea configs = XSDB [FLAF yaml] |
| S16 | existing WrCoffea configs (2022/2023 configs; XSDB [FLAF yaml, retrieved 2026-09-24] = 0.2505 +- 0.001043) |

## 8. Current configs: INVALID entries and entries not carried into the drafts

### 8.1 DAS status audit of the 1230 distinct paths in the current configs (65 INVALID, all others VALID)
| config file | DAS path (status) |
|---|---|
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-1J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-200to400_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-400to600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-40to100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM` (INVALID) |
| 2024/RunIII2024Summer24/archived/RunIII2024Summer24_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_Bin-2J-MLL-50-PTLL-600_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_njet.json | `/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_njet.json | `/DYto2L-2Jets_MLL-50_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v3/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_njet.json | `/DYto2L-2Jets_MLL-50_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-100to200_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-200to400_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-400to600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_1J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23/archived/Run3Summer23_mc_dy_nlo_ptll.json | `/DYto2L-2Jets_MLL-50_PTLL-600_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/Run3Summer23BPix_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22EE/archived/Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/Run3Summer22_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2022/Run3Summer22/archived/Run3Summer22_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/Run3Summer23BPix_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` (INVALID) |
| 2023/Run3Summer23BPix/archived/Run3Summer23BPix_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` (INVALID) |

### 8.2 Current (non-archived) MC entries that are not in the drafts
| current config | DAS path | status | events | why not in the drafts |
|---|---|---|---|---|
| Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | INVALID | 10,411,115 | superseded version (INVALID on DAS) |
| Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v1/NANOAODSIM` | INVALID | 142,936,007 | superseded version (INVALID on DAS) |
| Run3Summer23BPix_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | INVALID | 36,265,983 | superseded version (INVALID on DAS) |
| Run3Summer22EE_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | INVALID | 98,847,366 | superseded version (INVALID on DAS) |
| Run3Summer22_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` | INVALID | 29,650,452 | superseded version (INVALID on DAS) |
| Run3Summer23BPix_mc_dy_lo_ht.json | `/DYto2L-4Jets_MLL-50to120_HT-70to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v2/NANOAODSIM` | INVALID | 29,460,714 | superseded version (INVALID on DAS) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/TQbartoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 32,560,860 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/TQbartoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 9,082,425 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/TQbartoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 9,253,000 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/TQbartoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | VALID | 18,672,000 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v4/NANOAODSIM` | VALID | 335,985 | old-MadGraph TTLNu-1Jets (v5-v1 style, ~0.1-0.5M events): replaced by the mg35x regeneration (20x more events; the only version available in 2024) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` | VALID | 111,308 | old-MadGraph TTLNu-1Jets (v5-v1 style, ~0.1-0.5M events): replaced by the mg35x regeneration (20x more events; the only version available in 2024) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 256,078 | old-MadGraph TTLNu-1Jets (v5-v1 style, ~0.1-0.5M events): replaced by the mg35x regeneration (20x more events; the only version available in 2024) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | VALID | 460,305 | old-MadGraph TTLNu-1Jets (v5-v1 style, ~0.1-0.5M events): replaced by the mg35x regeneration (20x more events; the only version available in 2024) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/TbarQtoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 19,529,636 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/TbarQtoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 5,719,520 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/TbarQtoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 5,767,000 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/TbarQtoLNu-t-channel_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v2/NANOAODSIM` | VALID | 11,546,000 | 5FS t-channel (TQbartoLNu/TbarQtoLNu): replaced by the 4FS LNu samples used in 2024; also no 2024 equivalent exists; xsec 130.0/78.56 looks like the undecayed inclusive value (see xsec section) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | VALID | 372,212,243 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | VALID | 100,572,216 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | VALID | 94,723,033 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | VALID | 185,590,464 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 1,710,744 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 477,881 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 522,080 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM` | VALID | 1,031,962 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 1,629,239 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 514,983 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | VALID | 501,900 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | VALID | 920,162 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | VALID | 48,425,013 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v1/NANOAODSIM` | VALID | 13,533,118 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | VALID | 13,133,728 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | VALID | 26,674,759 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v3/NANOAODSIM` | VALID | 461,210,013 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v3/NANOAODSIM` | VALID | 146,630,714 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM` | VALID | 135,155,699 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM` | VALID | 274,569,490 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 5,427,609 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 1,432,995 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 1,431,810 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-0to120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v5/NANOAODSIM` | VALID | 2,874,681 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 7,266,679 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 2,132,100 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 2,087,902 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-100to400_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | VALID | 4,175,521 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 1,717,045 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 486,619 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 498,831 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-1500to2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM` | VALID | 901,686 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 1,770,386 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 489,635 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 483,512 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-2500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v4/NANOAODSIM` | VALID | 1,027,603 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 1,675,012 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 506,334 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | VALID | 510,857 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-400to800_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | VALID | 1,042,427 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 14,999,600 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 4,532,184 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v2/NANOAODSIM` | VALID | 4,335,809 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-40to100_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v5/NANOAODSIM` | VALID | 9,109,151 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22EE_mc_dy_lo_ht.json, Run3Summer22EE_mc_dy_lo_inc.json, Run3Summer22EE_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM` | VALID | 1,702,529 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer22_mc_dy_lo_ht.json, Run3Summer22_mc_dy_lo_inc.json, Run3Summer22_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer22NanoAODv12-130X_mcRun3_2022_realistic_v5-v2/NANOAODSIM` | VALID | 483,978 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23BPix_mc_dy_lo_ht.json, Run3Summer23BPix_mc_dy_lo_inc.json, Run3Summer23BPix_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v6-v3/NANOAODSIM` | VALID | 494,697 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |
| Run3Summer23_mc_dy_lo_ht.json, Run3Summer23_mc_dy_lo_inc.json, Run3Summer23_mc_dy_nlo_inc.json | `/WtoLNu-4Jets_MLNu-120_HT-800to1500_TuneCP5_13p6TeV_madgraphMLM-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v15-v3/NANOAODSIM` | VALID | 915,044 | HT/MLNu-binned W+jets: replaced by the jet-binned WtoLNu-4Jets_1J..4J scheme of the 2024 config (still VALID, can be swapped back) |

### 8.3 HT×M(lν)-binned W+jets in RunIII2024Summer24 (available, not used – 2024-style equivalent of the 2022/2023 HT-binned W scheme)
| DAS path | status | events |
|---|---|---|
| `/WtoLNu-4Jets_Bin-HT-100to400-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 375,281,609 |
| `/WtoLNu-4Jets_Bin-HT-100to400-MLNu-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 22,905,968 |
| `/WtoLNu-4Jets_Bin-HT-1500to2500-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 15,480,230 |
| `/WtoLNu-4Jets_Bin-HT-1500to2500-MLNu-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 15,147,485 |
| `/WtoLNu-4Jets_Bin-HT-2500-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 13,634,921 |
| `/WtoLNu-4Jets_Bin-HT-2500-MLNu-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 14,086,002 |
| `/WtoLNu-4Jets_Bin-HT-400to800-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 124,386,168 |
| `/WtoLNu-4Jets_Bin-HT-400to800-MLNu-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 14,880,703 |
| `/WtoLNu-4Jets_Bin-HT-40to100-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 361,395,701 |
| `/WtoLNu-4Jets_Bin-HT-40to100-MLNu-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 43,791,645 |
| `/WtoLNu-4Jets_Bin-HT-800to1500-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 15,283,623 |
| `/WtoLNu-4Jets_Bin-HT-800to1500-MLNu-120_TuneCP5_13p6TeV_madgraphMLM-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v2/NANOAODSIM` | VALID | 12,681,807 |

## 9. Signal

| era | datasets | processed-dataset versions | events (min–max per point) | size |
|---|---|---|---|---|
| Run3Summer22 | 200 (100 `WRtoNEltoElElJJ` + 100 `WRtoNMutoMuMuJJ`) | 195 × `v5-v2`, 4 × `v5-v3`, 1 × `v5-v4` | 13.82 M (42 k–70 k) | 60 GB |
| Run3Summer22EE | 200 | 195 × `postEE_v6-v2`, 3 × `v6-v3`, 2 × `v6-v4` | 41.74 M (199 k–210 k) | 156 GB |
| Run3Summer23 | 200 | 193 × `v15-v2`, 5 × `v15-v3`, 2 × `v15-v4` | 13.86 M (61 k–70 k) | 53 GB |
| Run3Summer23BPix | 200 | 198 × `postBPix_v6-v2`, 2 × `v6-v4` | 7.00 M (34 k–35 k) | 28 GB |
| RunIII2024Summer24 | 200 (placeholder = Run3Summer23BPix datasets, copied verbatim) | as 23BPix | 7.00 M | 28 GB |

* Mass grid (identical in all four eras, both channels): M_WR ∈ {2000, 4000, 6000, 8000} GeV with M_N = 100, 300, …, M_WR − 100 GeV (10 + 20 + 30 + 40 = 100 points per channel). The campaigns also contain 438 `WRtoNTautoTauTauJJ_WR<M>_N<N>` datasets (different mass grid, `WR` instead of `MWR` in the name) which are not used.
* The chosen DAS sets are identical to the current signal configs (all VALID); 2–8 superseded INVALID `-v2` versions per era exist for the points that were re-produced as `-v3/-v4`. `WRtoNMutoMuMuJJ_MWR6000_N3500` in Run3Summer23 has two VALID versions (`v15-v1` 61 k events and `v15-v2` 70 k); `-v2` is kept.
* **RunIII2024Summer24: no WR signal on DAS** (`/WRto*/RunIII2024Summer24*/NANOAODSIM` → 0, any status); **NanoAODv15: no WR signal for any era**; no 2025 signal.
* Cross sections per point are copied from the current configs (identical across eras, e.g. `MWR2000_N100` 0.1668 pb); their provenance was not re-derived.
* Full tables: `configs_run3/REPORT_signal.md`.

## 10. Cross-section check

Public references (XSDB and the CMS twikis are not reachable, see §3): LHCPhysics `TtbarNNLO` (tt̄ 923.6 pb NNLO+NNLL, 13.6 TeV), LHCPhysics `SingleTopNNLORef` (t-channel t 145.0, t̄ 87.2 pb NNLO; s-channel 11.778 pb; tW 87.9 pb NLO+NNLL), `cms-flaf/FLAF config/crossSections13p6TeV.yaml` (XSDB values with retrieval dates), `uhh-cms/cmsdb` (`ewk.py`, `top.py`). BR(W→ℓν, 3 flavours) = 0.3259 (PDG 2019 0.1086 × 3).

| process (drafts) | value used [pb] | reference | Δ / comment |
|---|---|---|---|
| TTto2L2Nu | 96.978 | 923.6 × 0.3259² = 98.09 (FLAF) | −1.1 % (configs use BR² = 0.105) |
| TTtoLNu2Q | 404.537 | 923.6 × 2 × 0.6741 × 0.3259 = 405.8 | −0.3 % |
| TWminusto2L2Nu / TbarWplusto2L2Nu | 4.61475 each | 87.9 × 0.4995 × 0.1062 = 4.663 / 87.9 × 0.5005 × 0.1062 = 4.672 (FLAF) | −1.0 % / −1.2 % |
| TBbarQtoLNu-t-channel-4FS | 38.6 | XSDB 38.6 (FLAF, 2025-09-22); NNLO × BR = 145.0 × 0.3259 = 47.25 | identical to XSDB; NNLO×BR would be +22 % (FLAF marks this "FIXME") |
| TbarBQtoLNu-t-channel-4FS | 23.34 | XSDB 23.34; NNLO × BR = 87.2 × 0.3259 = 28.42 | identical to XSDB; +22 % for NNLO×BR |
| TQbartoLNu-t-channel / TbarQtoLNu-t-channel (5FS, current 2022/2023 configs, not in drafts) | 130.0 / 78.56 | NNLO inclusive 145.0 / 87.2; × BR = 47.25 / 28.42 | **values are undecayed-inclusive-sized (ratio 0.90 to NNLO) although the samples are LNu-only → probable missing BR(W→ℓν) factor (×3.07 too high)**; verify against the sample's GenXsecAnalyzer |
| TBbartoLplusNuBbar / TbarBtoLminusNuB s-channel 4FS | 2.278 / 1.43 | FLAF: 7.244 × 0.3259 = 2.361 / 4.534 × 0.3259 = 1.478 (NNLO split by XSDB ratio) | −3.5 % / −3.3 % (XSDB NLO-type values) |
| WtoLNu-4Jets 1J / 2J / 3J / 4J | 9141 / 2931 / 864.6 / 417.8 | XSDB Summer24 entries (cmsdb) 9141 / 2931 / 864.6 / 417.8; XSDB Summer22 entries (FLAF) 9084 / 2925 / 861.7 / 416.5 | 2022/2023 drafts use the Summer24 numbers (+0.6 % / +0.2 % / +0.3 % / +0.3 % w.r.t. the Summer22 XSDB entries) |
| WtoLNu-4Jets MLNu-0to120 HT 40to100 / 100to400 / 400to800 / 800to1500 / 1500to2500 (current 2022/2023 configs) | 4264 / 1637 / 60.07 / 6.223 / 0.4498 | XSDB (FLAF) 4254 / 1626 / 59.99 / 6.23 / 0.4477 | ≤0.7 % |
| TTLL_MLL-50 | 0.08646 | XSDB 0.08646 (FLAF, 2026-09-24) | exact |
| TTLNu-1Jets | 0.2502 | XSDB 0.2505 ± 0.001043 (FLAF, 2026-09-24) | −0.1 % |
| TTLNu-EWK | 0.01769 | XSDB 0.01769 (FLAF) | exact |
| WW / WZ / ZZ | 80.23 / 29.1 / 12.75 | XSDB 80.23 / 29.1 / 12.75 (FLAF); cmsdb GenXsecAnalyzer 80.22 / 29.17 | exact (LO pythia8 values; NNLO WW ≈ 122 pb, ZZ nNNLO 24.97 pb per cmsdb – normalisation choice) |
| WWW / WWZ / WZZ / ZZZ | 0.2328 / 0.1851 / 0.06206 / 0.01591 | XSDB (FLAF) identical | exact |
| DY inclusive (LO 4Jets and NLO 2Jets) | 6340.2 (= 3 × 2113.4 per flavour) | XSDB: NLO 6688 ± 84, LO 5467 ± 13 (FLAF); MATRIX NNLO QCD + NLO EW per flavour 2094.2 (cmsdb) → 6282.6 | −5.2 % vs XSDB NLO, +0.9 % vs MATRIX; consistent across eras; origin of 2113.4 not documented in the repo |
| DY powheg MLL bins (9 × e, 9 × μ) | 2219 … 3.519e-08 | XSDB (FLAF, 2024-04-03) | exact |
| DY MLL-10to50: LO 4Jets / NLO 2Jets / powheg e,μ | 17380 / 20950 / 6744 (new) | XSDB (FLAF) | raw XSDB values, no k-factor (the inclusive LO/NLO samples are normalised to 6340.2, i.e. k = 1.16 for LO, 0.95 for NLO) |
| DY NLO 0J / 1J / 2J (2022/2023) | 5365 / 1015 / 382.8 | XSDB 5378 / 1017 / 385.5 (FLAF; cmsdb 5378 for 0J) | −0.2 % / −0.2 % / −0.7 % |
| DY NLO 0J / 1J / 2J, 2024 flavour-split (`DYto2E-…`, `DYto2Mu-…`) | 5382 / 972.3, 975.5 / 372.4, 374 | per-flavour expectation = all-flavour / 3 = 1793 / 339 / 128.5 | **×3 inconsistency inside the 2024 configs** (the 2024 inclusive flavour-split entries use 2113.4 = 6340.2/3; Σ(2024 NJ per flavour) = 6727 vs 2113); kept in the draft with a `note`, must be verified |
| DY NLO nested PTLL-100 / 200 / 400 / 600 | 108.257 / 11.157 / 0.61601 / 0.08411 (new) | FLAF (sum of XSDB `PTLL-<bin>_{1J,2J}` above threshold) | nested, overlapping samples |
| DY NLO PTLL-<bin>_{1J,2J} (archived configs, INVALID) | 472.0, 176.2, 45.66, 54.09, 3.371, 7.16, 0.117, 0.4354, 0.01405, 0.07061 | XSDB (FLAF) 475.3, 179.3, 45.42, … | ≤2 %, irrelevant (samples INVALID) |
| DY LO 1J / 2J / 3J / 4J | 973.1 / 312.4 / 93.93 / 45.43 (new) | XSDB (FLAF, cmsdb) | raw LO values |
| DY LO PTLL 40to100 / 100to200 / 200to400 / 400to600 / 600 | 403.7 / 58.46 / 6.678 / 0.3833 / 0.06843 (new) | XSDB via cmsdb (`dy_m50toinf_pt*`, LO) | raw LO values |
| DY LO HT bins (14) | 316.7 … 0.0001247 (current configs) | no public reference found (XSDB unreachable; cmsdb/FLAF carry only 13 TeV HT bins) | reused from the archived configs as instructed, unverified |
| WR signal | per point from current configs | – | not re-derived |

Discrepancies to act on: (i) 2024 `nlo_njet` per-flavour values (×3); (ii) 5FS t-channel 130.0/78.56 in the current 2022/2023 configs (BR); (iii) choose one normalisation convention for the low-mass/binned LO DY (raw XSDB) vs the inclusive (6340.2); (iv) tt̄/tW BR convention 0.105 vs 0.1062 (1 %); (v) W jet-binned: Summer22 vs Summer24 XSDB entries differ by ≤0.6 % – one value per process is used in all eras, as required.

## 11. Unresolved / decisions left to the user

1. Whether to adopt the NanoAODv15 set for 2022/2023 (data complete; MC complete only for the inclusive DY families and 17 of 22 backgrounds; no signal).
2. Whether to keep the HT-binned W+jets (12 samples, VALID in all eras incl. 2024) instead of the jet-binned scheme used in the drafts.
3. Whether to include the `MLL-10to50` companions (large: 2024 `DYto2E/Mu-4Jets_Bin-MLL-10to50` 404 + 466 M events, `DYto2E/Mu_Bin-MLL-10to50` powheg 348 + 342 M events) – marked `optional` in the `note`.
4. The nested pT(ll) samples need a stitching/filter scheme (FLAF: `LHE_NpNLO >= 1` and no `40to100` sample) before they can be used; the archived `nlo_ptll` configs are unusable (all samples INVALID).
5. `genEventSumw` must be computed for all `NEW` entries (and combined for nominal + `ext1`).
6. 2022B/2023B/2024B golden runs (§5.3) are not in any config; 2023B could be added to Run3Summer23 (same conditions), 2022B to Run3Summer22.
7. The 2025 era name `RunIII2025Summer25` is a placeholder; the `year` field of the two 2022 signal configs is an integer (as in the current files) while all other configs use strings.
8. Cross-section items (i)–(iv) of §10; the HT-binned DY and signal cross sections could not be checked against a public source.
9. RunIII2024Summer24 signal does not exist; the placeholder (23BPix samples) remains.

## 12. Appendix – query list and references

* DBS REST: `datasets?dataset=<pattern>&dataset_access_type=*&detail=true`; `filesummaries?dataset=`; `outputconfigs?dataset=`; `datasetparents?dataset=`; `runs?dataset=`. Patterns: `/<P>/Run3Summer22NanoAOD*/NANOAODSIM`, `/<P>/Run3Summer22EENanoAOD*/NANOAODSIM`, `/<P>/Run3Summer23NanoAOD*/NANOAODSIM`, `/<P>/Run3Summer23BPixNanoAOD*/NANOAODSIM`, `/<P>/RunIII2024Summer24NanoAOD*/NANOAODSIM`, `/<P>/*2025*/NANOAODSIM`, `/<P>/*Winter25*/NANOAODSIM`, `/<P>/*Summer25*/NANOAODSIM` for P ∈ {`DYto2*`, `DY*Jets*`, `TTto2L2Nu*`, `TTtoLNu2Q*`, `TTto4Q*`, `TWminus*`, `TbarWplus*`, `TBbar*`, `TbarB*`, `TQbar*`, `TbarQ*`, `WtoLNu*`, `WW*`, `WZ*`, `ZZ*`, `WWW*`, `WWZ*`, `WZZ*`, `ZZZ*`, `TTLL*`, `TTLNu*`, `TTW*`, `TTZ*`, `WRtoN*`, `WRto*`}; `/*/Run3Winter25NanoAOD*/NANOAODSIM`; `/TTto2L2Nu*/RunIII2024Summer24NanoAODv15*realistic_v3*/NANOAODSIM`, `…realistic_v4*`, `/*/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v3*/NANOAODSIM`, `…_v4*`, `/TTto2L2Nu*/RunIII2024Summer24NanoAODv16*/NANOAODSIM`; `/WRtoN*/*NanoAODv15*/NANOAODSIM`; data patterns as in §3.
* References: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO ; https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopNNLORef ; https://github.com/cms-flaf/FLAF (config/crossSections13p6TeV.yaml, config/Run3_2022/datasets.yaml, PR #336) ; https://github.com/cms-flaf/H_mumu/pull/74 ; https://github.com/uhh-cms/cmsdb (cmsdb/processes/ewk.py, top.py) ; https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions25/ ; https://cms-pdmv.gitbook.io/project/mccontact/rules-for-run3-2024-dataset-names (naming rules). Not reachable: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MATRIXCrossSectionsat13p6TeV (SSO), https://twiki.cern.ch/twiki/bin/view/CMS/StandardModelCrossSectionsat13p6TeV (404/SSO), XSDB.

## 13. Other DY-type / W samples seen on DAS and deliberately not configured

VALID samples matching `DYto2L/E/Mu*` or `WtoLNu-4Jets[_Bin]-MLNu-*` in the era campaigns that are neither in the drafts nor systematic-variation copies. The powheg MiNNLO+photos mass-binned family (`DYto2E/Mu[_Bin]-MLL-{50to130,130to200,200to400,400to600,600to800,800to1000,1000to1500,1500to2000,2000to4000,4000to6000,6000to13600}_powhegMINNLO-pythia8-photos`, some bins still PRODUCTION) is a complete NNLO alternative to the NLO powheg MLL family in every era, but no public cross sections were found for it, so it is listed only.
| era | kind | primary dataset (processed string) | events |
|---|---|---|---|
| Run3Summer22 | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 14,940,000 |
| Run3Summer22 | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 14,940,000 |
| Run3Summer22 | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 7,325,014 |
| Run3Summer22 | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 7,325,014 |
| Run3Summer22 | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_v5-v2`) | 4,488,000 |
| Run3Summer22 | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_v5-v2`) | 4,488,000 |
| Run3Summer22 | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_v5-v2`) | 4,282,000 |
| Run3Summer22 | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_v5-v2`) | 4,282,000 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-120to200_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 34,500,663 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-1500to2500_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 946,791 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-200to400_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 4,564,839 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-2500to4000_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 892,866 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-4000to6000_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 908,947 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-400to800_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 1,010,617 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-6000_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 999,715 |
| Run3Summer22 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-800to1500_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_v5_ext1-v2`) | 1,009,338 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-120to200_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 4,996,988 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-1500to2500_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 1,986,500 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-200to400_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 2,986,500 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-2500to4000_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 999,251 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-4000to6000_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 990,988 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-400to800_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 2,995,500 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-50to120_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v3`) | 9,987,936 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-6000_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 489,500 |
| Run3Summer22 | keepMDSHits special re-generation | `DYto2Mu_MLL-800to1500_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 1,988,750 |
| Run3Summer22 | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_v5-v3`) | 100,881,508 |
| Run3Summer22 | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_v5-v3`) | 100,881,508 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-1000to1500_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 1,091,504 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-1500to2000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-2000to4000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 211,662 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-4000to6000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 68,628 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-400to600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-6000to13600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 68,115 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-600to800_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 69,281 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-1000to1500_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-10to50_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 12,972,997 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 1,097,196 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-1500to2000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-2000to4000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 69,305 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 219,278 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-4000to6000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-400to600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-50to130_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 39,712,884 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-6000to13600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 68,576 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-600to800_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 68,508 |
| Run3Summer22 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_v5-v2`) | 70,000 |
| Run3Summer22 | pythia8-only / NNPDF31 variants | `DYto2E_M-50_NNPDF31-powheg-pythia8` (`130X_mcRun3_2022_realistic_v5-v2`) | 27,528,904 |
| Run3Summer22EE | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 44,843,999 |
| Run3Summer22EE | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 44,843,999 |
| Run3Summer22EE | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 24,957,222 |
| Run3Summer22EE | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 24,957,222 |
| Run3Summer22EE | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 15,497,000 |
| Run3Summer22EE | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 15,497,000 |
| Run3Summer22EE | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 15,040,000 |
| Run3Summer22EE | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 15,040,000 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-120to200_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 123,093,091 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-1500to2500_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v3`) | 3,302,117 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-200to400_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 16,353,112 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-2500to4000_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 3,351,427 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-4000to6000_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 3,222,531 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-400to800_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 3,598,198 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-6000_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 3,668,638 |
| Run3Summer22EE | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-800to1500_madgraphMLM-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v4`) | 3,295,801 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-120to200_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 14,712,750 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-1500to2500_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 5,969,127 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-200to400_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 8,781,750 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-2500to4000_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 2,952,000 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-4000to6000_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 2,925,000 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-400to800_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 8,985,000 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-50to120_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 29,755,600 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-6000_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 1,482,750 |
| Run3Summer22EE | keepMDSHits special re-generation | `DYto2Mu_MLL-800to1500_keepMDSHits_powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 5,859,942 |
| Run3Summer22EE | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v3`) | 195,727,615 |
| Run3Summer22EE | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v3`) | 195,727,615 |
| Run3Summer22EE | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6_ext1-v3`) | 150,148,133 |
| Run3Summer22EE | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6_ext1-v3`) | 150,148,133 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-1000to1500_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 250,000 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-10to50_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 43,634,971 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-1500to2000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 249,301 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-2000to4000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 249,325 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 726,626 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-1000to1500_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 250,000 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 3,567,212 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 736,485 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-400to600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 250,000 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-50to130_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 117,940,245 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-600to800_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 241,252 |
| Run3Summer22EE | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 248,566 |
| Run3Summer22EE | pythia8-only / NNPDF31 variants | `DYto2E_M-50_NNPDF31-powheg-pythia8` (`130X_mcRun3_2022_realistic_postEE_v6-v2`) | 95,080,708 |
| Run3Summer23 | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2023_realistic_v15-v2`) | 29,400,000 |
| Run3Summer23 | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2023_realistic_v15-v2`) | 29,400,000 |
| Run3Summer23 | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_v15-v2`) | 17,763,042 |
| Run3Summer23 | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_v15-v2`) | 17,763,042 |
| Run3Summer23 | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2023_realistic_v15-v2`) | 10,476,000 |
| Run3Summer23 | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2023_realistic_v15-v2`) | 10,476,000 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-120to200_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v5`) | 69,132,435 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-1500to2500_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v3`) | 1,878,810 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-200to400_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v5`) | 9,074,553 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-2500to4000_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v5`) | 1,872,530 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-4000to6000_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v5`) | 1,927,781 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-400to800_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v4`) | 1,935,706 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-6000_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v4`) | 1,865,656 |
| Run3Summer23 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-800to1500_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_v14-v4`) | 1,928,804 |
| Run3Summer23 | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_v15-v5`) | 202,880,249 |
| Run3Summer23 | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_v15-v5`) | 202,880,249 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-1000to1500_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 160,000 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 2,446,170 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-2000to4000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 155,986 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 500,000 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-400to600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 152,120 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-6000to13600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v3`) | 158,669 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 160,000 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v2`) | 2,460,000 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-1500to2000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 158,610 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 498,590 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-400to600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 160,000 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-50to130_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v2`) | 82,387,945 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-6000to13600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 156,440 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-600to800_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 158,594 |
| Run3Summer23 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_v15-v4`) | 156,495 |
| Run3Summer23BPix | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 14,810,000 |
| Run3Summer23BPix | EWK Z+2j (VBF-type) | `DYto2L-2Jets-EWK_MLL-50_madgraph-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 14,810,000 |
| Run3Summer23BPix | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 9,035,307 |
| Run3Summer23BPix | H->mumu window sample | `DYto2Mu-2Jets_MLL-105To160_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 9,035,307 |
| Run3Summer23BPix | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 6,000,000 |
| Run3Summer23BPix | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 6,000,000 |
| Run3Summer23BPix | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 5,984,000 |
| Run3Summer23BPix | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 5,984,000 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-120to200_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v4`) | 32,920,196 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-1500to2500_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v4`) | 933,675 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-200to400_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v4`) | 4,345,386 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-2500to4000_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v5`) | 1,015,183 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-4000to6000_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v4`) | 936,192 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-400to800_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v5`) | 990,330 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-6000_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v5`) | 974,688 |
| Run3Summer23BPix | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_MLNu-800to1500_madgraphMLM-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v2-v4`) | 955,236 |
| Run3Summer23BPix | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v6-v3`) | 101,353,986 |
| Run3Summer23BPix | other | `DYto2L-2Jets_MLL-4to10_amcatnloFXFX-pythia8` (`130X_mcRun3_2023_realistic_postBPix_v6-v3`) | 101,353,986 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 1,399,245 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-1500to2000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 87,927 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-2000to4000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 90,000 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-6000to13600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 87,400 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-600to800_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 87,828 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 90,000 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-1000to1500_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 88,554 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-130to200_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 1,394,000 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-1500to2000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 90,000 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-2000to4000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 85,854 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-200to400_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 270,000 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-4000to6000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 89,318 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-400to600_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 90,000 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-50to130_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v2`) | 45,761,975 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-600to800_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 89,311 |
| Run3Summer23BPix | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_MLL-800to1000_powhegMINNLO-pythia8-photos` (`130X_mcRun3_2023_realistic_postBPix_v6-v4`) | 90,000 |
| RunIII2024Summer24 | H->mumu window sample | `DYto2Mu-2Jets_Bin-MLL-105to160_amcatnloFXFX-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 92,601,022 |
| RunIII2024Summer24 | H->mumu window sample | `DYto2Mu-2Jets_Bin-MLL-105to160_amcatnloFXFX-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 92,601,022 |
| RunIII2024Summer24 | H->mumu window sample | `DYto2Mu-2Jets_Bin-MLL-105to160_amcatnloFXFX-pythia8` (`150X_mcRun3_2024_realistic_v2_ext1-v2`) | 102,858,186 |
| RunIII2024Summer24 | H->mumu window sample | `DYto2Mu-2Jets_Bin-MLL-105to160_amcatnloFXFX-pythia8` (`150X_mcRun3_2024_realistic_v2_ext1-v2`) | 102,858,186 |
| RunIII2024Summer24 | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_Bin-MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`150X_mcRun3_2024_realistic_v2-v4`) | 63,999,000 |
| RunIII2024Summer24 | Sherpa 5-jet NLO/LO merged | `DYto2E-5Jets-2NLO3LO_Bin-MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`150X_mcRun3_2024_realistic_v2-v4`) | 63,999,000 |
| RunIII2024Summer24 | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_Bin-MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`150X_mcRun3_2024_realistic_v2-v2`) | 63,812,000 |
| RunIII2024Summer24 | Sherpa 5-jet NLO/LO merged | `DYto2Mu-5Jets-2NLO3LO_Bin-MLL-40_TuneSherpaDef_13p6TeV_sherpaMEPS` (`150X_mcRun3_2024_realistic_v2-v2`) | 63,812,000 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-120to200_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v3`) | 342,435,331 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-1500to2500_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v3`) | 22,759,819 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-200to400_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v3`) | 43,683,629 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-2500to4000_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v3`) | 10,371,580 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-4000to6000_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v4`) | 9,113,967 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-400to800_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v3`) | 34,702,179 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-6000_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v4`) | 9,659,513 |
| RunIII2024Summer24 | W M(lnu)-binned (high-mass tails) | `WtoLNu-4Jets_Bin-MLNu-800to1500_madgraphMLM-pythia8` (`150X_mcRun3_2024_realistic_v2-v3`) | 18,785,532 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-1000to1500_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 983,148 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-130to200_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 14,761,420 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-1500to2000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 1,000,000 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-2000to4000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 999,271 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-200to400_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 2,991,420 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-4000to6000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 997,842 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-400to600_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 990,700 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-6000to13600_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 975,754 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-600to800_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 999,225 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2E_Bin-MLL-800to1000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 992,370 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-1000to1500_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 991,464 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-10to50_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 179,034,776 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-130to200_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 14,863,503 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-1500to2000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 963,293 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-2000to4000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 999,225 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-200to400_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 2,985,870 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-4000to6000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 996,927 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-400to600_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 998,426 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-50to130_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 497,239,329 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-6000to13600_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 972,131 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-600to800_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 999,228 |
| RunIII2024Summer24 | powheg MiNNLO+photos MLL-binned (NNLO) | `DYto2Mu_Bin-MLL-800to1000_powhegMINNLO-pythia8-photos` (`150X_mcRun3_2024_realistic_v2-v2`) | 979,772 |
| RunIII2024Summer24 | very low mass 1-10 GeV | `DYto2L_Bin-1J-MLL-1to10-PTLL-0to70_madgraph-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 4,963,992 |
| RunIII2024Summer24 | very low mass 1-10 GeV | `DYto2L_Bin-1J-MLL-1to10-PTLL-100to200_madgraph-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 4,949,000 |
| RunIII2024Summer24 | very low mass 1-10 GeV | `DYto2L_Bin-1J-MLL-1to10-PTLL-200to400_madgraph-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 500,000 |
| RunIII2024Summer24 | very low mass 1-10 GeV | `DYto2L_Bin-1J-MLL-1to10-PTLL-400to600_madgraph-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 50,000 |
| RunIII2024Summer24 | very low mass 1-10 GeV | `DYto2L_Bin-1J-MLL-1to10-PTLL-600toInf_madgraph-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 50,000 |
| RunIII2024Summer24 | very low mass 1-10 GeV | `DYto2L_Bin-1J-MLL-1to10-PTLL-70to100_madgraph-pythia8` (`150X_mcRun3_2024_realistic_v2-v2`) | 9,992,000 |
