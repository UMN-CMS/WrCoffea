# Inputs: dataset configs, cross sections and filesets

```
data/configs/<Run3|RunII>/<year>/<era>/<era>_<family>.json                        datasets + metadata
data/filesets/<Run3|RunII>/<year>/<era>/skimmed/<era>_<family>_fileset.json       what the analyzer reads
data/filesets/<Run3|RunII>/<year>/<era>/unskimmed/<era>_<family>_fileset.json     for --unskimmed runs
```

- One file per family: `data`, `signal`, one `mc_dy_<variant>` per Drell-Yan variant (`lo_ht`,
  `lo_inc`, `lo_njet`, `lo_ptll`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_njet_mll`, `nlo_ptll`; the
  list per era is `dy_variants` in `wrcoffea/config.yaml`). Each MC family holds every background
  plus one DY variant. The analyzer reads `lo_ht` unless `--dy` names another (`default_mc_tag`).
- UL16, UL17 and UL18 also have a `signal_scan` family (the pre-legacy multi-model scan samples).
  The `nanov15/` sub-folders of the 2022 and 2023 configs hold unused NanoAODv15 variants.
- The analyzer never reads the configs: `skim.py plan` copies their metadata into the campaign
  plan, `skim.py fileset` writes it into the skimmed filesets, `skim.py unskimmed-fileset` into the
  unskimmed ones.

## Config entries

One JSON object keyed by the DAS path of each dataset. A data entry has only the naming keys
(`das_name`, `run`, `year`, `era`, `dataset`, `physics_group`, `datatype: "data"`). An MC entry:

```json
"/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM": {
  "das_name": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
  "run": "Run3", "year": "2024", "era": "RunIII2024Summer24",
  "dataset": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
  "physics_group": "tt_tW",
  "xsec": 98.036, "order": "NNLO+NNLL", "total_uncertainty": 0.01223,
  "datatype": "mc",
  "xsec_source": {"generator": 762.1, "generator_unc": 0.09511, "header": "inclusive",
                  "factor": 0.106146, "factor_reason": "all events are 2L2Nu but the header is inclusive ...",
                  "kfactor": 1.21191, "reference": 923.6, "reference_source": "NNLO+NNLL, TOP twiki TtbarNNLO, mt = 172.5 GeV",
                  "normalize": "reference", "release": "CMSSW_15_0_21", "events_run": 43210,
                  "tool": "GenXSecAnalyzer", "date": "2026-10-05"}
}
```

| key | entries | meaning |
| --- | --- | --- |
| `das_name` | all | The DAS path, identical to the key. A private sample uses its `dataset` name for both. |
| `run`, `year`, `era` | all | `Run3` or `RunII`; the data-taking year; the analysis era. |
| `dataset` | all | Short name for outputs and plan keys. MC: the primary dataset name (an extension dataset normally carries the same name, see below). Data: `<primary dataset>_<run period>`, e.g. `Muon_Run2022C`. |
| `physics_group` | all | `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `Signal`, or the data streams `Muon`, `EGamma` (the Run 2 `SingleMuon` and `SingleElectron` datasets are in `Muon` and `EGamma`). `tt_tW` = dileptonic tt + dileptonic tW (UL: the `NoFullyHadronicDecays` tW, both decay modes, as in Run 2); `Nonprompt` = W+jets + single top s/t + semileptonic tt + the Run 3 semileptonic tW (`LNu2Q`); `Other` = multiboson + ttV. |
| `datatype` | all | `mc` or `data`. |
| `xsec` | MC | The cross section in pb that normalises the sample, the only cross-section field the analysis uses. |
| `order` | MC | The perturbative order `xsec` corresponds to (table below). |
| `total_uncertainty` | MC | The statistical uncertainty of the generator number scaled like the value, in pb. Not used by the analyzer. |
| `xsec_source` | MC | Provenance of `xsec` from the GenXSecAnalyzer campaign (below). The UL multi-model scan samples have none. |
| `note` | some | Free text. |
| `optional` | some | `true` for samples the default selection does not need (the DY `MLL-10to50` companions, reference-only inclusive W+jets). Not skimmed. |
| `scan` | some (UL) | `true` for the pre-legacy multi-model WR scan samples: skimmed whole and split per `GenModel_*` flag with `skim.py split-scan`. |
| `files`, `file_sizes`, `file_events`, `file_adler32` | private | The `root://` URLs of a private sample with per-file size, events and checksum, so the skim verifies transfers as for DBS files. Not copied into plans or filesets. |

Private samples: the 2024 signal (237 points) is a private Summer24 production at
`root://cmsxrootd.hep.wisc.edu//store/user/kalpana/WRAnalyzer/SignalSamples/Run3/RunIII2024Summer24`
(premixed; GTs `140X_mcRun3_2024_realistic_v26` / `150X_mcRun3_2024_realistic_v2`; HLT
`/frozen/2024/2e34/v1.4`; NanoAODv15 content), one merged file per mass point. Its entries are
written by `scripts/private_signal_config.py`, which lists the directory over WebDAV, checks every
file and takes the cross section from the sample (`header: "LHE"`). Besides the 200 points of the
official Run 3 grid it has an electron-only m_WR = 3200 GeV scan in m_N; four files (N = 100, 250,
2910, 3070) are empty or lack the generator branches and are not in the config.
`data/signal_points/RunIII2024Summer24_mass_points.csv` lists the 137 usable (m_WR, m_N) pairs.
Event IDs repeat across the merged jobs (every job wrote run 1, lumi 1); nothing relies on them.

### `xsec_source`

| key | meaning |
| --- | --- |
| `tool`, `release`, `events_run`, `date` | GenXSecAnalyzer, the CMSSW release, the events used, the date of the write. |
| `generator`, `generator_unc` | GenXSecAnalyzer's "After filter: final cross section" on the sample's own MiniAOD, in pb, and its statistical uncertainty. |
| `header` | What the generator number stands for: `inclusive` (all decay modes even when the sample restricts them: POWHEG ttbar, tW), `includes decay` (MG5_aMC, POWHEG t-channel with MadSpin), `per flavour` (POWHEG DY mass bins), `three-flavour` (MG5_aMC single-flavour DY), `LHE` (private samples). |
| `factor` | Header to physical cross section: BR(W→ℓν)² = 0.3258² = 0.106146 for dileptonic ttbar and tW, 2 × 0.3258 × 0.6742 = 0.439309 for semileptonic, 1/3 for single-flavour DY from a three-flavour header, else 1. `factor_reason` says why. |
| `kfactor` | `reference / (generator × factor)` for ttbar, tW, t-channel; for DY and W+jets the inclusive NNLO reference over the inclusive header of the sample's generator class in its era; 1 otherwise. |
| `reference`, `reference_source` | The reference cross section the k-factor is computed from, with its order and source. |
| `normalize` | `reference` (scaled to the reference) or `generator` (generator × factor). |
| `copied_from` | Official signal whose MiniAOD could not be read in this era: the generator value measured on the same primary dataset in the nearest sibling UL era (the UL campaigns share the 13 TeV gridpacks), or on the other-flavour sample of the same point (the two agree to about 1 %). `generator`, `release` and `events_run` are the donor's. |
| `jobs`, `compared_to` | Written by `scripts/private_signal_config.py`: the per-event `LHEWeight_originalXWGTUP` times the events of each generator job, averaged over the complete jobs; `compared_to` is the same point in a reference config. Replaced by `genxsec_campaign.py apply`, so no current entry carries them. |
| `base_source` | Signal scaled by its k-factor without a GenXSecAnalyzer measurement (the 2024 private samples): where the LO value comes from. |

## How `xsec` is built

`xsec = generator × factor × kfactor`

| Samples | Rule | References |
| --- | --- | --- |
| Drell-Yan, every sample, LO and NLO | k = inclusive NNLO for m(ℓℓ) > 50 GeV over the inclusive header of the sample's generator class in its era, so every DY variant carries the same total | 13.6 TeV: MATRIX v2.1.0 NNLO QCD, 2113.4 pb per flavour, 6340.2 pb for three (twiki MATRIXCrossSectionsat13p6TeV r12). 13 TeV: FEWZ NNLO 6077.22 pb (twiki StandardModelCrossSectionsat13TeV r28). k: LO MLM header 1.162 to 1.167 in Run 3, 1.136 in UL18; NLO FxFX header 0.937 to 0.951, UL18 0.947; POWHEG mass bins above 50 GeV 0.942. A "per flavour" number is ×3 for `DYto2L`/`DYJetsToLL`, as is for `DYto2E`/`DYto2Mu`. `order`: `LO x k(NNLO)` or `NLO x k(NNLO)` |
| ttbar, tW, t-channel | the inclusive reference × branching fraction (the generator value cancels) | 13.6 TeV, m_t = 172.5 GeV: ttbar 923.6 pb (NNLO+NNLL), tW 43.95 pb per charge (NLO+NNLL), t-channel 145.0 (top) and 87.2 pb (antitop) (NNLO, 5FS). 13 TeV: the `13TeV` entries of the same tables in `scripts/genxsec_campaign.py` |
| WW, WZ, ZZ (inclusive Pythia 8) | the reference | 13.6 TeV: MATRIX NNLO QCD × NLO EW as quoted by the first Run 3 measurements, WW 125.8, WZ 54.7, ZZ 16.7 pb. UL: the 13 TeV NNLO references of the CMS measurements, WW 118.8, WZ 50.7, ZZ 16.5 pb |
| LO MLM W+jets | generator × k, k = inclusive NNLO W→ℓν over the inclusive LO MLM header of the era | 13.6 TeV MATRIX 63650.1 pb for three flavours; 13 TeV FEWZ 61526.7 pb. The inclusive `WtoLNu-4Jets` / `WJetsToLNu` samples stay in the `lo_ht` configs as optional reference entries. Replaced the flat Run 2 factor 1.21 |
| Signal, official and private | LO × the point's own NLO/LO k-factor, `order` `LO x k(NLO)` | our MG5_aMC NLO/LO runs per point (`scripts/signal_kfactor/`) |
| Tribosons, ttV, s-channel | generator × factor | |

| `order` | meaning | examples |
| --- | --- | --- |
| `LO`, `NLO`, `NNLO` | Generator or reference value at that order (× BR where the sample restricts the decay) | UL ttW/ttZ (LO); tribosons, TTLL/TTLNu, s-channel (NLO); t-channel, WW/WZ/ZZ (NNLO) |
| `NLO+NNLL`, `NNLO+NNLL` | With soft-gluon resummation | tW; ttbar |
| `LO x k(NLO)` | LO scaled by an NLO/LO k-factor | signal |
| `LO x k(NNLO)` | LO scaled by an NNLO/LO k-factor | LO MLM DY and W+jets |
| `NLO x k(NNLO)` | NLO scaled by an NNLO/NLO k-factor | DY amcatnloFXFX and POWHEG |
| `unknown` | No value, or one whose provenance the rules cannot read | none today |

- `scripts/genxsec_campaign.py` implements the rules: `run` measures with GenXSecAnalyzer, `review`
  classifies every sample, `apply` writes `xsec`, `order`, `total_uncertainty` and `xsec_source`
  (dry run without `--write`), `annotate-order` recomputes `order`. After a write:
  `bin/skim.py refresh-metadata --plan-dir ...`, then `bin/skim.py fileset ...`.
- Extension datasets (`..._ext1-vN`) are more events of the same sample: one cross section divided
  by the generator-weight sum of both. With the same `dataset` name, `skim.py plan` makes one
  dataset and `skim.py fileset` one entry with the summed `genEventSumw`. With different names (the
  UL configs, the 2022 semileptonic tW) they are skimmed separately and `fileset` gives every entry
  of the primary dataset the combined `genEventSumw` (its own in `genEventSumw_own`, the partners in
  `normalised_with`).
- The analyzer: `weight = genWeight × xsec × lumi[fb⁻¹] × 1000 / genEventSumw`, both from the
  fileset metadata. `genEventSumw` is not a config key: `skim.py fileset` sums it from the Runs trees
  of the skimmed parts and refuses an MC dataset whose parts carry none; `--unskimmed` runs sum
  `genWeight` themselves. Nothing in `xsec_source`, `order` or `total_uncertainty` enters the analysis.

## Skimmed filesets

```bash
python3 bin/skim.py fileset --plan-dir <campaign root>/plans/v2 --out <campaign root> --remote <remote> --era RunIII2024Summer24
```

- Written from the campaign plan and the part sidecars ([skimming.md](skimming.md)); neither is in
  the repository.
- Files: the skims_v2 parts at Wisconsin, `root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2/...`
  (`--local` points at local parts). Every skimmed fileset of UL16APV, UL16, UL17, UL18 and 2022 to
  2024 points there.
- Normalisation: `genEventSumw`, `genEventCount` and the theory-weight sums (`LHEScaleSumw`,
  `LHEPdfSumw`, `PSSumw`) from the skims' own Runs trees; a sum the input lacks is derived by the
  skimmer from the events.
- Metadata: cross sections, `order`, `xsec_source` and `physics_group` from the configs through the
  plan; after a config change run `skim.py refresh-metadata --plan-dir ...` first.
- Output: one file per family under `data/filesets/` (`--fileset-dir` overrides).
- Data-only eras: for `RunIII2025Summer25` (`mc_alias` in `config.yaml`) `fileset` writes the MC
  filesets from the 2024 configs and skims with the era rewritten (`skims_from_era`). Without the
  2024 campaign root, `skim.py alias-filesets --era RunIII2025Summer25` writes the same from the
  committed 2024 filesets.

## Unskimmed filesets

The original NanoAOD files through an XRootD redirector, for `--unskimmed` runs; a grid proxy is
all it needs:

```bash
python3 bin/skim.py unskimmed-fileset --era RunIII2024Summer24            # every family of the era
python3 bin/skim.py unskimmed-fileset --all                                # every era with configs
python3 bin/skim.py unskimmed-fileset --era Run3Summer22 --family data signal
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy lo_inc
```

- Files: the valid files of each DAS dataset as `root://cmsxrootd.fnal.gov//store/...`
  (`--redirector` changes the prefix), sorted; a private sample keeps its own list.
- Metadata: the config entry plus `sample`, `nfiles`, `nevts` and a `listing` record. No
  `genEventSumw`; `--runs-sums` adds the Runs-tree sums (one read per file), needed only for the
  theory-weight systematics on unskimmed files.
- An `optional` dataset without valid files is skipped with a warning; a required one stops the
  run. The `signal_scan` families are left out. Rerunning overwrites deterministically, so `git diff`
  shows what changed in DAS. `mc_alias` is not applied: 2025 gets its data fileset only.

## Fileset format

```
{"<dataset key>": {"files": {"<url>": "Events", ...}, "metadata": {"sample": ..., "physics_group": ..., "xsec": ..., ...}}, ...}
```

`metadata.sample` is the only key the loader requires; `physics_group` selects the sample group,
`xsec` and `genEventSumw` normalise MC. Check a fileset without processing, or pass any fileset
explicitly:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --preflight-only
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3800_N1600 --fileset data/filesets/RunII/2018/RunIISummer20UL18/skimmed/RunIISummer20UL18_signal_scan_fileset.json
```
