# Inputs: dataset configs, cross sections and filesets

What this page answers: where the list of datasets of an era lives, how each MC cross section is
built and recorded, and how the filesets the analyzer reads are produced. Verified on 2026-10-12.

```
data/configs/<Run3|RunII>/<year>/<era>/<era>_<family>.json      configs: datasets + metadata
data/filesets/<Run3|RunII>/<year>/<era>/skimmed/<era>_<family>_fileset.json      what the analyzer reads
data/filesets/<Run3|RunII>/<year>/<era>/unskimmed/<era>_<family>_fileset.json    for --unskimmed runs
```

One file per *family*: `data`, `signal`, and one `mc_dy_<variant>` per Drell-Yan variant
(`lo_ht`, `lo_inc`, `lo_njet`, `lo_ptll`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_njet_mll`,
`nlo_ptll`; the list per era is `dy_variants` in `wrcoffea/config.yaml`). Each MC family holds all
backgrounds plus one DY variant, so a background sample appears once in every MC file of its era.
The analyzer reads `lo_ht` unless `--dy` names another variant (`default_mc_tag`). UL16, UL17 and
UL18 also have a `signal_scan` family (the pre-legacy multi-model scan samples). The `nanov15/`
sub-folders of the 2022, 2022EE, 2023 and 2023BPix configs hold unused NanoAODv15 variants and are
not read by any tool.

The analyzer never reads the configs directly: `skim.py plan` copies their metadata into the
campaign plan, `skim.py fileset` writes it into the skimmed filesets, and
`skim.py unskimmed-fileset` into the unskimmed ones.

## Config entries

A config is one JSON object keyed by the DAS path of each dataset. A complete MC entry (2024
dileptonic ttbar):

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

A data entry has only the naming keys (`das_name`, `run`, `year`, `era`, `dataset`,
`physics_group`, `datatype: "data"`).

| key | entries | meaning |
| --- | --- | --- |
| `das_name` | all | The DAS path, identical to the entry's key. A private sample (not in DBS) uses its `dataset` name for both. |
| `run`, `year`, `era` | all | `Run3` or `RunII`; the data-taking year; the analysis era (`Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`, `RunIII2025Summer25`, `RunIISummer20UL16APV`, `RunIISummer20UL16`, `RunIISummer20UL17`, `RunIISummer20UL18`). |
| `dataset` | all | Short name used for outputs and plan keys. MC: the primary dataset name; an extension dataset normally carries the same name and is skimmed and normalised together with the nominal (see below). Data: `<primary dataset>_<run period>`, e.g. `Muon_Run2022C`. |
| `physics_group` | all | The group the analyzer runs it in: `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `Signal`, or the data streams `Muon`, `EGamma` in every era (the Run 2 `SingleMuon` and `SingleElectron` primary datasets are in `Muon` and `EGamma`). |
| `datatype` | all | `mc` or `data`. |
| `xsec` | MC | The cross section in pb that normalises the sample, the only cross-section field the analysis uses. |
| `order` | MC | The perturbative order `xsec` corresponds to (table below). |
| `total_uncertainty` | MC | For campaign values: the statistical uncertainty of the generator number scaled like the value, in pb. Not a theory uncertainty, not used by the analyzer. |
| `xsec_source` | MC, every era | Provenance of `xsec` from the GenXSecAnalyzer campaign (next section). The UL campaign: [studies/ul_xsec_campaign.md](studies/ul_xsec_campaign.md). The UL multi-model scan samples have none. |
| `note` | some | Free text: low-mass companions, edge-era history, scan samples. |
| `optional` | some | `true` for samples the default selection does not need (the DY `MLL-10to50` companions, reference-only inclusive W+jets). Not skimmed. |
| `scan` | some (UL) | `true` for the pre-legacy multi-model WR scan samples: skimmed whole and split per `GenModel_*` flag with `skim.py split-scan`. |
| `files`, `file_sizes`, `file_events`, `file_adler32` | private | The explicit `root://` URLs of a private sample with the per-file size, events and site checksum, so the skim verifies transfers as it does for DBS files. Kept out of the plan and fileset metadata. |

**Private samples.** The 2024 signal (237 points) is a private Summer24 production at
`root://cmsxrootd.hep.wisc.edu//store/user/kalpana/WRAnalyzer/SignalSamples/Run3/RunIII2024Summer24`
(premixed; GTs `140X_mcRun3_2024_realistic_v26` / `150X_mcRun3_2024_realistic_v2`; HLT
`/frozen/2024/2e34/v1.4`; NanoAODv15 content), one hadd-merged file per mass point. Its entries
are written by `scripts/private_signal_config.py`, which lists the directory over WebDAV, checks
every file and takes the cross section from the sample itself (`header: "LHE"`). Besides the 200
points of the official Run 3 grid it has an electron-only m_WR = 3200 GeV scan in m_N; four files
(N = 100, 250, 2910, 3070) are empty or lack the generator branches and are not in the config.
`data/signal_points/RunIII2024Summer24_mass_points.csv` lists the 137 usable (m_WR, m_N) pairs.
Event IDs repeat across the merged jobs (every job wrote run 1, lumi 1), which nothing relies on.

### `xsec_source`

| key | meaning |
| --- | --- |
| `tool`, `release`, `events_run`, `date` | GenXSecAnalyzer, the CMSSW release it ran in, the events used, the date of the write. |
| `generator`, `generator_unc` | The generator cross section in pb that GenXSecAnalyzer reports on the sample's own MiniAOD ("After filter: final cross section"), and its statistical uncertainty. |
| `header` | What the generator number stands for: `inclusive` (all decay modes even when the sample restricts them: POWHEG ttbar, tW), `includes decay` (the branching fraction is already in: MG5_aMC, POWHEG t-channel with MadSpin), `per flavour` (POWHEG DY mass bins), `three-flavour` (MG5_aMC single-flavour DY, whose header is the e+μ+τ value), `LHE` (private samples). |
| `factor` | Header to physical cross section: BR(W→ℓν)² = 0.3258² = 0.106146 for dileptonic ttbar and tW, 2 × 0.3258 × 0.6742 = 0.439309 for semileptonic, 1/3 for single-flavour DY from a three-flavour header, otherwise 1. |
| `factor_reason` | Why. |
| `kfactor` | The ratio applied on top: `reference / (generator × factor)` for ttbar, tW, t-channel; for DY and W+jets the inclusive NNLO reference over the inclusive header of the sample's generator class in its era; 1 otherwise. |
| `reference`, `reference_source` | The reference cross section the k-factor is computed from, and its order and source. |
| `normalize` | `reference` (scaled to the reference, the convention) or `generator` (generator × factor). |
| `copied_from` | Official signal whose MiniAOD could not be read in this era: the generator value measured on the same primary dataset in the nearest sibling UL era (the UL campaigns share the 13 TeV gridpacks), or, failing that, on the other-flavour sample of the same mass point (same gridpack parameters; the two flavours' values agree to about 1 %). The note names the donor; `generator`, `release` and `events_run` are the donor's. |
| `jobs`, `compared_to` | Written by `scripts/private_signal_config.py` for a private sample: the cross section is the per-event `LHEWeight_originalXWGTUP` times the events of each generator job, averaged over the complete jobs; `compared_to` is the same point in a reference config. `genxsec_campaign.py apply` replaces that record when it applies the signal k-factor, so no current entry carries them. |
| `base_source` | Signal scaled by its k-factor without a GenXSecAnalyzer measurement (the 2024 private samples): where the LO value the k-factor multiplies comes from (the private production's LHE header, or an LO value already in the config). |

## How `xsec` is built

```
xsec = generator × factor × kfactor
```

- **Drell-Yan, every sample, LO and NLO** (2026-10-10): k = the inclusive NNLO cross section for
  m(ℓℓ) > 50 GeV over the inclusive header of the sample's own generator class in its own era, so
  every DY variant carries the same total; the Z-pT reweight and the jet-pT reshape correct the
  shapes ([analysis.md](analysis.md)). References: 13.6 TeV MATRIX v2.1.0 NNLO QCD, 2113.4 pb per
  flavour, 6340.2 pb for three (CMS twiki MATRIXCrossSectionsat13p6TeV r12); 13 TeV FEWZ NNLO
  6077.22 pb (twiki StandardModelCrossSectionsat13TeV r28). Denominators: the LO MLM inclusive
  header (k 1.162 to 1.167 in Run 3, 1.136 in UL18), the NLO FxFX inclusive header (k 0.937 to
  0.951, UL18 0.947), the sum of the POWHEG mass bins above 50 GeV (k 0.942). A twiki number "per
  flavour" is ×3 for `DYto2L`/`DYJetsToLL` samples and used as is for `DYto2E`/`DYto2Mu`. `order`
  is `LO x k(NNLO)` or `NLO x k(NNLO)`.
- **ttbar, tW, t-channel**: the inclusive reference × branching fraction (the generator value
  cancels): ttbar 923.6 pb (NNLO+NNLL), tW 43.95 pb per charge (NLO+NNLL), t-channel 145.0 (top) and
  87.2 pb (antitop) (NNLO, 5FS), 13.6 TeV, m_t = 172.5 GeV. 13 TeV values:
  [studies/ul_xsec_campaign.md](studies/ul_xsec_campaign.md).
- **WW, WZ, ZZ** (inclusive Pythia 8, 2026-10-08): the MATRIX NNLO QCD × NLO EW references quoted by
  the first Run 3 measurements, WW 125.8 pb, WZ 54.7 pb, ZZ 16.7 pb. Run 2 UL (2026-10-10): the
  13 TeV NNLO references quoted by the CMS 13 TeV measurements, WW 118.8 pb, WZ 50.7 pb, ZZ 16.5 pb.
- **LO MLM W+jets** (2026-10-10): generator × k with k = the inclusive NNLO W→ℓν cross section over
  the inclusive LO MLM header of the era (13.6 TeV MATRIX 63650.1 pb for three flavours; 13 TeV FEWZ
  61526.7 pb), the DY construction. The inclusive `WtoLNu-4Jets` / `WJetsToLNu` samples stay in the
  `lo_ht` configs as optional reference-only entries. This replaced the flat Run 2 factor 1.21 (the
  same ratio with the 2015 LO header).
- **Signal** (2026-10-10, official and private, every era): LO × the point's own direct NLO/LO
  k-factor, `order` `LO x k(NLO)`; derivation in [studies/signal_kfactor.md](studies/signal_kfactor.md).
- **Everything else** (tribosons, ttV, s-channel): generator × factor.

`scripts/genxsec_campaign.py` implements the rules: `review` classifies every sample, `apply`
(dry run by default, `--write` to modify) writes `xsec`, `order`, `total_uncertainty` and
`xsec_source`, `annotate-order` recomputes `order`. After a write: `bin/skim.py refresh-metadata
--plan-dir ...` and `bin/skim.py fileset ...` carry the values into the plans and filesets.

| `order` | meaning | examples |
| --- | --- | --- |
| `LO`, `NLO`, `NNLO` | Generator or reference value at that order (× BR where the sample restricts the decay). | UL ttW/ttZ (LO); tribosons, TTLL/TTLNu, s-channel (NLO); t-channel, WW/WZ/ZZ (NNLO) |
| `NLO+NNLL`, `NNLO+NNLL` | With soft-gluon resummation. | tW; ttbar |
| `LO x k(NLO)` | LO scaled by an NLO/LO k-factor. | signal (per-point direct k) |
| `LO x k(NNLO)` | LO scaled by an NNLO/LO k-factor. | LO MLM DY and W+jets (inclusive and every bin) |
| `NLO x k(NNLO)` | NLO scaled by an NNLO/NLO k-factor. | DY amcatnloFXFX and POWHEG |
| `unknown` | No value, or a value whose provenance the rules cannot read. | none in the current configs |

**Extension datasets.** An extension (`..._ext1-vN`) is more events of the same sample, so nominal
and extension share one cross section divided by the generator-weight sum of both. With the same
`dataset` name (the convention) `skim.py plan` makes one planned dataset and `skim.py fileset` one
entry with the summed `genEventSumw`. With different names (`X`, `X_ext1`: the UL configs, the
2022 semileptonic tW) they are skimmed separately and `skim.py fileset` gives every entry of the
primary dataset the combined `genEventSumw` (its own in `genEventSumw_own`, the partners in
`normalised_with`).

**How the analyzer uses it.** For MC, `weight = genWeight × xsec × lumi[fb⁻¹] × 1000 / genEventSumw`,
with `xsec` and `genEventSumw` from the fileset metadata. `genEventSumw` is not a config key:
`skim.py fileset` sums it from the Runs trees of the skimmed parts and refuses an MC dataset whose
parts carry none. `--unskimmed` runs sum `genWeight` over the files themselves. Nothing in
`xsec_source`, `order` or `total_uncertainty` enters the analysis.

## Skimmed filesets

The skim campaign writes them from the campaign plan and the part sidecars
([skimming.md](skimming.md)):

```bash
python3 bin/skim.py fileset --plan-dir <campaign root>/plans/v2 --out <campaign root> --era RunIII2024Summer24
```

The plan (`plan.json`, written by `skim.py plan`) and the part sidecars are not in the
repository: they sit in the campaign root, `--out` (the default is Billy's).

- **Files**: the skims_v2 parts at Wisconsin,
  `root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2/...` (`--local` points at
  the local parts).
- **Normalisation**: `genEventSumw`, `genEventCount` and the theory-weight sums (`LHEScaleSumw`,
  `LHEPdfSumw`, `PSSumw`) come from the skims' own Runs trees (for an input whose Runs tree lacks a
  sum, the skimmer derives it from the events, see skimming.md), so no DAS pass is needed. The UL
  filesets of the 2026-10 skims predate that and carry a `PSSumw` estimated from a few original files
  per sample by `scripts/estimate_ps_sums.py` (`PSSumw_source` says how); rerun it after a fileset
  regeneration until the UL skims are remade.
- **Metadata**: cross sections, `order`, `xsec_source` and `physics_group` from the configs through
  the plan; after a config change run `bin/skim.py refresh-metadata --plan-dir ...` first.
- **Output**: one file per family under `data/filesets/` (`--fileset-dir` overrides). Every active
  skimmed fileset of UL16APV, UL16, UL17, UL18 and 2022 to 2024 points at skims_v2; 2025 has no skims yet.
- **Data-only eras**: for `RunIII2025Summer25` (`mc_alias` in `config.yaml`) `skim.py fileset`
  writes the MC filesets from the 2024 configs and skims, with the era rewritten and
  `skims_from_era` recording the source.

## Unskimmed filesets

The original NanoAOD files of every dataset through an XRootD redirector, for `--unskimmed` runs.
Written from the configs and DBS; a grid proxy is all it needs:

```bash
python3 bin/skim.py unskimmed-fileset --era RunIII2024Summer24            # every family of the era
python3 bin/skim.py unskimmed-fileset --all                                # every era with configs
python3 bin/skim.py unskimmed-fileset --era Run3Summer22 --family data signal
```

- **Files**: the valid files of each DAS dataset as `root://cmsxrootd.fnal.gov//store/...`
  (`--redirector` changes the prefix), sorted; a private sample keeps the config's own file list.
- **Metadata**: the config entry (minus its file bookkeeping and note) plus `sample`, `nfiles`,
  `nevts` and a `listing` record (source, date, redirector). No `genEventSumw`: `--unskimmed` runs
  sum `genWeight` themselves. `--runs-sums` adds `genEventSumw`, `genEventCount` and the
  theory-weight sums from every file's Runs tree (one read per file; needed only for the
  theory-weight systematics on unskimmed files).
- An `optional` dataset without valid files is skipped with a warning; a required one stops the
  run. The `signal_scan` families are left out (the scan samples are split only when skimmed).
- Rerunning overwrites the files deterministically; a `git diff` then shows what changed in DAS.
- `unskimmed-fileset` does not apply `mc_alias`: for the data-only `RunIII2025Summer25` it writes
  only the data fileset, so there is no unskimmed 2025 MC fileset.

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy lo_inc
```

## Fileset format and validation

```
{"<dataset key>": {"files": {"<url>": "Events", ...}, "metadata": {"sample": ..., "physics_group": ..., "xsec": ..., ...}}, ...}
```

`metadata.sample` is the only key the loader insists on; `physics_group` selects the sample group,
`xsec` and `genEventSumw` normalise MC. Check a fileset without processing:

```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --preflight-only
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
```

Any fileset can be passed explicitly with `--fileset PATH`, for example the UL18 private-scan
fileset `data/filesets/RunII/2018/RunIISummer20UL18/skimmed/RunIISummer20UL18_signal_scan_fileset.json`.
