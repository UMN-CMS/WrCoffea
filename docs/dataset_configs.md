# Dataset configs

The dataset configs list every data stream and MC sample of an era, with the
metadata the skim campaign and the analysis need: names, physics group, cross
section and its provenance. They live in

```
data/configs/<Run3|RunII>/<year>/<era>/<era>_<family>.json
```

with one file per family: `<era>_data.json`, `<era>_signal.json`, and one
`<era>_mc_dy_<variant>.json` per Drell-Yan variant (`lo_inc`, `lo_ht`,
`lo_njet`, `lo_ptll`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_ptll`, ...). Each
MC file holds all backgrounds plus one DY variant, so a background sample
appears once in every MC file of its era. The analyzer reads the `lo_ht` one
unless `--dy` names another (`default_mc_tag` in `wrcoffea/config.yaml`). Sub-folders `archived/` (previous
versions) and `nanov15/` (unused NanoAODv15 variants for 2022/2023) are not
read by any tool.

The analyzer does not read the configs directly. `skim.py plan` and
`skim.py refresh-metadata` copy their metadata into the campaign plan, and
`skim.py fileset` writes it into the filesets under `data/filesets/`, which is
what `run_analysis.py` reads.

## Entry layout

A config file is one JSON object keyed by the DAS path of each dataset. A
complete MC entry (2024 dileptonic ttbar):

```json
"/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM": {
  "das_name": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v3/NANOAODSIM",
  "run": "Run3",
  "year": "2024",
  "era": "RunIII2024Summer24",
  "dataset": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8",
  "physics_group": "tt_tW",
  "xsec": 98.036,
  "order": "NNLO+NNLL",
  "total_uncertainty": 0.01223,
  "genEventSumw": 37820112169.87066,
  "datatype": "mc",
  "xsec_source": {
    "generator": 762.1,
    "generator_unc": 0.09511,
    "header": "inclusive",
    "factor": 0.106146,
    "factor_reason": "all events are 2L2Nu but the header is inclusive: the sibling samples TTtoLNu2Q carry the same header value",
    "kfactor": 1.21191,
    "reference": 923.6,
    "reference_source": "NNLO+NNLL, TOP twiki TtbarNNLO, mt = 172.5 GeV",
    "normalize": "reference",
    "release": "CMSSW_15_0_21",
    "events_run": 43210,
    "tool": "GenXSecAnalyzer",
    "date": "2026-10-05"
  }
}
```

A data entry has only the naming keys:

```json
"/Muon/Run2022C-22Sep2023-v1/NANOAOD": {
  "das_name": "/Muon/Run2022C-22Sep2023-v1/NANOAOD",
  "run": "Run3",
  "year": "2022",
  "era": "Run3Summer22",
  "dataset": "Muon_Run2022C",
  "physics_group": "Muon",
  "datatype": "data"
}
```

### Entry keys

| key | entries | meaning |
|---|---|---|
| `das_name` | all | The DAS path, identical to the entry's key. For a private sample (not in DBS) both are the `dataset` name. |
| `run` | all | `Run3` or `RunII`. |
| `year` | all | Data-taking year, `"2022"`, ..., `"2016"`. |
| `era` | all | Analysis era: `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`, `RunIII2025Summer25`, `RunIISummer20UL16APV`, `RunIISummer20UL16`, `RunIISummer20UL17`, `RunIISummer20UL18`. |
| `dataset` | all | Short name used for outputs and plan keys. MC: the primary dataset name; an extension dataset normally carries the same name, so it is planned, skimmed and written as one dataset with its nominal (see "Extension datasets"). Data: `<primary dataset>_<run period>`, e.g. `Muon_Run2022C`; several DAS versions of one period share the name. |
| `physics_group` | all | Group the analyzer runs it in: `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `Signal`, or the data streams `Muon`, `EGamma` (`SingleMuon`/`SingleElectron` in Run 2). |
| `datatype` | all | `mc` or `data`. |
| `xsec` | MC | The cross section in pb that normalises the sample (see "How `xsec` is built"). This is the only cross-section field the analysis uses. `null` where no value is known yet. |
| `order` | MC | The perturbative order of `xsec` as it is used (see "The `order` key"). Always directly after `xsec`. |
| `total_uncertainty` | MC | For values from the GenXSecAnalyzer campaign: the statistical uncertainty of the generator number (`generator_unc`) scaled by `factor` and `kfactor`, in pb. It is **not** the theory uncertainty of the value (for example ±5 % for NNLO+NNLL ttbar). For older Run 2 values: whatever the source quoted, e.g. the PDF uncertainty 14.78 pb of the DY 6077.22 pb. Not used by the analyzer. |
| `genEventSumw` | MC | Sum of generator weights of the dataset as recorded in the config. The analyzer normalises with the value in the **fileset** metadata, which `skim.py fileset` sums from the Runs trees of the skimmed parts (exactly the files that were processed; nominal and `ext` datasets of one sample merged). The config value can be `0`, `null`, or cover only the nominal dataset. |
| `xsec_source` | MC (Run 3) | Provenance of `xsec` from the GenXSecAnalyzer campaign (next section). Run 2 entries do not have it yet; their provenance is in `docs/configs/run2_ul_dataset_report.md`. |
| `note` | some | Free text, e.g. low-mass companions, edge-era history, scan samples. |
| `optional` | some | `true` for samples the default selection does not need (the DY `MLL-10to50` companions). Not skimmed in skims_v2. |
| `scan` | some (UL) | `true` for the pre-legacy multi-model WR scan samples: skimmed whole and split per `GenModel_*` flag with `skim.py split-scan`; the per-model cross sections (LO × ⟨k⟩) are in `archived/RunIISummer20UL18_signal_private.json`. |
| `LHEScaleSumw`, `LHEPdfSumw`, `PSSumw` | some | Dataset-level weighted means of the theory weights (`scripts/gen_event_sumw.py`), the denominators of the scale/PDF/parton-shower variations. For skims_v2 the fileset metadata takes them from the skims instead. |
| `files` | private | Explicit `root://` URLs of a private sample (not in DBS); the skim planner takes these instead of a DBS file list. |
| `file_sizes`, `file_events`, `file_adler32` | private | Per-URL size in bytes, event count and site adler32, so the skim verifies transfers and event counts as it does for DBS files. Kept out of the plan and fileset metadata. |

### Private samples

The 2024 signal (`RunIII2024Summer24_signal.json`, 237 points) is a private
production stored at
`root://cmsxrootd.hep.wisc.edu//store/user/kalpana/WRAnalyzer/SignalSamples/Run3/RunIII2024Summer24`:
the Summer24 chain (premixed, GTs `140X_mcRun3_2024_realistic_v26` /
`150X_mcRun3_2024_realistic_v2`, HLT `/frozen/2024/2e34/v1.4`, NanoAODv15
content, gridpacks built in CMSSW_13_2_9), one hadd-merged file per mass
point. Its entries are written by `scripts/private_signal_config.py`, which
lists the directory over WebDAV, checks every file, and takes the cross
section from the sample itself (`xsec_source.header = "LHE"`, see below).
Besides the 200 points of the official Run 3 grid it has an electron-only
MWR = 3200 GeV scan in N; four of its files (N = 100, 250, 2910, 3070) are
empty or lack the generator branches and are not in the config.
`data/signal_points/RunIII2024Summer24_mass_points.csv` lists the 137 usable
(WR, N) pairs, so `--mass` and `--signal-points` accept the MWR = 3200
points (they select the electron dataset only). Event IDs
repeat across the merged jobs (every job wrote run 1, lumi 1, events from 1),
which nothing in the skim or the analyzer relies on. The Run3Summer23BPix
placeholder it replaces is in `archived/RunIII2024Summer24_signal_placeholder_Run3Summer23BPix.json`.

### `xsec_source` keys

| key | meaning |
|---|---|
| `tool`, `release`, `events_run`, `date` | GenXSecAnalyzer, the CMSSW release it ran in, how many events it used, and when the value was written. |
| `generator` | The generator cross section in pb that GenXSecAnalyzer reports ("After filter: final cross section"), measured on the sample's own MiniAOD for this era. |
| `generator_unc` | Its statistical uncertainty in pb. |
| `header` | What the generator number stands for: `inclusive` (all decay modes, even when the sample restricts them: POWHEG ttbar and tW), `includes decay` (the branching fraction is already applied: MG5_aMC, POWHEG t-channel with MadSpin), `per flavour` (POWHEG DY mass bins), `three-flavour` (MG5_aMC single-flavour DY, whose header is the e+μ+τ value). Decided from the LHE content of 20,000 events, sibling samples and the reference values. |
| `factor` | Multiplier from the header to the physical cross section of the sample: BR(W→ℓν)² = 0.3258² = 0.106146 for dileptonic ttbar/tW, 2 × 0.3258 × 0.6742 = 0.439309 for semileptonic, 1/3 for single-flavour DY from a three-flavour header, otherwise 1. BR(W→ℓν) = 0.3258 is the PDG lepton-universality value (3 × 0.1086), ℓ = e, μ, τ. |
| `factor_reason` | Why that factor was chosen. |
| `kfactor` | Ratio applied on top of the generator value: `reference / (generator × factor)` for ttbar, tW and t-channel; for DY the inclusive NNLO reference over the inclusive header of the sample's own generator class in its era (LO MLM, NLO FxFX, or the sum of the POWHEG per-flavour mass bins); 1 otherwise. |
| `reference` | The reference cross section in pb the k-factor is computed from (`null` if none): the inclusive higher-order value for ttbar/tW/t-channel; for DY the inclusive NNLO cross section, m(ll) > 50 GeV (three flavours: 6340.2 pb at 13.6 TeV, 6077.22 pb at 13 TeV; per flavour, 2113.4 pb, for the POWHEG samples). |
| `reference_source` | Order and source of `reference`. |
| `normalize` | `reference`: scaled to the reference where one exists (the convention used); `generator`: generator value × factor. |
| `status`, `kept_value`, `reason` | Present when no GenXSecAnalyzer value exists (`status: "not measured"`, e.g. MiniAOD only on tape): the earlier config value was kept. |
| `copied_from`, `copy_note` | The record was copied from another era's config for the same sample (the archived 2024 placeholder signal, which used the Run3Summer23BPix samples and their values). |
| `jobs`, `compared_to` | Private samples (`header: "LHE"`): the cross section is the per-event `LHEWeight_originalXWGTUP` times the events of each generator job (MadGraph `event_norm = sum`), averaged over the complete jobs weighted by their events (jobs that stopped early are left out; their events stay in the sample). `jobs` is the number of jobs used, `compared_to` the value of the same mass point in the reference config (the Run3Summer23BPix GenXSecAnalyzer value for the 2024 signal: ratio 0.9992 ± 0.0047 over 200 points). |

## How `xsec` is built

```
xsec = generator × factor × kfactor
```

The rules (analysis decision, 2026-10-05):

- **DY, every sample, LO and NLO** (decided 2026-10-10): generator value × factor × k, with k = the inclusive NNLO
  cross section for m(ll) > 50 GeV over the inclusive header of the sample's own generator class in its own era, so
  every DY variant carries the same total and k is flat within a class (the Z-pT reweight and the jet-pT reshape
  correct the shapes).
  - Reference at 13.6 TeV: MATRIX v2.1.0 NNLO QCD, 2113.4 pb per flavour (+0.7/−1.2 % scale, 1.0 % PDF), three
    flavours 6340.2 pb (CMS twiki MATRIXCrossSectionsat13p6TeV r12, 2023-10-25; `docs/xsec/CMS_MATRIXCrossSectionsat13p6TeV.pdf`).
    The NNLO QCD × NLO EW row (2091.7 pb) is not used. At 13 TeV: FEWZ NNLO 6077.22 pb (2025.74 per flavour;
    twiki StandardModelCrossSectionsat13TeV r28), the Run 2 number.
  - Denominators (GenXSec headers of the era): LO MLM inclusive (2022 5458, 2022EE 5431, 2023 5435, 2023BPix 5432,
    2024 mean of 5469 and 5442 → k 1.162 to 1.167); NLO FxFX inclusive (6769, 6668, 6736, 6739, 2024 mean 6671.5 →
    k 0.937 to 0.951); POWHEG per flavour: the sum of the nine mass bins above 50 GeV, 2244.0 pb → k 0.9418.
  - The FxFX header sits 4 to 7 % above the NNLO reference at both energies (13 TeV: 6415 vs 6077.22), so NLO
    samples are scaled down; HT-, jet-, pT- and mass-binned samples and the low-mass (10 to 50 GeV) companions take
    the k of their class. `order` is `LO x k(NNLO)` (MLM) or `NLO x k(NNLO)` (FxFX, POWHEG).
  - UL18 (13 TeV): k = 6077.22 / 5350.0 = 1.136 for LO MLM and 6077.22 / 6415.0 = 0.947 for FxFX, written with the
    UL18 campaign (see `docs/configs/run2_ul_dataset_report.md`). Before 2026-10-10 the Run 3 LO samples were scaled
    to the FxFX header instead (k 1.223 to 1.241, `LO x k(NLO)`) and the NLO samples kept their headers.
- **ttbar, tW, t-channel**: the inclusive reference × branching fraction, i.e. `reference × factor`
  (the generator value cancels): ttbar 923.6 pb (NNLO+NNLL), tW 43.95 pb per charge (approximate NNLO = NLO+NNLL), t-channel 145.0 pb (top) and 87.2 pb (antitop) (NNLO, 5-flavour scheme), all at 13.6 TeV, mt = 172.5 GeV.
- **WW, WZ, ZZ** (the inclusive pure-Pythia 8 samples, all decays; decided 2026-10-08): the MATRIX
  NNLO QCD × NLO EW reference at 13.6 TeV quoted by the first Run 3 measurements, each of which agrees
  with it: WW 125.8 ± 3.9 pb (including gg→WW; CMS, arXiv:2406.05101), WZ 54.7 +1.2/−1.1 pb
  (W⁺Z + W⁻Z, 60 < m(Z) < 120 GeV; CMS, arXiv:2412.02477), ZZ 16.7 ± 0.5 pb (66 < m(Z) < 116 GeV;
  ATLAS, arXiv:2311.09715). The Pythia LO values (about 80, 29 and 12.8 pb) are scaled by 1.57, 1.87 and
  1.31. The Z mass windows and Pythia's γ* contributions differ at the percent level, as in the usual CMS
  treatment of these samples. Run 2 UL keeps WW 118.7 pb (NNLO) and WZ 47.13 / ZZ 16.523 pb (NLO) for now.
- **LO MLM W+jets** (decided 2026-10-10, replacing the flat Run 2 factor 1.21): generator value × k with
  k = the inclusive NNLO W → ℓν cross section over the inclusive LO MLM W+jets header of the era, the DY construction.
  References: 13.6 TeV MATRIX NNLO QCD, W⁻ 9045.8 + W⁺ 12170.9 pb per flavour, three flavours 63650.1 pb (twiki
  MATRIXCrossSectionsat13p6TeV r12); 13 TeV FEWZ 3.1 NNLO 20508.9 pb per flavour, 61526.7 pb (twiki
  StandardModelCrossSectionsat13TeV r28). The denominators are the inclusive `WtoLNu-4Jets` (Run 3; for 2024 the
  Run3Winter24 JMENano sample of the same gridpack, no Summer24 production exists) and `WJetsToLNu` (UL18) samples,
  kept in the `lo_ht` configs as optional reference-only entries (never skimmed). The old 1.21 was this ratio with
  the 2015 LO header (61526.7 / 50690), carried unchanged to every later campaign.
- **Everything else** (tribosons, ttV, s-channel; signal before its k-factor): the generator value × factor.

`scripts/genxsec_campaign.py` implements this: `review` classifies every sample
and `apply` (dry run by default, `--write` to modify) writes `xsec`, `order`,
`total_uncertainty` and `xsec_source`. After `apply`, `bin/skim.py
refresh-metadata --plan-dir ...` and `bin/skim.py fileset ...` carry the new
values into the plans and filesets.

## Extension datasets

An extension (`..._ext1-vN`) is more events of the same sample, so the nominal
and its extensions must be normalised together: one cross section, divided by
the generator-weight sum of all of them.

- **Same `dataset` name (the convention).** `skim.py plan` puts the nominal and
  the extension into one planned dataset, and `skim.py fileset` writes one entry
  with the files of both and `genEventSumw` summed over both. Examples: 2022
  ttbar, tW (dileptonic), TTLL and inclusive DY; the 2024 `DYto2Mu-2Jets_Bin-1J`
  extension added 2026-10-08.
- **Different `dataset` names** (`X` and `X_ext1`; the UL configs, and the 2022
  and 2022EE semileptonic tW). They are planned and skimmed as separate
  datasets and written as separate fileset entries. `skim.py fileset` then gives
  every MC entry that shares a primary dataset name the combined
  `genEventSumw` (keeping the entry's own sum in `genEventSumw_own` and listing
  the merged samples in `normalised_with`), so the sample is still counted once.
  Entries of one sample with different cross sections are an error. Before this
  step was added (2026-10-08) the 2022/2022EE semileptonic tW was normalised once
  per dataset, i.e. counted twice.

## How the analyzer uses it

For MC the event weight is

```
weight = genWeight × xsec × lumi[fb⁻¹] × 1000 / genEventSumw
```

with `xsec` and `genEventSumw` taken from the fileset metadata. Nothing in
`xsec_source`, `order` or `total_uncertainty` enters the analysis.

## The `order` key

`order` says at a glance which perturbative order the normalisation of the
sample corresponds to.

| value | meaning | examples |
|---|---|---|
| `LO` | Leading-order generator value, unchanged. | UL ttWJets/ttZJets (madgraphMLM); nothing else since the 2026-10-10 writes |
| `NLO` | Next-to-leading-order generator or reference value. | tribosons, TTLL/TTLNu, s-channel (× BR where the sample is leptonic); DY amcatnloFXFX and POWHEG before 2026-10-10 |
| `NNLO` | Next-to-next-to-leading-order reference (× branching fraction where the sample restricts the decay). | t-channel and WW/WZ/ZZ in every era (13.6 TeV: MATRIX NNLO QCD × NLO EW; 13 TeV: t-channel 134.2 / 80.0 pb, WW 118.8, WZ 50.7, ZZ 16.5, `docs/configs/run2_ul_dataset_report.md`) |
| `NLO+NNLL` | NLO plus next-to-next-to-leading-log soft-gluon resummation, also called approximate NNLO. | tW |
| `NNLO+NNLL` | NNLO plus NNLL resummation. | ttbar |
| `LO x k(NLO)` | LO generator value scaled by an inclusive NLO/LO k-factor: the normalisation is NLO, the shapes and the k-factor's flatness are LO assumptions. | signal in every era since 2026-10-10 (official and 2024 private: LO × the point's own direct NLO/LO k, `docs/signal_kfactor.md`); Run 3 LO DY before 2026-10-10 (FxFX / LO MLM inclusive) |
| `LO x k(NNLO)` | LO generator value scaled by an inclusive NNLO/LO k-factor. | LO MLM DY (inclusive and every HT/jet/pT bin: NNLO / LO MLM inclusive header; MATRIX 6340.2 pb at 13.6 TeV, FEWZ 6077.22 pb at 13 TeV); W+jets HT bins in every era (NNLO W / LO MLM inclusive header; 1.21 before 2026-10-10) |
| `NLO x k(NNLO)` | NLO generator value scaled by an inclusive NNLO/NLO k-factor. | DY amcatnloFXFX and POWHEG (MATRIX NNLO / FxFX inclusive header, or / sum of the POWHEG mass bins) |
| `unknown` | No value yet (`xsec: null`), or a value whose provenance the rules cannot read. | UL16/UL16APV/UL17 (still the 2026-10-04 drafts): DY NLO mass bins and 0J/1J/2J high-mass bins, DY M-10to50 jet bins, official signal points; UL18 has none since the 2026-10-10 campaign write |

`apply` sets `order` together with `xsec`. To recompute it for every MC entry
(for instance after editing values by hand):

```bash
python3 scripts/genxsec_campaign.py annotate-order            # dry run: counts per era and order
python3 scripts/genxsec_campaign.py annotate-order --write
```

For Run 3 the order follows from `xsec_source`; for Run 2 entries without
`xsec_source` it follows the documented provenance of each committed value
(`UL_ORDER_RULES` in the script, from `docs/configs/run2_ul_dataset_report.md`
and AN-19-083 Tables 6-8).

Since the writes of 2026-10-10 the eras agree: LO MLM W+jets carry LO × k(NNLO W / LO MLM inclusive),
the signal LO × its own per-point NLO/LO k, the dibosons and the t-channel
NNLO references, and every DY sample the inclusive NNLO total of its energy.
What `order` still shows as different: UL ttW/ttZ are LO madgraphMLM where
Run 3 has NLO TTLL/TTLNu, and UL16/UL16APV/UL17 keep their draft values.

## Worked examples

### Dileptonic ttbar (2024, `NNLO+NNLL`)

The entry shown above.

1. `generator` = 762.1 pb is what POWHEG reports: the inclusive NLO ttbar
   cross section for all decay modes, although this sample holds only
   dileptonic events. The semileptonic sibling `TTtoLNu2Q` reports the same
   762.1 pb, which shows the header is inclusive.
2. `factor` = 0.106146 = BR(W→ℓν)², the fraction of ttbar events in which both
   W bosons decay to e, μ or τ plus a neutrino. At NLO the dileptonic cross
   section is 762.1 × 0.106146 = 80.89 pb.
3. `reference` = 923.6 pb is the NNLO+NNLL inclusive ttbar cross section at
   13.6 TeV for mt = 172.5 GeV.
4. `kfactor` = 923.6 / 762.1 = 1.21191 scales NLO to NNLO+NNLL.
5. `xsec` = 762.1 × 0.106146 × 1.21191 = 923.6 × 0.106146 = **98.036 pb**.

### LO Drell-Yan, single flavour (2024, `LO x k(NNLO)`)

```json
"xsec": 2118.6,
"order": "LO x k(NNLO)",
"xsec_source": {"generator": 5469.0, "header": "three-flavour", "factor": 0.333333,
                "kfactor": 1.16216, "reference": 6340.2,
                "reference_source": "LO x k(NNLO): NNLO QCD, MATRIX v2.1.0, pp -> e+e- with m(ll) > 50 GeV: 2113.4 pb per flavour (...), x3 flavours = 6340.2 pb; CMS twiki MATRIXCrossSectionsat13p6TeV r12 (...); k = reference / LO MLM inclusive three-flavour header of this era (GenXSec)", ...}
```

`DYto2E-4Jets_Bin-MLL-50` contains only electron pairs, but MG5 reports the
e+μ+τ cross section (5469.0 pb), so `factor` = 1/3. The k-factor is the MATRIX
NNLO QCD inclusive cross section (6340.2 pb for three flavours) over the LO MLM
inclusive header of the era (5455.5 pb, mean of the electron and muon samples,
5469.0 and 5442.0): 1.16216. `xsec` = 5469.0 / 3 × 1.16216 = **2118.6 pb**,
within 0.25 % of 2113.4 (the electron and muon headers of the same process
scatter around their mean). The HT-binned LO samples of the same era get the
same k-factor; the FxFX samples get 6340.2 / 6671.5 = 0.9503 and the POWHEG mass
bins 2113.4 / 2244.0 = 0.9418 (`NLO x k(NNLO)`).

### t-channel single top (2024, `NNLO`)

```json
"xsec": 47.241,
"order": "NNLO",
"xsec_source": {"generator": 38.6, "header": "includes decay", "factor": 1.0,
                "kfactor": 1.22386, "reference": 145.0,
                "reference_source": "NNLO 5FS, TOP twiki SingleTopNNLORef", ...}
```

`TBbarQtoLNu-t-channel-4FS` is t → bW with W → ℓν (every event has one charged
lepton and one neutrino, e/μ/τ in equal shares). The POWHEG 4-flavour NLO
header, 38.6 pb, already includes BR(W→ℓν): it is 82 % of reference × BR but
only 27 % of the bare reference, so `factor` = 1. `xsec` = 145.0 × 0.3258 =
**47.241 pb**; the antitop sample gets 87.2 × 0.3258 = 28.41 pb.

### NLO Drell-Yan mass bin (2022, `NLO`)

```json
"xsec": 2219.0,
"order": "NLO",
"xsec_source": {"generator": 2219.0, "header": "per flavour", "factor": 1.0, "kfactor": 1.0, ...}
```

POWHEG's `DYto2E_MLL-50to120` header is already the single-flavour cross
section (the POWHEG mass bins above 50 GeV sum to the FxFX three-flavour value
divided by 3), so the generator value is used as is.

### Signal (2023, `LO`) and a point that could not be measured

A measured point carries its MG5 LO generator value (`kfactor` 1). For points
whose MiniAOD exists only on tape, `xsec_source` is

```json
{"status": "not measured", "kept_value": ..., "tool": "GenXSecAnalyzer",
 "reason": "MiniAOD only on tape (no disk replica in Rucio); ...", "date": "2026-10-07"}
```

and `xsec` keeps the earlier configuration value (also MG5 LO, so `order` is
`LO`).

### Run 2 UL value without a campaign record (UL18, `LO x k(NNLO)`)

```json
"/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM": {
  "...": "...",
  "xsec": 435.237,
  "order": "LO x k(NNLO)",
  "total_uncertainty": 3.128,
  "genEventSumw": 58225632.0,
  "datatype": "mc"
}
```

The UL W+jets HT bins carried the 2016 LO values × 1.21 (NNLO/LO inclusive
k-factor) used by AN-19-083 until the UL18 campaign write of 2026-10-10, and
since then their own GenXSec LO × k(NNLO W / LO MLM inclusive header), `order`
`LO x k(NNLO)` in both cases.
