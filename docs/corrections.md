# Corrections and their payloads

What this page answers: which correction each era applies, from which payload, where the payload
comes from, and what is a placeholder. The mapping is `wrcoffea/config.yaml`; the files are under
`data/jsonpog/<POG>/<Run>/<era>/` (POG JSONs), `data/reweights/` (our own payloads),
`data/muon_highpt/` and `data/pdf_sets/`. How each correction enters the event weight is in
[analysis.md](analysis.md). Verified on 2026-10-11.

## Where POG payloads come from, and how to check them

The authority is the CAT metadata area on cvmfs,
`/cvmfs/cms-griddata.cern.ch/cat/metadata/<POG>/<campaign>/latest/` (readable at the LPC), with
one campaign per era (`Run3-22CDSep23-Summer22-NanoAODv12`, `Run3-22EFGSep23-Summer22EE-NanoAODv12`,
`Run3-23CSep23-Summer23-NanoAODv12`, `Run3-23DSep23-Summer23BPix-NanoAODv12`,
`Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15`, `Run3-25Prompt-Summer24-NanoAODv15`, and the
UL campaigns). The older `jsonpog-integration` mirror stopped in September 2025 and is stale. Compare
by the sha256 of the canonical JSON, not of the gzip file. Last full check, all ten eras:
2026-10-11; every file identical to CAT except the 2024 `EGM/electron.json.gz`, which is CAT's plus
a HEEP working point CAT does not ship (kept on purpose).

## Per era

Era columns: the four UL eras, then 2022, 2022EE, 2023, 2023BPix, 2024, 2025.

### Jets

| Era | JEC (MC / data) | JER | Jet veto map | Veto rule | Jet ID |
| --- | --- | --- | --- | --- | --- |
| UL16APV | Summer19UL16APV_V7 | Summer20UL16APV_JRV5 | Summer19UL16_V1 | drop the jet | `jetId >= 6` |
| UL16 | Summer19UL16_V7 | Summer20UL16_JRV5 | Summer19UL16_V1 | drop the jet | `jetId >= 6` |
| UL17 | Summer19UL17_V5 | Summer19UL17_JRV4 | Summer19UL17_V1 | drop the jet | `jetId >= 6` |
| UL18 | Summer19UL18_V5 | Summer19UL18_JRV3 | Summer19UL18_V1 | drop the jet | `jetId >= 6` |
| 2022 | Summer22_22Sep2023_V4 | Summer22_22Sep2023_JRV2 | Summer22_23Sep2023_RunCD_V1 | reject the event | `jetId >= 6` |
| 2022EE | Summer22EE_22Sep2023_V4 | Summer22EE_22Sep2023_JRV2 | Summer22EE_23Sep2023_RunEFG_V1 | reject the event | `jetId >= 6` |
| 2023 | Summer23Prompt23_V4 | Summer23Prompt23_RunCv1234_JRV3 | Summer23Prompt23_RunC_V1 | reject the event | `jetId >= 6` |
| 2023BPix | Summer23BPixPrompt23_V4 | Summer23BPixPrompt23_RunD_JRV3 | Summer23BPixPrompt23_RunD_V1 | reject the event | `jetId >= 6` |
| 2024 | Summer24Prompt24_V5 | Summer24Prompt24_JRV2 | Summer24Prompt24_RunBCDEFGHI_V1 | reject the event | `jetid.json.gz` |
| 2025 | Summer24Prompt25_V3 | Summer24Prompt25_JRV2 | Summer24Prompt25_RunCDEFG_V1 | reject the event | `jetId >= 6` |

AK4 is CHS in UL and Puppi in Run 3, AK8 is Puppi everywhere; the JES uncertainty source is `Total`
(`CMS_scale_j_Total`), the UL18 file also ships the regrouped sources. The UL JERC files are the
CAT repackaging (installed 2026-10-11; values identical to the old files except the UL17 JER
resolution, ≤ 3 %). The JME recommendation and the veto-map screenshots are referenced in
`docs/pog_recommendations/` (not tracked).

### Pileup

`puWeights.json.gz` per era with the golden-JSON profile: `Collisions16_UltraLegacy_goldenJSON`,
`Collisions17_UltraLegacy_goldenJSON`, `Collisions18_UltraLegacy_goldenJSON`,
`Collisions2022_355100_357900_eraBCD_GoldenJson`, `Collisions2022_359022_362760_eraEFG_GoldenJson`,
`Collisions2023_366403_369802_eraBC_GoldenJson`, `Collisions2023_369803_370790_eraD_GoldenJson`,
`Collisions24_BCDEFGHI_goldenJSON` (`puWeights_BCDEFGHI.json.gz`), and for the 2024 MC used with 2025
data `Collisions25_goldenJSON` (`puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz`).

### Electrons

| Correction | UL16APV, UL16, UL17, UL18 | 2022, 2022EE, 2023, 2023BPix | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| reco SF | `electron.json.gz`, `UL-Electron-ID-SF`, RecoBelow20 / RecoAbove20 | `electron.json.gz`, `Electron-ID-SF`, Reco20to75 / RecoAbove75, clipped at 500 GeV | same | 2025 file |
| HEEP ID SF | flat per year (barrel / endcap): 2016 0.985 / 0.990, 2017 0.979 / 0.987, 2018 0.973 / 0.980 | flat central EGM values: 2022 pre-EE 1.007 / 1.019, post-EE 0.997 / 0.997, 2023 pre-BPix 1.006 / 0.988, post-BPix 1.009 / 0.988 (the 2022/2023 payloads have no HEEP working point) | the `HEEPID` working point of the 2024 payload, binned in pT, clipped at 1 TeV | 2025 file, HEEPID |
| trigger SF | the Run 2 HEEP trigger-OR efficiency per year, converted from the Run 2 analyzer (`electronHlt_WREGammaTrigger_run2legacy_v4.json.gz`, 2016 shared by both UL16 eras) | `electronHlt.json.gz`, `HLT_SF_Ele30_TightID` as a proxy for the OR | same | **placeholder**: the 2024 file |
| scale and smearing | none (UL NanoAOD is already corrected; no variation inputs) | `electronSS_EtDependent.json.gz` (scale on data, smearing on MC, with variations) | same | 2025 file |
| loose electron (boosted partner) | reco SF only; the loose-ID efficiency is in the LSF SF | same | same | same |

Above the last measured bin the EGM payloads hold a 1 ± 20 % placeholder; the analyzer clips reco
SFs to 500 GeV and HEEP ID SFs to 1 TeV instead of using it (open item in analysis.md).

### Muons

| Correction | UL16APV, UL16, UL17, UL18 | 2022, 2022EE, 2023, 2023BPix | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| reco, ID, isolation SFs | `muon_HighPt.json.gz` (global-muon reco vs \|η\| and p, HighPt ID, loose tracker isolation) | same | same | 2025 file |
| trigger SF | the leading muon's SF from `muon_HighPt.json.gz` (`NUM_HLT_DEN_HighPtLooseRelIsoProbes`); two SFs are never multiplied | per-leg data/MC efficiency OR from `muon_Z.json.gz`, entry `NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose` (our exact OR with the HighPt ID and loose tk-iso denominator; variation as Run 2: data efficiency up with MC efficiency down and the reverse, since 2026-10-12) | per-leg efficiency OR from `ScaleFactors_Muon_highPt_HLT_2024_schemaV2.json` | per-leg OR from the 2025 high-pT file |
| momentum | TuneP; **placeholder** uncertainties (flat 5 %/TeV scale, no smearing) | same placeholder | `data/muon_highpt/Run3/2024/RunIII2024Summer24.json`: MuonScaRe (`muon_scalesmearing.json.gz`) below 200 GeV, the generalized-endpoint correction on data and extra smearing on MC above ([studies/muon_momentum.md](studies/muon_momentum.md)) | placeholder (no 2025 payload yet) |
| loose muon (boosted partner) | reco and ID SFs, no isolation | same | same | same |

The reco SF's last bin is open-ended (p > 1.5 TeV; 2024 p > 0.9 TeV) with era-dependent central
values; see the open items in analysis.md. Every Run 3 muon file is the CAT release of 2026-06-18
(the 2022/2023 scale-smearing parameters were re-derived by the POG on 2026-04-28).

### Other event-level corrections

| Correction | Eras | Payload / value |
| --- | --- | --- |
| L1 prefiring | UL16APV, UL16, UL17, UL18 | NanoAOD `L1PreFiringWeight` Nom/Up/Dn (about −3 % on 2017 DY; the muon part only in 2018) |
| HEM 15/16 veto | UL18 | electrons with η < −1.25, −1.62 < φ < −0.82, data and MC, whole year |
| LSF SF (e / μ) | UL16APV, UL16 | 1.04 (+0.09 −0.08) / 1.01 ± 0.06 (AN-19-083 Table 60) |
| | UL17 | 1.02 ± 0.08 / 0.98 ± 0.07 |
| | UL18 | 1.05 (+0.07 −0.06) / 1.04 (+0.06 −0.05) |
| | 2022 to 2025 | **placeholder** 1.00 ± 0.10 per flavour until a Run 3 measurement exists |
| DY Z-pT reweight | all | `data/reweights/RunII/<year>/<era>/dy_zpt_reweight_ul_20261011.json` per UL era; `data/reweights/Run3/dy_zpt_reweight_run3_2022_2023_20261011.json` (2022, 2022EE, 2023, 2023BPix); `data/reweights/Run3/2024/RunIII2024Summer24/dy_zpt_reweight_run3_20261011.json` (2024, 2025) |
| DY jet-pT reshape | all but 2025 | `dy_reshape_run2jetpt_20261011.json` per UL era; `data/reweights/Run3/2022/`, `Run3/2023/`, `Run3/2024/RunIII2024Summer24/` per data-taking year |
| signal k-factor | all | `signal_kfactor_13tev_20261010.json` (UL), `signal_kfactor_13p6tev_20261010.json` (Run 3): per-point k(m(ℓN)) curves, [studies/signal_kfactor.md](studies/signal_kfactor.md) |
| PDF sets | all | `data/pdf_sets/<Run>/<year>/<era>.json` (`scripts/pdf_sets.py`): the LHA ID of each sample's `LHEPdfWeight` (NNPDF3.1 325300/325500 everywhere; UL also 306000 symmhessian+α_S and 320900 replicas); 2025 uses the 2024 table |

The Run 3 DY and signal payloads of 2026-10-11 are derivations of this repository
(`scripts/derive_dy_zpt_reweight.py`, `scripts/derive_dy_reshape.py`, `scripts/signal_kfactor/`),
with the recipe of AN-19-083; the Run 2 payloads converted from the Run 2 analyzer
(`scripts/convert_run2_*.py`) are kept next to them for comparison.

## Placeholders at a glance

| What | Where | Until |
| --- | --- | --- |
| LSF SF 1.00 ± 0.10 | Run 3 | a Run 3 measurement |
| Electron trigger SF = Ele30 TightID proxy | Run 3 | a tag-and-probe measurement of the OR |
| Electron trigger SF = the 2024 file | 2025 | a 2025 EGM HLT payload |
| TuneP momentum with flat 5 %/TeV scale, no smearing | UL, 2022, 2023, 2025 | the high-pT recipe payloads of those eras |
| EGM 1 ± 20 % above 500 GeV (reco) / 1 TeV (HEEP), clipped away | Run 3 | a decision on the high-pT treatment |
| No electron scale/smearing variations | UL | a prescription (see analysis.md) |
| No parton-shower variations | UL | a reskim with the event-derived PSSumw |
