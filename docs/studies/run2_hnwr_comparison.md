# WrCoffea versus the Run 2 HNWRAnalyzer

Comparison of this analyzer, for UL18 and for 2024, with the Run 2 WR analysis
code in [jedori0228/SKFlatAnalyzer](https://github.com/jedori0228/SKFlatAnalyzer),
branch `HNWR/Run2Legacy_v4` (commit `06f274c`, "final", 2022-05-13):
`Analyzers/src/HNWRAnalyzer.C` plus the object IDs in `DataFormats/src`.
That branch produced the `HNWRAnalyzer_SkimTree_LRSMHighPt_*` files under
`WR_Plotter/signal_fitting/run2_results/` (the `ReweightedQCDErrorEWCorr` and
`Reshaped` suffixes are the two DY corrections described below).  Run 2 ran on
the pre-UL 2018 samples (its sample table has 11.5 M events of
`DYJetsToLL_M-50_HT-100to200`; UL18 has 26.2 M).

The UL18 column is what matters for a Run 2 closure.  In the 2024 column,
differences that are era updates (triggers, POG payloads) are noted but not
treated as discrepancies; "as UL18" means the same code path.  Measured
differences are in [Measured comparison](#measured-comparison).

## Objects

| Object | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
|---|---|---|---|---|
| Electron kinematics | pT > 53, \|SC η\| < 2.4 (`SelectElectrons("HNWRLT", 53, 2.4)`) | pT > 53, \|η\| < 2.5, ECAL crack veto on SC η | as UL18 | **Differs**: η range 2.4 vs 2.5; Run 2 cuts on SC η |
| Electron energy | MiniAOD EGM scale and smearing | already applied in UL NanoAOD | EGM `electronSS_EtDependent` applied in the analyzer | Same intent |
| Electron tight | HEEP (`HNWRTight2018Prompt` for 2018) | `cutBased_HEEP` | as UL18 | Same |
| Electron loose | `HNWRLoose` (`DataFormats/src/Electron.C:401`): the POG Fall17 loose cuts (σiηiη, ΔηSeed, Δφin, 1/E−1/p, ≤ 1 missing hit, conversion veto) with the POG H/E cut replaced by **H/E < 0.5**, no isolation | Boosted: every `vidNestedWPBitmap` cut at Loose except isolation, so the POG H/E cut is kept, OR HEEP. Resolved: `cutBased == 2` (unused by any region) | as UL18 | **Differs**: H/E. Likely cause of the ee boosted DY CR deficit (nonprompt 0.39×, tt 0.63× of Run 2); not yet tested |
| Muon momentum | TuneP; Rochester below 200 GeV and the Generalized-Endpoint bias correction above 200 GeV on MC (`AnalyzerCore.C` `UseTunePMuon`), no smearing. SFs evaluated at the uncorrected MiniAOD TuneP pT | TuneP (`pt × tunepRelPt`), **uncorrected**; SFs at that pT | as UL18 | **Differs**: no momentum correction (the Run 3 GE κ values exist in the POG page; 2024 values only in an indico deck) |
| Muon kinematics | pT > 53, \|η\| < 2.4 | same, on TuneP pT | as UL18 | Same |
| Muon tight | POG HighPt ID and TrkIso/TuneP pT < 0.1 | `highPtId == 2` and `tkRelIso < 0.1` | as UL18 | Same |
| Muon loose | POG HighPt ID | `highPtId == 2` | as UL18 | Same |
| AK4 jets | TightLepVeto, pT > 40, \|η\| < 2.4, no lepton cleaning (`JetsVetoLeptonInside` exists but is never called); ΔR > 0.4 is a selection cut | same; JEC `Summer19UL18_V5`, JER `JRV2` | same; jet ID from the JME `jetid.json`, JEC `Summer24Prompt24_V5`, JER `JRV2` | Same |
| Jet veto map | none | `Summer19UL18_V1`, data and MC, whole year; covers 97 % of the HEM box and removes 18–20 % of selected events | `Summer24Prompt24_RunBCDEFGHI_V1`: 3.7 % of the \|η\| < 2.4 acceptance, outside the HEM box | **Differs**: the main UL18 yield difference (flat ~0.82) |
| AK8 jets | Tight ID, SD mass > 40, pT > 200, \|η\| < 2.4; SR additionally LSF > 0.75 | Same thresholds; SR uses `lsf3 > 0.75` | as UL18 | Same |
| b-tagging | DeepCSV medium, used to split SR and e-μ CRs into BJet / NoBJet | none | none | **Missing** (only needed for the b-jet sub-regions, which are not part of the fit) |

## Event selection

| Step | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
|---|---|---|---|---|
| MET filters | goodVertices, globalSuperTightHalo2016, HBHENoise, HBHENoiseIso, EcalDeadCellTP, BadPFMuon, eeBadSc (data only), ecalBadCalibReduced (2017+) | Same set; eeBadSc also on MC; BadPFMuonDz and hfNoisyHits when present | common set + `ecalBadCalibFilter` + the optional ones | Same, negligible differences |
| HEM 15/16 | drop events with a selected electron at η < −1.25, −1.62 < φ < −0.82 (data and MC, whole year, `isHEMVetoEvent`) | ported (`hem_veto`), on the two leptons each region is built from, electron η as in Run 2 | none (not needed) | Same for UL18 |
| Triggers | e: Ele32_WPTight, Photon200, Ele115_CaloIdVT_GsfTrkIdT; μ: Mu50, OldMu100, TkMu100 | Same | e: Ele30_WPTight, Photon200, Ele115_CaloIdVT_GsfTrkIdT, Ele50_CaloIdVT_GsfTrkIdT_PFJet165; μ: Mu50, HighPtTkMu100, CascadeMu100 | Same for UL18 |
| e-μ trigger | Muon triggers | Muon triggers | as UL18 | Same |
| Resolved baseline | **Exactly** 2 tight leptons (3-tight events fall to boosted and fail its extra-tight veto), leading pT > 60, ≥ 2 AK4 jets, ΔR > 0.4 among {l1, l2, j1, j2} | Same, plus the explicit sub-leading pT > 53 (implied in Run 2 by the object cut) | as UL18 | Same |
| Resolved SR | m(ll) > 400 and m(lljj) > 800 | Same | as UL18 | Same |
| Resolved DY CR | 60 ≤ m(ll) < 150 and m(lljj) > 800 | Same | as UL18 | Same |
| Resolved extras | LowWRCR (m(ll) > 200, m(lljj) < 800), DYCR1/2 (60–100 / 100–150), DYCR3 (200 < m(ll) < 400), BJet/NoBJet | `_nomass` study variants only | as UL18 | Missing sub-regions (cosmetic unless the m(ll) 200–400 CR is wanted) |
| Boosted tag | Not resolved | `~resolved` on the same objects | as UL18 | Same |
| Boosted DY CR | Tight lead (pT > 60), same-flavour loose partner with 60 ≤ m(ll) < 150 (first match), AK8 without LSF with \|Δφ(l, J)\| > 2, N = J or J + l₂ depending on ΔR < 0.8, m(WR) > 800 | Same | as UL18 | Same |
| Boosted SR | No DY pair, LSF AK8 with \|Δφ\| > 2, same-flavour loose lepton (pT > 53) within ΔR < 0.8, no opposite-flavour loose lepton in the jet, no extra tight lepton, m(ll) > 200, m(lJ) > 800 | Same | as UL18 | Same |
| Boosted e-μ CR | As SR but opposite-flavour lepton in the jet | Same (`flavor_cr`) | as UL18 | Same |
| Extra tight lepton veto | Any tight lepton other than the two used | Loose leptons passing the tight criteria, ΔR-matched away from the two used | as UL18 | Same |
| Blinding | — | SR data allowed with `--unblind` (Run 2 only) | SR data never written; ee/μμ data cutflows stop before m(ll) > 200 | — |

## Samples and normalization

| Item | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
|---|---|---|---|---|
| Normalization | xsec × lumi × genWeight / ΣgenWeight, lumi 59.736 fb⁻¹ | same, 59.83 fb⁻¹ | same, 109.70 fb⁻¹ (processed lumi of the skims) | Same |
| DY sample | LO madgraphMLM HT-binned `DYJetsToLL_M-50` | same (UL), `dy_lo_ht` | LO madgraphMLM `DYto2L-4Jets` binned in HT × MLL (HT > 40), `dy_lo_ht` | Same type |
| DY cross sections | 2016 CUETP8M1 LO table × 1.23 in every year (SKFlat `CommonSampleInfo`) | bare UL18 LO (XSDB); Run 2 / ours per HT bin (70–100 … 2500+) = 1.49, 1.30, 1.31, 1.35, 1.34, 1.39, 1.43, 1.46. The reshape carries the normalization | LO × 1.16217 = MATRIX NNLO 6340.2 pb / LO MLM header since 2026-10-10 (before: LO × 1.223, normalized to the NLO FxFx inclusive 6671.5 pb; every measurement below used that) | **Differs** in where the normalization lives; the product is what matters |
| tt + tW | `TT_TW` | `TTTo2L2Nu` + dileptonic tW | `TTto2L2Nu` + dileptonic tW | Same |
| Nonprompt | W+jets + single top s/t (AN-19-083 Sec. 5) | W+jets HT-binned + single top s/t + **TTToSemiLeptonic** | W+jets HT-binned + single top s/t + **TTtoLNu2Q** + semileptonic tW | **Differs**: we add semileptonic tt (and tW in 2024) |
| Other | `Others` (VV, VVV, ttV) | VV, VVV, ttW, ttZ | VV, VVV, ttll, ttlν | Same |
| Signal | Official FullSim in the final limits (HNWR_Plotter `src/Make_ShapeForLimit.C:12`, `UseOfficial = true` since commit `6fe1824`, 2020-12-18): triggers, trigger data/MC SFs and the HEEP ID as for the background MC; normalized to LO × ⟨k⟩ with `AveragedKFactor_WROfficial.txt`; PDF and αS templates per region, but no QCD-scale templates for the official samples (written only when `UseOfficial` is false). The analyzer also runs the private FastSim grid that AN-19-083 describes (triggers forced true, trigger SF = data efficiency, FastSim electron IDs) | AN LO × ⟨k⟩ cross sections | private RunIII2024Summer24 FullSim production (237 points); since 2026-10-10 LO × its direct NLO/LO k-factor (+28–62 %; before: the LO LHE cross section with no k-factor) | Same convention since 2026-10-10 |

## Weights and corrections

| Correction | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
|---|---|---|---|---|
| Pileup | custom histogram in nPileUp, with an nPileUp ≥ 5 floor in 2018 | correctionlib on `Pileup.nTrueInt` (`Collisions18_UltraLegacy_goldenJSON`) | `Collisions24_BCDEFGHI_goldenJSON` | Same intent |
| L1 prefiring | 2016/2017 only; `GetPrefireWeight` returns 1 for 2018 | none | none | Same |
| Electron SFs | RECO and HEEP ID on tight electrons; HEEP trigger-OR efficiencies (`WREGammaTriggerEff.root`) combined per event with the OR formula | same components; the ported Run 2 HEEP efficiencies (`scripts/convert_run2_electron_trigger_sf.py`) | EGM `2024Prompt` RECO and HEEP ID (clipped to the last measured bin: 500 GeV RECO, 1 TeV ID); trigger SF from the EGM `HLT_SF_Ele30_TightID` proxy | Same for UL18 |
| Muon SFs | RECO (HighPt, vs p), ID, ISO on tight muons; Mu50 trigger SF | same components; leading-muon trigger SF | same; trigger SF from the efficiency OR formula | Same |
| Boosted loose lepton | No RECO/ID SF on the loose partner (it only enters the muon trigger SF) | RECO+ID on loose muons, RECO on loose electrons | as UL18 | Small difference; ours is the more complete treatment |
| LSF scale factor | Flat per year and jet flavour (2018: e-jet 1.05 +0.07/−0.06, μ-jet 1.04 +0.06/−0.05; AN-19-083 Table 60) on boosted SR and e-μ CR | ported (`lsf_sf`), by lepton-in-jet flavour; `--systs lsf` | none | Same for UL18; Run 3 needs its own measurement |
| DY Z-pT reweight | QCD NLO/LO ratio × EW correction vs generator Z pT, DY only, with QCD scale/PDF/αS and three EW variations. Events without a generator SFOS pair get 0.949 × 1.004 | ported (`wrcoffea/dy_reweight.py`); events without a pair get 1 | Run 3 derivation (study, via the config overlay; not the default), see [below](#run-3-dy-corrections-2024) | Same for UL18 (no-pair fallback differs) |
| DY reshape | Data/MC ratio from the DY CRs vs leading-jet pT (resolved) and the **selected** AK8 pT (boosted), ee + μμ combined (`DYReshapeJetPt_2018.root`, 1.25 at 40–100 GeV falling to 0.69 above 1 TeV). The analyzer's `DYReshapeSyst` shifts the ratio by its bin error, which `Get_DYReshapeRatioJetPt.C` builds from the data and MC statistics **and** 23 DY MC systematics (JES, JER, lepton SFs, Z-pT, pileup, ...) added in quadrature: ±0.08–0.11 per resolved bin, ±0.09–0.29 per boosted bin (8.5–11.6 % on DY, AN-19-083 Table 21). The ee-vs-μμ branch exists but is never invoked | `wrcoffea/dy_reshape.py`, payload from `scripts/derive_dy_reshape.py` (same binning, same variables: the boosted `pt_leading_AK8Jets` histogram is filled with the region's DY-candidate AK8); stat and ee-vs-μμ variations (`--systs dyreshape`). Level ~1.5 → 0.85 vs jet pT, because it carries the DY normalization | Run 3 derivation (study, via the config overlay; not the default), see [below](#run-3-dy-corrections-2024) | Same nominal for UL18; **the uncertainty is narrower**: our stat variation is 1–2 % at 100–600 GeV where Run 2's is 8–9 %, because it lacks the DY MC systematics |
| Signal k-factor | k(m_lN)/⟨k⟩ per mass point (`kfactor.root`, `AveragedKFactor.txt`; AN-19-083 App. A) | ported (`wrcoffea/signal_kfactor.py`), shape only; since 2026-10-10 the 13 TeV per-point curves (`signal_kfactor_13tev_20261010.json`) | since 2026-10-10 the 13.6 TeV per-point curves (`signal_kfactor_13p6tev_20261010.json`), shape only | Same |

## Fit model (Run 2)

Not part of the analyzer, recorded here for the closure: DY and tt+tW
normalizations are unconstrained rate parameters per (resolved/boosted, year);
nonprompt has a 100 % and other a 50 % lnN per region and year; the DY reshape
uncertainty is bin-by-bin and uncorrelated; MC statistics via Barlow-Beeston.

## Systematic sources

Run 2: JetRes, JetEn, JetMass (SD mass), MuonReco/En/ID/ISO/Trigger,
ElectronReco/Res/En/ID/Trigger, LSFSF, PU, ZPtRw (+QCD scale/PDF/αS, EW1–3),
Prefire, DYReshape (the ratio's bin error: statistics plus the DY MC
systematics), BTag, and the signal PDF and αS (`RunXsecSyst`; no QCD-scale
templates for the official signal).

WrCoffea families (`--systs`): `lumi`, `pileup`, `sf` (all lepton components),
`jer`, `jes`, `elescale`, `elesmear`, `muscale`, `musmear`, `dyzpt`, `lsf`,
`dyreshape`, `scale`, `ps`.  `dyzpt`, `dyreshape` and `lsf` exist only where
the era has a payload.  Not covered: the AK8 soft-drop mass scale, b-tagging (no
b-tag regions).  `muscale` is a flat 5 %/TeV TuneP placeholder and `musmear`
maps to nominal.  Prefiring is not needed for 2018 or 2024.

## Measured comparison

`WR_Plotter/signal_fitting/run2_results/compare_legacy_new.py` reads the Run 2
files and any WrCoffea sets (`label=[<era>/]<dir>`), scales every set to the
Run 2 luminosity and writes yields, closure and cutflow tables plus per-variable
plots.  It masks Run 3 signal-region data and the ee/μμ data cutflow rows from
m(ll) > 200 on, whatever the file holds (the 2024 `20261008_skims_v2` data
predates analyzer-side blinding and has them filled).  The numbers below are
from `dyreshape_20260907` (+ `_closure` for DY) for UL18 and `20261008_skims_v2`
for 2024.

Data/MC per region:

| Region | Run 2 | UL18 | 2024 |
|---|---|---|---|
| Resolved DY CR, ee / μμ | 0.99 / 1.01 | 1.00 / 1.00 | 1.21 / 1.25 |
| Resolved eμ CR (e / μ data) | 0.95 / 0.97 | 1.02 / 1.04 | 0.98 / 1.00 |
| Boosted DY CR, ee / μμ | 1.04 / 0.96 | 1.04 / 0.97 | 1.04 / 0.96 |
| Boosted eμ / μe CR | 0.75 / 0.93 | 0.76 / 0.93 | 1.00 / 0.99 |
| Resolved SR, ee / μμ | 1.02 / 1.01 | 1.08 / 1.04 | blinded |
| Boosted SR, ee / μμ | 1.01 / 0.90 | 1.14 / 0.92 | blinded |

* **UL18**: every MC group and the data sit at 0.80–0.86 of Run 2, uniformly
  through the cutflow from the 2 leptons + 2 jets stage on: the UL jet veto
  map.  The data shapes in the DY and eμ CRs agree within a few percent.  Our
  DY SR/CR ratio is 0.85 (ee) / 0.90 (μμ) of Run 2 while the data SR/CR agrees
  to 1 %, which is why the resolved SR closes at 1.08 / 1.04 rather than 1.02 /
  1.01: our raw UL DY is softer and the Run 2 Z-pT ratio, derived on the 2016
  HT weighting, over-corrects it.  The boosted ee CRs show the loose-electron
  H/E difference.
* **2024** (CRs only): per fb⁻¹, data at the 2 leptons + 2 jets + ΔR stage is
  0.92 (μμ), 0.95 (ee), 0.95–0.99 (eμ) of Run 2 and tt+tW MC 0.96–1.01,
  despite the ~11 % higher 13.6 TeV tt cross section.  The data shapes in the DY
  CRs match Run 2 within 5–10 % except electron η < −1.25 (the 2018 HEM veto)
  and the Z-peak shape.  The resolved DY CRs are the one disagreement: data/MC
  1.21 / 1.25 with a slope in leading-jet pT (~1.75 at 40–60 GeV to 1.0 near
  1 TeV) and in lepton pT.  Bare LO DY / 1.23 reproduces the Run 2 raw-DY data/MC
  vs leading-jet pT within ~5 % per bin, so this is the uncorrected LO MLM
  behaviour Run 2 removed with its two DY corrections.  The boosted DY CRs
  close (1.04 / 0.96) without them, as a cancellation.

## UL16, UL16APV and UL17 data against the Run 2 data (2026-10-11)

Event counts per fb⁻¹ in the data control regions of the first UL16/UL16APV/UL17
productions (`20261011_skims_v2`, skims_v2, brilcalc luminosities 19.50 + 16.81
and 42.04) against the Run 2 `Regions` histograms (35.92 and 41.53 fb⁻¹), with
UL18 (`20261010_skims_v2`, 59.52 vs 59.74) for reference.  The first pass had
the electron channel 1.5-2.2× too high: the EGamma group held the SingleElectron
and SinglePhoton primary datasets without overlap removal.  Run 2 used
SingleElectron only (and, in 2017, no Ele115), which the configs now follow.

| year | region | Run 2 /fb | ours /fb | ours/Run 2 |
|---|---|---|---|---|
| 2016 | resolved DY CR ee / mumu | 270 / 422 | 287 / 402 | 1.06 / 0.95 |
| 2016 | boosted DY CR ee / mumu | 31 / 31 | 32 / 36 | 1.00 / 1.16 |
| 2016 | resolved e-mu CR | 24 | 25 | 1.04 |
| 2017 | resolved DY CR ee / mumu | 296 / 444 | 282 / 406 | 0.96 / 0.92 |
| 2017 | boosted DY CR ee / mumu | 29 / 34 | 35 / 44 | 1.18 / 1.31 |
| 2017 | resolved e-mu CR | 27 | 25 | 0.94 |
| 2018 | resolved DY CR ee / mumu | 285 / 444 | 279 / 417 | 0.98 / 0.94 |
| 2018 | boosted DY CR ee / mumu | 28 / 31 | 33 / 40 | 1.17 / 1.29 |
| 2018 | resolved e-mu CR | 27 | 26 | 0.98 |

The boosted DY CR excess is the corrected-vs-raw soft-drop mass plus the UL/EOY
reconstruction difference already seen for UL18 (`local_notes/2026-10-10_ul18_boosted_dycr`).

## The ported DY Z-pT reweighting

Run 2 applied, for `DYJets` samples only and before any selection:

```
weight *= ZPtReweight(ptZ) * ZPtEWCorr(ptZ)
```

* `ptZ` is the generator-level Z pT from `GenFinderForDY`: the first two
  hard-process final-state electrons or muons (status 1) or decayed
  hard-process taus (status 2), required to be same flavour and opposite
  sign.  `wrcoffea.dy_reweight.find_gen_z` reproduces this from `GenPart`
  (`statusFlags` bit 8 is `fromHardProcess`).
* `ZPtReweight` is `h_Ratio_PDFError` of `HNWRDYPtReweight/ZPtQCDWithError.root`
  (28 bins, 0–8000 GeV; `ptZ ≥ 8000` evaluated at 7999).  The bin error is the
  PDF uncertainty; `h_Ratio_ScaleUp/Down` and `h_Ratio_AlphaSUp/Down` are the
  QCD scale and αS variations.  The ratio is about 0.95 below 20 GeV, 1.28 at
  70–90 GeV and falls to 0.28 above 2 TeV.  It was made by
  `Analyzers/src/GetZPtReweight.C` and `HNWR_Plotter/python/Get_ZPtReweight.py`:
  generator Z with m(Z) ≥ 50, histogrammed per sample at 1 pb⁻¹ with the SKFlat
  cross sections in the inclusive LO madgraphMLM sample (`DYJets_MG`) and the
  inclusive NLO amcatnloFXFX sample (`DYJets`), both 6077.22 pb in SKFlat, and
  divided as unit-normalized shapes (the ~0.95 at low pT is only possible with a
  full-phase-space LO denominator; the HT-binned set alone would give ≫ 1 there).
  The scale / αS variations are the varied NLO spectrum over the nominal LO, not
  renormalized.
* `ZPtEWCorr` is `hist_v` of `HNWRDYPtReweight/ZPtEWCorr.root` (42 bins,
  30–6500 GeV; `ptZ < 30` evaluated at 31, `ptZ ≥ 6500` at 6499), from 1.00 at
  low pT to 0.69 at 3 TeV.  `hist_e1/e2/e3` carry three uncertainty components
  as bin errors.

`scripts/convert_run2_dy_reweight.py` converts both ROOT files into
`data/reweights/RunII/2018/RunIISummer20UL18/dy_zpt_reweight_run2legacy_v4_2018.json`
(provenance recorded in the file).  `config.yaml` maps `RunIISummer20UL18` to
it; the analyzer applies the nominal weight to `DYJets` whenever the era has a
payload (`--no-dy-zpt-reweight` turns it off) and `--systs dyzpt` produces the
six Up/Down pairs.  Every MC dataset records `dy_zpt_reweight` in its
`dataset_meta` (`applied`, `identity`, `off`, or `unconfigured`).

## Our own Z-pT reweight for the UL eras (2026-10-11)

Derived with `scripts/derive_dy_zpt_reweight.py` the way AN-19-083 did it: the
inclusive NLO amcatnloFXFX sample over the inclusive LO madgraphMLM sample, both
at the NNLO cross section, generator Z with m(Z) ≥ 50, every file of both
samples (UL17: 103M LO + 196M NLO events), Run 2 binning, scale/PDF/αS
variations from the NLO LHE weights, the Run 2 EW block copied.  Payloads
`data/reweights/RunII/<year>/<era>/dy_zpt_reweight_ul_20261011.json`, mapped
for all four UL eras (UL18 switched from the converted Run 2 file).

| generator Z pT [GeV] | UL16APV | UL16 | UL17 | UL18 | Run 2 2016 | Run 2 2017 | Run 2 2018 |
|---|---|---|---|---|---|---|---|
| 0-2 | 0.94 | 0.94 | 0.94 | 0.94 | 0.97 | 0.95 | 0.95 |
| 20-30 | 1.09 | 1.11 | 1.09 | 1.09 | 1.05 | 1.10 | 1.09 |
| 50-60 | 1.24 | 1.29 | 1.25 | 1.24 | 1.09 | 1.24 | 1.26 |
| 100-120 | 1.35 | 1.39 | 1.35 | 1.34 | 1.17 | 1.21 | 1.26 |
| 250-350 | 1.33 | 1.34 | 1.33 | 1.34 | 1.06 | 0.97 | 0.97 |
| 500-1000 | 1.32 ± 0.07 | 1.19 ± 0.07 | 1.25 ± 0.05 | 1.19 ± 0.05 | 0.89 | 0.70 | 0.70 |
| 1000-2000 | 2.2 ± 0.7 | 0.8 ± 0.4 | 1.1 ± 0.3 | 1.5 ± 0.4 | 0.67 | 0.46 | 0.45 |

The integrated NLO/LO is 0.996-0.999 in every era (shape-only weight); ee and
μμ agree within 1 %.  Up to ~60 GeV the four eras and the Run 2 files agree;
above 100 GeV our ratio stays at 1.3-1.4 where Run 2's falls below 1.  The
Run 2 LO/NLO input histograms are not in the cloned repositories, so what made
their ratio fall is not known.  Above 1 TeV the inclusive LO sample holds a few
tens of events per era, so that bin is statistically weak in any derivation of
this kind (Run 2's included); the empty 2-8 TeV bin takes the 1-2 TeV value.

## Our own jet-pT reshapes for the UL eras (2026-10-11)

`scripts/derive_dy_reshape.py` per era: (data − non-DY MC) / DY MC in the DY
CRs, ee + μμ combined, DY with the own Z-pT payload and no reshape, Run 2
DYReshapeJetPt bins (resolved 40, 100, 200, 400, 600, 800, 1000, 2000 GeV;
boosted 200, 400, 600, 800, 1000, 2000 GeV; the first bin is empty by
construction).  Payloads `dy_reshape_run2jetpt_20261011.json` per era, mapped.
Run 2 values from `SKFlatAnalyzer/data/Run2Legacy_v4/<year>/HNWRDYReshape/DYReshapeJetPt_<year>.root`
(`Resolved_ratio_AllCh`, `Boosted_ratio_AllCh`).

| year | resolved: ours (per era) | resolved: Run 2 | boosted: ours | boosted: Run 2 |
|---|---|---|---|---|
| 2016 | UL16APV 1.31, 1.10, 0.95, 0.84, 0.72, 0.81, 0.75; UL16 1.48, 1.14, 0.95, 0.86, 0.72, 0.77, 0.64 | 1.41, 1.24, 1.09, 0.98, 0.84, 0.90, 0.76 | 0.85, 0.69, 0.50, 0.35, 0.12; 0.82, 0.82, 0.80, 0.59, 0.64 | 0.92, 0.86, 0.76, 0.57, 0.24 |
| 2017 | 1.33, 1.13, 0.97, 0.87, 0.77, 0.68, 0.78 | 1.32, 1.23, 1.09, 1.00, 0.87, 0.79, 0.83 | 0.96, 0.88, 0.73, 0.67, 0.61 | 1.11, 1.14, 0.99, 0.85, 0.95 |
| 2018 | 1.47, 1.18, 1.01, 0.90, 0.82, 0.75, 0.57 | 1.25, 1.13, 1.03, 0.95, 0.86, 0.85, 0.69 | 0.97, 0.82, 0.74, 0.48, 0.71 | 0.98, 0.92, 0.98, 0.63, 0.98 |

Same shape as Run 2 (a 1.3-1.5 excess of data at 40-100 GeV falling to
0.6-0.8 above 1 TeV); our resolved ratio sits 5-10 % below Run 2's in the
middle bins because our Z-pT ratio is higher there.  The boosted ratios fall
further than Run 2's at high jet pT, where the corrected-vs-raw soft-drop mass
choice acts (parked).  The two UL16 eras differ by their statistics.

## Run 3 DY corrections (2024)

### Z-pT reweight

> **Superseded 2026-10-11.** Billy asked for the derivation to be done exactly
> as AN-19-083 did it: the inclusive NLO sample over the inclusive LO sample, no
> binned samples and no stitching.  `scripts/derive_dy_zpt_reweight.py` was
> rewritten to that recipe (roles `lo` = `dy_lo_inc`, `nlo` = `dy_nlo_inc`,
> MLL-50 only, 10M events per sample); the UL eras were derived with it the same
> day, and the 2024 payload described below (stitched: HT-binned LO + PTLL-binned
> NLO) is to be re-derived with it before any further use.  The text below is
> kept as the record of what the 2026-10-08/10 payload contains.

The 2026-10-08 version of `scripts/derive_dy_zpt_reweight.py` repeated the Run 2
`GetZPtReweight.C` recipe on the 2024 samples with stitched inputs and wrote a
payload in the format of the Run 2 conversion, `data/reweights/Run3/2024/RunIII2024Summer24/dy_zpt_reweight_run3_20261008.json`,
mapped in `config.yaml` (`dy_zpt_reweight: RunIII2024Summer24`).

* `fill` reads generator information from unskimmed NanoAOD over XRootD (the
  skims' reco selection would bias the generator spectrum): ~500 k events per
  dataset spread over up to 8 files, 22 datasets, ~5 min on the laptop.  The
  per-dataset spectra are cached (`--workdir`; the 2026-10-08 cache is in
  `~/work/skim_scratch/dy_zpt_spectra/RunIII2024Summer24`).  `combine` stitches
  and divides.
* Generator Z from `find_gen_z` (the analyzer's own `GenFinderForDY`), m(Z) ≥ 50,
  ee + μμ only (the NLO inclusive samples are `DYto2E` / `DYto2Mu`), Run 2 binning.
* LO = the 14 `dy_lo_ht` samples at their configured cross sections (generator
  LO × 1.223 at the time; the 2026-10-10 NNLO normalization moved LO and NLO by the
  same ×0.950, and the payload re-derived then, `dy_zpt_reweight_run3_20261010.json`,
  is identical bin by bin) plus the LO inclusive `DYto2E/2Mu-4Jets_Bin-MLL-50` below
  LHE_HT = 40, so the denominator is the full LO spectrum although the analysis
  sample has no events there.
* NLO = amcatnloFXFX `DYto2E/2Mu-2Jets_Bin-MLL-50` below LHE_Vpt = 100 plus
  `DYto2L-2Jets_Bin-MLL-50-PTLL-{100,200,400,600}`.  These are **open-ended**
  (PTLL-X means pT(ll) > X), so each is restricted to [X, next X).  The
  analyzer's own `dy_nlo_ptll` variant does not do this: it adds the four
  overlapping samples (DY above 200 GeV counted 2–4 times) and has nothing
  below 100 GeV, so it should not be used as it stands.
* Variations on the NLO numerator: 7-point scale envelope (`LHEScaleWeight`
  0,1,3,5,7,8), Hessian PDF (`LHEPdfWeight` 1–100 of LHA 325300), αS (101/102).
  The EW block is the Run 2 13 TeV `ZPtEWCorr`, copied (the EW Sudakov factor
  depends weakly on √s).
* Checks (in the payload provenance): integrated NLO/LO = 0.999; inclusive vs
  binned in every overlapping slice: LO 1.00 ± 0.01 (HT 40–400), NLO 1.01 ± 0.02
  (pT(ll) 100–200), 1.10 ± 0.05 (200–400), compatible within statistics above;
  ee and μμ ratios agree within 2–3 %.

| generator Z pT [GeV] | 0–2 | 2–4 | 10–20 | 20–30 | 30–100 | 100–200 | 200–350 | 350–500 | 500–1000 | 1000–2000 | 2000–8000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2024 NLO/LO | 0.69 | 0.80 | 0.96–1.01 | 1.13 | 1.23–1.32 | 1.27–1.31 | 1.24–1.26 | 1.16 | 1.02 | 0.84 | 0.42 ± 0.16 |
| Run 2 2018 | 0.95 | 0.94 | 0.92–0.97 | 1.09 | 1.22–1.28 | 1.14–1.26 | 0.97–1.10 | 0.86 | 0.70 | 0.46 | 0.28 |

Between 5 and 100 GeV the two agree; above ~150 GeV the 2024 ratio falls far
less.  The Run 2 denominator carried the 2016 per-HT-bin cross sections, so its
ratio is tied to that HT weighting; 2024 raw DY is flatter in Z pT, and the Run 2
ratio copied to 2024 would over-correct the tail.  The last bin is statistics
limited.

With the Z-pT weight on (DY rerun `20261008_dyzpt`), the 2024 DY grows by
~16 % in the resolved and ~5 % in the boosted DY CRs, and the CR closure moves
from 1.21 / 1.25 to 1.05 / 1.09 (resolved ee / μμ) and from 1.04 / 0.96 to
0.99 / 0.93 (boosted).  The lepton-pT and η trends of the uncorrected DY flatten
out; the leading-jet pT slope stays (~1.85 at 40–60 GeV, 1.45 at 80–100 GeV,
1.0 above 300 GeV), as it did in Run 2 before its reshape.

### Reshape

`scripts/derive_dy_reshape.py --dy-rootfiles` takes the data and non-DY
backgrounds from `20261008_skims_v2` and the Z-pT-reweighted DY from
`20261008_dyzpt` (Run 2 binning, ee + μμ combined):
`data/reweights/Run3/2024/RunIII2024Summer24/dy_reshape_run2jetpt_20261008.json`.

| leading AK4 pT [GeV] | 40–100 | 100–200 | 200–400 | 400–600 | 600–800 | 800–1000 | 1000–2000 |
|---|---|---|---|---|---|---|---|
| 2024 (ee+μμ) | 1.54 ± 0.03 | 1.26 | 1.05 | 0.92 | 0.84 | 0.74 ± 0.05 | 0.68 ± 0.07 |
| 2024 ee / μμ | 1.54 / 1.53 | 1.24 / 1.28 | 1.02 / 1.06 | 0.90 / 0.94 | 0.85 / 0.84 | 0.66 / 0.79 | 0.66 / 0.69 |
| Run 2 2018 | 1.25 | | | | | | 0.69 (> 1 TeV) |

| DY-candidate AK8 pT [GeV] | 200–400 | 400–600 | 600–800 | 800–1000 | 1000–2000 |
|---|---|---|---|---|---|
| 2024 (ee+μμ) | 1.05 ± 0.02 | 0.91 | 0.82 | 0.71 ± 0.09 | 0.63 ± 0.14 |
| 2024 ee / μμ | 1.09 / 1.01 | 0.93 / 0.88 | 0.87 / 0.79 | 0.78 / 0.66 | 0.78 / 0.54 |

The 40–100 GeV bin averages a data/MC that falls from ~1.85 to ~1.45 inside it,
so the Run 2 binning cannot close the lowest jet-pT bins exactly.

### Comparison with the AN-19-083 reshape

AN-19-083 Sec. 5.1 (pp. 28–51): after the Z-pT correction "the simulation
predicts harder jets than is observed in the data", attributed to the missing
loop corrections of the LO sample and visible in m(lljj) and m(lJ).  The fix is
the bin-by-bin (Data − non-DY)/DY ratio vs leading-jet pT in the DY sidebands,
ee + μμ combined because the effect should not depend on the lepton flavour,
applied to DY in every region (Fig. 24).  Its uncertainty is the statistical
and systematic uncertainty of the ratio, uncorrelated across bins (8.5–11.6 % on
DY, Table 21).  Two validations: the ratio derived in DY CR1 (60 < m(ll) < 100)
improves data/MC in DY CR2 (100–150), Figs. 27–30; with an NLO DY sample as
pseudo-data, NLO/LO agrees with Data/LO in the CRs (Fig. 31) and the CR and SR
NLO/LO agree within 2σ (Fig. 32).  The DY rate parameters then fit to 1.00 ±
0.07–0.16 (Table 19).

Exact 2018 values (`DYReshapeJetPt_2018.root`, errors stat + syst) next to 2024:

| leading AK4 pT [GeV] | 40–100 | 100–200 | 200–400 | 400–600 | 600–800 | 800–1000 | 1000–2000 |
|---|---|---|---|---|---|---|---|
| Run 2 2018 Data/LO | 1.25 ± 0.11 | 1.13 | 1.03 | 0.95 | 0.86 | 0.85 | 0.69 ± 0.11 |
| 2024 Data/LO (Z-pT on) | 1.54 ± 0.03 | 1.26 | 1.05 | 0.92 | 0.84 | 0.74 | 0.68 ± 0.07 |
| 2024 NLO/LO (Z-pT on) | 1.31 ± 0.05 | 1.21 | 1.08 | 1.03 | 0.86 | 0.75 | 0.65 ± 0.09 |

(2018 boosted: 0.98, 0.92, 0.98, 0.63, 0.98 in the AK8 bins of the table above.)

* **Same problem**: the 2024 LO DY is too hard in jet pT after the Z-pT weight,
  ee and μμ agree (flavour independent), and the high-pT end matches Run 2
  (0.68 vs 0.69 above 1 TeV).  The 2024 NLO FxFx sample reproduces Data/LO from
  100 GeV up, as Run 2 found, and its shape is close to the Run 2 Data/LO.
* **What is new**: at 40–100 GeV 2024 Data/LO is 1.54 where NLO/LO is 1.31, so
  data has ~18 % more soft-leading-jet events than even NLO predicts (Data/NLO =
  1.18, 1.04, 0.97, 0.89, 0.98, 0.99, 1.05 across the bins).  Run 2 did not see
  this.  Its origin (jet response or pileup jets at low pT, or the generator) is
  open; the data-driven reshape absorbs it in the CR either way.
* **Transfer to the SR** (Fig. 32 analogue, NLO as pseudo-data,
  `20261008_skims_v2_nlo_inc_dy` over `20261008_dyzpt`): LO SR × (NLO/LO)
  per jet-pT bin from the CR predicts the NLO SR yield to 1.02 ± 0.04 resolved
  (ee 1.04, μμ 1.01; a flat CR scale gives 0.96) and 0.91 ± 0.08 boosted (flat:
  0.90).  This validates the NLO-like part of the correction only; the extra
  low-pT component is untested in the SR.
* **CR1 → CR2 test** (study `--study dy-cr-split`, sets `20261008_dycr_split` and
  `20261008_dycr_split_cr1reshape`): CR2 (100 < m(ll) < 150) sits 5–7 % above CR1 at
  leading-jet pT 100–400 GeV (2.3–2.5σ), as in Run 2 (3–11 %).  The reshape derived
  in CR1 brings the resolved CR2 from data/MC 1.09 to 1.03 (event by event: ee 1.075
  → 1.012, μμ 1.107 → 1.043; m(lljj) flat within ~5 % below 2 TeV); Run 2's CR2 after
  its reshape is 1.03.  Boosted CR2: 1.05 → 1.08 (Run 2: 1.08), statistics limited.
  The paper's intermediate-m(ll) check (150 < m(ll) < 400 / 200 GeV) is not done yet.
* **Fix, ours vs Run 2**: the same ratio, binning, variables and ee + μμ
  combination.  Run 2's analyzer variation is the full quoted uncertainty: it
  shifts the ratio by its bin error, which holds the statistics and the DY MC
  systematics (~±0.1 per bin; the ee-vs-μμ branch is never invoked).  Ours
  provides stat and the ee-vs-μμ difference (`--systs dyreshape`), so it is
  narrower.  At 2024 statistics the stat part is ±0.01–0.07, so the
  systematic part (and an SR-transfer term from the test above, ~4 % resolved,
  ~10 % boosted) dominates and should be added.

### Closure and how to run with the corrections

Neither payload is mapped in `config.yaml` (every run in every session reads
it).  They are switched on per process with the overlay
`data/reweights/Run3/2024/RunIII2024Summer24/dy_corrections_study_overlay.yaml`:

```
WRCOFFEA_CONFIG_OVERLAY=data/reweights/Run3/2024/RunIII2024Summer24/dy_corrections_study_overlay.yaml \
    python bin/run_analysis.py RunIII2024Summer24 DYJets --dir <tag> ...
```

DY rerun with both corrections, `20261008_dyzpt_reshape` (data and non-DY from
`20261008_skims_v2`):

| Region | uncorrected | Z-pT | Z-pT + reshape | Run 2 |
|---|---|---|---|---|
| Resolved DY CR data/MC, ee / μμ | 1.21 / 1.25 | 1.05 / 1.09 | 0.98 / 1.01 | 0.99 / 1.01 |
| Boosted DY CR data/MC, ee / μμ | 1.04 / 0.96 | 0.99 / 0.93 | 1.03 / 0.97 | 1.04 / 0.96 |
| Resolved SR DY (MC), ee / μμ | 496 / 627 | 580 / 734 | 711 / 900 | — |
| Boosted SR DY (MC), ee / μμ | 124 / 119 | 148 / 144 | 142 / 137 | — |

In the resolved DY CRs data/MC is flat at ~1.0 in lepton pT and η, jet pT,
m(ll) and m(lljj) after both corrections (Run 2 kept a 0.85 → 1.2 slope in
lepton pT); the leading-jet bins at 40–60 GeV stay at 1.1–1.2 because the
40–100 GeV reshape bin averages over them.  The full correction scales the
resolved DY by 1.25 in the CRs and 1.43 in the SRs, an SR/CR ratio of 1.15
(Run 2, corrected over raw DY files: 1.12).  In the boosted regions it is 1.00
in the CRs and 1.15 in the SRs; Run 2's Z-pT weight suppressed the
high-Z-pT boosted CR (0.75) far more than ours.

Open items before mapping the corrections by default:

* the intermediate-m(ll) check of the paper (150 < m(ll) < 400 / 200 GeV; CR1 → CR2 is done, see above);
* re-deriving the reshape after the 2026-10-10 NNLO DY normalization (the payloads absorbed the old one, now 5.2 % low);
* a reshape systematic beyond the stat and ee-vs-μμ variations (Run 2 varied
  the ratio by stat ⊕ DY MC systematics, ~±0.1 per bin), plus an SR-transfer
  term (~4 % resolved, ~10 % boosted from the NLO pseudo-data test);
* finer binning below 100 GeV, or a smooth fit, for the soft-leading-jet excess;
* a 13.6 TeV EW correction instead of the copied 13 TeV one;
* the same derivation for 2022 and 2023 (the script is era-generic; it needs
  each era's `dy_lo_ht`, `dy_lo_inc`, `dy_nlo_inc` and `dy_nlo_ptll` configs);
* stitching for the `dy_nlo_ptll` variant before it is used anywhere.
