# WrCoffea versus the Run 2 HNWRAnalyzer

What this page answers: where this analyzer differs from the Run 2 analysis code
for UL18 and 2024, object by object, cut by cut and weight by weight, and how the
Run 2 DY corrections compare with our own.  Verified on 2026-10-12.

The Run 2 WR analysis code is
[jedori0228/SKFlatAnalyzer](https://github.com/jedori0228/SKFlatAnalyzer),
branch `HNWR/Run2Legacy_v4` (commit `06f274c`, "final", 2022-05-13):
`Analyzers/src/HNWRAnalyzer.C` plus the object IDs in `DataFormats/src`.
That branch produced the `HNWRAnalyzer_SkimTree_LRSMHighPt_*` files under
`WR_Plotter/signal_fitting/run2_results/` (the `ReweightedQCDErrorEWCorr` and
`Reshaped` suffixes are the two DY corrections described below).  Run 2 ran on
the pre-UL 2018 samples (its sample table has 11.5 M events of
`DYJetsToLL_M-50_HT-100to200`; UL18 has 26.2 M).

The UL18 column is what matters for a Run 2 closure.  In the 2024 column,
differences that are era updates (triggers, POG payloads) are noted but not
treated as discrepancies; "as UL18" means the same code path.  The script that
measures the differences in yields is described in
[Measured comparison](#measured-comparison).

## Objects

| Object | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
| --- | --- | --- | --- | --- |
| Electron kinematics | pT > 53, \|SC η\| < 2.4 (`SelectElectrons("HNWRLT", 53, 2.4)`) | pT > 53, \|η\| < 2.5, ECAL crack veto on SC η | as UL18 | **Differs**: η range 2.4 vs 2.5; Run 2 cuts on SC η |
| Electron energy | MiniAOD EGM scale and smearing | already applied in UL NanoAOD | EGM `electronSS_EtDependent` applied in the analyzer | Same intent |
| Electron tight | HEEP (`HNWRTight2018Prompt` for 2018) | `cutBased_HEEP` | as UL18 | Same |
| Electron loose | `HNWRLoose` (`DataFormats/src/Electron.C:401`): the POG Fall17 loose cuts (σiηiη, ΔηSeed, Δφin, 1/E−1/p, ≤ 1 missing hit, conversion veto) with the POG H/E cut replaced by **H/E < 0.5**, no isolation | Boosted: every `vidNestedWPBitmap` cut at Loose except isolation, so the POG H/E cut is kept, OR HEEP. Resolved: `cutBased == 2` (unused by any region) | as UL18 | **Differs**: H/E. Likely cause of the ee boosted DY CR deficit (nonprompt 0.39×, tt 0.63× of Run 2); not yet tested |
| Muon momentum | TuneP; Rochester below 200 GeV and the Generalized-Endpoint bias correction above 200 GeV on MC (`AnalyzerCore.C` `UseTunePMuon`), no smearing. SFs evaluated at the uncorrected MiniAOD TuneP pT | TuneP (`pt × tunepRelPt`), **uncorrected**; SFs at that pT | TuneP; MuonScaRe below 200 GeV, the Generalized-Endpoint correction on data and extra smearing on MC above (`muon_highpt`, [muon_momentum.md](muon_momentum.md)) | **Differs**: UL18 has no momentum correction; 2024 follows the Run 3 Muon POG recipe (GE applied to data, not to MC) |
| Muon kinematics | pT > 53, \|η\| < 2.4 | same, on TuneP pT | as UL18 | Same |
| Muon tight | POG HighPt ID and TrkIso/TuneP pT < 0.1 | `highPtId == 2` and `tkRelIso < 0.1` | as UL18 | Same |
| Muon loose | POG HighPt ID | `highPtId == 2` | as UL18 | Same |
| AK4 jets | TightLepVeto, pT > 40, \|η\| < 2.4, no lepton cleaning (`JetsVetoLeptonInside` exists but is never called); ΔR > 0.4 is a selection cut | same; JEC `Summer19UL18_V5`, JER `Summer19UL18_JRV3` (the JRV2 scale factors, retagged) | same; jet ID from the JME `jetid.json`, JEC `Summer24Prompt24_V5`, JER `JRV2` | Same |
| Jet veto map | none | `Summer19UL18_V1`, data and MC, whole year; covers 97 % of the HEM box. Jets in vetoed cells are dropped and the event is kept (`jetveto_mode: jet`; jets with a PF muon within ΔR < 0.2 are spared) | `Summer24Prompt24_RunBCDEFGHI_V1`: 3.7 % of the \|η\| < 2.4 acceptance, outside the HEM box. The event is rejected (the JME Run 3 rule; 13–15 % of 2024 events) | **Differs**: Run 2 had no map. Until 2026-10-10 UL18 also rejected the event, which cost it 17–20 % of the events against Run 2 |
| AK8 jets | Tight ID, SD mass > 40 on the uncorrected-subjet mass, pT > 200, \|η\| < 2.4; SR additionally LSF > 0.75 | Same thresholds, but SD mass from NanoAOD's corrected `msoftdrop` (`ak8_msoftdrop_mode: raw` cuts on the uncorrected mass instead); SR uses `lsf3 > 0.75` | as UL18 | **Differs**: the SD-mass definition, about 9 % of the UL18 boosted DY CR yield |
| b-tagging | DeepCSV medium, used to split SR and e-μ CRs into BJet / NoBJet | none | none | **Missing** (only needed for the b-jet sub-regions, which are not part of the fit) |

## Event selection

| Step | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
| --- | --- | --- | --- | --- |
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
| --- | --- | --- | --- | --- |
| Normalization | xsec × lumi × genWeight / ΣgenWeight, lumi 59.736 fb⁻¹ | same, 59.52 fb⁻¹ (processed lumi of the skims) | same, 109.70 fb⁻¹ (processed lumi of the skims) | Same |
| DY sample | LO madgraphMLM HT-binned `DYJetsToLL_M-50` | same (UL), `dy_lo_ht` | LO madgraphMLM `DYto2L-4Jets` binned in HT × MLL (HT > 40), `dy_lo_ht` | Same type |
| DY cross sections | 2016 CUETP8M1 LO table × 1.23 in every year (SKFlat `CommonSampleInfo`) | LO × 1.136 = FEWZ NNLO 6077.22 pb / LO MLM inclusive header since 2026-10-10 ([ul_xsec_campaign.md](ul_xsec_campaign.md); before: bare LO) | LO × 1.16217 = MATRIX NNLO 6340.2 pb / LO MLM header since 2026-10-10 (before: LO × 1.223, normalized to the NLO FxFx inclusive 6671.5 pb) | **Differs** in where the normalization lives (flat 1.23 vs an NNLO k-factor); the product with the reshape is what matters |
| tt + tW | `TT_TW` | `TTTo2L2Nu` + dileptonic tW | `TTto2L2Nu` + dileptonic tW | Same |
| Nonprompt | W+jets + single top s/t (AN-19-083 Sec. 5) | W+jets HT-binned + single top s/t + **TTToSemiLeptonic** | W+jets HT-binned + single top s/t + **TTtoLNu2Q** + semileptonic tW | **Differs**: we add semileptonic tt (and tW in 2024) |
| Other | `Others` (VV, VVV, ttV) | VV, VVV, ttW, ttZ | VV, VVV, ttll, ttlν | Same |
| Signal | Official FullSim in the final limits (HNWR_Plotter `src/Make_ShapeForLimit.C:12`, `UseOfficial = true` since commit `6fe1824`, 2020-12-18): triggers, trigger data/MC SFs and the HEEP ID as for the background MC; normalized to LO × ⟨k⟩ with `AveragedKFactor_WROfficial.txt`; PDF and αS templates per region, but no QCD-scale templates for the official samples (written only when `UseOfficial` is false). The analyzer also runs the private FastSim grid that AN-19-083 describes (triggers forced true, trigger SF = data efficiency, FastSim electron IDs) | official UL FullSim samples (105 points per flavour); since 2026-10-10 LO × the point's direct NLO/LO k-factor (before: the AN LO × ⟨k⟩) | private RunIII2024Summer24 FullSim production (237 points); since 2026-10-10 LO × its direct NLO/LO k-factor (+28–62 %; before: the LO LHE cross section with no k-factor) | Close: the direct per-point k instead of ⟨k⟩, within 1–3 % at the shared points ([signal_kfactor.md](signal_kfactor.md)) |

## Weights and corrections

| Correction | Run 2 HNWRAnalyzer (2018) | WrCoffea UL18 | WrCoffea 2024 | Status |
| --- | --- | --- | --- | --- |
| Pileup | custom histogram in nPileUp, with an nPileUp ≥ 5 floor in 2018 | correctionlib on `Pileup.nTrueInt` (`Collisions18_UltraLegacy_goldenJSON`) | `Collisions24_BCDEFGHI_goldenJSON` | Same intent |
| L1 prefiring | 2016/2017 only; `GetPrefireWeight` returns 1 for 2018 | NanoAOD `L1PreFiringWeight` (`prefiring_eras`; muon part only in 2018, a few per mille); `--systs prefire` | none | **Differs** for UL18 by a few per mille |
| Electron SFs | RECO and HEEP ID on tight electrons; HEEP trigger-OR efficiencies (`WREGammaTriggerEff.root`) combined per event with the OR formula | same components; the ported Run 2 HEEP efficiencies (`scripts/convert_run2_electron_trigger_sf.py`) | EGM `2024Prompt` RECO and HEEP ID (clipped to the last measured bin: 500 GeV RECO, 1 TeV ID); trigger SF from the EGM `HLT_SF_Ele30_TightID` proxy | Same for UL18 |
| Muon SFs | RECO (HighPt, vs p), ID, ISO on tight muons; Mu50 trigger SF | same components; leading-muon trigger SF | same; trigger SF from the efficiency OR formula | Same |
| Boosted loose lepton | No RECO/ID SF on the loose partner (it only enters the muon trigger SF) | RECO+ID on loose muons, RECO on loose electrons | as UL18 | Small difference; ours is the more complete treatment |
| LSF scale factor | Flat per year and jet flavour (2018: e-jet 1.05 +0.07/−0.06, μ-jet 1.04 +0.06/−0.05; AN-19-083 Table 60) on boosted SR and e-μ CR | ported (`lsf_sf`), by lepton-in-jet flavour; `--systs lsf` | placeholder 1.00 ± 0.10 per flavour (since 2026-10-11) | Same for UL18; Run 3 needs its own measurement |
| DY Z-pT reweight | QCD NLO/LO ratio × EW correction vs generator Z pT, DY only, with QCD scale/PDF/αS and three EW variations. Events without a generator SFOS pair get 0.949 × 1.004 | own derivation the AN-19-083 way (`dy_zpt_reweight_ul_20261011.json`, [below](#our-own-z-pt-reweight-for-the-ul-eras-2026-10-11)); the converted Run 2 payload stays for reference; `wrcoffea/dy_reweight.py`; events without a pair get 1 | the same derivation on the 2024 samples (`dy_zpt_reweight_run3_20261011.json`), mapped by default; 2022 and 2023 share one payload | Same method; above 100 GeV our ratio stays at 1.3–1.4 where Run 2's falls below 1; the no-pair fallback differs |
| DY reshape | Data/MC ratio from the DY CRs vs leading-jet pT (resolved) and the **selected** AK8 pT (boosted), ee + μμ combined (`DYReshapeJetPt_2018.root`, 1.25 at 40–100 GeV falling to 0.69 above 1 TeV). The analyzer's `DYReshapeSyst` shifts the ratio by its bin error, which `Get_DYReshapeRatioJetPt.C` builds from the data and MC statistics **and** 23 DY MC systematics (JES, JER, lepton SFs, Z-pT, pileup, ...) added in quadrature: ±0.08–0.11 per resolved bin, ±0.09–0.29 per boosted bin (8.5–11.6 % on DY, AN-19-083 Table 21). The ee-vs-μμ branch exists but is never invoked | `wrcoffea/dy_reshape.py`, payload from `scripts/derive_dy_reshape.py` (same binning, same variables: the boosted `pt_leading_AK8Jets` histogram is filled with the region's DY-candidate AK8); stat and ee-vs-μμ variations (`--systs dyreshape`). Payload `dy_reshape_run2jetpt_20261011.json`, values [below](#our-own-jet-pt-reshapes-for-the-ul-eras-2026-10-11) | the same derivation, one payload per data-taking year from the 2026-10-11 production (`dy_reshape_run2jetpt_20261011.json`), mapped by default | Same nominal for UL18; **the uncertainty is narrower**: our stat variation is 1–2 % at 100–600 GeV where Run 2's is 8–9 %, because it lacks the DY MC systematics |
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

WrCoffea families (`--systs`): `lumi`, `pileup`, `prefire`, `sf` (all lepton
components), `jer`, `jes`, `elescale`, `elesmear`, `muscale`, `musmear`, `dyzpt`,
`lsf`, `dyreshape`, `scale`, `ps`, `pdf`.  A family exists only where the era
has its payload: `prefire` in the UL eras, `dyzpt`, `dyreshape` and `lsf` where
they are mapped; the UL eras have no `elescale` / `elesmear`
([analysis.md](../analysis.md#systematics)).  Not covered: the AK8 soft-drop
mass scale, b-tagging (no b-tag regions).  In 2024 `muscale` and `musmear`
follow the Muon POG recipe ([muon_momentum.md](muon_momentum.md)); UL18 keeps
the TuneP placeholder (flat 5 %/TeV scale, smearing equal to nominal).

## Measured comparison

`WR_Plotter/signal_fitting/run2_results/compare_legacy_new.py` reads the Run 2
files and any WrCoffea sets (`label=[<era>/]<dir>`), scales every set to the
Run 2 luminosity and writes yields, closure and cutflow tables plus per-variable
plots.  It masks Run 3 signal-region data and the ee/μμ data cutflow rows from
m(ll) > 200 on, whatever the file holds.  Run it on the current productions for
numbers; dated results are not kept on this page.

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
(provenance recorded in the file).  `config.yaml` mapped `RunIISummer20UL18` to
it until 2026-10-11, when our own derivation (next section) replaced it; the
file stays for reference.  The analyzer applies the nominal weight of the
mapped payload to the DY datasets named by `dy_reweight_datasets`
(`--no-dy-zpt-reweight` turns it off), and `--systs dyzpt` produces the Up/Down
pairs.  Every MC dataset records `dy_zpt_reweight` in its `dataset_meta`
(`applied`, `identity`, `not_for_sample`, `off`, or `unconfigured`).

## Our own Z-pT reweight for the UL eras (2026-10-11)

Derived with `scripts/derive_dy_zpt_reweight.py` the way AN-19-083 did it: the
inclusive NLO amcatnloFXFX sample over the inclusive LO madgraphMLM sample, both
at the NNLO cross section, generator Z with m(Z) ≥ 50, every file of both
samples (UL17: 103M LO + 196M NLO events), Run 2 binning, scale/PDF/αS
variations from the NLO LHE weights, the Run 2 EW block copied.  Payloads
`data/reweights/RunII/<year>/<era>/dy_zpt_reweight_ul_20261011.json`, mapped
for all four UL eras (UL18 switched from the converted Run 2 file).

| generator Z pT [GeV] | UL16APV | UL16 | UL17 | UL18 | Run 2 2016 | Run 2 2017 | Run 2 2018 |
| --- | --- | --- | --- | --- | --- | --- | --- |
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
construction).  Payloads `dy_reshape_run2jetpt_20261011.json` per era, mapped;
UL16, UL17 and UL18 were re-derived the same day on the prefiring-weighted
backgrounds.  The values below are those of the committed payloads.
Run 2 values from `SKFlatAnalyzer/data/Run2Legacy_v4/<year>/HNWRDYReshape/DYReshapeJetPt_<year>.root`
(`Resolved_ratio_AllCh`, `Boosted_ratio_AllCh`).

| year | resolved: ours (per era) | resolved: Run 2 | boosted: ours | boosted: Run 2 |
| --- | --- | --- | --- | --- |
| 2016 | UL16APV 1.41, 1.15, 0.99, 0.86, 0.75, 0.83, 0.77; UL16 1.60, 1.20, 0.99, 0.89, 0.74, 0.79, 0.66 | 1.41, 1.24, 1.09, 0.98, 0.84, 0.90, 0.76 | 0.89, 0.72, 0.52, 0.37, 0.13; 0.87, 0.86, 0.83, 0.62, 0.67 | 0.92, 0.86, 0.76, 0.57, 0.24 |
| 2017 | 1.46, 1.21, 1.02, 0.91, 0.80, 0.70, 0.80 | 1.32, 1.23, 1.09, 1.00, 0.87, 0.79, 0.83 | 1.02, 0.92, 0.76, 0.71, 0.64 | 1.11, 1.14, 0.99, 0.85, 0.95 |
| 2018 | 1.48, 1.18, 1.01, 0.90, 0.82, 0.75, 0.58 | 1.25, 1.13, 1.03, 0.95, 0.86, 0.85, 0.69 | 0.97, 0.82, 0.74, 0.49, 0.71 | 0.98, 0.92, 0.98, 0.63, 0.98 |

Same shape as Run 2 (a 1.4-1.6 excess of data at 40-100 GeV falling to
0.6-0.8 above 1 TeV); our resolved ratio sits 2-12 % below Run 2's in the
middle bins because our Z-pT ratio is higher there.  The boosted ratios fall
further than Run 2's at high jet pT, where the corrected-vs-raw soft-drop mass
choice acts (parked).  The two UL16 eras differ by their statistics.

## The Run 2 DY reshape (AN-19-083)

AN-19-083 Sec. 5.1 (pp. 28–51): after the Z-pT correction the LO simulation
still has harder jets than the data, which the note attributes to the missing
loop corrections of the LO sample; it shows in m(lljj) and m(lJ).  The fix is
the bin-by-bin (Data − non-DY)/DY ratio vs leading-jet pT in the DY sidebands,
ee + μμ combined because the effect should not depend on the lepton flavour,
applied to DY in every region (Fig. 24).  Its uncertainty is the statistical
and systematic uncertainty of the ratio, uncorrelated across bins (8.5–11.6 % on
DY, Table 21).  Two validations: the ratio derived in DY CR1 (60 < m(ll) < 100)
improves data/MC in DY CR2 (100–150), Figs. 27–30; with an NLO DY sample as
pseudo-data, NLO/LO agrees with Data/LO in the CRs (Fig. 31) and the CR and SR
NLO/LO agree within 2σ (Fig. 32).  The DY rate parameters then fit to 1.00 ±
0.07–0.16 (Table 19).

Exact 2018 values (`DYReshapeJetPt_2018.root`, errors stat + syst):
resolved 1.25 ± 0.11, 1.13, 1.03, 0.95, 0.86, 0.85, 0.69 ± 0.11 in the leading-AK4
bins of the previous section; boosted 0.98, 0.92, 0.98, 0.63, 0.98 in its AK8
bins.  Ours uses the same ratio, binning, variables and ee + μμ combination.
Run 2's analyzer variation shifts the ratio by its full bin error (statistics
and DY MC systematics, ~±0.1 per bin); ours provides the statistical and the
ee-vs-μμ variations (`--systs dyreshape`), so it is narrower.
