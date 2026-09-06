# WrCoffea versus the Run 2 HNWRAnalyzer

Comparison of this analyzer with the Run 2 WR analysis code in
[jedori0228/SKFlatAnalyzer](https://github.com/jedori0228/SKFlatAnalyzer),
branch `HNWR/Run2Legacy_v4` (commit `06f274c`, "final", 2022-05-13):
`Analyzers/src/HNWRAnalyzer.C` plus the object IDs in `DataFormats/src`.
That branch produced the `HNWRAnalyzer_SkimTree_LRSMHighPt_*` files under
`WR_Plotter/signal_fitting/run2_results/` (the `ReweightedQCDErrorEWCorr` and
`Reshaped` suffixes are the two DY corrections described below).

The 2018 column is what matters for the UL18 closure; Run 3 differences that
are simply era updates (triggers, POG payloads) are noted but not treated as
discrepancies.

## Objects

| Object | Run 2 HNWRAnalyzer | WrCoffea | Status |
|---|---|---|---|
| Electron kinematics | pT > 53, \|SC η\| < 2.4 (`SelectElectrons("HNWRLT", 53, 2.4)`) | pT > 53, \|η\| < 2.5, ECAL crack veto on SC η | **Differs**: η range 2.4 vs 2.5 (known, see memory note). Run 2 cuts on SC η. |
| Electron tight | HEEP v7 (`HNWRTight2018Prompt` for 2018) | `cutBased_HEEP` | Same |
| Electron loose | Custom `HNWRLoose` cut-based, no isolation (H/E, σiηiη, ΔηSeed, Δφin, 1/E−1/p, ≤1 missing hit, conversion veto) | Boosted: POG loose from `vidNestedWPBitmap` with the isolation bit ignored, OR HEEP. Resolved: `cutBased == 2` (unused by any region) | Equivalent intent (loose, non-isolated) with a POG working point instead of the hand-tuned one |
| Muon kinematics | TuneP pT > 53, \|η\| < 2.4 | `Muon.pt` (PF) > 53 with the momentum scale/smearing applied, \|η\| < 2.4 | **Differs**: Run 2 used TuneP momentum for the selection and kinematics; SFs were evaluated at the MiniAOD (PF) pT |
| Muon tight | POG HighPt ID and TrkIso/TuneP pT < 0.1 | `highPtId == 2` and `tkRelIso < 0.1` | Same |
| Muon loose | POG HighPt ID | `highPtId == 2` | Same |
| AK4 jets | TightLepVeto ID, pT > 40, \|η\| < 2.4 | TightLepVeto, pT > 40, \|η\| < 2.4, JEC/JER applied | Same |
| AK8 jets | Tight ID, SD mass > 40, pT > 200, \|η\| < 2.4; SR additionally LSF > 0.75 | Same thresholds; SR uses `lsf3 > 0.75` | Same |
| b-tagging | DeepCSV medium, used to split SR and e-μ CRs into BJet / NoBJet | None | **Missing** (only needed for the b-jet sub-regions, which are not part of the fit) |

## Event selection

| Step | Run 2 HNWRAnalyzer | WrCoffea | Status |
|---|---|---|---|
| MET filters | goodVertices, globalSuperTightHalo2016, HBHENoise, HBHENoiseIso, EcalDeadCellTP, BadPFMuon, eeBadSc (data only), ecalBadCalibReduced (2017+) | Same set for UL18; eeBadSc also on MC; BadPFMuonDz and hfNoisyHits when present; Run 3 uses the era-specific ECAL filter | Same, negligible differences |
| HEM 15/16 | 2018: drop events with a selected electron at η < −1.25, −1.62 < φ < −0.82 (data and MC, whole year, `isHEMVetoEvent`) | **Now ported**: `hem_veto` in `config.yaml` (UL18 only), `SEL_HEM_VETO` in the resolved regions and folded into the boosted region masks; evaluated on the two leptons each region is built from, on electron η like Run 2 | Same for UL18 |
| Triggers (2018) | e: Ele32_WPTight, Photon200, Ele115_CaloIdVT_GsfTrkIdT; μ: Mu50, OldMu100, TkMu100 | Same for `RunII`; Run 3 uses Ele30/Photon200/Ele115/Ele50_PFJet165 and Mu50/HighPtTkMu100/CascadeMu100 | Same |
| e-μ trigger | Muon triggers | Muon triggers | Same |
| Resolved baseline | Exactly 2 tight leptons, leading pT > 60, ≥2 AK4 jets, ΔR > 0.4 among {l1, l2, j1, j2} | Same, plus the explicit sub-leading pT > 53 (implied in Run 2 by the object cut) | Same |
| Resolved SR | m(ll) > 400 and m(lljj) > 800 | Same | Same |
| Resolved DY CR | 60 ≤ m(ll) < 150 and m(lljj) > 800 | Same | Same |
| Resolved extras | LowWRCR (m(ll) > 200, m(lljj) < 800), DYCR1/2 (60–100 / 100–150), DYCR3 (200 < m(ll) < 400), BJet/NoBJet | `_nomass` study variants only | Missing sub-regions (cosmetic unless the m(ll) 200–400 CR is wanted) |
| Boosted tag | Not resolved | `~resolved` on the same objects | Same |
| Boosted DY CR | Tight lead (pT > 60), same-flavour loose partner with 60 ≤ m(ll) < 150 (first match), AK8 without LSF with \|Δφ(l, J)\| > 2, N = J or J + l₂ depending on ΔR < 0.8, m(WR) > 800 | Same | Same |
| Boosted SR | No DY pair, LSF AK8 with \|Δφ\| > 2, same-flavour loose lepton (pT > 53) within ΔR < 0.8, no opposite-flavour loose lepton in the jet, no extra tight lepton, m(ll) > 200, m(lJ) > 800 | Same | Same |
| Boosted e-μ CR | As SR but opposite-flavour lepton in the jet | Same (`flavor_cr`) | Same |
| Extra tight lepton veto | Any tight lepton other than the two used | Loose leptons passing the tight criteria, ΔR-matched away from the two used | Same |

## Weights and corrections

| Correction | Run 2 HNWRAnalyzer | WrCoffea | Status |
|---|---|---|---|
| Normalization | xsec × lumi × genWeight / ΣgenWeight | Same | Same |
| Pileup | Custom histogram in nPileUp | correctionlib on `Pileup.nTrueInt` | Same |
| L1 prefiring | Applied for 2016/2017 only; `GetPrefireWeight` returns 1 for 2018 | None | Same for UL18 (only matters if 2016/2017 are ever added) |
| Electron SFs | RECO and HEEP ID on tight electrons; HEEP trigger-OR efficiencies (`WREGammaTriggerEff.root`) combined per event with the OR formula | Same components; UL18 uses the ported Run 2 HEEP efficiencies (`scripts/convert_run2_electron_trigger_sf.py`), Run 3 the EGM Ele30 proxy | Same for UL18 |
| Muon SFs | RECO (HighPt, vs p), ID, ISO on tight muons; Mu50 trigger SF | Same components | Same |
| Boosted loose lepton | No RECO/ID SF on the loose partner (it only enters the muon trigger SF) | RECO+ID on loose muons, RECO on loose electrons | Small difference; ours is the more complete treatment |
| LSF scale factor | Flat per year and jet flavour (2018: e-jet 1.05 +0.07/−0.06, μ-jet 1.04 +0.06/−0.05; AN-19-083 Table 60, injected-lepton method) on boosted SR and e-μ CR | **Now ported**: `lsf_sf` in `config.yaml` (UL18), applied in `_lepton_sfs_for_region` to the boosted SR / flavour-CR regions by lepton-in-jet flavour; `--systs lsf` for the variation | Same for UL18; Run 3 needs its own measurement |
| DY Z-pT reweight | QCD NLO/LO ratio × EW correction vs generator Z pT, DY only, with QCD scale/PDF/αS and three EW variations | **Now ported** (`wrcoffea/dy_reweight.py`, see below) | Same for UL18 |
| DY reshape | Data/MC ratio from the DY CR applied per Resolved/Boosted with a stat and an ee-vs-μμ variation. The final production (`script/Submit/submit_2018.sh`, flags `ApplyDYReshape,JetPt`) used the **leading-jet pT** variant (`DYReshapeJetPt_2018.root`, 40–2000 GeV, ratio 1.25 at 40–100 GeV falling to 0.69 above 1 TeV), not the m(lljj) one | **Machinery ported**: `wrcoffea/dy_reshape.py` applies a payload from `scripts/derive_dy_reshape.py` (same binning and ee+μμ combination as Run 2, applied to every DY event in every region; stat and ee-vs-μμ variations via `--systs dyreshape`). UL18 payload `dy_reshape_run2jetpt_20260907.json` derived from a local production (data + bkg, DY lo_ht, Z-pT reweight on). Its level (~1.5 → 0.85 vs jet pT) absorbs the NNLO/LO normalization Run 2 carried in its DY cross sections (~1.35× ours); normalized to its mean, the shape matches the Run 2 ratio to within ~10% per bin | Same for UL18 |
| Signal k-factor | k(m_lN)/⟨k⟩ per mass point (`kfactor.root`, `AveragedKFactor.txt`; AN-19-083 App. A). Shape only: the Run 2 sample tables carry xsec 1 and the normalization LO × ⟨k⟩ enters at the plotting stage | **Now ported**: `wrcoffea/signal_kfactor.py`, payload from `scripts/convert_run2_signal_kfactor.py`, `signal_kfactor` in `config.yaml` (UL18), `--no-signal-kfactor`. Our signal configs already hold the AN's LO × ⟨k⟩, so the same shape-only weight applies. Gen finder mirrors `GenFinderForHNWRSignal` on `GenPart` | Same for UL18 |
| Muon momentum | TuneP + ScaleMuons/SmearMuons systematics | Scale/smear placeholder (flat 5 %/TeV) | Placeholder, already flagged |

## Systematic sources

Run 2: JetRes, JetEn, JetMass (SD mass), MuonReco/En/ID/ISO/Trigger,
ElectronReco/Res/En/ID/Trigger, LSFSF, PU, ZPtRw (+QCD scale/PDF/αS, EW1–3),
Prefire, DYReshape, BTag.

WrCoffea: lumi, pileup, sf (all lepton components), jer, jes, elescale,
elesmear, muscale, musmear, and now dyzpt. Not covered: AK8 soft-drop mass
scale, LSF SF, DY reshape, b-tagging (no b-tag regions). Prefiring is not
needed for 2018.

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
  70–90 GeV and falls to 0.28 above 2 TeV.
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

The payload was derived for the 13 TeV 2018 LO madgraphMLM HT-binned
`DYJetsToLL_M-50` samples, which is our UL18 `dy_lo_ht` fileset.  It is not
mapped to the 13.6 TeV Run 3 eras on purpose: the NLO/LO ratio depends on the
generator setup and centre-of-mass energy, and the Run 3 `DYto2L-4Jets` MLM
sample needs its own derivation (`Analyzers/src/GetZPtReweight.C` in the Run 2
repository shows how the numerator and denominator spectra were produced).
