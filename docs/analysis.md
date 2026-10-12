# The analysis as implemented

What this page answers: what the analyzer does to an event, cut by cut and weight by weight, which
regions it builds, how data is blinded, which systematic variations exist and what is still open.
The code is the reference (`wrcoffea/analyzer.py`, `regions.py`, `systematics.py`,
`scale_factors.py`, cut values in `wrcoffea/config.yaml: cuts`); the payload of every correction
per era is in [corrections.md](corrections.md). The selection follows the Run 2 analysis
CMS-EXO-20-002 (AN-19-083); the differences are listed in
[studies/run2_hnwr_comparison.md](studies/run2_hnwr_comparison.md). Verified on 2026-10-12.

## Signal and topologies

W_R → ℓN → ℓℓqq' with ℓ = e or μ, in two topologies:

| Topology | Final state | Observable | Fit bins (GeV) |
| --- | --- | --- | --- |
| resolved | two tight leptons, two AK4 jets | m(ℓℓjj) (`mass_fourobject`) | 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000 |
| boosted | one tight lepton, one AK8 jet containing the second lepton | m(ℓJ) (`mass_twoobject`) | 800, 1000, 1200, 1500, 1800, 8000 |

Eras: Run 2 UL `RunIISummer20UL16APV`, `UL16`, `UL17`, `UL18` and Run 3 `Run3Summer22`,
`Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`; `RunIII2025Summer25` is
data-only and uses the 2024 MC. Signal samples: the official grids (UL: the electron and muon
samples of a point processed together; Run 3 2022/2023: 100 points per era) and for 2024 a private
Summer24 production (137 points, [inputs.md](inputs.md)).

## Event filters

Applied to data and MC unless stated.

- **Golden JSON** (data): the certified lumisections of the era ([luminosity.md](luminosity.md)).
- **Noise filters** (`_COMMON_NOISE_FLAGS`, `_OPTIONAL_NOISE_FLAGS` and `apply_noise_filter` in
  `analyzer.py`): goodVertices, globalSuperTightHalo2016Filter, EcalDeadCellTriggerPrimitiveFilter,
  BadPFMuonFilter, eeBadScFilter, plus BadPFMuonDzFilter and hfNoisyHitsFilter where the branch
  exists; Run 2 UL adds HBHENoiseFilter and HBHENoiseIsoFilter in every year and ecalBadCalibFilter
  for 2017 and 2018; 2024 adds ecalBadCalibFilter. 2025 has no era-specific list yet: it gets the
  common flags only (the log says "Noise filters are not applied", but the common ones are). For 2022
  and 2023 the NanoAOD ecalBadCalibFilter is not used; instead the JME recipe for runs 362433 to
  367144 rejects an event with PuppiMET > 100 GeV and a jet (pT > 50 GeV, −0.5 < η < −0.1,
  −2.1 < φ < −1.8, neutral or charged EM fraction > 0.9) back to back with it, Δφ > 2.9
  (`ecal_bad_calib_filter`; on PuppiMET since 2026-10-11, PF MET before, a difference of a few
  events per million). A required flag the input lacks, or a missing or unreadable
  golden JSON, stops the run (since 2026-10-11); the two optional flags are applied when present.
- **Jet veto maps** (JME): a jet with pT > 15 GeV, EM fraction < 0.9 and the tight-lepton-veto ID in a
  vetoed cell. Run 3 rejects the **event** (the JME recommendation; 13 to 15 % of 2024 events). The
  UL eras drop only those **jets** and keep the event (`jetveto_mode: jet`), as the Run 2 analysis
  did; jets with a PF muon within ΔR < 0.2 are spared. The event rule had cost UL18 17 to 20 % of
  its events against Run 2 (2026-10-10).
- **HEM 15/16 veto** (UL18 only, data and MC, whole year, as in Run 2): the event is dropped when one of
  the two leptons the region is built from is an electron with η < −1.25 and −1.62 < φ < −0.82.

## Objects

Jets are corrected with the era's JEC (and JER smearing on MC) before anything else, then ordered
by their corrected pT (j1, j2 are the two leading jets of that order); electrons and muons get their
scale and smearing corrections ([corrections.md](corrections.md)).

| Object | Definition |
| --- | --- |
| tight electron | pT > 53 GeV, \|η\| < 2.5, ECAL crack 1.4442 to 1.566 vetoed on the supercluster η, `cutBased_HEEP` |
| tight muon | TuneP pT > 53 GeV, \|η\| < 2.4, `highPtId == 2` (global high-pT ID), `tkRelIso < 0.1` |
| loose electron (boosted partner) | the cut-based Loose working point from the VID bitmap with the isolation bits ignored (a HEEP electron that fails it is not a loose electron; 0.5 to 1 % of HEEP electrons in 2024 signal and tt̄); the extra-tight-lepton veto counts the HEEP electrons among these |
| loose muon (boosted partner) | `highPtId == 2`, no isolation requirement |
| AK4 jet | pT > 40 GeV, \|η\| < 2.4, tight ID with lepton veto (2024: the correctionlib `jetid.json`; other eras `jetId >= 6`); CHS in UL, Puppi in Run 3 |
| AK8 jet | pT > 200 GeV, \|η\| < 2.4, tight ID, soft-drop mass > 40 GeV; the SR and flavour-CR jet also needs `lsf3 > 0.75` |

Notes on the AK8 jet. The soft-drop mass cut uses NanoAOD's JEC-corrected `msoftdrop` by default.
An optional per-era key `ak8_msoftdrop_mode` (`corrected` or `raw`, read by
`scale_factors.ak8_msoftdrop_mode`; an era without an entry is `corrected`) switches to the mass of
the uncorrected subjets, which is what Run 2 cut on. `config.yaml` sets it for no era today; a study
can set it through `WRCOFFEA_CONFIG_OVERLAY`. The choice accounts for about 9 % of the UL18 boosted
DY CR yield (decision pending). NanoAOD's `lsf3` is −1 for jets without a lepton (Run 2 stored 0)
and counts non-PF leptons; `FatJet_muonIdx3SJ` / `electronIdx3SJ` are stale in UL NanoAOD and are
never used.

## Triggers

Each region uses one trigger OR; the data stream follows the trigger.

| Era | Electron | Muon |
| --- | --- | --- |
| UL16APV, UL16 | Ele27_WPTight_Gsf, Photon175, Ele115_CaloIdVT_GsfTrkIdT | Mu50, TkMu50 |
| UL17 | Ele35_WPTight_Gsf, Photon200 (Ele115 left out: off for the first ~5 fb⁻¹, as in Run 2) | Mu50, OldMu100, TkMu100 |
| UL18 | Ele32_WPTight_Gsf, Photon200, Ele115_CaloIdVT_GsfTrkIdT | Mu50, OldMu100, TkMu100 |
| Run 3 | Ele30_WPTight_Gsf, Photon200, Ele115_CaloIdVT_GsfTrkIdT, Ele50_CaloIdVT_GsfTrkIdT_PFJet165 | Mu50, HighPtTkMu100, CascadeMu100 |

The resolved eμ region uses the muon trigger; the boosted flavour CRs use the trigger of the
leading lepton. Run 2 data are the SingleElectron and SingleMuon primary datasets (the Photon
paths count on the SingleElectron stream, as in Run 2), except UL18, whose primary datasets are
EGamma and SingleMuon; Run 3 has EGamma and Muon (plus SingleMuon in 2022B and C). The analyzer
calls the two streams `EGamma` and `Muon` in every era.
Both data files fill every region with no overlap removal, so each region reads exactly one
stream: EGamma for the ee regions and `wr_emu_boosted_flavor_cr`, Muon for the μμ regions,
`wr_resolved_flavor_cr` and `wr_mue_boosted_flavor_cr`.

## Regions

Cut values: `lead_lepton_pt_min` 60, `sublead_lepton_pt_min` 53, `mll_dy_low/high` 60/150,
`mll_sr_high_min` 400 (resolved), `mll_sr_min` 200 (boosted), `mlljj_min` 800, `dr_min` 0.4,
`dphi_boosted_min` 2.0, `dr_ak8_loose` 0.8 (`config.yaml: cuts`).

**Resolved** (base: exactly two tight leptons of the flavour, leading pT > 60 GeV, sub-leading
pT > 53 GeV, the flavour's trigger, at least two AK4 jets, ΔR > 0.4 among all pairs of
{ℓ1, ℓ2, j1, j2}, HEM veto in UL18; m(ℓℓjj) from the two leading leptons and the two leading jets):

| Region | Flavour | m(ℓℓ) | m(ℓℓjj) | Stream |
| --- | --- | --- | --- | --- |
| `wr_ee_resolved_dy_cr`, `wr_mumu_resolved_dy_cr` | ee, μμ | 60 to 150 GeV | > 800 GeV | EGamma, Muon |
| `wr_ee_resolved_sr`, `wr_mumu_resolved_sr` | ee, μμ | > 400 GeV | > 800 GeV | EGamma, Muon |
| `wr_resolved_flavor_cr` | eμ (muon trigger) | > 400 GeV | > 800 GeV | Muon |

**Boosted** (events that fail the resolved-like tag; leading tight lepton pT > 60 GeV; any extra
tight lepton vetoed):

| Region | Jet | Second lepton | Masses | Stream |
| --- | --- | --- | --- | --- |
| `wr_ee_boosted_dy_cr`, `wr_mumu_boosted_dy_cr` | leading AK8 without the LSF cut, \|Δφ(ℓ, J)\| > 2 | a same-flavour loose lepton with 60 < m(ℓ_T, ℓ_L) < 150 GeV sends the event here, never to the SR | m(ℓJ) > 800 GeV; the loose lepton is added to the mass when it is outside the jet (ΔR ≥ 0.8) | EGamma, Muon |
| `wr_ee_boosted_sr`, `wr_mumu_boosted_sr` | leading AK8 with `lsf3 > 0.75` and \|Δφ\| > 2 | a same-flavour loose lepton inside the jet (ΔR < 0.8) and no opposite-flavour one | m(ℓ_T, ℓ_L) > 200 GeV, m(ℓ_T, J) > 800 GeV | EGamma, Muon |
| `wr_emu_boosted_flavor_cr`, `wr_mue_boosted_flavor_cr` | as the SR | an opposite-flavour loose lepton in the jet, no same-flavour one; named leading-then-jet (`emu` = leading electron, muon in the jet, electron trigger) | as the SR | EGamma, Muon |

Region variants and studies (`--study`, registry in `regions.py`): `transfer-factor` adds the SR
and flavour CRs without the mass cut (`<region>_nomass`, for the eμ transfer factor R = SR / flavour
CR measured on tt̄, AN-19-083 Appendix J); `dy-cr-split` splits the DY CRs at m(ℓℓ) = 100 GeV
(AN-19-083 DY CR1 and CR2, for the reshape closure). Run 2 regions not built here: resolved DYCR3
(200 < m(ℓℓ) < 400 GeV), the low-W_R CR (m < 800 GeV) and the b-jet splits.

## Event weights

MC base weight: `genWeight × xsec × lumi × 1000 / genEventSumw`, times the pileup weight, the DY
Z-pT weight, the signal k(m(ℓN))/⟨k⟩ weight and, in the UL eras, the L1 prefiring weight. Per
region it is multiplied by the lepton scale factors that enter that region and by the DY reshape:

- ee regions: electron reco, HEEP ID and trigger SFs of the tight electrons; μμ regions: muon reco,
  ID, isolation and trigger SFs of the tight muons; eμ regions: both, and the trigger SF of the
  trigger that selects the region;
- boosted regions: loose-muon reco and ID (no isolation) or loose-electron reco SFs of the partner
  (the loose-electron ID efficiency is absorbed in the LSF SF, as in Run 2);
- boosted SR and flavour CRs: the LSF SF by the flavour of the lepton in the jet;
- DY (the LO HT-binned sample only): the jet-pT reshape.

Trigger SFs: electrons, one SF for the OR (Run 3: the EGM Ele30 TightID SF as a proxy; UL: the
Run 2 HEEP trigger-OR efficiency); muons, the per-leg data/MC efficiency OR for 2022 to 2025 and
the leading muon's SF alone in the UL eras (two SFs are never multiplied). The muon variation of
the efficiency OR follows the Run 2 analysis (SKFlat `MCCorrection::MuonTrigger_SF`): every muon's
data efficiency moves up by its error while its MC efficiency moves down, and the reverse, so the
two errors add in the ratio (decided 2026-10-12; shifting both the same way, as before, cancelled
them to about 0.01 %). Cutflows use the base weight without the per-region lepton SFs.

### Drell-Yan modelling

DY is the LO madgraphMLM HT-binned sample (`lo_ht`) in every era, with two corrections derived
exactly as in AN-19-083 and applied only to that sample (`dy_reweight_datasets`; other DY variants
are cross-checks and get the identity):

- **Z-pT reweight** (`wrcoffea/dy_reweight.py`, payload per era): the NLO/LO ratio versus the
  generator Z pT (dilepton system, m ≥ 50 GeV) of the inclusive NLO FxFX over the inclusive LO MLM
  sample, both at the NNLO cross section, so the weight changes the shape only (integral 0.996 to
  0.999); in the 28 bins of the Run 2 derivation; scale, PDF and α_S variations of the ratio
  (unrenormalized); times the electroweak correction of Lindert et al. (arXiv:1705.04664) with its
  three nuisances. Derived by `scripts/derive_dy_zpt_reweight.py` on the full samples; one payload
  per UL era, one for 2022+2023 and one for 2024 (the Summer24 NLO sample is about 5 % softer).
- **Jet-pT reshape** (`wrcoffea/dy_reshape.py`): (data − non-DY MC) / DY MC versus the leading jet pT
  in the DY control regions, ee and μμ summed, in the Run 2 bins (resolved 0, 40, 100, 200, 400,
  600, 800, 1000, 2000 GeV on the AK4 jet; boosted 0, 200, 400, 600, 800, 1000, 2000 GeV on the AK8
  jet), applied to DY in every region; nuisances: the statistical uncertainty and the ee-vs-μμ
  difference. Derived by `scripts/derive_dy_reshape.py` from a production without it; one payload
  per UL era and per Run 3 data-taking year.

Both are switched off with `--no-dy-zpt-reweight` (implies `--no-dy-reshape`) and `--no-dy-reshape`.
The HT-binned sets have no events below their first bin, LHE H_T = 40 GeV in Run 3 and 70 GeV in
UL (`default_mc_tag` in `config.yaml`). In 2024 this costs 9 to 16 % of the boosted-SR DY, about 1 %
of the resolved-SR DY and nothing in the control regions; the choice (2026-10-08) is to live with it
rather than stitch.

### Signal normalization

The config cross section of every point is LO × the point's own direct NLO/LO k-factor, and the
analyzer applies the k(m(ℓN))/⟨k⟩ shape reweight, both from the derivation in
[studies/signal_kfactor.md](studies/signal_kfactor.md) (`wrcoffea/signal_kfactor.py`,
`--no-signal-kfactor` switches the shape part off).

## Blinding

Data of a blinded run (the default) never builds the signal regions or their variants, and its ee
and μμ cutflows stop before the m(ℓℓ) > 200 GeV cut; control regions and the eμ cutflows are
complete; MC is never blinded. The manifest records what a file withholds. `--unblind` is allowed
for the Run 2 eras only. Operational details: [run_analysis.md](run_analysis.md#blinding).

## Systematics

What is finished and what remains, per Run: [systematics_status.md](systematics_status.md).

Every source is registered in `wrcoffea/systematics.py` with its family, its kind (weight, shape,
normalization or members), the regions it can affect and the eras that have its inputs; a family
an era cannot produce is an error, never a silent copy, and the coverage contract refuses partial
templates. Variations are written as `syst_<label lower>_<region>` folders.

| Family | Sources | Kind | What it varies |
| --- | --- | --- | --- |
| `lumi` | Lumi | normalization | the era's luminosity uncertainty, applied at export |
| `pileup` | Pileup | weight | the pileup profile reweighting |
| `prefire` | L1Prefiring | weight | the NanoAOD `L1PreFiringWeight` (UL only) |
| `sf` | MuonRecoSf, MuonIdSf, MuonIsoSf, MuonTrigSf, LooseMuonRecoSf, LooseMuonIdSf, ElectronRecoSf, ElectronIdSf, ElectronTrigSf, LooseElectronRecoSf | weight | each lepton scale-factor component, only in the regions it enters |
| `lsf` | LsfSf | weight | the LSF scale factor (boosted SR and flavour CRs) |
| `jer`, `jes` | JER, JES (the `Total` source) | shape | jet energy resolution and scale; the selection and the masses are recomputed |
| `elescale`, `elesmear` | EleScale, EleSmear | shape | electron energy scale and resolution (Run 3) |
| `muscale`, `musmear` | MuScale, MuSmear | shape | muon momentum scale and resolution (2024: the high-pT recipe; other eras a placeholder, see open items) |
| `dyzpt` | DYZptQcdScale, DYZptQcdPdf, DYZptQcdAlphaS, DYZptEw1, DYZptEw2, DYZptEw3 | weight | the Z-pT reweight's QCD and EW uncertainties (DY only) |
| `dyreshape` | DYReshapeStat, DYReshapeChannel | weight | the reshape's statistical and ee-vs-μμ uncertainties (DY only) |
| `scale` | ScaleMuR, ScaleMuF, ScaleMuRF | weight | QCD renormalization and factorization scales from `LHEScaleWeight`, each normalized to the dataset's mean so only acceptance and shape move; the plotter takes the envelope |
| `ps` | PSISR, PSFSR | weight | parton-shower ISR and FSR from `PSWeight`, normalized the same way |
| `pdf` | PDF, AlphaS | members | every `LHEPdfWeight` member stored per PDF set (`pdfmembers_<region>/<var>_lha<id>`); the Hessian quadrature and the α_S shift (scaled to 0.0015) are formed at the final binning by the plotter and the datacards |

What each era can produce (2026-10-12; from the registry and the payloads of
[corrections.md](corrections.md)):

| Family | UL16APV, UL16, UL17, UL18 | 2022, 2022EE, 2023, 2023BPix | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| `lumi`, `pileup`, `sf`, `jer`, `jes`, `dyzpt`, `lsf`, `scale`, `pdf` | yes | yes | yes | yes |
| `muscale`, `musmear` | placeholder (TuneP, flat 5 %/TeV scale, smearing = nominal) | placeholder | yes (the high-pT recipe) | placeholder |
| `prefire` | yes | no (no prefiring in Run 3) | no | no |
| `elescale`, `elesmear` | no (no scale-and-smearing payload; UL NanoAODv9 MC has no scale uncertainty branch) | yes | yes | yes |
| `dyreshape` | yes | yes | yes | not yet (no 2025 reshape) |
| `ps` | yes: the UL filesets carry PSSumw estimated from a few million unselected events of every original sample (`scripts/estimate_ps_sums.py`, 2026-10-11; exact sums come with the next skim, which derives them from the events) | yes | yes | yes |

Samples without generator weights (the Pythia 8 WW, WZ, ZZ) get their nominal for `scale` and
contribute no `pdf` members, with a warning. A dataset whose fileset records why it has no
dataset-level means (`PSSumw_source.status`, e.g. original files unreachable for the UL estimate)
gets its nominal for that family, also with a warning. Missing means without
such a record are an error. The plotter's band and the systematics tables take every source present
in the files ([WR_Plotter/docs/plotting.md](../WR_Plotter/docs/plotting.md)).

## Open items

Decisions and measurements still outstanding, with the date they were last looked at.

- **Muon momentum at high pT.** The Muon POG Run 3 recipe (MuonScaRe below 200 GeV, the
  generalized-endpoint correction on data and extra smearing on MC above) is applied to 2024 only;
  UL, 2022 and 2023 keep the TuneP momentum with a flat 5 %/TeV scale uncertainty and no smearing
  (2026-10-10).
- **Muon reco SF above 1.5 TeV.** The POG high-pT reco SF has an open-ended last bin (p > 1.5 TeV,
  2024 p > 0.9 TeV) whose barrel values range from 0.68 (2023BPix) to 1.02 (2022) with large errors,
  and high-mass signal muons land there. Options: clip to the last measured bin, clip with the
  spread as a systematic, or a common high-p value; not decided (2026-10-10).
- **Electron SFs above the measured range.** EGM reco SFs are clipped at 500 GeV and HEEP ID SFs at
  1 TeV (the payloads hold a 1 ± 20 % placeholder above); the 2024 HEEP bins between 700 GeV and
  1 TeV (1.08 to 1.12) are statistics-limited (2026-10-04).
- **Electron trigger SF.** Run 3 uses the EGM Ele30 TightID SF as a proxy for the trigger OR; no
  central SF for the OR exists, a tag-and-probe measurement is needed (2025 uses the 2024 file).
- **LSF SF.** Run 2 values per year from AN-19-083 Table 60; Run 3 carries a 1.00 ± 0.10 placeholder
  until a measurement exists (2026-10-11).
- **UL electron scale and smearing variations.** UL NanoAODv9 MC has `Electron_dEscaleUp/Down`
  identically zero (the MC scale systematics were not included in the UL productions, per the EGM
  twiki); options are the data scale uncertainty mirrored onto MC or EgammaPostRecoTools at MiniAOD
  level (2026-10-11).
- **UL parton-shower means.** The UL filesets carry PSSumw estimated from a few million unselected
  events per sample (statistical precision well below 0.1 %); the next UL skim replaces them by the
  exact sums. A fileset regeneration drops the estimate until `scripts/estimate_ps_sums.py` is rerun
  (2026-10-11).
- **Soft-drop mass definition.** Corrected (default) versus the raw subjet mass Run 2 used, about
  9 % of the UL18 boosted DY CR (2026-10-10).
- **Official signal samples have no N → ℓtb.** The official gridpacks' `jj` excludes the top and the N
  width counts only the generated channels, so for m_N above a few hundred GeV the samples put the
  full rate into ℓℓ + light jets while about a third should go to ℓtb; parked (2026-10-10).
- **Luminosity correlations.** The correlated 2022 to 2024 components belong in the datacards
  (2026-10-10).
- **2025.** Data-only era, configured but not skimmed; electron trigger SF and muon momentum are
  placeholders (2026-10-11).
