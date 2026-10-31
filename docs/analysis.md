# The analysis as implemented

The code is the reference: `wrcoffea/analyzer.py`, `regions.py`, `systematics.py`,
`scale_factors.py`, with the cut values in `wrcoffea/config.yaml: cuts`. The payload of every
correction per era is in [corrections.md](corrections.md). The selection follows the Run 2
analysis CMS-EXO-20-002 (AN-19-083).

## Signal and topologies

W_R → ℓN → ℓℓqq' with ℓ = e or μ, in two topologies:

| Topology | Final state | Observable | Fit bins (GeV) |
| --- | --- | --- | --- |
| resolved | two tight leptons, two AK4 jets | m(ℓℓjj) (`mass_fourobject`) | 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000 |
| boosted | one tight lepton, one AK8 jet containing the second lepton | m(ℓJ) (`mass_twoobject`) | 800, 1000, 1200, 1500, 1800, 8000 |

Eras: Run 2 UL `RunIISummer20UL16APV`, `UL16`, `UL17`, `UL18`; Run 3 `Run3Summer22`,
`Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24`; `RunIII2025Summer25`
is data-only on the 2024 MC. Signal: the official grids (UL: the electron and muon samples of a
point processed together; 2022 and 2023: 100 points per era) and for 2024 a private Summer24
production of 137 points ([inputs.md](inputs.md)).

## Event filters

Data and MC unless stated.

- Golden JSON (data): the certified lumisections of the era ([luminosity.md](luminosity.md)).
- Noise filters (`_COMMON_NOISE_FLAGS`, `_OPTIONAL_NOISE_FLAGS`, `apply_noise_filter` in
  `analyzer.py`): goodVertices, globalSuperTightHalo2016Filter, EcalDeadCellTriggerPrimitiveFilter,
  BadPFMuonFilter, eeBadScFilter; BadPFMuonDzFilter and hfNoisyHitsFilter where the branch exists.
  UL adds HBHENoiseFilter and HBHENoiseIsoFilter, and ecalBadCalibFilter in 2017 and 2018. 2024 adds
  ecalBadCalibFilter. 2025 uses the common flags only (JME lists ecalBadCalibFilter as under review
  for 2025 and 2026). 2022 and 2023 use the JME veto for runs 362433 to 367144 instead of the
  NanoAOD flag: an event with PuppiMET > 100 GeV and a jet with pT > 50 GeV, −0.5 < η < −0.1,
  −2.1 < φ < −1.8, neutral or charged EM fraction > 0.9, back to back with it (Δφ > 2.9) is
  rejected (`ecal_bad_calib_filter`). A required flag the input lacks, or a missing or unreadable
  golden JSON, stops the run.
- Jet veto maps (JME): a jet with pT > 15 GeV, EM fraction < 0.9 and the tight-lepton-veto ID in a
  vetoed cell. Run 3 rejects the event (13 to 15 % of 2024 events). UL drops those jets and keeps
  the event (`jetveto_mode: jet`), as Run 2 did; jets with a PF muon within ΔR < 0.2 are spared.
- HEM 15/16 veto (UL18, data and MC, whole year, as in Run 2): the event is dropped when one of the
  two leptons the region is built from is an electron with η < −1.25 and −1.62 < φ < −0.82.

## Objects

Jets get the era's JEC, and JER smearing on MC, before anything else, and are ordered by corrected
pT (j1, j2 are the two leading jets of that order). Electrons and muons get their scale and
smearing corrections ([corrections.md](corrections.md)).

| Object | Definition |
| --- | --- |
| tight electron | pT > 53 GeV, \|η\| < 2.5, ECAL crack 1.4442 to 1.566 vetoed on the supercluster η, `cutBased_HEEP` |
| tight muon | TuneP pT > 53 GeV, \|η\| < 2.4, `highPtId == 2`, `tkRelIso < 0.1` |
| loose electron (boosted partner) | the cut-based Loose working point from the VID bitmap with the isolation bits ignored. A HEEP electron that fails it is not loose (0.5 to 1 % of HEEP electrons in 2024 signal and tt̄); the extra-tight-lepton veto counts those |
| loose muon (boosted partner) | `highPtId == 2`, no isolation requirement |
| AK4 jet | pT > 40 GeV, \|η\| < 2.4, tight ID with lepton veto: 2024 and 2025 the correctionlib `jetid.json` (NanoAODv15 has no `jetId`); 2022 and 2023 the JME repair of the NanoAODv12 `jetId` (Tight bit, muEF < 0.8 and chEmEF < 0.8 for \|η\| ≤ 2.7); UL the lepton-veto bit. CHS in UL, Puppi in Run 3. The jet veto maps use the same ID |
| AK8 jet | pT > 200 GeV, \|η\| < 2.4, tight ID, soft-drop mass > 40 GeV; the SR and flavour-CR jet also needs `lsf3 > 0.75` |

AK8 soft-drop mass:

- Run 3 follows the JME jet mass recommendation (`_softdrop_from_subjets`): the two soft-drop
  subjets get the AK4 PUPPI JEC from their raw momenta and, on MC, the AK4 JER with gen subjets
  matched within ΔR < 0.2; m_SD is the mass of their sum. A jet without two subjets keeps NanoAOD's
  value. UL keeps NanoAOD's `msoftdrop` (the UL JERC payloads carry no AK4 PUPPI corrections).
- Checks: on 2023 DY the AK4 JEC alone reproduces NanoAOD's m_SD to 0.05 % and the JER adds a
  spread of 1.6 %; on 2024 signal m_SD is 1 % below NanoAOD's (newer JEC). The 40 GeV cut moves
  under 0.2 % of the jets.
- `ak8_msoftdrop_mode` per era (`corrected`, the default for every era, or `raw`: the mass of the
  uncorrected subjets, which Run 2 cut on; `scale_factors.ak8_msoftdrop_mode`). The two differ by
  about 9 % of the UL18 boosted DY CR yield. A study sets it through `WRCOFFEA_CONFIG_OVERLAY`.
- NanoAOD's `lsf3` is −1 for jets without a lepton (Run 2 stored 0) and counts non-PF leptons.
  `FatJet_muonIdx3SJ` / `electronIdx3SJ` are stale in UL NanoAOD and are not used.

## Triggers

Each region uses one trigger OR; the data stream follows the trigger.

| Era | Electron | Muon |
| --- | --- | --- |
| UL16APV, UL16 | Ele27_WPTight_Gsf, Photon175, Ele115_CaloIdVT_GsfTrkIdT | Mu50, TkMu50 |
| UL17 | Ele35_WPTight_Gsf, Photon200 (Ele115 left out: off for the first ~5 fb⁻¹, as in Run 2) | Mu50, OldMu100, TkMu100 |
| UL18 | Ele32_WPTight_Gsf, Photon200, Ele115_CaloIdVT_GsfTrkIdT | Mu50, OldMu100, TkMu100 |
| Run 3 | Ele30_WPTight_Gsf, Photon200, Ele115_CaloIdVT_GsfTrkIdT, Ele50_CaloIdVT_GsfTrkIdT_PFJet165 | Mu50, HighPtTkMu100, CascadeMu100, Mu50_L1SingleMuShower (exists from 2022E; absent in 2022 B-D and Run3Summer22 MC) |

- The resolved eμ region uses the muon trigger; the boosted flavour CRs use the trigger of the
  leading lepton.
- Run 2 data: the SingleElectron and SingleMuon primary datasets (the Photon paths count on the
  SingleElectron stream, as in Run 2); UL18 has EGamma and SingleMuon. Run 3: EGamma and Muon,
  plus SingleMuon in 2022B and C. The analyzer calls the two streams `EGamma` and `Muon` everywhere.
- Both data files fill every region with no overlap removal, so each region reads one stream:
  EGamma for the ee regions and `wr_emu_boosted_flavor_cr`; Muon for the μμ regions,
  `wr_resolved_flavor_cr` and `wr_mue_boosted_flavor_cr`.

## Regions

Cut values (`config.yaml: cuts`): `lead_lepton_pt_min` 60, `sublead_lepton_pt_min` 53,
`mll_dy_low/high` 60/150, `mll_sr_high_min` 400 (resolved), `mll_sr_min` 200 (boosted),
`mlljj_min` 800, `dr_min` 0.4, `dphi_boosted_min` 2.0, `dr_ak8_loose` 0.8.

Resolved base: exactly two tight leptons of the flavour, leading pT > 60 GeV, sub-leading
pT > 53 GeV, the flavour's trigger, at least two AK4 jets, ΔR > 0.4 among all pairs of
{ℓ1, ℓ2, j1, j2}, HEM veto in UL18; m(ℓℓjj) from the two leading leptons and the two leading jets.

| Region | Flavour | m(ℓℓ) | m(ℓℓjj) | Stream |
| --- | --- | --- | --- | --- |
| `wr_ee_resolved_dy_cr`, `wr_mumu_resolved_dy_cr` | ee, μμ | 60 to 150 GeV | > 800 GeV | EGamma, Muon |
| `wr_ee_resolved_sr`, `wr_mumu_resolved_sr` | ee, μμ | > 400 GeV | > 800 GeV | EGamma, Muon |
| `wr_resolved_flavor_cr` | eμ (muon trigger) | > 400 GeV | > 800 GeV | Muon |

Boosted base: events that fail the resolved-like tag, leading tight lepton pT > 60 GeV, any extra
tight lepton vetoed.

| Region | Jet | Second lepton | Masses | Stream |
| --- | --- | --- | --- | --- |
| `wr_ee_boosted_dy_cr`, `wr_mumu_boosted_dy_cr` | leading AK8 without the LSF cut, \|Δφ(ℓ, J)\| > 2 | a same-flavour loose lepton with 60 < m(ℓ_T, ℓ_L) < 150 GeV sends the event here, never to the SR | m(ℓJ) > 800 GeV; the loose lepton is added to the mass when it is outside the jet (ΔR ≥ 0.8) | EGamma, Muon |
| `wr_ee_boosted_sr`, `wr_mumu_boosted_sr` | leading AK8 with `lsf3 > 0.75` and \|Δφ\| > 2 | a same-flavour loose lepton inside the jet (ΔR < 0.8), no opposite-flavour one | m(ℓ_T, ℓ_L) > 200 GeV, m(ℓ_T, J) > 800 GeV | EGamma, Muon |
| `wr_emu_boosted_flavor_cr`, `wr_mue_boosted_flavor_cr` | as the SR | an opposite-flavour loose lepton in the jet, no same-flavour one; named leading-then-jet (`emu`: leading electron, muon in the jet, electron trigger) | as the SR | EGamma, Muon |

Studies (`--study`, registry in `regions.py`, combinable):

- `transfer-factor`: the SR and flavour CRs without the mass cut (`<region>_nomass`), for the eμ
  transfer factor R = SR / flavour CR on tt̄ (AN-19-083 Appendix J).
- `dy-cr-split`: the DY CRs split at m(ℓℓ) = 100 GeV (AN-19-083 DY CR1 and CR2).
- `no-lsf`: the boosted SR and flavour CRs on the leading AK8 jet with |Δφ| > 2 and no LSF
  requirement (`<region>_nolsf`, AN-19-083 Figs. 12 and 93-94); no LSF scale factor.

Run 2 regions not built: resolved DYCR3 (200 < m(ℓℓ) < 400 GeV), the low-W_R CR (m < 800 GeV) and
the b-jet splits.

## Event weights

MC base weight: `genWeight × xsec × lumi × 1000 / genEventSumw`, times the pileup weight, the
signal k(m(ℓN))/⟨k⟩ weight, the L1 prefiring weight in UL and, with `--dy-reweight`, the DY Z-pT
weight. Per region it is multiplied by:

- ee regions: electron reco, HEEP ID and trigger SFs of the tight electrons; μμ regions: muon reco,
  ID, isolation and trigger SFs of the tight muons; eμ regions: both, and the trigger SF of the
  trigger that selects the region;
- boosted regions: loose-muon reco and ID (no isolation) or loose-electron reco SFs of the partner
  (the loose-electron ID efficiency is in the LSF SF, as in Run 2);
- boosted SR and flavour CRs: the LSF SF by the flavour of the lepton in the jet;
- DY, with `--dy-reweight zpt+reshape`: the jet-pT reshape.

Trigger SFs:

- Electrons: one SF for the OR. Run 3 uses the EGM Ele30 TightID SF as a proxy; UL the Run 2 HEEP
  trigger-OR efficiency.
- Muons, 2022 to 2025: the per-leg data/MC efficiency OR. The payloads cover Mu50, CascadeMu100 and
  HighPtTkMu100; Mu50_L1SingleMuShower is not covered. Variation in 2024 and 2025 as in Run 2
  (SKFlat `MCCorrection::MuonTrigger_SF`): every muon's data efficiency up by its error with its MC
  efficiency down, and the reverse. In 2022 and 2023 the `muon_Z` payloads carry no efficiency-level
  systematic, so every muon's data efficiency moves by the relative uncertainty of the POG per-muon
  SF (statistical and tag-and-probe fit variations in quadrature) with the MC efficiency nominal.
- Muons, UL: the leading muon's SF alone. Two SFs are never multiplied.

Cutflows use the base weight without the DY Z-pT reweight and without the per-region SFs. Each
complete chain ends with `with_region_sfs`, the last cut times the lepton and LSF SFs of its
region, equal to that region's yield without the DY corrections.

### Drell-Yan

DY is the LO madgraphMLM HT-binned sample (`lo_ht`) in every era, at LO × the inclusive NNLO
k-factor of its generator class ([inputs.md](inputs.md)) and not reweighted by default. Two
corrections derived as in AN-19-083 switch on with `--dy-reweight`, for that sample only
(`dy_reweight_datasets`; other DY variants get the identity):

- Z-pT reweight (`wrcoffea/dy_reweight.py`): the NLO/LO ratio versus generator Z pT (dilepton
  system, m ≥ 50 GeV) of the inclusive NLO FxFX over the inclusive LO MLM sample, both at the NNLO
  cross section, in the 28 bins of the Run 2 derivation (integral 0.996 to 0.999: shape only), with
  scale, PDF and α_S variations, times the electroweak correction of Lindert et al.
  (arXiv:1705.04664) with its three nuisances. `scripts/derive_dy_zpt_reweight.py`; one payload per
  UL era, one for 2022+2023, one for 2024 (the Summer24 NLO sample is about 5 % softer).
- Jet-pT reshape (`wrcoffea/dy_reshape.py`): (data − non-DY MC) / DY MC versus the leading jet pT
  in the DY CRs, ee and μμ summed, in the Run 2 bins (resolved 0, 40, 100, 200, 400, 600, 800,
  1000, 2000 GeV on the AK4 jet; boosted 0, 200, 400, 600, 800, 1000, 2000 GeV on the AK8 jet),
  applied to DY in every region; nuisances: statistics and the ee-vs-μμ difference.
  `scripts/derive_dy_reshape.py` from a production without it; one payload per UL era and per Run 3
  data-taking year.

`--dy-reweight zpt` applies the first, `zpt+reshape` both (the reshape was derived on top of the
reweight and never comes alone); `none` is the default. The `dyzpt` and `dyreshape` families need
the matching correction on. Productions before 2026-10-12 have both corrections in their default
DY; `--no-dy-zpt-reweight` and `--no-dy-reshape` still work and mean `none` and `zpt`.

The HT-binned sets have no events below their first bin, LHE H_T = 40 GeV in Run 3 and 70 GeV in UL.
In 2024 this costs 9 to 16 % of the boosted-SR DY, about 1 % of the resolved-SR DY and nothing in
the control regions; the samples are not stitched.

### Signal normalization

The config cross section of every point is LO × the point's own NLO/LO k-factor, and the analyzer
applies the k(m(ℓN))/⟨k⟩ shape reweight; both come from our MG5_aMC NLO/LO runs per mass point
(`scripts/signal_kfactor/`, `wrcoffea/signal_kfactor.py`). `--no-signal-kfactor` switches the
shape part off.

## Blinding

Data of a blinded run (the default) never builds the signal regions or their variants, and its ee
and μμ cutflows stop before the m(ℓℓ) > 200 GeV cut; control regions and the eμ cutflows are
complete; MC is never blinded. The manifest records what a file withholds. `--unblind` is allowed
for the Run 2 eras only ([run_analysis.md](run_analysis.md#blinding)).

## Systematics

Finished and remaining, per Run: [systematics_status.md](systematics_status.md).

Every source is registered in `wrcoffea/systematics.py` with its family, kind (weight, shape,
normalization, members), the regions it can affect and the eras that have its inputs. A family an
era cannot produce is an error, never a silent copy; the coverage contract refuses partial
templates. Variations are written as `syst_<label lower>_<region>` folders.

| Family | Sources | Kind | What it varies |
| --- | --- | --- | --- |
| `lumi` | Lumi | normalization | the era's luminosity uncertainty, applied at export |
| `pileup` | Pileup | weight | the pileup profile reweighting |
| `prefire` | L1Prefiring | weight | the NanoAOD `L1PreFiringWeight` (UL only) |
| `sf` | MuonRecoSf, MuonIdSf, MuonIsoSf, MuonTrigSf, LooseMuonRecoSf, LooseMuonIdSf, ElectronRecoSf, ElectronIdSf, ElectronTrigSf, LooseElectronRecoSf | weight | each lepton scale-factor component, only in the regions it enters |
| `lsf` | LsfSf | weight | the LSF scale factor (boosted SR and flavour CRs) |
| `jer`, `jes` | JER, JES (the `Total` source) | shape | jet energy resolution and scale; the selection and the masses are recomputed (Run 3: the AK8 soft-drop mass too, through both subjets) |
| `jms`, `jmr` | JMS, JMR | shape | the residual soft-drop mass scale, (1 ± 0.01) m_SD, and resolution, m_gen + (m_SD − m_gen)(1 ± 0.02) with m_gen from the matched gen subjets (jets without a match unchanged); they move the m_SD > 40 GeV cut (JME jet mass recommendation, validated for 50 ≤ m_SD < 300 GeV and pT ≥ 100 GeV) |
| `elescale`, `elesmear` | EleScale, EleSmear | shape | electron energy scale and resolution (Run 3) |
| `muscale`, `musmear` | MuScale, MuSmear | shape | muon momentum scale and resolution (2022 to 2024: the high-pT recipe; UL and 2025: a scale placeholder and no `musmear`) |
| `dyzpt` | DYZptQcdScale, DYZptQcdPdf, DYZptQcdAlphaS, DYZptEw1, DYZptEw2, DYZptEw3 | weight | the Z-pT reweight's QCD and EW uncertainties (DY only; needs `--dy-reweight zpt` or `zpt+reshape`) |
| `dyreshape` | DYReshapeStat, DYReshapeChannel | weight | the reshape's statistical and ee-vs-μμ uncertainties (DY only; needs `--dy-reweight zpt+reshape`) |
| `scale` | ScaleMuR, ScaleMuF, ScaleMuRF | weight | QCD renormalization and factorization scales from `LHEScaleWeight`, each normalized to the dataset's mean so only acceptance and shape move; the plotter takes the envelope |
| `ps` | PSISR, PSFSR | weight | parton-shower ISR and FSR from `PSWeight`, normalized the same way |
| `pdf` | PDF, AlphaS | members | every `LHEPdfWeight` member stored per PDF set (`pdfmembers_<region>/<var>_lha<id>`); the Hessian quadrature and the α_S shift (scaled to 0.0015) are formed at the final binning by the plotter and the datacards |

What each era can produce:

| Family | UL16APV, UL16, UL17, UL18 | 2022, 2022EE, 2023, 2023BPix | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| `lumi`, `pileup`, `sf`, `jer`, `jes`, `jms`, `jmr`, `dyzpt`, `lsf`, `scale`, `pdf` | yes | yes | yes | yes |
| `muscale` | placeholder (TuneP, a 5 %/TeV × pT shift of both charges) | yes (the high-pT recipe) | yes | placeholder |
| `musmear` | no (no high-pT muon payload) | yes | yes | no |
| `prefire` | yes | no (no prefiring in Run 3) | no | no |
| `elescale`, `elesmear` | no (no scale-and-smearing payload; UL NanoAODv9 MC has no scale uncertainty branch) | yes | yes | yes |
| `dyreshape` | yes | yes | yes | no (no 2025 reshape) |
| `ps` | yes (the UL skims derive PSSumw from the events) | yes | yes | yes |

Samples without generator weights (the Pythia 8 WW, WZ, ZZ) get their nominal for `scale` and
contribute no `pdf` members, with a warning. A dataset without the dataset-level means its family
needs is an error. The plotter's band and the systematics tables take every source present in the
files ([WR_Plotter/docs/plotting.md](../WR_Plotter/docs/plotting.md)).

## Open items

- Low-mass closure of the data-driven tt̄ prediction (AN-19-083 App. J): the SR and flavour-CR
  selections with m < 800 GeV, data against R × flavour-CR data. Deferred; needs a decision on
  whether data may fill those SR-like regions.
- LSF3 above 1: NanoAOD's lsf3 exceeds 1.1 for 7 % (tt̄) and 20 % (W+jets) of the jets passing the
  0.75 cut, 2 for 1.3 % and 3.8 %, with values up to about 27 (2024 skims). Not explained.
- Muon momentum at high pT: the Muon POG Run 3 recipe is applied to 2022, 2023 and 2024; UL and
  2025 keep TuneP with a 5 %/TeV × pT scale shift and no resolution variation. UL needs the Run 2 GE
  maps and a Rochester-type correction below 200 GeV; 2025 has no POG recipe yet.
- Muon reco SF above 1.5 TeV: the POG high-pT reco SF has an open-ended last bin (p > 1.5 TeV, 2024
  p > 0.9 TeV) whose barrel values range from 0.68 (2023BPix) to 1.02 (2022) with large errors, where
  high-mass signal muons land. Options: clip to the last measured bin, clip with the spread as a
  systematic, a common high-p value.
- Electron SFs above the measured range: reco SFs are clipped at 500 GeV and HEEP ID SFs at 1 TeV
  (the payloads hold 1 ± 20 % above); the 2024 HEEP bins between 700 GeV and 1 TeV (1.08 to 1.12)
  are statistics-limited.
- Electron trigger SF: Run 3 uses the EGM Ele30 TightID SF as a proxy for the OR; a tag-and-probe
  measurement is needed. 2025 uses the 2024 file.
- LSF SF: Run 2 values per year from AN-19-083 Table 60; Run 3 carries 1.00 ± 0.10 until measured.
- UL electron scale and smearing variations: UL NanoAODv9 MC has `Electron_dEscaleUp/Down`
  identically zero; options are the data scale uncertainty mirrored onto MC or EgammaPostRecoTools
  at MiniAOD level.
- Soft-drop mass definition: corrected (default) versus the raw subjet mass Run 2 used, about 9 % of
  the UL18 boosted DY CR.
- Soft-drop mass in UL: the JME recipe needs the AK4 PUPPI JEC and JER, which the UL JERC payloads do
  not carry; UL keeps NanoAOD's m_SD, and JES and JER do not move it there.
- JMS and JMR are validated for 50 ≤ m_SD < 300 GeV; the cut sits at 40 GeV.
- Official signal samples have no N → ℓtb: the gridpacks' `jj` excludes the top and the N width
  counts only the generated channels, so for m_N above a few hundred GeV the samples put the full
  rate into ℓℓ + light jets while about a third should go to ℓtb.
- Signal cutflows (issue #107): the ee and μμ chains count only events of their own generator
  flavour since each point's file holds both flavour samples; takes effect with the next signal
  production, then close the issue.
- Luminosity correlations between 2022 and 2024 belong in the datacards.
- 2025: data-only, configured, skims in progress; electron trigger SF and muon momentum are
  placeholders.
