# History

What this page answers: which choices were retired, when, and what replaced them, so that old
histogram sets, notes and plots can be read correctly. Newest first. Dates are the dates of the
change in this repository; the commit is given where one commit carries it.

## 2026-10

- **2026-10-12** The last three official UL signal datasets without a cross section (UL16APV e WR6000_N2100,
  UL16 e WR4000_N1700, UL16 mu WR6000_N1700: MiniAOD on tape in their own era, no measurement in any era's
  results) take the generator value measured on the same primary dataset in the nearest sibling UL era, as
  the 24 UL18 points did on 2026-10-10 (`xsec_source.copied_from`); `scripts/genxsec_campaign.py apply`
  now finds that donor itself, with the other-flavour sample of the same mass point as the fallback, and
  no longer rewrites the date of entries whose values it leaves unchanged. The three points were missing
  from every UL16 and UL16APV histogram set made before this date.
- **2026-10-11** A missing required noise-filter flag, and a missing or unreadable golden JSON, stop a
  run instead of warning (review finding R6). The mass-point CSV is read only for runs that involve
  Signal, so a data-only era without one (2025) runs. One `ERA_SUBDIRS` definition (`era_utils`)
  replaces the two copies in `das_utils` and `skim.plan`. The UL filesets carry parton-shower weight
  means estimated from original files (`scripts/estimate_ps_sums.py`), making `--systs ps` producible
  for UL until a reskim provides the exact sums.
- **2026-10-11** The 2022/2023 ECAL bad-calibration veto (runs 362433 to 367144) uses PuppiMET, as
  the JME recipe prescribes; it had used PF MET while the first skims lacked PuppiMET. On the
  skimmed 2022G and 2023C data the two reject the same events to within 3 to 8 per million.
- **2026-10-11** The pre-rebuild config copies, the retired DY variant configs, the UL18
  standard-NanoAOD data and private-scan configs and the skims_v1 filesets were removed from the
  tree (`dbdd2d2`); they stay in the history. `scripts/full_fileset.py` went the same day: the
  unskimmed filesets now come from `bin/skim.py unskimmed-fileset` and the skimmed ones from
  `bin/skim.py fileset`. The `studies/` folder (a 2026-01 m(ℓℓ) optimisation script reading
  outputs that no longer existed) was deleted.
- **2026-10-11** UL16, UL16APV and UL17 data: the SinglePhoton primary datasets were dropped from
  the configs and filesets. Both streams filled every region without overlap removal, so the
  electron channel was counted 1.5 to 2.2 times against Run 2. The Photon trigger paths count on
  the SingleElectron stream, as in the Run 2 analysis. UL17 electrons trigger on
  Ele35_WPTight_Gsf or Photon200 only (Ele115 was off for the first ~5 fb⁻¹ and Run 2 left it out).
- **2026-10-11** The DY Z-pT reweight is derived the AN-19-083 way for every era (inclusive NLO
  FxFx over inclusive LO MLM, Run 2 bins, both samples at the NNLO cross section, so the weight is
  a pure shape). The stitched LO/NLO derivation tried on 2026-10-08/10 was never used for a
  production. The UL jet-pT reshapes were re-derived per era on the prefiring-weighted
  backgrounds (`ad92e05`: ratios moved by up to 8 to 10 % resolved, 6 to 7 % boosted).
- **2026-10-11** L1 prefiring weights (UL), the muon trigger OR from the POG efficiencies for
  2022/2023, central HEEP SFs for 2022/2023, a 1.00 ± 0.10 LSF placeholder for Run 3, the UL JERC
  payloads in the CAT layout, the 2025 era configuration and PDF-set tables for every era
  (`103a9a3`). The 2024 golden JSON was replaced by the file re-issued on 2026-08-04 (run 385387
  LS 1-2 dropped, 0.0009 fb⁻¹).
- **2026-10-11** Every Run 3 Muon POG payload replaced by the CAT release of 2026-06-18; the
  2022/2023 scale-smearing parameters had been the pre-2026-04-28 ones (`db2ece7`).
- **2026-10-10** Luminosities: every era uses the brilcalc (normtag_PHYSICS) luminosity of the
  lumisections in its skims (UL18 59.52 instead of the published 59.83 fb⁻¹, UL17 42.04 instead of
  41.48). The 2026 Run 2 luminosity table gives UL18 0.84 %.
- **2026-10-10** Cross sections: every DY sample (LO and NLO) scaled to the inclusive NNLO
  cross section of its energy (MATRIX 6340.2 pb at 13.6 TeV, FEWZ 6077.22 pb at 13 TeV) by a
  k-factor per generator class; before, the Run 3 LO samples were scaled to the FxFX header
  (k 1.223 to 1.241) and the NLO samples kept their headers. LO MLM W+jets: the flat Run 2 factor
  1.21 replaced by the same construction with the NNLO W cross section. Signal: LO times the
  point's own NLO/LO k-factor in every era (`docs/studies/signal_kfactor.md`).
- **2026-10-10** UL18 jet veto maps drop the vetoed jets and keep the event (`jetveto_mode: jet`),
  as Run 2 did; the Run 3 eras reject the event. The event rule had cost UL18 17 to 20 % of the
  events against Run 2.
- **2026-10-10** High-pT muon momentum (MuonScaRe below 200 GeV, generalized-endpoint correction and
  extra smearing above) switched on for 2024 (`wrcoffea/muon_momentum.py`); histogram sets made
  before 2026-10-10 12:00 CDT have uncorrected muons.
- **2026-10-08** Default DY sample `lo_ht` (LO madgraphMLM HT-binned) in every era; the LO
  inclusive and NLO samples are cross-checks. Blinding moved into the analyzer: data files carry
  no signal-region folders, `--unblind` is Run 2 only.
- **2026-10-04 to 10-08** skims_v2: every era reskimmed with the `wrcoffea/skim` package (branch
  policy, zstd:19, 2 GB parts, LuminosityBlocks trees, verification contract). The skims made
  before October 2026 (`WRAnalyzer/skims/` at Wisconsin: 1209 branches, about 3.2 kB per event,
  zlib:1, no LuminosityBlocks tree, genEventSumw from a separate DAS pass, UL18 without rho) are
  no longer used by any fileset. The dataset configs were rebuilt from DBS with measured cross
  sections (`1bb790d`).

## 2026-09

- **2026-09-07** Run 2 ports for UL18: DY Z-pT reweight (the Run 2 payload), JES/JER, HEEP
  trigger-OR efficiency, HEM 15/16 veto, LSF SF, signal k-factor. The systematics registry
  (families, sources, coverage contract, manifests).
- **2026-09-06** Region variants and studies (`--study transfer-factor`, `dy-cr-split`,
  `--split-groups`).

## Earlier

- **2026-02-19** Unskimmed filesets regenerated for 2024 and UL18; older ones removed.
- **2026-01-18** `studies/` folder added (removed 2026-10-11).
