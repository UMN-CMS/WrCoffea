# Dataset configs: what changed on 2026-10-04 and why

The per-era config JSONs under `data/configs/` were rebuilt from DBS for the
full reskim. The research behind them is in this directory:

- `run2_ul_dataset_report.md`: RunIISummer20UL16APV, UL16, UL17, UL18 (NanoAODv9)
- `run3_dataset_report.md` and `run3_signal_report.md`: Run3Summer22, 22EE, 23, 23BPix,
  RunIII2024Summer24, RunIII2025Summer25 (data only)

Previous versions of every replaced config are kept under each era's
`archived/pre_20261004/`; the UL18 private 490-point signal grid and the
standard (non-GT36) UL18 data lists are under `RunIISummer20UL18/archived/`.

## Decisions taken with Billy (2026-10-04)

- NanoAOD versions: UL NanoAODv9; 2022/2023 stay on NanoAODv12 (the v15
  reprocessing has no WR signal and no binned DY; v15 configs are in `nanov15/`);
  2024 MINIv6NANOv15; 2025 is data only (PromptReco NanoAODv15) and reuses the
  2024 MC through `mc_alias` in `wrcoffea/config.yaml`.
- Run 2 data: the 15Feb2022 re-reco with the high-pT muon fix (`_GT36`
  NanoAODv9) for SingleMuon 2017C-F and SingleMuon and EGamma 2018A-D;
  SingleMuon 2017B stays standard (its GT36 version lacks 25 % of certified
  lumisections).
- Edge eras added: 2022B (SingleMuon, EGamma), 2023B (Muon0/1, EGamma0/1),
  2024B (Muon0/1, EGamma0/1, PromptReco NanoAOD).
- Low-mass DY companions (MLL 10-50) are flagged `"optional": true` and left
  out of skim plans by default.
- Semileptonic tW (`TWminustoLNu2Q`, `TbarWplustoLNu2Q`) added to every Run 3
  MC config as Nonprompt, matching the Run 2 `NoFullyHadronicDecays` tW.
- UL inclusive DY cross section corrected to 6077.22 pb (the committed
  2025.74 pb was the per-flavour value on a three-flavour sample).

## Cross sections: flavour convention (verified 2026-10-04 with GenXSecAnalyzer)

The Run 3 single-flavour DY samples (`DYto2E-*`, `DYto2Mu-*`) carry the
three-flavour LHE normalization: GenXSecAnalyzer on the inclusive
`DYto2E-2Jets_Bin-MLL-50` (electrons only in `LHEPart`) gives 6728 +- 52 pb,
the e+mu+tau NLO value, and on `DYto2E-2Jets_Bin-0J-MLL-50` 5382 pb. Generator
and XSDB numbers for these samples must therefore be divided by three; the
inclusive entries already follow that convention (2113.4 = 6340/3), and the
jet-binned entries were corrected on 2026-10-04 (0J 1794.0, 1J 324.1/325.2,
2J 124.1/124.7 pb). The three-flavour `DYto2L-*` samples and the UL
`DYJetsToLL_M-50` sample use the full values (6077.22 pb for UL inclusive).

GenXSecAnalyzer runs in the OrbStack VM (`~/work/skims_v2/_genxsec/run_genxsec.sh`)
on MiniAOD files copied to the Mac first; about one minute per file.

- Run 3 W+jets: HT x M(lnu)-binned (12 samples per era, including 2024) as in
  Run 2; the jet-binned 1J-4J set was removed on 2026-10-04. t-channel single
  top: 4FS in Run 3, 5FS in UL.
- Pre-legacy WR scan samples (`WRtoNLtoLLJJ_MWR500to3500`, `3500to5000`,
  `5000to6000`, rpscan, NanoAODv7 for Autumn18/Fall17/Summer16) carry one
  `GenModel_*` flag per mass point; they are the origin of the private UL18
  490-point grid and are to be skimmed and split per model (`skim.py split-scan`).

## Open items

- Cross sections marked `null` or `VERIFY` in the configs: UL18 and the Run 3
  eras were settled by the GenXSecAnalyzer campaign (written 2026-10-10;
  `docs/dataset_configs.md`, `run2_ul_dataset_report.md`). UL16, UL16APV and
  UL17 still carry the 2026-10-04 drafts with 233-236 `null` values each (the
  amcatnloFXFX DY mass bins, the `DYJetsToLL_{0,1,2}J_MLL_*` bins,
  `DY{1-4}JetsToLL_M-10to50`, the official signal points): GenXSecAnalyzer runs
  on their MiniAOD parents (`~/work/skims_v2/_genxsec/run_genxsec.sh`) settle
  them the same way, without XSDB access.
- The `nlo_mll` family means amcatnloFXFX mass bins in Run 2 and powheg
  per-flavour mass bins in Run 3; stitching of the binned families against the
  inclusive samples is not implemented downstream.

The meaning of every key in the config JSONs (including `xsec`, `order` and `xsec_source`) is documented in [`docs/dataset_configs.md`](../dataset_configs.md).
