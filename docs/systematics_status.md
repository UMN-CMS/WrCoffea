# Systematics: finished and remaining

What this page answers: which systematic uncertainties are finished for Run 2 UL and for Run 3,
meaning implemented, fed by a real input and produced in the final histogram sets, and which are
still placeholders, missing or waiting for a decision. How each family works is in
[analysis.md](analysis.md#systematics); the payloads are in [corrections.md](corrections.md);
the open GitHub issues are linked. Verified on 2026-10-11.

Run 2 = UL16APV, UL16, UL17, UL18. Run 3 = 2022, 2022EE, 2023, 2023BPix, 2024 (2025 has no
production yet). The final sets are `--dir 20261011_systs` in every era.

## Finished

| Source | Run 2 UL | Run 3 | Notes |
| --- | --- | --- | --- |
| Luminosity (`lumi`) | yes | yes | normtag_PHYSICS luminosity of the skimmed lumisections; uncertainties from the LUM POG tables (2024 and 2025 preliminary). Era correlations: see Remaining. |
| Pileup (`pileup`) | yes | yes | golden-JSON profiles per era |
| Muon reco, ID, isolation SFs (`sf`) | yes | yes | POG high-pT payloads; last-bin treatment above 1.5 TeV: see Remaining |
| Muon trigger SF (`sf`) | yes | yes | UL: the leading muon's SF; 2022/2023: per-leg efficiency OR from the POG `muon_Z` entry of our exact OR; 2024: per-leg OR from the high-pT HLT file |
| Electron reco SF (`sf`) | yes | yes | EGM payloads; Run 3 clipped at 500 GeV (see Remaining) |
| Electron HEEP ID SF (`sf`) | yes | yes | UL: flat per year; 2022/2023: the central EGM flat values; 2024: the HEEPID working point, clipped at 1 TeV |
| Electron trigger SF (`sf`) | yes | proxy | UL: the Run 2 HEEP trigger-OR efficiencies; Run 3: the Ele30 TightID SF stands in for the OR (Remaining, #94) |
| Loose-lepton SFs (`sf`) | yes | yes | boosted partner: muon reco + ID, electron reco |
| LSF SF (`lsf`) | yes | placeholder | UL: AN-19-083 Table 60 per year; Run 3: 1.00 ± 0.10 (Remaining, #95) |
| JES, JER (`jes`, `jer`) | yes | yes | the `Total` JES source; the regrouped sources are not split out |
| Electron energy scale and smearing (`elescale`, `elesmear`) | no | yes | UL NanoAODv9 MC carries no scale uncertainty (Remaining, #96) |
| Muon momentum scale and smearing (`muscale`, `musmear`) | placeholder | 2024 yes; 2022/2023 placeholder | the high-pT recipe is applied to 2024 only; elsewhere a flat 5 %/TeV scale and no smearing (Remaining, #91) |
| DY Z-pT reweight, 6 nuisances (`dyzpt`) | yes | yes | own derivations, AN-19-083 recipe, 2026-10-11; one payload per UL era, 2022+2023 together, 2024 alone |
| DY jet-pT reshape, 2 nuisances (`dyreshape`) | yes | yes | per UL era; per Run 3 data-taking year; none for 2025 yet |
| QCD scales μR, μF (`scale`) | yes | yes | normalized per dataset; the plotter takes the envelope |
| Parton shower ISR, FSR (`ps`) | yes (estimated means) | yes | UL filesets carry PSSumw estimated from the original files (`scripts/estimate_ps_sums.py`); exact sums at the next UL skim. Three UL signal points whose original file is unreachable keep their nominal (#104) |
| PDF and α_S (`pdf`) | yes | yes | every member stored; Hessian quadrature and the α_S shift formed at the fit binning |
| L1 prefiring (`prefire`) | yes | not applicable | NanoAOD `L1PreFiringWeight` |

"yes" means the variation is in the `20261011_systs` files of every era of that Run (the UL sets are
being remade with `ps` as this page is written) and the plotter's band and the systematics tables
([plotting.md](plotting.md)) pick it up.

## Remaining

| Item | Run 2 UL | Run 3 | What closes it | Issue |
| --- | --- | --- | --- | --- |
| Electron trigger-OR scale factor | done | proxy (Ele30 TightID) | a tag-and-probe measurement of the OR per era, a correctionlib file, the `ElectronTrigSf` variation rebuilt from its uncertainties | #94 |
| LSF scale factor | done | placeholder 1.00 ± 0.10 | a Run 3 measurement (semileptonic-top tag-and-probe, as in Run 2) | #95 |
| Muon momentum calibration at high pT | placeholder | 2024 done; 2022/2023 placeholder | payloads for the other eras (UL: Run 2 generalized-endpoint maps; 2022/2023: the POG maps) | #91 |
| Electron energy scale and smearing variations | missing | done | a prescription for UL (data scale uncertainty mirrored onto MC plus the MC smearing, or EgammaPostRecoTools) | #96 |
| Muon reco SF above the last measured bin (p > 1.5 TeV; 2024 > 0.9 TeV) | open | open | a treatment: clip, clip with the spread as a systematic, or a common high-p value | #92 |
| Electron SFs above the measured range (reco > 500 GeV, HEEP ID > 1 TeV, clipped) | not affected (flat SFs) | open | a high-pT prescription (freeze point and uncertainty) | #93 |
| Parton-shower means | estimated | done | a UL reskim (the skimmer now derives the exact sums) | #104 |
| Luminosity correlations between eras | open | open | the LUM POG correlated and uncorrelated components in the datacards | #105 |
| JES split by source | `Total` only | `Total` only | per-source nuisances in the production and the datacards, if the fit needs them | |
| Background normalization and MC statistics in the fit | not in the production | not in the production | datacard-level: free DY and tt̄+tW rates, the lnN priors of the other backgrounds, Barlow-Beeston (Run 2 choices in [analysis.md](analysis.md)) | |
| Signal cross-section uncertainty and the missing N → ℓtb channel | open | open | a decision on the per-point k-factor uncertainty and on the official samples' missing top channel | #99 |
| 2025 | | no production | skims, processed luminosity, a 2025 reshape, the 2025 electron trigger and muon momentum payloads | #100 |
| AK8 soft-drop mass definition (corrected vs raw) | open | open | not a systematic but a selection choice that moves the boosted DY CR by ~9 % in UL18 | #97 |

## How to check the state of a production

```bash
cd WR_Plotter
../.venv/bin/python bin/make_systematics_tables.py --era <era> --dir 20261011_systs      # which sources move which process, by how much
```

A source missing from a file is listed by the plotter when the band is built (a sample's nominal
stands in, never zero), and the analyzer's manifest next to each ROOT file records the requested
families, the produced and the unaffected variations per region.
