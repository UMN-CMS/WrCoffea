# UL cross-section campaign (13 TeV)

What this page answers: how the cross sections of the Run 2 UL configs were measured and normalised, which
13 TeV reference each process is scaled to, the Drell-Yan and W+jets k-factors per UL era, which official signal
points took their value from another UL era, and why the official and the private (Run 2) signal grids differ.
The config keys (`xsec`, `order`, `xsec_source`) are explained in [inputs.md](../inputs.md). Verified on 2026-10-12.

## The campaign

`scripts/genxsec_campaign.py` measured the UL samples with GenXSecAnalyzer: one MiniAOD file per sample, Condor at
the LPC, CMSSW_10_6_30 in the cc7 image. The raw results stay in the campaign work area, keyed
`primary|energy|era`. Each written config entry carries its own `xsec_source`. The UL18 values were written on
2026-10-10, the UL16APV, UL16 and UL17 values later the same day.

## 13 TeV references

The same references serve all four UL eras. Only the k-factors differ, because each era's generator headers are
measured on that era's samples.

| process | reference [pb] | order | source | how the sample gets it |
| --- | --- | --- | --- | --- |
| Z/γ* → ℓℓ, m(ℓℓ) > 50 GeV | 6077.22 for three flavours (2025.74 per flavour) | NNLO | FEWZ 3.1.b2; twiki StandardModelCrossSectionsat13TeV r28 (2023-08-03) | k = reference / inclusive header of the generator class (next section) |
| W → ℓν | 61526.7 for three flavours (W+ 11811.4 + W− 8677.3 = 20508.9 per flavour) | NNLO | FEWZ 3.1; same twiki and revision | k = reference / LO MLM inclusive W+jets header |
| ttbar | 833.9 | NNLO+NNLL | TOP twiki TtbarNNLO, m_t = 172.5 GeV | × 0.106146 dileptonic (88.515 pb), × 0.439309 semileptonic (366.34 pb) |
| tW | 35.85 per charge (71.7 for both) | NLO+NNLL | TOP twiki SingleTopRefXsec | × 0.545454 for `NoFullyHadronicDecays` (19.555 pb) |
| t-channel, 5FS, inclusive decays | 134.2 (top), 80.0 (antitop) | NNLO | TOP twiki SingleTopNNLORef (arXiv:2012.01574) | as is |
| s-channel, 4FS, leptonic | 10.32 (top + antitop) | NLO | TOP twiki SingleTopRefXsec | × BR(W → ℓν) (3.3623 pb) |
| WW | 118.8 | NNLO | NNLO QCD for qq (arXiv:1408.5243) with gg → WW × 1.4, NLO EW −1.2 % on qq (CMS, arXiv:2009.00119: measured 117.6 ± 6.8 pb) | as is |
| WZ | 50.7 | NNLO | NNLO QCD × NLO EW, MATRIX v2.0.0, W+Z + W−Z with 60 < m(Z) < 120 GeV (CMS, arXiv:2110.11231 Table 6: measured 50.6 ± 2.1 pb) | as is |
| ZZ | 16.5 | NNLO | NNLO QCD × NLO EW for qq, NLO QCD for gg → ZZ, MATRIX v2.0.0, 60 < m(Z) < 120 GeV (CMS, arXiv:2009.01186 Sec. 8: measured 17.4 ± 0.8 pb) | as is |
| tribosons, ttW, ttZ | none | LO or NLO | the generator value | generator × factor |

## Drell-Yan and W+jets k-factors

The LO madgraphMLM DY samples (inclusive, jet- and HT-binned) are scaled with the two inclusive M > 50 GeV samples
of the same campaign. Both are measured with GenXSecAnalyzer on the era's MiniAOD: the "After filter: final cross
section", after FxFx merging or MLM matching. Both are three-flavour values. UL18:

| sample | generator | cross section [pb] | events used |
| --- | --- | --- | --- |
| `DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8` | NLO, FxFx | 6415.0 ± 22.5 | 166,028 |
| `DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8` | LO, MLM | 5350.0 ± 13.4 | 94,313 |

Decision of 2026-10-10, every era: each DY sample is scaled to the inclusive NNLO cross section for
m(ℓℓ) > 50 GeV by k = NNLO / inclusive header of its own generator class. LO MLM and FxFX then share one total. At
13 TeV the reference is the FEWZ NNLO 6077.22 pb. For UL18, k = 6077.22 / 5350.0 = **1.136** for the LO MLM
samples (`order` `LO x k(NNLO)`) and 6077.22 / 6415.0 = **0.947** for the FxFX ones (`NLO x k(NNLO)`). At 13.6 TeV
the reference is MATRIX NNLO QCD, 6340.2 pb (2024: LO 1.162, FxFX 0.950).

LO MLM W+jets is built the same way, with the NNLO W → ℓν reference over the inclusive LO MLM `WJetsToLNu` header
of the era. It replaced the flat Run 2 factor 1.21.

| era | LO MLM DY header [pb] | k, LO MLM DY | FxFX DY header [pb] | k, FxFX DY | LO MLM W header [pb] | k, W+jets |
| --- | --- | --- | --- | --- | --- | --- |
| UL16APV | 5358.0 | 1.134 | 6407.0 | 0.949 | 53460 | 1.151 |
| UL16 | 5399.0 | 1.126 | 6394.0 | 0.950 | 54130 | 1.137 |
| UL17 | 5342.0 | 1.138 | 6422.0 | 0.946 | 53520 | 1.150 |
| UL18 | 5350.0 | 1.136 | 6415.0 | 0.947 | 53560 | 1.149 |

The earlier definition, NLO FxFX / LO MLM = 6415.0 / 5350.0 = 1.199 (Run 3 2024: 1.22289), scaled LO to the FxFX
header. That header sits 5.6 % above NNLO at 13 TeV and 5.2 % at 13.6 TeV. Run 2 (AN-19-083, SKFlat) used a flat
1.23 from the k-factor column of the generator group's SummaryTable1G25ns twiki, which states no derivation. A flat
inclusive ratio does not correct the shape: the Z-pT reweight and the DY reshape do.

## Official signal: LO x k_point

The official `WRtoN{El,Mu}to..JJ` samples get their GenXSecAnalyzer LO value times the point's own NLO/LO k-factor
(`order` `LO x k(NLO)`, `xsec_source.reference_source` "direct per-point sigma_NLO / sigma_LO = k"). k_point =
c x sigma_NLO / sigma_LO comes from our own MG5_aMC 2.9.18 runs of p p > N l with the official gridpack model,
parameter and run cards: fixed-order NLO [QCD] against the LO of the official card. c = 1.0184 +- 0.0043 restores the
2.1 % of the N resonance that the samples' decay chain truncates at bwcutoff = 15. The result is the NLO cross
section of that mass point. The method is AN-19-083 App. A. The derivation, the comparison with Run 2's <k> (within
1-3 % at the shared points) and the payloads are in [signal_kfactor.md](signal_kfactor.md). k rises from 1.24 at
m_WR = 1 TeV to 1.42 at 4 TeV, and to 1.45-1.66 at 8 TeV. The analyzer applies the same point's k(m_lN) / <k> as a
shape-only weight (`data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_13tev_20261010.json`).

History: on 2026-10-09 the 97 official points missing from Run 2's <k> table had been given the genWeight-weighted
mean of Run 2's shared k(m_lN) over their generated events (`"source": "ul18_official"` in
`signal_kfactor_run2legacy_v4_2018.json`, kept for reference). That shared-curve average overshoots the direct ratio
by 2-12 % at high mass, because the neighbouring points' tails leak into each core. The write of 2026-10-10 uses the
direct per-point k instead ("option 2").

## Official points measured in another UL era

The UL18 MiniAOD of 24 official points cannot be opened (tape only). The UL campaigns use the same 13 TeV
gridpacks, so these points take the value measured on the UL17 MiniAOD (UL16 for one point). It is recorded as
`xsec_source.copied_from`:

| sample | measured on | LO [pb] | k_point | LO x k_point [pb] |
| --- | --- | --- | --- | --- |
| `WRtoNEltoElElJJ_MWR4000_N900` | RunIISummer20UL17 | 0.001369 | 1.4133 | 0.0019349 |
| `WRtoNEltoElElJJ_MWR6000_N1500` | RunIISummer20UL17 | 6.029e-05 | 1.4381 | 8.6702e-05 |
| `WRtoNEltoElElJJ_MWR6000_N700` | RunIISummer20UL17 | 9.104e-05 | 1.3973 | 0.00012721 |
| `WRtoNEltoElElJJ_MWR8000_N100` | RunIISummer20UL17 | 3.713e-05 | 1.2704 | 4.7169e-05 |
| `WRtoNEltoElElJJ_MWR8000_N1100` | RunIISummer20UL17 | 8.805e-06 | 1.4379 | 1.2661e-05 |
| `WRtoNEltoElElJJ_MWR8000_N300` | RunIISummer20UL17 | 2.591e-05 | 1.3094 | 3.3928e-05 |
| `WRtoNEltoElElJJ_MWR8000_N3900` | RunIISummer20UL17 | 2.1e-06 | 1.6012 | 3.3625e-06 |
| `WRtoNEltoElElJJ_MWR8000_N5900` | RunIISummer20UL17 | 7.599e-07 | 1.6116 | 1.2247e-06 |
| `WRtoNMutoMuMuJJ_MWR1000_N100` | RunIISummer20UL17 | 3.495 | 1.2356 | 4.3183 |
| `WRtoNMutoMuMuJJ_MWR2000_N1500` | RunIISummer20UL16 | 0.03556 | 1.3105 | 0.046603 |
| `WRtoNMutoMuMuJJ_MWR2000_N1900` | RunIISummer20UL17 | 0.00261 | 1.3147 | 0.0034313 |
| `WRtoNMutoMuMuJJ_MWR4000_N3100` | RunIISummer20UL17 | 0.0002749 | 1.4167 | 0.00038945 |
| `WRtoNMutoMuMuJJ_MWR6000_N3700` | RunIISummer20UL17 | 2.48e-05 | 1.4429 | 3.5783e-05 |
| `WRtoNMutoMuMuJJ_MWR6000_N4100` | RunIISummer20UL17 | 1.88e-05 | 1.4390 | 2.7054e-05 |
| `WRtoNMutoMuMuJJ_MWR6000_N4500` | RunIISummer20UL17 | 1.307e-05 | 1.4422 | 1.885e-05 |
| `WRtoNMutoMuMuJJ_MWR6000_N5500` | RunIISummer20UL17 | 1.986e-06 | 1.4443 | 2.8684e-06 |
| `WRtoNMutoMuMuJJ_MWR8000_N1900` | RunIISummer20UL17 | 4.62e-06 | 1.5353 | 7.0931e-06 |
| `WRtoNMutoMuMuJJ_MWR8000_N3500` | RunIISummer20UL17 | 2.303e-06 | 1.5953 | 3.674e-06 |
| `WRtoNMutoMuMuJJ_MWR8000_N3700` | RunIISummer20UL17 | 2.22e-06 | 1.5989 | 3.5494e-06 |
| `WRtoNMutoMuMuJJ_MWR8000_N4700` | RunIISummer20UL17 | 1.522e-06 | 1.6070 | 2.4459e-06 |
| `WRtoNMutoMuMuJJ_MWR8000_N4900` | RunIISummer20UL17 | 1.381e-06 | 1.6029 | 2.2136e-06 |
| `WRtoNMutoMuMuJJ_MWR8000_N6900` | RunIISummer20UL17 | 2.442e-07 | 1.6130 | 3.9388e-07 |
| `WRtoNMutoMuMuJJ_MWR8000_N7900` | RunIISummer20UL17 | 7.242e-09 | 1.6557 | 1.199e-08 |
| `WRtoNMutoMuMuJJ_MWR8000_N900` | RunIISummer20UL17 | 1.11e-05 | 1.4045 | 1.559e-05 |

Control points measured in both UL18 and UL17/UL16 agree within 0.2-1.2 % (UL17 slightly higher). That is the
precision of the gridpack's own integration (0.7 % for WR2000_N100 in its `results.dat`):

| control sample | UL18 [pb] | other UL era [pb] | ratio |
| --- | --- | --- | --- |
| `WRtoNEltoElElJJ_MWR2000_N100` | 0.1447 | 0.1464 (RunIISummer20UL17) | 1.0117 |
| `WRtoNEltoElElJJ_MWR4000_N2100` | 0.0008077 | 0.0008133 (RunIISummer20UL17) | 1.0069 |
| `WRtoNMutoMuMuJJ_MWR2000_N100` | 0.1445 | 0.145 (RunIISummer20UL17) | 1.0035 |
| `WRtoNMutoMuMuJJ_MWR2000_N100` | 0.1445 | 0.145 (RunIISummer20UL16) | 1.0035 |
| `WRtoNMutoMuMuJJ_MWR4000_N100` | 0.001935 | 0.001941 (RunIISummer20UL17) | 1.0031 |
| `WRtoNMutoMuMuJJ_MWR6000_N100` | 0.0001648 | 0.0001652 (RunIISummer20UL17) | 1.0024 |
| `WRtoNMutoMuMuJJ_MWR6000_N5900` | 2.433e-07 | 2.444e-07 (RunIISummer20UL17) | 1.0045 |

The same rule filled the last three entries without a value on 2026-10-12. Their MiniAOD is on tape in their own
era, and no era's results held a measurement of them:

- UL16APV `WRtoNEltoElElJJ_MWR6000_N2100` from UL16: 4.862e-05 pb LO, 7.0367e-05 pb with k_point;
- UL16 `WRtoNEltoElElJJ_MWR4000_N1700` from UL17: 0.001024 and 0.0014505 pb;
- UL16 `WRtoNMutoMuMuJJ_MWR6000_N1700` from UL17: 5.528e-05 and 7.9785e-05 pb.

`scripts/genxsec_campaign.py apply` finds the donor itself: the same primary dataset in the nearest sibling UL era
first, then the other-flavour sample of the same mass point. The two flavours' generator values agree to about 1 %
(control table above).

## Official UL grid vs the private (Run 2) grid

At the 8 shared points the official per-flavour LO x <k> is above the AN-19-083 value (the private config value / 2):
x1.10 at WR2000_N100, x1.19 at WR4000_N100, x1.78 at WR6000_N5900. Both use the MG5_aMC model `EffLRSM_NLO` and the
process `p p > N l, N > l j j`. The gridpack cards on cvmfs (official
`phys_generator/gridpacks/RunII/13TeV/MadGraph5_aMCatNLO/v265/WRtoNLtoLLJJ_emu/`, private
`phys_generator/gridpacks/2017/13TeV/madgraph/V5_2.6.5/WRtoNLtoLLJJ/`) differ in two ways:

1. **Light heavy neutrinos.** The official samples decouple the other flavour's N (mass 10^6 GeV). The private grid
   has N_e and N_mu degenerate. The W_R width is 10 % larger in the private grid, and BR(W_R -> l N) per flavour is
   9.09 % instead of 10.0 %: a factor 1.10 everywhere. The paper (CMS-EXO-20-002) states the N "couple exclusively
   to leptons of the corresponding flavor, and have different masses".
2. **PDF.** The private gridpacks use `NNPDF31_nnlo_hessian_pdfas` (306000), which is negative at large x. They
   produce negative-weight events, and the signed cross section falls below |sigma| by 1 % (WR2000_N100), 10 %
   (WR4000_N100) and 40 % (WR6000_N5900). The AN values are the signed ones. The official gridpacks use the UL
   standard `NNPDF31_nnlo_as_0118_mc_hessian_pdfas` (325300) and have no negative weights.

The product reproduces the measured ratios (1.09, 1.18, 1.77). Each private sample is consistent with its own AN
cross section, since its events carry the same negative weights. But the two grids are different models: do not mix
them in one limit scan or interpolation without accounting for both effects.
