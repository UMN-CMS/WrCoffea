# Signal NLO/LO k-factor

How the QCD NLO/LO correction of W_R -> l N production is derived and applied, following AN-19-083 App. A.
Everything here is reproducible with `scripts/signal_kfactor/` (first derivation 2026-10-09/10).

## What is applied (decided 2026-10-10, "option 2")

- **Normalisation.** The cross section of a signal sample in the configs is `LO x k_point`: the GenXSecAnalyzer LO
  value of the sample times the point's own `c x sigma_NLO / sigma_LO` (payload `direct_k`;
  `scripts/genxsec_campaign.py apply --signal-norm direct`, the default), i.e. the NLO cross section of that mass
  point. Points without a dedicated run (the private UL18 grid) take a linear interpolation in
  (m_WR, m_N / m_WR) of the points that have one.
- **Shape.** Every event gets the weight `k(m_lN) / <k>`: `k` is the point's own NLO/LO ratio as a histogram in
  the generator-level mass of the heavy neutrino and the primary lepton (the W_R* virtuality; payload
  `point_curves`, the shared curve of all points as fallback), `<k>` the genWeight-weighted mean of that `k` over
  all generated events of the sample and lepton channel (payload `averages`, from `average_k.py`), so the weight
  averages to 1 and only moves events along m_lN (`wrcoffea/signal_kfactor.py`, payload per era in
  `config.yaml: signal_kfactor`). Points without a run use the nearest run point's shape scaled to their `k_point`.

Run 2 (AN-19-083) normalised with `LO x <k>` of one shared curve instead (`--signal-norm average`). With one curve
summed over all points, the integrated `<k>` of a sample is 2-12 % above the point's direct ratio at high mass
(the neighbouring points' far tails, where `k` is 1.6-1.75, leak into its resonance core), while the direct ratios
reproduce Run 2's published `<k>` within 1-3 %; hence option 2.

## Derivation (one campaign per centre-of-mass energy)

For 24 points per energy (m_WR = 1, 2, ..., 8 TeV x m_N = 100 GeV, ~m_WR/2, m_WR - 100 GeV; plus m_WR = 9 and
10 TeV for the high-mass end), MadGraph5_aMC@NLO 2.9.18 runs

- the fixed-order NLO QCD cross section of `p p > n2 mu+` and `mu-` (the N decay and the lepton flavour do not
  enter a QCD correction of the production), with m_lN histogrammed for the full NLO result and for its Born part,
  with the 9-point muR/muF variation (`analysis_HwU_mlN.f`);
- the LO cross section and 100k events (400k for m_WR >= 8 TeV) with the **official LO run_card** of the CMS
  samples (`lhaid 325300`, the default MadEvent scale), from which the LO m_lN histogram is taken (`lhe_mln.py`).

Model (`EffLRSM_NLO`), param_card and run_card come from the official gridpacks on cvmfs
(`RunII/13TeV/MadGraph5_aMCatNLO/v265/WRtoNLtoLLJJ_emu` for 13 TeV, `RunIII/13p6TeV/.../v2918/WRtoNLtoLLJJ_mu`
for 13.6 TeV; the two models are identical); the W_R width is recomputed per point (`decay 34 auto`, which
reproduces the gridpack values). Each energy's jobs run on LPC Condor in the cc7 image with CMSSW_12_4_8
(`condor/submit.sh`, 4 points per job, ~15 min).

```
k_point        = c x sigma_NLO / sigma_LO                        (direct_k; the config normalisation)
k_point(m_lN)  = c x NLO_point(m_lN) / LO_point(m_lN)            (point_curves; 200 GeV bins, 0-10 TeV, bins with
                                                                  < 20 LO events take k_point)
k_shared(m_lN) = c x sum_points NLO(m_lN) / sum_points LO(m_lN)  (kfactor; the Run 2-style shared curve, fallback)
<k>_sample     = sum_events w_gen k_point(m_lN) / sum_events w_gen   (averages; average_k.py, unskimmed NanoAOD)
```

In the shared curve every point enters with its own cross section, as in the Run 2 derivation from one mixed
sample. `c` is the ratio of our production-only LO (`p p > n2 mu`, official run_card) to the samples' own LO
(GenXSecAnalyzer on the official production) at the points both share:

| energy | shared points | c |
| --- | --- | --- |
| 13 TeV | 105 | 1.0184 +- 0.0043 |
| 13.6 TeV | 95 | 1.0182 +- 0.0082 |

It is the Breit-Wigner truncation of the samples' decay chain, not a MadGraph-version effect (verified
2026-10-10). The official gridpacks generate `p p > nn ll, nn > ll jj jj` with `bwcutoff = 15`: the N is
integrated over |m - m_N| < 15 Gamma_N only, which keeps (2/pi) arctan(30) = 97.9 % of the resonance. With
Gamma_N / m_N = 7e-11 that window is 1e-7 GeV wide, so the missing 2.1 % is purely resonant, part of
sigma x BR physically, and leaves no kinematic trace in the sample. Checks (13.6 TeV, WR2000_N100, official
cards, MG 2.9.18): the gridpack's own process gives 0.1673 +- 0.0005 pb with bwcutoff 15 (the sample: 0.1675)
and 0.1708 +- 0.0006 pb with bwcutoff 1000 (production only: 0.1701). `c` is flat in both masses, as a universal
factor must be: 1.016-1.020 for m_WR = 1-8 TeV, the same with the N -> mu t b channel closed (m_N < 180 GeV)
or open. And MG 2.6.5, the version of the 13 TeV samples, reproduces 2.9.18 at four 13 TeV points to 0.3 % in
sigma_NLO and 0.1 % in sigma_LO (table below; run with a local LHAPDF 6.2.3 through `KF_LHAPDF_CONFIG` in
`run_points.sh`, because 2.6.5's compiled PDF code cannot find `pdfsets.index` with the CMSSW LHAPDF).

`c` therefore restores the full resonance: `sigma(sample) x k_point = sigma_NLO(production) x BR` for the
shared points. The measured mean is used rather than the pure-Breit-Wigner 1.0217 (0.4 % apart; the gridpack
integrations carry ~1 % errors of their own). The Run 2 private grid was generated the same way
(`p p > nn ll, nn > ll jj jj` with both N flavours, bwcutoff 15), and AN-19-083's LO x <k> applies no such
factor, so that normalisation sits ~2 % below sigma_NLO x BR.

MG 2.6.5 against 2.9.18, 13 TeV, official cards, req_acc_FO 0.002 / 100k LO events (2026-10-10):

| point | sigma_NLO 2.6.5 [pb] | sigma_NLO 2.9.18 [pb] | ratio | sigma_LO(card) 2.6.5 [pb] | 2.9.18 [pb] | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| WR1000_N100 | 4.300 +- 0.006 | 4.312 +- 0.006 | 0.997 | 3.5547 +- 0.0018 | 3.5538 | 1.000 |
| WR2000_N100 | 0.1886 +- 0.0003 | 0.1885 +- 0.0002 | 1.000 | 0.1473 +- 0.0001 | 0.1472 | 1.000 |
| WR4000_N2100 | 1.146e-3 +- 1.4e-6 | 1.147e-3 +- 1.3e-6 | 0.999 | 8.238e-4 +- 3.6e-7 | 8.238e-4 | 1.000 |
| WR6000_N5900 | 3.563e-7 +- 3.6e-10 | 3.571e-7 +- 3.4e-10 | 0.998 | 2.475e-7 +- 1.1e-10 | 2.476e-7 | 0.999 |

## Results

Direct per-point k = c x sigma_NLO / sigma_LO (official card), scale band = 9-point envelope on sigma_NLO
(final campaign: 130 points at 13 TeV, 159 at 13.6 TeV):

| point | 13 TeV | 13.6 TeV | Run 2 <k> (AN-19-083) |
| --- | --- | --- | --- |
| WR1000_N100 | 1.236 (+1.7/-1.6 %) | 1.229 | 1.213 |
| WR2000_N100 | 1.304 (+1.8/-2.4 %) | 1.297 | 1.291 |
| WR2000_N1900 | 1.315 (+2.5/-2.9 %) | 1.310 | 1.314 |
| WR4000_N100 | 1.370 (+2.4/-3.2 %) | 1.372 | 1.344 |
| WR4000_N3900 | 1.411 (+4.0/-4.6 %) | 1.408 | 1.417 |
| WR6000_N100 | 1.312 (+2.0/-2.5 %) | 1.315 | 1.271 |
| WR6000_N5900 | 1.469 (+4.4/-5.3 %) | 1.450 | 1.439 |
| WR8000_N100 | 1.270 (+1.8/-2.0 %) | 1.278 | - |
| WR8000_N7900 | 1.656 (+6.4/-7.4 %) | 1.614 | - |

The 13 and 13.6 TeV values agree within 1 %; both agree with Run 2 within 1-3 % (ours slightly higher, by about
`c`). The curves and the averaged-k plots (the figures of AN-19-083 App. A, with the Run 2 curves overlaid at
13 TeV) are in `~/work/skims_v2/_kfactor/plots/` (`kfactor_vs_mlN_<E>.pdf`, `kfactor_<E>.pdf`).

Two features of `k(m_lN)` that Run 2's smooth curve did not have, both properties of the samples:

- **The official card's default scale.** Relative to the NLO run's own Born (default scale H_T/2) the LO with the
  official card is lower by 6 % at m_WR = 1 TeV, 12 % at 2 TeV and 20 % at 6 TeV (the factorisation-scale
  dependence of the NNLO PDF set grows with x). NLO over that Born is a smooth 1.15-1.25.
- **MadEvent's far off-shell tails.** Between the resonance cores the LO (MadEvent) distribution is 25-30 % lower
  than the Born of the NLO run, also when generated with the same H_T/2 scale (checked with `lo_diag.sh`;
  `bwcutoff` has no effect); the official 2022 sample's gen-level m_lN shape matches the LO run. The weight is
  therefore larger in the tails (`k` ~ 1.6-1.75 between cores, ~1.2-1.5 at the cores).

Because one curve serves every sample, the integrated `<k>` of a point is 3-5 % above its direct ratio at
m_WR >= 4 TeV (the neighbouring points' tails raise the curve in its core); Run 2's Fig. 55 shows the same offset
("Integrated" vs "Inclusive").

## Validation of the payloads (2026-10-10)

- The averages reproduce the direct ratios where they must: `<k>_sample / k_point` = 1.002 +- 0.003 over the 210
  UL18 official samples, 1.002 +- 0.003 over the 200 Run3Summer22 ones and 1.002 +- 0.003 over the 237 samples of the
  2024 private production (every sample has a gen match for all of its events). For the 490 private UL18 points the
  shapes are borrowed from the nearest run point, so the spread is wider (1.004 +- 0.027, extremes 0.95 and 1.21);
  that changes only the event-by-event shape weight, which averages to 1 by construction, not the normalisation.
- `WRtoNMutoMuMuJJ_MWR2000_N100` runs end to end with the new payload at 13 TeV (unskimmed input) and 13.6 TeV
  (skim); the 13.6 TeV output carries the `signal_kfactor` weight component.
- `config.yaml: signal_kfactor` maps UL18 and every Run 3 era to the new payloads since 2026-10-10 02:50. Signal
  histograms made before that carry no k-factor shape weight in Run 3 and the Run 2 one in UL18.
- The config cross sections are `LO x k_point` since 2026-10-10 (UL18 written at 05:35, the five Run 3 eras at
  06:55; `order` = `LO x k(NLO)`; the skimmed filesets were regenerated and installed, backups under
  `~/work/skims_v2/`). Run 3 signal histograms made before 06:55 are normalised at LO, 28 to 62 % low; UL18 sets
  made before 05:35 had cross sections for the 16 AN-19-083 Table 23 points only.

## Open: the official samples have no N -> l t b (parked 2026-10-10)

What is established:

- The official gridpacks (13 and 13.6 TeV) generate `p p > nn ll, nn > ll jj jj` with
  `define jj = u d c s b u~ d~ c~ s~ b~`: no top. The Run 2 private grid (`WRtoNLtoLLJJ`, AN-19-083) used
  `jj = u d c s b t u~ d~ c~ s~ b~ t~` and both N flavours.
- The per-point `c` = sigma_LO(production) / sigma_LO(sample) is flat, 1.016-1.020, from m_N = 100 to 7900 GeV at
  both energies. A decay-chain cross section is sigma_prod x Gamma(generated channels) / Gamma_N(card); if the
  card's N width included N -> l t b, `c` would rise by 1 / BR(l j j) once that channel opens (m_N > m_t + m_b).
  It does not, so the N width in the official param_cards counts only the generated light-quark channels.
- Physically N -> l W_R* -> l q q~' has the channels u d, c s and t b with the same coupling, so for m_N >> m_t
  about one third of the N decays go to l t b (phase-space suppressed near threshold, m_N ~ 200-400 GeV). The
  official samples contain no such events and normalise l l + light jets to the full W_R -> l N rate.

What this means for the analysis (not evaluated): for m_N above a few hundred GeV the true l l q q'(light) rate
is ~2/3 of what the official samples claim, and the l t b events (t -> b W) have a different acceptance in both
topologies, so the predicted SR yield at those points is off by something between 0 and -33 % depending on that
acceptance. Run 2's efficiencies include the l t b events; ours (official samples) do not, which also matters for
the Run 2 comparison at high m_N.

How to pick it up:

1. Read `DECAY 9900014` of a heavy official point (e.g. WR6000_N3000, gridpack on cvmfs,
   `/cvmfs/cms.cern.ch/phys_generator/gridpacks/RunII/13TeV/MadGraph5_aMCatNLO/v265/WRtoNLtoLLJJ_emu/` and
   `.../RunIII/13p6TeV/MadGraph5_aMCatNLO/v2918/WRtoNLtoLLJJ_mu/`) and compare it with MG `compute_widths 9900014`
   on the full model (with t b): the ratio is 1 / BR(l j j_light). Do it for a few m_N to get BR(m_N).
2. Decide how to treat it: scale the official-sample normalisation by BR(N -> l j j_light)(m_N) and model the
   l t b part separately (a private production with the Run 2-style `jj`, or an acceptance study on the Run 2
   private grid, which has the events), or document the assumption. Either way the AN must state it.
3. The same `jj` definition applies to the 2024 private production (same gridpack cards); check its proc_card.

Tooling and records from the 2026-10-10 checks (LPC login node, `~/nobackup/kfactor/`): `test_job136/dec_test.sh`
(the bwcutoff test, 13.6 TeV WR2000_N100), `test_job13_lhapdf/` (the 2.6.5 route: `go.sh`, `go3.sh`,
`out/13_WR*`), `lhapdf-6.2.3/` built by `lhapdf_build/build.sh`, gridpack cards under `~/nobackup/provdump/cards/`
(`off_2000_100`, `priv_2000_100`). Copies of the two scripts: `scripts/signal_kfactor/build_lhapdf.sh` and
`scripts/signal_kfactor/decay_chain_test.sh`.

## Payloads

- 13 TeV: `data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_13tev_20261010.json` (600 `direct_k` and
  `point_curves`, 587 `averages`; `direct_k`,
  `point_curves` and the shared `kfactor` from `kfactor_analysis.py --payload --interpolate-to <the 490 private
  points>`; `averages` from `average_k.py` for the 210 official points and `--scan` for the 490 private points of
  the three NanoAODv7 scans). The points run directly: the whole official grid (105), m_WR = 3, 5, 7, 9, 10 TeV
  points for the curve, and m_WR = 600 and 800 GeV points for the private grid. UL16APV, UL16 and UL17 map to the
  same payload (`config.yaml`, 2026-10-10): their samples come from the same 13 TeV gridpacks, and both the curves
  and the averages are generator-level quantities of a mass point, not of an era.
- 13.6 TeV: `data/reweights/Run3/signal_kfactor_13p6tev_20261010.json` (159 direct points, 137 averages), shared by
  the 2022/2023 official samples and the 2024 private production (one average per mass point and channel; the
  samples of one point are the same gridpack in every era; the electron-only 2024 points have an `ee` average only).

Run 2's `k(m_lN)` and averages (`signal_kfactor_run2legacy_v4_2018.json`) stay in the repository for reference.
