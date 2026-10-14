# Running the analyzer

What this page answers: how to run `bin/run_analysis.py`, what it writes, how blinding and the
systematics production work, and what every flag does. What the selection and the corrections
are is in [analysis.md](analysis.md) and [corrections.md](corrections.md). Verified on 2026-10-12.

## Usage

```bash
python3 bin/run_analysis.py <era> <sample or mode> [--dir TAG] [--dy VARIANT] [--systs FAMILY ...] [...]
```

```bash
python3 bin/run_analysis.py RunIII2024Summer24 all --dir 20261011           # everything, LO HT-binned DY
python3 bin/run_analysis.py RunIII2024Summer24 bkg --dir 20261011           # the four background groups
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dir 20261011 --dy nlo_inc
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --dir 20261011
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --systs lumi pileup sf jer jes jms jmr elescale elesmear muscale musmear lsf scale ps pdf --dir 20261011
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --dy-reweight zpt+reshape --dir 20261011   # WRAnalyzer_DYJets_lo_ht_zpt_reshape.root
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --maxfiles 1 --maxchunks 1 --dir smoke   # a quick test
python3 bin/run_analysis.py RunIISummer20UL18 data --unblind --dir 20261011                     # Run 2 only
```

Eras (`--list-eras`): `RunIISummer20UL16APV`, `RunIISummer20UL16`, `RunIISummer20UL17`,
`RunIISummer20UL18`, `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`,
`RunIII2024Summer24`, `RunIII2025Summer25` (data-only; its MC is the 2024 MC through `mc_alias`).

| Mode | Samples |
| --- | --- |
| `all` | EGamma, Muon, DYJets, tt_tW, Nonprompt, Other, Signal |
| `data` | EGamma, Muon |
| `bkg` | DYJets, tt_tW, Nonprompt, Other |
| `mc` | DYJets, tt_tW, Nonprompt, Other, Signal |
| `signal` | Signal: the 9-point default grid (lowest, median and highest N for m_WR = 2, 4, 6 TeV), or `--signal-points` |
| single | `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal --mass WR<wr>_N<n>`; also the subgroups of the split groups (`TTbar`, `tW`, `WJets`, `TTSemiLep`, `SingleTop`, `Diboson`, `Triboson`, `ttV`, see `--split-groups`) |

Composite modes run the samples one after the other locally (3 Dask workers by default; a failing
sample is logged and the next one runs) and in parallel with `--condor` at the LPC
([condor.md](condor.md)). `--signal-points all` runs every point of
`data/signal_points/<era>_mass_points.csv`, `--mwr-range MIN MAX` restricts the points.

**Signal samples.** Run 3: the official grid (100 points per 2022/2023 era) and the 2024 private
production (137 points). UL: the official pair `WRtoNEltoElElJJ` + `WRtoNMutoMuMuJJ` per point
(105 points per era), processed into one file. The UL private pre-legacy scan (`WRtoNLtoLLJJ`,
490 points) runs only when its fileset is named, for checks:

```bash
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3800_N1600 --dir my_scan_check \
    --fileset data/filesets/RunII/2018/RunIISummer20UL18/skimmed/RunIISummer20UL18_signal_scan_fileset.json
```

## Output

```text
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>.root
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>_systematics.json   (the manifest)
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_signal_WR<wr>_N<n>.root    (one file per signal point)
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_DYJets_<tag>.root          (a DY sample other than the default)
```

`--dir` names the production; without it the files go directly under `<Era>/`. `--name` is
inserted after the prefix: `WRAnalyzer_<name>_<sample>.root`. Every DY sample goes into the
same `--dir` (since 2026-10-12): the default (`lo_ht`, not reweighted) is
`WRAnalyzer_DYJets.root`, any other `WRAnalyzer_DYJets_<tag>.root` with `<tag>` the `--dy`
variant plus `_zpt` or `_zpt_reshape` for `--dy-reweight zpt` or `zpt+reshape`
(`WRAnalyzer_DYJets_nlo_inc.root`, `WRAnalyzer_DYJets_lo_ht_zpt_reshape.root`). Before, every
variant was `WRAnalyzer_DYJets.root` in a directory of its own, `<dir>_<tag>_dy` by
convention. Inside a file:

- `<region>/<var>_<region>`: the nominal histograms (MC scaled to σ × L), one folder per region
  ([analysis.md](analysis.md) lists them). Variations, where produced:
  `syst_<label lower>_<region>/<var>_syst_<label lower>_<region>`, the label in lower case with
  only letters and digits kept (`MuonTrigSfUp` → `muontrigsfup`); PDF members:
  `pdfmembers_<region>/<var>_lha<id>` (TH2: variable × member).
- `cutflow_resolved/`, `cutflow_boosted/`: per channel (`ee`, `mumu`, `em`; boosted also `me`, the
  μ(e-jet) flavour CR, since 2026-10-14) the `cumulative` and
  `onecut` cutflows, weighted and `_unweighted`, with the cut names as bin labels. The weighted bins
  use the base event weight without the DY Z-pT reweight and without the per-region scale factors;
  a complete chain (not blinded) ends with `with_region_sfs`, the last cut times the lepton and LSF
  SFs of its region, equal to that region's yield without the DY corrections (since 2026-10-14).
  For signal, the `ee` and `mumu` chains count only events of their own generator flavour (outgoing LHE
  leptons), so a mass point's file, which holds both flavour samples, has single-flavour cutflows.
- `event_weights/<region>/` (MC, nominal pass only, not with `--unskimmed`): `positive` and
  `negative` (event counts vs log10|w| by the sign of the region's nominal weight) and
  `components/<name>` (one distribution per factor of w: generator weight, normalization, pileup,
  L1 prefiring (UL), DY Z-pT, signal k-factor, DY reshape, each lepton SF; `corrections` is their
  product).
- With `--save-events`: the `events_<region>` TTrees of the signal regions (`mass`, `weight`).

The manifest records the datasets and their chunk counts, the requested families, the
observables covered, per region the produced, normalization and unaffected variations, the
PDF-set layouts, the `blinding` record and, for DY, the `dy` record (`variant`, `reweight`; since
2026-10-12).

## Blinding

Data is blinded by default; MC never is.

- Data never builds the signal regions (`wr_ee_resolved_sr`, `wr_mumu_resolved_sr`,
  `wr_ee_boosted_sr`, `wr_mumu_boosted_sr`) or their variants (`_nomass`, ...): no folders, no
  `--save-events` trees. The DY and flavour control regions are filled as usual.
- The data `ee` and `mumu` cutflows stop before the m(ℓℓ) > 200 GeV cut (resolved chains end at
  `mlljj_gt800`, boosted at `hem_veto_boosted_sr`); the cut axis is shorter, not zero-filled. The
  `em` and `me` cutflows are complete.
- The manifest's `blinding` entry says what the file withholds (`applied`, `signal_regions`,
  `cutflow_truncated_before`). Files written before blinding existed have no entry, and the plotter
  never shows their signal-region data.
- `--unblind` fills the data signal regions and full cutflows. Allowed for the Run 2 eras only;
  every Run 3 era is refused at start-up. Blinded and unblinded outputs are never merged. Run 2
  workflows that need observed signal-region data (observed limits) need data made with `--unblind`.

## Systematics production

`--systs FAMILY ...` adds the up/down variations of those families to every MC histogram (data
always gets nominal only). The families, their sources, and what each era can produce are in
[analysis.md, Systematics](analysis.md#systematics). Rules the production enforces:

- A family or source the era cannot produce is an error at start-up, never a silent skip
  (UL `elescale`/`elesmear`, Run 3 `prefire`, `musmear` outside the eras with a high-pT muon
  payload: UL and 2025). A deliberate omission of one source is recorded
  with `--omit-systs <Source>` and listed in the manifest.
- `scale` and `ps` need the dataset-level weight means in the fileset (`LHEScaleSumw`, `PSSumw`).
  A dataset without them is an error when its first chunk is processed, not at start-up.
- Weight sources share one pass; shape sources (`jer`, `jes`, `jms`, `jmr`, `elescale`, `elesmear`,
  `muscale`, `musmear`) re-run the selection on shifted kinematics. `lumi` is applied at export.
- Coverage: every chunk records the (region, variation) pairs it completed, with the production
  configuration (families, omissions, observables, regions, blinding); chunks or datasets produced
  with different configurations are never merged, and a dataset that did not complete every
  applicable variation in every chunk makes the writer raise instead of writing a partial template.
  A source that cannot affect a region (muon scale factors in an `ee` region) writes no folder there
  and is listed as `unaffected`.
- Which histograms receive the variations (`--syst-observables`): by default (`fit`) the fit
  observables, `mass_fourobject` and `mass_twoobject`; `all` every histogram, for the productions
  behind the control plots (the AN kinematics), which then carry the full band; an explicit list names
  them. The 2D eta-phi jet maps are never varied, even with `all`. Nominal is always filled everywhere.
  Control plots of a production made with the default show the MC statistical band only.
- In zsh pass the families as separate words; a `$VAR` holding the list does not word-split.
- Samples without generator weights (the Pythia 8 WW/WZ/ZZ have no `LHEScaleWeight` or
  `LHEPdfWeight`) get their nominal weight for `scale` and contribute no PDF members, with one
  warning per dataset; events of the few 2022 files that lack the LHE weights get their nominal
  ([skimming.md](skimming.md), "Known input defects"). Any other mismatch in the number of weights
  is an error.
- Streaming the skims from Wisconsin works for systematics runs at a low round-trip time; raise
  `--xrootd-timeout` (120 to 300 s) for them.

## Flags

| Flag | Meaning |
| --- | --- |
| `--list-eras`, `--list-samples`, `--list-masses` | Print the choices and exit (`--list-masses` per era). |
| `--preflight-only` | Validate the fileset path, schema and selection, then exit. |
| `--mass WR<wr>_N<n>` | The signal point (required for `Signal`). |
| `--signal-points default`, `all` or `POINT ...`, `--mwr-range MIN MAX` | Signal points of a composite mode. |
| `--dy VARIANT` | DY variant (`dy_variants` per era in `config.yaml`); default `lo_ht`. `nlo_ptll` runs with a warning but is not a DY prediction: its pT(ll) slices are added unstitched (the comment in `validate_arguments`). |
| `--region resolved`, `boosted` or `both` | Topologies to fill (default both). |
| `--unblind` | Fill the data signal regions and full cutflows (Run 2 only). |
| `--dy-reweight none`, `zpt` or `zpt+reshape` | Corrections of the LO HT-binned DY sample: none (the default since 2026-10-12; the sample at its NNLO-normalized cross section), the Z-pT reweight, or the Z-pT reweight and the jet-pT reshape. `--systs dyzpt` needs `zpt` or `zpt+reshape`, `--systs dyreshape` needs `zpt+reshape`. Only the default variant (`lo_ht`) takes it. The output is `WRAnalyzer_DYJets_lo_ht_zpt[_reshape].root`. The old `--no-dy-zpt-reweight` and `--no-dy-reshape` still work, with a warning, and mean `none` and `zpt`. |
| `--no-signal-kfactor` | Switch the signal k(m(ℓN))/⟨k⟩ reweight off. |
| `--unskimmed` | Read the unskimmed filesets (the original NanoAOD files); sums genWeight itself. |
| `--fileset PATH` | An explicit fileset; with `Signal`, `--mass` only has to be a point of that fileset. |
| `--dir TAG`, `--name SUFFIX` | Output sub-directory and file-name suffix. |
| `--debug` | Run without saving histograms. |
| `--study transfer-factor` or `dy-cr-split` | Add a study's extra regions (`<region>_nomass`; the DY CR split at m(ℓℓ) = 100 GeV). |
| `--save-events` | Write the signal-region (mass, weight) TTrees. |
| `--split-groups GROUP ...` | Composite runs: write `tt_tW` as TTbar + tW, `Nonprompt` as WJets + TTSemiLep + SingleTop, `Other` as Diboson + Triboson + ttV. |
| `--systs FAMILY ...` | `lumi pileup prefire sf jer jes jms jmr elescale elesmear muscale musmear dyzpt lsf dyreshape scale ps pdf`. |
| `--syst-observables fit`, `all` or `NAME ...` | Histograms that receive the variations: default `fit` (the m(W_R) masses); `all` every histogram but the eta-phi maps. |
| `--omit-systs SOURCE ...` | Leave out named sources although their family is enabled (recorded). |
| `--condor`, `--max-workers N`, `--worker-wait-timeout N`, `--threads-per-worker N` | Execution: Condor at the LPC, Dask workers (local default 3; Condor 50 single, 200 composite skimmed, 500 composite unskimmed). |
| `--chunksize N`, `--maxchunks N`, `--maxfiles N` | Events per chunk (250000), chunks per file, files per dataset (for tests). |
| `--xrootd-timeout S` | XRootD read timeout per chunk (default 60 with `--condor`, 10 locally). |
| `--xrd-fallback`, `--xrd-fallback-timeout`, `--xrd-fallback-retries-per-redirector`, `--xrd-fallback-sleep` | Redirector fallback for unskimmed preprocessing (on by default with `--unskimmed`). |

### Transfer-factor study

The eμ transfer factor R = SR / flavour CR is measured on tt̄ and its closure checked below
800 GeV (AN-19-083 Appendix J):

```bash
python3 bin/run_analysis.py RunIII2024Summer24 bkg --study transfer-factor --split-groups tt_tW --dir tf_study
cd WR_Plotter && ../.venv/bin/python bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir tf_study --tf-study --input WRAnalyzer_TTbar.root
```

## Troubleshooting

- **Unskimmed runs crash on I/O.** The original files are read through redirectors and see more
  transient errors than the skims. The `Runner(...)` call in `bin/run_analysis.py` has
  `skipbadfiles=False`, so an unreadable file makes the sample fail. Setting it to `True` there
  skips such files, at the price of silently losing their events; there is no flag for it.
- **"Operation expired" while streaming skims** under `--systs`: raise `--xrootd-timeout`; on a
  high-latency link copy the skims locally and run with `--fileset` on a rewritten fileset.
- **A smoke run's data/MC ratio** (`--maxfiles 1`) means nothing: one file carries the full sum of
  weights.
