# Running the analyzer

## Usage

```bash
python3 bin/run_analysis.py <era> <sample or mode> [--dir TAG] [--dy VARIANT] [--systs FAMILY ...] [...]

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
| `signal` | Signal: every point of `data/signal_points/<era>_mass_points.csv` (2025: the 2024 CSV, its `mc_alias`), or `--signal-points` |
| single | `DYJets`, `tt_tW`, `Nonprompt`, `Other`, `EGamma`, `Muon`, `Signal --mass WR<wr>_N<n>`; the subgroups of `--split-groups` (`TTbar`, `tW`, `tWSemiLep`, `WJets`, `TTSemiLep`, `SingleTop`, `Diboson`, `Triboson`, `ttV`) |

- An era's production is one command, `bin/run_production.py <era> [--condor]`: `data`, `bkg` and
  `signal` at once, with `--all-dy` the DY corrections and every DY variant that is a prediction,
  with `--only` any of them by name ([condor.md](condor.md#running)).
- Composite modes run the samples one after the other locally (3 Dask workers; a failing sample is
  logged and the next one runs) and in parallel with `--condor` at the LPC ([condor.md](condor.md)).
  `--signal-points grid` runs the 9-point grid (lowest, median and highest N for m_WR = 2, 4, 6 TeV)
  instead of every point, `--mwr-range MIN MAX` a range.
- Signal samples: Run 3 the official grid (100 points per 2022/2023 era) and the 2024 private
  production (1323 points, of which the 1167 with both an electron and a muon sample are in the
  mass-point CSV and run; the k-factor payload covers every point); UL the official pair `WRtoNEltoElElJJ` + `WRtoNMutoMuMuJJ`
  per point (105 points per era) into one file. The UL private pre-legacy scan (`WRtoNLtoLLJJ`, 490 points) runs
  only when its fileset is named: `--fileset data/filesets/RunII/2018/RunIISummer20UL18/skimmed/RunIISummer20UL18_signal_scan_fileset.json`.

## Output

```text
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>.root
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>_systematics.json   the manifest
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_signal_WR<wr>_N<n>.root    one file per signal point
WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_DYJets_<tag>.root          a DY sample other than the default
```

- `--dir` names the production; without it the files go under `<Era>/`. `--name` gives
  `WRAnalyzer_<name>_<sample>.root`.
- Every DY sample goes into the same `--dir`: the default (`lo_ht`, not reweighted) is
  `WRAnalyzer_DYJets.root`, any other `WRAnalyzer_DYJets_<tag>.root` with `<tag>` the `--dy` variant
  plus `_zpt` or `_zpt_reshape` for `--dy-reweight`. Productions before 2026-10-12 kept every variant
  as `WRAnalyzer_DYJets.root` in its own directory, `<dir>_<tag>_dy`.
- `<region>/<var>_<region>`: the nominal histograms, MC scaled to σ × L. Variations:
  `syst_<label lower>_<region>/<var>_syst_<label lower>_<region>`, the label in lower case with only
  letters and digits (`MuonTrigSfUp` → `muontrigsfup`). PDF members: `pdfmembers_<region>/<var>_lha<id>`
  (TH2: variable × member).
- `cutflow_resolved/`, `cutflow_boosted/`: per channel (`ee`, `mumu`, `em`; boosted also `me`) the
  `cumulative` and `onecut` cutflows, weighted and `_unweighted`, cut names as bin labels. Weighted
  bins use the base weight without the DY Z-pT reweight and the per-region SFs; a complete chain
  ends with `with_region_sfs`, equal to the region's yield without the DY corrections. Signal `ee`
  and `mumu` chains count only events of their own generator flavour. Boosted chains open with
  object rows (AK8 jet pT and |η|, soft-drop mass, tight ID; the lepton's pT and |η|, then ID); the
  LSF requirement and the loose partner have rows before the |Δφ| and in-jet rows. Older files carry
  shorter chains; the plotter reads the rows from the file.
- `event_weights/<region>/` (MC, nominal pass, not with `--unskimmed`): `positive` and `negative`
  (counts vs log10|w|) and `components/<name>`, one distribution per factor of w (generator weight,
  normalization, pileup, L1 prefiring in UL, DY Z-pT, signal k-factor, DY reshape, each lepton SF;
  `corrections` is their product).
- With `--save-events`: `events_<region>` TTrees of the signal regions (`mass`, `weight`).
- The manifest: the datasets and chunk counts, the requested families, the observables covered, per
  region the produced, normalization and unaffected variations, the PDF-set layouts, the `blinding`
  record and, for DY, `dy` (`variant`, `reweight`).

## Blinding

Data is blinded by default; MC never is.

- Data never builds the signal regions (`wr_ee_resolved_sr`, `wr_mumu_resolved_sr`,
  `wr_ee_boosted_sr`, `wr_mumu_boosted_sr`) or their variants (`_nomass`, `_nolsf`): no folders, no
  `--save-events` trees. The control regions are filled.
- The data `ee` and `mumu` cutflows stop before the m(ℓℓ) > 200 GeV cut (resolved chains end at
  `mlljj_gt800`, boosted at `hem_veto_boosted_sr`); the axis is shorter, not zero-filled. `em` and
  `me` are complete.
- The manifest's `blinding` entry lists `applied`, `signal_regions`, `cutflow_truncated_before`.
  Files written before blinding existed have no entry; the plotter never shows their signal-region
  data.
- `--unblind` fills the data signal regions and full cutflows, Run 2 only; Run 3 is refused at
  start-up. Blinded and unblinded outputs are never merged.

## Systematics production

`--systs FAMILY ...` adds the up/down variations of those families to every MC histogram; data
gets nominal only. Families and sources: `wrcoffea/systematics.py`; what each era has produced:
[systematics_status.md](systematics_status.md).

- A family or source the era cannot produce is an error at start-up (UL `elescale`/`elesmear`,
  Run 3 `prefire`, `musmear` in UL). A deliberate omission is `--omit-systs <Source>`,
  listed in the manifest.
- `scale` and `ps` need the dataset-level weight means in the fileset (`LHEScaleSumw`, `PSSumw`); a
  dataset without them fails at its first chunk.
- Weight sources share one pass; shape sources (`jer`, `jes`, `jms`, `jmr`, `elescale`, `elesmear`,
  `muscale`, `musmear`) re-run the selection on shifted kinematics. `lumi` is applied at export.
- Coverage: every chunk records the (region, variation) pairs it completed with the production
  configuration; chunks or datasets produced with different configurations are never merged, and a
  dataset that did not complete every applicable variation in every chunk makes the writer raise. A
  source that cannot affect a region (muon SFs in an `ee` region) writes no folder there and is
  listed as `unaffected`.
- `--syst-observables`: `fit` (default) varies `mass_fourobject` and `mass_twoobject`; `all` every
  histogram but the 2D eta-phi maps (for the productions behind the control plots); or a list.
  Control plots of a default production show the MC statistical band only.
- In zsh pass the families as separate words; a `$VAR` holding the list does not word-split.
- Samples without generator weights (the Pythia 8 WW/WZ/ZZ) get their nominal for `scale` and no
  PDF members, with one warning per dataset; events of the few 2022 files that lack the LHE weights
  get their nominal ([skimming.md](skimming.md#known-input-defects-skims_v2)). Any other mismatch in
  the number of weights is an error.
- Streaming the skims from Wisconsin for systematics runs: raise `--xrootd-timeout` to 120 to 300 s.

## Flags

| Flag | Meaning |
| --- | --- |
| `--list-eras`, `--list-samples`, `--list-masses` | Print the choices and exit (`--list-masses` per era). |
| `--preflight-only` | Validate the fileset path, schema and selection, then exit. |
| `--mass WR<wr>_N<n>` | The signal point (required for `Signal`). |
| `--signal-points all` (default), `grid` or `POINT ...`; `--mwr-range MIN MAX` | Signal points of a composite mode. |
| `--dy VARIANT` | DY variant (`dy_variants` per era in `config.yaml`); default `lo_ht`. `nlo_ptll` runs with a warning: its pT(ll) slices are added unstitched, so it is not a DY prediction. |
| `--region resolved`, `boosted` or `both` | Topologies to fill (default both). |
| `--unblind` | Fill the data signal regions and full cutflows (Run 2 only). |
| `--dy-reweight none`, `zpt` or `zpt+reshape` | Corrections of the LO HT-binned DY: none (default), the Z-pT reweight, or both. `--systs dyzpt` needs `zpt` or `zpt+reshape`, `--systs dyreshape` needs `zpt+reshape`. Only `lo_ht` takes it; output `WRAnalyzer_DYJets_lo_ht_zpt[_reshape].root`. `--no-dy-zpt-reweight` and `--no-dy-reshape` still work and mean `none` and `zpt`. |
| `--no-signal-kfactor` | Switch the signal k(m(ℓN))/⟨k⟩ reweight off; also runs points the payload has no average for. |
| `--unskimmed` | Read the unskimmed filesets (the original NanoAOD files); sums genWeight itself. |
| `--fileset PATH` | An explicit fileset; with `Signal`, `--mass` only has to be a point of that fileset. |
| `--dir TAG`, `--name SUFFIX` | Output sub-directory and file-name suffix. |
| `--debug` | Run without saving histograms. |
| `--study STUDY ...` | Extra regions: `transfer-factor` (`<region>_nomass`), `dy-cr-split` (the DY CR split at m(ℓℓ) = 100 GeV), `no-lsf` (`<region>_nolsf`). |
| `--save-events` | Write the signal-region (mass, weight) TTrees. |
| `--split-groups GROUP ...` | Composite runs: `tt_tW` as TTbar + tW (the UL `NoFullyHadronicDecays` tW samples are all `tW`), `Nonprompt` as WJets + TTSemiLep + SingleTop + tWSemiLep, `Other` as Diboson + Triboson + ttV. |
| `--systs FAMILY ...` | `lumi pileup prefire sf jer jes jms jmr elescale elesmear muscale musmear dyzpt lsf dyreshape scale ps pdf`. |
| `--syst-observables fit`, `all` or `NAME ...` | Histograms that receive the variations. |
| `--omit-systs SOURCE ...` | Leave out named sources of an enabled family (recorded). |
| `--condor`, `--max-workers N`, `--worker-memory SIZE`, `--worker-wait-timeout S`, `--threads-per-worker N` | Condor at the LPC; Dask workers (local 3; Condor 50 single, 200 composite, 500 unskimmed composite); worker memory (6GB, 8GB with `--systs`); the wait for the first worker (3600 s). [condor.md](condor.md#settings). |
| `--scheduler FILE` | Run on the workers of `bin/condor_cluster.py` (implies `--condor`; refused if the workers' copy of `wrcoffea/` or `data/` differs). [condor.md](condor.md#one-cluster-for-several-commands). |
| `--chunksize N`, `--maxchunks N`, `--maxfiles N` | Events per chunk (100000 with `--condor`, 250000 locally), chunks per file, files per dataset. |
| `--xrootd-timeout S` | XRootD read timeout per chunk (60 with `--condor`, 10 locally). |
| `--xrd-fallback`, `--xrd-fallback-timeout`, `--xrd-fallback-retries-per-redirector`, `--xrd-fallback-sleep` | Redirector fallback for unskimmed preprocessing (on by default with `--unskimmed`). |

Transfer-factor study (R = SR / flavour CR on tt̄, closure below 800 GeV, AN-19-083 Appendix J):

```bash
python3 bin/run_analysis.py RunIII2024Summer24 bkg --study transfer-factor --split-groups tt_tW --dir tf_study
cd WR_Plotter && ../.venv/bin/python bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir tf_study --tf-study --input WRAnalyzer_TTbar.root
```

## Troubleshooting

- Unskimmed runs crash on I/O: the original files are read through redirectors. `Runner(...)` in
  `bin/run_analysis.py` has `skipbadfiles=False`, so an unreadable file fails the sample; `True`
  skips such files and silently loses their events.
- "Operation expired" while streaming skims under `--systs`: raise `--xrootd-timeout`; on a
  high-latency link copy the skims locally and run with `--fileset` on a rewritten fileset.
- A smoke run's data/MC ratio (`--maxfiles 1`) means nothing: one file carries the full sum of weights.
