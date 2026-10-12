# Luminosity

What this page answers: which golden JSON each era uses, how the luminosity of every era is
computed (brilcalc over the lumisections actually in the skims), the numbers in use and their
uncertainties. The numbers live in `wrcoffea/config.yaml` (`lumis`, `lumi_unc`, `lumi_jsons`) and
`WR_Plotter/data/lumi.yaml`; this page explains them. Verified on 2026-10-11.

## Golden JSONs

The analyzer applies the certified-lumisection JSON to data (`lumi_jsons` in `config.yaml`); MC
is untouched. The files are kept in `data/lumis/` and were compared byte for byte with the DQM
certification site (`https://cms-service-dqmdc.web.cern.ch/CAF/certification/`) on 2026-10-11.

| Era | File (`data/lumis/...`) | Note |
| --- | --- | --- |
| UL16APV, UL16 | `RunII/2016/RunIISummer20UL16/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt` | one legacy file for both halves |
| UL17 | `RunII/2017/RunIISummer20UL17/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt` | |
| UL18 | `RunII/2018/RunIISummer20UL18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt` | |
| 2022, 2022EE | `Run3/2022/Cert_Collisions2022_355100_362760_Golden.json` | |
| 2023, 2023BPix | `Run3/2023/Cert_Collisions2023_366442_370790_Golden.json` | |
| 2024 | `Run3/2024/RunIII2024Summer24/Cert_Collisions2024_378981_386951_Golden.json` | re-issued by DQM on 2026-08-04 (tracker review): run 385387 LS 1-2 dropped, 0.0009 fb⁻¹; installed 2026-10-11 |
| 2025 | `Run3/2025/RunIII2025Summer25/Cert_Collisions2025_391658_398903_Golden.json` | the latest listed (2026-07-30) |

## How the luminosity of an era is computed

The LUM POG recommendation is to compute the luminosity with the JSON actually applied, with
brilcalc and the physics normtag. Here that is taken one step further: the luminosity of an era is
the brilcalc luminosity of the golden lumisections **present in its skims**, so a lumisection lost
to a missing or broken input file never counts (decision of 2026-10-10). `scripts/processed_lumi.py`
does it:

1. A per-lumisection brilcalc table of the era's golden JSON (`--byls`, normtag_PHYSICS). Made once,
   on a cmslpc node in the BRIL container (about 3 minutes for 2024, more than 15 for 2016):

   ```bash
   IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/brilws-docker:latest
   apptainer -s exec -B /cvmfs -B /uscms_data -B /uscms/home -B /uscms/homes \
       --env PYTHONPATH=/home/bril/.local/lib/python3.10/site-packages $IMG \
       brilcalc lumi -c web --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
       -i <golden.json> -u /fb --byls --output-style csv -o golden_byls.csv
   ```

   (On lxplus: `source /cvmfs/cms-bril.cern.ch/cms-lumi-pog/brilws-docker/brilws-env`, then the same
   `brilcalc lumi ...` command.) The csv header prints "Norm tag: None" even with `--normtag`; the
   values are the normtag ones.
2. The `<dataset>_processed_lumis.json` masks that `skim.py validate --remote-lumis` writes for every
   data stream: the (run, lumisection) pairs in the skimmed parts.
3. `scripts/processed_lumi.py --byls golden_byls.csv --golden <golden.json> --masks <masks> --out lumi.json`
   sums, per channel, the luminosity of every golden lumisection weighted by the fraction of the
   channel's events present (Run 3 halves such as Muon0/Muon1 share every lumisection and hold half
   the events each; a family without halves counts 1 if present). The era's number is the mean of
   the electron and the muon channel.

The masks and the results are in the repository under
`data/lumis/<Run>/<Year>[/<era>]/processed_skims_v2/` (`*_processed_lumis.json`,
`processed_lumi_normtag_PHYSICS.json`); the byls tables (tens of MB) are not.

## Numbers in use (fb⁻¹)

| Era | Used | Electron stream | Muon stream | Golden total of the era's runs | Uncertainty |
| --- | ---: | ---: | ---: | ---: | ---: |
| UL16APV | 19.50 | 19.497 | 19.495 | 19.502 | 1.2 % |
| UL16 | 16.81 | 16.811 | 16.811 | 16.812 | 1.2 % |
| UL17 | 42.04 | 42.063 | 42.016 | 42.068 | 0.82 % |
| UL18 | 59.52 | 59.500 | 59.544 | 59.561 | 0.84 % |
| 2022 | 8.08 | | | | 1.4 % |
| 2022EE | 26.64 | | | | 1.4 % |
| 2022 total | 34.72 | 34.700 | 34.744 | 34.761 | |
| 2023 | 18.59 | | | | 1.3 % |
| 2023BPix | 9.675 | | | | 1.3 % |
| 2023 total | 28.27 | 28.261 | 28.265 | 28.281 | |
| 2024 | 109.70 | 109.722 | 109.675 | 109.843 | 1.6 % (preliminary) |
| 2025 | 110.63 | | | | 5 % (preliminary; no skims yet) |

- The 2022 and 2023 results were computed per data-taking year with the sub-era split taken from
  the run periods (`channel_by_period` in the result files); the sub-era numbers in `config.yaml`
  are the ones used.
- The UL numbers differ from the published Run 2 ones (41.48, 59.83) because the physics normtag
  of December 2025 recalibrated 2017 (+1.4 %) and 2018 (−0.5 %), and because only the skimmed
  lumisections count (0.03 to 0.1 % below the golden totals).
- Uncertainties: Run 2 from the 2026 LUM POG Run 2 table (UL18 0.84 %), Run 3 from
  LumiRecommendationsRun3 r39 (2022 1.4 %, 2023 1.3 %, 2024 1.6 % preliminary, 2025 5 %
  preliminary). They enter as the `lumi` systematic family, uncorrelated between eras; the
  correlated components the LUM POG gives for 2022 to 2024 belong in the datacards and are not
  implemented yet.
- 2025 is data-only (2024 MC through `mc_alias`) and has no skims: 110.63 fb⁻¹ is provisional.

## Per-era notes

Certified lumisections that exist in no file of the official datasets, and runs missing from a
reprocessing, are the reason the processed luminosity sits below the golden total.

- **2024** (2026-10-06): golden JSON 109.95 fb⁻¹ before the 2026-08-04 re-issue (C to I 109.82 plus
  2024B 0.13). The skims lack runs 386615 to 386617 (0.105 fb⁻¹, present only in
  `Run2024I-PromptReco-v1`, not in the `MINIv6NANOv15` reprocessing) and the lumisections absent
  from the datasets themselves (worst: Muon0 2024E, 1.9 % of that period's Muon0 luminosity).
- **2023** (2026-10-07): golden 28.281 fb⁻¹ (B 0.640, C 17.964, D 9.677); Run3Summer23 = 2023B + C,
  Run3Summer23BPix = 2023D.
- **2022** (2026-10-08): golden 34.761 fb⁻¹ (B 0.097, C 5.022, D 2.967, E 5.806, F 17.783, G 3.086);
  Run3Summer22 = 2022B + C + D, Run3Summer22EE = E + F + G. EGamma 2022E lacks 0.88 % of its
  certified lumisections in the dataset itself. 2022C has both `SingleMuon` and `Muon` (disjoint
  run ranges); the script takes their union per lumisection.
- **UL18** (2026-10-09): golden 59.561 fb⁻¹ (2018A 13.96, B 7.02, C 6.87, D 31.65 / 31.69 by stream).
  Certified lumisections in no file: SingleMuon 2018C 46 (0.18 %), EGamma 2018D 963 (0.76 %).
- **UL17, UL16, UL16APV** (2026-10-10): golden 42.068, 16.812 and 19.502 fb⁻¹; the 2016 byls table
  takes more than 15 minutes and is streamed to the csv, so wait for brilcalc to exit before
  copying it. The 2016B version splits (`ver1`, `ver2`) are disjoint run ranges and are counted as
  a union, not as halves.
- **brilcalc and the normtag**: for a single run the byls csv sum equals the `--normtag` summary
  exactly (run 297050: 0.106374 fb⁻¹ both ways, 0.103287 without the normtag), and the 2017
  golden total in summary mode with the normtag is the same 42.068, although the csv header says
  "Norm tag: None".

## Re-doing it for a new era or a new skim

```bash
python3 bin/skim.py validate --plan-dir plans/v2_<era> --remote <campaign URL> --remote-lumis   # writes the masks
python3 scripts/processed_lumi.py --byls golden<year>_byls_physics.csv --golden data/lumis/.../Cert_....json \
    --masks <skims root>/<run>/<year>/<era>/data/*/*_processed_lumis.json --out processed_lumi_normtag_PHYSICS.json
```

Then write the era's number into `wrcoffea/config.yaml` (`lumis`) and `WR_Plotter/data/lumi.yaml`,
and copy the masks and the result into `data/lumis/.../processed_skims_v2/`. A golden JSON that is
re-issued under the same name (2024 in 2026) changes the data selection; re-check the files
against the DQM site when an era is revisited.
