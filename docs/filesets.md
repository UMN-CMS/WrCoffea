# Creating Filesets

The first step is to find the files that will be fed into the analyzer. This can either be skimmed files located at UMN and Wisconsin, or unskimmed files queried from DAS.

## Unskimmed Filesets

Use `scripts/full_fileset.py` to build fileset JSONs from DAS. This queries `dasgoclient` for the logical file names of each dataset in the config, and uses the FNAL XRootD redirector (`root://cmsxrootd.fnal.gov/`) which automatically routes to the nearest available replica.

```bash
python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_data.json
python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_signal.json
python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_mc_dy_lo_inc.json
```

Optionally filter to a single `physics_group` with `--dataset`:
```bash
python3 scripts/full_fileset.py --config data/configs/.../RunIII2024Summer24_data.json --dataset Muon
```

Output path:
```
data/filesets/Run3/2024/RunIII2024Summer24/unskimmed/RunIII2024Summer24_data_fileset.json
```

To run the analysis on unskimmed files, pass `--unskimmed`:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy lo_inc
```

## Skimmed Filesets

The skim campaign writes the skimmed filesets itself, from the campaign plan and the part sidecars (see [Skimming](skimming.md)):

```bash
python3 bin/skim.py fileset --plan-dir plans/v2 --out ~/work/skims_v2 --era RunIII2024Summer24
```

- **Files:** the entries point at the skims_v2 parts at Wisconsin (`--local` points at the local parts instead).
- **Normalisation:** `genEventSumw`, `genEventCount` and the theory-weight sums (`LHEScaleSumw`, `LHEPdfSumw`, `PSSumw`) are taken from the skims' own Runs trees, so no DAS pass is needed. Nominal and extension datasets of one sample are normalised together. The configs carry no `genEventSumw`, and an MC dataset whose skims carry none stops the step with an error.
- **Metadata:** cross sections, `order` and `xsec_source` come from the configs through the plan; after a config change, run `bin/skim.py refresh-metadata --plan-dir ...` first.
- **Output:** one file per config family, written into `data/filesets/` (`--fileset-dir` overrides), e.g.

```
data/filesets/Run3/2024/RunIII2024Summer24/skimmed/RunIII2024Summer24_mc_dy_lo_ht_fileset.json
```

Every active skimmed fileset (2022-2024, UL18) points at skims_v2.

### Earlier skims (skims_v1)

The filesets of the skims made before October 2026 are kept under `skimmed/archived/skims_v1/` for reference. Pass one with `--fileset` to run on those files; nothing writes new ones.

## Config Files

Dataset configurations live in `data/configs/` organized by Run/Year/Era:

```
data/configs/
  Run3/2024/RunIII2024Summer24/
    RunIII2024Summer24_data.json
    RunIII2024Summer24_signal.json
    RunIII2024Summer24_mc_dy_lo_ht.json      # default DY (one file per DY variant: lo_ht, lo_inc, nlo_inc, ...)
    RunIII2024Summer24_mc_dy_nlo_mll.json
  Run3/2022/Run3Summer22/
    ...
  RunII/2018/RunIISummer20UL18/
    ...
```

Each config JSON maps `physics_group` names to DAS dataset paths. The fileset scripts read these configs and produce the corresponding fileset JSONs under `data/filesets/`.

## Fileset Directory Layout

```
data/filesets/
  Run3/2024/RunIII2024Summer24/
    skimmed/       # Filesets pointing to pre-skimmed files
    unskimmed/     # Filesets pointing to full NanoAOD via XRootD
  Run3/2022/Run3Summer22/
    ...
  RunII/2018/RunIISummer20UL18/
    ...
```

## Validation

To check that a fileset is valid before running the analysis:
```bash
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --preflight-only
python3 bin/run_analysis.py RunIII2024Summer24 Signal --mass WR4000_N2100 --preflight-only
```
