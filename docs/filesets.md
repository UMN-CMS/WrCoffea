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
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy LO_inclusive
```

## Skimmed Filesets

Use `scripts/skimmed_fileset.py` to build filesets pointing to pre-existing skimmed NanoAOD files at Wisconsin or UMN storage. No `--dataset` argument is needed.

```bash
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_inc.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
```

For UMN storage instead of Wisconsin, add the `--umn` flag:
```bash
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json --umn
```

Output path:
```
data/filesets/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_inc_fileset.json
```

## Config Files

Dataset configurations live in `data/configs/` organized by Run/Year/Era:

```
data/configs/
  Run3/2024/RunIII2024Summer24/
    RunIII2024Summer24_data.json
    RunIII2024Summer24_signal.json
    RunIII2024Summer24_mc_dy_lo_inc.json
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
