# Create Filesets
The first step is to find the files that will be fed into the analyzer. This can either be skimmed files that are located at UMN and Wisconsin, or unskimmed files that we will query with rucio and DAS.

## Unskimmed filesets
To create a fileset of unskimmed files, use a command of the form
```
python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_inc.json --dataset tt_tW
python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json --dataset Signal
python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json --dataset EGamma
```
where `--dataset` filters by the `physics_group` field in the config JSON: `DYJets`, `tt_tW`, `Nonprompt`, `Other` (for backgrounds), `Signal` (for signal files), or `EGamma` or `Muon` (for data). Note that this does not work if running at UMN, use the script below instead.

The output file will be of the form
```
data/filesets/Run3/2022/Run3Summer22/Run3Summer22_tt_tW_fileset.json
```

## Skimmed filesets
Creating a fileset from skims is very similar, except one does not need the `--dataset` argument. For example,
```
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_inc.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json
```
The outputted `json` file locates the skimmed nanoAOD files from Wisconsin storage, and creates filesets from each dataset.

For running at UMN, add the `--umn` flag to create the fileset from the skims at UMN,
```
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json --umn
```
The output file will be of the form
```
data/filesets/Run3/2022/Run3Summer22/Run3Summer22_mc_dy_lo_inc_fileset.json
```
