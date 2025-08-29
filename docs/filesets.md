# Create Filesets
The first step is to find the files that will be fed into the analyzer. This can either be skimmed files that are located at UMN and Billy's EOS LPC area, or unskimmed files that we will query with rucio and DAS. 

## Unskimmed filesets
To create a fileset of unskimmed files, use a command of the form 
```
python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_lo_dy.json --dataset TTbar
python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json --dataset Signal
python3 scripts/full_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json --dataset EGamma
```
where `--dataset` can be `DYJets`, `TTbar`, `TW`, `WJets`, `SingleTop`, `TTbarSemileptonic`, `TTV`, `Diboson`, `Triboson` (for backgrounds), `Signal` (for signal files), or `EGamma` or `Muon` (for data). Note that this does not work if running at UMN, use the script below instead.

The output file will be of the form
```
data/jsons/Run3/2022/Run3Summer22/unskimmed/Run3Summer22_TTbar_fileset.json
```

## Skimmed filesets
Creating a fileset from skims is very similar, except one does not need the `dataset` argument. For example,
```
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_mc_lo_dy.json 
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json 
```
The outputted `json` file locates the skimmed nanoAOD files from Billy's EOS LPC area, and creates filesets from each dataset.

For running at UMN, add the `--umn` flag to create the fileset from the skims at UMN,
```
python3 scripts/skimmed_fileset.py --config data/configs/Run3/2022/Run3Summer22/Run3Summer22_data.json --umn
```
The output file will be of the form
```
data/jsons/Run3/2022/Run3Summer22/skimmed/Run3Summer22_mc_lo_dy_skimmed_fileset.json
```
