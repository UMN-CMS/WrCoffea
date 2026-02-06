# Create Filesets
The first step is to find the files that will be fed into the analyzer. This can either be skimmed files that are located at UMN and Wisconsin, or unskimmed files that we will query from DAS.

## Unskimmed filesets
To create a fileset of unskimmed files, use `scripts/full_fileset.py`. This queries DAS via `dasgoclient` for the logical file names of each dataset in the config, and builds fileset JSONs using the FNAL XRootD redirector (`root://cmsxrootd.fnal.gov/`) which automatically routes to the nearest available replica.

```
python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_data.json
python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_signal.json
python3 scripts/full_fileset.py --config data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_mc_dy_lo_inc.json
```
You can optionally filter to a single `physics_group` with `--dataset`:
```
python3 scripts/full_fileset.py --config data/configs/.../RunIII2024Summer24_data.json --dataset Muon
```

The output file will be of the form
```
data/filesets/Run3/2024/RunIII2024Summer24/unskimmed/RunIII2024Summer24_data_fileset.json
```

To run the analysis on unskimmed files, pass `--unskimmed`:
```
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy LO_inclusive
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
