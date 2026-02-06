## Basic Analysis
First, create the filesets (see the `filesets.md` README).

### Analyzing background samples
To run a basic analysis on a background sample, specify the era and the sample. For example,
```
python3 bin/run_analysis.py RunIII2024Summer24 DYJets
python3 bin/run_analysis.py RunIISummer20UL18 tt_tW
python3 bin/run_analysis.py Run3Summer22EE Nonprompt
```
Available eras and samples can be queried with `--list-eras` and `--list-samples` (see [Discovery helpers](#discovery-helpers) below). Background samples: `DYJets`, `tt_tW`, `Nonprompt`, `Other`.

By default the analyzer looks for the skimmed filesets, and the output histograms will be saved to
```
WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/WRAnalyzer_DYJets.root.
WR_Plotter/rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_DYJets.root.
```

### Analyzing signal samples
To analyze signal files, use the `--mass` flag with the desired signal point. For example,
```
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3200_N3000
python3 bin/run_analysis.py Run3Summer22 Signal --mass WR4000_N1900
```
Possible signal points can be listed with `python3 bin/run_analysis.py --list-masses`.

### Analyzing data samples
To analyze data, use a similar format
```
python3 bin/run_analysis.py RunIISummer20UL18 EGamma
python3 bin/run_analysis.py Run3Summer22EE Muon
```
where `EGamma` can also be replaced with `Muon`.


## Optional Arguments

#### Discovery helpers
To avoid drift with documentation, you can query the canonical CLI choices directly:
```
python3 bin/run_analysis.py --list-eras
python3 bin/run_analysis.py --list-samples
python3 bin/run_analysis.py Run3Summer22 --list-masses
```

To validate fileset paths and selection (including signal mass matching) without running processing:
```
python3 bin/run_analysis.py Run3Summer22 DYJets --preflight-only
python3 bin/run_analysis.py RunIISummer20UL18 Signal --mass WR3200_N3000 --preflight-only
```

#### `--dir`
One can further specify a directory to save to with the  `--dir` flag. For example, 
```
python3 bin/run_analysis.py RunIISummer20UL18 DYJets --dir 3jets
```
This will save the file to 
```
WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/3jets/WRAnalyzer_DYJets.root.
```

#### `--name`
Filenames can be modified with the `--name` flag. For instance,
```
python3 bin/run_analysis.py Run3Summer22 DYJets --name dr1p5
```
This will save the file 
```
WR_Plotter/rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_dr1p5_DYJets.root.
```
Of course, both flags can be used,
```
python3 bin/run_analysis.py Run3Summer22EE DYJets --dir 3jets --name dr1p5
```
This will save the file
```
WR_Plotter/rootfiles/Run3/2022/Run3Summer22EE/3jets/WRAnalyzer_dr1p5_DYJets.root.
```

#### `--debug`
When making changes to the analyzer, one may want to run the analyzer without saving histograms. This can be done with,
```
python3 bin/run_analysis.py Run3Summer22EE DYJets --debug
```

#### `--reweight`
If one wishes to reweight a particular sample, use 
```
python3 bin/run_analysis.py Run3Summer22EE DYJets --reweight reweight_file.json
```
where `reweight_file.json` is the output file of scripts/derive_reweights.py

#### `--unskimmed`
Run over unskimmed filesets (created by `scripts/full_fileset.py`) instead of the default skimmed filesets. Unskimmed filesets use the FNAL XRootD redirector for automatic site failover.
```
python3 bin/run_analysis.py RunIII2024Summer24 DYJets --unskimmed --dy LO_inclusive
python3 bin/run_analysis.py RunIII2024Summer24 Muon --unskimmed
python3 bin/run_analysis.py RunIII2024Summer24 Signal --unskimmed --mass WR2000_N1900
```

More information can be found in the `README.md` file in other folders.

## Analyzing all
To analyzer all backgrounds in one command, execute the script
```
./bin/analyze_all.sh bkg RunIISummer20UL18
```
To analyze all signal samples
```
./bin/analyze_all.sh signal Run3Summer22
```
Or to analyze both `EGamma` and `Muon`,
```
./bin/analyze_all.sh data Run3Summer22EE
```
Similar to above, one can also specifcy `--dir` and `--name` to save files to specific directories and filenames. For example,
```
./bin/analyze_all.sh bkg RunIISummer20UL18 --dir 3jets --name dr1p5
```
