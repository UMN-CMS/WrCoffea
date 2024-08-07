# WrCoffea

## Getting Started
Begin by cloning the repository,
```
git clone git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
```
## Running the analyzer
### The first time
```
bash bootstrap.sh
```
The `./shell` executable can then be used to start an apptainer shell with a coffea environment. For more information: https://github.com/CoffeaTeam/lpcjobqueue
### Running on data
```
./shell coffeateam/coffea-dask-almalinux8:latest
```
To run over all 2018 data, enter the command 
```
python3 run_analysis.py 2018 Data --hists Data.root --executor lpc
```
The shell should then show
```
Starting an LPC Cluster

Starting to analyze 2018 Data files
Analyzing 241608232 SingleMuon Run2018A events.
Analyzing 119918017 SingleMuon Run2018B events.
Analyzing 109986009 SingleMuon Run2018C events.
Analyzing 513909894 SingleMuon Run2018D events.
Analyzing 339013231 EGamma Run2018A events.
Analyzing 153792795 EGamma Run2018B events.
Analyzing 147827904 EGamma Run2018C events.
Analyzing 752524583 EGamma Run2018D events.

Computing histograms...
```
The output is a root file (`Data.root`) containing histograms of kinematic variables across all basic analysis regions.

### Running on MC
The command below will locally analyze one root file from the 2018 UL DY+Jets background MC sample:
```
python3 run_analysis.py 2018 DYJets --hists example_hists.root --max_files 1
```

### Arguments
To run the analyzer, a sample set and process must be specified as arguments:

#### Mandatory Arguments
 - Year: Currently, only `2018` exists, but there are also plans to include the rest of Run II (2016 and 2017).
 - Process: The process to be analyzed. Options for background processes are `DYJets`, `tt+tW`, `tt_semileptonic`, `WJets`, `Diboson`, `Triboson`, `ttX`, `SingleTop`. To analyze signal MC samples, use `Signal`, or `Data` for Data.
 - Signal Mass: If the process is `Signal`, then the signal masses must also be specified via the flag `--mass`, for example `--mass MWR3000_MN2900`. To see all possible signal points, use `--help`.

#### Optional Arguments

`--hists`: Generate a root file with histograms of kinematic observables.

`--masses`: Generate a root file with branches of the 3-object invariant mass ($m_{ljj}$) and 4-object invariant mass ($m_{lljj}$) (only implemented if the process is `Signal`).

`--max_files`: Generate a root file with branches of the 3-object invariant mass ($m_{ljj}$) and 4-object invariant mass ($m_{lljj}$) (only implemented if the process is `Signal`).

`--executor`: Specify whether or not to run on the LPC. To run on the LPC, one must also run `./shell coffeateam/coffea-dask-almalinux8:latest`.

To run the analyzer without computing any output files (perhaps for debugging purposes), omit both `--hists` and `--masses`.

For more information, enter:
```
python3 run_analysis.py --help
```
