# WrCoffea

## Getting Started
Begin by cloning the repository,
```
git clone git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
```
## Running the analyzer
### Each week
To set up a grid UI
```
voms-proxy-init --rfc --voms cms -valid 192:00
```
### Basic analysis
To run a basic analysis, 
```
python3 bin/run_analysis.py Run3Summer22 DYJets --hists --skimmed
```
More information can be found in the `README.md` file in other folders.
