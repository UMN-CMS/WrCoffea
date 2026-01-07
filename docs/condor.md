## Running on Condor on the LPC
From the working directory of your project, download and run the boostrap script:
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```
Note this only needs to be done once.

This creates two new files in this directory: `shell` and `.bashrc`. The `./shell` executable can then be used to start an apptainer shell with a coffea environment. The current image I use is
```
./shell coffeateam/coffea-base-almalinux9:0.7.29-py3.10
```
Then, run the analyzer with the `--condor` option,
```
python3 bin/run_analysis.py Run3Summer22EE DYJets --condor
```

### Environment note (preferred)
This repo is primarily developed with a requirements-driven environment (see `requirements.txt`).
If you use the apptainer shell above, install the pinned deps inside that environment before submitting jobs:
```
python3 -m pip install -r requirements.txt
```
