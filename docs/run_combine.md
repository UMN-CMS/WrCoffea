# Expected Limits with Combine

Compute expected asymptotic limits using the Higgs Combine framework. All scripts live in `combine_framework/`.

## Setup

Source the Combine environment:
```bash
source combine_framework/runcombine.sh
```

## Create Datacards

Generate datacards from the analyzer output histograms:
```bash
python3 combine_framework/datacards.py
```

## Run Asymptotic Limits

```bash
source combine_framework/v1_Asymptotic_limit.sh
```

## Analyze Output and Create Plots

Merge the Combine output files and produce limit plots:

```bash
cd combine_framework/plotting
hadd -f combine.root ../higgsCombine*root
make
./plotlimit input_combine.txt out_WRcombine.root WR
```

Add the `combine.root` file path in `input_combine.txt` before running the last command.

### 2D Exclusion Plot

```bash
root -b -q 'getExclusion.C("out_WRcombine.root")'
```

### 1D Limit Plot (vs WR Mass)

```bash
python3 limitPlotter.py limits_EE.root WR_cross_sections.txt output_limit_plot.png 1200
```

### 1D Limit Plot (vs N Mass)

```bash
python3 limitPlotter_mN.py limits_EE.root WR_cross_sections.txt output_limit_plot_mW.png 1200
```
