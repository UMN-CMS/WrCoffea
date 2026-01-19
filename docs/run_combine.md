## Run the combine
set up the environment -
```
source runcombine.sh
```
Create datacards -
```
python3 datacards.py
```
Get expected asymptotic limits
```
source v1_Asymptotic_limit.sh
```

## analyze the output & create the plots
```
cd plotting
hadd -f combine.root ../higgsCombine*root
make
./plotlimit input_combine.txt out_WRcombine.root WR
```

Add combine.root file path in the 'input_combine.txt' before running the above last command.

Get 2D plots 
```
root -b -q 'getExclusion.C("out_WRcombine.root")'
```

Get 1D limit plot as a function of WR mass
```
python3 limitPlotter.py limits_EE.root WR_cross_sections.txt output_limit_plot.png 1200
```

Get 1D limit plot as a function	of N mass
```
python3 limitPlotter_mN.py limits_EE.root WR_cross_sections.txt output_limit_plot_mW.png 1200
```