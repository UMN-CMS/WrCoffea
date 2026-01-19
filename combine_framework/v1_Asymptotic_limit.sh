#!/bin/bash

datacard_dir="/export/scratch/users/aalpana/combineSetup/SimuFits_Limits/shape_workflow/Datacards/Run3/2022/Run3Summer22"

for card in ${datacard_dir}/WR*_N*.txt; do
    base=$(basename "$card" .txt)
    wsfile="${datacard_dir}/${base}.root"

    # Extract WR and N masses from filename
    # Example: WR2000_N700 -> WR_mass=2000, N_mass=700
    WR_mass=$(echo "$base" | grep -oP '(?<=WR)\d+')
    N_mass=$(echo "$base" | grep -oP '(?<=_N)\d+')
    String=$(echo "$base" | grep -oP '(?<=WR_).*?(?=_WR)')
    # Encode masses into a single integer: WR_mass * 10000 + N_mass
    mH=$((WR_mass * 10000 + N_mass))

    echo ">>> Processing $base | WR=$WR_mass GeV, N=$N_mass GeV | mH=$mH | region=$String"
    
    # Convert datacard to workspace
    text2workspace.py "$card" -o "$wsfile"

    # Run Combine using encoded mass
    combine -M AsymptoticLimits "$wsfile" -t -1 -m $mH -n "_${base}"

    echo "----------------------------------------"
done
