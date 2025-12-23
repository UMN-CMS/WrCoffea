source Asd
datacard_dir="Datacards/"

# List of WR_N mass combinations
masses=(
    "WR1200_N200" "WR1200_N400" "WR1200_N600" "WR1200_N800" "WR1200_N1100"
    "WR1600_N400" "WR1600_N600" "WR1600_N800" "WR1600_N1200" "WR1600_N1500"
    "WR2000_N400" "WR2000_N800" "WR2000_N1000" "WR2000_N1400" "WR2000_N1900"
    "WR2400_N600" "WR2400_N800" "WR2400_N1200" "WR2400_N1800" "WR2400_N2300"
    "WR2800_N600" "WR2800_N1000" "WR2800_N1400" "WR2800_N2000" "WR2800_N2700"
    "WR3200_N800" "WR3200_N1200" "WR3200_N1600" "WR3200_N2400" "WR3200_N3000"
)

# masses=(
#     "WR1200_N600"
#     "WR1600_N800"
#     "WR2000_N1000"
#     "WR2400_N1200"
#     "WR2800_N1400"
#     "WR3200_N800" "WR3200_N1200" "WR3200_N1600" "WR3200_N2400" "WR3200_N3000"
# )

#masses=("WR3200_N1600")

for mass in "${masses[@]}"; do
    # Match files like: Datacards/*WR_3jets*_{mass}.txt
    for card in ${datacard_dir}/*WR*E*_"${mass}".txt; do
        
        # Skip if file doesn't exist (avoid literal pattern)
        [[ -e "$card" ]] || continue

        base=$(basename "$card" .txt)
        wsfile="${datacard_dir}/${base}.root"

        # Extract WR and N from "WRxxxx_Nyyyy"
        WR_mass=$(echo "$mass" | grep -oP '(?<=WR)\d+')
        N_mass=$(echo "$mass" | grep -oP '(?<=_N)\d+')

        # Determine region string
        # e.g. WR_3jets_resolved_WR1400_N800 → extract text between "WR_" and "_WR"
        String=$(echo "$base" | grep -oP '(?<=WR_).*?(?=_WR)')

        # Encode mass (same as before)
        mH=$((WR_mass * 10000 + N_mass))

        echo ">>> Processing $base | WR=$WR_mass GeV, N=$N_mass GeV | mH=$mH | region=$String"

        # Convert datacard to workspace
        text2workspace.py "$card" -o "$wsfile"

        # Run combine
        combine -M AsymptoticLimits "$wsfile" -t -1 -m $mH -n "_${base}"

        echo "----------------------------------------"
    done
done
