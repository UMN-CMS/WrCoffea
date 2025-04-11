#!/bin/bash
set -o nounset
set -o pipefail

# ERA options available in the script
ERA_OPTIONS=(
  RunIISummer20UL18
  Run3Summer22
)

# Data options for Run3Summer22
DATA_OPTIONS=(
  EGamma
  Muon
)

# MC options (for all eras)
MC_OPTIONS=(
  DYJets
  TTbar
  tW
  WJets
  TTbarSemileptonic
  SingleTop
  TTX
  Diboson
  Triboson
)

# MASS options for signal mode
MASS_OPTIONS=(
  WR1200_N200 WR1200_N400 WR1200_N600 WR1200_N800 WR1200_N1100
  WR1600_N400 WR1600_N600 WR1600_N800 WR1600_N1200 WR1600_N1500
  WR2000_N400 WR2000_N800 WR2000_N1000 WR2000_N1400 WR2000_N1900
  WR2400_N600 WR2400_N800 WR2400_N1200 WR2400_N1800 WR2400_N2300
  WR2800_N600 WR2800_N1000 WR2800_N1400 WR2800_N2000 WR2800_N2700
  WR3200_N800 WR3200_N1200 WR3200_N1600 WR3200_N2400 WR3200_N3000
)

# Validate mandatory arguments: mode and era
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 {data|bkg|signal} era [additional options]"
  exit 1
fi

MODE="$1"
if [ "${MODE}" != "data" ] && [ "${MODE}" != "bkg" ] && [ "${MODE}" != "signal" ]; then
  echo "Error: First argument must be 'data', 'bkg', or 'signal'"
  exit 1
fi

# Mandatory era argument
SELECTED_ERA="$2"
valid=false
for era in "${ERA_OPTIONS[@]}"; do
  if [ "${era}" == "${SELECTED_ERA}" ]; then
    valid=true
    break
  fi
done
if [ "${valid}" != "true" ]; then
  echo "Error: Unknown era '${SELECTED_ERA}'. Valid options: ${ERA_OPTIONS[*]}"
  exit 1
fi

# Shift off the mode and era arguments, leaving additional options (if any)
if [ "$#" -ge 3 ]; then
  shift 2
  EXTRA_ARGS=("$@")
else
  EXTRA_ARGS=()
fi

# Function for data and bkg modes
run_analysis() {
  local era="$1"
  local process="$2"
#  echo "Running analysis for era ${era} and --process ${process}"
  python3 bin/run_analysis.py "${era}" "${process}" "${EXTRA_ARGS[@]}" || {
    echo "Error running analysis for process ${process} with era ${era}. Skipping..."
    return 1
  }
}

# Function for signal mode (with mass option)
run_signal_analysis() {
  local era="$1"
  local mass="$2"
#  echo "Running analysis for era ${era} and signal with --mass ${mass}"
  python3 bin/run_analysis.py "${era}" "Signal" --mass "${mass}" "${EXTRA_ARGS[@]}" || {
    echo "Error running signal analysis for mass ${mass} with era ${era}. Skipping..."
    return 1
  }
  python3 bin/run_analysis.py "${era}" "Signal" --mass "${mass}" --dir 3jets "${EXTRA_ARGS[@]}" || {
    echo "Error running signal analysis for mass ${mass} with era ${era} and 3jets. Skipping..."
    return 1
  }
}

# Run analysis based on mode
if [ "${MODE}" == "data" ]; then
  for process in "${DATA_OPTIONS[@]}"; do
    run_analysis "${SELECTED_ERA}" "${process}"
  done
elif [ "${MODE}" == "bkg" ]; then
  for process in "${MC_OPTIONS[@]}"; do
    run_analysis "${SELECTED_ERA}" "${process}"
  done
elif [ "${MODE}" == "signal" ]; then
  for mass in "${MASS_OPTIONS[@]}"; do
    run_signal_analysis "${SELECTED_ERA}" "${mass}"
  done
fi

echo "All analyses complete!"
