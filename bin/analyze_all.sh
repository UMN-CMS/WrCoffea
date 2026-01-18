#!/bin/bash
set -o nounset
set -o pipefail

# Always run from repo root so relative paths work no matter where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the active virtualenv, otherwise fall back to the repo-local .venv.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="$(command -v python)"
fi

# ERA options available in the script
ERA_OPTIONS=(
  RunIISummer20UL18
  RunIII2024Summer24
)

# Data options
DATA_OPTIONS=(
  Muon
  EGamma
)

# MC options (for all eras)
MC_OPTIONS=(
  DYJets
  tt_tW
  Nonprompt
  Other
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

# Build MASS_OPTIONS dynamically from the repo's canonical CSV file for the era.
MASS_OPTIONS=()
MASS_CSV="${REPO_ROOT}/data/signal_points/${SELECTED_ERA}_mass_points.csv"
if [ -f "${MASS_CSV}" ]; then
  while IFS= read -r mass; do
    if [ -n "${mass}" ]; then
      MASS_OPTIONS+=( "${mass}" )
    fi
  done < <(awk -F, 'NR>1{gsub(/^[ \t]+|[ \t]+$/,"",$1); gsub(/^[ \t]+|[ \t]+$/,"",$2); if($1!="" && $2!="") print "WR"$1"_N"$2}' "${MASS_CSV}")
else
  echo "Error: Signal mass CSV not found: ${MASS_CSV}"
  exit 1
fi

# Now shift off mode & era, EXTRA_ARGS will remain the same
shift 2
EXTRA_ARGS=( "$@" )

# Shift off the mode and era arguments, leaving additional options (if any)
#if [ "$#" -ge 3 ]; then
#  shift 2
#  EXTRA_ARGS=("$@")
#else
#  EXTRA_ARGS=()
#fi

# Function for data and bkg modes
run_analysis() {
  local era="$1"
  local process="$2"
#  echo "Running analysis for era ${era} and --process ${process}"
  "${PYTHON}" bin/run_analysis.py "${era}" "${process}" "${EXTRA_ARGS[@]}" || {
    echo "Error running analysis for process ${process} with era ${era}. Skipping..."
    return 1
  }
}

# Function for signal mode (with mass option)
run_signal_analysis() {
  local era="$1"
  local mass="$2"
#  echo "Running analysis for era ${era} and signal with --mass ${mass}"
  "${PYTHON}" bin/run_analysis.py "${era}" "Signal" --mass "${mass}" "${EXTRA_ARGS[@]}" || {
    echo "Error running signal analysis for mass ${mass} with era ${era}. Skipping..."
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
