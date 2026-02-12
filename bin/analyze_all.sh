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

# ERA options derived from the Python config (single source of truth).
mapfile -t ERA_OPTIONS < <("${PYTHON}" -c "from wrcoffea.era_utils import ERA_MAPPING; print('\n'.join(ERA_MAPPING))")

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
  echo "Usage: $0 {data|bkg|signal|all} era [additional options]"
  exit 1
fi

MODE="$1"
if [ "${MODE}" != "data" ] && [ "${MODE}" != "bkg" ] && [ "${MODE}" != "signal" ] && [ "${MODE}" != "all" ]; then
  echo "Error: First argument must be 'data', 'bkg', 'signal', or 'all'"
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

# Mass CSV path (only checked when signal mode is actually used).
MASS_CSV="${REPO_ROOT}/data/signal_points/${SELECTED_ERA}_mass_points.csv"

# Now shift off mode & era, EXTRA_ARGS will remain the same
shift 2
EXTRA_ARGS=( "$@" )

# Set worker counts based on whether we're running on unskimmed files
UNSKIMMED=false
for arg in "$@"; do
  if [ "${arg}" = "--unskimmed" ]; then
    UNSKIMMED=true
    break
  fi
done

if [ "${UNSKIMMED}" = true ]; then
  WORKERS=400
else
  WORKERS=50
fi

# Filter out --systs for data mode (data doesn't use systematics)
DATA_ARGS=()
skip_systs=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [ "${skip_systs}" = true ]; then
    # Skip arguments after --systs until we hit another flag
    if [[ "${arg}" =~ ^-- ]]; then
      skip_systs=false
      DATA_ARGS+=("${arg}")
    fi
    # Otherwise skip this syst value (e.g., "lumi", "pileup", "sf")
  elif [ "${arg}" = "--systs" ]; then
    # Start skipping --systs and its values
    skip_systs=true
  else
    DATA_ARGS+=("${arg}")
  fi
done

# Function for data and bkg modes
run_analysis() {
  local era="$1"
  local process="$2"
  shift 2
  local mode_args=("$@")
#  echo "Running analysis for era ${era} and --process ${process}"
  "${PYTHON}" bin/run_analysis.py "${era}" "${process}" "${mode_args[@]}" "${EXTRA_ARGS[@]}" || {
    echo "Error running analysis for process ${process} with era ${era}. Skipping..."
    return 1
  }
}

# Function for signal mode (with mass option)
run_signal_analysis() {
  local era="$1"
  local mass="$2"
#  echo "Running analysis for era ${era} and signal with --mass ${mass}"
  "${PYTHON}" bin/run_analysis.py "${era}" "Signal" --mass "${mass}" --max-workers 10 --chunksize 50000 "${EXTRA_ARGS[@]}" || {
    echo "Error running signal analysis for mass ${mass} with era ${era}. Skipping..."
    return 1
  }
}

# Run analysis based on mode
if [ "${MODE}" == "data" ]; then
  # Use DATA_ARGS (filtered to exclude --systs) for data
  EXTRA_ARGS=("${DATA_ARGS[@]}")
  for process in "${DATA_OPTIONS[@]}"; do
    run_analysis "${SELECTED_ERA}" "${process}" --max-workers ${WORKERS}
  done
elif [ "${MODE}" == "bkg" ]; then
  for process in "${MC_OPTIONS[@]}"; do
    run_analysis "${SELECTED_ERA}" "${process}" --max-workers ${WORKERS}
  done
elif [ "${MODE}" == "all" ]; then
  echo "=== Running data ==="
  # Temporarily use DATA_ARGS (no --systs) for data
  SAVED_EXTRA_ARGS=("${EXTRA_ARGS[@]}")
  EXTRA_ARGS=("${DATA_ARGS[@]}")
  for process in "${DATA_OPTIONS[@]}"; do
    run_analysis "${SELECTED_ERA}" "${process}" --max-workers ${WORKERS}
  done
  # Restore EXTRA_ARGS (with --systs) for MC
  EXTRA_ARGS=("${SAVED_EXTRA_ARGS[@]}")
  echo "=== Running backgrounds ==="
  for process in "${MC_OPTIONS[@]}"; do
    run_analysis "${SELECTED_ERA}" "${process}" --max-workers ${WORKERS}
  done
  # Signal excluded from 'all' mode - use 'signal' mode explicitly if needed
fi

if [ "${MODE}" == "signal" ]; then
  if [ ! -f "${MASS_CSV}" ]; then
    echo "Error: Signal mass CSV not found: ${MASS_CSV}"
    exit 1
  fi
  # Default signal behavior: pick ~9 points: WR=2000/4000/6000, and for each WR take
  # N = (min, median, max) from the era's mass-point CSV. If those WR values don't exist
  # for the era, fall back to 9 evenly spaced points from the full list.
  mapfile -t MASS_OPTIONS < <("${PYTHON}" - <<PY
import csv
from pathlib import Path

csv_path = Path(r"${MASS_CSV}")
rows = []
with csv_path.open() as f:
  reader = csv.reader(f)
  header = next(reader, None)
  for r in reader:
    if not r or len(r) < 2:
      continue
    try:
      wr = int(str(r[0]).strip())
      n = int(str(r[1]).strip())
    except Exception:
      continue
    rows.append((wr, n))

by_wr = {}
for wr, n in rows:
  by_wr.setdefault(wr, set()).add(n)

target_wrs = [2000, 4000, 6000]
selected = []
for wr in target_wrs:
  if wr not in by_wr:
    continue
  ns = sorted(by_wr[wr])
  if not ns:
    continue
  picks = [ns[0]]
  if len(ns) > 2:
    picks.append(ns[len(ns)//2])
  if len(ns) > 1:
    picks.append(ns[-1])
  # unique, preserve order
  seen = set()
  for n in picks:
    if n in seen:
      continue
    seen.add(n)
    selected.append(f"WR{wr}_N{n}")

if not selected:
  # Fallback: choose 9 evenly-spaced points from the full CSV list.
  all_masses = sorted({f"WR{wr}_N{n}" for wr, n in rows})
  if not all_masses:
    raise SystemExit(0)
  k = min(9, len(all_masses))
  if k == 1:
    chosen = [all_masses[0]]
  else:
    idxs = [round(i * (len(all_masses) - 1) / (k - 1)) for i in range(k)]
    chosen = []
    seen = set()
    for ix in idxs:
      m = all_masses[int(ix)]
      if m in seen:
        continue
      seen.add(m)
      chosen.append(m)
  selected = chosen

for m in selected:
  print(m)
PY
)
  echo "Selected ${#MASS_OPTIONS[@]} signal mass points (default grid):"
  for m in "${MASS_OPTIONS[@]}"; do
    echo "  ${m}"
  done

  for mass in "${MASS_OPTIONS[@]}"; do
    run_signal_analysis "${SELECTED_ERA}" "${mass}"
  done
fi

echo "All analyses complete!"
