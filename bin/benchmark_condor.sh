#!/bin/bash
# benchmark_condor.sh -- Sweep max-workers x chunksize on Condor for DYJets
# and produce a summary table of execution times.
#
# Usage:
#   bin/benchmark_condor.sh [era] [sample]
#
# Defaults to RunIII2024Summer24 DYJets if no arguments are given.

set -o nounset
set -o pipefail

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

# ── benchmark parameters ─────────────────────────────────────────────
ERA="${1:-RunIII2024Summer24}"
SAMPLE="${2:-DYJets}"
WORKERS=(50 100 150 200)
CHUNKSIZES=(100000 250000 500000)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_FILE="${REPO_ROOT}/benchmark_results_${TIMESTAMP}.log"

echo "Benchmark started at $(date)" | tee "${RESULTS_FILE}"
echo "Era: ${ERA}  Sample: ${SAMPLE}" | tee -a "${RESULTS_FILE}"
echo "Workers: ${WORKERS[*]}" | tee -a "${RESULTS_FILE}"
echo "Chunksizes: ${CHUNKSIZES[*]}" | tee -a "${RESULTS_FILE}"
echo "Total runs: $(( ${#WORKERS[@]} * ${#CHUNKSIZES[@]} ))" | tee -a "${RESULTS_FILE}"
echo "==========================================" | tee -a "${RESULTS_FILE}"

declare -A TIMES

for w in "${WORKERS[@]}"; do
  for chunk in "${CHUNKSIZES[@]}"; do
    LABEL="w${w}_c${chunk}"
    echo "" | tee -a "${RESULTS_FILE}"
    echo ">>> Running: --max-workers ${w} --chunksize ${chunk}" | tee -a "${RESULTS_FILE}"
    echo "    Started at $(date)" | tee -a "${RESULTS_FILE}"

    OUTPUT=$("${PYTHON}" bin/run_analysis.py "${ERA}" "${SAMPLE}" \
        --condor \
        --debug \
        --max-workers "${w}" \
        --chunksize "${chunk}" \
        2>&1)
    EXIT_CODE=$?

    echo "${OUTPUT}" >> "${RESULTS_FILE}"

    if [ ${EXIT_CODE} -ne 0 ]; then
      echo "    FAILED (exit ${EXIT_CODE})" | tee -a "${RESULTS_FILE}"
      TIMES["${LABEL}"]="FAILED"
    else
      EXEC_TIME=$(echo "${OUTPUT}" | grep -oP 'Execution took \K[0-9]+\.[0-9]+')
      if [ -n "${EXEC_TIME}" ]; then
        echo "    Completed in ${EXEC_TIME} minutes" | tee -a "${RESULTS_FILE}"
        TIMES["${LABEL}"]="${EXEC_TIME}"
      else
        echo "    Completed but could not parse time" | tee -a "${RESULTS_FILE}"
        TIMES["${LABEL}"]="N/A"
      fi
    fi
  done
done

# ── summary table ────────────────────────────────────────────────────
echo "" | tee -a "${RESULTS_FILE}"
echo "==========================================" | tee -a "${RESULTS_FILE}"
echo "SUMMARY (time in minutes)" | tee -a "${RESULTS_FILE}"
echo "==========================================" | tee -a "${RESULTS_FILE}"

# Header row
HEADER=$(printf "%-12s" "workers")
for chunk in "${CHUNKSIZES[@]}"; do
  HEADER+=$(printf "  %-12s" "${chunk}")
done
echo "${HEADER}" | tee -a "${RESULTS_FILE}"
echo "----------------------------------------------------" | tee -a "${RESULTS_FILE}"

# Data rows
for w in "${WORKERS[@]}"; do
  ROW=$(printf "%-12s" "${w}")
  for chunk in "${CHUNKSIZES[@]}"; do
    LABEL="w${w}_c${chunk}"
    VAL="${TIMES[${LABEL}]:-N/A}"
    ROW+=$(printf "  %-12s" "${VAL}")
  done
  echo "${ROW}" | tee -a "${RESULTS_FILE}"
done

echo "" | tee -a "${RESULTS_FILE}"
echo "Benchmark finished at $(date)" | tee -a "${RESULTS_FILE}"
echo "Full results saved to: ${RESULTS_FILE}"
