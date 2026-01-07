#!/bin/bash
set -e

echo "[$(date)] Starting job on ${HOSTNAME}"
echo "Args: $@"

# ---------- Parse arguments ----------
FILE_LIST=""
OUT_FILE=""
MWR=""
MN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file-list)
      FILE_LIST="$2"
      shift 2
      ;;
    --out-file)
      OUT_FILE="$2"
      shift 2
      ;;
    --mwr)
      MWR="$2"
      shift 2
      ;;
    --mn)
      MN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$FILE_LIST" || -z "$OUT_FILE" || -z "$MWR" || -z "$MN" ]]; then
  echo "Missing required arguments"
  exit 1
fi

echo "FILE_LIST = $FILE_LIST"
echo "OUT_FILE  = $OUT_FILE"
echo "MWR       = $MWR"
echo "MN        = $MN"

#echo "PWD: $(pwd)"
#ls

# ---------- Unpack venv ----------
echo "Unpacking venv_wr.tgz..."
tar xzf venv_wr.tgz

PYTHON=./venv_wr/bin/python
echo "Using Python at: $PYTHON"
$PYTHON -V

# ---------- Go to scratch dir ----------
#cd "${_CONDOR_SCRATCH_DIR:-/srv}"
#echo "Now in scratch dir: $(pwd)"
#echo "Contents of scratch dir:"
#ls

# We already have: make_run2_signals.py, wr_signals_eos.txt, etc. here
EOS_LIST="$FILE_LIST"

# ---------- Process one EOS file at a time ----------
i=0
while read -r LFN; do
  [[ -z "$LFN" ]] && continue

  i=$((i+1))
  BASENAME=$(basename "$LFN")

  echo ""
  echo "=== [$i] Processing LFN: ${LFN} ==="
  echo "  xrdcp root://cmseos.fnal.gov/${LFN} ${BASENAME}"
  xrdcp "root://cmseos.fnal.gov/${LFN}" "${BASENAME}"

  # Create single-file list for this call
  echo "${BASENAME}" > single_file_list.txt

  echo "  Running make_run2_signals.py on ${BASENAME}"
  time $PYTHON make_run2_signals.py \
    --file-list single_file_list.txt \
    --out-file "$OUT_FILE" \
    --mwr "$MWR" \
    --mn "$MN"

  echo "  Cleaning up ${BASENAME}"
  rm -f "${BASENAME}" single_file_list.txt
done < "$EOS_LIST"

echo "Finished processing all files in ${EOS_LIST}"

# ---------- Copy output back to EOS ----------
# Allow override to avoid hard-coding usernames/paths.
EOS_OUT_DIR="${EOS_OUT_DIR:-/store/user/${USER}/WR_run2_signals/splits}"
echo "Copying output to EOS: root://cmseos.fnal.gov/${EOS_OUT_DIR}/${OUT_FILE}"
xrdcp -f "$OUT_FILE" "root://cmseos.fnal.gov/${EOS_OUT_DIR}/${OUT_FILE}"

echo "Done. Cleaning up local output:"
ls
rm -f "$OUT_FILE"

echo "[$(date)] Job finished."
