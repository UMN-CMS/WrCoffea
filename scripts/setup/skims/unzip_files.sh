#!/bin/bash

source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# Ensure an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

DATASET_NAME="$1"

echo "-------------------------------------------"
echo "Processing tarballs in: $(pwd)"
echo "-------------------------------------------"

shopt -s nullglob
# Extract only tarballs that match this dataset
TARS=( "${DATASET_NAME}"_*.tar.gz )
if (( ${#TARS[@]} == 0 )); then
    echo "[warn] No tarballs found for '${DATASET_NAME}_*.tar.gz' in $(pwd)"
fi
for TAR_FILE in "${TARS[@]}"; do
    echo "Extracting: $TAR_FILE"
    tar -xzf "$TAR_FILE"
    rm -f "$TAR_FILE"
done
shopt -u nullglob

# If the dataset directory doesn't exist, bail out.
if [[ ! -d "$DATASET_NAME" ]]; then
    echo "[error] Expected directory '$DATASET_NAME' after extraction, but it was not found."
    exit 2
fi

# For EGamma or Muon datasets: skip merging entirely.
if [[ "$DATASET_NAME" == *EGamma* || "$DATASET_NAME" == *Muon* ]]; then
    echo "-------------------------------------------"
    echo "Dataset '$DATASET_NAME' matches EGamma/Muon. Skipping hadd; just cleaning up."
    echo "-------------------------------------------"

    # Remove only tarballs for this dataset
    # --- Step 1: Remove everything except the dataset directory
    find . -mindepth 1 -maxdepth 1 ! -name "$DATASET_NAME" -exec rm -rf {} +

    # Move extracted files up one level and remove the directory
    shopt -s nullglob
    FILES_TO_MOVE=( "${DATASET_NAME}/"* )
    if (( ${#FILES_TO_MOVE[@]} > 0 )); then
        mv -v "${FILES_TO_MOVE[@]}" .
    fi
    shopt -u nullglob
    rmdir "$DATASET_NAME" 2>/dev/null || true

    echo "-------------------------------------------"
    echo "All tarballs extracted and cleaned up (no merging performed)."
    echo "-------------------------------------------"
    exit 0
fi

# Otherwise: proceed with counting and merging
cd "$DATASET_NAME" || { echo "[error] cd failed"; exit 3; }

echo "-------------------------------------------"
echo "Counting events and merging files..."
echo "-------------------------------------------"

TOTAL_EVENTS=0
MERGED_FILE_COUNT=0
EVENT_THRESHOLD=${EVENT_THRESHOLD:-1000000}   # Max events per merged file; override with env var
CURRENT_EVENT_COUNT=0
MERGE_LIST=()

# Gather ROOT files
mapfile -t ROOT_FILES < <(ls *.root 2>/dev/null | sort)
if (( ${#ROOT_FILES[@]} == 0 )); then
    echo "[warn] No .root files found in $(pwd)"
    cd ..
    exit 0
fi

TOTAL_FILES=${#ROOT_FILES[@]}
FILE_IDX=0

# Helper: merge and cleanup current MERGE_LIST (fail on bad hadd warnings)
merge_flush() {
    if (( ${#MERGE_LIST[@]} == 0 )); then
        return 0
    fi

    local MERGED_FILE="${DATASET_NAME}_skim${MERGED_FILE_COUNT}.root"
    local LOG
    LOG="$(mktemp -t hadd_log.XXXXXX)"

    # Treat this warning as fatal; extend pattern if needed
    local BAD_WARN_PATTERN='Warning in <TTree::CopyEntries>.*not present in the import TTree'

    echo "Merging ${#MERGE_LIST[@]} files into: $MERGED_FILE (events ~${CURRENT_EVENT_COUNT})"

    # Run hadd and capture ALL output
    hadd -f "$MERGED_FILE" "${MERGE_LIST[@]}" >"$LOG" 2>&1
    local HADD_STATUS=$?

    if [[ $HADD_STATUS -ne 0 ]]; then
        echo "[ERROR] hadd failed (exit $HADD_STATUS). Keeping source files. See log: $LOG"
        tail -n 50 "$LOG"
        [[ -f "$MERGED_FILE" ]] && rm -f "$MERGED_FILE"
        return 1
    fi

    # Treat specific warnings as fatal
    if grep -E -q "$BAD_WARN_PATTERN" "$LOG"; then
        echo "[ERROR] Detected fatal warning from hadd:"
        grep -E "$BAD_WARN_PATTERN" "$LOG" | sed 's/^/  /'
        echo "Refusing to keep merged file and refusing to delete source files."
        rm -f "$MERGED_FILE"
        echo "Full log at: $LOG"
        return 2
    fi

    # If we got here, the merge is clean enough. Proceed to delete sources.
    echo "Deleting merged source files..."
    rm -f "${MERGE_LIST[@]}"

    # Reset state only on success
    MERGE_LIST=()
    CURRENT_EVENT_COUNT=0
    ((MERGED_FILE_COUNT++))

    # Optional: remove log (or keep for auditing)
    rm -f "$LOG"
    return 0
}

for ROOT_FILE in "${ROOT_FILES[@]}"; do
    ((FILE_IDX++))
    if [[ ! -f "$ROOT_FILE" ]]; then
        continue
    fi

    # Count events using ROOT directly (quiet)
    EVENTS=$(root -l -b -q -e \
        "TFile f(\"$ROOT_FILE\"); TTree *t=(TTree*)f.Get(\"Events\"); if(t){cout<<t->GetEntries()<<endl;} else {cout<<0<<endl;}" \
        2>/dev/null)
    # Fallback if ROOT returned nothing
    if [[ -z "$EVENTS" ]]; then
        EVENTS=0
    fi

    # If adding this file would exceed threshold, flush current batch first
    if (( CURRENT_EVENT_COUNT > 0 && CURRENT_EVENT_COUNT + EVENTS > EVENT_THRESHOLD )); then
        if ! merge_flush; then
            echo "[abort] Merge failed; stopping further processing to avoid data loss."
            echo "-------------------------------------------"
            echo "Total events processed so far: $TOTAL_EVENTS"
            echo "Merged into $MERGED_FILE_COUNT files"
            echo "Some merges may have failed; check logs above."
            echo "-------------------------------------------"
            cd ..
            exit 4
        fi
    fi

    # Add file to current batch
    MERGE_LIST+=("$ROOT_FILE")
    CURRENT_EVENT_COUNT=$((CURRENT_EVENT_COUNT + EVENTS))
    TOTAL_EVENTS=$((TOTAL_EVENTS + EVENTS))
done

# Merge leftovers
if ! merge_flush; then
    echo "[abort] Final merge failed; keeping original files. Check logs."
    cd ..
    exit 5
fi

echo "-------------------------------------------"
echo "Total events processed: $TOTAL_EVENTS"
echo "Merged into $MERGED_FILE_COUNT files"
echo "Original files deleted."
echo "-------------------------------------------"

cd ..

# Remove only tarballs for this dataset
# --- Step 1: Remove everything except the dataset directory
find . -mindepth 1 -maxdepth 1 ! -name "$DATASET_NAME" -exec rm -rf {} +

# --- Step 2: Move all files from inside the dataset directory up one level
if [[ -d "$DATASET_NAME" ]]; then
    shopt -s nullglob
    FILES_TO_MOVE=( "${DATASET_NAME}/"* )
    if (( ${#FILES_TO_MOVE[@]} > 0 )); then
        mv -v "${FILES_TO_MOVE[@]}" .
    fi
    shopt -u nullglob
    # --- Step 3: Remove now-empty dataset directory
    rmdir "$DATASET_NAME" 2>/dev/null || true
fi

echo "-------------------------------------------"
echo "All tarballs extracted and cleaned up."
echo "-------------------------------------------"
