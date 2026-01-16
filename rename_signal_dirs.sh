#!/usr/bin/env bash
set -euo pipefail

BASE='davs://cmsxrootd.hep.wisc.edu:1094/store/user/wijackso/WRAnalyzer/skims/Run3/2024/RunIII2024Summer24/Signals'
SLEEP_SEC=1   # change to 2, 5, etc. if you want it gentler

# List only directories, extract the name (last column), then rename:
#   WR...pythia8__Run3Summer...__NANOAODSIM  ->  WR...pythia8
gfal-ls -l "$BASE" \
  | awk '$1 ~ /^d/ {print $NF}' \
  | while read -r old; do
      [[ -z "$old" ]] && continue

      new="${old%%__*}"   # keep only the part before the first "__"

      # Skip if already in desired form
      if [[ "$old" == "$new" ]]; then
        echo "[skip] already clean: $old"
        continue
      fi

      src="${BASE}/${old}"
      dst="${BASE}/${new}"

      # Safety: don't overwrite an existing target
      if gfal-stat "$dst" >/dev/null 2>&1; then
        echo "[WARN] target exists, skipping: $dst"
        continue
      fi

      echo "[rename] $old  ->  $new"
      gfal-rename "$src" "$dst"
      sleep "$SLEEP_SEC"
    done
