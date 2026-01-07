#!/bin/bash
# Merge *_skim.root files sequentially, splitting outputs so each merged file
# contains roughly MAX_EVENTS events.  Uses uproot (preferred) or ROOT fallback.
# Verifies that total event counts before and after merging are consistent.

# --- Environment -------------------------------------------------------------
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# ---------------------------------------------------------------------------

set -euo pipefail

MAX_EVENTS=1000000          # target max events per output file
OUT_SUFFIX="_part"          # output naming pattern
TREE_NAME="Events"          # default tree name (override with --tree)

usage() {
  echo "Usage: $0 [--tree TREE_NAME] <dataset_dir>"
  exit 1
}

# --- Parse args --------------------------------------------------------------
if [[ $# -lt 1 ]]; then usage; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree) TREE_NAME="${2:-Events}"; shift 2;;
    -h|--help) usage;;
    *) DATASET_DIR="$1"; shift;;
  esac
done

[[ -d "${DATASET_DIR:-}" ]] || { echo "❌ Directory not found: ${DATASET_DIR:-<nil>}"; exit 1; }
cd "$DATASET_DIR"

DATASET_NAME="$(basename "$PWD")"
mapfile -t FILES < <(ls -1 *_skim.root 2>/dev/null || true)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "❌ No *_skim.root files in $PWD"
  exit 1
fi

echo "🔍 Found ${#FILES[@]} input ROOT files."
echo "🌲 Using tree name: '${TREE_NAME}' (auto-detect fallback enabled)"
echo "📦 Splitting outputs at ~${MAX_EVENTS} events per file"
echo

# --- Helpers -----------------------------------------------------------------
has_uproot() {
  python3 - <<'PY' >/dev/null 2>&1
import importlib.util
assert importlib.util.find_spec("uproot") is not None
PY
}

get_nevents_uproot() {
  local f="$1" tname="$2"
  python3 - "$f" "$tname" <<'PY'
import sys, uproot
path, pref = sys.argv[1], sys.argv[2]
try:
    with uproot.open(path) as f:
        tree = None
        if pref in f:
            from uproot.behaviors.TTree import TTree
            obj = f[pref]
            if isinstance(obj, TTree): tree = obj
        if tree is None:
            for k, cls in f.classnames().items():
                if cls.startswith("TTree"):
                    tree = f[k]; break
        print(tree.num_entries if tree is not None else 0)
except Exception:
    print(0)
PY
}

get_nevents_root() {
  local f="$1" tname="$2"
  root -l -b -q <<'RCMD' 2>/dev/null | tail -n1
{
  gErrorIgnoreLevel = kError;
  TString path = "$f";
  TString pref = "$tname";
  Long64_t n = 0;
  TFile *file = TFile::Open(path,"READ");
  if (file && !file->IsZombie()) {
    TTree *tr = nullptr;
    if (!pref.IsNull()) {
      TObject *obj = file->Get(pref);
      if (obj && obj->InheritsFrom(TTree::Class())) tr=(TTree*)obj;
    }
    if (!tr) {
      TIter next(file->GetListOfKeys());
      while (TObject *ko=next()) {
        TObject *obj=file->Get(ko->GetName());
        if (obj && obj->InheritsFrom(TTree::Class())) { tr=(TTree*)obj; break; }
      }
    }
    if (tr) n=tr->GetEntries();
  }
  if (file) file->Close();
  std::cout<<n<<std::endl;
}
RCMD
}

get_nevents() {
  local f="$1"
  if has_uproot; then
    get_nevents_uproot "$f" "$TREE_NAME"
  else
    get_nevents_root "$f" "$TREE_NAME"
  fi
}

# --- Merge loop --------------------------------------------------------------
part=1
cur_events=0
cur_list=()
total_in_events=0

flush_part() {
  local outfile="${DATASET_NAME}${OUT_SUFFIX}${part}.root"
  if [[ ${#cur_list[@]} -eq 0 ]]; then return; fi
  echo "🧩 Merging ${#cur_list[@]} files into ${outfile} (≈ ${cur_events} events)"
  hadd -f "${outfile}" "${cur_list[@]}"
  echo "✅ Wrote $(ls -lh "${outfile}" | awk '{print $5}')"
  part=$((part+1))
  cur_events=0
  cur_list=()
}

for f in "${FILES[@]}"; do
  nev="$(get_nevents "$f" || echo 0)"
  [[ "$nev" =~ ^[0-9]+$ ]] || nev=0
  printf "📄 %-60s : %s events\n" "$f" "$nev"
  total_in_events=$((total_in_events + nev))

  if (( cur_events + nev > MAX_EVENTS )) && (( ${#cur_list[@]} > 0 )); then
    flush_part
  fi
  cur_list+=("$f")
  cur_events=$((cur_events + nev))
done

flush_part

echo
echo "🎉 Merge complete."
echo "🔢 Total events (input files): ${total_in_events}"

# --- Post-merge verification -------------------------------------------------
echo
echo "🔍 Verifying merged outputs..."
mapfile -t MERGED < <(ls -1 "${DATASET_NAME}"${OUT_SUFFIX}*.root 2>/dev/null || true)

total_out_events=0
for f in "${MERGED[@]}"; do
  nev="$(get_nevents "$f" || echo 0)"
  [[ "$nev" =~ ^[0-9]+$ ]] || nev=0
  printf "📦 %-60s : %s events\n" "$f" "$nev"
  total_out_events=$((total_out_events + nev))
done

echo
echo "📊 Input total : ${total_in_events}"
echo "📊 Output total: ${total_out_events}"

if (( total_in_events == total_out_events )); then
  echo "✅ Event counts match perfectly!"
  echo
  echo "🧹 Removing original *_skim.root files..."
  rm -f *_skim.root
  echo "✅ Cleanup complete."
else
  echo "⚠️  Mismatch detected! Original files have been preserved."
  diff=$((total_out_events - total_in_events))
  echo "   Difference: ${diff} events (output - input)"
fi

echo
echo "📂 Generated $(ls -1 "${DATASET_NAME}"${OUT_SUFFIX}*.root | wc -l) merged file(s):"
ls -lh "${DATASET_NAME}"${OUT_SUFFIX}*.root
