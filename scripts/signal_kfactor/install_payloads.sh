#!/bin/bash
# Merge the computed <k> averages into the campaign payloads and install them under data/reweights.
#   install_payloads.sh <campaign dir, e.g. ~/work/skims_v2/_kfactor> <date tag, e.g. 20261010>
# Writes
#   data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_13tev_<tag>.json   (UL18 official + private grid; the
#                                                                                   other UL eras map to it too)
#   data/reweights/Run3/signal_kfactor_13p6tev_<tag>.json                         (2022/2023 official + 2024 private)
# and prints the config.yaml lines; it does not touch config.yaml (other sessions' runs read it at start).
set -e
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
K=$(readlink -f ${1:-$HOME/work/skims_v2/_kfactor}); TAG=${2:-$(date +%Y%m%d)}
PY=$REPO/.venv/bin/python
AVG=$REPO/scripts/signal_kfactor/average_k.py
cd $K
# 13 TeV: the official samples, then the private scan models, into one payload
$PY $AVG --merge averages_UL18_official.json --payload payload_13.json --write-payload payload_13_merged.json --source ul18_official
$PY $AVG --merge averages_UL18_scans.json --payload payload_13_merged.json --write-payload payload_13_merged.json --source ul18_private_scan
# 13.6 TeV: the 2022 official samples (same points in 2022EE/2023/2023BPix) and the 2024 private production
$PY $AVG --merge averages_Run3Summer22.json --payload payload_13p6.json --write-payload payload_13p6_merged.json --source run3summer22_official
$PY $AVG --merge averages_RunIII2024Summer24.json --payload payload_13p6_merged.json --write-payload payload_13p6_merged.json --source runiii2024_private
$PY - "$K" "$REPO" "$TAG" <<'EOF'
import json, sys
from pathlib import Path
K, REPO, TAG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
for src, dst in (("payload_13_merged.json", f"data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_13tev_{TAG}.json"),
                 ("payload_13p6_merged.json", f"data/reweights/Run3/signal_kfactor_13p6tev_{TAG}.json")):
    p = json.loads((K / src).read_text())
    pts = set(p["direct_k"]) | set(p["point_curves"])
    missing_avg = sorted(x for x in pts if x not in p["averages"])
    bad = sorted(x for x, a in p["averages"].items() if a.get("ee") is None and a.get("mm") is None)
    p["provenance"]["installed"] = f"{TAG}: {len(p['direct_k'])} direct_k, {len(p['point_curves'])} point_curves, {len(p['averages'])} averages"
    p["provenance"]["note"] = ("direct_k: the point's c x sigma_NLO / sigma_LO(official card) (config normalisation, "
                               "genxsec_campaign.py --signal-norm direct); point_curves: its own NLO/LO m_lN ratio (the analyzer's "
                               "k(m_lN)); averages: <k> of each sample and channel over its generated events (average_k.py); "
                               "kfactor: the shared curve of all points (fallback). docs/studies/signal_kfactor.md")
    out = REPO / dst
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    print(f"{dst}: {p['provenance']['installed']}; points without an average: {len(missing_avg)}"
          f"{' e.g. ' + ', '.join(missing_avg[:5]) if missing_avg else ''}; empty averages: {bad[:5]}")
print("\nconfig.yaml signal_kfactor:")
for era in ("RunIISummer20UL18", "RunIISummer20UL17", "RunIISummer20UL16", "RunIISummer20UL16APV"):   # same 13 TeV gridpacks
    print(f"  {era}: data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_13tev_{TAG}.json")
for era in ("Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix", "RunIII2024Summer24"):
    print(f"  {era}: data/reweights/Run3/signal_kfactor_13p6tev_{TAG}.json")
EOF
