#!/usr/bin/env python3
"""
Verify genEventSumw in a JSON by recomputing it from DAS files:
- For each dataset entry (keyed by DAS dataset name):
  1) query DAS for files in that dataset
  2) read each file over xrootd with uproot
  3) sum Events.genWeight across the whole dataset
  4) compare to JSON's genEventSumw
  5) report mismatches

Notes:
- Uses the "multiple TTrees" safe form for uproot.iterate: {file_url: "Events"}.
- Default redirector: root://cms-xrd-global.cern.ch
- You need: dasgoclient, uproot, numpy, a valid VOMS proxy.

Example:
  ./check_genEventSumw.py -i dy_2024.json --rel-tol 1e-6 --abs-tol 1e-3 --step-size "200 MB"
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import uproot


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr.strip()}")
    return p.stdout


def das_list_files(dataset: str) -> List[str]:
    out = run_cmd(["dasgoclient", "-query", f"file dataset={dataset}", "-limit", "0"])
    return [x.strip() for x in out.splitlines() if x.strip()]


def lfn_to_root_url(lfn: str, redirector: str) -> str:
    # lfn like /store/...
    return f"{redirector.rstrip('/')}/{lfn}"


def sum_genweight(file_urls: List[str], step_size: str) -> float:
    total = 0.0
    filespec = {u: "Events" for u in file_urls}  # IMPORTANT: files have multiple TTrees

    count = 0
    for arrays in uproot.iterate(filespec, ["genWeight"], step_size=step_size, library="np"):
        total += float(np.sum(arrays["genWeight"], dtype=np.float64))
        count += 1
        print(count)

    return total


def is_close(a: float, b: float, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--infile", required=True, help="Input JSON file (your DY config)")
    ap.add_argument("--redirector", default="root://cms-xrd-global.cern.ch", help="XRootD redirector")
    ap.add_argument("--step-size", default="200 MB", help='uproot.iterate step_size, e.g. "200 MB"')
    ap.add_argument("--rel-tol", type=float, default=1e-6, help="Relative tolerance for mismatch flag")
    ap.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute tolerance for mismatch flag")
    ap.add_argument("--max-datasets", type=int, default=0, help="If >0, only process first N datasets")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between datasets")
    ap.add_argument("--report", default="genEventSumw_check_report.json", help="Output report JSON")
    args = ap.parse_args()

    data: Dict[str, Any] = json.loads(Path(args.infile).read_text())
    items = list(data.items())
    if args.max_datasets and args.max_datasets > 0:
        items = items[: args.max_datasets]

    report = {
        "checked": [],
        "mismatches": [],
        "errors": [],
        "tolerances": {"rel_tol": args.rel_tol, "abs_tol": args.abs_tol},
        "redirector": args.redirector,
        "step_size": args.step_size,
    }

    print(f"[i] Datasets to check: {len(items)}")

    for idx, (das_key, entry) in enumerate(items, start=1):
        ds = entry.get("das_name", das_key)
        json_sumw = float(entry.get("genEventSumw", 0.0))

        print(f"\n[{idx}/{len(items)}] {ds}")
        try:
            lfns = das_list_files(ds)
            if not lfns:
                raise RuntimeError("No files returned by DAS")

            file_urls = [lfn_to_root_url(lfn, args.redirector) for lfn in lfns]
            print(f"  files: {len(file_urls)}")
            print(f"  first: {file_urls[0]}")

            calc_sumw = sum_genweight(file_urls, step_size=args.step_size)

            ok = is_close(calc_sumw, json_sumw, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
            diff = calc_sumw - json_sumw
            frac = (diff / json_sumw) if json_sumw != 0 else float("inf")

            line = {
                "dataset": ds,
                "json_genEventSumw": json_sumw,
                "calc_genEventSumw": calc_sumw,
                "diff": diff,
                "frac_diff": frac,
                "n_files": len(file_urls),
                "status": "OK" if ok else "MISMATCH",
            }
            report["checked"].append(line)

            if ok:
                print(f"  OK: calc={calc_sumw:.6g}  json={json_sumw:.6g}  diff={diff:.6g}")
            else:
                print(f"  MISMATCH!")
                print(f"    calc={calc_sumw:.12g}")
                print(f"    json ={json_sumw:.12g}")
                print(f"    diff ={diff:.12g}  frac={frac:.6g}")
                report["mismatches"].append(line)

        except Exception as e:
            print(f"  [ERROR] {e}")
            report["errors"].append({"dataset": ds, "error": str(e)})

        if args.sleep > 0:
            import time
            time.sleep(args.sleep)

    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n[i] Report written: {args.report}")
    print(f"[i] Mismatches: {len(report['mismatches'])} | Errors: {len(report['errors'])} | Checked: {len(report['checked'])}")


if __name__ == "__main__":
    main()
