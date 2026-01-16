#!/usr/bin/env python3
"""
Fill genEventSumw for each dataset by summing Runs.genEventSumw across ALL files in DAS.

Per dataset:
  - remove "nevts"
  - ensure "total_uncertainty": 0
  - compute "genEventSumw" by summing Runs.genEventSumw over all files
  - write output JSON

Robustness:
  - per-file timeout
  - retry a failing file N times with exponential backoff
  - if any file still fails after retries: dataset marked INCOMPLETE and genEventSumw left None

Usage:
  python3 fill_runs_genEventSumw_retry.py -i input.json -o output.json --sleep-files 0.02 --per-file-timeout 120 --retries 6 --retry-sleep 2
"""

import argparse
import json
import subprocess
import time
import signal
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import uproot


KEY_ORDER = [
    "das_name",
    "run",
    "year",
    "era",
    "dataset",
    "physics_group",
    "xsec",
    "total_uncertainty",
    "genEventSumw",
    "datatype",
]


class Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise Timeout()


def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr.strip()}")
    return p.stdout


def das_list_files(dataset: str) -> List[str]:
    out = run_cmd(["dasgoclient", "-query", f"file dataset={dataset}", "-limit", "0"])
    return [x.strip() for x in out.splitlines() if x.strip()]


def lfn_to_root_url(lfn: str, redirector: str) -> str:
    return f"{redirector.rstrip('/')}/{lfn}"


def read_runs_genEventSumw(url: str) -> float:
    """Single attempt: open file and sum Runs.genEventSumw."""
    with uproot.open(url) as f:
        if "Runs" not in f:
            raise RuntimeError("No 'Runs' tree in file")
        runs = f["Runs"]
        if "genEventSumw" not in runs.keys():
            raise RuntimeError("No 'genEventSumw' branch in Runs tree")
        arr = runs["genEventSumw"].array(library="np")
        return float(np.sum(arr, dtype=np.float64))


def read_with_retries(
    url: str,
    per_file_timeout_s: int,
    retries: int,
    retry_sleep_s: float,
    backoff: float,
) -> Tuple[float, int]:
    """
    Try to read Runs.genEventSumw with retries.
    Returns (sumw, attempts_used). Raises last exception if all fail.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            signal.alarm(per_file_timeout_s)
            s = read_runs_genEventSumw(url)
            signal.alarm(0)
            return s, attempt
        except Exception as e:
            signal.alarm(0)
            last_err = e
            if attempt < retries:
                sleep_t = retry_sleep_s * (backoff ** (attempt - 1))
                print(f"      retry {attempt}/{retries} failed: {e} -> sleeping {sleep_t:.2f}s")
                time.sleep(sleep_t)
            else:
                break
    raise last_err  # type: ignore


def sum_dataset(
    file_urls: List[str],
    per_file_timeout_s: int,
    sleep_s: float,
    retries: int,
    retry_sleep_s: float,
    backoff: float,
) -> Tuple[float, List[dict]]:
    total = 0.0
    bad: List[dict] = []

    signal.signal(signal.SIGALRM, _alarm_handler)

    n = len(file_urls)
    for i, url in enumerate(file_urls, start=1):
        print(f"    [file {i:4d}/{n}] {url}")
        t0 = time.time()
        try:
            s, used = read_with_retries(
                url=url,
                per_file_timeout_s=per_file_timeout_s,
                retries=retries,
                retry_sleep_s=retry_sleep_s,
                backoff=backoff,
            )
            total += s
            dt = time.time() - t0
            print(f"      sumw(file)={s:.6g}  attempts={used}  dt={dt:.1f}s  running_total={total:.6g}")
        except Exception as e:
            dt = time.time() - t0
            print(f"      [FAILED] after {retries} attempts  dt={dt:.1f}s  err={e}")
            bad.append({"file": url, "reason": str(e), "attempts": retries})

        if sleep_s > 0:
            time.sleep(sleep_s)

    return total, bad


def ordered_entry(entry: Dict[str, Any]) -> OrderedDict:
    o = OrderedDict()
    for k in KEY_ORDER:
        o[k] = entry.get(k)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--infile", required=True)
    ap.add_argument("-o", "--outfile", required=True)
    ap.add_argument("--redirector", default="root://cms-xrd-global.cern.ch")
    ap.add_argument("--per-file-timeout", type=int, default=120)
    ap.add_argument("--sleep-files", type=float, default=0.0)

    ap.add_argument("--retries", type=int, default=6, help="Attempts per file")
    ap.add_argument("--retry-sleep", type=float, default=2.0, help="Initial sleep before retry")
    ap.add_argument("--backoff", type=float, default=1.6, help="Exponential backoff factor")

    ap.add_argument("--report", default="fill_runs_genEventSumw_report.json")
    args = ap.parse_args()

    data: Dict[str, Any] = json.loads(Path(args.infile).read_text())

    out: "OrderedDict[str, OrderedDict]" = OrderedDict()
    report = {"complete": [], "incomplete": [], "errors": []}

    items = list(data.items())
    for idx, (top_key, entry0) in enumerate(items, start=1):
        entry = dict(entry0)
        ds = entry.get("das_name", top_key)

        print(f"\n[{idx}/{len(items)}] {ds}")

        entry.pop("nevts", None)
        entry["total_uncertainty"] = 0

        try:
            lfns = das_list_files(ds)
            if not lfns:
                raise RuntimeError("DAS returned 0 files")

            file_urls = [lfn_to_root_url(lfn, args.redirector) for lfn in lfns]
            print(f"  files: {len(file_urls)}")
            print(f"  first: {file_urls[0]}")

            total, bad = sum_dataset(
                file_urls=file_urls,
                per_file_timeout_s=args.per_file_timeout,
                sleep_s=args.sleep_files,
                retries=args.retries,
                retry_sleep_s=args.retry_sleep,
                backoff=args.backoff,
            )

            if bad:
                entry["genEventSumw"] = None
                report["incomplete"].append({
                    "dataset": ds,
                    "n_files": len(file_urls),
                    "n_bad_files": len(bad),
                    "bad_files": bad,
                    "partial_sumw": total,
                })
                print(f"  [INCOMPLETE] bad_files={len(bad)} -> genEventSumw not written (see report)")
            else:
                entry["genEventSumw"] = float(total)
                report["complete"].append({
                    "dataset": ds,
                    "n_files": len(file_urls),
                    "genEventSumw": float(total),
                })
                print(f"  genEventSumw(dataset) = {entry['genEventSumw']:.12g}")

        except Exception as e:
            entry["genEventSumw"] = None
            report["errors"].append({"dataset": ds, "error": str(e)})
            print(f"  [ERROR] {e}")

        out[ds] = ordered_entry(entry)

    Path(args.outfile).write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"\n[i] Wrote updated JSON: {args.outfile}")
    print(f"[i] Wrote report:       {args.report}")
    print(f"[i] Complete: {len(report['complete'])} | Incomplete: {len(report['incomplete'])} | Errors: {len(report['errors'])}")


if __name__ == "__main__":
    main()
