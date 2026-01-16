#!/usr/bin/env python3
"""
fill_signal_config_sumw_and_xsdb.py

Given a CMS config JSON like:
  data/configs/Run3/2022/Run3Summer22/Run3Summer22_signal.json

For each dataset entry:
  1) delete "nevts"
  2) add "genEventSumw" by querying DAS for files and summing Runs.genEventSumw
     - uses FNAL redirector permanently: root://cmsxrootd.fnal.gov//
     - per-file open timeout 5s, many retries
  3) add "total_uncertainty" by querying XSDB using "DAS=<process_name>" (process_name = entry["dataset"])
     - HTTP timeout 5s, many retries

Writes an updated JSON (in-place by default).

Requirements (typical on cmslpc):
  - dasgoclient in PATH
  - python packages: uproot, awkward (optional), numpy, requests
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --- Optional imports with clear error messages ---
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("Missing dependency: numpy") from e

try:
    import uproot
except Exception as e:
    raise RuntimeError("Missing dependency: uproot (pip install uproot)") from e

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception as e:
    raise RuntimeError("Missing dependency: requests (pip install requests)") from e


FNAL_REDIRECTOR_PREFIX = "root://cmsxrootd.fnal.gov//"


@dataclass
class RetryCfg:
    max_tries: int
    timeout_s: float
    base_sleep_s: float
    max_sleep_s: float
    jitter_s: float


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def run_cmd(cmd: List[str], timeout_s: Optional[float] = None) -> str:
    """Run a command and return stdout; raise on nonzero exit."""
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
    )
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout


def das_list_files(das_dataset: str, cmd_timeout_s: float = 60.0) -> List[str]:
    """
    Return list of file LFN paths (e.g. /store/...) for a DAS dataset.
    """
    q = f"file dataset={das_dataset}"
    out = run_cmd(["dasgoclient", "-query", q, "-limit", "0"], timeout_s=cmd_timeout_s)
    files = [line.strip() for line in out.splitlines() if line.strip()]
    return files


def to_fnal_root_url(lfn_or_url: str) -> str:
    """
    Force FNAL redirector use.
    Accepts:
      - /store/...
      - root://...//store/...
    Returns:
      - root://cmsxrootd.fnal.gov//store/...
    """
    s = lfn_or_url.strip()
    # If it's already a root URL, try to extract the //store/... suffix.
    m = re.search(r"(//store/.*)$", s)
    if m:
        return FNAL_REDIRECTOR_PREFIX + m.group(1).lstrip("/")  # keep double-slash semantics consistent
    if s.startswith("/store/"):
        return FNAL_REDIRECTOR_PREFIX + s.lstrip("/")
    # last resort: if someone gave store/... without leading slash
    if s.startswith("store/"):
        return FNAL_REDIRECTOR_PREFIX + s
    return s  # leave as-is


def backoff_sleep(attempt: int, cfg: RetryCfg) -> None:
    # Exponential-ish backoff with cap + jitter
    sleep_s = min(cfg.max_sleep_s, cfg.base_sleep_s * (2 ** min(attempt, 10)))
    sleep_s += random.uniform(0.0, cfg.jitter_s)
    time.sleep(sleep_s)


def read_runs_genEventSumw(root_url: str, timeout_s: float) -> float:
    """
    Read Runs.genEventSumw and sum over all entries.
    Notes:
      - uproot forwards options to fsspec; xrootd backends may respect timeout/connect_timeout.
      - We still enforce an overall per-attempt wall timeout by relying on backend timeouts.
    """
    # Options here are "best effort"; different site envs/backends may interpret differently.
    # The key constraint is: we retry aggressively, so occasional hangs should be limited by backend.
    with uproot.open(root_url, timeout=timeout_s) as f:
        if "Runs" not in f:
            raise KeyError("No 'Runs' TTree in file")
        runs = f["Runs"]
        if "genEventSumw" not in runs.keys():
            raise KeyError("No 'genEventSumw' branch in Runs")
        arr = runs["genEventSumw"].array(library="np")
        return float(np.sum(arr))


def compute_genEventSumw_for_dataset(das_dataset: str, retry: RetryCfg, limit_files: Optional[int] = None) -> float:
    files = das_list_files(das_dataset)
    if not files:
        raise RuntimeError(f"No files returned by DAS for dataset: {das_dataset}")

    if limit_files is not None:
        files = files[:limit_files]

    total = 0.0
    n_ok = 0
    n_fail = 0

    for i, lfn in enumerate(files, start=1):
        url = to_fnal_root_url(lfn)
        last_err: Optional[Exception] = None

        for attempt in range(retry.max_tries):
            try:
                val = read_runs_genEventSumw(url, timeout_s=retry.timeout_s)
                total += val
                n_ok += 1
                last_err = None
                break
            except Exception as e:
                last_err = e
                # Try again with backoff
                backoff_sleep(attempt, retry)

        if last_err is not None:
            n_fail += 1
            eprint(f"[WARN] failed to read genEventSumw after {retry.max_tries} tries: {url}\n       last error: {last_err}")

    if n_ok == 0:
        raise RuntimeError(f"All files failed for dataset {das_dataset}; cannot compute genEventSumw.")

    if n_fail > 0:
        eprint(f"[WARN] dataset {das_dataset}: {n_fail}/{len(files)} files failed; genEventSumw is partial.")

    return total


def make_requests_session(max_retries: int) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def parse_total_uncertainty_from_json(obj: Any) -> Optional[float]:
    """
    Attempt to find total_uncertainty in common XSDB-like JSON shapes.
    Returns pb value if found.
    """
    # Common shapes:
    #  - {"data":[{... "total_uncertainty": 0.00005 ...}], ...}
    #  - [{"total_uncertainty": ...}]
    #  - {"results":[{...}]}
    candidates = []
    if isinstance(obj, dict):
        for k in ["data", "results", "items", "rows"]:
            if k in obj and isinstance(obj[k], list):
                candidates.extend(obj[k])
        # sometimes a single record
        candidates.append(obj)
    elif isinstance(obj, list):
        candidates = obj

    for rec in candidates:
        if isinstance(rec, dict) and "total_uncertainty" in rec:
            try:
                return float(rec["total_uncertainty"])
            except Exception:
                continue

    return None


def parse_total_uncertainty_from_html(html: str) -> Optional[float]:
    """
    Fallback: crude regex scraping.
    Looks for something like a table cell containing a float near 'total_uncertainty'.
    """
    # First, try to find a JSON blob embedded in the page (some apps do this).
    m = re.search(r"({.*\"total_uncertainty\".*})", html, flags=re.DOTALL)
    if m:
        blob = m.group(1)
        try:
            return parse_total_uncertainty_from_json(json.loads(blob))
        except Exception:
            pass

    # Otherwise, attempt a simple number capture near the label.
    # This is intentionally permissive.
    m2 = re.search(r"total_uncertainty[^0-9\-]*([0-9]*\.[0-9]+|[0-9]+)", html, flags=re.IGNORECASE)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None


def xsdb_total_uncertainty_pb(
    process_name: str,
    timeout_s: float,
    http_retries: int,
    endpoint_override: Optional[str] = None,
    overall_timeout_s: float = 30.0,
) -> float:
    """
    Query XSDB for total_uncertainty (pb) for DAS=<process_name>.
    Hard-caps total wall time via overall_timeout_s.
    Disables proxy/env influence for stability on batch nodes.
    """
    query_expr = f"DAS={process_name}"

    # Prefer API-style endpoints only. Do NOT hit the "/" UI.
    if endpoint_override:
        candidates = [
            (endpoint_override, {"query": query_expr}),
            (endpoint_override, {"q": query_expr}),
        ]
    else:
        base = "https://xsdb-xsdb-official.app.cern.ch"
        candidates = [
            (f"{base}/api/search", {"query": query_expr}),
            (f"{base}/api/v1/search", {"query": query_expr}),
            (f"{base}/api/search", {"q": query_expr}),
            (f"{base}/api/v1/search", {"q": query_expr}),
        ]

    sess = requests.Session()
    sess.trust_env = False  # IMPORTANT: ignore HTTPS_PROXY etc.

    deadline = time.time() + overall_timeout_s
    last_err: Optional[str] = None

    for url, params in candidates:
        for attempt in range(1, http_retries + 1):
            if time.time() > deadline:
                raise RuntimeError(
                    f"XSDB query exceeded overall timeout ({overall_timeout_s}s) for {process_name}. "
                    f"Last error: {last_err}"
                )
            try:
                eprint(f"[XSDB] GET {url} attempt {attempt}/{http_retries}")
                r = sess.get(url, params=params, timeout=(timeout_s, timeout_s))
                ct = (r.headers.get("content-type") or "").lower()

                # Try JSON first (even if mislabeled)
                obj = None
                try:
                    obj = r.json()
                except Exception:
                    obj = None

                if obj is not None:
                    val = parse_total_uncertainty_from_json(obj)
                    if val is not None:
                        return val

                # Fallback: HTML scrape (rare with API endpoints, but safe)
                val = parse_total_uncertainty_from_html(r.text)
                if val is not None:
                    return val

                last_err = f"no total_uncertainty found at {r.url} (status {r.status_code})"
            except Exception as e:
                last_err = f"request failed at {url}: {e}"

            # small backoff (bounded)
            time.sleep(min(2.0, 0.2 * attempt))

    raise RuntimeError(f"XSDB query failed for {process_name}. Last error: {last_err}")

def update_config(
    in_path: str,
    out_path: str,
    *,
    sumw_retry: RetryCfg,
    xsdb_timeout_s: float,
    xsdb_http_retries: int,
    xsdb_endpoint: Optional[str],
    limit_files: Optional[int],
    overwrite: bool,
    dry_run: bool,
) -> None:
    with open(in_path, "r") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config JSON root must be a dict mapping dataset DAS name -> metadata dict")

    for das_key, meta in cfg.items():
        if not isinstance(meta, dict):
            eprint(f"[WARN] skipping non-dict entry for key: {das_key}")
            continue

        das_name = meta.get("das_name", das_key)
        process_name = meta.get("dataset")  # what you type into XSDB as DAS=<process_name>

        if not process_name:
            eprint(f"[WARN] missing 'dataset' field for {das_key}; skipping XSDB total_uncertainty")
        # 1) delete nevts
        if "nevts" in meta:
            del meta["nevts"]

        # 2) genEventSumw
        if overwrite or ("genEventSumw" not in meta):
            eprint(f"[INFO] computing genEventSumw for: {das_name}")
            if not dry_run:
                meta["genEventSumw"] = compute_genEventSumw_for_dataset(
                    das_dataset=das_name,
                    retry=sumw_retry,
                    limit_files=limit_files,
                )
            else:
                meta["genEventSumw"] = meta.get("genEventSumw", None)

        # 3) total_uncertainty from XSDB
        if process_name and (overwrite or ("total_uncertainty" not in meta)):
            eprint(f"[INFO] querying XSDB total_uncertainty for: {process_name}")
            if not dry_run:
                meta["total_uncertainty"] = xsdb_total_uncertainty_pb(
                    process_name=process_name,
                    timeout_s=xsdb_timeout_s,
                    http_retries=xsdb_http_retries,
                    endpoint_override=xsdb_endpoint,
                )
            else:
                meta["total_uncertainty"] = meta.get("total_uncertainty", None)

    if dry_run:
        eprint("[INFO] dry-run: not writing output")
        return

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp_path, out_path)
    eprint(f"[OK] wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fill genEventSumw (from DAS+ROOT) and total_uncertainty (from XSDB) into a signal config JSON."
    )
    ap.add_argument("config", help="Input config JSON (e.g. Run3Summer22_signal.json)")
    ap.add_argument("-o", "--out", default=None, help="Output path (default: in-place overwrite input)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing genEventSumw/total_uncertainty if present")
    ap.add_argument("--dry-run", action="store_true", help="Do not query; do not write output")

    # genEventSumw / DAS / ROOT reading controls
    ap.add_argument("--sumw-timeout", type=float, default=5.0, help="Per-attempt ROOT open timeout (seconds)")
    ap.add_argument("--sumw-max-tries", type=int, default=30, help="Max retries per file when reading ROOT")
    ap.add_argument("--sumw-base-sleep", type=float, default=0.5, help="Base backoff sleep (seconds)")
    ap.add_argument("--sumw-max-sleep", type=float, default=15.0, help="Max backoff sleep (seconds)")
    ap.add_argument("--sumw-jitter", type=float, default=0.3, help="Random jitter added to sleeps (seconds)")
    ap.add_argument("--limit-files", type=int, default=None, help="If set, only use first N files per dataset (testing)")

    # XSDB controls
    ap.add_argument("--xsdb-timeout", type=float, default=5.0, help="XSDB HTTP timeout (seconds)")
    ap.add_argument("--xsdb-http-retries", type=int, default=10, help="HTTP-layer retries for XSDB queries")
    ap.add_argument("--xsdb-endpoint", default=None, help="Override XSDB endpoint URL (if site API differs)")

    args = ap.parse_args()

    in_path = args.config
    out_path = args.out or in_path

    sumw_retry = RetryCfg(
        max_tries=args.sumw_max_tries,
        timeout_s=args.sumw_timeout,
        base_sleep_s=args.sumw_base_sleep,
        max_sleep_s=args.sumw_max_sleep,
        jitter_s=args.sumw_jitter,
    )

    update_config(
        in_path=in_path,
        out_path=out_path,
        sumw_retry=sumw_retry,
        xsdb_timeout_s=args.xsdb_timeout,
        xsdb_http_retries=args.xsdb_http_retries,
        xsdb_endpoint=args.xsdb_endpoint,
        limit_files=args.limit_files,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
