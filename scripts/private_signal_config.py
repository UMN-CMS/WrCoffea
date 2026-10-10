#!/usr/bin/env python3
"""Config entries for a private WR signal production stored at an XRootD site.

Lists the production directory over WebDAV (an ``xrdfs ls`` through the
Wisconsin redirector can hang for many minutes, a PROPFIND on port 1094
answers in seconds), opens every NanoAOD file and writes one config entry per
mass point with an explicit ``files`` list, which is how the skim planner
takes samples that are not in DBS.  ``file_sizes``, ``file_events`` and the
site's adler32 (``file_adler32``) give the skim the same transfer checksum and
event-count validation as DBS files.

Per file it checks the event count, the Runs sums (genEventCount,
genEventSumw, LHE/PS sum vectors), genWeight and the per-job
LHEWeight_originalXWGTUP.  Files without events or without the generator
branches are reported and left out of the config.

Cross section: MadGraph gridpacks with ``event_norm = sum`` give every event
of a job XWGTUP = sigma_job / N_job, with N_job the events the job generated.
The merged files keep one Runs row per job (genEventCount = N_job), so
sigma_job = XWGTUP * N_job for every job that processed all its events.  A
job whose event count is not a multiple of ``--job-size-quantum`` stopped
early: its XWGTUP still refers to the requested size, so it is left out of
the cross section (its events stay in the sample and in genEventSumw).  The
value is the event-weighted mean over complete jobs; the uncertainty is the
spread of the job estimates propagated to that mean.

Example (the 2024 production)::

    python3 scripts/private_signal_config.py \\
        --dir root://cmsxrootd.hep.wisc.edu//store/user/kalpana/WRAnalyzer/SignalSamples/Run3/RunIII2024Summer24 \\
        --run Run3 --year 2024 --era RunIII2024Summer24 \\
        --reference data/configs/Run3/2024/RunIII2024Summer24/archived/RunIII2024Summer24_signal_placeholder_Run3Summer23BPix.json \\
        --out data/configs/Run3/2024/RunIII2024Summer24/RunIII2024Summer24_signal.json \\
        --report private_signal_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import uproot

PROXY = os.environ.get("X509_USER_PROXY", os.path.expanduser("~/.x509up_proxy"))
CA_DIR = os.environ.get("X509_CERT_DIR", os.path.expanduser("~/.grid-security/certificates"))

# Kalpana's file names: WRtoNEltoElElJJ_MWR_2000_MN_100_TuneCP5_13p6TeV_cff_merged.root
FILE_RE = re.compile(r"^(?P<proc>WRtoN(?:El|Mu)to(?:El|Mu){2}JJ)_MWR_(?P<wr>\d+)_MN_(?P<n>\d+)_(?P<tune>TuneCP5_13p6TeV)")
# Official names, used as dataset names so every downstream mass-point parser works unchanged.
DATASET_FMT = "{proc}_MWR{wr}_N{n}_{tune}_madgraph-pythia8"
REF_RE = re.compile(r"(WRtoN(?:El|Mu)to(?:El|Mu){2}JJ)_MWR(\d+)_N(\d+)_")
NS = {"D": "DAV:"}


def _sig(v: float, n: int = 5) -> float:
    return float(f"{v:.{n}g}")


def split_root_url(url: str) -> tuple[str, str]:
    """``root://host//store/x`` -> (``host``, ``/store/x``)."""
    m = re.match(r"^root://([^/]+)/(/.*)$", url.rstrip("/"))
    if not m:
        raise ValueError(f"expected root://host//path, got {url!r}")
    return m.group(1), m.group(2)


def propfind(host: str, path: str) -> list[dict]:
    """One WebDAV directory listing (Depth: 1) with the grid proxy."""
    cmd = ["curl", "-sS", "-m", "120", "-L", "--cert", PROXY, "--key", PROXY, "--capath", CA_DIR,
           "-X", "PROPFIND", "-H", "Depth: 1", f"https://{host}:1094{path.rstrip('/')}/"]
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and "multistatus" in r.stdout:
            break
        time.sleep(5)
    else:
        raise RuntimeError(f"PROPFIND {path} failed: {r.returncode} {r.stderr.strip()[:200]}")
    out = []
    for resp in ET.fromstring(r.stdout).findall("D:response", NS):
        href = re.sub("/+", "/", unquote(resp.find("D:href", NS).text)).rstrip("/")
        if href == path.rstrip("/"):
            continue
        prop = resp.find("D:propstat/D:prop", NS)
        out.append({"path": href, "dir": prop.find("D:resourcetype/D:collection", NS) is not None,
                    "size": int(prop.findtext("D:getcontentlength", "0", NS) or 0)})
    return out


def list_root_files(host: str, top: str) -> list[dict]:
    files, todo = [], [top]
    while todo:
        nxt = []
        for entry in (e for d in todo for e in propfind(host, d)):
            if entry["dir"]:
                nxt.append(entry["path"])
            elif entry["path"].endswith(".root"):
                files.append(entry)
        todo = nxt
    return sorted(files, key=lambda f: f["path"])


def adler32_at_site(host: str, path: str) -> str:
    """Server-side adler32 (``xrdfs query checksum``), or "" if the site does not answer."""
    try:
        r = subprocess.run(["xrdfs", host, "query", "checksum", path], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return ""
    m = re.match(r"^adler32\s+([0-9a-f]{8})\s*$", r.stdout.strip())
    return m.group(1) if (r.returncode == 0 and m) else ""


def inspect_file(url: str, quantum: int) -> dict:
    """Event count, Runs sums and per-job cross sections of one merged file."""
    for attempt in range(3):
        try:
            with uproot.open(url, timeout=120) as f:
                ev, runs = f["Events"], f["Runs"]
                rec = {"events": int(ev.num_entries), "branches": len(ev.keys()), "runs_keys": list(runs.keys())}
                needed = {"genWeight", "LHEWeight_originalXWGTUP", "nLHEScaleWeight", "nLHEPdfWeight", "nPSWeight"}
                if rec["events"] == 0 or "genEventSumw" not in runs or not needed <= set(ev.keys()):
                    return rec
                r = runs.arrays(["genEventCount", "genEventSumw"], library="np")
                a = ev.arrays(sorted(needed), library="np")
            break
        except Exception as e:  # network hiccups: retry, then report
            err = e
            time.sleep(5)
    else:
        return {"error": f"{type(err).__name__}: {err}"[:300]}
    n_job = r["genEventCount"].astype(np.int64)
    rec["genEventSumw"] = float(r["genEventSumw"].sum())
    rec["genWeight_sum"] = float(a["genWeight"].sum())
    rec["genEventCount"] = int(n_job.sum())
    for k in ("nLHEScaleWeight", "nLHEPdfWeight", "nPSWeight"):
        v, c = np.unique(a[k], return_counts=True)
        rec[k] = {str(int(x)): int(y) for x, y in zip(v, c)}
    if rec["genEventCount"] != rec["events"]:
        return rec  # Runs rows do not map onto the events: no job split possible
    edges = np.concatenate([[0], np.cumsum(n_job)])
    xw = a["LHEWeight_originalXWGTUP"]
    jobs = []
    for i, n in enumerate(n_job):
        if n == 0:
            continue
        seg = xw[edges[i]:edges[i + 1]]
        jobs.append({"events": int(n), "xwgtup": float(seg[0]), "constant": bool(seg.min() == seg.max())})
    rec["jobs"] = jobs
    cpl = [j for j in jobs if j["events"] % quantum == 0 and j["constant"]]
    if cpl:
        n = np.array([j["events"] for j in cpl], dtype=float)
        s = np.array([j["xwgtup"] * j["events"] for j in cpl])
        xs = float((n * s).sum() / n.sum())
        var = float(((n * (s - xs)) ** 2).sum() / n.sum() ** 2) * len(s) / max(len(s) - 1, 1)
        rec.update(xsec=xs, xsec_unc=float(np.sqrt(var)), jobs_complete=len(cpl), events_complete=int(n.sum()),
                   job_rel_spread=float(s.std(ddof=1) / s.mean()) if len(s) > 1 else 0.0)
    rec["jobs_truncated"] = [j for j in jobs if j not in cpl]
    return rec


def problems_of(rec: dict) -> list[str]:
    if "error" in rec:
        return [rec["error"]]
    if rec["events"] == 0:
        return ["no events"]
    out = []
    if "genEventSumw" not in rec:
        missing = [] if "genEventSumw" in rec["runs_keys"] else ["Runs.genEventSumw"]
        return [f"generator branches missing ({rec['branches']} Events branches; {', '.join(missing) or 'Events weights'})"]
    if rec["genEventCount"] != rec["events"]:
        out.append(f"Runs genEventCount {rec['genEventCount']} != {rec['events']} events")
    if abs(rec["genEventSumw"] - rec["genWeight_sum"]) > 1e-6 * max(1.0, abs(rec["genWeight_sum"])):
        out.append(f"Runs genEventSumw {rec['genEventSumw']} != sum(genWeight) {rec['genWeight_sum']}")
    if "xsec" not in rec:
        out.append("no complete job to take the cross section from")
    return out


def load_reference(path: str | None) -> dict[tuple[str, int, int], float]:
    if not path:
        return {}
    ref = {}
    for meta in json.loads(Path(path).read_text()).values():
        m = REF_RE.search(str(meta.get("dataset", "")))
        if m:
            src = meta.get("xsec_source") or {}
            ref[(m.group(1), int(m.group(2)), int(m.group(3)))] = float(src.get("generator") or meta["xsec"])
    return ref


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", required=True, help="root://host//store/... directory of the production")
    p.add_argument("--run", required=True)
    p.add_argument("--year", required=True)
    p.add_argument("--era", required=True)
    p.add_argument("--out", required=True, help="config JSON to write")
    p.add_argument("--reference", help="config whose generator cross sections the new values are compared to")
    p.add_argument("--report", help="JSON report with the per-file inventory")
    p.add_argument("--note", default="", help="text for every entry's 'note' (production details)")
    p.add_argument("--job-size-quantum", type=int, default=500,
                   help="complete jobs generated a multiple of this many events")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    host, top = split_root_url(args.dir)
    files = list_root_files(host, top)
    print(f"{len(files)} ROOT files under {top} ({sum(f['size'] for f in files) / 1e9:.1f} GB)")

    named, unnamed = [], []
    for f in files:
        m = FILE_RE.match(Path(f["path"]).name)
        (named if m else unnamed).append((f, m))
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(lambda fm: inspect_file(f"root://{host}/{fm[0]['path']}", args.job_size_quantum), named))
        cksums = list(ex.map(lambda fm: adler32_at_site(host, fm[0]["path"]), named))

    ref = load_reference(args.reference)
    today = time.strftime("%Y-%m-%d")
    entries, skipped, report = {}, [], []
    for (f, m), rec, ck in zip(named, recs, cksums):
        url = f"root://{host}/{f['path']}"
        name = DATASET_FMT.format(**m.groupdict())
        probs = problems_of(rec)
        report.append({"file": url, "dataset": name, "size": f["size"], "adler32": ck, "problems": probs, **rec})
        if probs:
            skipped.append((name, url, probs))
            continue
        key = (m["proc"], int(m["wr"]), int(m["n"]))
        source = {"generator": _sig(rec["xsec"]), "generator_unc": _sig(rec["xsec_unc"], 4), "header": "LHE",
                  "factor": 1.0, "factor_reason": "generator value taken as is", "kfactor": 1.0,
                  "reference": None, "reference_source": "", "normalize": "reference",
                  "events_run": rec["events_complete"], "jobs": rec["jobs_complete"],
                  "tool": "LHEWeight_originalXWGTUP x events per job (event_norm=sum), event-weighted over complete jobs",
                  "date": today}
        if key in ref:
            source["compared_to"] = {"generator": ref[key], "config": Path(args.reference).name}
        entry = {"das_name": name, "run": args.run, "year": args.year, "era": args.era, "dataset": name,
                 "physics_group": "Signal", "xsec": _sig(rec["xsec"]), "order": "LO",
                 "total_uncertainty": _sig(rec["xsec_unc"], 4), "genEventSumw": rec["genEventSumw"],
                 "datatype": "mc", "xsec_source": source, "files": [url], "file_sizes": {url: f["size"]},
                 "file_events": {url: rec["events"]}}
        if ck:
            entry["file_adler32"] = {url: ck}
        if args.note:
            entry["note"] = args.note
        entries[name] = entry

    def order(name):
        m = REF_RE.search(name)
        return int(m.group(2)), int(m.group(3)), m.group(1)

    entries = {k: entries[k] for k in sorted(entries, key=order)}
    Path(args.out).write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {len(entries)} entries to {args.out}")

    if unnamed:
        print(f"{len(unnamed)} file(s) with unrecognised names (not in the config):")
        for f, _ in unnamed:
            print(f"  {f['path']}")
    if skipped:
        print(f"{len(skipped)} file(s) left out:")
        for name, url, probs in skipped:
            print(f"  {name}: {'; '.join(probs)}")
    trunc = [(r["dataset"], r["jobs_truncated"]) for r in report if r.get("jobs_truncated")]
    if trunc:
        print(f"{len(trunc)} file(s) with jobs left out of the cross section (stopped early):")
        for name, js in trunc:
            print(f"  {name}: " + ", ".join(f"{j['events']} events" for j in js))
    ratios = [e["xsec"] / e["xsec_source"]["compared_to"]["generator"] for e in entries.values()
              if "compared_to" in e["xsec_source"]]
    if ratios:
        r = np.array(ratios)
        print(f"new / reference cross section over {len(r)} points: mean {r.mean():.4f}, std {r.std():.4f}, "
              f"range [{r.min():.4f}, {r.max():.4f}]")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1) + "\n")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
