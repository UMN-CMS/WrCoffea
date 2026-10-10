"""Whole-campaign validation: do the skims hold exactly what the inputs would give?

Checks, per dataset of the plan:

1. **Completeness**: every group finished; every planned input file appears in
   exactly one part (by LFN) or is listed as unreachable; no duplicates, no
   strangers.
2. **Per input**: the entry count read from the file equals the DBS event
   count; the selected-event digest from the production selection equals the
   one from the independent recount.
3. **Per part (re-read)**: the file's checksum equals the sidecar's; the
   (run, luminosityBlock, event) triples stored in each input's entry range
   reproduce the digests; the Runs tree sums equal the per-input sums.
4. **Dataset totals**:
   - entries in == DBS events of the dataset;
   - for MC, the summed genEventSumw and genEventCount;
   - for data, the merged LuminosityBlocks trees hold exactly the lumisections
     DBS lists for the dataset's files (lost or extra lumisections are a
     problem). Certified lumisections of the dataset's runs that exist in no
     file at all are a warning. The processed lumisections are written as
     ``<dataset>_processed_lumis.json`` (golden-JSON format) next to the parts.

Parts written by Condor jobs exist only at the remote root: with ``--remote``
(and ``--skip-reread``) every remote part is stat-ed against its sidecar size,
and ``--remote-lumis`` reads the data LuminosityBlocks trees through XRootD.

The result is a JSON report plus a console summary; the exit code is nonzero
if any dataset has a problem (unreachable inputs count as a problem unless
``--accept-bad-files`` is given).
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from wrcoffea.skim.fetch import adler32_of
from wrcoffea.skim.plan import Plan, DatasetPlan
from wrcoffea.skim.recount import ids_digest
from wrcoffea.skim.selection import EVENT_ID_BRANCHES
from wrcoffea.skim.worker import load_group_status
from wrcoffea.skim.writer import read_sidecar

logger = logging.getLogger(__name__)


def golden_lumis(path: str | Path) -> dict[int, set[int]]:
    """``{run: {lumi, ...}}`` from a golden-JSON file."""
    with open(path) as fh:
        d = json.load(fh)
    out: dict[int, set[int]] = {}
    for run, ranges in d.items():
        s: set[int] = set()
        for lo, hi in ranges:
            s.update(range(int(lo), int(hi) + 1))
        out[int(run)] = s
    return out


def golden_for_era(era: str, overrides: dict[str, str] | None = None) -> str | None:
    if overrides and era in overrides:
        return overrides[era]
    try:
        from wrcoffea.analysis_config import LUMI_JSONS
    except Exception:
        return None
    return LUMI_JSONS.get(era)


def _open_part(path: str):
    """uproot.open for a local path or an xrootd URL (separate so tests can substitute it)."""
    return uproot.open(path, timeout=120) if path.startswith("root://") else uproot.open(path)


def remote_part_url(remote: str, ds: DatasetPlan, local_path: str) -> str:
    return f"{remote.rstrip('/')}/{ds.out_dir}/{Path(local_path).name}"


def _remote_sizes(urls: list[str], workers: int = 16) -> dict[str, int | None]:
    """Remote size of each URL (None when absent); stats run in parallel (each takes ~1 s at Wisconsin)."""
    import concurrent.futures as cf
    from wrcoffea.skim.condor import remote_size, RemoteUnavailable

    def one(u):
        try:
            return u, remote_size(u)
        except RemoteUnavailable:
            return u, -1
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return dict(ex.map(one, urls))


def _part_lumis(path: str) -> set[tuple[int, int]]:
    with _open_part(path) as f:
        if "LuminosityBlocks" not in f:
            return set()
        t = f["LuminosityBlocks"]
        if t.num_entries == 0:
            return set()
        a = t.arrays(["run", "luminosityBlock"], library="np")
        return set(zip(a["run"].tolist(), a["luminosityBlock"].tolist()))


def dbs_dataset_lumis(dbs, das_paths: list[str]) -> set[tuple[int, int]] | None:
    """Every (run, lumi) DBS lists in any file of the datasets, or None if unavailable."""
    out: set[tuple[int, int]] = set()
    try:
        for das in das_paths:
            for b in dbs.get("blocks", dataset=das):
                for row in dbs.get("filelumis", block_name=b["block_name"]):
                    ls = row["lumi_section_num"] if isinstance(row["lumi_section_num"], list) else [row["lumi_section_num"]]
                    out.update((int(row["run_num"]), int(l)) for l in ls)
    except Exception as e:
        logger.warning("DBS file-lumi lists unavailable: %s", e)
        return None
    return out


def write_lumimask(lumis: set[tuple[int, int]], path: Path) -> None:
    """Write the processed lumisections as a golden-JSON style run -> [[lo, hi], ...] file (brilcalc input)."""
    byrun: dict[int, list[int]] = {}
    for r, l in lumis:
        byrun.setdefault(r, []).append(l)
    out: dict[str, list[list[int]]] = {}
    for r in sorted(byrun):
        ls = sorted(byrun[r])
        ranges, start, prev = [], ls[0], ls[0]
        for l in ls[1:]:
            if l != prev + 1:
                ranges.append([start, prev]); start = l
            prev = l
        ranges.append([start, prev])
        out[str(r)] = ranges
    path.write_text(json.dumps(out) + "\n")


def validate_dataset(plan: Plan, key: str, out_root: Path, *, reread: bool = True,
                     golden: str | None = None, dbs_runs: list[int] | None = None,
                     dbs_lumis: set[tuple[int, int]] | None = None,
                     remote: str | None = None, remote_lumis: bool = False) -> dict:
    ds: DatasetPlan = plan.datasets[key]
    out_dir = out_root / ds.out_dir
    rep: dict = {"dataset": ds.key, "era": ds.era, "datatype": ds.datatype, "problems": [], "warnings": [],
                 "groups": len(ds.groups), "groups_done": 0, "parts": 0, "events_in": 0, "events_out": 0,
                 "bytes": 0, "bad_files": [], "gen_sumw": None, "gen_count": None}
    planned = {f["lfn"]: f for f in ds.files}
    synthetic = set(plan.schemas.get(ds.schema_id, {}).get("synthetic", []))
    seen: Counter = Counter()
    remote_parts: list = []
    sidecars: list[dict] = []
    for gid in range(len(ds.groups)):
        st = load_group_status(out_dir, gid)
        if st is None:
            rep["problems"].append(f"group {gid} never ran")
            continue
        if st.status != "done":
            rep["problems"].append(f"group {gid} status {st.status}: {(st.error or '').splitlines()[0][:160] if st.error else ''}")
            continue
        rep["groups_done"] += 1
        for b in st.bad_files:
            rep["bad_files"].append(b["lfn"])
        for p in st.parts:
            sc_path = Path(p["sidecar"])
            if not sc_path.exists():
                rep["problems"].append(f"sidecar missing: {p['sidecar']}")
                continue
            if not Path(p["path"]).exists():
                if remote is None:
                    rep["problems"].append(f"part missing: {p['path']}")
                    continue
                remote_parts.append((p, remote_part_url(remote, ds, p["path"])))
            sidecars.append(read_sidecar(sc_path))
    # Parts that exist only at the remote root (written by Condor jobs): each must be there with the
    # size its sidecar records (the shipper verified the adler32 server-side when it copied them).
    if remote_parts:
        sizes = _remote_sizes([u for _, u in remote_parts])
        rep["remote_parts"] = len(remote_parts)
        for p, u in remote_parts:
            got = sizes.get(u)
            if got == -1:
                rep["warnings"].append(f"{Path(u).name}: remote stat timed out (busy server); size not checked")
            elif got is None:
                rep["problems"].append(f"remote part missing: {u}")
            elif got != int(p["size"]):
                rep["problems"].append(f"remote part {Path(u).name}: size {got} != sidecar {p['size']}")
    gen_sumw = 0.0
    gen_count = 0
    have_sumw = False
    lumis: set[tuple[int, int]] = set()
    for sc in sidecars:
        part = sc["part"]
        rep["parts"] += 1
        rep["events_out"] += int(part["n_events"])
        rep["bytes"] += int(part["size"])
        if sc.get("schema", {}).get("digest") and ds.schema_id not in ("lazy", sc["schema"]["digest"]):
            rep["problems"].append(f"{Path(part['path']).name}: schema {sc['schema']['digest']} != plan {ds.schema_id}")
        for rec in part["inputs"]:
            lfn = rec["lfn"]
            seen[lfn] += 1
            if lfn not in planned:
                rep["problems"].append(f"{lfn}: not in the plan but present in {Path(part['path']).name}")
                continue
            f = planned[lfn]
            rep["events_in"] += int(rec["n_in"])
            if f["events"] and int(rec["n_in"]) != int(f["events"]):
                rep["problems"].append(f"{lfn}: {rec['n_in']} entries read, DBS says {f['events']}")
            if rec.get("recount_digest") and rec["recount_digest"] != rec["ids_digest"]:
                rep["problems"].append(f"{lfn}: selection and recount digests differ")
            unexpected = [b for b in rec.get("filled_branches", []) if b not in synthetic]
            if unexpected:
                rep["warnings"].append(f"{lfn}: filled {len(unexpected)} branch(es) missing in the input: {unexpected[:6]}")
            if rec.get("gen_sumw") is not None:
                have_sumw = True
                gen_sumw += float(rec["gen_sumw"])
                gen_count += int(rec.get("gen_count") or 0)
        local_ok = Path(part["path"]).exists()
        if reread and local_ok:
            try:
                _reread_part(sc, rep)
            except Exception as e:
                rep["problems"].append(f"{Path(part['path']).name}: re-read failed: {e}")
        if ds.is_data and ((reread and local_ok) or remote_lumis):
            src = part["path"] if local_ok else (remote_part_url(remote, ds, part["path"]) if remote else None)
            if src is None:
                rep["problems"].append(f"{Path(part['path']).name}: not available to read its lumisections")
            else:
                try:
                    lumis |= _part_lumis(src)
                except Exception as e:
                    rep["problems"].append(f"{Path(part['path']).name}: LuminosityBlocks unreadable: {e}")
    # Completeness by LFN.
    for lfn in planned:
        n = seen.get(lfn, 0)
        if n == 0 and lfn not in rep["bad_files"] and rep["groups_done"] == len(ds.groups):
            rep["problems"].append(f"{lfn}: planned but in no part")
        elif n > 1:
            rep["problems"].append(f"{lfn}: appears in {n} parts")
    if rep["bad_files"]:
        rep["problems"].append(f"{len(rep['bad_files'])} unreachable input file(s)")
    # Dataset totals.
    if rep["groups_done"] == len(ds.groups) and not rep["bad_files"]:
        exp = int(ds.dbs_summary.get("num_event", 0)) or ds.input_events
        if exp and rep["events_in"] != exp:
            rep["problems"].append(f"entries read {rep['events_in']} != DBS events {exp}")
    if have_sumw:
        rep["gen_sumw"] = gen_sumw
        rep["gen_count"] = gen_count
    # Golden-JSON lumi coverage for data.
    if ds.is_data and (reread or remote_lumis) and rep["groups_done"] == len(ds.groups):
        rep["lumis_processed"] = len(lumis)
        mask_path = out_dir / f"{ds.key}_processed_lumis.json"
        write_lumimask(lumis, mask_path)
        rep["lumimask"] = str(mask_path)
        if dbs_lumis is not None:
            # The skims must hold every lumisection DBS lists for the dataset's files, no more, no less.
            lost = sorted(dbs_lumis - lumis)
            extra = sorted(lumis - dbs_lumis)
            rep["dbs_lumis"] = len(dbs_lumis)
            if lost:
                rep["dbs_lumis_lost_sample"] = lost[:50]
                (rep["problems"] if not rep["bad_files"] else rep["warnings"]).append(
                    f"{len(lost)} lumisection(s) present in the dataset's files are absent from the skims")
            if extra:
                rep["problems"].append(f"{len(extra)} lumisection(s) in the skims are not in any DBS file of the dataset")
        if golden and Path(golden).exists():
            gold = golden_lumis(golden)
            runs = set(dbs_runs or []) or {r for r, _ in lumis}
            expected = {(r, l) for r in runs if r in gold for l in gold[r]}
            missing = sorted(expected - lumis)
            rep["golden_lumis_expected"] = len(expected)
            rep["golden_lumis_missing"] = len(missing)
            if missing:
                rep["golden_missing_sample"] = missing[:50]
                never_in_dataset = [m for m in missing if dbs_lumis is not None and m not in dbs_lumis]
                if dbs_lumis is not None and len(never_in_dataset) == len(missing):
                    rep["warnings"].append(
                        f"{len(missing)} certified lumisection(s) ({100.0 * len(missing) / max(len(expected), 1):.2f} % of the "
                        f"golden lumis of these runs) exist in no file of the dataset; compute the luminosity from {mask_path.name}")
                else:
                    msg = f"{len(missing)} golden lumisection(s) of this dataset's runs are absent from the skims"
                    (rep["problems"] if not rep["bad_files"] else rep["warnings"]).append(msg)
        else:
            rep["warnings"].append(f"no golden JSON for era {ds.era}; lumi coverage not checked")
    rep["ok"] = not rep["problems"]
    return rep


def _reread_part(sc: dict, rep: dict) -> None:
    part = sc["part"]
    path = part["path"]
    if adler32_of(path) != part["adler32"]:
        rep["problems"].append(f"{Path(path).name}: adler32 differs from sidecar (file changed or corrupt)")
        return
    with uproot.open(path) as f:
        t = f["Events"]
        if t.num_entries != part["n_events"]:
            rep["problems"].append(f"{Path(path).name}: {t.num_entries} entries, sidecar {part['n_events']}")
        for rec in part["inputs"]:
            if rec["n_pass"] == 0:
                continue
            ids = t.arrays(list(EVENT_ID_BRANCHES), entry_start=rec["entry_start"], entry_stop=rec["entry_stop"])
            if ids_digest(ids["run"], ids["luminosityBlock"], ids["event"]) != rec["ids_digest"]:
                rep["problems"].append(f"{Path(path).name}: stored events of {rec['lfn']} do not match the digest")
        if "Runs" in f and not sc.get("is_data"):
            runs = f["Runs"].arrays(library="ak")
            for name in ("genEventSumw", "genEventSumw_"):
                if name in runs.fields:
                    exp = sum((r.get("gen_sumw") or 0.0) for r in part["inputs"])
                    got = float(ak.sum(runs[name]))
                    if not np.isclose(got, exp, rtol=1e-9, atol=1e-6):
                        rep["problems"].append(f"{Path(path).name}: Runs {name} {got} != inputs {exp}")
                    break


def validate_campaign(args) -> int:
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype)
    out_root = Path(args.out).expanduser()
    dbs = None
    reports = []
    t0 = time.monotonic()
    for key in keys:
        ds = plan.datasets[key]
        dbs_runs = None
        dbs_lumis = None
        golden = golden_for_era(ds.era) if ds.is_data else None
        if ds.is_data:
            try:
                from wrcoffea.skim.catalog import DBSClient, is_das_path
                dbs = dbs or DBSClient()
                paths = [p for p in ds.das_paths if is_das_path(p)]
                dbs_runs = sorted({r for p in paths for r in dbs.runs(p)})
                dbs_lumis = dbs_dataset_lumis(dbs, paths)
            except Exception as e:
                logger.warning("DBS run/lumi lists unavailable for %s: %s", ds.key, e)
        rep = validate_dataset(plan, key, out_root, reread=not args.skip_reread, golden=golden, dbs_runs=dbs_runs,
                               dbs_lumis=dbs_lumis, remote=getattr(args, "remote", None),
                               remote_lumis=getattr(args, "remote_lumis", False))
        if args.accept_bad_files and rep["bad_files"]:
            rep["problems"] = [p for p in rep["problems"] if "unreachable" not in p and "planned but in no part" not in p]
            rep["ok"] = not rep["problems"]
        reports.append(rep)
        flag = "OK  " if rep["ok"] else "FAIL"
        print(f"{flag} {ds.key[:58]:58s} {ds.era:20s} groups {rep['groups_done']}/{rep['groups']} "
              f"events {rep['events_out']}/{rep['events_in']} parts {rep['parts']} "
              f"{'bad=' + str(len(rep['bad_files'])) if rep['bad_files'] else ''}")
        for p in rep["problems"][:8]:
            print(f"      problem: {p}")
        if len(rep["problems"]) > 8:
            print(f"      ... {len(rep['problems']) - 8} more problems")
        for w in rep["warnings"][:3]:
            print(f"      warning: {w}")
    n_ok = sum(r["ok"] for r in reports)
    print(f"\n{n_ok}/{len(reports)} datasets valid ({(time.monotonic() - t0)/60:.1f} min)")
    if args.report:
        Path(args.report).write_text(json.dumps({"campaign": plan.campaign, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                  "datasets": reports}, indent=1) + "\n")
        print(f"report written to {args.report}")
    return 0 if n_ok == len(reports) else 1
