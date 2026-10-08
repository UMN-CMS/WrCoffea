"""Split a skimmed multi-model (scan) sample into one dataset per mass point.

The pre-legacy WR signal scans (``WRtoNLtoLLJJ_MWR500to3500`` and friends)
hold many mass points in one dataset, flagged event by event through boolean
``GenModel_<model>`` branches.  The skim keeps those branches and the worker
records, per input file and model, the number of generated events and the
sums of genWeight and genWeight^2 before the selection.  :func:`split_scan`
then reads the skimmed parts once per batch of models and writes, for every
model, ordinary skim parts with a Runs tree built from those sums, a sidecar,
a group status and a plan entry, so ``validate``, ``upload`` and ``fileset``
treat the model datasets like any other.
"""

from __future__ import annotations

import json
import logging
import os
import re
import resource
import time
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from wrcoffea.skim.plan import Plan, DatasetPlan
from wrcoffea.skim.recount import IdDigest
from wrcoffea.skim.schema import OutputSchema
from wrcoffea.skim.selection import EVENT_ID_BRANCHES
from wrcoffea.skim.worker import GroupStatus, load_group_status, write_group_status, verify_part
from wrcoffea.skim.writer import PartWriter, InputRecord, read_sidecar

logger = logging.getLogger(__name__)


def model_name(branch: str) -> str:
    """``GenModel_WRtoNLtoLLJJ_MWR4800_MN200_TuneCP5_13TeV_madgraph_pythia8`` -> ``WRtoNLtoLLJJ_MWR4800_MN200``."""
    m = re.match(r"GenModel_(.+?_MWR\d+_MN\d+)", branch)
    return m.group(1) if m else branch[len("GenModel_"):]


def _raise_open_files(n: int = 4096) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < n:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(n, hard), hard))


def split_scan(plan: Plan, key: str, out_root: str | Path, *, model_meta: dict | None = None,
               compression: str | None = None, flush_events: int = 5_000, part_max_bytes: float | None = None,
               models_per_pass: int = 120, chunk_entries: int = 50_000, keep_flags: bool = False,
               force: bool = False, plan_dir: str | Path | None = None) -> list[str]:
    """Split the finished scan dataset *key*; returns the new dataset keys."""
    ds = plan.datasets[key]
    if not ds.scan:
        raise ValueError(f"{key} is not a scan dataset")
    out_root = Path(out_root)
    scan_dir = out_root / ds.out_dir
    sidecars = []
    for gid in range(len(ds.groups)):
        st = load_group_status(scan_dir, gid)
        if st is None or st.status != "done":
            raise RuntimeError(f"{key}: group {gid} is not done")
        if st.bad_files:
            raise RuntimeError(f"{key}: group {gid} has unreachable inputs; split refused")
        sidecars += [read_sidecar(p["sidecar"]) for p in st.parts]
    sidecars.sort(key=lambda s: s["part"]["path"])

    # Models and their generated sums, from the input records of every part.
    sums: dict[str, dict] = {}
    for sc in sidecars:
        for rec in sc["part"]["inputs"]:
            for m, c in (rec.get("scan_counts") or {}).items():
                e = sums.setdefault(m, {"count": 0, "sumw": 0.0, "sumw2": 0.0})
                e["count"] += int(c)
                e["sumw"] += float(rec["scan_sumw"][m])
                e["sumw2"] += float(rec["scan_sumw2"][m])
    if not sums:
        raise RuntimeError(f"{key}: no scan sums recorded in the sidecars (was it skimmed with scan=true?)")
    models = sorted(sums)
    logger.info("%s: %d models, %d parts", key, len(models), len(sidecars))

    settings = plan.settings
    compression = compression or settings.get("compression", "zstd:19")
    part_max_bytes = float(part_max_bytes or settings.get("part_max_bytes", 2.0e9))
    scan_schema = plan.schema_for(key)
    model_schema = OutputSchema([b for b in scan_schema.branches if keep_flags or not b.name.startswith("GenModel_")],
                                scan_schema.dropped, scan_schema.unknown_kept, scan_schema.policy_version, scan_schema.synthetic)
    plan.schemas.setdefault(model_schema.digest(), model_schema.to_dict())
    read_names = [b.name for b in model_schema.branches]
    _raise_open_files()
    new_keys: list[str] = []
    model_meta = model_meta or {}

    for b0 in range(0, len(models), models_per_pass):
        batch = models[b0:b0 + models_per_pass]
        writers: dict[str, PartWriter] = {}
        for m in batch:
            name = model_name(m)
            out_dir = out_root / f"{ds.out_dir.rsplit('/', 2)[0]}/signals/{name}"
            if load_group_status(out_dir, 0) is not None and load_group_status(out_dir, 0).status == "done" and not force:
                logger.info("%s already split; skipping", name)
                continue
            for stale in out_dir.glob(f"{name}_g0000p*"):
                stale.unlink()
            writers[m] = PartWriter(out_dir, name, model_schema, is_data=False, compression=compression,
                                    flush_events=flush_events, part_max_bytes=part_max_bytes,
                                    campaign={"campaign": plan.campaign, "code_version": plan.code_version, "cuts": plan.cuts,
                                              "policy_version": plan.policy_version, "group": 0, "era": ds.era,
                                              "scan_source": key, "model": m},
                                    name_fmt="{base}_g0000p{index:02d}.root")
        if not writers:
            continue
        t0 = time.monotonic()
        for sc in sidecars:
            part = sc["part"]
            with uproot.open(part["path"]) as f:
                t = f["Events"]
                keys = t.keys()
                flags_here = [m for m in writers if m in keys]
                recs = {m: InputRecord(lfn=part["path"], size=int(part["size"]), adler32=part["adler32"], n_in=int(part["n_events"]))
                        for m in writers}
                digests = {m: IdDigest() for m in writers}
                for m, w in writers.items():
                    w.begin_input(recs[m], available_branches=[k for k in keys if not k.startswith("GenModel_")])
                for start in range(0, t.num_entries, chunk_entries):
                    stop = min(start + chunk_entries, t.num_entries)
                    if not flags_here:
                        break
                    flags = t.arrays(flags_here, entry_start=start, entry_stop=stop)
                    data = None
                    for m in flags_here:
                        mask = np.asarray(ak.to_numpy(flags[m]), dtype=bool)
                        n = int(mask.sum())
                        if n == 0:
                            continue
                        if data is None:
                            data = t.arrays([n_ for n_ in read_names if n_ in keys], entry_start=start, entry_stop=stop)
                        sel = data[mask]
                        digests[m].update(sel["run"], sel["luminosityBlock"], sel["event"])
                        writers[m].add_chunk(sel, n)
                # per-model Runs rows for this part: the generated sums of the inputs it holds
                for m, w in writers.items():
                    recs[m].ids_digest = recs[m].recount_digest = digests[m].hexdigest()
                    count = sum(int((r.get("scan_counts") or {}).get(m, 0)) for r in part["inputs"])
                    sw = sum(float((r.get("scan_sumw") or {}).get(m, 0.0)) for r in part["inputs"])
                    sw2 = sum(float((r.get("scan_sumw2") or {}).get(m, 0.0)) for r in part["inputs"])
                    runs = {"run": ak.Array(np.array([1], dtype=np.uint32)),
                            "genEventCount": ak.Array(np.array([count], dtype=np.int64)),
                            "genEventSumw": ak.Array(np.array([sw], dtype=np.float64)),
                            "genEventSumw2": ak.Array(np.array([sw2], dtype=np.float64))}
                    recs[m].gen_count, recs[m].gen_sumw = count, sw
                    w.end_input(runs, None)
        for m, w in writers.items():
            parts = w.close()
            for p in parts:
                verify_part(p, False)
            name = model_name(m)
            out_dir = Path(parts[0].path).parent if parts else out_root / f"{ds.out_dir.rsplit('/', 2)[0]}/signals/{name}"
            st = GroupStatus(dataset=name, gid=0, status="done", started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code_version=plan.code_version,
                             n_inputs=len(sidecars), n_done=len(sidecars), events_in=sums[m]["count"],
                             events_out=sum(p.n_events for p in parts),
                             parts=[{"path": p.path, "sidecar": p.sidecar, "index": p.index, "n_events": p.n_events,
                                     "size": p.size, "adler32": p.adler32, "n_inputs": len(p.inputs)} for p in parts])
            write_group_status(out_dir, st)
            meta = dict(model_meta.get(name, {}))
            meta.setdefault("physics_group", "Signal")
            meta.update({"dataset": name, "datatype": "mc", "era": ds.era, "run": ds.run, "year": ds.year,
                         "das_name": meta.get("das_name", name), "scan_source": key})
            plan.datasets[name] = DatasetPlan(
                key=name, era=ds.era, run=ds.run, year=ds.year, datatype="mc", physics_group="Signal", category="signals",
                das_paths=[key], metadata=meta,
                files=[{"lfn": sc["part"]["path"], "size": sc["part"]["size"], "events": sc["part"]["n_events"],
                        "adler32": sc["part"]["adler32"], "valid": True, "dataset": key} for sc in sidecars],
                groups=[[sc["part"]["path"] for sc in sidecars]], schema_id=model_schema.digest(),
                dbs_summary={"num_file": len(sidecars), "num_event": 0, "file_size": 0},
                out_dir=str(out_dir.relative_to(out_root)), notes=[f"split from scan {key}; genEventSumw from GenModel sums"],
            )
            new_keys.append(name)
        logger.info("batch of %d models written in %.1f min", len(writers), (time.monotonic() - t0) / 60)
        if plan_dir:
            plan.save(plan_dir)
    return new_keys
