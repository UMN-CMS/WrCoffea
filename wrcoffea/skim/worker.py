"""Process one plan group: fetch inputs, skim, recount, write parts, verify.

This is the unit of work for both the local runner and a Condor job.  For
every input file of the group it

1. downloads the file to scratch with checksum verification (prefetching the
   next file while the current one is skimmed);
2. checks the entry count against DBS;
3. computes the skim decision with the production selection *and* the
   independent recount implementation, chunk by chunk, and stops if they ever
   disagree;
4. streams the selected events into the current part;
5. records the input's Runs (and, for data, LuminosityBlocks) rows and the
   digest of its selected event identifiers;
6. deletes the local copy.

After the last input it closes the parts and re-reads every part to verify
that the stored event identifiers of every input match the recorded digests
and that the Runs sums add up.  A group status JSON (``_groups/gNNNN.json``)
records the outcome, every bad file with its attempt log, and timings.
"""

from __future__ import annotations

import json
import lzma
import zlib
import logging
import os
import shutil
import socket
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from wrcoffea.skim.catalog import RucioClient, xrootd_sources, is_das_path
from wrcoffea.skim.fetch import fetch_file, adler32_of, FetchResult
from wrcoffea.skim.plan import Plan, DatasetPlan, code_version
from wrcoffea.skim.recount import recount_mask, IdDigest, ids_digest
from wrcoffea.skim.schema import OutputSchema, build_schema, branch_types_of_tree, load_policy
from wrcoffea.skim.selection import skim_mask, selection_branches, EVENT_ID_BRANCHES
from wrcoffea.skim.runs import per_model_branch
from wrcoffea.skim.weight_sums import EventWeightSums, add_derived_runs_columns, derivable
from wrcoffea.skim.writer import PartWriter, InputRecord, PartRecord

logger = logging.getLogger(__name__)

STATUS_VERSION = 1


class SelectionMismatch(RuntimeError):
    """The production selection and the recount disagree on an event."""


class VerificationError(RuntimeError):
    """A written part does not match its sidecar."""


@dataclass
class GroupStatus:
    dataset: str
    gid: int
    status: str = "pending"            # pending | running | done | failed
    started: str = ""
    finished: str = ""
    host: str = ""
    pid: int = 0
    code_version: str = ""
    n_inputs: int = 0
    n_done: int = 0
    events_in: int = 0
    events_out: int = 0
    parts: list[dict] = field(default_factory=list)
    bad_files: list[dict] = field(default_factory=list)
    error: str | None = None
    seconds: float = 0.0
    fetch_seconds: float = 0.0
    skim_seconds: float = 0.0
    status_version: int = STATUS_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GroupStatus":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def group_status_path(out_dir: str | Path, gid: int) -> Path:
    return Path(out_dir) / "_groups" / f"g{gid:04d}.json"


class CorruptInput(RuntimeError):
    """An input that passed its checksum but cannot be read (a basket that does not decompress or
    deserialize): the file is corrupt at the source, every replica is identical, retrying is futile."""


def is_corrupt_error(e: BaseException) -> bool:
    if isinstance(e, (lzma.LZMAError, zlib.error)):
        return True
    name = type(e).__name__
    msg = str(e).lower()
    return name in ("DecompressionError", "DeserializationError") or "data error" in msg or "corrupt" in msg


PERMANENT_BAD = ("corrupt", "unavailable")   # unreadable at the source / no replica anywhere: never retried


def known_corrupt_inputs(ds: DatasetPlan, prev: GroupStatus | None) -> set[str]:
    """Inputs recorded as permanently bad by the plan (``bad_inputs``) or by this group's earlier attempt."""
    bad = set(getattr(ds, "bad_inputs", None) or [])
    if prev is not None:
        bad |= {b["lfn"] for b in prev.bad_files if b.get("category") in PERMANENT_BAD}
    return bad


def needs_retry(st: GroupStatus) -> bool:
    """A done group is worth redoing only when some bad input was unreachable (transient), not permanently bad."""
    return any(b.get("category", "unreachable") not in PERMANENT_BAD for b in st.bad_files)


class GroupInterrupted(Exception):
    """Raised in a worker that was told to stop (SIGTERM); the group is recorded as failed."""


def group_in_progress(st: GroupStatus | None) -> bool:
    """True when *st* says ``running`` and a Python process with that pid is still alive here.

    Used to refuse running a group twice (an orphaned worker of an earlier ``run`` may
    still be processing it).  A status whose process is gone is a crashed attempt.
    """
    if st is None or st.status != "running" or not st.pid:
        return False
    if st.host and st.host != socket.gethostname():
        return False      # written on another machine (a Condor worker, say): nothing to check here
    try:
        os.kill(st.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", str(st.pid)], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception:
        return True
    return "python" in cmd.lower()


def _relativize_parts(parts: list[dict], out_dir: Path) -> list[dict]:
    """Part and sidecar paths under *out_dir* become relative to it (the tree can then be moved as a whole)."""
    out = []
    root = str(out_dir.resolve())
    for p in parts:
        q = dict(p)
        for key in ("path", "sidecar"):
            v = q.get(key)
            if v and os.path.isabs(v):
                try:
                    q[key] = os.path.relpath(os.path.abspath(v), root) if os.path.abspath(v).startswith(root + os.sep) else v
                except ValueError:
                    pass
        out.append(q)
    return out


def _absolutize_parts(parts: list[dict], out_dir: Path) -> list[dict]:
    out = []
    for p in parts:
        q = dict(p)
        for key in ("path", "sidecar"):
            v = q.get(key)
            if v and not os.path.isabs(v):
                q[key] = str(out_dir / v)
        out.append(q)
    return out


def classify_unreachable(plan: Plan, out_root: str | Path, keys: list[str], rucio) -> dict:
    """Turn ``unreachable`` bad files that have no disk replica anywhere (Rucio) into ``unavailable``.

    Run where Rucio is reachable (laptop or login node) on the mirrored statuses; the rewritten
    statuses stop the groups from being retried for files that cannot be fetched by anyone.
    Returns ``{dataset key: {"unavailable": n, "events": n, "still_unreachable": n}}``.
    """
    out_root = Path(out_root)
    summary: dict[str, dict] = {}
    for key in keys:
        ds = plan.datasets[key]
        events_of = {f["lfn"]: int(f.get("events", 0)) for f in ds.files}
        out_dir = out_root / ds.out_dir
        for gid in range(len(ds.groups)):
            st = load_group_status(out_dir, gid)
            if st is None or not st.bad_files:
                continue
            changed = False
            for b in st.bad_files:
                if b.get("category", "unreachable") != "unreachable":
                    continue
                attempts = b.get("attempts", [])
                if attempts and not all(a.get("category") == "missing" for a in attempts):
                    continue      # timeouts/network errors: genuinely transient, keep retrying
                try:
                    reps = rucio.replicas(b["lfn"])
                except Exception as e:
                    logger.warning("Rucio lookup failed for %s: %s", b["lfn"], e)
                    continue
                disk = [r["rse"] for r in reps if r.get("state") == "AVAILABLE" and "Tape" not in r.get("rse", "")]
                if disk:
                    continue      # a disk replica exists: the redirectors just could not serve it, retry
                b["category"] = "unavailable"
                b["replicas"] = sorted({r["rse"] for r in reps})
                b["error"] = "no disk replica anywhere (DBS valid, Rucio lists none); skipped for good"
                changed = True
                d = summary.setdefault(key, {"unavailable": 0, "events": 0, "still_unreachable": 0})
                d["unavailable"] += 1
                d["events"] += events_of.get(b["lfn"], 0)
            d = summary.setdefault(key, {"unavailable": 0, "events": 0, "still_unreachable": 0})
            d["still_unreachable"] += sum(1 for b in st.bad_files if b.get("category", "unreachable") == "unreachable")
            if changed:
                write_group_status(out_dir, st)
    return {k: v for k, v in summary.items() if v["unavailable"] or v["still_unreachable"]}


def pending_groups(plan, out_root: str | Path, keys: list[str], *, force: bool = False,
                   include_failed: bool = True, retry_bad: bool = True, skip: set | None = None,
                   log=print) -> list[tuple[str, int]]:
    """Groups to (re)run: never run, failed, left ``running`` by a dead process, or done with
    unreachable inputs.  Groups in *skip* (for example those queued on Condor) are left out."""
    todo = []
    out_root = Path(out_root)
    for key in keys:
        ds = plan.datasets[key]
        out_dir = out_root / ds.out_dir
        for gid in range(len(ds.groups)):
            if skip and (key, gid) in skip:
                continue
            st = load_group_status(out_dir, gid)
            if st is None or force:
                todo.append((key, gid))
            elif st.status == "done":
                if retry_bad and needs_retry(st):
                    todo.append((key, gid))   # inputs unreachable last time (often transient): try again
            elif st.status == "failed" and include_failed:
                todo.append((key, gid))
            elif st.status == "running":
                if group_in_progress(st):
                    log(f"  {key} g{gid:04d}: still being processed by pid {st.pid} (started {st.started}); not rerunning it")
                    continue
                todo.append((key, gid))     # left "running" by a dead process
    return todo


def load_group_status(out_dir: str | Path, gid: int) -> GroupStatus | None:
    p = group_status_path(out_dir, gid)
    if not p.exists():
        return None
    try:
        st = GroupStatus.from_dict(json.loads(p.read_text()))
        st.parts = _absolutize_parts(st.parts, Path(out_dir).resolve())
        return st
    except Exception as e:  # unreadable status counts as pending
        logger.warning("unreadable status %s: %s", p, e)
        return None


def write_group_status(out_dir: str | Path, st: GroupStatus) -> Path:
    p = group_status_path(out_dir, st.gid)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.partial")
    doc = st.to_dict()
    doc["parts"] = _relativize_parts(doc["parts"], Path(out_dir))
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, p)
    return p


def _runs_table(fin) -> dict | None:
    if "Runs" not in fin:
        return None
    arr = fin["Runs"].arrays(library="ak")
    return {k: arr[k] for k in arr.fields}


def _lumi_table(fin) -> dict | None:
    if "LuminosityBlocks" not in fin:
        return None
    arr = fin["LuminosityBlocks"].arrays(library="ak")
    return {k: arr[k] for k in arr.fields}


def _remove_part_files(part: PartRecord) -> None:
    for p in (part.path, part.sidecar):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def verify_part(part: PartRecord, is_data: bool) -> None:
    """Re-read a finished part and check it against its own records."""
    with uproot.open(part.path) as f:
        tree = f["Events"]
        if tree.num_entries != part.n_events:
            raise VerificationError(f"{part.path}: {tree.num_entries} entries, sidecar says {part.n_events}")
        total = 0
        for rec in part.inputs:
            if rec.n_pass != rec.entry_stop - rec.entry_start:
                raise VerificationError(f"{part.path}: inconsistent entry range for {rec.lfn}")
            total += rec.n_pass
            if rec.n_pass == 0:
                continue
            ids = tree.arrays(list(EVENT_ID_BRANCHES), entry_start=rec.entry_start, entry_stop=rec.entry_stop)
            got = ids_digest(ids["run"], ids["luminosityBlock"], ids["event"])
            if got != rec.ids_digest:
                raise VerificationError(f"{part.path}: event ids of {rec.lfn} differ from the skim digest")
            if rec.recount_digest and got != rec.recount_digest:
                raise VerificationError(f"{part.path}: event ids of {rec.lfn} differ from the recount digest")
        if total != part.n_events:
            raise VerificationError(f"{part.path}: inputs sum to {total} events, part has {part.n_events}")
        if not is_data and "Runs" in f:
            runs = f["Runs"].arrays(library="ak")
            for name in ("genEventSumw", "genEventSumw_"):
                if name in runs.fields:
                    expected = sum(r.gen_sumw or 0.0 for r in part.inputs)
                    got_sumw = float(ak.sum(runs[name]))
                    if not np.isclose(got_sumw, expected, rtol=1e-9, atol=1e-6):
                        raise VerificationError(f"{part.path}: Runs {name}={got_sumw} but inputs sum to {expected}")
                    break
            for name in ("genEventCount", "genEventCount_"):
                if name in runs.fields:
                    expected_n = sum(r.gen_count or 0 for r in part.inputs)
                    if int(ak.sum(runs[name])) != expected_n:
                        raise VerificationError(f"{part.path}: Runs {name} != sum over inputs")
                    break
    if adler32_of(part.path) != part.adler32:
        raise VerificationError(f"{part.path}: adler32 changed after writing")


def skim_input(local_path: str, rec: InputRecord, writer: PartWriter, schema: OutputSchema, *,
               is_data: bool, chunk_entries: int = 20_000, cuts: dict | None = None,
               scan: bool = False,
               decompression_threads: int = 1) -> InputRecord:
    """Skim one local NanoAOD file into *writer* (see module docstring, steps 2-5).

    *decompression_threads* > 1 decompresses baskets in parallel (LZMA releases the GIL).  The
    executor is created per input: uproot shuts the executor down when it closes the file.
    """
    from wrcoffea.skim.selection import SKIM_CUTS
    cuts = cuts or SKIM_CUTS
    t0 = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=int(decompression_threads)) if int(decompression_threads) > 1 else None
    with uproot.open(local_path, decompression_executor=executor) as fin:
        if "Events" not in fin:
            raise RuntimeError(f"{local_path}: no Events tree")
        tree = fin["Events"]
        keys = tree.keys()
        n_in = int(tree.num_entries)
        rec.n_in = n_in
        if rec.events_dbs and n_in != rec.events_dbs:
            raise RuntimeError(f"{rec.lfn}: {n_in} entries in file but DBS lists {rec.events_dbs} events")
        if n_in == 0:
            # An empty file (DBS lists it with 0 events; such NanoAOD files may not even define the
            # event branches): nothing to select, but its Runs / LuminosityBlocks rows still count
            # (a data file can hold lumisections in which no event passed the stream's trigger).
            writer.begin_input(rec)
            empty = IdDigest().hexdigest()
            rec.ids_digest = rec.recount_digest = empty
            runs = _runs_table(fin)
            lumis = _lumi_table(fin) if is_data else None
            rec.skim_seconds = time.monotonic() - t0
            if executor is not None:
                executor.shutdown(wait=False)
            writer.end_input(runs, lumis)
            return rec
        sel_names = selection_branches(keys) + list(EVENT_ID_BRANCHES)
        # Theory-weight sums the input's Runs tree lacks (PSSumw in UL NanoAODv9) are accumulated from
        # every event, before the selection, and written as Runs branches (wrcoffea.skim.weight_sums).
        # Scan samples keep per-model sums and are left alone.
        runs_keys = list(fin["Runs"].keys()) if "Runs" in fin else []
        derive = ([] if is_data or scan or any(per_model_branch(k) for k in runs_keys)
                  else derivable(runs_keys, keys))
        sums = EventWeightSums(derive) if derive else None
        if sums is not None:
            sel_names = sel_names + [b for b in sums.branches if b not in sel_names]
        gen_models = sorted(k for k in keys if k.startswith("GenModel_")) if scan else []
        if gen_models:
            sel_names = sel_names + gen_models + (["genWeight"] if "genWeight" in keys else [])
            counts = {m: 0 for m in gen_models}
            sumw = {m: 0.0 for m in gen_models}
            sumw2 = {m: 0.0 for m in gen_models}
        kept = [n for n in schema.names if n in keys]
        writer.begin_input(rec, available_branches=keys)
        d_prod, d_rec = IdDigest(), IdDigest()
        for start in range(0, n_in, chunk_entries):
            stop = min(start + chunk_entries, n_in)
            sel = tree.arrays(sel_names, entry_start=start, entry_stop=stop)
            if sums is not None:
                sums.add(sel)
            if gen_models:
                w = np.asarray(ak.to_numpy(sel["genWeight"]), dtype=np.float64) if "genWeight" in keys else np.ones(stop - start)
                for m in gen_models:
                    flag = np.asarray(ak.to_numpy(sel[m]), dtype=bool)
                    counts[m] += int(flag.sum())
                    sumw[m] += float(w[flag].sum())
                    sumw2[m] += float((w[flag] ** 2).sum())
            m_prod = skim_mask(sel, cuts)
            m_rec = recount_mask(sel, cuts)
            if not np.array_equal(m_prod, m_rec):
                bad = np.nonzero(m_prod != m_rec)[0]
                raise SelectionMismatch(
                    f"{rec.lfn}: selection and recount disagree on {len(bad)} event(s) in "
                    f"entries [{start}, {stop}), first at {start + int(bad[0])}"
                )
            n_sel = int(m_prod.sum())
            if n_sel == 0:
                continue
            ids = sel[m_prod]
            d_prod.update(ids["run"], ids["luminosityBlock"], ids["event"])
            ids2 = sel[m_rec]
            d_rec.update(ids2["run"], ids2["luminosityBlock"], ids2["event"])
            # to_packed copies only the selected events: a boolean-masked awkward array otherwise keeps
            # the whole chunk's buffers (all branches) alive in the writer's buffer until the next flush,
            # which grew to several GB per worker for low pass rates.
            data = ak.to_packed(tree.arrays(kept, entry_start=start, entry_stop=stop)[m_prod])
            writer.add_chunk(data, n_sel)
        rec.ids_digest = d_prod.hexdigest()
        rec.recount_digest = d_rec.hexdigest()
        if gen_models:
            rec.scan_counts, rec.scan_sumw, rec.scan_sumw2 = counts, sumw, sumw2
        runs = _runs_table(fin)
        if sums is not None and runs:
            runs = add_derived_runs_columns(runs, sums)
            rec.derived_runs_branches = list(sums.names)
        lumis = _lumi_table(fin) if is_data else None
    if executor is not None:
        executor.shutdown(wait=False)
    rec.skim_seconds = time.monotonic() - t0
    writer.end_input(runs, lumis)
    return rec


def _input_sources(lfn: str, rucio: RucioClient | None) -> list[str]:
    if lfn.startswith("/store/"):
        return xrootd_sources(lfn, rucio)
    return [lfn]  # explicit URL or local path (private sample)


def _fetch_one(lfn: str, frec: dict, scratch: Path, rucio: RucioClient | None, streams: int) -> FetchResult | str:
    if os.path.exists(lfn) and not lfn.startswith("/store/"):
        return lfn  # local file: use in place
    dest = scratch / os.path.basename(lfn)
    if dest.exists() and frec.get("adler32") and adler32_of(str(dest)) == frec["adler32"]:
        res = FetchResult(lfn=lfn, dest=str(dest), ok=True, size=dest.stat().st_size,
                          adler32=frec["adler32"], source="scratch")
        return res
    return fetch_file(lfn, str(dest), _input_sources(lfn, rucio), adler32=frec.get("adler32") or None,
                      size=int(frec.get("size", 0)), streams=streams)


def process_group(plan: Plan, key: str, gid: int, *, out_root: str | Path, scratch: str | Path,
                  rucio_account: str | None = None, keep_inputs: bool = False,
                  chunk_entries: int = 20_000, compression: str | None = None,
                  flush_events: int | None = None, part_max_bytes: float | None = None,
                  force: bool = False, prefetch: bool = True, streams: int = 4,
                  decompression_threads: int = 1) -> GroupStatus:
    """Run one group end to end; returns (and writes) its :class:`GroupStatus`."""
    ds: DatasetPlan = plan.datasets[key]
    lfns = ds.groups[gid]
    files = ds.file_by_lfn()
    out_dir = Path(out_root) / ds.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = load_group_status(out_dir, gid)
    if prev is not None and prev.pid != os.getpid() and group_in_progress(prev) and not force:
        raise RuntimeError(f"{ds.key} g{gid:04d} is being processed by pid {prev.pid} on {prev.host} "
                           f"(started {prev.started}); refusing to run it twice (use --force to override)")
    if prev is not None and prev.status == "done" and not (prev.bad_files and needs_retry(prev)) and not force:
        logger.info("%s g%04d already done; skipping", ds.key, gid)
        return prev
    if prev is not None and prev.status == "done" and prev.bad_files:
        logger.info("%s g%04d finished earlier with %d unreachable input(s); redoing it", ds.key, gid, len(prev.bad_files))
        for p in prev.parts:
            _remove_part_files(PartRecord(path=p["path"], index=p["index"], n_events=p["n_events"], size=p["size"],
                                          adler32=p["adler32"], inputs=[], runs={}, n_lumis=None, schema_digest="",
                                          sidecar=p["sidecar"]))

    settings = plan.settings
    compression = compression or settings.get("compression", "zstd:19")
    flush_events = int(flush_events or settings.get("flush_events", 50_000))
    part_max_bytes = float(part_max_bytes or settings.get("part_max_bytes", 2.0e9))

    st = GroupStatus(dataset=ds.key, gid=gid, status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     host=socket.gethostname(), pid=os.getpid(), code_version=code_version(), n_inputs=len(lfns))
    write_group_status(out_dir, st)
    t_start = time.monotonic()

    # Remove leftovers of an earlier attempt of this group.
    for stale in out_dir.glob(f"{ds.key}_g{gid:04d}p*"):
        stale.unlink()

    if ds.schema_id == "lazy":
        schema = None
    else:
        schema = plan.schema_for(key)

    # One scratch directory per group: groups of the same dataset run in
    # parallel and each removes only its own inputs when it finishes.
    scratch_dir = Path(scratch) / ds.key / f"g{gid:04d}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    rucio = RucioClient(rucio_account) if rucio_account else None

    corrupt = known_corrupt_inputs(ds, prev)
    prev_cat = {b["lfn"]: b.get("category") for b in (prev.bad_files if prev else [])}
    for lfn in lfns:
        if lfn in corrupt:
            cat = prev_cat.get(lfn) if prev_cat.get(lfn) in PERMANENT_BAD else "corrupt"
            st.bad_files.append({"lfn": lfn, "category": cat, "error": f"known {cat} input, skipped"})
            logger.warning("BAD FILE %s: known %s input, skipped", lfn, cat)
    lfns = [lfn for lfn in lfns if lfn not in corrupt]

    writer: PartWriter | None = None
    pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
    current_lfn = None
    try:
        fut = pool.submit(_fetch_one, lfns[0], files.get(lfns[0], {}), scratch_dir, rucio, streams) if (pool and lfns) else None
        for i, lfn in enumerate(lfns):
            current_lfn = lfn
            frec = files.get(lfn, {})
            res = fut.result() if pool else _fetch_one(lfn, frec, scratch_dir, rucio, streams)
            if pool and i + 1 < len(lfns):
                fut = pool.submit(_fetch_one, lfns[i + 1], files.get(lfns[i + 1], {}), scratch_dir, rucio, streams)
            if isinstance(res, str):
                local, fetch_s, source = res, 0.0, "local"
            else:
                if not res.ok:
                    st.bad_files.append({"lfn": lfn, "attempts": [asdict(a) for a in res.attempts]})
                    logger.error("BAD FILE %s: unreachable from every source (%d attempts)", lfn, len(res.attempts))
                    continue
                local, fetch_s, source = res.dest, sum(a.seconds for a in res.attempts), res.source
            st.fetch_seconds += fetch_s
            if schema is None:
                with uproot.open(local) as fin:
                    schema = build_schema([branch_types_of_tree(fin["Events"])], load_policy())
                plan.schemas[schema.digest()] = schema.to_dict()
                logger.info("%s: schema derived from first input (%d branches)", ds.key, len(schema.branches))
            if writer is None:
                writer = PartWriter(out_dir, ds.key, schema, is_data=ds.is_data, compression=compression,
                                    flush_events=flush_events, part_max_bytes=part_max_bytes,
                                    campaign={"campaign": plan.campaign, "code_version": plan.code_version,
                                              "cuts": plan.cuts, "policy_version": plan.policy_version,
                                              "group": gid, "era": ds.era, "das_paths": ds.das_paths},
                                    name_fmt="{base}_g{gid:04d}p{index:02d}.root", name_fields={"gid": gid})
            rec = InputRecord(lfn=lfn, dataset=frec.get("dataset", ""), size=int(frec.get("size", 0)),
                              adler32=frec.get("adler32", ""), events_dbs=int(frec.get("events", 0)),
                              source=source, fetch_seconds=fetch_s)
            try:
                skim_input(local, rec, writer, schema, is_data=ds.is_data, chunk_entries=chunk_entries, cuts=plan.cuts,
                           decompression_threads=decompression_threads,
                           scan=getattr(ds, "scan", False))
            except Exception as e:
                if is_corrupt_error(e):
                    # Checksum-valid but unreadable: record it so the next run of this group skips it
                    # (the parts of this attempt are discarded below; the retry converges).
                    st.bad_files.append({"lfn": lfn, "category": "corrupt", "error": f"{type(e).__name__}: {e}"[:300]})
                    raise CorruptInput(f"{lfn}: {type(e).__name__}: {e}") from e
                raise
            finally:
                if not keep_inputs and not isinstance(res, str):
                    try:
                        os.remove(local)
                    except FileNotFoundError:
                        pass
            st.n_done += 1
            st.events_in += rec.n_in
            st.events_out += rec.n_pass
            st.skim_seconds += rec.skim_seconds
            logger.info("%s g%04d [%d/%d] %s: %d -> %d (%.2f%%) fetch %.0fs skim %.0fs",
                        ds.key, gid, i + 1, len(lfns), os.path.basename(lfn), rec.n_in, rec.n_pass,
                        100.0 * rec.n_pass / max(rec.n_in, 1), fetch_s, rec.skim_seconds)
        parts = writer.close() if writer is not None else []
        for part in parts:
            verify_part(part, ds.is_data)
        st.parts = [{"path": p.path, "sidecar": p.sidecar, "index": p.index, "n_events": p.n_events,
                     "size": p.size, "adler32": p.adler32, "n_inputs": len(p.inputs)} for p in parts]
        st.status = "done"
    except Exception as e:
        logger.error("%s g%04d FAILED: %s", ds.key, gid, e)
        if writer is not None:
            writer.abort()
            for part in writer.parts:
                _remove_part_files(part)
        st.status = "failed"
        st.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
        if not keep_inputs:
            shutil.rmtree(scratch_dir, ignore_errors=True)
    st.finished = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    st.seconds = time.monotonic() - t_start
    write_group_status(out_dir, st)
    return st
