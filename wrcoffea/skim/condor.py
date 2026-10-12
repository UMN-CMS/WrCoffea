"""Condor submission for the skim campaign (LPC-style vanilla universe inside an apptainer image).

One job = one group.  The job unpacks a small code tarball, runs ``skim.py group`` on a per-dataset
slice of the plan inside the coffea container on the worker, and ships the results with
``skim.py ship``: parts and sidecars are copied with xrdcp (adler32-verified) to the campaign's
remote root, the group status last, so an interrupted copy never looks finished.  Statuses and
sidecars store part names relative to the dataset directory, so the remote tree has exactly the
layout of a local campaign root and ``skim.py mirror`` can pull the small JSON files back to a
local root for ``status``/``validate``/``fileset``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

from wrcoffea.skim.plan import Plan
from wrcoffea.skim.worker import load_group_status, pending_groups
from wrcoffea.skim.writer import read_sidecar

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-almalinux9-noml:2026.9.0-py3.12"
CODE_PATHS = ("bin/skim.py", "wrcoffea")      # relative to the repo root; __pycache__ excluded

WRAPPER = r'''#!/bin/bash
# Condor job: one skim group.  Arguments: dataset key, group id, remote campaign root (xrootd URL).
KEY="$1"; GID="$2"; REMOTE="$3"; STREAMS="${4:-4}"; THREADS="${5:-1}"
echo "=== $(date -u) $(hostname) $KEY g$GID -> $REMOTE ==="
JOB=$PWD
mkdir -p code out scratch && tar xzf skim_code.tgz -C code || exit 3
export PYTHONPATH=$JOB/code
python3 code/bin/skim.py group --plan-dir "$JOB" --out "$JOB/out" --scratch "$JOB/scratch" \
    --dataset "$KEY" --gid "$GID" --streams "$STREAMS" --decompression-threads "$THREADS"
rc=$?
echo "=== $(date -u) group rc=$rc; shipping outputs ==="
python3 code/bin/skim.py ship --plan-dir "$JOB" --out "$JOB/out" --remote "$REMOTE" --dataset "$KEY" --gid "$GID"
src=$?
echo "=== $(date -u) ship rc=$src ==="
rm -rf "$JOB/out" "$JOB/scratch" "$JOB/code"
[ $rc -eq 0 ] && [ $src -eq 0 ]
'''

JDL = '''universe                = vanilla
executable              = run_group.sh
arguments               = "$(key) $(gid) {remote} {streams} {threads}"
+SingularityImage       = "{image}"
+SingularityBindCVMFS   = True
+SkimGroup              = "$(era)/$(key):$(gid)"
job_machine_attrs       = Machine
job_machine_attrs_history_length = 4
requirements            = (TARGET.Machine =!= MY.MachineAttrMachine1) && (TARGET.Machine =!= MY.MachineAttrMachine2) && (TARGET.Machine =!= MY.MachineAttrMachine3){avoid}
periodic_hold           = (JobStatus == 2) && ((time() - EnteredCurrentStatus) > {max_seconds})
periodic_hold_reason    = "running longer than {max_hours} h (overloaded node?); released to restart elsewhere"
periodic_release        = (HoldReasonCode == 3) && (NumJobStarts < 4)
use_x509userproxy       = true
x509userproxy           = {proxy}
{environment}request_cpus            = {cpus}
request_memory          = {memory_mb}
request_disk            = {disk_kb}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = skim_code.tgz, plans/$(key)/plan.json, plans/$(key)/schemas.json
transfer_output_files   = ""
output                  = logs/$(key)_g$(gid).out
error                   = logs/$(key)_g$(gid).err
log                     = logs/condor.log
notification            = never
queue key,gid,era from groups.txt
'''


def make_code_tarball(repo_root: str | Path, dest: str | Path) -> Path:
    """tar.gz of the code a job needs (``bin/skim.py`` and the ``wrcoffea`` package, no caches)."""
    repo_root, dest = Path(repo_root), Path(dest)

    def _filter(ti: tarfile.TarInfo):
        if "__pycache__" in ti.name or ti.name.endswith((".pyc", ".pyo")):
            return None
        return ti

    with tarfile.open(dest, "w:gz") as tar:
        for rel in CODE_PATHS:
            tar.add(repo_root / rel, arcname=rel, filter=_filter)
    return dest


def corrupt_inputs_from_statuses(plan: Plan, key: str, out_root: str | Path) -> list[str]:
    """LFNs any earlier attempt of *key* recorded as corrupt (statuses under *out_root*)."""
    ds = plan.datasets[key]
    out_dir = Path(out_root) / ds.out_dir
    from wrcoffea.skim.worker import PERMANENT_BAD
    bad: set[str] = set(ds.bad_inputs or [])
    for gid in range(len(ds.groups)):
        st = load_group_status(out_dir, gid)
        if st is not None:
            bad |= {b["lfn"] for b in st.bad_files if b.get("category") in PERMANENT_BAD}
    return sorted(bad)


def write_plan_slice(plan: Plan, key: str, dest_dir: str | Path, bad_inputs: list[str] | None = None) -> Path:
    """A plan directory holding only *key* and its schema (what one job needs)."""
    import dataclasses
    ds = plan.datasets[key]
    if bad_inputs:
        ds = dataclasses.replace(ds, bad_inputs=sorted(set(ds.bad_inputs or []) | set(bad_inputs)))
    sliced = Plan(**{**{k: v for k, v in plan.__dict__.items() if k not in ("datasets", "schemas")},
                    "datasets": {key: ds},
                    "schemas": {ds.schema_id: plan.schemas[ds.schema_id]} if ds.schema_id in plan.schemas else {}})
    return sliced.save(dest_dir)


def _era_from_iwd(iwd: str | None, key: str) -> str | None:
    """Era of a legacy (era-less) job: read it from the plan slice in the job's submit directory."""
    if not iwd:
        return None
    try:
        doc = json.loads((Path(iwd) / "plans" / key / "plan.json").read_text())
        return doc["datasets"][key]["era"]
    except Exception:
        return None


def inflight_groups(text: str | None = None) -> set[tuple[str | None, str, int]]:
    """``(era, key, gid)`` of every group with a job still queued or running (SkimGroup ClassAd).

    *text* is the output of ``condor_q -af SkimGroup JobStatus Iwd``; when None it is obtained by
    running condor_q here (not possible inside the container: the host wrapper passes a file).
    Tags are ``era/key:gid``; older jobs were tagged ``key:gid`` and their era is read from the
    plan slice in their submit directory (None if that fails, which callers treat as any era).
    """
    if text is None:
        if shutil.which("condor_q") is None:
            return set()
        try:
            text = subprocess.run(["condor_q", "-af", "SkimGroup", "JobStatus", "Iwd"], capture_output=True,
                                  text=True, timeout=120).stdout
        except Exception as e:
            logger.warning("condor_q failed (%s); assuming no jobs in flight", e)
            return set()
    found: set = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] in ("undefined", ""):
            continue
        tag = parts[0].strip('"')
        if parts[1] not in ("1", "2", "5", "6", "7"):       # idle, running, held, transferring
            continue
        era = None
        if "/" in tag:
            era, tag = tag.split("/", 1)
        key, _, gid = tag.rpartition(":")
        if not key or not gid.isdigit():
            continue
        if era is None:
            era = _era_from_iwd(parts[2] if len(parts) > 2 else None, key)
        found.add((era, key, int(gid)))
    return found


def prepare_jobs(plan: Plan, keys: list[str], *, out_root: str | Path, remote: str, jobs_dir: str | Path,
                 repo_root: str | Path, proxy: str, image: str = DEFAULT_IMAGE, cpus: int = 1,
                 memory_mb: int = 4000, disk_gb: float = 20.0, streams: int = 4, decompression_threads: int = 1,
                 max_hours: float = 8.0, avoid_hosts: tuple[str, ...] = (), max_groups: int | None = None,
                 force: bool = False, retry_bad: bool = True, skip_inflight: bool = True,
                 rucio_account: str | None = None, inflight_text: str | None = None) -> dict:
    """Write a complete Condor submission for the pending groups of *keys*.

    *out_root* is where the campaign's statuses are read from (a local or mounted copy of the remote
    tree, kept current with ``skim.py mirror``); *remote* is the xrootd URL the jobs write to.
    *rucio_account* is set as ``RUCIO_ACCOUNT`` in the jobs' environment, the account their Rucio
    replica lookup (the fallback when no redirector serves a file) authenticates as.
    """
    jobs_dir = Path(jobs_dir)
    (jobs_dir / "logs").mkdir(parents=True, exist_ok=True)
    (jobs_dir / "plans").mkdir(exist_ok=True)
    inflight = inflight_groups(inflight_text) if skip_inflight else set()
    # a dataset key can exist in several eras (2023 and 2023BPix share sample names): match the era too
    skip = {(k, g) for era, k, g in inflight if k in plan.datasets and era in (None, plan.datasets[k].era)}
    todo = pending_groups(plan, out_root, keys, force=force, retry_bad=retry_bad, skip=skip, log=logger.info)
    if max_groups:
        todo = todo[:max_groups]
    for key in sorted({k for k, _ in todo}):
        write_plan_slice(plan, key, jobs_dir / "plans" / key, bad_inputs=corrupt_inputs_from_statuses(plan, key, out_root))
    make_code_tarball(repo_root, jobs_dir / "skim_code.tgz")
    (jobs_dir / "groups.txt").write_text("".join(f"{k} {g} {plan.datasets[k].era}\n" for k, g in todo))
    wrapper = jobs_dir / "run_group.sh"
    wrapper.write_text(WRAPPER)
    wrapper.chmod(0o755)
    avoid = "".join(f' && (TARGET.Machine =!= "{h}")' for h in avoid_hosts)
    (jobs_dir / "submit.jdl").write_text(JDL.format(remote=remote, streams=streams, threads=decompression_threads,
                                                    image=image, proxy=proxy, cpus=cpus, memory_mb=memory_mb,
                                                    disk_kb=int(disk_gb * 1e6), avoid=avoid,
                                                    max_hours=f"{max_hours:g}", max_seconds=int(max_hours * 3600),
                                                    environment=(f'environment             = "RUCIO_ACCOUNT={rucio_account}"\n'
                                                                 if rucio_account else "")))
    n_files = sum(len(plan.datasets[k].groups[g]) for k, g in todo)
    gb = sum(sum(plan.datasets[k].file_by_lfn()[l]["size"] for l in plan.datasets[k].groups[g]) for k, g in todo) / 1e9
    summary = {"jobs_dir": str(jobs_dir), "groups": len(todo), "datasets": len({k for k, _ in todo}),
               "input_files": n_files, "input_gb": round(gb, 1), "skipped_inflight": len(skip),
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (jobs_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# ----------------------------------------------------------------------------------------------
# shipping a finished group to the remote root (job side)
# ----------------------------------------------------------------------------------------------

def _xrd(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return -9, f"{cmd[0]} timed out after {timeout:.0f} s"
    return proc.returncode, (proc.stderr or proc.stdout or "").strip()


def _split_url(url: str) -> tuple[str, str]:
    """``root://host//path`` -> (``root://host``, ``/path``)."""
    assert url.startswith("root://"), url
    host, _, path = url[len("root://"):].partition("/")
    return "root://" + host, "/" + path.lstrip("/")


class RemoteUnavailable(RuntimeError):
    """The storage did not answer a stat within the allowed time (busy server), file state unknown."""


def remote_size(url: str, timeout: float = 120, attempts: int = 3) -> int | None:
    """Size of *url*, None when it does not exist; raises RemoteUnavailable when the server keeps timing out."""
    host, path = _split_url(url)
    rc, out = -9, ""
    for attempt in range(1, attempts + 1):
        rc, out = _xrd(["xrdfs", host, "stat", path], timeout * attempt)
        if rc != -9:
            break
        logger.warning("stat attempt %d/%d timed out for %s", attempt, attempts, path)
    if rc == -9:
        raise RemoteUnavailable(f"stat of {url} timed out {attempts} times")
    if rc != 0:
        return None
    for line in out.splitlines():
        if line.strip().lower().startswith("size:"):
            return int(line.split(":", 1)[1])
    return None


def ship_file(local: str | Path, url: str, *, adler32: str | None = None, attempts: int = 3,
              timeout: float = 3600) -> None:
    """xrdcp *local* to *url*, checksum-verified when *adler32* is given, size-verified always."""
    local = Path(local)
    cmd = ["xrdcp", "-f", "--nopbar"]
    if adler32:
        cmd += ["--cksum", f"adler32:{adler32}"]
    cmd += [str(local), url]
    last = ""
    for attempt in range(1, attempts + 1):
        rc, msg = _xrd(cmd, timeout)
        if rc == 0:
            try:
                size = remote_size(url)
            except RemoteUnavailable as e:
                if adler32:
                    # xrdcp already verified the server-side adler32 of the copy: that is the stronger check.
                    logger.warning("%s: %s; accepting the checksum-verified copy", local.name, e)
                    return
                size = None
                last = str(e)
            if size == local.stat().st_size:
                return
            last = last or f"remote size {size} != {local.stat().st_size}"
        else:
            last = f"rc={rc} {msg[-300:]}"
        logger.warning("ship attempt %d/%d failed for %s: %s", attempt, attempts, local.name, last)
        time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"could not ship {local} to {url}: {last}")


def ship_group(plan: Plan, key: str, gid: int, *, out_root: str | Path, remote: str) -> dict:
    """Copy one finished group's parts, sidecars and status to ``<remote>/<out_dir>/`` (status last)."""
    ds = plan.datasets[key]
    out_dir = Path(out_root) / ds.out_dir
    st = load_group_status(out_dir, gid)
    if st is None:
        raise RuntimeError(f"{key} g{gid:04d}: no status under {out_dir}")
    rdir = f"{remote.rstrip('/')}/{ds.out_dir}"
    shipped = []
    for p in st.parts:
        sc = read_sidecar(p["sidecar"])
        ship_file(p["path"], f"{rdir}/{Path(p['path']).name}", adler32=sc["part"]["adler32"])
        ship_file(p["sidecar"], f"{rdir}/{Path(p['sidecar']).name}")
        shipped += [Path(p["path"]).name, Path(p["sidecar"]).name]
    status_path = out_dir / "_groups" / f"g{gid:04d}.json"
    ship_file(status_path, f"{rdir}/_groups/{status_path.name}")
    shipped.append(f"_groups/{status_path.name}")
    logger.info("%s g%04d: shipped %d file(s) to %s (status %s)", key, gid, len(shipped), rdir, st.status)
    return {"key": key, "gid": gid, "status": st.status, "files": shipped, "remote_dir": rdir}


# ----------------------------------------------------------------------------------------------
# mirroring the small JSON files of a remote tree into a local campaign root
# ----------------------------------------------------------------------------------------------

def _is_missing(message: str) -> bool:
    """xrootd error text for a path that does not exist (Wisconsin's HDFS-backed server says
    "[3011] Too many DFS read attempts" instead of "no such file")."""
    m = message.lower()
    return "no such file" in m or "not found" in m or "[3011]" in m


def remote_listing(url: str, timeout: float = 300) -> list[str]:
    """Paths (absolute, without host) of the files below *url*."""
    host, path = _split_url(url)
    rc, out = _xrd(["xrdfs", host, "ls", "-R", path], timeout)
    if rc != 0:
        raise RuntimeError(f"xrdfs ls {url}: {out[-300:]}")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _fetch_json(host: str, rpath: str, local: Path, timeout: float = 120) -> bool:
    """xrdcp one small remote file; False when it does not exist (or cannot be read)."""
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".part")
    rc, msg = _xrd(["xrdcp", "-f", "--nopbar", f"{host}/{rpath}", str(tmp)], timeout)
    if rc != 0:
        tmp.unlink(missing_ok=True)
        if _is_missing(msg) or "not found" in msg.lower() or rc in (54, 51):
            return False
        raise RuntimeError(f"xrdcp {host}/{rpath}: {msg[-200:]}")
    tmp.replace(local)
    return True


def mirror_metadata(plan: Plan, keys: list[str], *, out_root: str | Path, remote: str, workers: int = 8,
                    refresh: bool = False) -> dict:
    """Fetch the group statuses and part sidecars of *keys* from *remote* into *out_root*.

    No directory listings: at Wisconsin a listing takes 30 s on a good day and times out
    (> 5 min) on a busy one, while fetching a small file by name takes 1-2 s and a missing one
    fails as fast.  The status of every group not yet done locally is therefore probed by name
    (in parallel) and the sidecars named in it are fetched.
    """
    import concurrent.futures as cf
    host, _ = _split_url(remote)
    out_root = Path(out_root)
    todo = []
    for key in keys:
        ds = plan.datasets[key]
        for gid in range(len(ds.groups)):
            local_st = load_group_status(out_root / ds.out_dir, gid)
            if local_st is not None and local_st.status == "done" and not refresh:
                continue
            todo.append((key, gid))

    def one(item):
        key, gid = item
        ds = plan.datasets[key]
        rdir = f"{remote.rstrip('/')}/{ds.out_dir}"
        _, rpath = _split_url(rdir)
        local_dir = out_root / ds.out_dir
        if not _fetch_json(host, f"{rpath}/_groups/g{gid:04d}.json", local_dir / "_groups" / f"g{gid:04d}.json"):
            return 0
        n = 1
        st = load_group_status(local_dir, gid)
        for p in (st.parts if st else []):
            name = Path(p["sidecar"]).name
            if _fetch_json(host, f"{rpath}/{name}", local_dir / name):
                n += 1
        return n

    n_files = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for n in ex.map(one, todo):
            n_files += n
    return {"files": n_files, "datasets": len(keys), "probed": len(todo)}
