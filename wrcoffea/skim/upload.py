"""Upload finished parts to remote storage and write the analyzer filesets.

Uploads use ``xrdcp`` with the part's adler32 (from the sidecar) so a corrupt
transfer fails; a remote file whose size already matches is skipped.  The
sidecar JSON is uploaded next to each part so the provenance travels with the
data.  Filesets are written in the layout the analyzer reads
(``data/filesets/<run>/<year>/<era>/skimmed/<era>_<family>_fileset.json``),
one per config JSON, with ``genEventSumw`` and the theory-weight sums taken
from the skims' own Runs trees rather than from a separate DAS pass.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wrcoffea.skim.plan import Plan, DatasetPlan, load_config_entries, dataset_key_for, ERA_SUBDIRS
from wrcoffea.skim.worker import load_group_status
from wrcoffea.skim.writer import read_sidecar

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# remote helpers
# ---------------------------------------------------------------------------

def _split_root_url(url: str) -> tuple[str, str]:
    """``root://host//path`` -> (host, /path)."""
    assert url.startswith("root://")
    rest = url[len("root://"):]
    host, _, path = rest.partition("/")
    return host, "/" + path.lstrip("/")


def remote_size(url: str, timeout: int = 60) -> int | None:
    host, path = _split_root_url(url)
    try:
        r = subprocess.run(["xrdfs", host, "stat", path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        if line.strip().startswith("Size:"):
            return int(line.split()[1])
    return None


def remote_mkdir(url: str, timeout: int = 60) -> None:
    host, path = _split_root_url(url)
    subprocess.run(["xrdfs", host, "mkdir", "-p", path], capture_output=True, text=True, timeout=timeout)


def xrdcp_upload(local: str, url: str, *, adler32: str | None, streams: int = 8, retries: int = 3,
                 timeout: float = 3600.0) -> tuple[bool, str]:
    cmd = ["xrdcp", "-f", "--nopbar", "--streams", str(streams)]
    if adler32:
        cmd += ["--cksum", f"adler32:{adler32}"]
    cmd += [local, url]
    last = ""
    for attempt in range(1, retries + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = "timeout"
            continue
        if r.returncode == 0:
            return True, ""
        last = (r.stderr or "").strip()[-300:]
        time.sleep(min(60, 5 * 2 ** (attempt - 1)))
    return False, last


def finished_parts(plan: Plan, key: str, out_root: Path, *, allow_bad: bool = False) -> list[dict] | None:
    """Sidecar dicts of all parts of a dataset, or None if it is not complete."""
    ds = plan.datasets[key]
    out_dir = out_root / ds.out_dir
    parts = []
    for gid in range(len(ds.groups)):
        st = load_group_status(out_dir, gid)
        if st is None or st.status != "done":
            return None
        if st.bad_files and not allow_bad:
            from wrcoffea.skim.worker import needs_retry
            if needs_retry(st):
                return None     # transient fetch failures: the group will be redone
            # corrupt / unavailable inputs only: the group is final; the sums of weights come from
            # the parts actually written, so the dataset is normalised correctly without them
        for p in st.parts:
            sc = read_sidecar(p["sidecar"])
            parts.append(sc)
    return parts


def upload_campaign(args) -> int:
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype)
    out_root = Path(args.out).expanduser()
    remote_root = args.remote.rstrip("/")
    jobs: list[tuple[str, str, str | None]] = []
    skipped_datasets = []
    for key in keys:
        ds = plan.datasets[key]
        parts = finished_parts(plan, key, out_root)
        if parts is None:
            skipped_datasets.append(ds.key)
            continue
        rdir = f"{remote_root}/{ds.out_dir}"
        for sc in parts:
            p = sc["part"]
            jobs.append((p["path"], f"{rdir}/{Path(p['path']).name}", p["adler32"]))
            jobs.append((p["sidecar"], f"{rdir}/{Path(p['sidecar']).name}", None))
    print(f"{len(jobs)//2} part(s) from {len(keys) - len(skipped_datasets)} dataset(s) to upload; "
          f"{len(skipped_datasets)} incomplete dataset(s) skipped")
    if args.dry_run:
        for local, url, _ in jobs:
            print(f"  {local} -> {url}")
        return 0
    state_path = out_root / "_upload_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    made_dirs: set[str] = set()

    def _one(job):
        local, url, adler = job
        size = os.path.getsize(local)
        if remote_size(url) == size:
            return url, "exists", ""
        d = url.rsplit("/", 1)[0]
        if d not in made_dirs:
            remote_mkdir(d)
            made_dirs.add(d)
        ok, err = xrdcp_upload(local, url, adler32=adler)
        if ok and remote_size(url) != size:
            ok, err = False, "size mismatch after upload"
        return url, ("ok" if ok else "failed"), err

    n_ok = n_fail = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for url, status, err in ex.map(_one, jobs):
            state[url] = {"status": status, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "error": err}
            if status == "failed":
                n_fail += 1
                print(f"FAILED {url}: {err}", flush=True)
            else:
                n_ok += 1
            state_path.write_text(json.dumps(state, indent=1) + "\n")
    print(f"uploaded/verified {n_ok}, failed {n_fail} in {(time.monotonic() - t0)/60:.1f} min")
    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# filesets
# ---------------------------------------------------------------------------

def _weighted_ratio_mean(parts: list[dict], name: str) -> list[float] | None:
    vals, weights = [], []
    for sc in parts:
        runs = sc["part"].get("runs") or {}
        if name in runs and "genEventSumw" in runs:
            vals.append(runs[name])
            weights.append(float(runs["genEventSumw"]))
    if not vals:
        return None
    n = len(vals[0])
    if any(len(v) != n for v in vals):
        logger.warning("%s: parts disagree on the number of entries; skipping", name)
        return None
    tot = sum(weights)
    if tot == 0:
        return [sum(v[i] for v in vals) / len(vals) for i in range(n)]
    return [sum(w * v[i] for v, w in zip(vals, weights)) / tot for i in range(n)]


def dataset_metadata_from_parts(ds: DatasetPlan, parts: list[dict]) -> dict:
    """Analyzer metadata for one dataset, with sums taken from the skims."""
    meta = {k: v for k, v in ds.metadata.items() if k not in ("files", "file_sizes", "note")}
    meta.setdefault("das_name", ds.das_paths[0] if ds.das_paths else ds.key)
    meta["sample"] = ds.key
    meta.pop("dataset", None)
    meta["skim_campaign"] = parts[0]["campaign"].get("campaign") if parts else None
    meta["nevts"] = sum(int(sc["part"]["n_events"]) for sc in parts)
    if not ds.is_data:
        sumw = sum(float(sc["part"].get("runs", {}).get("genEventSumw", 0.0)) for sc in parts)
        count = sum(int(sc["part"].get("runs", {}).get("genEventCount", 0)) for sc in parts)
        if sumw:
            meta["genEventSumw"] = sumw
        if count:
            meta["genEventCount"] = count
        for name in ("LHEScaleSumw", "LHEPdfSumw", "PSSumw"):
            v = _weighted_ratio_mean(parts, name)
            if v is not None:
                meta[name] = v
    return meta


def _mc_alias() -> dict:
    try:
        from wrcoffea.analysis_config import MC_ALIAS
        return dict(MC_ALIAS)
    except Exception:
        return {}


def write_filesets(args) -> int:
    plan = Plan.load(args.plan_dir)
    out_root = Path(args.out).expanduser()
    remote_root = None if args.local else args.remote.rstrip("/")
    fileset_dir = Path(args.fileset_dir)
    aliases = _mc_alias()
    eras = args.era or sorted({d.era for d in plan.datasets.values()} | set(aliases))
    n_written = 0
    for era in eras:
        sub = ERA_SUBDIRS.get(era)
        if sub is None:
            logger.warning("unknown era %s; skipped", era)
            continue
        cfg_dir = Path(__file__).resolve().parents[2] / "data" / "configs" / sub
        configs = [(era, p) for p in sorted(cfg_dir.glob("*.json")) if "archived" not in p.parts]
        # A data-only era borrows the MC configs (and skims) of its alias era.
        src_era = aliases.get(era)
        if src_era:
            src_dir = Path(__file__).resolve().parents[2] / "data" / "configs" / ERA_SUBDIRS[src_era]
            configs += [(src_era, p) for p in sorted(src_dir.glob("*.json"))
                        if "archived" not in p.parts and not p.name.endswith("_data.json")]
        for cfg_era, cfg in configs:
            family = cfg.stem[len(cfg_era) + 1:] if cfg.stem.startswith(cfg_era + "_") else cfg.stem
            fileset: dict = {}
            missing = []
            for das, meta, _ in load_config_entries([cfg], include_optional=True):
                dkey = dataset_key_for(das, meta)
                ds = plan.datasets.get(dkey)
                if ds is None or ds.era != cfg_era:
                    missing.append(dkey)
                    continue
                if args.dataset and not any(s in dkey for s in args.dataset):
                    continue
                parts = finished_parts(plan, dkey, out_root, allow_bad=False)
                if parts is None:
                    missing.append(dkey)
                    continue
                if any(e.get("metadata", {}).get("sample") == dkey for e in fileset.values()):
                    continue  # data sub-eras share one dataset
                files = {}
                for sc in sorted(parts, key=lambda s: s["part"]["path"]):
                    local = sc["part"]["path"]
                    if remote_root:
                        files[f"{remote_root}/{ds.out_dir}/{Path(local).name}"] = "Events"
                    else:
                        files[local] = "Events"
                meta_out = dataset_metadata_from_parts(ds, parts)
                if cfg_era != era:
                    # Borrowed MC: the analyzer must treat it as the data-only era.
                    meta_out["era"] = era
                    meta_out["year"] = str(plan.datasets[next(k for k, d in plan.datasets.items() if d.era == era)].year) \
                        if any(d.era == era for d in plan.datasets.values()) else meta_out.get("year")
                    meta_out["skims_from_era"] = cfg_era
                fileset[das] = {"files": files, "metadata": meta_out}
            if not fileset:
                logger.warning("%s: no finished datasets; fileset not written", cfg.name)
                continue
            out = fileset_dir / sub / "skimmed" / f"{era}_{family}_fileset.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(fileset, indent=2, sort_keys=True) + "\n")
            n_written += 1
            print(f"wrote {out} ({len(fileset)} datasets{'; missing: ' + ', '.join(missing[:5]) if missing else ''})")
    print(f"{n_written} fileset(s) written")
    return 0
