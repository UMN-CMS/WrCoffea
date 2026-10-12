"""Upload finished parts to remote storage and write the analyzer filesets.

Uploads use ``xrdcp`` with the part's adler32 (from the sidecar) so a corrupt
transfer fails; a remote file whose size already matches is skipped.  The
sidecar JSON is uploaded next to each part so the provenance travels with the
data.  Filesets are written in the layout the analyzer reads
(``data/filesets/<run>/<year>/<era>/skimmed/<era>_<family>_fileset.json``),
one per config JSON, with ``genEventSumw`` and the theory-weight sums taken
from the skims' own Runs trees rather than from a separate DAS pass.  The
configs carry no ``genEventSumw``: an MC dataset whose skims have none is an
error.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from wrcoffea.skim.plan import Plan, DatasetPlan, load_config_entries, dataset_key_for, ERA_SUBDIRS, FILE_LIST_KEYS
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
                       physics_groups=args.physics_group or None, datatype=args.datatype,
                       exclude=getattr(args, "exclude", None))
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
    """Analyzer metadata for one dataset, with sums taken from the skims.

    For MC the normalisation is the parts' own summed ``genEventSumw``; a dataset whose
    parts carry none raises ``ValueError``.  A ``genEventSumw`` in the plan metadata
    (copied from configs that used to carry one) is never used.
    """
    meta = {k: v for k, v in ds.metadata.items() if k not in (*FILE_LIST_KEYS, "note", "genEventSumw")}
    meta.setdefault("das_name", ds.das_paths[0] if ds.das_paths else ds.key)
    meta["sample"] = ds.key
    meta.pop("dataset", None)
    meta["skim_campaign"] = parts[0]["campaign"].get("campaign") if parts else None
    meta["nevts"] = sum(int(sc["part"]["n_events"]) for sc in parts)
    if not ds.is_data:
        sumw = sum(float(sc["part"].get("runs", {}).get("genEventSumw", 0.0)) for sc in parts)
        count = sum(int(sc["part"].get("runs", {}).get("genEventCount", 0)) for sc in parts)
        if not sumw:
            raise ValueError(f"{ds.key}: its skimmed parts carry no genEventSumw (no Runs sums in their "
                             "sidecars), so it cannot be normalised. Stopping: re-skim it or fix its sidecars.")
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


def combine_extension_sumw(fileset: dict) -> list[str]:
    """Normalise the datasets of one physics sample together.

    A sample can be split over several fileset entries when its nominal and extension datasets were
    planned under different ``dataset`` names (e.g. ``X`` and ``X_ext1``).  Each entry is then
    weighted by ``xsec / genEventSumw`` on its own, which would count the sample once per dataset.
    MC entries that share a primary dataset name get the combined ``genEventSumw`` (and
    ``genEventCount``) of all of them, so the sample is normalised once; the entry's own sum is kept
    as ``genEventSumw_own``.  Returns one note per combined sample.
    """
    by_primary: dict[str, list[str]] = {}
    for das, entry in fileset.items():
        meta = entry.get("metadata", {})
        if meta.get("datatype") == "data" or not das.startswith("/"):
            continue
        by_primary.setdefault(das.split("/")[1], []).append(das)
    notes = []
    for primary, keys in sorted(by_primary.items()):
        if len(keys) < 2:
            continue
        metas = [fileset[k]["metadata"] for k in keys]
        xsecs = {m.get("xsec") for m in metas}
        if len(xsecs) != 1:
            raise ValueError(f"{primary}: datasets of one sample carry different cross sections {sorted(map(str, xsecs))}")
        total = sum(float(m.get("genEventSumw_own", m.get("genEventSumw")) or 0.0) for m in metas)
        count = sum(int(m.get("genEventCount") or 0) for m in metas)
        samples = sorted(str(m.get("sample")) for m in metas)
        for m in metas:
            m.setdefault("genEventSumw_own", m.get("genEventSumw"))
            m["genEventSumw"] = total
            if count:
                m["genEventCount"] = count
            m["normalised_with"] = samples
        notes.append(f"{primary}: {len(keys)} datasets ({', '.join(samples)}) normalised together, genEventSumw {total:.6g}")
    return notes


def _entry_selected(dkey: str, meta: dict, ds: DatasetPlan | None, *, datasets=None, physics_groups=None,
                    datatype=None, exclude=None) -> bool:
    """The ``--dataset/--physics-group/--datatype/--exclude`` filters of the other subcommands (``Plan.select``),
    applied to one config entry; the plan's dataset decides when it exists, the config entry otherwise."""
    kind = ds.datatype if ds is not None else meta.get("datatype")
    group = ds.physics_group if ds is not None else meta.get("physics_group", "")
    if exclude and any(s in dkey for s in exclude):
        return False
    if physics_groups and str(group).lower() not in [g.lower() for g in physics_groups]:
        return False
    if datatype and kind != datatype:
        return False
    if datasets and not any(s in dkey for s in datasets):
        return False
    return True


def _group_progress(plan: Plan, key: str, out_root: Path) -> str:
    """``"3/5 groups done"`` (or the reason a finished-looking dataset is not final)."""
    ds = plan.datasets[key]
    n_done = sum(1 for gid in range(len(ds.groups))
                 if (st := load_group_status(out_root / ds.out_dir, gid)) is not None and st.status == "done")
    if n_done == len(ds.groups):
        return f"{n_done}/{len(ds.groups)} groups done, transient bad inputs to redo"
    return f"{n_done}/{len(ds.groups)} groups done"


def write_filesets(args) -> int:
    """Write one fileset per config family of every selected era.

    The selection flags (``--dataset``, ``--physics-group``, ``--datatype``, ``--exclude``) pick config
    entries as ``Plan.select`` picks datasets.  Without them each family file is rewritten from the
    campaign; with them only the selected entries of an existing family file are replaced or added,
    and its other entries are kept.  A family whose selected datasets are not all finished is NOT
    written (exit code 1) and every missing dataset is listed, unless ``--allow-missing`` is given;
    a family with nothing finished is skipped with a warning.  Optional config entries the plan left
    out (``plan --include-optional``) are not counted as missing.
    """
    plan = Plan.load(args.plan_dir)
    out_root = Path(args.out).expanduser()
    remote_root = None if args.local else args.remote.rstrip("/")
    fileset_dir = Path(args.fileset_dir)
    config_root = Path(getattr(args, "config_root", None) or Path(__file__).resolve().parents[2] / "data" / "configs")
    allow_missing = bool(getattr(args, "allow_missing", False))
    sel = {"datasets": args.dataset or None, "physics_groups": getattr(args, "physics_group", None) or None,
           "datatype": getattr(args, "datatype", None), "exclude": getattr(args, "exclude", None) or None}
    narrowed = any(v for v in sel.values())
    aliases = _mc_alias()
    eras = args.era or sorted({d.era for d in plan.datasets.values()} | set(aliases))
    n_written, refused, empty = 0, [], []
    for era in eras:
        sub = ERA_SUBDIRS.get(era)
        if sub is None:
            logger.warning("unknown era %s; skipped", era)
            continue
        cfg_dir = config_root / sub
        configs = [(era, p) for p in sorted(cfg_dir.glob("*.json")) if "archived" not in p.parts]
        # A data-only era borrows the MC configs (and skims) of its alias era.
        src_era = aliases.get(era)
        if src_era:
            src_dir = config_root / ERA_SUBDIRS[src_era]
            configs += [(src_era, p) for p in sorted(src_dir.glob("*.json"))
                        if "archived" not in p.parts and not p.name.endswith("_data.json")]
        for cfg_era, cfg in configs:
            family = cfg.stem[len(cfg_era) + 1:] if cfg.stem.startswith(cfg_era + "_") else cfg.stem
            out = fileset_dir / sub / "skimmed" / f"{era}_{family}_fileset.json"
            fileset: dict = {}
            missing: list[str] = []
            n_selected = 0
            for das, meta, _ in load_config_entries([cfg], include_optional=True):
                dkey = dataset_key_for(das, meta)
                ds = plan.datasets.get(dkey)
                if ds is not None and ds.era != cfg_era:
                    ds = None
                if not _entry_selected(dkey, meta, ds, **sel):
                    continue
                n_selected += 1
                if ds is None:
                    if meta.get("optional"):
                        logger.info("%s: optional %s is not in the plan; left out", cfg.name, dkey)
                    else:
                        missing.append(f"{dkey} (not in the plan)")
                    continue
                targets = [(das, ds)]
                if ds.scan:
                    # A multi-model scan sample is analysed as its per-model datasets (``skim.py split-scan``),
                    # keyed by the model name as the private-grid filesets always were.
                    models = sorted((k, d) for k, d in plan.datasets.items()
                                    if d.metadata.get("scan_source") == dkey and d.era == cfg_era)
                    if not models:
                        missing.append(f"{dkey} (scan not split)")
                        continue
                    targets = [(d.metadata.get("das_name", k), d) for k, d in models]
                for das_t, ds_t in targets:
                    parts = finished_parts(plan, ds_t.key, out_root, allow_bad=False)
                    if parts is None:
                        missing.append(f"{ds_t.key} ({_group_progress(plan, ds_t.key, out_root)})")
                        continue
                    if any(e.get("metadata", {}).get("sample") == ds_t.key for e in fileset.values()):
                        continue  # data sub-eras share one dataset
                    files = {}
                    for sc in sorted(parts, key=lambda s: s["part"]["path"]):
                        local = sc["part"]["path"]
                        if remote_root:
                            files[f"{remote_root}/{ds_t.out_dir}/{Path(local).name}"] = "Events"
                        else:
                            files[local] = "Events"
                    meta_out = dataset_metadata_from_parts(ds_t, parts)
                    if cfg_era != era:
                        # Borrowed MC: the analyzer must treat it as the data-only era.
                        meta_out["era"] = era
                        meta_out["year"] = str(plan.datasets[next(k for k, d in plan.datasets.items() if d.era == era)].year) \
                            if any(d.era == era for d in plan.datasets.values()) else meta_out.get("year")
                        meta_out["skims_from_era"] = cfg_era
                    fileset[das_t] = {"files": files, "metadata": meta_out}
            if not n_selected:
                continue
            if not fileset:
                empty.append(out.name)
                logger.warning("%s: none of the %d selected dataset(s) is finished; %s not written",
                               cfg.name, n_selected, out.name)
                continue
            if missing:
                listing = "\n    ".join(missing)
                if not allow_missing:
                    refused.append(out.name)
                    logger.error("%s: %d of the selected dataset(s) are not finished; %s NOT written "
                                 "(--allow-missing writes it without them):\n    %s",
                                 cfg.name, len(missing), out.name, listing)
                    continue
                logger.warning("%s: written WITHOUT %d unfinished dataset(s) (--allow-missing):\n    %s",
                               out.name, len(missing), listing)
            if narrowed and out.is_file():
                merged = json.loads(out.read_text())
                kept = len([k for k in merged if k not in fileset])
                merged.update(fileset)
                fileset = merged
                logger.info("%s: %d selected entr(ies) replaced or added, %d other entr(ies) kept",
                            out.name, len(fileset) - kept, kept)
            for note in combine_extension_sumw(fileset):
                logger.info("%s: %s", cfg.name, note)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_text(json.dumps(fileset, indent=2, sort_keys=True) + "\n")
            tmp.replace(out)
            n_written += 1
            print(f"wrote {out} ({len(fileset)} datasets" + (f"; {len(missing)} unfinished left out" if missing else "") + ")")
    print(f"{n_written} fileset(s) written"
          + (f"; {len(refused)} refused because datasets are unfinished: {', '.join(refused)}" if refused else "")
          + (f"; {len(empty)} with nothing finished: {', '.join(empty)}" if empty else ""))
    return 1 if refused else 0
