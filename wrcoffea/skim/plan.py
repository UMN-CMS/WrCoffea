"""Campaign plan: datasets, their input files, and how they are grouped into jobs.

``skim plan`` reads the analysis config JSONs, resolves every DAS path to its
list of files through DBS (sizes, event counts, adler32), samples one file per
DAS path to learn the branch layout, applies the branch policy to build one
:class:`~wrcoffea.skim.schema.OutputSchema` per dataset, and splits the files
into *groups*: the unit of work one worker (or one Condor job) processes into
one or more output parts.  The plan is written as JSON and is the single
source of truth for the whole campaign: every later step (run, status,
validate, upload, fileset) works from it, never from a fresh DAS query.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from wrcoffea.skim.catalog import DBSClient, FileRecord, is_das_path
from wrcoffea.skim.schema import OutputSchema, build_schema, branch_types_of_tree, load_policy
from wrcoffea.skim.selection import SKIM_CUTS

logger = logging.getLogger(__name__)

PLAN_VERSION = 1

# Default group sizes in input bytes; a group is one unit of work.
DEFAULT_GROUP_INPUT_BYTES = {"data": 60e9, "mc": 30e9, "signal": 1e12}

ERA_SUBDIRS = {
    "RunIISummer20UL16APV": "RunII/2016/RunIISummer20UL16APV",
    "RunIISummer20UL16": "RunII/2016/RunIISummer20UL16",
    "RunIISummer20UL17": "RunII/2017/RunIISummer20UL17",
    "RunIISummer20UL18": "RunII/2018/RunIISummer20UL18",
    "Run3Summer22": "Run3/2022/Run3Summer22",
    "Run3Summer22EE": "Run3/2022/Run3Summer22EE",
    "Run3Summer23": "Run3/2023/Run3Summer23",
    "Run3Summer23BPix": "Run3/2023/Run3Summer23BPix",
    "RunIII2024Summer24": "Run3/2024/RunIII2024Summer24",
    "RunIII2025Summer25": "Run3/2025/RunIII2025Summer25",
}


def code_version() -> str:
    """``git describe``-style identifier of the checked-out code (best effort)."""
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty", "--tags"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def infer_category(dataset_name: str, physics_group: str | None, datatype: str) -> str:
    if datatype == "data":
        return "data"
    if (physics_group or "").lower() == "signal" or "WR" in dataset_name:
        return "signals"
    return "backgrounds"


@dataclass
class DatasetPlan:
    key: str
    era: str
    run: str
    year: str
    datatype: str                      # "data" | "mc"
    physics_group: str
    category: str                      # data | backgrounds | signals
    das_paths: list[str]
    metadata: dict
    files: list[dict]                  # FileRecord dicts, LFN-sorted
    groups: list[list[str]]            # LFNs per group
    schema_id: str
    dbs_summary: dict = field(default_factory=dict)
    out_dir: str = ""                  # relative to the campaign root
    notes: list[str] = field(default_factory=list)
    scan: bool = False                 # multi-model sample split per GenModel_* flag after skimming
    bad_inputs: list[str] = field(default_factory=list)   # LFNs known to be corrupt (unreadable); skipped, recorded

    @property
    def is_data(self) -> bool:
        return self.datatype == "data"

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def input_bytes(self) -> int:
        return sum(int(f["size"]) for f in self.files)

    @property
    def input_events(self) -> int:
        return sum(int(f["events"]) for f in self.files)

    def file_by_lfn(self) -> dict[str, dict]:
        return {f["lfn"]: f for f in self.files}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetPlan":
        return cls(**d)


@dataclass
class Plan:
    campaign: str
    created: str
    code_version: str
    cuts: dict
    policy_version: int
    settings: dict
    datasets: dict[str, DatasetPlan]
    schemas: dict[str, dict]
    plan_version: int = PLAN_VERSION

    # ------------------------------------------------------------ persistence
    def save(self, plan_dir: str | Path) -> Path:
        plan_dir = Path(plan_dir)
        plan_dir.mkdir(parents=True, exist_ok=True)
        head = {k: v for k, v in asdict(self).items() if k not in ("datasets", "schemas")}
        head["datasets"] = {k: d.to_dict() for k, d in self.datasets.items()}
        for name, payload in (("plan.json", json.dumps(head, indent=1) + "\n"),
                              ("schemas.json", json.dumps(self.schemas, indent=0) + "\n")):
            tmp = plan_dir / f".{name}.tmp"
            tmp.write_text(payload)
            tmp.replace(plan_dir / name)       # atomic: workers of a running campaign reload the plan per group
        return plan_dir / "plan.json"

    @classmethod
    def load(cls, plan_dir: str | Path) -> "Plan":
        plan_dir = Path(plan_dir)
        if plan_dir.is_file():
            plan_dir = plan_dir.parent
        head = json.loads((plan_dir / "plan.json").read_text())
        schemas = json.loads((plan_dir / "schemas.json").read_text())
        datasets = {k: DatasetPlan.from_dict(d) for k, d in head.pop("datasets").items()}
        return cls(datasets=datasets, schemas=schemas, **head)

    def schema_for(self, key: str) -> OutputSchema:
        return OutputSchema.from_dict(self.schemas[self.datasets[key].schema_id])

    def select(self, *, eras: list[str] | None = None, datasets: list[str] | None = None,
               physics_groups: list[str] | None = None, datatype: str | None = None,
               exclude: list[str] | None = None) -> list[str]:
        """Dataset keys matching the filters (substring match for datasets and exclusions)."""
        keys = []
        for k, d in self.datasets.items():
            if eras and d.era not in eras:
                continue
            if exclude and any(s in k for s in exclude):
                continue
            if physics_groups and d.physics_group.lower() not in [p.lower() for p in physics_groups]:
                continue
            if datatype and d.datatype != datatype:
                continue
            if datasets and not any(s in k for s in datasets):
                continue
            keys.append(k)
        return keys


# ----------------------------------------------------------------------------
# Config loading
# ----------------------------------------------------------------------------

def load_config_entries(config_paths: list[str | Path], *, physics_group: str | None = None,
                        include_optional: bool = False) -> list[tuple[str, dict, Path]]:
    """``[(das_or_key, metadata, config_path)]`` from config JSONs, deduplicated by key.

    Entries flagged ``"optional": true`` (for example the low-mass DY companions
    that the analysis selection never reaches) are skipped unless
    *include_optional* is set.
    """
    out = []
    seen = set()
    for cp in config_paths:
        cp = Path(cp)
        with open(cp) as fh:
            cfg = json.load(fh)
        for key, meta in cfg.items():
            if "no longer available" in str(meta.get("note", "")).lower():
                logger.info("skipping %s (marked unavailable)", key)
                continue
            if meta.get("optional") and not include_optional:
                logger.info("skipping optional %s (use --include-optional)", key)
                continue
            if physics_group and str(meta.get("physics_group", "")).lower() != physics_group.lower():
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append((key, meta, cp))
    return out


def dataset_key_for(key: str, meta: dict) -> str:
    """Output dataset name: the config's ``dataset`` field (data sub-eras share one)."""
    name = meta.get("dataset")
    if name:
        return str(name)
    if is_das_path(key):
        return key.strip("/").split("/")[0]
    return key


# ----------------------------------------------------------------------------
# Branch sampling (cached)
# ----------------------------------------------------------------------------

def _cache_path(cache_dir: Path, das_path: str) -> Path:
    h = hashlib.sha1(das_path.encode()).hexdigest()[:16]
    return cache_dir / f"{h}.json"


def sample_branch_table(das_path: str, files: list[FileRecord], *, cache_dir: Path,
                        redirector: str = "root://cmsxrootd.fnal.gov/", n_files: int = 1,
                        timeout: int = 90) -> list[dict]:
    """Branch tables of up to *n_files* files of a dataset (first and last by LFN), cached."""
    import uproot

    cache = _cache_path(cache_dir, das_path)
    if cache.exists():
        d = json.loads(cache.read_text())
        if d.get("n_files", 0) >= n_files:
            return [{k: tuple(v) for k, v in t.items()} for t in d["tables"]]
    picks = []
    if files:
        if n_files >= len(files):
            picks = list(files)
        elif n_files == 1:
            picks = [files[0]]
        else:
            picks = [files[i] for i in sorted({0, len(files) - 1, len(files) // 2})][:n_files]
    tables = []
    for f in picks:
        url = redirector.rstrip("/") + "/" + f.lfn
        last = None
        for attempt in range(2):
            try:
                with uproot.open(url, timeout=timeout) as fh:
                    tables.append(branch_types_of_tree(fh["Events"]))
                break
            except Exception as e:
                last = e
                time.sleep(2 * (attempt + 1))
        else:
            logger.warning("could not sample %s via %s: %s", f.lfn, redirector, last)
    if not tables and files:
        # Every picked file failed: try a few other files before giving up.
        for f in [x for x in files if x not in picks][:3]:
            try:
                with uproot.open(redirector.rstrip("/") + "/" + f.lfn, timeout=timeout) as fh:
                    tables.append(branch_types_of_tree(fh["Events"]))
                break
            except Exception as e:
                logger.warning("could not sample %s: %s", f.lfn, e)
    if not tables:
        logger.error("no file of %s could be sampled; the worker will derive the schema from the first input", das_path)
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"das_path": das_path, "n_files": n_files,
                                 "tables": [{k: list(v) for k, v in t.items()} for t in tables]}))
    return tables


# ----------------------------------------------------------------------------
# Plan building
# ----------------------------------------------------------------------------

def make_groups(files: list[FileRecord], group_bytes: float) -> list[list[str]]:
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for f in files:
        if cur and cur_bytes + f.size > group_bytes:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.append(f.lfn)
        cur_bytes += f.size
    if cur:
        groups.append(cur)
    return groups


def build_plan(config_paths: list[str | Path], campaign: str, *, out_settings: dict | None = None,
               dbs: DBSClient | None = None, cache_dir: str | Path | None = None,
               physics_group: str | None = None, eras: list[str] | None = None,
               dataset_filter: list[str] | None = None, sample_files: int = 1,
               group_input_bytes: dict | None = None, policy: dict | None = None,
               skip_private: bool = False, include_optional: bool = False) -> Plan:
    """Build a :class:`Plan` from config JSONs (see module docstring)."""
    dbs = dbs or DBSClient()
    policy = policy or load_policy()
    cache_dir = Path(cache_dir) if cache_dir else Path(os.path.expanduser("~/.cache/wrcoffea/branch_tables"))
    group_bytes = dict(DEFAULT_GROUP_INPUT_BYTES)
    group_bytes.update(group_input_bytes or {})

    entries = load_config_entries(config_paths, physics_group=physics_group, include_optional=include_optional)
    # Group config entries by (era, dataset key).
    grouped: dict[tuple[str, str], list[tuple[str, dict, Path]]] = {}
    for key, meta, cp in entries:
        era = str(meta.get("era", ""))
        if eras and era not in eras:
            continue
        dkey = dataset_key_for(key, meta)
        if dataset_filter and not any(s in dkey for s in dataset_filter):
            continue
        grouped.setdefault((era, dkey), []).append((key, meta, cp))

    datasets: dict[str, DatasetPlan] = {}
    schemas: dict[str, dict] = {}
    for (era, dkey), items in grouped.items():
        meta0 = items[0][1]
        datatype = str(meta0.get("datatype", "mc"))
        pg = str(meta0.get("physics_group", ""))
        category = infer_category(dkey, pg, datatype)
        scan = bool(meta0.get("scan", False))
        run = str(meta0.get("run", ""))
        year = str(meta0.get("year", ""))
        files: list[FileRecord] = []
        das_paths: list[str] = []
        tables: list[dict] = []
        notes: list[str] = []
        summary = {"num_file": 0, "num_event": 0, "file_size": 0}
        for key, meta, cp in items:
            if is_das_path(key):
                recs = dbs.files(key, valid_only=True)
                all_recs = dbs.files(key, valid_only=False)
                n_invalid = len(all_recs) - len(recs)
                if n_invalid:
                    notes.append(f"{key}: {n_invalid} file(s) marked invalid in DBS are excluded")
                s = dbs.summary(key)
                for k in summary:
                    summary[k] += int(s.get(k, 0))
                das_paths.append(key)
                files.extend(recs)
                # Scan samples carry different GenModel_* flags in different
                # files, so every file is sampled to get the full set.
                tables.extend(sample_branch_table(key, recs, cache_dir=cache_dir,
                                                  n_files=len(recs) if scan else sample_files))
                logger.info("%s: %d files, %.1f GB, %d events", key.split("/")[1], len(recs),
                            sum(f.size for f in recs) / 1e9, sum(f.events for f in recs))
            else:
                # Private sample: the config must carry explicit file URLs.
                urls = meta.get("files") or []
                if not urls:
                    msg = f"{key}: not a DAS path and no 'files' list in {cp.name}; skipped"
                    if skip_private:
                        logger.warning(msg)
                        notes.append(msg)
                        continue
                    raise ValueError(msg)
                das_paths.append(key)
                files.extend(FileRecord(lfn=u, size=int(meta.get("file_sizes", {}).get(u, 0)), events=0,
                                        adler32="", valid=True, dataset=key) for u in urls)
                notes.append(f"{key}: private sample with {len(urls)} explicit files (no DBS checksums)")
        if not files:
            logger.warning("%s/%s has no files; skipped", era, dkey)
            continue
        files = sorted(files, key=lambda f: f.lfn)
        schema = build_schema(tables, policy) if tables else None
        if schema is None:
            notes.append("no branch sample available; schema will be derived from the first input by the worker")
            sid = "lazy"
        else:
            sid = schema.digest()
            schemas.setdefault(sid, schema.to_dict())
        kind = "signal" if category == "signals" else datatype
        groups = make_groups(files, float(group_bytes.get(kind, group_bytes["mc"])))
        subdir = ERA_SUBDIRS.get(era, f"{run}/{year}/{era}")
        datasets[dkey if dkey not in datasets else f"{dkey}__{era}"] = DatasetPlan(
            key=dkey, era=era, run=run, year=year, datatype=datatype, physics_group=pg,
            category=category, das_paths=das_paths,
            metadata={k: v for k, v in meta0.items() if k not in ("files", "file_sizes")},
            files=[f.to_dict() for f in files], groups=groups, schema_id=sid,
            dbs_summary=summary, out_dir=f"{subdir}/{category}/{dkey}", notes=notes, scan=scan,
        )
    settings = {"group_input_bytes": group_bytes, "sample_files": sample_files}
    settings.update(out_settings or {})
    return Plan(campaign=campaign, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                code_version=code_version(), cuts=dict(SKIM_CUTS),
                policy_version=int(policy.get("version", 0)), settings=settings,
                datasets=datasets, schemas=schemas)


def refresh_metadata(plan: "Plan", config_paths: list[str | Path], *, keys: tuple[str, ...] = (),
                     include_optional: bool = True) -> dict[str, dict[str, tuple]]:
    """Replace each dataset's config metadata in *plan* with the current config entries.

    Files, groups and schemas are untouched (the plan stays the truth for what is skimmed);
    only the metadata that the configs own (cross sections, bookkeeping fields) is refreshed,
    all keys by default or just *keys*.  Returns ``{dataset key: {field: (old, new)}}`` for
    the datasets whose metadata changed.
    """
    fresh: dict[str, dict] = {}
    for das, meta, _ in load_config_entries(config_paths, include_optional=include_optional):
        fresh[das] = meta
    changed: dict[str, dict[str, tuple]] = {}
    for dkey, ds in plan.datasets.items():
        metas = [fresh[d] for d in ds.das_paths if d in fresh]
        if not metas:
            continue
        new_meta = {k: v for k, v in metas[0].items() if k not in ("files", "file_sizes")}
        if keys:
            new_meta = {k: v for k, v in new_meta.items() if k in keys}
        diff = {}
        for k, v in new_meta.items():
            if ds.metadata.get(k) != v:
                diff[k] = (ds.metadata.get(k), v)
                ds.metadata[k] = v
        if diff:
            changed[dkey] = diff
    return changed


def plan_summary(plan: Plan) -> str:
    lines = [f"campaign {plan.campaign}: {len(plan.datasets)} datasets, {len(plan.schemas)} schemas"]
    by_era: dict[str, list[DatasetPlan]] = {}
    for d in plan.datasets.values():
        by_era.setdefault(d.era, []).append(d)
    for era, ds in sorted(by_era.items()):
        nf = sum(d.n_files for d in ds)
        ng = sum(len(d.groups) for d in ds)
        tb = sum(d.input_bytes for d in ds) / 1e12
        ne = sum(d.input_events for d in ds)
        lines.append(f"  {era:22s} {len(ds):4d} datasets {nf:6d} files {ng:5d} groups {ne/1e9:7.2f} Gevt {tb:6.2f} TB")
    return "\n".join(lines)
