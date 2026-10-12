"""Unskimmed filesets: the original NanoAOD files of every dataset of an era.

``bin/skim.py unskimmed-fileset --era <era> ...`` reads the era's configs
(``data/configs/<run>/<year>/<era>/<era>_<family>.json``), lists the valid
files of every DAS dataset through DBS (the client the skim plan uses; a grid
proxy is all it needs) and writes one fileset per config family to
``data/filesets/<run>/<year>/<era>/unskimmed/<era>_<family>_fileset.json``,
in the layout ``bin/run_analysis.py --unskimmed`` reads:

    {<dataset key>: {"files": {<url>: "Events", ...}, "metadata": {...}}, ...}

- ``files``: ``<redirector>/<lfn>`` (default ``root://cmsxrootd.fnal.gov/``),
  sorted.  A private sample that lists its files in the config (the 2024
  signal) keeps those URLs.
- ``metadata``: the config entry without its file bookkeeping and note, plus
  ``sample`` (the short dataset name), ``nevts`` and ``nfiles`` from DBS and a
  ``listing`` record (source, date, redirector).  No ``genEventSumw``: in
  ``--unskimmed`` mode the analyzer sums ``genWeight`` over the files itself.
- ``--runs-sums`` adds ``genEventSumw``, ``genEventCount`` and the ratio-valued
  theory-weight sums from every file's ``Runs`` tree (one read per file, so
  slow for a whole era), which the theory-weight systematics need.
- An ``optional`` config entry without valid files in DBS is skipped with a
  warning; a required one stops the run.
- The ``signal_scan`` families (multi-model samples, split per mass point only
  when skimmed) are left out.

Rerunning overwrites the files; the output is deterministic for a given DBS
state, so a diff shows what changed in DAS.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import awkward as ak

from wrcoffea.era_utils import get_era_details
from wrcoffea.skim.plan import FILE_LIST_KEYS
from wrcoffea.skim.runs import (collapse_runs, collapse_scan_runs, concat_runs, is_scan_runs, pad_ratio_branches,
                                pad_scan_runs)

logger = logging.getLogger(__name__)

DEFAULT_REDIRECTOR = "root://cmsxrootd.fnal.gov/"
SKIPPED_FAMILIES = frozenset({"signal_scan"})
# config keys that do not belong in a fileset's metadata (as for the skimmed filesets)
DROPPED_KEYS = frozenset(FILE_LIST_KEYS) | {"note", "genEventSumw"}
TREE = "Events"


def era_dir(root: str | Path, era: str) -> Path:
    run, year, _ = get_era_details(era)
    return Path(root) / run / year / era


def config_files(config_root: str | Path, era: str, families: Iterable[str] | None = None) -> dict[str, Path]:
    """``{family: config path}`` of *era*, in name order; ``signal_scan`` left out unless asked for."""
    directory = era_dir(config_root, era)
    if not directory.is_dir():
        raise FileNotFoundError(f"no configs for {era}: {directory}")
    wanted = set(families) if families else None
    out: dict[str, Path] = {}
    for path in sorted(directory.glob(f"{era}_*.json")):
        family = path.stem[len(era) + 1:]
        if wanted is None and family in SKIPPED_FAMILIES:
            continue
        if wanted is not None and family not in wanted:
            continue
        out[family] = path
    if wanted:
        missing = sorted(wanted - set(out))
        if missing:
            raise FileNotFoundError(f"{era}: no config for famil{'y' if len(missing) == 1 else 'ies'} {', '.join(missing)}")
    return out


def url_for(lfn: str, redirector: str) -> str:
    """``root://host//store/...`` for a ``/store`` LFN; URLs and local paths pass through."""
    if not redirector or lfn.startswith(("root://", "file:", "http://", "https://")) or not lfn.startswith("/store/"):
        return lfn
    return redirector.rstrip("/") + "/" + lfn


def _runs_table(url: str, *, timeout: int, retries: int) -> dict[str, ak.Array]:
    import uproot
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with uproot.open(url, timeout=timeout) as f:
                if "Runs" not in f:
                    return {}
                arr = f["Runs"].arrays(library="ak")
                return {k: arr[k] for k in arr.fields}
        except Exception as exc:  # noqa: BLE001 - retried, then reported with the file
            last = exc
            if attempt < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not read the Runs tree of {url}: {last}")


def runs_sums(urls: Sequence[str], *, workers: int = 16, timeout: int = 120, retries: int = 2) -> dict:
    """``genEventSumw``, ``genEventCount`` and the ratio-valued sums over the files' Runs trees.

    The per-file tables are collapsed exactly as a skim part's are
    (:mod:`wrcoffea.skim.runs`); data files (no sums) give ``{}``.
    """
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        tables = [t for t in pool.map(lambda u: _runs_table(u, timeout=timeout, retries=retries), urls) if t]
    if not tables:
        return {}
    if any(is_scan_runs(t) for t in tables):
        collapsed = collapse_scan_runs(concat_runs(pad_scan_runs(tables)))
    else:
        collapsed = collapse_runs(concat_runs(pad_ratio_branches(tables)))
    out: dict = {}
    for name in ("genEventSumw", "genEventSumw_"):
        if name in collapsed:
            out["genEventSumw"] = float(collapsed[name][0])
            break
    for name in ("genEventCount", "genEventCount_"):
        if name in collapsed:
            out["genEventCount"] = int(collapsed[name][0])
            break
    for name in ("LHEScaleSumw", "LHEPdfSumw", "PSSumw"):
        if name in collapsed:
            out[name] = [float(x) for x in ak.to_list(collapsed[name][0])]
    return out


def dataset_entry(key: str, cfg: Mapping, *, dbs, redirector: str, today: str,
                  runs: bool = False, workers: int = 16) -> dict | None:
    """``{"files": ..., "metadata": ...}`` of one config entry, or None for an optional dataset without files."""
    das = cfg.get("das_name") or key
    listing: dict = {"date": today}
    if cfg.get("files"):
        urls = sorted(cfg["files"])
        events = cfg.get("file_events") or {}
        nevts = int(sum(events.values())) if events else None
        listing.update(source="files listed in the config")
    elif isinstance(das, str) and das.startswith("/"):
        records = dbs.files(das, valid_only=True)
        if not records:
            if cfg.get("optional"):
                logger.warning("%s: no valid files in DBS; optional, skipped", das)
                return None
            raise ValueError(f"{das}: no valid files in DBS and the entry is not optional")
        urls = sorted(url_for(r.lfn, redirector) for r in records)
        nevts = int(sum(r.events for r in records))
        listing.update(source="DBS valid files", redirector=redirector)
    else:
        raise ValueError(f"{key}: neither a DAS path nor a config with a file list")
    metadata = {k: v for k, v in cfg.items() if k not in DROPPED_KEYS}
    metadata.setdefault("das_name", das)
    metadata["sample"] = cfg.get("dataset") or (das.split("/")[1] if das.startswith("/") else key)
    metadata.pop("dataset", None)          # coffea's preprocess refuses a "dataset" metadata key
    metadata["nfiles"] = len(urls)
    if nevts is not None:
        metadata["nevts"] = nevts
    metadata["listing"] = listing
    if runs and cfg.get("datatype", "mc") != "data":
        metadata.update(runs_sums(urls, workers=workers))
    return {"files": {u: TREE for u in urls}, "metadata": metadata}


def build_family(config_path: str | Path, *, dbs, redirector: str = DEFAULT_REDIRECTOR, runs: bool = False,
                 workers: int = 16, today: str | None = None) -> tuple[dict, list[str]]:
    """``(fileset, skipped dataset keys)`` of one config file."""
    today = today or time.strftime("%Y-%m-%d")
    config = json.loads(Path(config_path).read_text())
    fileset: dict = {}
    skipped: list[str] = []
    for key, cfg in config.items():
        entry = dataset_entry(key, cfg, dbs=dbs, redirector=redirector, today=today, runs=runs, workers=workers)
        if entry is None:
            skipped.append(key)
            continue
        fileset[key] = entry
    return fileset, skipped


def fileset_path(out_root: str | Path, era: str, family: str) -> Path:
    return era_dir(out_root, era) / "unskimmed" / f"{era}_{family}_fileset.json"


def write_era_filesets(era: str, *, config_root: str | Path, out_root: str | Path, dbs,
                       families: Iterable[str] | None = None, redirector: str = DEFAULT_REDIRECTOR,
                       runs: bool = False, workers: int = 16, dry_run: bool = False,
                       today: str | None = None) -> list[Path]:
    """Write the unskimmed fileset of every config family of *era*; returns the paths."""
    written: list[Path] = []
    for family, config_path in config_files(config_root, era, families).items():
        path = fileset_path(out_root, era, family)
        if dry_run:
            logger.info("%s %s: %s -> %s", era, family, config_path, path)
            written.append(path)
            continue
        fileset, skipped = build_family(config_path, dbs=dbs, redirector=redirector, runs=runs, workers=workers,
                                        today=today)
        if not fileset:
            logger.warning("%s %s: no dataset with files; nothing written", era, family)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fileset, indent=1) + "\n")
        n_files = sum(len(v["files"]) for v in fileset.values())
        n_evts = sum(int(v["metadata"].get("nevts") or 0) for v in fileset.values())
        logger.info("%s %s: %d datasets, %d files, %s events -> %s%s", era, family, len(fileset), n_files,
                    f"{n_evts:,}" if n_evts else "?", path,
                    f" (skipped optional: {', '.join(k.split('/')[1] if k.startswith('/') else k for k in skipped)})" if skipped else "")
        written.append(path)
    return written


def eras_with_configs(config_root: str | Path) -> list[str]:
    """Every era that has a config directory, from the directory layout."""
    root = Path(config_root)
    return sorted(p.name for p in root.glob("*/*/*") if p.is_dir() and any(p.glob(f"{p.name}_*.json")))
