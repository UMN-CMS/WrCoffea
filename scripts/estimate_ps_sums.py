#!/usr/bin/env python3
"""Estimate the parton-shower weight means (PSSumw) of skimmed filesets whose skims lack them.

The ``ps`` systematic family normalizes each PSWeight variation to its mean over the whole
generated sample, ``PSSumw[i] = sum(genWeight * PSWeight[i]) / sum(genWeight)``.  Run 3
NanoAOD stores it in the Runs tree and the skims carry it over; UL NanoAODv9 has no such
branch, so the UL filesets made from the 2026-10 skims have none and ``--systs ps`` is
refused for UL.  A mean over a few million unselected events pins the number to well below
0.1 %, so this script reads ``genWeight`` and ``PSWeight`` from a few ORIGINAL files of every
dataset (the DAS file list via DBS, through XRootD redirectors), computes the means with
the same accumulator the skimmer uses, and writes them into the fileset metadata as
``PSSumw`` with a ``PSSumw_source`` record (files, events, date).

    scripts/estimate_ps_sums.py --era RunIISummer20UL18 [--files-per-dataset 2] [--workers 8] \\
        [--cache ps_estimate_cache.json] [--dry-run]

Datasets whose metadata already holds a 4-entry ``PSSumw`` are left alone; a dataset whose
PSWeight does not have four entries gets only the ``PSSumw_source`` record (no PS variation
is defined for it).  Every MC fileset of the era is processed (the DY variants share their
backgrounds; each dataset is read once).  A file that cannot be opened is retried on the
next redirector; a dataset that still fails is recorded as failed, the others are written,
and the exit code is 1 so a rerun (which reads the ``--cache`` of finished datasets) can
complete it.  ``bin/skim.py fileset`` overwrites these values: after a fileset regeneration
rerun this script, until a reskim with the event-derived PSSumw makes it unnecessary
(``wrcoffea/skim/weight_sums.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrcoffea.era_utils import era_subdir  # noqa: E402
from wrcoffea.skim.weight_sums import EventWeightSums  # noqa: E402

logger = logging.getLogger("estimate_ps_sums")

REDIRECTORS = ("root://cmsxrootd.fnal.gov/", "root://cms-xrd-global.cern.ch/", "root://xrootd-cms.infn.it/")
BRANCHES = ["genWeight", "PSWeight"]


def pick_files(lfns: list[str], n: int) -> list[str]:
    """*n* files spread evenly over the sorted list (all of them when there are fewer)."""
    lfns = sorted(lfns)
    if len(lfns) <= n:
        return lfns
    step = len(lfns) / n
    return [lfns[int(i * step)] for i in range(n)]


def _open(lfn: str, redirectors: tuple[str, ...], *, timeout: int, retries: int):
    """Open an LFN through the redirectors in turn (a local path or full URL as is)."""
    import uproot
    urls = [lfn] if not lfn.startswith("/store/") else [r.rstrip("/") + "/" + lfn for r in redirectors]
    last: Exception | None = None
    for attempt in range(retries):
        for url in urls:
            try:
                return uproot.open(url, timeout=timeout)
            except Exception as exc:  # noqa: BLE001 - every failure is retried, then reported
                last = exc
        time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"could not open {lfn} through {', '.join(urls)}: {last}")


def ps_means(lfns: list[str], *, redirectors=REDIRECTORS, step: int = 500_000, timeout: int = 180,
             retries: int = 3) -> tuple[list[float], int, int]:
    """``(means, n_events, n_entries)`` of PSWeight over every event of *lfns*."""
    sums = EventWeightSums(["PSSumw"])
    for lfn in lfns:
        with _open(lfn, redirectors, timeout=timeout, retries=retries) as f:
            tree = f["Events"]
            if "PSWeight" not in tree.keys():
                return [], 0, 0
            for arrays in tree.iterate(BRANCHES, step_size=step, library="ak"):
                sums.add(arrays)
    ratios = sums.ratios()["PSSumw"]
    return [float(x) for x in ratios], sums.n_events, len(ratios)


def estimate_dataset(das: str, *, dbs, redirectors=REDIRECTORS, files_per_dataset: int) -> dict:
    """The estimate of one dataset; a failure is returned as a ``status``, never raised."""
    try:
        records = dbs.files(das, valid_only=True)
        if not records:
            return {"status": "no valid files in DBS", "date": time.strftime("%Y-%m-%d")}
        lfns = pick_files([r.lfn for r in records], files_per_dataset)
        means, n_events, n_entries = ps_means(lfns, redirectors=redirectors)
    except Exception as exc:  # noqa: BLE001
        return {"status": f"failed: {exc}", "date": time.strftime("%Y-%m-%d")}
    out = {"method": "mean of genWeight x PSWeight over the events of a few original files",
           "files": len(lfns), "events": n_events, "date": time.strftime("%Y-%m-%d")}
    if n_entries == 4:
        out["PSSumw"] = means
    else:
        out["status"] = f"PSWeight has {n_entries} entries; no 4-entry parton-shower variations for this sample"
    return out


def fileset_paths(fileset_dir: str | Path, era: str) -> list[Path]:
    d = Path(fileset_dir) / era_subdir(era) / "skimmed"
    return sorted(p for p in d.glob(f"{era}_*_fileset.json") if "_data_" not in p.name and "scan" not in p.name)


def _load_cache(path: Path | None) -> dict:
    if path and path.is_file():
        return json.loads(path.read_text())
    return {}


def _save_cache(path: Path | None, cache: dict) -> None:
    if path:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=1) + "\n")
        tmp.replace(path)


def update_era(era: str, *, dbs, fileset_dir: str | Path, redirectors=REDIRECTORS, files_per_dataset: int,
               workers: int, dry_run: bool, cache: dict | None = None, cache_path: Path | None = None) -> dict:
    """Estimate the missing PSSumw of every MC fileset of *era*; returns ``{das: result}`` (the cache).

    Finished datasets (a result with ``PSSumw`` or a definitive ``status``) in *cache* are not
    read again; failed ones are.
    """
    cache = {} if cache is None else cache
    paths = fileset_paths(fileset_dir, era)
    todo: dict[str, None] = {}
    entries: list[tuple[Path, dict, str, dict]] = []
    for path in paths:
        fileset = json.loads(path.read_text())
        for key, spec in fileset.items():
            md = spec.get("metadata", {})
            if md.get("datatype") == "data":
                continue
            das = md.get("das_name") or key
            if len(md.get("PSSumw") or []) == 4 or not isinstance(das, str) or not das.startswith("/"):
                continue
            entries.append((path, fileset, key, md))
            done = cache.get(das)
            if not done or str(done.get("status", "")).startswith("failed"):
                todo[das] = None
    logger.info("%s: %d filesets, %d dataset entries without PSSumw, %d distinct datasets to read (%d from the cache)",
                era, len(paths), len(entries), len(todo), len({md.get("das_name") or k for _, _, k, md in entries}) - len(todo))
    if dry_run:
        return cache
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(estimate_dataset, das, dbs=dbs, redirectors=redirectors,
                               files_per_dataset=files_per_dataset): das for das in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            das = futures[fut]
            result = fut.result()
            cache[das] = result
            _save_cache(cache_path, cache)
            logger.info("  [%d/%d] %s: %s", i, len(futures), das.split("/")[1][:60],
                        f"{result['events']:,} events of {result['files']} files -> {['%.4f' % x for x in result['PSSumw']]}"
                        if "PSSumw" in result else result.get("status"))
    written: dict[Path, dict] = {}
    for path, fileset, key, md in entries:
        res = cache.get(md.get("das_name") or key)
        if not res:
            continue
        if "PSSumw" in res:
            md["PSSumw"] = res["PSSumw"]
        md["PSSumw_source"] = {k: v for k, v in res.items() if k != "PSSumw"}
        written.setdefault(path, fileset)
    for path, fileset in written.items():
        path.write_text(json.dumps(fileset, indent=1) + "\n")
        logger.info("  wrote %s", path)
    failed = sorted(d for d in todo if str(cache.get(d, {}).get("status", "")).startswith("failed"))
    if failed:
        logger.error("%s: %d dataset(s) failed; rerun to complete them: %s", era, len(failed),
                     ", ".join(d.split("/")[1][:40] for d in failed[:8]) + (" ..." if len(failed) > 8 else ""))
    return cache


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", nargs="+", required=True)
    ap.add_argument("--files-per-dataset", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redirectors", nargs="+", default=list(REDIRECTORS), help="tried in this order for every file")
    ap.add_argument("--fileset-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "filesets"))
    ap.add_argument("--cache", default=None, help="JSON of finished datasets; a rerun resumes from it")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from wrcoffea.skim.catalog import DBSClient
    dbs = None if args.dry_run else DBSClient()
    cache_path = Path(args.cache) if args.cache else None
    cache = _load_cache(cache_path)
    rc = 0
    for era in args.era:
        update_era(era, dbs=dbs, fileset_dir=args.fileset_dir, redirectors=tuple(args.redirectors),
                   files_per_dataset=args.files_per_dataset, workers=args.workers, dry_run=args.dry_run,
                   cache=cache, cache_path=cache_path)
        if any(str(v.get("status", "")).startswith("failed") for v in cache.values()):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
