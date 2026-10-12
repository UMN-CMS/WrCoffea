#!/usr/bin/env python3
"""Skim campaign driver.

    skim.py plan              build the campaign plan from config JSONs (DBS + branch sampling)
    skim.py run               process plan groups locally with N workers (fetch, skim, verify)
    skim.py group             process one group (the unit a Condor job runs)
    skim.py refresh-metadata  copy the current config metadata (xsec etc.) into an existing plan
    skim.py condor            write (and with --submit, submit) one Condor job per pending group
    skim.py ship              copy one finished group to the remote campaign root (Condor job side)
    skim.py mirror            pull the sidecars and group statuses of a remote campaign into a local root
    skim.py bad-files         list bad inputs; --classify marks files without any replica as unavailable
    skim.py status            summarize progress and problems from the group status files
    skim.py validate          whole-campaign validation against the plan, DBS and the golden JSON
    skim.py upload            copy finished parts to remote storage with checksum verification
    skim.py split-scan        split skimmed multi-model scan samples into one dataset per mass point
    skim.py fileset           write the analyzer fileset JSONs from the plan and sidecars
    skim.py cuts              print the skim selection and branch policy

See docs/skimming.md for the workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("skim")

CONFIGS_ROOT = REPO_ROOT / "data" / "configs"
DEFAULT_OUT = Path(os.path.expanduser("~/work/skims_v2"))
DEFAULT_REMOTE = "root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2"
DEFAULT_SCRATCH = Path(os.path.expanduser("~/work/skim_scratch"))


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _configs_for(args) -> list[Path]:
    from wrcoffea.skim.plan import ERA_SUBDIRS
    paths: list[Path] = [Path(c) for c in (args.config or [])]
    for era in args.era or []:
        sub = ERA_SUBDIRS.get(era)
        if sub is None:
            raise SystemExit(f"unknown era {era!r}; known: {sorted(ERA_SUBDIRS)}")
        cfg_dir = CONFIGS_ROOT / sub
        found = sorted(p for p in cfg_dir.glob("*.json") if "archived" not in p.parts)
        if not found:
            raise SystemExit(f"no config JSONs under {cfg_dir}")
        paths.extend(found)
    if not paths:
        raise SystemExit("give --config JSON... or --era ERA...")
    return paths


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def cmd_plan(args) -> int:
    from wrcoffea.skim.plan import build_plan, plan_summary
    group_bytes = {}
    for item in args.group_gb or []:
        k, _, v = item.partition("=")
        group_bytes[k.strip()] = float(v) * 1e9
    settings = {"compression": args.compression, "flush_events": args.flush_events,
                "part_max_bytes": args.part_max_gb * 1e9}
    plan = build_plan(_configs_for(args), args.campaign, out_settings=settings,
                      physics_group=args.physics_group, eras=args.era or None,
                      dataset_filter=args.dataset or None, sample_files=args.sample_files,
                      group_input_bytes=group_bytes, skip_private=args.skip_private,
                      include_optional=args.include_optional)
    path = plan.save(args.plan_dir)
    print(plan_summary(plan))
    print(f"plan written to {path}")
    for d in plan.datasets.values():
        for n in d.notes:
            print(f"  note: {n}")
    return 0


# ---------------------------------------------------------------------------
# run / group
# ---------------------------------------------------------------------------

def _pending_groups(plan, out_root: Path, keys: list[str], *, force: bool, include_failed: bool = True,
                    retry_bad: bool = True):
    from wrcoffea.skim.worker import pending_groups
    return pending_groups(plan, out_root, keys, force=force, include_failed=include_failed, retry_bad=retry_bad,
                          log=lambda m: print(m, flush=True))


def _run_one(task) -> dict:
    """Pool task: process one group in its own process with a per-group log file."""
    plan_dir, key, gid, opts = task
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.worker import process_group
    plan = Plan.load(plan_dir)
    ds = plan.datasets[key]
    out_root = Path(opts["out_root"])
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{ds.key}_g{gid:04d}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    def _on_term(signum, frame):
        # Stop promptly: kill the transfer in flight, then unwind process_group, which
        # records the group as failed and removes its partial parts and scratch files.
        from wrcoffea.skim import fetch
        from wrcoffea.skim.worker import GroupInterrupted
        fetch.STOP.set()
        fetch.abort_active()
        raise GroupInterrupted(f"worker terminated by signal {signum}")

    signal.signal(signal.SIGTERM, _on_term)
    try:
        st = process_group(plan, key, gid, out_root=out_root, scratch=opts["scratch"],
                           rucio_account=opts.get("rucio_account"), keep_inputs=opts.get("keep_inputs", False),
                           chunk_entries=opts.get("chunk_entries", 20_000), force=opts.get("force", False),
                           streams=opts.get("streams", 4), compression=opts.get("compression"),
                           decompression_threads=opts.get("decompression_threads", 1),
                           part_max_bytes=opts.get("part_max_bytes"))
        return {"key": key, "gid": gid, "status": st.status, "events_in": st.events_in, "events_out": st.events_out,
                "parts": len(st.parts), "bad": len(st.bad_files), "seconds": st.seconds, "error": st.error,
                "log": str(log_path)}
    except Exception as e:   # e.g. the group is already being processed by another worker
        logging.getLogger("skim").error("%s g%04d: %s", key, gid, e)
        return {"key": key, "gid": gid, "status": "failed", "events_in": 0, "events_out": 0, "parts": 0, "bad": 0,
                "seconds": 0.0, "error": f"{type(e).__name__}: {e}", "log": str(log_path)}
    finally:
        fh.close()


def cmd_run(args) -> int:
    from wrcoffea.skim.plan import Plan
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype, exclude=getattr(args, 'exclude', None))
    if not keys:
        raise SystemExit("no datasets match the filters")
    out_root = Path(args.out).expanduser()
    todo = _pending_groups(plan, out_root, keys, force=args.force, retry_bad=not args.keep_bad)
    if args.limit_groups:
        todo = todo[: args.limit_groups]
    n_files = sum(len(plan.datasets[k].groups[g]) for k, g in todo)
    gb = sum(sum(plan.datasets[k].file_by_lfn()[l]["size"] for l in plan.datasets[k].groups[g]) for k, g in todo) / 1e9
    print(f"{len(keys)} dataset(s), {len(todo)} group(s) to run, {n_files} input files, {gb:.1f} GB to read")
    if args.dry_run:
        for k, g in todo:
            print(f"  {k} g{g:04d}: {len(plan.datasets[k].groups[g])} files")
        return 0
    if not todo:
        print("nothing to do")
        return 0
    opts = {"out_root": str(out_root), "scratch": str(Path(args.scratch).expanduser()),
            "rucio_account": args.rucio_account, "keep_inputs": args.keep_inputs,
            "chunk_entries": args.chunk_entries, "force": args.force, "streams": args.streams,
            "compression": args.compression, "decompression_threads": args.decompression_threads,
            "part_max_bytes": args.part_max_gb * 1e9 if args.part_max_gb else None}
    tasks = [(str(args.plan_dir), k, g, opts) for k, g in todo]
    t0 = time.monotonic()
    results = []
    ctx = mp.get_context("spawn")

    def _on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _on_term)   # pkill/kill -TERM stops the workers too, not just this process
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        try:
            for i, res in enumerate(pool.imap_unordered(_run_one, tasks), 1):
                results.append(res)
                flag = "OK " if res["status"] == "done" else "FAIL"
                extra = f" bad_files={res['bad']}" if res["bad"] else ""
                print(f"[{i}/{len(tasks)}] {flag} {res['key']} g{res['gid']:04d}: {res['events_in']} -> {res['events_out']} "
                      f"events, {res['parts']} part(s), {res['seconds']/60:.1f} min{extra}", flush=True)
                if res["error"]:
                    print(f"      error: {res['error'].splitlines()[0][:200]}  (log: {res['log']})", flush=True)
        except KeyboardInterrupt as e:
            print(f"interrupted ({e or 'Ctrl-C'}); terminating workers (groups in flight are recorded as failed "
                  f"and redone by the next run)", flush=True)
            pool.terminate()
            pool.join()
            return 143 if str(e).startswith("signal") else 130
    n_ok = sum(r["status"] == "done" for r in results)
    n_bad = sum(r["bad"] for r in results)
    print(f"done: {n_ok}/{len(results)} groups succeeded, {n_bad} bad input file(s), "
          f"{(time.monotonic() - t0)/60:.1f} min")
    return 0 if n_ok == len(results) and n_bad == 0 else 1


def cmd_group(args) -> int:
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.worker import process_group
    plan = Plan.load(args.plan_dir)
    st = process_group(plan, args.dataset, args.gid, out_root=Path(args.out).expanduser(),
                       scratch=Path(args.scratch).expanduser(), rucio_account=args.rucio_account,
                       keep_inputs=args.keep_inputs, chunk_entries=args.chunk_entries, force=args.force,
                       streams=args.streams, compression=args.compression,
                       decompression_threads=args.decompression_threads,
                       part_max_bytes=args.part_max_gb * 1e9 if args.part_max_gb else None)
    print(json.dumps({k: v for k, v in st.to_dict().items() if k != "parts"}, indent=1))
    return 0 if st.status == "done" else 1     # bad inputs are recorded in the status; `run`/`condor` decide retries


# ---------------------------------------------------------------------------
# refresh-metadata / condor / ship / mirror / bad-files / status
# ---------------------------------------------------------------------------

def cmd_refresh_metadata(args) -> int:
    """Copy the current config metadata (cross sections etc.) into an existing plan."""
    from wrcoffea.skim.plan import Plan, ERA_SUBDIRS, refresh_metadata
    plan = Plan.load(args.plan_dir)
    cfgs = list(args.config or [])
    if not cfgs:
        for era in sorted({ds.era for ds in plan.datasets.values()}):
            sub = ERA_SUBDIRS.get(era)
            if sub is None:
                print(f"warning: no config directory known for era {era}", file=sys.stderr)
                continue
            cfgs += [str(p) for p in sorted((REPO_ROOT / "data" / "configs" / sub).glob("*.json")) if "archived" not in p.parts]
    changed = refresh_metadata(plan, cfgs, keys=tuple(args.key or ()))
    for dkey, diff in sorted(changed.items()):
        for k, (old, new) in diff.items():
            o = old if not isinstance(old, dict) else "{...}"
            n = new if not isinstance(new, dict) else "{...}"
            print(f"  {dkey}: {k}: {o} -> {n}")
    if changed and not args.dry_run:
        plan.save(args.plan_dir)
    print(f"{len(changed)} dataset(s) with changed metadata{' (dry run, plan not written)' if args.dry_run else ''}")
    return 0


def cmd_condor(args) -> int:
    """Prepare (and optionally submit) one Condor job per pending group."""
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.condor import prepare_jobs
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype, exclude=getattr(args, 'exclude', None))
    if not keys:
        raise SystemExit("no datasets match the filters")
    proxy = args.proxy or os.environ.get("X509_USER_PROXY") or f"/tmp/x509up_u{os.getuid()}"
    from wrcoffea.skim.condor import DEFAULT_IMAGE
    args.image = args.image or DEFAULT_IMAGE
    summary = prepare_jobs(plan, keys, out_root=Path(args.out).expanduser(), remote=args.remote,
                           jobs_dir=Path(args.jobs_dir).expanduser(), repo_root=REPO_ROOT, proxy=proxy,
                           image=args.image, cpus=args.cpus, memory_mb=args.memory_mb, disk_gb=args.disk_gb,
                           streams=args.streams, decompression_threads=args.decompression_threads,
                           max_hours=args.max_hours, avoid_hosts=tuple(args.avoid_hosts or ()),
                           max_groups=args.max_groups, force=args.force,
                           retry_bad=not args.keep_bad, skip_inflight=not args.ignore_queue,
                           inflight_text=Path(args.inflight_file).read_text() if args.inflight_file else None)
    print(f"{summary['groups']} job(s) for {summary['datasets']} dataset(s), {summary['input_files']} input files, "
          f"{summary['input_gb']} GB to read; {summary['skipped_inflight']} group(s) already queued/running skipped")
    print(f"submission written to {summary['jobs_dir']}")
    if summary["groups"] == 0:
        return 0
    if not args.submit:
        print(f"submit with: cd {summary['jobs_dir']} && condor_submit submit.jdl")
        return 0
    import subprocess
    proc = subprocess.run(["condor_submit", "submit.jdl"], cwd=summary["jobs_dir"], capture_output=True, text=True)
    print(proc.stdout.strip() or proc.stderr.strip())
    return proc.returncode


def cmd_ship(args) -> int:
    """Copy one finished group (parts, sidecars, then the status) to the remote campaign root."""
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.condor import ship_group
    plan = Plan.load(args.plan_dir)
    res = ship_group(plan, args.dataset, args.gid, out_root=Path(args.out).expanduser(), remote=args.remote)
    print(json.dumps(res))
    return 0 if res["status"] == "done" else 1


def cmd_mirror(args) -> int:
    """Pull the JSON metadata (sidecars, group statuses) of a remote campaign tree into a local root."""
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.condor import mirror_metadata
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype, exclude=getattr(args, 'exclude', None))
    res = mirror_metadata(plan, keys, out_root=Path(args.out).expanduser(), remote=args.remote)
    print(f"mirrored {res['files']} JSON file(s) for {res['datasets']} dataset(s) into {Path(args.out).expanduser()}")
    return 0


def cmd_bad_files(args) -> int:
    """List the bad inputs of a campaign; with --classify, mark the ones without any replica as unavailable."""
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.worker import load_group_status, classify_unreachable
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype, exclude=args.exclude)
    out_root = Path(args.out).expanduser()
    if args.mark_corrupt:
        from wrcoffea.skim.worker import write_group_status
        for lfn in args.mark_corrupt:
            hits = [(k, g) for k in keys for g, lfns in enumerate(plan.datasets[k].groups) if lfn in lfns]
            if not hits:
                print(f"  {lfn}: not in any selected dataset")
                continue
            for key, gid in hits:
                out_dir = out_root / plan.datasets[key].out_dir
                st = load_group_status(out_dir, gid)
                if st is None:
                    print(f"  {key} g{gid:04d}: no status yet; nothing to mark (the job records corrupt inputs itself)")
                    continue
                if any(b["lfn"] == lfn and b.get("category") == "corrupt" for b in st.bad_files):
                    print(f"  {key} g{gid:04d}: {lfn} already marked corrupt")
                    continue
                st.bad_files = [b for b in st.bad_files if b["lfn"] != lfn]
                st.bad_files.append({"lfn": lfn, "category": "corrupt",
                                     "error": args.reason or "marked corrupt by hand (verified unreadable)"})
                write_group_status(out_dir, st)
                print(f"  {key} g{gid:04d}: {lfn} marked corrupt (status {st.status})")
    if args.classify:
        from wrcoffea.skim.catalog import RucioClient
        res = classify_unreachable(plan, out_root, keys, RucioClient(args.rucio_account))
        for k, v in sorted(res.items()):
            print(f"  {k}: {v['unavailable']} file(s) with no replica marked unavailable ({v['events']} events), "
                  f"{v['still_unreachable']} still unreachable (retried)")
    tally: dict[str, dict] = {}
    for key in keys:
        ds = plan.datasets[key]
        ev = {f["lfn"]: int(f.get("events", 0)) for f in ds.files}
        for gid in range(len(ds.groups)):
            st = load_group_status(out_root / ds.out_dir, gid)
            for b in (st.bad_files if st else []):
                d = tally.setdefault(key, {})
                c = b.get("category", "unreachable")
                d.setdefault(c, [0, 0])
                d[c][0] += 1
                d[c][1] += ev.get(b["lfn"], 0)
    for key, cats in sorted(tally.items()):
        total = sum(int(f.get("events", 0)) for f in plan.datasets[key].files)
        parts = ", ".join(f"{c}: {n} file(s), {e} events ({100 * e / max(total, 1):.1f} %)" for c, (n, e) in sorted(cats.items()))
        print(f"{key}: {parts}")
    if not tally:
        print("no bad inputs recorded")
    return 0


def cmd_status(args) -> int:
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.worker import load_group_status
    plan = Plan.load(args.plan_dir)
    keys = plan.select(eras=args.era or None, datasets=args.dataset or None,
                       physics_groups=args.physics_group or None, datatype=args.datatype, exclude=getattr(args, 'exclude', None))
    out_root = Path(args.out).expanduser()
    tot = {"groups": 0, "done": 0, "failed": 0, "running": 0, "pending": 0, "bad_files": 0,
           "events_in": 0, "events_out": 0, "bytes": 0}
    rows = []
    for key in keys:
        ds = plan.datasets[key]
        out_dir = out_root / ds.out_dir
        c = {"done": 0, "failed": 0, "running": 0, "pending": 0, "bad": 0, "ein": 0, "eout": 0, "bytes": 0, "parts": 0}
        for gid in range(len(ds.groups)):
            st = load_group_status(out_dir, gid)
            if st is None:
                c["pending"] += 1
                continue
            c[st.status if st.status in c else "pending"] += 1
            c["bad"] += len(st.bad_files)
            if st.status == "done":
                c["ein"] += st.events_in
                c["eout"] += st.events_out
                c["bytes"] += sum(p["size"] for p in st.parts)
                c["parts"] += len(st.parts)
            if args.verbose and (st.status != "done" or st.bad_files):
                rows.append(f"    g{gid:04d} {st.status}{' bad=' + str(len(st.bad_files)) if st.bad_files else ''}"
                            f"{(' ' + st.error.splitlines()[0][:120]) if st.error else ''}")
        n = len(ds.groups)
        state = "DONE" if c["done"] == n and c["bad"] == 0 else ("done+bad" if c["done"] == n else
                 ("FAILED" if c["failed"] else ("running" if c["running"] else ("partial" if c["done"] else "pending"))))
        frac = f"{100.0 * c['eout'] / c['ein']:.2f}%" if c["ein"] else "-"
        print(f"{ds.key[:60]:60s} {ds.era:20s} {c['done']:4d}/{n:<4d} {state:9s} {c['eout']:>11d}/{c['ein']:<12d} {frac:>7s} "
              f"{c['bytes']/1e9:7.2f} GB {c['parts']:4d} parts {('bad=' + str(c['bad'])) if c['bad'] else ''}")
        for r in rows:
            print(r)
        rows = []
        for k in ("done", "failed", "running", "pending"):
            tot[k] += c[k]
        tot["groups"] += n
        tot["bad_files"] += c["bad"]
        tot["events_in"] += c["ein"]
        tot["events_out"] += c["eout"]
        tot["bytes"] += c["bytes"]
    print(f"\ntotal: {tot['done']}/{tot['groups']} groups done, {tot['failed']} failed, {tot['running']} running, "
          f"{tot['pending']} pending; {tot['bad_files']} bad input files (see bad-files); "
          f"{tot['events_out']}/{tot['events_in']} events kept; {tot['bytes']/1e9:.1f} GB written")
    return 0


# ---------------------------------------------------------------------------
# cuts
# ---------------------------------------------------------------------------

def cmd_cuts(args) -> int:
    from wrcoffea.skim.selection import SKIM_CUTS, describe_cuts
    from wrcoffea.skim.schema import load_policy
    print("Skim selection:")
    for k, v in SKIM_CUTS.items():
        print(f"  {k:24s} {v:g}")
    print(f"  => {describe_cuts()}")
    pol = load_policy()
    print(f"\nBranch policy v{pol.get('version')}:")
    print(f"  keep collections : {', '.join(pol.get('keep_collections', []))}")
    print(f"  drop collections : {', '.join(pol.get('drop_collections', []))}")
    print(f"  keep flat        : {', '.join(pol.get('keep_flat', []))}")
    print(f"  unknown branches : {pol.get('unknown_policy')}")
    trig = pol.get("triggers", {})
    print(f"  required triggers: {', '.join(trig.get('required', []))}")
    print(f"  trigger patterns : {', '.join(trig.get('patterns', []))}")
    return 0


# ---------------------------------------------------------------------------
# validate / upload / fileset (implemented in wrcoffea.skim.validate / upload)
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    from wrcoffea.skim.validate import validate_campaign
    return validate_campaign(args)


def cmd_upload(args) -> int:
    from wrcoffea.skim.upload import upload_campaign
    return upload_campaign(args)


def cmd_split_scan(args) -> int:
    from wrcoffea.skim.plan import Plan
    from wrcoffea.skim.scan import split_scan
    plan = Plan.load(args.plan_dir)
    keys = [k for k, d in plan.datasets.items() if d.scan and (not args.dataset or any(s in k for s in args.dataset))]
    if not keys:
        raise SystemExit("no scan datasets match")
    model_meta = {}
    if args.model_config:
        for cfg in args.model_config:
            for k, v in json.load(open(cfg)).items():
                model_meta[v.get("dataset", k)] = v
    total = []
    for key in keys:
        new = split_scan(plan, key, Path(args.out).expanduser(), model_meta=model_meta, compression=args.compression,
                         part_max_bytes=args.part_max_gb * 1e9 if args.part_max_gb else None,
                         models_per_pass=args.models_per_pass, force=args.force, plan_dir=args.plan_dir)
        print(f"{key}: {len(new)} model datasets written")
        total += new
    plan.save(args.plan_dir)
    print(f"{len(total)} model datasets registered in the plan")
    return 0


def cmd_fileset(args) -> int:
    from wrcoffea.skim.upload import write_filesets
    return write_filesets(args)


def cmd_unskimmed_fileset(args) -> int:
    """Unskimmed filesets (the original NanoAOD files) of one or more eras, from the configs and DBS."""
    from wrcoffea.skim.catalog import DBSClient
    from wrcoffea.skim.unskimmed import eras_with_configs, write_era_filesets
    eras = list(args.era or [])
    if args.all:
        eras = eras_with_configs(args.config_root)
    if not eras:
        logger.error("give --era ERA [ERA ...] or --all")
        return 2
    dbs = None if args.dry_run else DBSClient()
    rc = 0
    for era in eras:
        try:
            write_era_filesets(era, config_root=args.config_root, out_root=args.fileset_dir, dbs=dbs,
                               families=args.family, redirector=args.redirector, runs=args.runs_sums,
                               workers=args.workers, dry_run=args.dry_run)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            logger.error("%s: %s", era, exc)
            rc = 1
    return rc


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def _add_selection_args(p):
    p.add_argument("--era", nargs="+", metavar="ERA", help="restrict to these eras")
    p.add_argument("--dataset", nargs="+", metavar="SUBSTR", help="restrict to datasets whose key contains any SUBSTR")
    p.add_argument("--physics-group", nargs="+", metavar="GROUP", help="restrict to these physics groups")
    p.add_argument("--datatype", choices=["data", "mc"], default=None)
    p.add_argument("--exclude", nargs="+", metavar="SUBSTR", help="leave out datasets whose key contains any SUBSTR")


def _add_worker_args(p):
    p.add_argument("--out", default=str(DEFAULT_OUT), help=f"campaign output root (default {DEFAULT_OUT})")
    p.add_argument("--scratch", default=str(DEFAULT_SCRATCH), help=f"scratch for downloaded inputs (default {DEFAULT_SCRATCH})")
    p.add_argument("--rucio-account", default=os.environ.get("RUCIO_ACCOUNT", "wijackso"),
                   help="Rucio account for replica lookups (set empty to disable)")
    p.add_argument("--keep-inputs", action="store_true", help="keep downloaded inputs in scratch")
    p.add_argument("--chunk-entries", type=int, default=20_000, help="input entries per processing chunk")
    p.add_argument("--streams", type=int, default=4, help="xrdcp parallel streams per download")
    p.add_argument("--decompression-threads", type=int, default=1, help="threads for decompressing input baskets (LZMA dominates the CPU time)")
    p.add_argument("--force", action="store_true", help="rerun groups that are already done")
    p.add_argument("--compression", default=None, help="override the plan's output compression (algo:level)")
    p.add_argument("--part-max-gb", type=float, default=None, help="override the plan's part size cap")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="skim.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="build the campaign plan")
    p.add_argument("--campaign", required=True, help="campaign name, e.g. v2")
    p.add_argument("--plan-dir", required=True, help="directory for plan.json and schemas.json")
    p.add_argument("--config", nargs="+", metavar="JSON", help="config JSON files")
    p.add_argument("--era", nargs="+", metavar="ERA", help="all non-archived configs of these eras")
    p.add_argument("--physics-group", default=None)
    p.add_argument("--dataset", nargs="+", metavar="SUBSTR", help="only datasets whose name contains any SUBSTR")
    p.add_argument("--sample-files", type=int, default=1, help="files per DAS path to sample for the branch layout")
    p.add_argument("--compression", default="zstd:19", help="output compression algo:level (zstd|zlib|lz4|lzma); see docs/skimming.md")
    p.add_argument("--flush-events", type=int, default=50_000, help="events per output basket (flush)")
    p.add_argument("--part-max-gb", type=float, default=2.0, help="roll output parts above this size")
    p.add_argument("--group-gb", nargs="+", metavar="KIND=GB", help="input GB per group, e.g. data=60 mc=30 signal=1000")
    p.add_argument("--skip-private", action="store_true", help="skip config entries that are not DAS paths and carry no file list")
    p.add_argument("--include-optional", action="store_true", help="also plan config entries flagged optional (low-mass DY companions)")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("run", help="process pending groups locally")
    p.add_argument("--plan-dir", required=True)
    _add_selection_args(p)
    _add_worker_args(p)
    p.add_argument("--workers", type=int, default=3, help="parallel group workers (each also prefetches one file)")
    p.add_argument("--limit-groups", type=int, default=0, help="process at most N groups (0 = all)")
    p.add_argument("--keep-bad", action="store_true", help="do not retry groups that finished with unreachable inputs")
    p.add_argument("--dry-run", action="store_true", help="list the groups that would run")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("group", help="process one group (Condor job entry point)")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--dataset", required=True, help="dataset key in the plan")
    p.add_argument("--gid", type=int, required=True)
    _add_worker_args(p)
    p.set_defaults(func=cmd_group)

    p = sub.add_parser("refresh-metadata", help="copy the current config metadata (xsec etc.) into an existing plan")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--config", nargs="+", metavar="JSON", help="config JSON files (default: all non-archived configs of the plan's eras)")
    p.add_argument("--key", nargs="+", metavar="FIELD", help="only these metadata fields (default: every field the configs own)")
    p.add_argument("--dry-run", action="store_true", help="report the changes without writing the plan")
    p.set_defaults(func=cmd_refresh_metadata)
    p = sub.add_parser("condor", help="prepare (and optionally submit) one Condor job per pending group")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT), help="campaign root holding the group statuses (local copy or mount)")
    p.add_argument("--remote", default=DEFAULT_REMOTE, help="xrootd URL the jobs write their outputs to")
    p.add_argument("--jobs-dir", required=True, help="where to write the submission (JDL, wrapper, code tarball, plan slices)")
    _add_selection_args(p)
    p.add_argument("--proxy", default=None, help="x509 proxy file for the jobs (default: $X509_USER_PROXY)")
    p.add_argument("--image", default=None, help="apptainer image (default: the coffea almalinux9 image on cvmfs)")
    p.add_argument("--cpus", type=int, default=1)
    p.add_argument("--memory-mb", type=int, default=4000)
    p.add_argument("--disk-gb", type=float, default=20.0)
    p.add_argument("--streams", type=int, default=4)
    p.add_argument("--decompression-threads", type=int, default=1, help="threads for decompressing input baskets")
    p.add_argument("--max-groups", type=int, default=None, help="submit at most this many groups (tests)")
    p.add_argument("--max-hours", type=float, default=8.0,
                   help="hold a job running longer than this and restart it on another machine (overloaded nodes)")
    p.add_argument("--avoid-hosts", nargs="+", metavar="HOST", help="never run on these worker nodes")
    p.add_argument("--force", action="store_true", help="rerun groups that are already done")
    p.add_argument("--keep-bad", action="store_true", help="do not rerun groups done with unreachable inputs")
    p.add_argument("--ignore-queue", action="store_true", help="do not skip groups that condor_q reports as queued or running")
    p.add_argument("--inflight-file", default=None,
                   help="output of `condor_q -af SkimGroup JobStatus Iwd` saved on the host (condor_q is not visible inside the container)")
    p.add_argument("--submit", action="store_true", help="run condor_submit on the written JDL")
    p.set_defaults(func=cmd_condor)
    p = sub.add_parser("ship", help="copy one finished group to the remote campaign root (job side)")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--remote", required=True)
    p.add_argument("--dataset", required=True, help="dataset key in the plan")
    p.add_argument("--gid", type=int, required=True)
    p.set_defaults(func=cmd_ship)
    p = sub.add_parser("mirror", help="pull sidecars and group statuses of a remote campaign tree into a local root")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--remote", default=DEFAULT_REMOTE)
    _add_selection_args(p)
    p.set_defaults(func=cmd_mirror)
    p = sub.add_parser("bad-files", help="list bad inputs; --classify marks files without any replica as unavailable (Rucio)")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    _add_selection_args(p)
    p.add_argument("--classify", action="store_true")
    p.add_argument("--mark-corrupt", nargs="+", metavar="LFN", help="record these inputs as corrupt in their group statuses (skipped from then on)")
    p.add_argument("--reason", default=None, help="text stored with --mark-corrupt")
    p.add_argument("--rucio-account", default="wijackso")
    p.set_defaults(func=cmd_bad_files)
    p = sub.add_parser("status", help="progress summary")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    _add_selection_args(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="validate the campaign")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    _add_selection_args(p)
    p.add_argument("--accept-bad-files", action="store_true", help="allow datasets with unreachable inputs")
    p.add_argument("--report", default=None, help="write the JSON report here")
    p.add_argument("--skip-reread", action="store_true", help="skip re-reading every part (sidecar-only checks)")
    p.add_argument("--remote", default=None,
                   help="campaign root URL holding parts that are not on local disk (Condor output); their sizes are "
                        "checked with a stat, e.g. " + DEFAULT_REMOTE)
    p.add_argument("--remote-lumis", action="store_true",
                   help="for data, read the LuminosityBlocks trees (remote parts through xrootd) for the lumi checks")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("upload", help="upload finished parts to remote storage")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--remote", default="root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2")
    _add_selection_args(p)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("split-scan", help="split skimmed multi-model scan samples into one dataset per mass point")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--dataset", nargs="+", metavar="SUBSTR", help="restrict to scan datasets whose key contains any SUBSTR")
    p.add_argument("--model-config", nargs="+", metavar="JSON", help="config JSON(s) with per-model metadata (xsec), keyed by model dataset name")
    p.add_argument("--compression", default=None)
    p.add_argument("--part-max-gb", type=float, default=None)
    p.add_argument("--models-per-pass", type=int, default=120)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_split_scan)

    p = sub.add_parser("fileset", help="write analyzer fileset JSONs")
    p.add_argument("--plan-dir", required=True)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--remote", default="root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2",
                   help="URL prefix for the files in the fileset; use --local to point at the local parts")
    p.add_argument("--local", action="store_true", help="reference the local parts instead of the remote copies")
    _add_selection_args(p)
    p.add_argument("--fileset-dir", default=str(REPO_ROOT / "data" / "filesets"))
    p.set_defaults(func=cmd_fileset)

    p = sub.add_parser("unskimmed-fileset",
                       help="write the unskimmed filesets (original NanoAOD files via DBS) of an era, one per config family")
    p.add_argument("--era", nargs="+", metavar="ERA", help="era(s), e.g. RunIII2024Summer24")
    p.add_argument("--all", action="store_true", help="every era that has configs")
    p.add_argument("--family", nargs="+", metavar="FAMILY",
                   help="config families to write (data, signal, mc_dy_lo_ht, ...); default: all but signal_scan")
    p.add_argument("--redirector", default="root://cmsxrootd.fnal.gov/", help="XRootD redirector prefix for the LFNs")
    p.add_argument("--runs-sums", action="store_true",
                   help="also read every file's Runs tree for genEventSumw and the theory-weight sums (slow)")
    p.add_argument("--workers", type=int, default=16, help="parallel Runs-tree reads with --runs-sums")
    p.add_argument("--config-root", default=str(REPO_ROOT / "data" / "configs"))
    p.add_argument("--fileset-dir", default=str(REPO_ROOT / "data" / "filesets"))
    p.add_argument("--dry-run", action="store_true", help="list the configs and output paths without querying DBS")
    p.set_defaults(func=cmd_unskimmed_fileset)

    p = sub.add_parser("cuts", help="print the selection and branch policy")
    p.set_defaults(func=cmd_cuts)

    args = ap.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
