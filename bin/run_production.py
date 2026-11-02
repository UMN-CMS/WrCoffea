"""Run a production: the run_analysis.py commands of an era, at once on Condor workers started here
(LPC, inside ./shell) or one after the other locally.

    python bin/run_production.py RunIII2024Summer24 --condor               # data, backgrounds and every signal point of 2024
    python bin/run_production.py RunIII2024Summer24 --condor --all-dy      # also every other DY setup, all at once
    python bin/run_production.py RunIII2024Summer24 --condor --only data   # one command by name; also bkg, signal, dy_nlo_inc, ...
    python bin/run_production.py RunIII2024Summer24                        # laptop: one command after the other
    python bin/run_production.py RunIII2024Summer24 --plan --all-dy        # print the commands, run nothing
    python bin/run_production.py RunIII2024Summer24 --condor -- --systs lumi pileup sf jer jes   # flags for every command
    python bin/run_production.py all --condor                              # every era with a data fileset

The commands of an era come from config.yaml; data, bkg and signal run unless --all-dy or --only
says otherwise:

    data            data                                both data streams
    bkg             bkg                                 the four background groups with the default DY
    signal          signal                              every signal point of the era
    dy_zpt          DYJets --dy-reweight zpt            eras with a dy_zpt_reweight payload
    dy_zpt_reshape  DYJets --dy-reweight zpt+reshape    eras with a dy_reshape payload as well
    dy_<variant>    DYJets --dy <variant>               every dy_variants entry but the default and nlo_ptll

Flags after "--" go to every command (--signal-points and --mwr-range to signal only); with --systs
among them, dy_zpt also gets dyzpt and dy_zpt_reshape gets dyzpt and dyreshape. Every command first runs with --preflight-only, so a missing
fileset or a flag the era refuses stops the production before any worker is requested. Each command
writes its log to logs/<dir>/; the summary names the failed commands and the line that reruns them.
The output goes to WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/ (--dir defaults to today's date).
See docs/condor.md.
"""
import run_analysis  # first: sets the environment the libraries read when they load

import argparse
import logging
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import NamedTuple

import condor_cluster
from wrcoffea.analysis_config import DY_RESHAPE_JSONS, DY_VARIANTS, DY_ZPT_REWEIGHT_JSONS
from wrcoffea.cli_utils import UNSTITCHED_DY_VARIANTS, build_fileset_path, default_dy_variant, list_eras
from wrcoffea.era_utils import get_era_details

DEFAULT_WORKERS = 250
NOMINAL = ("data", "bkg", "signal")                  # the commands that run without --all-dy or --only
SIGNAL_FLAGS = ("--signal-points", "--mwr-range")    # pass-through flags that only the signal command takes
PREFLIGHT_JOBS = 8          # preflights at a time; each is a python process that imports coffea
LOG_ROOT = Path("logs")
RUN_ANALYSIS = Path("bin") / "run_analysis.py"


class Command(NamedTuple):
    era: str
    name: str
    args: tuple     # the run_analysis.py arguments after the era

    @property
    def label(self):
        return f"{self.era} {self.name}"


class Result(NamedTuple):
    command: Command
    returncode: int
    minutes: float
    log: Path


def with_extra_systs(flags, families):
    """``flags`` with ``families`` added to its ``--systs`` list; unchanged without ``--systs``."""
    flags = list(flags)
    if "--systs" not in flags:
        return flags
    start = flags.index("--systs") + 1
    end = start
    while end < len(flags) and not flags[end].startswith("-"):
        end += 1
    return flags[:end] + [f for f in families if f not in flags[start:end]] + flags[end:]


def split_signal_flags(flags):
    """``flags`` as (the flags every command takes, the signal-point flags with their values)."""
    general, signal = [], []
    flags = list(flags)
    i = 0
    while i < len(flags):
        target = signal if flags[i] in SIGNAL_FLAGS else general
        target.append(flags[i])
        i += 1
        while i < len(flags) and not flags[i].startswith("-"):   # the flag's values
            target.append(flags[i])
            i += 1
    return general, signal


def plan_commands(era, extra=()):
    """The commands of an era's production, from config.yaml: the data, background and signal
    composites, the DY corrections the era has payloads for, and every DY variant that is a
    prediction. ``extra`` flags go to each, the signal-point flags to ``signal`` only."""
    extra, signal_flags = split_signal_flags(extra)
    commands = [Command(era, "data", ("data", *extra)),
                Command(era, "bkg", ("bkg", *extra)),
                Command(era, "signal", ("signal", *extra, *signal_flags))]
    if era in DY_ZPT_REWEIGHT_JSONS:
        commands.append(Command(era, "dy_zpt",
                                ("DYJets", "--dy-reweight", "zpt", *with_extra_systs(extra, ["dyzpt"]))))
        if era in DY_RESHAPE_JSONS:
            commands.append(Command(era, "dy_zpt_reshape",
                                    ("DYJets", "--dy-reweight", "zpt+reshape",
                                     *with_extra_systs(extra, ["dyzpt", "dyreshape"]))))
    default = default_dy_variant(era)
    for variant in DY_VARIANTS.get(era, []):
        if variant != default and variant not in UNSTITCHED_DY_VARIANTS:
            commands.append(Command(era, f"dy_{variant}", ("DYJets", "--dy", variant, *extra)))
    return commands


def select_commands(commands, *, only=(), skip=(), all_dy=False):
    """The commands that run: those named by ``only``, else every command with ``all_dy``, else data,
    bkg and signal; ``skip`` removes names. Names are checked against the whole plan."""
    names = list(dict.fromkeys(c.name for c in commands))
    unknown = sorted((set(only) | set(skip)) - set(names))
    if unknown:
        raise ValueError(f"Unknown command name(s): {', '.join(unknown)}. The commands are {', '.join(names)}.")
    if only:
        chosen = [c for c in commands if c.name in only]
    elif all_dy:
        chosen = list(commands)
    else:
        chosen = [c for c in commands if c.name in NOMINAL]
    return [c for c in chosen if c.name not in skip]


def resolve_eras(spec, *, unskimmed=False):
    """The eras named, or for ``all`` every era with a data fileset."""
    eras = list(dict.fromkeys(spec))
    if eras == ["all"]:
        kept, skipped = [], []
        for era in list_eras():
            has_data = build_fileset_path(era=era, sample="EGamma", unskimmed=unskimmed, dy=None).exists()
            (kept if has_data else skipped).append(era)
        if skipped:
            logging.info("Skipping %s: no %s data fileset.", ", ".join(skipped),
                         "unskimmed" if unskimmed else "skimmed")
        return kept
    unknown = [era for era in eras if era not in list_eras()]
    if unknown:
        raise ValueError(f"Unknown era(s): {', '.join(unknown)}. Choices: all, {', '.join(list_eras())}.")
    return eras


def output_dir(era, tag):
    """Where run_analysis.py --dir ``tag`` writes the era's files (wrcoffea/save_hists.py)."""
    run, year, era = get_era_details(era)
    return Path("WR_Plotter") / "rootfiles" / run / year / era / tag


def command_argv(command, *, tag, scheduler=None, wait_timeout=None, preflight=False):
    """The run_analysis.py command line of ``command``."""
    argv = [sys.executable, str(RUN_ANALYSIS), command.era, *command.args, "--dir", tag]
    if preflight:
        argv.append("--preflight-only")
    elif scheduler:
        argv += ["--scheduler", str(scheduler), "--worker-wait-timeout", str(wait_timeout)]
    return argv


def log_path(command, *, tag, preflight=False):
    return LOG_ROOT / tag / f"{command.era}_{command.name}{'.preflight' if preflight else ''}.log"


def last_line(path, width=200):
    """The last non-empty line of a log, for the report of a failed command."""
    try:
        lines = [line.strip() for line in Path(path).read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1][:width] if lines else ""


def run_commands(commands, *, argv_of, log_of, jobs, tick=None, tick_s=30, poll_s=5, verbose=True, popen=None):
    """Run the commands ``jobs`` at a time, each with its output in its log file, and call ``tick()``
    every ``tick_s`` seconds meanwhile; return a Result per command in the order they finished.
    Ctrl+C reaches the commands too (same process group): they are waited for, then stopped."""
    popen = popen or subprocess.Popen
    pending = list(commands)
    running = {}
    results = []
    last_tick = time.monotonic()
    try:
        while pending or running:
            while pending and len(running) < jobs:
                command = pending.pop(0)
                log = log_of(command)
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = open(log, "w")
                proc = popen(argv_of(command), stdout=handle, stderr=subprocess.STDOUT)
                running[command] = (proc, handle, time.monotonic())
                if verbose:
                    logging.info("%s: started, log %s", command.label, log)
            for command, (proc, handle, started) in list(running.items()):
                returncode = proc.poll()
                if returncode is None:
                    continue
                handle.close()
                del running[command]
                result = Result(command, returncode, (time.monotonic() - started) / 60, log_of(command))
                results.append(result)
                if returncode != 0:
                    logging.error("%s: FAILED after %.1f min (exit %d): %s", command.label,
                                  result.minutes, returncode, last_line(result.log))
                elif verbose:
                    logging.info("%s: done in %.1f min", command.label, result.minutes)
            if tick is not None and time.monotonic() - last_tick >= tick_s:
                tick()
                last_tick = time.monotonic()
            if pending or running:
                time.sleep(poll_s)
    except KeyboardInterrupt:
        logging.info("Interrupted: waiting for %d running command(s) to stop.", len(running))
        for proc, handle, _ in running.values():
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.terminate()
            handle.close()
        raise
    return results


def preflight(commands, *, tag, popen=None):
    """Run every command with --preflight-only; return the results of those that failed."""
    results = run_commands(commands, argv_of=lambda c: command_argv(c, tag=tag, preflight=True),
                           log_of=lambda c: log_path(c, tag=tag, preflight=True),
                           jobs=PREFLIGHT_JOBS, poll_s=1, verbose=False, popen=popen)
    return [r for r in results if r.returncode != 0]


@contextmanager
def production_cluster(*, workers, memory, wait_timeout, scheduler_file=condor_cluster.SCHEDULER_FILE):
    """Start the production's Condor workers, write the scheduler file its commands attach to and wait
    for the first worker; yield a function that resubmits dead workers; stop the workers on exit."""
    if Path(scheduler_file).exists():
        logging.warning("%s exists: to use the workers of a running bin/condor_cluster.py pass --scheduler %s. "
                        "Starting new workers and replacing the file.", scheduler_file, scheduler_file)
    identity = run_analysis._cluster_identity(n_workers=workers, memory=memory)
    cluster, client = run_analysis._start_condor_cluster(n_workers=workers, memory=memory)
    address = cluster.scheduler_address
    try:
        client.set_metadata(run_analysis.CLUSTER_IDENTITY_KEY, identity)
        condor_cluster.write_scheduler_file(scheduler_file, address, identity)
        logging.info("Scheduler %s: %d workers of %s requested.", address, workers, memory)
        run_analysis._wait_for_workers_or_raise(
            client, requested_workers=workers, min_workers=1, timeout_s=wait_timeout,
            label=f"run_production_{workers}workers",
            job_states=lambda: run_analysis._condor_job_states(cluster))
        logging.info("%d worker(s) connected; the others join as the pool frees slots.",
                     len(client.scheduler_info()["workers"]))
        yield lambda: cluster.scale(workers)
    finally:
        logging.info("Stopping the workers.")
        condor_cluster.remove_scheduler_file(scheduler_file, address)
        client.close()
        cluster.close()


def plan_text(commands, *, tag):
    width = max(len(c.label) for c in commands)
    lines = [f"Production {tag}: {len(commands)} command(s), logs in {LOG_ROOT / tag}/"]
    lines += [f"  {c.label:<{width}}  {shlex.join(c.args)}" for c in commands]
    lines += [f"  output {output_dir(era, tag)}/" for era in dict.fromkeys(c.era for c in commands)]
    return "\n".join(lines)


def summary_text(results, *, tag, wall_minutes, rerun_line):
    """Every command's outcome and time, the files written per era and, per era with failed commands,
    ``rerun_line(era, names)``: the command that reruns them."""
    ok = [r for r in results if r.returncode == 0]
    failed = [r for r in results if r.returncode != 0]
    lines = [f"{len(ok)} of {len(results)} command(s) succeeded in {wall_minutes:.0f} min"
             + (f", {len(failed)} FAILED." if failed else ".")]
    width = max(len(r.command.label) for r in results)
    for r in results:
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        detail = "" if r.returncode == 0 else f"  {r.log}: {last_line(r.log)}"
        lines.append(f"  {r.command.label:<{width}}  {status:<16} {r.minutes:6.1f} min{detail}")
    for era in dict.fromkeys(r.command.era for r in results):
        n_files = len(list(output_dir(era, tag).glob("WRAnalyzer_*.root")))
        lines.append(f"  {n_files} files in {output_dir(era, tag)}/")
    for era in dict.fromkeys(r.command.era for r in failed):
        lines.append("Rerun: " + rerun_line(era, [r.command.name for r in failed if r.command.era == era]))
    return "\n".join(lines)


def parse_args(argv):
    argv = list(argv)
    extra = []
    if "--" in argv:    # the flags of every command follow it
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eras", nargs="+", metavar="ERA",
                        help="Era(s) to produce, or 'all' for every era with a data fileset.")
    parser.add_argument("--dir", default=date.today().strftime("%Y%m%d"), metavar="TAG",
                        help="Output directory under WR_Plotter/rootfiles/<Run>/<Year>/<Era>/ and log directory "
                             "under logs/ (default: today's date).")
    parser.add_argument("--condor", action="store_true",
                        help="Start Condor workers (LPC, inside ./shell), run the commands on them at once, stop them.")
    parser.add_argument("--scheduler", default=None, metavar="FILE",
                        help="Run on the workers of a running bin/condor_cluster.py (its scheduler file) "
                             "instead of starting workers.")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Condor workers to start (default {DEFAULT_WORKERS}).")
    parser.add_argument("--worker-memory", default=None, metavar="SIZE",
                        help=f"Memory of each worker (default {run_analysis.CONDOR_WORKER_MEMORY}; "
                             f"{run_analysis.CONDOR_WORKER_MEMORY_SYSTS} when the commands get --systs).")
    parser.add_argument("--worker-wait-timeout", type=int, default=3600, metavar="SECONDS",
                        help="Seconds to wait for the first worker (default 3600; a busy pool may need 21600).")
    parser.add_argument("--jobs", type=int, default=None, metavar="N",
                        help="Commands at a time (default: with Condor every command of an era at once, locally one).")
    parser.add_argument("--all-dy", action="store_true",
                        help="Also run every other DY setup of the era: the Z-pT reweight, reweight+reshape and each "
                             "dy_variants entry but the default and nlo_ptll (default: data, bkg and signal).")
    parser.add_argument("--only", nargs="+", default=(), metavar="NAME",
                        help="Only these commands, from the whole plan (names as --plan --all-dy lists them).")
    parser.add_argument("--skip", nargs="+", default=(), metavar="NAME", help="Not these commands.")
    parser.add_argument("--plan", action="store_true", help="Print the commands and exit.")
    args = parser.parse_args(argv)
    if args.scheduler and (args.workers is not None or args.worker_memory is not None):
        parser.error("--workers and --worker-memory size the workers this script starts, not those of --scheduler.")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    args.extra = extra
    return args


def main(argv=None, *, popen=None, cluster=production_cluster):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tag = args.dir
    on_cluster = args.condor or bool(args.scheduler)
    scheduler = args.scheduler or (condor_cluster.SCHEDULER_FILE if args.condor else None)
    try:
        eras = resolve_eras(args.eras, unskimmed="--unskimmed" in args.extra)
        commands = select_commands([c for era in eras for c in plan_commands(era, args.extra)],
                                   only=args.only, skip=args.skip, all_dy=args.all_dy)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if not commands:
        raise SystemExit("error: no command to run.")
    print(plan_text(commands, tag=tag), flush=True)   # before the logging lines when stdout is a file
    if args.plan:
        return 0

    failed = preflight(commands, tag=tag, popen=popen)
    if failed:
        for r in failed:
            logging.error("%s: preflight failed: %s (%s)", r.command.label, last_line(r.log), r.log)
        logging.error("Nothing started: %d of %d command(s) failed the preflight.", len(failed), len(commands))
        return 2
    logging.info("Preflight passed for %d command(s).", len(commands))

    jobs = args.jobs or (max(sum(c.era == era for c in commands) for era in eras) if on_cluster else 1)
    memory = args.worker_memory or (run_analysis.CONDOR_WORKER_MEMORY_SYSTS if "--systs" in args.extra
                                    else run_analysis.CONDOR_WORKER_MEMORY)
    argv_of = lambda c: command_argv(c, tag=tag, scheduler=scheduler, wait_timeout=args.worker_wait_timeout)
    log_of = lambda c: log_path(c, tag=tag)
    t0 = time.monotonic()
    try:
        if args.condor and not args.scheduler:
            with cluster(workers=args.workers or DEFAULT_WORKERS, memory=memory,
                         wait_timeout=args.worker_wait_timeout) as keep_alive:
                results = run_commands(commands, argv_of=argv_of, log_of=log_of, jobs=jobs, tick=keep_alive, popen=popen)
        else:
            results = run_commands(commands, argv_of=argv_of, log_of=log_of, jobs=jobs, popen=popen)
    except KeyboardInterrupt:
        logging.info("Interrupted: nothing more runs. Rerun with --dir %s to continue.", tag)
        return 130
    results.sort(key=lambda r: commands.index(r.command))

    def rerun_line(era, names):
        mode = ["--scheduler", args.scheduler] if args.scheduler else (["--condor"] if args.condor else [])
        line = ["python", str(Path("bin") / "run_production.py"), era, *mode, "--dir", tag, "--only", *names]
        return shlex.join(line + (["--", *args.extra] if args.extra else []))

    print(summary_text(results, tag=tag, wall_minutes=(time.monotonic() - t0) / 60, rerun_line=rerun_line), flush=True)
    return 0 if not any(r.returncode for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
