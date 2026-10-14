"""Start Condor workers once and let several run_analysis.py commands use them (LPC, inside ./shell).

    python bin/condor_cluster.py [--workers 200] [--worker-memory 6GB] [--idle-timeout 1800]

then, in another tmux window on the same node, as many commands as needed, one after the other or
at the same time:

    python bin/run_analysis.py RunIII2024Summer24 all --scheduler condor_scheduler.json --dir <tag>
    python bin/run_analysis.py RunIII2024Summer24 DYJets --dy nlo_inc --scheduler condor_scheduler.json --dir <tag>

Each run then starts on workers that are already there instead of waiting for new ones. The workers
keep the copy of wrcoffea/ and data/ made when this cluster starts: a run whose checkout differs (or
that runs other package versions or another image) is refused, so after changing the code or the
payloads, stop this cluster and start it again. It stops on Ctrl+C, or by itself once no task has
run and no run has been attached for --idle-timeout seconds. See docs/condor.md.
"""
import run_analysis  # first: sets the environment the libraries read when they load

import argparse
import json
import logging
import os
import time
from pathlib import Path

SCHEDULER_FILE = "condor_scheduler.json"


class QuietBusyChecks(logging.Filter):
    """Drop the scheduler's INFO line for each check of scheduler_busy (one every 30 s)."""

    def filter(self, record):
        return not record.getMessage().startswith("Run out-of-band function 'scheduler_busy'")


def scheduler_busy(dask_scheduler, own_client_id):
    """Whether any task is running or waiting, or a run other than this script is attached.
    Runs on the scheduler (client.run_on_scheduler)."""
    if dask_scheduler.queued or dask_scheduler.unrunnable:
        return True
    if any(ws.processing for ws in dask_scheduler.workers.values()):
        return True
    return any(cid not in (own_client_id, "fire-and-forget") for cid in dask_scheduler.clients)


def write_scheduler_file(path, address, identity):
    """The scheduler file of this cluster: dask's format (Client(scheduler_file=...) reads
    "address") plus when it started and what its workers are."""
    record = {"type": "Scheduler", "address": address, "started": identity["started"],
              "workers": identity["workers"], "memory": identity["memory"], "host": os.uname().nodename}
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def remove_scheduler_file(path, address):
    """Remove the scheduler file, unless another cluster has written its own there since."""
    try:
        if json.loads(Path(path).read_text(encoding="utf-8")).get("address") == address:
            Path(path).unlink()
    except (OSError, ValueError):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=200, help="Number of Dask workers (Condor jobs); default 200.")
    parser.add_argument("--worker-memory", default=run_analysis.CONDOR_WORKER_MEMORY, metavar="SIZE",
                        help=f"Memory of each worker (default {run_analysis.CONDOR_WORKER_MEMORY}; use "
                             f"{run_analysis.CONDOR_WORKER_MEMORY_SYSTS} for runs with --systs).")
    parser.add_argument("--scheduler-file", default=SCHEDULER_FILE, metavar="PATH",
                        help=f"Where to write the scheduler file the runs read (default {SCHEDULER_FILE}).")
    parser.add_argument("--idle-timeout", type=int, default=1800, metavar="SECONDS",
                        help="Stop once no task has run and no run has been attached for this long, so "
                             "that idle workers do not hold slots (default 1800; 0 = never).")
    parser.add_argument("--worker-wait-timeout", type=int, default=3600, metavar="SECONDS",
                        help="Seconds to wait for the first worker while the jobs are idle in a busy pool "
                             "(default 3600).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")

    logging.getLogger("distributed.scheduler").addFilter(QuietBusyChecks())
    # What the workers get: recorded before the cluster starts, when lpcjobqueue copies it.
    identity = run_analysis._cluster_identity(n_workers=args.workers, memory=args.worker_memory)
    cluster, client = run_analysis._start_condor_cluster(n_workers=args.workers, memory=args.worker_memory)
    address = cluster.scheduler_address
    try:
        client.set_metadata(run_analysis.CLUSTER_IDENTITY_KEY, identity)
        write_scheduler_file(args.scheduler_file, address, identity)
        logging.info("Scheduler %s (%d workers of %s requested); scheduler file %s. Run with "
                     "--scheduler %s. Ctrl+C stops the workers.", address, args.workers,
                     args.worker_memory, args.scheduler_file, args.scheduler_file)
        run_analysis._wait_for_workers_or_raise(
            client,
            requested_workers=args.workers,
            min_workers=1,
            timeout_s=args.worker_wait_timeout,
            label=f"condor_cluster_{args.workers}workers",
            job_states=lambda: run_analysis._condor_job_states(cluster),
        )
        last_busy = last_report = time.monotonic()
        while True:
            time.sleep(30)
            # Resubmit the jobs of workers that died, to keep the requested number.
            cluster.scale(args.workers)
            if client.run_on_scheduler(scheduler_busy, own_client_id=client.id):
                last_busy = time.monotonic()
            idle = time.monotonic() - last_busy
            if time.monotonic() - last_report >= 300:
                last_report = time.monotonic()
                logging.info("%d workers; %s.", len(client.scheduler_info()["workers"]),
                             "busy" if idle < 30 else f"idle for {idle / 60:.0f} min")
            if args.idle_timeout and idle >= args.idle_timeout:
                logging.info("Idle for %d s: stopping the workers.", args.idle_timeout)
                break
    except KeyboardInterrupt:
        logging.info("Stopping the workers.")
    finally:
        remove_scheduler_file(args.scheduler_file, address)
        client.close()
        cluster.close()


if __name__ == "__main__":
    main()
