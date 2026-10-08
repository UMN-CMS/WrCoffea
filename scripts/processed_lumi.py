#!/usr/bin/env python3
"""Luminosity of the data actually in the skims, per data stream, era and channel.

Inputs: a per-lumisection brilcalc table of the era's golden JSON and the
``<dataset>_processed_lumis.json`` masks that ``skim.py validate`` writes next
to the parts of every data stream.  The analyzer applies the golden JSON to
data, so only golden lumisections count.

Run 3 splits each primary dataset in two halves (Muon0/Muon1, EGamma0/EGamma1)
that share every lumisection and hold about half of its events each.  Per
golden lumisection, the fraction of a channel's events present in the skims is
taken from the primary-dataset families of that channel: a family with halves
contributes (halves containing the lumisection) / (number of halves), a family
without halves 1 if it contains it, and the channel takes the largest family
fraction (2022C has SingleMuon and Muon for different runs; overlapping
families are not double counted).  The channel luminosity is the sum of
lumi x fraction, exact in expectation; with one halved family it equals the
mean of the two halves' processed luminosities.

Make the table once on a cmslpc node (about 3 minutes for 2024):

    IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cloud/brilws-docker:latest
    apptainer -s exec -B /cvmfs -B /uscms_data -B /uscms/home -B /uscms/homes \\
        --env PYTHONPATH=/home/bril/.local/lib/python3.10/site-packages $IMG \\
        brilcalc lumi -c web --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \\
        -i <golden.json> -u /fb --byls --output-style csv -o golden_byls.csv

then:

    python3 scripts/processed_lumi.py --byls golden_byls.csv --golden <golden.json> \\
        --masks ~/work/skims_v2/Run3/2024/RunIII2024Summer24/data/*/*_processed_lumis.json --out lumi.json
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

CHANNEL_OF = {"Muon": "muon", "SingleMuon": "muon", "EGamma": "electron", "SingleElectron": "electron"}


def read_byls(path: str) -> dict[tuple[int, int], float]:
    """(run, ls) -> recorded luminosity [/fb] from a ``brilcalc --byls --output-style csv`` table."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split(",")
            out[(int(p[0].split(":")[0]), int(p[1].split(":")[0]))] = float(p[6])
    return out


def read_mask(path: str | Path) -> set[tuple[int, int]]:
    d = json.loads(Path(path).read_text())
    return {(int(r), ls) for r, ranges in d.items() for a, b in ranges for ls in range(a, b + 1)}


def parse_stream(name: str) -> tuple[str, str, str]:
    """``Muon0_Run2024E`` -> ("Muon", "0", "Run2024E"); ``SingleMuon_Run2018A`` -> ("SingleMuon", "", "Run2018A")."""
    m = re.match(r"^([A-Za-z]+?)(\d?)_(Run\d{4}[A-Z])", name)
    if not m:
        raise ValueError(f"cannot parse data stream name {name!r}")
    return m.group(1), m.group(2), m.group(3)


def compute(byls: dict, golden: set, masks: dict[str, set]) -> dict:
    per_stream = {}
    runs_by_period = collections.defaultdict(set)
    for name, mask in sorted(masks.items()):
        pd, half, period = parse_stream(name)
        good = mask & golden
        per_stream[name] = {"pd": pd, "half": half, "period": period,
                            "lumi_fb": sum(byls.get(x, 0.0) for x in good),
                            "golden_ls": len(good), "golden_ls_without_lumi": sum(1 for x in good if x not in byls)}
        runs_by_period[period] |= {r for r, _ in mask}
    golden_by_period = {p: sum(v for (r, ls), v in byls.items() if r in runs and (r, ls) in golden)
                        for p, runs in runs_by_period.items()}
    covered = set().union(*runs_by_period.values()) if runs_by_period else set()
    orphan_runs = sorted({r for (r, _) in byls if r not in covered})
    orphan_lumi = sum(v for (r, _), v in byls.items() if r in orphan_runs)
    # family = one primary dataset (with or without halves) in one period
    families: dict[tuple[str, str, str], list[set]] = collections.defaultdict(list)
    for name, mask in masks.items():
        st = per_stream[name]
        families[(CHANNEL_OF.get(st["pd"], st["pd"]), st["period"], st["pd"])].append(mask & golden)
    period_of_run = {r: p for p, runs in runs_by_period.items() for r in runs}
    channel_lumi: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for ch in sorted({c for c, _, _ in families}):
        fams = [(halves) for (c, _, _), halves in families.items() if c == ch]
        frac: dict[tuple[int, int], float] = collections.defaultdict(float)
        for halves in fams:
            n = len(halves)
            count: collections.Counter = collections.Counter()
            for h in halves:
                count.update(h)
            for ls, k in count.items():
                frac[ls] = max(frac[ls], k / n)
        for ls, f in frac.items():
            channel_lumi[ch][period_of_run.get(ls[0], "?")] += byls.get(ls, 0.0) * f
    channel_lumi = {ch: dict(v) for ch, v in channel_lumi.items()}
    return {"per_stream": per_stream, "golden_by_period": golden_by_period,
            "channel_by_period": channel_lumi,
            "channel_total_fb": {ch: sum(v.values()) for ch, v in channel_lumi.items()},
            "golden_total_fb": sum(byls.values()),
            "golden_in_dataset_runs_fb": sum(golden_by_period.values()),
            "golden_runs_in_no_dataset": orphan_runs, "golden_runs_in_no_dataset_fb": orphan_lumi}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--byls", required=True, help="brilcalc --byls csv of the golden JSON")
    ap.add_argument("--golden", required=True, help="the golden JSON the analyzer applies")
    ap.add_argument("--masks", nargs="+", required=True, help="<dataset>_processed_lumis.json files")
    ap.add_argument("--normtag", default="normtag_PHYSICS.json", help="recorded in the output only")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    args = ap.parse_args(argv)
    byls = read_byls(args.byls)
    golden = read_mask(args.golden)
    masks = {Path(m).name.replace("_processed_lumis.json", ""): read_mask(m) for m in args.masks}
    res = compute(byls, golden, masks)
    res.update({"byls_table": str(args.byls), "golden_json": str(args.golden), "normtag": args.normtag})
    periods = sorted(res["golden_by_period"])
    chans = sorted(res["channel_by_period"])
    print(f"{'period':10s} {'golden':>9s} " + " ".join(f"{c:>9s}" for c in chans) + "   [recorded /fb, golden lumisections only]")
    for p in periods:
        print(f"{p:10s} {res['golden_by_period'][p]:9.3f} " + " ".join(f"{res['channel_by_period'][c].get(p, 0.0):9.3f}" for c in chans))
    print(f"{'total':10s} {res['golden_in_dataset_runs_fb']:9.3f} " + " ".join(f"{res['channel_total_fb'][c]:9.3f}" for c in chans))
    print(f"golden JSON total {res['golden_total_fb']:.3f} /fb; golden runs in no data stream: "
          f"{res['golden_runs_in_no_dataset']} ({res['golden_runs_in_no_dataset_fb']:.3f} /fb)")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
