#!/usr/bin/env python3
"""Output-format study for the skims: compression, basket size and file count.

Phase ``write``: take existing skim parts as input events, rewrite them with
the new writer under several settings and record size and write time.
Phase ``read``: time the analyzer-like column reads (250k-event chunks of the
~100 branches the analysis touches) over each variant, locally or from
remote URLs, plus the per-file open cost.  Results go to a JSON file.

    python3 scripts/skim_filesize_study.py write --inputs PART.root ... --out DIR
    python3 scripts/skim_filesize_study.py read  --files DIR/*.root --out results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import uproot

# Branches the analysis reads (from a grep of wrcoffea/*.py), by collection.
ANALYSIS_FIELDS = {
    "Electron": ["pt", "eta", "phi", "mass", "charge", "pdgId", "cutBased", "cutBased_HEEP", "deltaEtaSC", "vidNestedWPBitmap", "r9", "seedGain", "energyErr"],
    "Muon": ["pt", "eta", "phi", "mass", "charge", "pdgId", "highPtId", "tkRelIso", "tunepRelPt", "nTrackerLayers"],
    "Jet": ["pt", "eta", "phi", "mass", "area", "rawFactor", "genJetIdx", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity", "jetId", "isTightLeptonVeto"],
    "FatJet": ["pt", "eta", "phi", "mass", "msoftdrop", "lsf3", "area", "rawFactor", "genJetAK8Idx", "subJetIdx1", "subJetIdx2", "jetId", "isTight", "chHEF", "neHEF", "chEmEF", "neEmEF", "muEF", "chMultiplicity", "neMultiplicity"],
    "SubJet": ["pt", "eta", "phi", "mass", "area", "rawFactor"],
    "GenJet": ["pt", "eta", "phi", "mass"], "GenJetAK8": ["pt", "eta", "phi", "mass"],
    "GenPart": ["pt", "eta", "phi", "mass", "pdgId", "status", "statusFlags", "genPartIdxMother"],
    "Pileup": ["nTrueInt"], "Rho": ["fixedGridRhoFastjetAll"],
    "Flag": ["goodVertices", "globalSuperTightHalo2016Filter", "EcalDeadCellTriggerPrimitiveFilter", "BadPFMuonFilter", "eeBadScFilter", "BadPFMuonDzFilter", "hfNoisyHitsFilter", "ecalBadCalibFilter"],
    "HLT": ["Mu50", "HighPtTkMu100", "CascadeMu100", "OldMu100", "TkMu100", "Ele30_WPTight_Gsf", "Ele32_WPTight_Gsf", "Photon200", "Ele115_CaloIdVT_GsfTrkIdT", "Ele50_CaloIdVT_GsfTrkIdT_PFJet165"],
    "MET": ["pt", "phi"], "PuppiMET": ["pt", "phi"],
}
ANALYSIS_FLAT = ["run", "luminosityBlock", "event", "genWeight", "LHEScaleWeight", "LHEPdfWeight", "PSWeight", "fixedGridRhoFastjetAll"]


def analysis_columns(keys) -> list[str]:
    keys = set(keys)
    cols = [k for k in ANALYSIS_FLAT if k in keys]
    for coll, fields in ANALYSIS_FIELDS.items():
        cols += [f"{coll}_{f}" for f in fields if f"{coll}_{f}" in keys]
        if f"n{coll}" in keys:
            cols.append(f"n{coll}")
    return cols


# ----------------------------------------------------------------------------
# write phase
# ----------------------------------------------------------------------------

def cmd_write(args) -> int:
    from wrcoffea.skim.schema import build_schema, branch_types_of_tree
    from wrcoffea.skim.writer import PartWriter, InputRecord
    from wrcoffea.skim.selection import EVENT_ID_BRANCHES
    from wrcoffea.skim.recount import IdDigest

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = [p for pat in args.inputs for p in sorted(glob.glob(pat))]
    with uproot.open(inputs[0]) as f:
        schema = build_schema([branch_types_of_tree(f["Events"])])
    results = []

    def rewrite(tag, compression, flush, n_events, part_max_bytes=1e15):
        w = PartWriter(out, tag, schema, is_data=False, compression=compression, flush_events=flush,
                       part_max_bytes=part_max_bytes, name_fmt="{base}_p{index:02d}.root")
        t0 = time.monotonic()
        remaining = n_events
        for path in inputs:
            if remaining <= 0:
                break
            with uproot.open(path) as f:
                t = f["Events"]
                keys = t.keys()
                kept = [n for n in schema.names if n in keys]
                n_take = min(t.num_entries, remaining)
                rec = InputRecord(lfn=path, n_in=n_take)
                w.begin_input(rec, available_branches=keys)
                d = IdDigest()
                for s in range(0, n_take, args.chunk):
                    e = min(s + args.chunk, n_take)
                    arr = t.arrays(kept, entry_start=s, entry_stop=e)
                    ids = arr[list(EVENT_ID_BRANCHES)]
                    d.update(ids["run"], ids["luminosityBlock"], ids["event"])
                    w.add_chunk(arr, e - s)
                rec.ids_digest = rec.recount_digest = d.hexdigest()
                runs = f["Runs"].arrays(library="ak")
                w.end_input({k: runs[k] for k in runs.fields}, None)
                remaining -= n_take
        parts = w.close()
        dt = time.monotonic() - t0
        size = sum(p.size for p in parts)
        n = sum(p.n_events for p in parts)
        r = {"tag": tag, "compression": compression, "flush_events": flush, "n_parts": len(parts), "events": n,
             "bytes": size, "bytes_per_event": size / max(n, 1), "write_seconds": dt, "files": [p.path for p in parts]}
        results.append(r)
        print(f"{tag:28s} {compression:8s} flush={flush:>7d} parts={len(parts):2d} {n:8d} ev {size/1e6:8.1f} MB "
              f"{size/max(n,1):6.0f} B/ev  {dt:6.1f}s", flush=True)
        (out / "write_results.json").write_text(json.dumps(results, indent=1))

    n1 = args.events
    for comp in args.compressions:
        rewrite(f"comp_{comp.replace(':', '')}", comp, args.flush, n1)
    for flush in args.flushes:
        rewrite(f"flush_{flush}", args.base_compression, flush, n1)
    # file-count variants: same events split into k files
    for k in args.splits:
        per = int(np.ceil(args.split_events / k))
        # approximate the per-file cap by events: use a tiny part_max_bytes and limit via inputs? Simpler: roll by events.
        w_tag = f"split_{k}"
        rewrite_by_events(w_tag, args.base_compression, args.flush, args.split_events, per, inputs, schema, out, results, args.chunk)
    return 0


def rewrite_by_events(tag, compression, flush, n_events, per_file, inputs, schema, out, results, chunk):
    """Write exactly n_events into files of per_file events each (rolling by count)."""
    from wrcoffea.skim.writer import PartWriter, InputRecord
    from wrcoffea.skim.selection import EVENT_ID_BRANCHES
    from wrcoffea.skim.recount import IdDigest
    w = PartWriter(out, tag, schema, is_data=False, compression=compression, flush_events=flush,
                   part_max_bytes=1e15, name_fmt="{base}_p{index:02d}.root")
    t0 = time.monotonic()
    remaining = n_events
    in_file = 0
    for path in inputs:
        if remaining <= 0:
            break
        with uproot.open(path) as f:
            t = f["Events"]
            keys = t.keys()
            kept = [n for n in schema.names if n in keys]
            runs = f["Runs"].arrays(library="ak")
            runs = {k: runs[k] for k in runs.fields}
            s = 0
            while s < t.num_entries and remaining > 0:
                e = min(s + min(chunk, per_file - in_file, remaining), t.num_entries)
                rec = InputRecord(lfn=f"{path}#{s}", n_in=e - s)
                w.begin_input(rec, available_branches=keys)
                arr = t.arrays(kept, entry_start=s, entry_stop=e)
                d = IdDigest()
                ids = arr[list(EVENT_ID_BRANCHES)]
                d.update(ids["run"], ids["luminosityBlock"], ids["event"])
                w.add_chunk(arr, e - s)
                rec.ids_digest = rec.recount_digest = d.hexdigest()
                w.end_input(runs, None)
                in_file += e - s
                remaining -= e - s
                s = e
                if in_file >= per_file:
                    w.close_part()
                    in_file = 0
    parts = w.close()
    dt = time.monotonic() - t0
    size = sum(p.size for p in parts)
    n = sum(p.n_events for p in parts)
    r = {"tag": tag, "compression": compression, "flush_events": flush, "n_parts": len(parts), "events": n,
         "bytes": size, "bytes_per_event": size / max(n, 1), "write_seconds": dt, "files": [p.path for p in parts]}
    results.append(r)
    print(f"{tag:28s} {compression:8s} flush={flush:>7d} parts={len(parts):2d} {n:8d} ev {size/1e6:8.1f} MB "
          f"{size/max(n,1):6.0f} B/ev  {dt:6.1f}s", flush=True)
    (out / "write_results.json").write_text(json.dumps(results, indent=1))


# ----------------------------------------------------------------------------
# read phase
# ----------------------------------------------------------------------------

def time_reads(files: list[str], *, chunk: int = 250_000, timeout: int = 120, repeat: int = 1) -> dict:
    """Open each file, read the analysis columns chunk by chunk; return timings."""
    res = {"files": len(files), "open_seconds": 0.0, "read_seconds": 0.0, "events": 0, "bytes_compressed": 0,
           "columns": 0}
    for _ in range(repeat):
        for path in files:
            t0 = time.monotonic()
            f = uproot.open(path, timeout=timeout) if path.startswith("root://") else uproot.open(path)
            t = f["Events"]
            n = t.num_entries
            cols = analysis_columns(t.keys())
            res["columns"] = len(cols)
            res["open_seconds"] += time.monotonic() - t0
            t1 = time.monotonic()
            for s in range(0, n, chunk):
                arr = t.arrays(cols, entry_start=s, entry_stop=min(s + chunk, n))
                del arr
            res["read_seconds"] += time.monotonic() - t1
            res["events"] += n
            res["bytes_compressed"] += sum(t[c].compressed_bytes for c in cols)
            f.close()
    res["events_per_second"] = res["events"] / max(res["read_seconds"], 1e-9)
    res["MB_per_second"] = res["bytes_compressed"] / 1e6 / max(res["read_seconds"], 1e-9)
    return res


def cmd_read(args) -> int:
    groups: dict[str, list[str]] = {}
    for pat in args.files:
        for p in (sorted(glob.glob(pat)) if not pat.startswith("root://") else [pat]):
            tag = os.path.basename(p).rsplit("_p", 1)[0]
            groups.setdefault(tag, []).append(p)
    out = {}
    for tag, files in groups.items():
        r = time_reads(files, chunk=args.chunk, timeout=args.timeout, repeat=args.repeat)
        out[tag] = r
        print(f"{tag:28s} files={r['files']:2d} open={r['open_seconds']:6.2f}s read={r['read_seconds']:7.1f}s "
              f"{r['events_per_second']/1e3:7.1f} kev/s {r['MB_per_second']:6.1f} MB/s cols={r['columns']}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--events", type=int, default=1_000_000)
    p.add_argument("--chunk", type=int, default=50_000)
    p.add_argument("--flush", type=int, default=50_000)
    p.add_argument("--compressions", nargs="+", default=["zstd:3", "zstd:9", "zlib:1", "zlib:6", "lz4:4", "lzma:9"])
    p.add_argument("--base-compression", default="zstd:3")
    p.add_argument("--flushes", nargs="*", type=int, default=[10_000, 200_000])
    p.add_argument("--splits", nargs="*", type=int, default=[1, 4, 16])
    p.add_argument("--split-events", type=int, default=2_000_000)
    p.set_defaults(func=cmd_write)
    p = sub.add_parser("read")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--chunk", type=int, default=250_000)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_read)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
