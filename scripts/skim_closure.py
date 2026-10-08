#!/usr/bin/env python3
"""Analyzer-level closure test: skim parts versus the raw inputs they came from.

Runs ``bin/run_analysis.py`` twice with identical metadata, once on the raw
NanoAOD inputs of a dataset (kept with ``skim.py run --keep-inputs``) and once
on the skim parts made from exactly those inputs, then compares every
histogram of the two outputs bin by bin.  Because the skim cuts are looser
than every analysis selection, the two must agree to floating-point
precision; any difference means the skim lost or altered events.

    python3 scripts/skim_closure.py --plan-dir plans/v2 --dataset TTto2L2Nu --out ~/work/skims_v2 \
        --inputs-dir ~/work/skim_scratch/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8 --sample tt_tW
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import uproot

from wrcoffea.skim.plan import Plan
from wrcoffea.skim.upload import finished_parts, dataset_metadata_from_parts


def build_filesets(plan: Plan, key: str, out_root: Path, inputs_dir: Path, work: Path) -> tuple[Path, Path, list[str]]:
    ds = plan.datasets[key]
    parts = finished_parts(plan, key, out_root, allow_bad=True)
    if parts is None:
        raise SystemExit(f"{key}: not all groups are done")
    meta = dataset_metadata_from_parts(ds, parts)
    meta["physics_group"] = ds.physics_group
    skim_files = {sc["part"]["path"]: "Events" for sc in sorted(parts, key=lambda s: s["part"]["path"])}
    lfns = [i["lfn"] for sc in parts for i in sc["part"]["inputs"]]
    raw_files = {}
    missing = []
    for lfn in lfns:
        local = inputs_dir / os.path.basename(lfn)
        if local.exists():
            raw_files[str(local)] = "Events"
        else:
            missing.append(lfn)
    if missing:
        raise SystemExit(f"{len(missing)} raw input(s) not found under {inputs_dir}, e.g. {missing[0]}")
    das = ds.das_paths[0]
    work.mkdir(parents=True, exist_ok=True)
    raw_fs = work / "fileset_raw.json"
    skim_fs = work / "fileset_skim.json"
    raw_fs.write_text(json.dumps({das: {"files": raw_files, "metadata": meta}}, indent=1))
    skim_fs.write_text(json.dumps({das: {"files": skim_files, "metadata": meta}}, indent=1))
    return raw_fs, skim_fs, lfns


def run_analyzer(era: str, sample: str, fileset: Path, tag: str, name: str, *, mass: str | None,
                 workers: int, extra: list[str]) -> Path:
    cmd = [sys.executable, str(REPO / "bin" / "run_analysis.py"), era, sample, "--fileset", str(fileset),
           "--dir", tag, "--name", name, "--max-workers", str(workers)] + extra
    if mass:
        cmd += ["--mass", mass]
    print("running:", " ".join(cmd), flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True, cwd=str(REPO))
    print(f"  done in {(time.monotonic() - t0)/60:.1f} min", flush=True)
    from wrcoffea.era_utils import get_era_details
    run, year, era_name = get_era_details(era)
    out_dir = REPO / "WR_Plotter" / "rootfiles" / run / year / era_name / tag
    if sample == "Signal":
        return out_dir / f"WRAnalyzer_{name}_signal_{mass}.root"
    return out_dir / f"WRAnalyzer_{name}_{sample}.root"


def compare_root(a: Path, b: Path, *, rtol: float = 1e-6, atol: float = 0.0) -> dict:
    """Compare every histogram in two ROOT files; returns a report dict.

    Ordinary histograms must agree bin by bin.  Cutflow histograms count events
    before the selection too, and their first rows (no cuts, jet-only rows,
    single-lepton rows) legitimately differ because the skim removes those
    events; for them the rows from the first agreeing row onward must all
    agree, and the final row in particular.
    """
    rep = {"n_hists": 0, "mismatched": [], "only_a": [], "only_b": [], "max_rel_diff": 0.0, "cutflow_prefix_rows": {}}
    with uproot.open(a) as fa, uproot.open(b) as fb:
        ka = {k.split(";")[0] for k, c in fa.classnames(recursive=True).items() if c.startswith("TH")}
        kb = {k.split(";")[0] for k, c in fb.classnames(recursive=True).items() if c.startswith("TH")}
        rep["only_a"] = sorted(ka - kb)
        rep["only_b"] = sorted(kb - ka)
        for k in sorted(ka & kb):
            ha, hb = fa[k], fb[k]
            va, vb = ha.values(flow=True), hb.values(flow=True)
            ea, eb = ha.variances(flow=True), hb.variances(flow=True)
            rep["n_hists"] += 1
            if va.shape != vb.shape:
                rep["mismatched"].append({"hist": k, "reason": "shape"})
                continue
            close = np.isclose(va, vb, rtol=rtol, atol=atol)
            if ea is not None and eb is not None:
                close &= np.isclose(ea, eb, rtol=rtol, atol=atol)
            if "cutflow" in k.lower():
                if "onecut" in k.lower():
                    # One-cut-at-a-time rows apply a single cut to every event,
                    # so they cannot close against a skim; diagnostics only.
                    rep.setdefault("ignored", []).append(k)
                    continue
                main_a, main_b = ha.values(flow=False), hb.values(flow=False)
                close_main = np.isclose(main_a, main_b, rtol=rtol, atol=atol)
                agree_from = int(np.argmax(close_main)) if close_main.any() else len(close_main)
                tail_ok = close_main[agree_from:].all() if agree_from < len(close_main) else False
                rep["cutflow_prefix_rows"][k] = agree_from
                if not tail_ok or not close_main[-1]:
                    rep["mismatched"].append({"hist": k, "sum_a": float(va.sum()), "sum_b": float(vb.sum()),
                                              "reason": f"rows differ after row {agree_from}"})
                continue
            denom = np.maximum(np.abs(va), 1e-12)
            rel = float(np.max(np.abs(va - vb) / denom)) if va.size else 0.0
            rep["max_rel_diff"] = max(rep["max_rel_diff"], rel)
            if not close.all():
                rep["mismatched"].append({"hist": k, "sum_a": float(va.sum()), "sum_b": float(vb.sum()), "reason": "bins differ"})
    rep["ok"] = not rep["mismatched"] and not rep["only_a"] and not rep["only_b"]
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan-dir", required=True)
    ap.add_argument("--dataset", required=True, help="dataset key in the plan")
    ap.add_argument("--out", required=True, help="campaign output root")
    ap.add_argument("--inputs-dir", required=True, help="directory holding the kept raw inputs")
    ap.add_argument("--sample", required=True, help="run_analysis sample argument (physics group, or Signal)")
    ap.add_argument("--mass", default=None, help="signal mass point, e.g. WR2000_N100")
    ap.add_argument("--era", default=None, help="era for run_analysis (default: the dataset's era)")
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--work", default=None, help="where to write the filesets and the report")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--extra", nargs="*", default=[], help="extra run_analysis arguments, e.g. --region resolved")
    args = ap.parse_args(argv)

    plan = Plan.load(args.plan_dir)
    key = next((k for k in plan.datasets if args.dataset in k), None)
    if key is None:
        raise SystemExit(f"no dataset matches {args.dataset!r}")
    ds = plan.datasets[key]
    era = args.era or ds.era
    work = Path(args.work or (Path(args.out).expanduser() / "_closure" / ds.key))
    raw_fs, skim_fs, lfns = build_filesets(plan, key, Path(args.out).expanduser(), Path(args.inputs_dir).expanduser(), work)
    tag = f"closure_{plan.campaign}_{ds.key[:40]}"
    out_raw = run_analyzer(era, args.sample, raw_fs, tag, "raw", mass=args.mass, workers=args.max_workers, extra=args.extra)
    out_skim = run_analyzer(era, args.sample, skim_fs, tag, "skim", mass=args.mass, workers=args.max_workers, extra=args.extra)
    rep = compare_root(out_raw, out_skim, rtol=args.rtol)
    rep.update({"dataset": ds.key, "inputs": len(lfns), "raw": str(out_raw), "skim": str(out_skim)})
    (work / "closure_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(f"\n{ds.key}: {rep['n_hists']} histograms compared, max relative difference {rep['max_rel_diff']:.2e}, "
          f"{len(rep['mismatched'])} mismatched, only-raw {len(rep['only_a'])}, only-skim {len(rep['only_b'])}")
    for m in rep["mismatched"][:10]:
        print(f"   {m['hist']}: {m.get('reason', '')} raw {m.get('sum_a', 0):.6g} skim {m.get('sum_b', 0):.6g}")
    pre = rep.get("cutflow_prefix_rows", {})
    if pre:
        print(f"   cutflows: pre-skim rows differ in {sum(1 for v in pre.values() if v > 0)}/{len(pre)} histograms "
              f"(first agreeing row at most {max(pre.values())}), all later rows agree")
    print("CLOSURE OK" if rep["ok"] else "CLOSURE FAILED")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
