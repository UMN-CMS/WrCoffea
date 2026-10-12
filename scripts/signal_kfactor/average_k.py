#!/usr/bin/env python3
"""<k> per signal sample: the genWeight-weighted mean of the payload's k(m_lN) over every generated event,
per lepton channel (the N flavour), the number the analyzer divides by in its k(m_lN) / <k> weight and the
cross-section tool multiplies the LO value by.

Reads the unskimmed NanoAOD of the plan's signal datasets over xrootd, with the analyzer's own gen matching
(wrcoffea.signal_kfactor.find_gen_ln) and k(m_lN) table, as Run 2 did with CalculateAverageKFactor
(SKFlat) and script/KFactor/Read/run.py (HNWR_Plotter).

    average_k.py --era RunIISummer20UL18 --payload <k(m_lN) json> --out averages.json [--dataset SUBSTR ...]
    average_k.py --merge averages.json --payload <json> --write-payload <json>   # add the averages to a payload

Output: {dataset: {"mass_point": "WR2000_N100", "ee": <k> or null, "mm": <k> or null, "n_ee", "n_mm",
                   "events", "no_gen_match", "bad_files"}}.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from coffea.nanoevents.methods import vector

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from wrcoffea.signal_kfactor import find_gen_ln, _bin_index, mass_point_key  # noqa: E402
from wrcoffea.skim.plan import Plan  # noqa: E402

REDIRECTOR = "root://cmsxrootd.fnal.gov/"


def file_url(lfn: str, redirector: str) -> str:
    """An LFN through the redirector; a private sample's full URL or local path as it is."""
    return lfn if lfn.startswith("root://") or lfn.startswith("/") and not lfn.startswith("/store/") else redirector + lfn
BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_pdgId", "GenPart_genPartIdxMother",
            "genWeight"]
PLANS = Path("~/work/skims_v2/plans").expanduser()


def default_plan_dir(era: str) -> Path:
    """plans/v2_<era> where it exists (UL eras, 2022/2023), else plans/v2 (2024, the private signal)."""
    d = PLANS / f"v2_{era}"
    return d if (d / "plan.json").exists() else PLANS / "v2"
ak.behavior.update(vector.behavior)


def load_curve(path: str) -> dict:
    """The shared curve plus the per-point curves of the payload (as the analyzer's load_payload)."""
    raw = json.loads(Path(path).read_text())
    kf = raw["kfactor"]
    out = {"edges": np.asarray(kf["edges"], float), "values": np.asarray(kf["values"], float), "mln_min": float(kf["mln_min"]),
           "point_curves": {}}
    for name, c in (raw.get("point_curves") or {}).items():
        out["point_curves"][name] = {"edges": np.asarray(c["edges"], float), "values": np.asarray(c["values"], float),
                                     "mln_min": float(c["mln_min"])}
    return out


def curve_for(curves: dict, mass_point: str | None) -> dict:
    """The mass point's own curve when the payload has one (option 2 of docs/studies/signal_kfactor.md), else the shared one."""
    return curves["point_curves"].get(mass_point, curves)


def sample_averages(key: str, lfns: list[str], payload_path: str, redirector: str) -> dict:
    curve = curve_for(load_curve(payload_path), mass_point_key(key))
    sw = {9900012: 0.0, 9900014: 0.0}
    swk = {9900012: 0.0, 9900014: 0.0}
    n_ch = {9900012: 0, 9900014: 0}
    n = n_missing = 0
    bad = []
    for lfn in lfns:
        try:
            with uproot.open(file_url(lfn, redirector), timeout=180) as f:
                a = f["Events"].arrays(BRANCHES)
        except Exception as e:                      # unreachable input: recorded, left out
            bad.append(f"{lfn}: {type(e).__name__}")
            continue
        gp = ak.zip({"pt": a.GenPart_pt, "eta": a.GenPart_eta, "phi": a.GenPart_phi, "mass": a.GenPart_mass,
                     "pdgId": a.GenPart_pdgId, "genPartIdxMother": a.GenPart_genPartIdxMother},
                    with_name="PtEtaPhiMLorentzVector")
        m_ln, channel, found = find_gen_ln(types.SimpleNamespace(GenPart=gp))
        m = np.where(m_ln < 200.0, curve["mln_min"], m_ln)
        k = curve["values"][_bin_index(m, curve["edges"])]
        w = np.asarray(a.genWeight, dtype=np.float64)
        for ch in sw:
            sel = found & (channel == ch)
            sw[ch] += float(w[sel].sum()); swk[ch] += float((w * k)[sel].sum()); n_ch[ch] += int(sel.sum())
        n += len(w); n_missing += int((~found).sum())
    return {"key": key, "mass_point": mass_point_key(key),
            "ee": swk[9900012] / sw[9900012] if sw[9900012] else None,
            "mm": swk[9900014] / sw[9900014] if sw[9900014] else None,
            "n_ee": n_ch[9900012], "n_mm": n_ch[9900014], "events": n, "no_gen_match": n_missing, "bad_files": bad}


def scan_averages(key: str, lfns: list[str], payload_path: str, redirector: str, chunk: int = 200_000) -> dict:
    """Per-model <k> of a multi-model scan sample (one dataset, GenModel_<model> flags per event):
    {model: {"ee", "mm", "n_ee", "n_mm"}} plus the file bookkeeping, computed as sample_averages does."""
    curves = load_curve(payload_path)
    acc: dict[str, dict] = {}
    n = n_missing = 0
    bad = []
    for lfn in lfns:
        try:
            f = uproot.open(file_url(lfn, redirector), timeout=300)
            t = f["Events"]
            flags = [b for b in t.keys() if b.startswith("GenModel_")]
            for a in t.iterate(BRANCHES + flags, step_size=chunk):
                gp = ak.zip({"pt": a.GenPart_pt, "eta": a.GenPart_eta, "phi": a.GenPart_phi, "mass": a.GenPart_mass,
                             "pdgId": a.GenPart_pdgId, "genPartIdxMother": a.GenPart_genPartIdxMother},
                            with_name="PtEtaPhiMLorentzVector")
                m_ln, channel, found = find_gen_ln(types.SimpleNamespace(GenPart=gp))
                w = np.asarray(a.genWeight, dtype=np.float64)
                n += len(w); n_missing += int((~found).sum())
                for fl in flags:
                    sel_m = np.asarray(ak.to_numpy(a[fl]), dtype=bool)
                    if not sel_m.any():
                        continue
                    model = fl[len("GenModel_"):]
                    curve = curve_for(curves, mass_point_key(model))      # each model's own curve
                    m = np.where(m_ln < 200.0, curve["mln_min"], m_ln)
                    k = curve["values"][_bin_index(m, curve["edges"])]
                    d = acc.setdefault(model, {9900012: [0.0, 0.0, 0], 9900014: [0.0, 0.0, 0]})
                    for ch in d:
                        sel = sel_m & found & (channel == ch)
                        d[ch][0] += float(w[sel].sum()); d[ch][1] += float((w * k)[sel].sum()); d[ch][2] += int(sel.sum())
            f.close()
        except Exception as e:
            bad.append(f"{lfn}: {type(e).__name__}")
    models = {}
    for model, d in acc.items():
        models[model] = {"mass_point": mass_point_key(model),
                         "ee": d[9900012][1] / d[9900012][0] if d[9900012][0] else None,
                         "mm": d[9900014][1] / d[9900014][0] if d[9900014][0] else None,
                         "n_ee": d[9900012][2], "n_mm": d[9900014][2]}
    return {"key": key, "events": n, "no_gen_match": n_missing, "bad_files": bad, "models": models}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era")
    ap.add_argument("--payload", required=True, help="payload JSON with the k(m_lN) table")
    ap.add_argument("--plan-dir", default=None, help="skim plan with the signal datasets (default by run)")
    ap.add_argument("--dataset", nargs="*", default=None, help="only datasets whose key contains any SUBSTR")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redirector", default=REDIRECTOR)
    ap.add_argument("--local-dir", default=None,
                    help="directory holding local copies of the inputs (by file name): used instead of xrootd when present")
    ap.add_argument("--merge", default=None, help="averages JSON to merge into --payload -> --write-payload")
    ap.add_argument("--write-payload", default=None)
    ap.add_argument("--source", default=None, help='"source" label of the merged averages (default: the era)')
    ap.add_argument("--scan", action="store_true",
                    help="the plan's scan datasets (multi-model samples) instead of the single-point ones: one output "
                         "record per model, from the unskimmed inputs")
    args = ap.parse_args()

    if args.merge:
        raw = json.loads(Path(args.payload).read_text())
        avgs = json.loads(Path(args.merge).read_text())
        added = 0
        flat = {}
        for key, r in avgs.items():
            if "models" in r:                       # --scan output: one record per model
                flat.update(r["models"])
            else:
                flat[key] = r
        for key, r in flat.items():
            mp = r["mass_point"]
            if not mp:
                continue
            entry = raw.setdefault("averages", {}).setdefault(mp, {"source": args.source or args.era or "computed"})
            for ch in ("ee", "mm"):
                if r.get(ch) is not None:
                    entry[ch] = round(r[ch], 8); added += 1
        Path(args.write_payload).write_text(json.dumps(raw, indent=1) + "\n")
        print(f"merged {added} channel averages for {len(avgs)} datasets into {args.write_payload}")
        return

    plan_dir = Path(args.plan_dir).expanduser() if args.plan_dir else default_plan_dir(args.era)
    plan = Plan.load(plan_dir)
    def local_or_lfn(lfn: str) -> str:
        if args.local_dir:
            cand = Path(args.local_dir) / Path(lfn).name
            if cand.is_file() and cand.stat().st_size > 0:
                return str(cand)
        return lfn

    jobs = {k: [local_or_lfn(f["lfn"]) for f in d.files] for k, d in plan.datasets.items()
            if d.era == args.era and d.category == "signals" and bool(d.scan) == args.scan
            and not d.metadata.get("scan_source")          # split models hold skimmed parts, not the generated events
            and (not args.dataset or any(s in k for s in args.dataset))}
    print(f"{len(jobs)} {'scan' if args.scan else 'signal'} datasets of {args.era} in {plan_dir}", flush=True)
    res = {}
    if args.scan:
        # one worker per input file would need per-file merging; the three scans run in parallel instead
        with ProcessPoolExecutor(min(args.workers, len(jobs)) or 1) as ex:
            futs = {ex.submit(scan_averages, k, v, args.payload, args.redirector): k for k, v in jobs.items()}
            for i, fu in enumerate(as_completed(futs), 1):
                r = fu.result(); res[r["key"]] = r
                print(f"[{i}/{len(jobs)}] {r['key']}: {len(r['models'])} models, {r['events']} events, "
                      f"{r['no_gen_match']} unmatched, {len(r['bad_files'])} bad", flush=True)
    else:
      with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(sample_averages, k, v, args.payload, args.redirector): k for k, v in jobs.items()}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); res[r["key"]] = r
            print(f"[{i}/{len(jobs)}] {r['key'][:48]:48s} {r['mass_point'] or '?':14s} ee {r['ee'] if r['ee'] is None else round(r['ee'], 5)} "
                  f"mm {r['mm'] if r['mm'] is None else round(r['mm'], 5)} ({r['events']} ev, {r['no_gen_match']} unmatched, "
                  f"{len(r['bad_files'])} bad)", flush=True)
    out = args.out or f"averages_{args.era}.json"
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
