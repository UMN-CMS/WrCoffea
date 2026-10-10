#!/usr/bin/env python3
"""Derive the DY reshape ratio vs jet pT from the DY control regions.

Reproduces the Run 2 (AN-19-083) ``DYReshapeJetPt`` inputs from a WR_Plotter
rootfile directory produced by ``run_analysis.py``:

    ratio = (data - non-DY MC) / DY MC

in the leading AK4 jet pT of the resolved DY CRs and the DY-candidate AK8
jet pT of the boosted DY CRs, with the Run 2 binning.  The ee and mumu
control regions are combined for the nominal ratio (as in Run 2); the
per-channel ratios are kept and their absolute difference is stored as the
channel systematic.  Bins without DY MC (below the jet thresholds) get a
ratio of 1 with zero uncertainty.

The DY histograms must have been produced with the Z-pT reweighting on
(the default), because the reshape is applied on top of it.  When only DY
was rerun with the reweighting, ``--dy-rootfiles`` points at that directory
and the data and non-DY backgrounds still come from ``--rootfiles``.
``--region-suffix _mll60to100`` derives the ratio in the DY CR1 halves of the
``dy-cr-split`` study (AN-19-083 CR1 -> CR2 test) instead of the full DY CRs.

Example::

    python scripts/derive_dy_reshape.py --era RunIISummer20UL18 \
        --rootfiles WR_Plotter/rootfiles/RunII/2018/RunIISummer20UL18/<tag> \
        --out data/reweights/RunII/2018/RunIISummer20UL18/dy_reshape_<tag>.json

    python scripts/derive_dy_reshape.py --era RunIII2024Summer24 \
        --rootfiles WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/<tag> \
        --dy-rootfiles WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/<dy tag> \
        --out data/reweights/Run3/2024/RunIII2024Summer24/dy_reshape_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import uproot

# Run 2 DYReshapeJetPt binning.
RESOLVED_EDGES = [0.0, 40.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0]
BOOSTED_EDGES = [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0]

KINDS = {
    "resolved": {
        "variable": "pt_leading_jet",
        "edges": RESOLVED_EDGES,
        "regions": {"ee": "wr_ee_resolved_dy_cr", "mumu": "wr_mumu_resolved_dy_cr"},
    },
    "boosted": {
        "variable": "pt_leading_AK8Jets",
        "edges": BOOSTED_EDGES,
        "regions": {"ee": "wr_ee_boosted_dy_cr", "mumu": "wr_mumu_boosted_dy_cr"},
    },
}
DATA_FILES = {"ee": "WRAnalyzer_EGamma.root", "mumu": "WRAnalyzer_Muon.root"}
DY_FILE = "WRAnalyzer_DYJets.root"
NOT_BACKGROUND = ("DYJets", "EGamma", "Muon", "signal", "Signal", "Observed")


def _read(path: Path, key: str, edges_out):
    """Rebin a fine histogram (values, variances) onto *edges_out*."""
    with uproot.open(path) as f:
        if key not in f:
            raise KeyError(f"{path.name} has no histogram '{key}'")
        h = f[key]
        values, edges_in = h.to_numpy()
        variances = h.variances()
        if variances is None:
            variances = values.copy()
    edges_out = np.asarray(edges_out, dtype=np.float64)
    inside = np.isin(edges_out, edges_in)
    if not inside.all():
        raise ValueError(f"Target edges {edges_out[~inside]} are not edges of '{key}' in {path.name}")
    centers = 0.5 * (edges_in[:-1] + edges_in[1:])
    vals, _ = np.histogram(centers, bins=edges_out, weights=values)
    vars_, _ = np.histogram(centers, bins=edges_out, weights=variances)
    return vals, vars_


def background_files(rootfiles: Path) -> list[Path]:
    out = []
    for p in sorted(rootfiles.glob("WRAnalyzer_*.root")):
        stem = p.stem[len("WRAnalyzer_"):]
        if any(stem.startswith(tag) for tag in NOT_BACKGROUND):
            continue
        out.append(p)
    return out


def _ratio(data, var_data, bkg, var_bkg, dy, var_dy):
    num = data - bkg
    ratio = np.ones_like(dy)
    err = np.zeros_like(dy)
    ok = dy > 0
    ratio[ok] = num[ok] / dy[ok]
    err[ok] = np.sqrt((var_data[ok] + var_bkg[ok]) / dy[ok] ** 2 + (num[ok] ** 2) * var_dy[ok] / dy[ok] ** 4)
    return ratio, err, ok


def derive_kind(rootfiles: Path, kind: str, bkgs: list[Path], dy_rootfiles: Path | None = None,
                region_suffix: str = "") -> dict:
    cfg = KINDS[kind]
    edges = cfg["edges"]
    per_channel = {}
    totals = {"data": 0, "var_data": 0, "bkg": 0, "var_bkg": 0, "dy": 0, "var_dy": 0}
    for channel, region in cfg["regions"].items():
        region = region + region_suffix
        key = f"{region}/{cfg['variable']}_{region}"
        data, var_data = _read(rootfiles / DATA_FILES[channel], key, edges)
        dy, var_dy = _read((dy_rootfiles or rootfiles) / DY_FILE, key, edges)
        bkg = np.zeros_like(dy)
        var_bkg = np.zeros_like(dy)
        for p in bkgs:
            b, vb = _read(p, key, edges)
            bkg += b
            var_bkg += vb
        ratio, err, ok = _ratio(data, var_data, bkg, var_bkg, dy, var_dy)
        per_channel[channel] = {"ratio": ratio, "err": err, "defined": ok, "data": data, "bkg": bkg, "dy": dy}
        for name, arr in (("data", data), ("var_data", var_data), ("bkg", bkg), ("var_bkg", var_bkg),
                          ("dy", dy), ("var_dy", var_dy)):
            totals[name] = totals[name] + arr
    ratio, err, ok = _ratio(totals["data"], totals["var_data"], totals["bkg"], totals["var_bkg"],
                            totals["dy"], totals["var_dy"])
    both = per_channel["ee"]["defined"] & per_channel["mumu"]["defined"]
    channel_err = np.where(both, np.abs(per_channel["ee"]["ratio"] - per_channel["mumu"]["ratio"]), 0.0)
    return {
        "variable": cfg["variable"],
        "edges": edges,
        "ratio": ratio.tolist(),
        "stat_err": err.tolist(),
        "ee_ratio": per_channel["ee"]["ratio"].tolist(),
        "ee_stat_err": per_channel["ee"]["err"].tolist(),
        "mumu_ratio": per_channel["mumu"]["ratio"].tolist(),
        "mumu_stat_err": per_channel["mumu"]["err"].tolist(),
        "channel_err": channel_err.tolist(),
        "yields": {
            "data": totals["data"].tolist(),
            "non_dy_mc": totals["bkg"].tolist(),
            "dy_mc": totals["dy"].tolist(),
        },
    }


def derive(rootfiles: Path, era: str, note: str = "", dy_rootfiles: Path | None = None,
           region_suffix: str = "") -> dict:
    bkgs = background_files(rootfiles)
    if not bkgs:
        raise FileNotFoundError(f"No non-DY background files found in {rootfiles}")
    out = {
        "provenance": {
            "era": era,
            "rootfiles": str(rootfiles),
            "dy_rootfiles": str(dy_rootfiles or rootfiles),
            "regions": {k: {ch: r + region_suffix for ch, r in v["regions"].items()} for k, v in KINDS.items()},
            "dy_file": DY_FILE,
            "data_files": DATA_FILES,
            "background_files": [p.name for p in bkgs],
            "derived_on": date.today().isoformat(),
            "method": "Run 2 DYReshapeJetPt: (data - non-DY MC) / DY MC in the DY CRs, ee and mumu combined",
            "note": note,
        },
        "requires_dy_zpt_reweight": True,
    }
    for kind in KINDS:
        out[kind] = derive_kind(rootfiles, kind, bkgs, dy_rootfiles, region_suffix)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", required=True)
    parser.add_argument("--rootfiles", type=Path, required=True, help="WR_Plotter rootfile directory of one production")
    parser.add_argument("--dy-rootfiles", type=Path, default=None,
                        help="Directory holding WRAnalyzer_DYJets.root (default: --rootfiles)")
    parser.add_argument("--region-suffix", default="",
                        help="Suffix of the DY CR regions to use, e.g. _mll60to100 (study dy-cr-split)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--note", default="", help="Free-text provenance note (e.g. the production command)")
    args = parser.parse_args(argv)

    payload = derive(args.rootfiles, args.era, args.note, args.dy_rootfiles, args.region_suffix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    for kind in KINDS:
        block = payload[kind]
        print(f"{kind}: edges {block['edges']}")
        print(f"   ratio    {np.round(block['ratio'], 3).tolist()}")
        print(f"   stat_err {np.round(block['stat_err'], 3).tolist()}")
        print(f"   ee/mumu  {np.round(block['ee_ratio'], 3).tolist()} / {np.round(block['mumu_ratio'], 3).tolist()}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
