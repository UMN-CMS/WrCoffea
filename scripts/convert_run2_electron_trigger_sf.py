#!/usr/bin/env python3
"""Convert the Run 2 HNWR electron trigger efficiencies to a correctionlib file.

The Run 2 WR analysis (SKFlatAnalyzer ``HNWR/Run2Legacy_v4``) measured the
efficiency of the electron trigger OR for HEEP electrons and stored it in
``data/Run2Legacy_v4/<year>/ID/Electron/WREGammaTriggerEff.root`` as
``DataEff_TH2F_{Barrel,EndCap}``, ``MCEff_TH2F_{Barrel,EndCap}`` and
``SF_TH2F_{Barrel,EndCap}`` (x = supercluster eta, y = pT).
``MCCorrection::ElectronTrigger_SF`` combines the per-electron efficiencies
with the OR formula ``1 - prod(1 - eff)`` and takes data over MC, after
clamping pT to [50, 499] and |eta| to 2.5.

This script writes the same numbers as a correctionlib set with the
signature of the Run 3 EGM HLT payloads, so ``electron_trigger_sf`` can use
it unchanged:

    Electron-HLT-DataEff / Electron-HLT-McEff / Electron-HLT-SF
        (year: "2018", ValType: nom|up|down, Path: "WREGammaTrigger", eta, pt)

Per eta bin the pT binning of the matching histogram is used (barrel rows
from the Barrel histogram with its last [500, 2000) bin dropped so that the
Run 2 pT clamp at 499 is reproduced by ``flow="clamp"``; endcap rows from the
EndCap histogram).  The two crack rows (1.444 < |eta| < 1.566) hold 0, as in
the source file; the analysis vetoes crack electrons anyway.

Example::

    python scripts/convert_run2_electron_trigger_sf.py \
        --skflat-data /path/to/SKFlatAnalyzer/data/Run2Legacy_v4 --year 2018 \
        --out data/jsonpog/EGM/RunII/RunIISummer20UL18/electronHlt_WREGammaTrigger_run2legacy_v4.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import uproot
from correctionlib import schemav2 as cs

SOURCE_REPO = "https://github.com/jedori0228/SKFlatAnalyzer"
SOURCE_BRANCH = "HNWR/Run2Legacy_v4"
PATH_NAME = "WREGammaTrigger"
YEAR_KEYS = {2016: "2016", 2017: "2017", 2018: "2018"}
PT_CLAMP_MAX = 500.0   # Run 2: pt >= 500 -> 499, i.e. never above the [300, 500) barrel bin
VALTYPES = ("nom", "up", "down")


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read(f, name):
    h = f[name]
    values, xedges, yedges = h.to_numpy()
    return xedges, yedges, values, h.errors()


def _row_binning(pt_edges, values, errors, valtype, *, truncate_last):
    if truncate_last:
        pt_edges, values, errors = pt_edges[:-1], values[:-1], errors[:-1]
    if valtype == "up":
        content = values + errors
    elif valtype == "down":
        content = values - errors
    else:
        content = values
    return cs.Binning(nodetype="binning", input="pt", edges=[float(e) for e in pt_edges],
                      content=[float(c) for c in content], flow="clamp")


def _eta_binning(barrel, endcap, valtype):
    xedges_b, yedges_b, vals_b, errs_b = barrel
    xedges_e, yedges_e, vals_e, errs_e = endcap
    if not np.array_equal(xedges_b, xedges_e):
        raise RuntimeError("Barrel and EndCap histograms have different eta binning")
    abs_lo = np.abs(xedges_b[:-1])
    abs_hi = np.abs(xedges_b[1:])
    rows = []
    for i in range(len(xedges_b) - 1):
        lo, hi = min(abs_lo[i], abs_hi[i]), max(abs_lo[i], abs_hi[i])
        if hi <= 1.444 + 1e-9:            # barrel
            rows.append(_row_binning(yedges_b, vals_b[i], errs_b[i], valtype,
                                     truncate_last=yedges_b[-2] >= PT_CLAMP_MAX - 1e-9))
        elif lo >= 1.566 - 1e-9:          # endcap
            rows.append(_row_binning(yedges_e, vals_e[i], errs_e[i], valtype, truncate_last=False))
        else:                             # crack: empty in the source file
            rows.append(0.0)
    return cs.Binning(nodetype="binning", input="eta", edges=[float(e) for e in xedges_b],
                      content=rows, flow="clamp")


def _correction(name, description, barrel, endcap, year_key):
    valtype_node = cs.Category(
        nodetype="category", input="ValType",
        content=[cs.CategoryItem(key=v, value=cs.Category(
            nodetype="category", input="Path",
            content=[cs.CategoryItem(key=PATH_NAME, value=_eta_binning(barrel, endcap, v))],
        )) for v in VALTYPES],
    )
    return cs.Correction(
        name=name, description=description, version=1,
        inputs=[
            cs.Variable(name="year", type="string", description="Data-taking year"),
            cs.Variable(name="ValType", type="string", description="nom, up (value+error), down (value-error)"),
            cs.Variable(name="Path", type="string", description=f"Trigger selection; only '{PATH_NAME}'"),
            cs.Variable(name="eta", type="real", description="Electron supercluster eta"),
            cs.Variable(name="pt", type="real", description="Electron pT [GeV]"),
        ],
        output=cs.Variable(name="weight", type="real", description="Efficiency or scale factor"),
        data=cs.Category(nodetype="category", input="year",
                         content=[cs.CategoryItem(key=year_key, value=valtype_node)]),
    )


def convert(skflat_data: Path, year: int) -> cs.CorrectionSet:
    src = skflat_data / str(year) / "ID" / "Electron" / "WREGammaTriggerEff.root"
    with uproot.open(src) as f:
        hists = {prefix: (_read(f, f"{prefix}_TH2F_Barrel"), _read(f, f"{prefix}_TH2F_EndCap"))
                 for prefix in ("DataEff", "MCEff", "SF")}
    year_key = YEAR_KEYS[year]
    corrections = [
        _correction("Electron-HLT-DataEff", "Per-electron trigger efficiency in data (HEEP, electron trigger OR)",
                    *hists["DataEff"], year_key),
        _correction("Electron-HLT-McEff", "Per-electron trigger efficiency in MC (HEEP, electron trigger OR)",
                    *hists["MCEff"], year_key),
        _correction("Electron-HLT-SF", "Per-electron trigger scale factor data/MC (HEEP, electron trigger OR)",
                    *hists["SF"], year_key),
    ]
    description = (
        f"Run 2 WR electron trigger efficiencies for HEEP electrons, converted from "
        f"{SOURCE_REPO} ({SOURCE_BRANCH}, commit {_git_commit(skflat_data)}) file "
        f"{src.relative_to(skflat_data.parent)} on {date.today().isoformat()}. "
        "pT is clamped to [50, 500) as in HNWRAnalyzer (MCCorrection::ElectronTrigger_Eff)."
    )
    return cs.CorrectionSet(schema_version=2, description=description, corrections=corrections)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skflat-data", type=Path, required=True, help="SKFlatAnalyzer data/Run2Legacy_v4 directory")
    parser.add_argument("--year", type=int, default=2018, choices=sorted(YEAR_KEYS))
    parser.add_argument("--out", type=Path, required=True, help="Output .json or .json.gz path")
    args = parser.parse_args()

    cset = convert(args.skflat_data, args.year)
    text = cset.model_dump_json(exclude_unset=True, indent=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gz":
        with gzip.open(args.out, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
