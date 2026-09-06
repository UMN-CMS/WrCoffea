#!/usr/bin/env python3
"""Convert the Run 2 HNWRAnalyzer DY Z-pT reweighting histograms to JSON.

The Run 2 analysis (SKFlatAnalyzer, branch ``HNWR/Run2Legacy_v4``,
``Analyzers/src/HNWRAnalyzer.C``) reweights the LO MadGraph HT-binned DY
sample as a function of the generator-level Z pT with two factors:

* ``HNWRDYPtReweight/ZPtQCDWithError.root`` - the NLO/LO QCD ratio
  (``h_Ratio_PDFError`` carries the central value and the PDF uncertainty;
  ``h_Ratio_ScaleUp/Down`` and ``h_Ratio_AlphaSUp/Down`` the QCD scale and
  alpha_S variations).  ``ptZ >= 8000`` is clamped to 7999.
* ``HNWRDYPtReweight/ZPtEWCorr.root`` - the electroweak correction
  (``hist_v`` central; ``hist_e1/e2/e3`` carry three uncertainty
  components as bin errors).  ``ptZ < 30`` is evaluated at 31 and
  ``ptZ >= 6500`` at 6499.

This script writes both as one JSON payload that ``wrcoffea.dy_reweight``
evaluates with plain NumPy, so the analysis workers need neither ROOT nor
the SKFlat checkout.

Example::

    python scripts/convert_run2_dy_reweight.py \
        --skflat-data /path/to/SKFlatAnalyzer/data/Run2Legacy_v4 --year 2018 \
        --out data/reweights/RunII/2018/RunIISummer20UL18/dy_zpt_reweight_run2legacy_v4_2018.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import uproot

SOURCE_REPO = "https://github.com/jedori0228/SKFlatAnalyzer"
SOURCE_BRANCH = "HNWR/Run2Legacy_v4"

# Clamps used by HNWRAnalyzer::executeEvent when looking up the histograms.
QCD_PT_MAX = 7999.0
EW_PT_MIN = 31.0
EW_PT_MAX = 6499.0


def _th1(f, name):
    h = f[name]
    values, edges = h.to_numpy()
    return edges.tolist(), values.tolist(), h.errors().tolist()


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def convert(skflat_data: Path, year: int) -> dict:
    qcd_file = skflat_data / str(year) / "HNWRDYPtReweight" / "ZPtQCDWithError.root"
    ew_file = skflat_data / str(year) / "HNWRDYPtReweight" / "ZPtEWCorr.root"

    with uproot.open(qcd_file) as f:
        edges, central, pdf_err = _th1(f, "h_Ratio_PDFError")
        edges_c, central_plain, _ = _th1(f, "h_Ratio_Central")
        _, scale_up, _ = _th1(f, "h_Ratio_ScaleUp")
        _, scale_down, _ = _th1(f, "h_Ratio_ScaleDown")
        _, alphas_up, _ = _th1(f, "h_Ratio_AlphaSUp")
        _, alphas_down, _ = _th1(f, "h_Ratio_AlphaSDown")
    if edges != edges_c:
        raise RuntimeError("QCD ratio histograms have inconsistent binning")

    with uproot.open(ew_file) as f:
        ew_edges, ew_central, _ = _th1(f, "hist_v")
        _, e1_val, e1_err = _th1(f, "hist_e1")
        _, e2_val, e2_err = _th1(f, "hist_e2")
        _, e3_val, e3_err = _th1(f, "hist_e3")
    for other in (e1_val, e2_val, e3_val):
        if any(abs(a - b) > 1e-9 for a, b in zip(other, ew_central)):
            raise RuntimeError("EW error histograms do not share the central values of hist_v")

    return {
        "provenance": {
            "source_repo": SOURCE_REPO,
            "source_branch": SOURCE_BRANCH,
            "source_commit": _git_commit(skflat_data),
            "source_files": [
                str(qcd_file.relative_to(skflat_data.parent)),
                str(ew_file.relative_to(skflat_data.parent)),
            ],
            "year": year,
            "converted_on": date.today().isoformat(),
            "analyzer_reference": "Analyzers/src/HNWRAnalyzer.C::executeEvent (ApplyDYPtReweight)",
            "note": (
                "Derived for the 13 TeV 2018 LO madgraphMLM HT-binned DYJetsToLL_M-50 "
                "samples used by the Run 2 WR analysis (AN-19-083). The QCD ratio is "
                "NLO/LO as a function of generator-level Z pT; the EW correction is "
                "the electroweak K-factor with three uncertainty components."
            ),
        },
        "variable": "gen_z_pt",
        "qcd": {
            "edges": edges,
            "central": central,
            "central_plain": central_plain,
            "pdf_err": pdf_err,
            "scale_up": scale_up,
            "scale_down": scale_down,
            "alphas_up": alphas_up,
            "alphas_down": alphas_down,
            "pt_max": QCD_PT_MAX,
        },
        "ew": {
            "edges": ew_edges,
            "central": ew_central,
            "e1_err": e1_err,
            "e2_err": e2_err,
            "e3_err": e3_err,
            "pt_min": EW_PT_MIN,
            "pt_max": EW_PT_MAX,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skflat-data", type=Path, required=True,
                        help="SKFlatAnalyzer data/Run2Legacy_v4 directory")
    parser.add_argument("--year", type=int, default=2018, choices=(2016, 2017, 2018))
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    payload = convert(args.skflat_data, args.year)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"Wrote {args.out} ({len(payload['qcd']['central'])} QCD bins, {len(payload['ew']['central'])} EW bins)")


if __name__ == "__main__":
    main()
