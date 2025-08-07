#!/usr/bin/env python3
import argparse
import uproot
import json
import numpy as np
import os

# ─── Configuration ─────────────────────────────────────────────────────────────
ERA      = "RunIISummer20UL18"
RUN      = "RunII"
YEAR     = "2018"
LUMI_PB  = 59.83 * 1000   # 2018: 59.83 fb⁻¹ → 59830 pb⁻¹

# ─── Background files ──────────────────────────────────────────────────────────
# Paths under WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor
BKG_FILES = [
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_TTbar.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_TW.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_WJets.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_SingleTop.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_TTbarSemileptonic.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_TTV.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_Diboson.root",
    f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor/WRAnalyzer_Triboson.root",
]

def derive_sf(dy_path, ref_path, region, edges_out, variable, lumi_pb, bkg_paths):
    key = f"{region}/{variable}_{region}"

    # — DYJets MC yields
    f_dy = uproot.open(dy_path)
    counts_dy, edges_in = f_dy[key].to_numpy()
    counts_dy *= lumi_pb
    centers = 0.5 * (edges_in[:-1] + edges_in[1:])
    dy_rb, _ = np.histogram(centers, bins=edges_out, weights=counts_dy)

    # — "Data" in CR
    f_ref = uproot.open(ref_path)
    counts_ref, _ = f_ref[key].to_numpy()
    data_rb, _ = np.histogram(centers, bins=edges_out, weights=counts_ref)

    # — Sum non-DY backgrounds
    other_rb = np.zeros_like(dy_rb)
    for path in bkg_paths:
        h_bkg = uproot.open(path)[key]
        cnt_b, _ = h_bkg.to_numpy()
        cnt_b *= lumi_pb
        rb, _ = np.histogram(centers, bins=edges_out, weights=cnt_b)
        other_rb += rb

    # — Compute scale factors
    sf = np.ones_like(dy_rb, dtype=float)
    mask = dy_rb > 0
    sf[mask] = (data_rb[mask] - other_rb[mask]) / dy_rb[mask]

    return np.array(edges_out), sf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Derive per-bin DY scale factors for RunIISummer20UL18."
    )
    parser.add_argument(
        "--variable",
        required=True,
        help="Which variable to SF-correct, e.g. mass_dijet"
    )
    return parser.parse_args()

def main():
    args     = parse_args()
    variable = args.variable

    # build file paths for DY and data CRs
    base = f"WR_Plotter/rootfiles/{RUN}/{YEAR}/{ERA}/dy_ht_lo_kfactor"
    dy_file     = f"{base}/WRAnalyzer_DYJets.root"
    egamma_file = f"{base}/WRAnalyzer_EGamma.root"
    muon_file   = f"{base}/WRAnalyzer_Muon.root"

    # backgrounds
    bkg_paths = BKG_FILES

    # bin edges
    new_edges = [0, 200, 400, 600, 800, 1000, 1250, 1500, 2000, 4000]

    # control regions
    ee_cr = "wr_ee_resolved_dy_cr"
    mm_cr = "wr_mumu_resolved_dy_cr"
    edges, sf_EE = derive_sf(
        dy_file, egamma_file, ee_cr,
        new_edges, variable, LUMI_PB, bkg_paths
    )
    _, sf_MM = derive_sf(
        dy_file, muon_file, mm_cr,
        new_edges, variable, LUMI_PB, bkg_paths
    )

    # output
    out = {
        "edges": edges.tolist(),
        "sf_ee_resolved_dy_cr":   sf_EE.tolist(),
        "sf_mumu_resolved_dy_cr": sf_MM.tolist(),
    }
    out_dir = os.path.join("data", "reweights", RUN, YEAR, ERA)
    os.makedirs(out_dir, exist_ok=True)
    out_filename = os.path.join(out_dir, f"{variable}_sf.json")
    with open(out_filename, "w") as jf:
        json.dump(out, jf, indent=2)

    print(f"Wrote scale factors to {out_filename}")

if __name__ == "__main__":
    main()
