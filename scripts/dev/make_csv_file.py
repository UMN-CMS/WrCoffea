#!/usr/bin/env python3
import argparse
import csv
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import awkward as ak
import uproot

# ------------------------------------------------------------
# Dataset list
# ------------------------------------------------------------
DATASETS = [
    "/WRtoNEltoElElJJ_MWR2000_N1100_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
    "/WRtoNMutoMuMuJJ_MWR2000_N1100_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
    "/WRtoNEltoElElJJ_MWR4000_N2100_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
    "/WRtoNMutoMuMuJJ_MWR4000_N2100_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
    "/WRtoNEltoElElJJ_MWR6000_N3100_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
    "/WRtoNMutoMuMuJJ_MWR6000_N3100_TuneCP5_13p6TeV_madgraph-pythia8/Run3Summer22EENanoAODv12-130X_mcRun3_2022_realistic_postEE_v6-v2/NANOAODSIM",
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def list_files_from_das(dataset: str) -> List[str]:
    out = subprocess.check_output(
        ["dasgoclient", "-query", f"file dataset={dataset}"],
        text=True,
    )
    return [l.strip() for l in out.splitlines() if l.strip()]


def four_vector_from_pt_eta_phi_m(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    p = np.sqrt(px**2 + py**2 + pz**2)
    E = np.sqrt(p**2 + mass**2)
    return px, py, pz, E


def infer_output_name(dataset: str) -> str:
    base = dataset.strip("/").split("/")[0]
    if "_Tune" in base:
        base = base.split("_Tune")[0]
    return base + ".csv"


def infer_lep_pdg(dataset: str) -> Optional[int]:
    """Return 11 for ee samples, 13 for mumu samples, else None."""
    name = dataset.lower()
    if "eltoeleljj" in name:
        return 11
    if "mutomumujj" in name:
        return 13
    return None


# ------------------------------------------------------------
# Processing (reco-level leptons/jets only)
# ------------------------------------------------------------
def process_dataset(dataset: str, max_events: int, redirector: str) -> str:
    files = list_files_from_das(dataset)
    out_name = infer_output_name(dataset)
    lep_pdg = infer_lep_pdg(dataset)

    if lep_pdg == 11:
        lep_prefix = "Electron"
    elif lep_pdg == 13:
        lep_prefix = "Muon"
    else:
        raise RuntimeError("Could not determine lepton type from dataset name: %s" % dataset)

    print("\nProcessing %s" % dataset)
    print("  → %s (lep %s from %s collection)" % (out_name, lep_pdg, lep_prefix))

    # Only leptons + jets in the CSV
    cols = [
        "L1_mass","L1_pt","L1_eta","L1_phi","L1_E","L1_px","L1_py","L1_pz",
        "L2_mass","L2_pt","L2_eta","L2_phi","L2_E","L2_px","L2_py","L2_pz",
        "J1_mass","J1_pt","J1_eta","J1_phi","J1_E","J1_px","J1_py","J1_pz",
        "J2_mass","J2_pt","J2_eta","J2_phi","J2_E","J2_px","J2_py","J2_pz",
    ]

    total = 0
    DEBUG_MAX_SKIPS = 50   # max skipped-event messages per dataset
    skipped_printed = 0

    with open(out_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()

        # branches to read
        branches = [
            "%s_pt"   % lep_prefix,
            "%s_eta"  % lep_prefix,
            "%s_phi"  % lep_prefix,
            "%s_mass" % lep_prefix,
            "Jet_pt",
            "Jet_eta",
            "Jet_phi",
            "Jet_mass",
        ]

        for i_file, short in enumerate(files, 1):
            if max_events and total >= max_events:
                break
            full = redirector + short
            print("  [%d/%d] %s" % (i_file, len(files), full))

            try:
                root_file = uproot.open(full)
            except Exception as e:
                print("   [SKIP FILE] failed to open: %s" % e)
                continue

            if "Events" not in root_file:
                print("   [SKIP FILE] no 'Events' tree found")
                root_file.close()
                continue

            for arr in root_file["Events"].iterate(
                filter_name=branches,
                step_size=50000,
                library="ak",
            ):
                if max_events and total >= max_events:
                    break

                # Reco leptons (either electrons or muons)
                lep_pt   = arr["%s_pt"   % lep_prefix]
                lep_eta  = arr["%s_eta"  % lep_prefix]
                lep_phi  = arr["%s_phi"  % lep_prefix]
                lep_mass = arr["%s_mass" % lep_prefix]

                # Reco jets
                jet_pt   = arr["Jet_pt"]
                jet_eta  = arr["Jet_eta"]
                jet_phi  = arr["Jet_phi"]
                jet_mass = arr["Jet_mass"]

                # Simple multiplicity requirements: at least 2 leptons and 2 jets
                lep_counts = np.array(ak.num(lep_pt, axis=1))
                jet_counts = np.array(ak.num(jet_pt, axis=1))

                has_two_lep = lep_counts >= 2
                has_two_jet = jet_counts >= 2

                good = has_two_lep & has_two_jet

                # --------------------------------------------------------
                # Explain why events are skipped (up to DEBUG_MAX_SKIPS)
                # --------------------------------------------------------
                if skipped_printed < DEBUG_MAX_SKIPS:
                    n_events_chunk = len(lep_counts)
                    for iev in range(n_events_chunk):
                        if bool(good[iev]):
                            continue  # this event will be kept

                        reasons = []
                        if not has_two_lep[iev]:
                            reasons.append("n_leptons=%d < 2" % lep_counts[iev])
                        if not has_two_jet[iev]:
                            reasons.append("n_jets=%d < 2" % jet_counts[iev])
                        if not reasons:
                            reasons.append("failed selection for unknown reason")

                        print("    [SKIP EVENT] %s" % "; ".join(reasons))
                        skipped_printed += 1
                        if skipped_printed >= DEBUG_MAX_SKIPS:
                            break
                # --------------------------------------------------------

                if not ak.any(good):
                    # no events in this chunk survive; move to next chunk
                    continue

                # Order leptons/jets by pT and take the two leading
                lep_order = ak.argsort(lep_pt, axis=1, ascending=False)
                jet_order = ak.argsort(jet_pt, axis=1, ascending=False)

                lep_idx2 = lep_order[:, :2]
                jet_idx2 = jet_order[:, :2]

                lep_pt2   = lep_pt[lep_idx2]
                lep_eta2  = lep_eta[lep_idx2]
                lep_phi2  = lep_phi[lep_idx2]
                lep_mass2 = lep_mass[lep_idx2]

                jet_pt2   = jet_pt[jet_idx2]
                jet_eta2  = jet_eta[jet_idx2]
                jet_phi2  = jet_phi[jet_idx2]
                jet_mass2 = jet_mass[jet_idx2]

                # Keep only 'good' events
                lep_pt2_g   = lep_pt2[good]
                lep_eta2_g  = lep_eta2[good]
                lep_phi2_g  = lep_phi2[good]
                lep_mass2_g = lep_mass2[good]

                jet_pt2_g   = jet_pt2[good]
                jet_eta2_g  = jet_eta2[good]
                jet_phi2_g  = jet_phi2[good]
                jet_mass2_g = jet_mass2[good]

                # Convert to numpy
                lep_pt_np   = np.array(lep_pt2_g)   # (n_evt, 2)
                lep_eta_np  = np.array(lep_eta2_g)
                lep_phi_np  = np.array(lep_phi2_g)
                lep_mass_np = np.array(lep_mass2_g)

                jet_pt_np   = np.array(jet_pt2_g)   # (n_evt, 2)
                jet_eta_np  = np.array(jet_eta2_g)
                jet_phi_np  = np.array(jet_phi2_g)
                jet_mass_np = np.array(jet_mass2_g)

                if lep_pt_np.shape[0] == 0:
                    # Shouldn't happen if ak.any(good) is True, but be defensive.
                    continue

                # Split leptons/jets into 1 and 2
                L1_pt, L2_pt     = lep_pt_np[:, 0], lep_pt_np[:, 1]
                L1_eta, L2_eta   = lep_eta_np[:, 0], lep_eta_np[:, 1]
                L1_phi, L2_phi   = lep_phi_np[:, 0], lep_phi_np[:, 1]
                L1_mass, L2_mass = lep_mass_np[:, 0], lep_mass_np[:, 1]

                J1_pt, J2_pt     = jet_pt_np[:, 0], jet_pt_np[:, 1]
                J1_eta, J2_eta   = jet_eta_np[:, 0], jet_eta_np[:, 1]
                J1_phi, J2_phi   = jet_phi_np[:, 0], jet_phi_np[:, 1]
                J1_mass, J2_mass = jet_mass_np[:, 0], jet_mass_np[:, 1]

                # Build 4-vectors for leptons/jets
                L1_px, L1_py, L1_pz, L1_E = four_vector_from_pt_eta_phi_m(
                    L1_pt, L1_eta, L1_phi, L1_mass
                )
                L2_px, L2_py, L2_pz, L2_E = four_vector_from_pt_eta_phi_m(
                    L2_pt, L2_eta, L2_phi, L2_mass
                )

                J1_px, J1_py, J1_pz, J1_E = four_vector_from_pt_eta_phi_m(
                    J1_pt, J1_eta, J1_phi, J1_mass
                )
                J2_px, J2_py, J2_pz, J2_E = four_vector_from_pt_eta_phi_m(
                    J2_pt, J2_eta, J2_phi, J2_mass
                )

                n_rows = len(L1_pt)
                for idx in range(n_rows):
                    if max_events and total >= max_events:
                        break

                    w.writerow({
                        # Lepton 1
                        "L1_mass": L1_mass[idx],
                        "L1_pt":   L1_pt[idx],
                        "L1_eta":  L1_eta[idx],
                        "L1_phi":  L1_phi[idx],
                        "L1_E":    L1_E[idx],
                        "L1_px":   L1_px[idx],
                        "L1_py":   L1_py[idx],
                        "L1_pz":   L1_pz[idx],
                        # Lepton 2
                        "L2_mass": L2_mass[idx],
                        "L2_pt":   L2_pt[idx],
                        "L2_eta":  L2_eta[idx],
                        "L2_phi":  L2_phi[idx],
                        "L2_E":    L2_E[idx],
                        "L2_px":   L2_px[idx],
                        "L2_py":   L2_py[idx],
                        "L2_pz":   L2_pz[idx],
                        # Jet 1
                        "J1_mass": J1_mass[idx],
                        "J1_pt":   J1_pt[idx],
                        "J1_eta":  J1_eta[idx],
                        "J1_phi":  J1_phi[idx],
                        "J1_E":    J1_E[idx],
                        "J1_px":   J1_px[idx],
                        "J1_py":   J1_py[idx],
                        "J1_pz":   J1_pz[idx],
                        # Jet 2
                        "J2_mass": J2_mass[idx],
                        "J2_pt":   J2_pt[idx],
                        "J2_eta":  J2_eta[idx],
                        "J2_phi":  J2_phi[idx],
                        "J2_E":    J2_E[idx],
                        "J2_px":   J2_px[idx],
                        "J2_py":   J2_py[idx],
                        "J2_pz":   J2_pz[idx],
                    })
                    total += 1

            root_file.close()

    print("  Done: %d events → %s" % (total, out_name))
    return out_name


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-events", type=int, default=180000)
    p.add_argument("--redirector", default="root://cms-xrd-global.cern.ch/")
    args = p.parse_args()

    csv_files = []
    for ds in DATASETS:
        csv_files.append(
            process_dataset(ds, args.max_events, args.redirector)
        )

    # --- Zip everything ---
    zipname = "WR_datasets_reco.zip"
    print("\nZipping %d CSVs → %s" % (len(csv_files), zipname))
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for fpath in csv_files:
            z.write(fpath)
    print("All done ✅ → %s" % zipname)
