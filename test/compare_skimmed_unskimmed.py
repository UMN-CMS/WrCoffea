#!/usr/bin/env python3
"""Compare ROOT files between SKIMMED and UNSKIMMED directories."""

import uproot
import numpy as np
import os

SKIMMED = "WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/20260210_SKIMMED"
UNSKIMMED = "WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/20260210_UNSKIMMED"

def get_root_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith('.root')])

def is_histogram(obj):
    """Check if object is a histogram type we can compare."""
    classname = obj.classname if hasattr(obj, 'classname') else type(obj).__name__
    return any(t in classname for t in ['TH1', 'TH2', 'TH3'])

def compare_files(fname):
    skim_path = os.path.join(SKIMMED, fname)
    unskim_path = os.path.join(UNSKIMMED, fname)

    diffs = []

    with uproot.open(skim_path) as f_skim, uproot.open(unskim_path) as f_unskim:
        # Get all histogram keys (classname-filtered)
        skim_keys = set(f_skim.keys(filter_classname=["TH1*", "TH2*", "TH3*"], cycle=False))
        unskim_keys = set(f_unskim.keys(filter_classname=["TH1*", "TH2*", "TH3*"], cycle=False))

        only_skim = skim_keys - unskim_keys
        only_unskim = unskim_keys - skim_keys

        if only_skim:
            diffs.append(f"  Histograms only in SKIMMED: {sorted(only_skim)}")
        if only_unskim:
            diffs.append(f"  Histograms only in UNSKIMMED: {sorted(only_unskim)}")

        common_keys = sorted(skim_keys & unskim_keys)

        for key in common_keys:
            obj_skim = f_skim[key]
            obj_unskim = f_unskim[key]

            vals_s = obj_skim.values()
            vals_u = obj_unskim.values()

            sum_s = float(np.sum(vals_s))
            sum_u = float(np.sum(vals_u))

            # Compare bin counts
            if vals_s.shape != vals_u.shape:
                diffs.append(f"  {key}: SHAPE MISMATCH skim={vals_s.shape} vs unskim={vals_u.shape}")
                continue

            if not np.allclose(vals_s, vals_u, rtol=1e-6, atol=1e-10):
                max_diff = float(np.max(np.abs(vals_s - vals_u)))
                nonzero = np.abs(vals_u) > 0
                rel_diff = float(np.max(np.abs(vals_s - vals_u)[nonzero] / np.abs(vals_u)[nonzero])) if np.any(nonzero) else 0.0
                diffs.append(
                    f"  {key}: VALUES DIFFER | "
                    f"skim_sum={sum_s:.6f} unskim_sum={sum_u:.6f} "
                    f"diff={sum_s - sum_u:.6f} "
                    f"max_bin_diff={max_diff:.6e} max_rel_diff={rel_diff:.6e}"
                )
                # If cutflow, print bin-by-bin comparison
                if 'cutflow' in key.lower():
                    try:
                        axis = obj_skim.axis()
                        labels = list(axis.labels()) if hasattr(axis, 'labels') else None
                    except Exception:
                        labels = None

                    for i in range(len(vals_s)):
                        if not np.isclose(vals_s[i], vals_u[i], rtol=1e-6, atol=1e-10):
                            label = labels[i] if labels and i < len(labels) else f"bin{i}"
                            diffs.append(
                                f"    bin {i} ({label}): skim={vals_s[i]:.6f} unskim={vals_u[i]:.6f} "
                                f"diff={vals_s[i] - vals_u[i]:.6f}"
                            )

            # Check variances
            try:
                var_s = obj_skim.variances()
                var_u = obj_unskim.variances()
                if var_s is not None and var_u is not None:
                    if not np.allclose(var_s, var_u, rtol=1e-6, atol=1e-10):
                        max_var_diff = float(np.max(np.abs(var_s - var_u)))
                        diffs.append(
                            f"  {key}: VARIANCES DIFFER | max_diff={max_var_diff:.6e}"
                        )
            except Exception:
                pass

    return diffs


def main():
    skim_files = get_root_files(SKIMMED)
    unskim_files = get_root_files(UNSKIMMED)

    common = sorted(set(skim_files) & set(unskim_files))
    only_skim = sorted(set(skim_files) - set(unskim_files))
    only_unskim = sorted(set(unskim_files) - set(skim_files))

    print(f"SKIMMED files: {len(skim_files)}")
    print(f"UNSKIMMED files: {len(unskim_files)}")
    print(f"Common files: {len(common)}")

    if only_skim:
        print(f"\nFiles only in SKIMMED: {only_skim}")
    if only_unskim:
        print(f"\nFiles only in UNSKIMMED: {only_unskim}")

    print("\n" + "="*80)

    any_diff = False
    for fname in common:
        print(f"\n--- {fname} ---")
        diffs = compare_files(fname)
        if diffs:
            any_diff = True
            for d in diffs:
                print(d)
        else:
            print("  MATCH (all histograms identical)")

    print("\n" + "="*80)
    if any_diff:
        print("\nRESULT: DIFFERENCES FOUND (see above)")
    else:
        print("\nRESULT: ALL FILES MATCH")


if __name__ == "__main__":
    main()
