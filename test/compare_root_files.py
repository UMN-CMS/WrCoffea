#!/usr/bin/env python3
"""Compare ROOT files between skimmed and unskimmed directories."""

import uproot
import numpy as np
import sys

DIR_A = "/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/20260210_skimmed/"
DIR_B = "/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/feb_unskimmed/"

LABEL_A = "skimmed (20260210)"
LABEL_B = "unskimmed (feb)"

COMMON_FILES = ["WRAnalyzer_DYJets.root", "WRAnalyzer_Muon.root"]

summary = {}


def is_histogram(obj):
    """Check if an uproot object is a histogram (not a directory)."""
    return hasattr(obj, 'values') and hasattr(obj, 'axes') and not isinstance(obj, uproot.ReadOnlyDirectory)


def compare_histogram(hist_a, hist_b, key):
    """Compare two uproot histogram objects. Returns (match: bool, details: str)."""
    details_lines = []
    match = True

    try:
        vals_a = np.asarray(hist_a.values())
        vals_b = np.asarray(hist_b.values())
    except Exception as e:
        return False, f"  Could not read values: {e}"

    if vals_a.shape != vals_b.shape:
        details_lines.append(f"  SHAPE MISMATCH: {LABEL_A}={vals_a.shape}, {LABEL_B}={vals_b.shape}")
        return False, "\n".join(details_lines)

    if np.allclose(vals_a, vals_b, rtol=0, atol=0):
        details_lines.append(f"  Values: EXACT MATCH (nbins={len(vals_a)})")
    elif np.allclose(vals_a, vals_b, rtol=1e-12, atol=1e-12):
        max_diff = np.max(np.abs(vals_a - vals_b))
        max_rel = np.max(np.abs(vals_a - vals_b) / np.where(vals_a != 0, np.abs(vals_a), 1.0))
        details_lines.append(f"  Values: CLOSE MATCH (max_abs_diff={max_diff:.2e}, max_rel_diff={max_rel:.2e})")
    else:
        match = False
        diff = vals_a - vals_b
        nonzero_mask = diff != 0
        n_diff = np.sum(nonzero_mask)
        max_abs_diff = np.max(np.abs(diff))
        sorted_idx = np.argsort(np.abs(diff))[::-1]
        details_lines.append(f"  Values: DIFFER in {n_diff}/{len(vals_a)} bins, max_abs_diff={max_abs_diff:.6e}")
        n_show = min(10, int(n_diff))
        details_lines.append(f"  Top {n_show} differing bins:")
        for i in range(n_show):
            idx = sorted_idx[i]
            details_lines.append(
                f"    bin[{idx}]: {LABEL_A}={vals_a[idx]:.6e}, {LABEL_B}={vals_b[idx]:.6e}, diff={diff[idx]:.6e}"
            )

    int_a = np.sum(vals_a)
    int_b = np.sum(vals_b)
    if int_a == int_b:
        details_lines.append(f"  Integral: EXACT MATCH = {int_a:.6e}")
    else:
        rel_diff = abs(int_a - int_b) / abs(int_a) if int_a != 0 else float('inf')
        details_lines.append(
            f"  Integral: {LABEL_A}={int_a:.6e}, {LABEL_B}={int_b:.6e}, "
            f"diff={int_a - int_b:.6e}, rel={rel_diff:.6e}"
        )
        if not np.isclose(int_a, int_b, rtol=1e-12, atol=1e-12):
            match = False

    try:
        var_a = hist_a.variances()
        var_b = hist_b.variances()
        if var_a is not None and var_b is not None:
            var_a = np.asarray(var_a)
            var_b = np.asarray(var_b)
            if np.allclose(var_a, var_b, rtol=0, atol=0):
                details_lines.append(f"  Sumw2/variances: EXACT MATCH")
            elif np.allclose(var_a, var_b, rtol=1e-12, atol=1e-12):
                max_diff = np.max(np.abs(var_a - var_b))
                details_lines.append(f"  Sumw2/variances: CLOSE MATCH (max_abs_diff={max_diff:.2e})")
            else:
                match = False
                diff_v = var_a - var_b
                n_diff_v = np.sum(diff_v != 0)
                max_abs_diff_v = np.max(np.abs(diff_v))
                details_lines.append(
                    f"  Sumw2/variances: DIFFER in {n_diff_v}/{len(var_a)} bins, max_abs_diff={max_abs_diff_v:.6e}"
                )
                sorted_idx_v = np.argsort(np.abs(diff_v))[::-1]
                n_show_v = min(5, int(n_diff_v))
                for i in range(n_show_v):
                    idx = sorted_idx_v[i]
                    details_lines.append(
                        f"    var_bin[{idx}]: {LABEL_A}={var_a[idx]:.6e}, {LABEL_B}={var_b[idx]:.6e}"
                    )
        elif var_a is None and var_b is None:
            details_lines.append(f"  Sumw2/variances: both None")
        else:
            match = False
            details_lines.append(f"  Sumw2/variances: one is None")
    except Exception as e:
        details_lines.append(f"  Sumw2/variances: could not compare ({e})")

    try:
        flow_a = np.asarray(hist_a.values(flow=True))
        flow_b = np.asarray(hist_b.values(flow=True))
        uf_a, of_a = flow_a[0], flow_a[-1]
        uf_b, of_b = flow_b[0], flow_b[-1]
        if uf_a != uf_b or of_a != of_b:
            details_lines.append(
                f"  Flow: underflow {LABEL_A}={uf_a:.6e} vs {LABEL_B}={uf_b:.6e}; "
                f"overflow {LABEL_A}={of_a:.6e} vs {LABEL_B}={of_b:.6e}"
            )
            if not (np.isclose(uf_a, uf_b, rtol=1e-12) and np.isclose(of_a, of_b, rtol=1e-12)):
                match = False
        else:
            details_lines.append(f"  Flow: underflow={uf_a:.6e}, overflow={of_a:.6e} (match)")
    except Exception:
        pass

    return match, "\n".join(details_lines)


def print_cutflow(hist_a, hist_b, key):
    """Print cutflow histogram side-by-side with bin labels."""
    print(f"\n  === CUTFLOW: {key} ===")

    vals_a = np.asarray(hist_a.values())
    vals_b = np.asarray(hist_b.values())

    # Try to get labels from the axis
    labels = None
    try:
        axis_a = hist_a.axis()
        labels = list(axis_a.labels())
    except (AttributeError, TypeError):
        pass

    if labels is None:
        try:
            axis_a = hist_a.axis()
            labels = [str(axis_a.bin(i)) for i in range(len(vals_a))]
        except Exception:
            labels = [f"bin_{i}" for i in range(len(vals_a))]

    max_label_len = max(len(str(l)) for l in labels) if labels else 10

    header = f"  {'Bin Label':<{max_label_len}}  {'Skimmed':>16}  {'Unskimmed':>16}  {'Diff':>16}  {'RelDiff':>12}"
    print(header)
    print("  " + "-" * len(header))

    for i in range(min(len(vals_a), len(vals_b))):
        label = labels[i] if i < len(labels) else f"bin_{i}"
        va = vals_a[i]
        vb = vals_b[i]
        diff = va - vb
        rel = diff / vb if vb != 0 else (0.0 if diff == 0 else float('inf'))
        marker = " <--" if abs(diff) > 1e-12 else ""
        print(f"  {str(label):<{max_label_len}}  {va:>16.6f}  {vb:>16.6f}  {diff:>16.6f}  {rel:>12.2e}{marker}")

    if len(vals_a) > len(vals_b):
        print(f"\n  {LABEL_A} has {len(vals_a) - len(vals_b)} extra bins")
    elif len(vals_b) > len(vals_a):
        print(f"\n  {LABEL_B} has {len(vals_b) - len(vals_a)} extra bins")


def compare_file(filename):
    """Compare a single ROOT file between the two directories."""
    path_a = DIR_A + filename
    path_b = DIR_B + filename

    print(f"\n{'='*80}")
    print(f"COMPARING: {filename}")
    print(f"  A ({LABEL_A}): {path_a}")
    print(f"  B ({LABEL_B}): {path_b}")
    print(f"{'='*80}")

    file_a = uproot.open(path_a)
    file_b = uproot.open(path_b)

    # Get histogram keys only (filter out directories by using cycle-stripped classnames)
    keys_a = set(file_a.keys(filter_classname=["TH1*", "TH2*", "TH3*", "TProfile*"]))
    keys_b = set(file_b.keys(filter_classname=["TH1*", "TH2*", "TH3*", "TProfile*"]))

    # Also get ALL keys for reporting
    all_keys_a = set(file_a.keys())
    all_keys_b = set(file_b.keys())

    common_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f"\n  Total keys in {LABEL_A}: {len(all_keys_a)}")
    print(f"  Total keys in {LABEL_B}: {len(all_keys_b)}")
    print(f"  Histogram keys in {LABEL_A}: {len(keys_a)}")
    print(f"  Histogram keys in {LABEL_B}: {len(keys_b)}")
    print(f"  Common histogram keys: {len(common_keys)}")

    if only_a:
        print(f"\n  Histogram keys ONLY in {LABEL_A} ({len(only_a)}):")
        for k in only_a:
            print(f"    {k}")
    if only_b:
        print(f"\n  Histogram keys ONLY in {LABEL_B} ({len(only_b)}):")
        for k in only_b:
            print(f"    {k}")

    file_summary = {}
    cutflow_keys = []

    print(f"\n--- Comparing {len(common_keys)} common histograms ---")

    for key in common_keys:
        obj_a = file_a[key]
        obj_b = file_b[key]

        type_name_a = type(obj_a).__name__
        type_name_b = type(obj_b).__name__

        print(f"\n  [{key}] type_a={type_name_a}, type_b={type_name_b}")

        if is_histogram(obj_a) and is_histogram(obj_b):
            match, details = compare_histogram(obj_a, obj_b, key)
            print(details)
            file_summary[key] = {"match": match, "details": details}

            if "cutflow" in key.lower() or "yield" in key.lower():
                cutflow_keys.append(key)
        else:
            print(f"  Skipping non-histogram object")
            file_summary[key] = {"match": None, "details": "non-histogram"}

    # Print cutflow details
    if cutflow_keys:
        print(f"\n{'~'*60}")
        print(f"CUTFLOW / YIELD HISTOGRAMS for {filename}")
        print(f"{'~'*60}")
        for key in cutflow_keys:
            print_cutflow(file_a[key], file_b[key], key)

    summary[filename] = file_summary
    file_a.close()
    file_b.close()


# ---- Main ----
print("ROOT File Comparison Script")
print(f"uproot version: {uproot.__version__}")
print(f"numpy version: {np.__version__}")

for fname in COMMON_FILES:
    compare_file(fname)

# ---- Final Summary ----
print(f"\n\n{'#'*80}")
print(f"FINAL SUMMARY")
print(f"{'#'*80}")

all_match = True
for fname, file_results in summary.items():
    print(f"\n  {fname}:")
    n_match = 0
    n_differ = 0
    n_skip = 0
    differ_keys = []
    for key, info in file_results.items():
        if info["match"] is None:
            n_skip += 1
        elif info["match"]:
            n_match += 1
        else:
            n_differ += 1
            differ_keys.append(key)
            all_match = False

    print(f"    Histograms matching:  {n_match}")
    print(f"    Histograms differing: {n_differ}")
    print(f"    Skipped:              {n_skip}")

    if differ_keys:
        print(f"    Differing histograms:")
        for k in differ_keys:
            print(f"      - {k}")

if all_match:
    print(f"\nOVERALL: All common histograms MATCH across both files.")
else:
    print(f"\nOVERALL: Some histograms DIFFER. See details above.")
