#!/usr/bin/env python3
"""WR signal NLO/LO k-factor from the fixed-order MG5_aMC runs of run_points.sh (AN-19-083 App. A method).

Reads out/<energy>_WR<m>_N<n>/{nlo.HwU, lo.json} and produces, per energy:

- kfactor_<energy>.json: per point sigma_NLO, sigma_Born (NLO run), sigma_LO (official LO run_card), k = NLO/LO
  with the 9-point scale envelope, and the combined k(m_lN) = sum NLO(m_lN) / sum Born(m_lN) over all points
  (the Run 2 "kfactor_all" histogram, Fig. 54), with its scale envelope;
- the Run 2 plots (HNWR_Plotter/python/Draw_KFactor.py): k vs m_lN (kfactor_vs_mlN.pdf) and the averaged
  k-factor vs m_WR for the inclusive points and for fixed m_N (kfactor.pdf), both with our curve and, at 13 TeV,
  the Run 2 one for comparison;
- optionally the analyzer payload (--payload), in the format of data/reweights/.../signal_kfactor_*.json.

HwU format: blocks "<histogram> nbins "title"" followed by one line per bin: xmin xmax central dy wgt_1 ... wgt_n
(the extra weights are the scale/PDF variations, named in the header line).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- HwU reading
def read_hwu(path: str | Path) -> dict[str, dict]:
    """{title: {"edges", "y", "dy", "cols": {column name: array}}}; the column names come from the ``##&`` header
    (xmin, xmax, central value, dy, delta_mu_cen/min/max @aux, then one column per scale / PDF weight)."""
    hists: dict[str, dict] = {}
    names: list[str] = []
    cur = None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("##&"):
            names = [c.strip() for c in s[3:].split("&")]
            continue
        if s.startswith("<histogram>"):
            m = re.match(r"<histogram>\s*(\d+)\s*\"(.*)\"", s)
            title = m.group(2).split("|")[0].strip()
            cur = hists[title] = {"edges": [], "rows": []}
            continue
        if cur is None or not s or s.startswith("<"):
            continue
        try:
            vals = [float(x) for x in s.split()]
        except ValueError:
            continue
        if not cur["edges"]:
            cur["edges"].append(vals[0])
        cur["edges"].append(vals[1])
        cur["rows"].append(vals)
    for h in hists.values():
        rows = np.asarray(h.pop("rows"))
        h["edges"] = np.asarray(h["edges"])
        h["y"], h["dy"] = rows[:, 2], rows[:, 3]
        h["cols"] = {names[i] if i < len(names) else f"col{i}": rows[:, i] for i in range(rows.shape[1])}
    return hists


def scale_columns(h: dict) -> list[str]:
    """The names of the scale-variation weight columns (muR/muF = 0.5, 1, 2 combinations) of an HwU histogram."""
    return [n for n in h["cols"] if n.startswith("dyn=") and "muR=" in n and "muF=" in n]


def hist_by_prefix(hists: dict, prefix: str) -> dict:
    for k, v in hists.items():
        if k.lower().startswith(prefix.lower()):
            return v
    raise KeyError(prefix)


# ----------------------------------------------------------------------------- per point
def point_result(out_dir: Path) -> dict | None:
    tag = out_dir.name                           # <energy>_WR<m>_N<n>
    m = re.match(r"(13p6|13)_WR(\d+)_N(\d+)$", tag)
    if not m or not (out_dir / "nlo.HwU").exists():
        return None
    energy, mwr, mn = m.group(1), int(m.group(2)), int(m.group(3))
    hw = read_hwu(out_dir / "nlo.HwU")
    tot = hist_by_prefix(hw, "total rate")
    born = hist_by_prefix(hw, "total rate Born")
    mln = hist_by_prefix(hw, "mlN")
    mln_b = hist_by_prefix(hw, "mlN Born")
    s_nlo, ds_nlo = float(tot["y"].sum()), float(np.sqrt((tot["dy"] ** 2).sum()))
    s_born = float(born["y"].sum())
    # scale envelope of the NLO total over the muR/muF variations (reweight_scale = True)
    sc = scale_columns(tot)
    scale_tot = np.array([tot["cols"][n].sum() for n in sc]) if sc else np.array([s_nlo])
    lo = json.loads((out_dir / "lo.json").read_text()) if (out_dir / "lo.json").exists() else {}
    s_lo = lo.get("xsec")
    res = {"energy": energy, "mwr": mwr, "mn": mn,
           "sigma_nlo": s_nlo, "sigma_nlo_err": ds_nlo, "sigma_born": s_born,
           "sigma_nlo_scale_min": float(scale_tot.min()), "sigma_nlo_scale_max": float(scale_tot.max()),
           "sigma_lo_official_card": s_lo, "sigma_lo_official_card_err": lo.get("xsec_err"),
           "k_nlo_over_born": s_nlo / s_born if s_born else None,
           "k_nlo_over_lo": (s_nlo / s_lo) if s_lo else None,
           "mln_edges": mln["edges"].tolist(), "mln_nlo": mln["y"].tolist(), "mln_born": mln_b["y"].tolist(),
           "mln_nlo_scale": np.column_stack([mln["cols"][n] for n in scale_columns(mln)]).tolist() if sc else None,
           "mln_lo_counts": lo.get("mln_counts"), "mln_lo_edges": lo.get("mln_edges")}
    return res


REBIN_GEV = 200.0     # the Run 2 k(m_lN) binning (kfactor_all rebinned by 200 x 1 GeV bins)


def _rebin(edges: np.ndarray, y: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Sum *y* (per-bin totals) into bins of *width* GeV (the input binning must divide it)."""
    w0 = edges[1] - edges[0]
    n = int(round(width / w0))
    assert abs(n * w0 - width) < 1e-6 and len(y) % n == 0, (w0, width, len(y))
    return edges[::n], y.reshape(-1, n).sum(axis=1)


def lo_card_hist(p: dict) -> np.ndarray | None:
    """The official-card LO cross section per m_lN bin: sigma_LO x (LHE events in the bin / all events)."""
    if not p.get("mln_lo_counts") or not p.get("sigma_lo_official_card"):
        return None
    c = np.asarray(p["mln_lo_counts"], dtype=float)
    assert np.allclose(p["mln_lo_edges"], p["mln_edges"])
    return p["sigma_lo_official_card"] * c / c.sum()


def combined_curve(points: list[dict]) -> dict:
    """k(m_lN) = sum_points NLO(m_lN) / sum_points LO(m_lN), every point entering with its own cross section (the
    Run 2 derivation from one mixed sample of (mWR, mN) points), in REBIN_GEV bins.  The denominator is the LO run
    with the official LO run_card (the normalisation our samples carry); the NLO run's own Born term gives the
    fixed-scale-choice cross-check k_born.  The scale envelope comes from the 9 muR/muF weights of the NLO numerator."""
    edges0 = np.asarray(points[0]["mln_edges"])
    nlo = np.zeros(len(edges0) - 1); born = np.zeros_like(nlo); lo = np.zeros_like(nlo)
    nsc = min(len(p["mln_nlo_scale"][0]) for p in points if p["mln_nlo_scale"]) if any(p["mln_nlo_scale"] for p in points) else 0
    nlo_sc = np.zeros((len(nlo), nsc)) if nsc else None
    n_lo = 0
    for p in points:
        assert np.allclose(p["mln_edges"], edges0)
        nlo += np.asarray(p["mln_nlo"]); born += np.asarray(p["mln_born"])
        h = lo_card_hist(p)
        if h is not None:
            lo += h; n_lo += 1
        if nsc:
            nlo_sc += np.asarray(p["mln_nlo_scale"])[:, :nsc]
    edges, nlo_r = _rebin(edges0, nlo, REBIN_GEV)
    _, born_r = _rebin(edges0, born, REBIN_GEV)
    _, lo_r = _rebin(edges0, lo, REBIN_GEV)
    den = lo_r if n_lo == len(points) else born_r
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(den > 0, nlo_r / den, np.nan)
        k_born = np.where(born_r > 0, nlo_r / born_r, np.nan)
    out = {"edges": edges.tolist(), "k": k.tolist(), "k_born": k_born.tolist(), "nlo": nlo_r.tolist(),
           "born": born_r.tolist(), "lo_card": lo_r.tolist(),
           "denominator": "LO (official run_card)" if n_lo == len(points) else "Born of the NLO run"}
    if nsc:
        sc_r = np.stack([_rebin(edges0, nlo_sc[:, j], REBIN_GEV)[1] for j in range(nsc)], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_sc = np.where(den[:, None] > 0, sc_r / den[:, None], np.nan)
        out["k_scale_min"] = np.where(den > 0, np.nanmin(np.where(np.isnan(k_sc), np.inf, k_sc), axis=1), np.nan).tolist()
        out["k_scale_max"] = np.where(den > 0, np.nanmax(np.where(np.isnan(k_sc), -np.inf, k_sc), axis=1), np.nan).tolist()
    return out


def average_k_from_curve(curve: dict, lo_edges, lo_counts) -> float | None:
    """<k> of a sample = sum_bins k(m_lN) x N_LO(m_lN) / N_LO: the Run 2 'Integrated' averaged k-factor."""
    if not lo_counts:
        return None
    k = np.asarray(curve["k"])
    e = np.asarray(curve["edges"])
    ctr = 0.5 * (np.asarray(lo_edges[:-1]) + np.asarray(lo_edges[1:]))
    idx = np.clip(np.searchsorted(e, ctr, side="right") - 1, 0, len(k) - 1)
    kk = k[idx]
    w = np.asarray(lo_counts, dtype=float)
    ok = ~np.isnan(kk) & (w > 0)
    return float((kk[ok] * w[ok]).sum() / w[ok].sum()) if w[ok].sum() else None


# ----------------------------------------------------------------------------- plots (Draw_KFactor.py)
def run2_curve(path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The Run 2 k(m_lN) histogram from our converted payload (values per bin of 'edges')."""
    try:
        raw = json.loads(Path(path).read_text())
    except OSError:
        return None
    return np.asarray(raw["kfactor"]["edges"]), np.asarray(raw["kfactor"]["values"])


def make_plots(energy: str, points: list[dict], curve: dict, out_dir: Path, run2_payload: str | None,
               run2_inclusive: str | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_e = "13 TeV" if energy == "13" else "13.6 TeV"
    # --- k vs m_lN (Fig. 54)
    fig, ax = plt.subplots(figsize=(7, 6))
    e = np.asarray(curve["edges"]); k = np.asarray(curve["k"])
    ax.step(e, np.append(k, k[-1]), where="post", color="k", lw=1.5, label=f"this work, {label_e}")
    if "k_scale_min" in curve:
        lo = np.asarray(curve["k_scale_min"]); hi = np.asarray(curve["k_scale_max"])
        ax.fill_between(e, np.append(lo, lo[-1]), np.append(hi, hi[-1]), step="post", color="k", alpha=0.15,
                        label="scale envelope (9-point)")
    kb = np.asarray(curve["k_born"])
    ax.step(e, np.append(kb, kb[-1]), where="post", color="tab:blue", lw=0.8, ls=":", label="NLO / Born of the NLO run (same scale)")
    r2 = run2_curve(run2_payload) if run2_payload else None
    if r2 is not None:
        e2, k2 = r2
        ax.step(e2, np.append(k2, k2[-1]), where="post", color="r", lw=1.2, ls="--", label="Run 2 (AN-19-083 App. A)")
    ax.set_xlim(0, 8000); ax.set_ylim(0.8, 1.8)
    ax.set_xlabel(r"$m_{\ell N}$ (GeV)"); ax.set_ylabel("k-factor")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / f"kfactor_vs_mlN_{energy}.pdf"); plt.close(fig)

    # --- averaged k-factor vs m_WR (Fig. 55): inclusive sigma_NLO/sigma_LO per point and the integrated
    # <k> from the curve, for m_N = 100, m_WR - 100 and fixed m_N values
    fig, ax = plt.subplots(figsize=(7, 6))
    # the three series of Run 2's Fig. 53/55: m_N = 100 GeV, m_N ~ m_WR / 2, m_N = m_WR - 100 GeV
    import itertools
    by_n: dict[str, list] = {}
    by_wr: dict[int, list] = {}
    for p in points:
        by_wr.setdefault(p["mwr"], []).append(p)
    for mwr, ps in sorted(by_wr.items()):
        for p in sorted(ps, key=lambda p: p["mn"]):
            if p["mn"] == 100:
                by_n.setdefault("m_N = 100 GeV", []).append(p)
            elif p["mn"] == p["mwr"] - 100:
                by_n.setdefault("m_N = m_WR - 100 GeV", []).append(p)
        mid = min(ps, key=lambda p: abs(p["mn"] - mwr / 2))          # the point nearest m_WR / 2 (full grids
        if 0.35 * mwr <= mid["mn"] <= 0.65 * mwr:                     # have many N: only this one is drawn)
            by_n.setdefault("m_N = m_WR / 2", []).append(mid)
    colors = {"m_N = 100 GeV": "k", "m_N = m_WR - 100 GeV": "r", "m_N = m_WR / 2": "b"}
    extra = itertools.cycle(["g", "m", "y", "c", "tab:orange", "tab:brown"])
    for key, ps in by_n.items():
        c = colors.get(key) or next(extra)
        x = [p["mwr"] for p in ps]
        # the inclusive k-factor of a point: sigma_NLO / sigma_LO(official card) x c (Run 2's "Inclusive")
        incl = [p.get("k_nlo_over_lo_sample") or p["k_nlo_over_lo"] or p["k_nlo_over_born"] for p in ps]
        integ = [average_k_from_curve(curve, p["mln_lo_edges"], p["mln_lo_counts"]) for p in ps]
        if len(x) > 1:
            ax.plot(x, incl, color=c, lw=1.5, label=f"Inclusive, {key}")
            ax.plot(x, integ, color=c, lw=1.5, ls=":", label=f"Integrated, {key}")
        else:
            ax.plot(x, incl, "o", color=c, label=f"Inclusive, {key}")
            ax.plot(x, integ, "x", color=c, label=f"Integrated, {key}")
    if run2_inclusive and Path(run2_inclusive).exists():
        rows = [l.split() for l in Path(run2_inclusive).read_text().splitlines() if l and not l.startswith("m(")]
        ax.plot([float(r[0]) for r in rows], [float(r[1]) for r in rows], color="k", lw=0.8, ls="-.", alpha=0.6,
                label="Run 2 inclusive, m_N = 100 GeV")
        ax.plot([float(r[0]) for r in rows], [float(r[3]) for r in rows], color="r", lw=0.8, ls="-.", alpha=0.6,
                label="Run 2 inclusive, m_N = m_WR - 100 GeV")
    ax.set_xlim(0, 8000); ax.set_ylim(0.8, 1.8)
    ax.set_xlabel(r"$m_{W_R}$ (GeV)"); ax.set_ylabel("Averaged k-factor")
    ax.set_title(label_e, loc="right", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / f"kfactor_{energy}.pdf"); plt.close(fig)


# ----------------------------------------------------------------------------- payload
def direct_k_table(points: list[dict], c: float, interpolate_to: list[tuple[int, int]] | None = None) -> dict:
    """{"WR<m>_N<n>": {"k", "k_scale_up", "k_scale_down", "sigma_nlo", "sigma_lo_card", "source"}}: the direct
    c x sigma_NLO / sigma_LO(official card) of every point run, and a linear interpolation in (m_WR, m_N / m_WR)
    for the requested points without a run (the private grids)."""
    out = {}
    for p in points:
        if not p["k_nlo_over_lo"]:
            continue
        kk = p["k_nlo_over_lo"] * c
        out[f"WR{p['mwr']}_N{p['mn']}"] = {
            "k": round(kk, 5), "k_scale_up": round(kk * p["sigma_nlo_scale_max"] / p["sigma_nlo"], 5),
            "k_scale_down": round(kk * p["sigma_nlo_scale_min"] / p["sigma_nlo"], 5),
            "sigma_nlo": p["sigma_nlo"], "sigma_nlo_err": p["sigma_nlo_err"], "sigma_lo_card": p["sigma_lo_official_card"],
            "source": "direct"}
    if interpolate_to:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        xy = np.array([[p["mwr"], p["mn"] / p["mwr"]] for p in points if p["k_nlo_over_lo"]], float)
        for key in ("k", "k_scale_up", "k_scale_down"):
            z = np.array([out[f"WR{p['mwr']}_N{p['mn']}"][key] for p in points if p["k_nlo_over_lo"]])
            lin, near = LinearNDInterpolator(xy, z), NearestNDInterpolator(xy, z)
            for mwr, mn in interpolate_to:
                name = f"WR{mwr}_N{mn}"
                if name in out and out[name]["source"] == "direct":
                    continue
                v = lin(mwr, mn / mwr)
                v = float(near(mwr, mn / mwr)) if np.isnan(v) else float(v)
                out.setdefault(name, {"source": "interpolated (linear in m_WR, m_N/m_WR from the direct points)"})[key] = round(v, 5)
    return out


def point_curves(points: list[dict], c: float, min_lo_events: int = 20) -> dict:
    """Per-point k(m_lN) = c x NLO(m_lN) / LO(m_lN) of that point alone, in REBIN_GEV bins (the shape correction of
    one sample without the neighbouring points' tails).  Bins with fewer than *min_lo_events* LO events (far tails,
    a few per mille of the events) take the point's direct k."""
    out = {}
    for p in points:
        h = lo_card_hist(p)
        if h is None or not p["k_nlo_over_lo"]:
            continue
        edges0 = np.asarray(p["mln_edges"])
        edges, nlo = _rebin(edges0, np.asarray(p["mln_nlo"]), REBIN_GEV)
        _, lo = _rebin(edges0, h, REBIN_GEV)
        _, cnt = _rebin(edges0, np.asarray(p["mln_lo_counts"], float), REBIN_GEV)
        kd = p["k_nlo_over_lo"] * c
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where((cnt >= min_lo_events) & (lo > 0), c * nlo / lo, kd)
        out[f"WR{p['mwr']}_N{p['mn']}"] = {"edges": edges.tolist(), "values": np.nan_to_num(k, nan=kd).tolist(),
                                            "direct_k": round(kd, 5), "mln_min": float(edges[int(np.argmax(cnt > 0))])}
    return out


def write_payload(curve: dict, energy: str, path: Path, points: list[dict],
                  interpolate_to: list[tuple[int, int]] | None = None) -> None:
    e = np.asarray(curve["edges"]); k = np.asarray(curve["k"])
    first = int(np.argmax(~np.isnan(k)))
    k = np.where(np.isnan(k), k[first], k)            # empty bins (below threshold / above the top) take the nearest
    payload = {
        "provenance": {"method": "AN-19-083 App. A: sigma_NLO(m_lN)/sigma_Born(m_lN) from MG5_aMC fixed-order runs, "
                                 "EffLRSM_NLO, 5FS, lhaid 325300, points summed with their own cross sections",
                       "energy": "13p6TeV" if energy == "13p6" else "13TeV",
                       "points": [f"WR{p['mwr']}_N{p['mn']}" for p in points],
                       "lo_norm_factor": curve.get("lo_norm_factor", 1.0),
                       "lo_norm_note": "k(m_lN) = c x NLO(m_lN) / LO(m_lN, official run_card, MG 2.9.18); c = mean of "
                                       "LO(2.9.18) / LO(sample, GenXSecAnalyzer) over the points shared with the official "
                                       "grid, so that sigma(sample) x <k> = sigma_NLO; c is the Breit-Wigner truncation of the samples' decay chain "
                       "(bwcutoff = 15 keeps 97.9 % of the resonance), not a MadGraph-version effect (2.6.5 and 2.9.18 agree "
                       "to 0.1 %): docs/studies/signal_kfactor.md",
                       "script": "scripts/signal_kfactor/run_points.sh + kfactor_analysis.py"},
        "variable": "gen_mln",
        "kfactor": {"edges": e.tolist(), "values": k.tolist(), "mln_min": float(e[first])},
        "averages": {},
        # per-point normalisation alternative to LO x <k>: sigma_NLO / sigma_LO(sample) directly (see docs/studies/signal_kfactor.md)
        "direct_k": direct_k_table(points, curve.get("lo_norm_factor", 1.0), interpolate_to),
        # the shape correction of each point from its own NLO/LO m_lN ratio (integrates to direct_k by construction)
        "point_curves": point_curves(points, curve.get("lo_norm_factor", 1.0)),
    }
    # points without a run (the private grids): the shape of the nearest run point in (m_WR, m_N / m_WR), scaled
    # to the interpolated direct k
    run_pts = [(p["mwr"], p["mn"]) for p in points if f"WR{p['mwr']}_N{p['mn']}" in payload["point_curves"]]
    for name, d in payload["direct_k"].items():
        if name in payload["point_curves"] or not run_pts:
            continue
        m = re.match(r"WR(\d+)_N(\d+)", name)
        mwr, mn = int(m.group(1)), int(m.group(2))
        near = min(run_pts, key=lambda q: ((q[0] - mwr) / 1000.0) ** 2 + (q[1] / q[0] - mn / mwr) ** 2)
        src = payload["point_curves"][f"WR{near[0]}_N{near[1]}"]
        scale = d["k"] / src["direct_k"]
        payload["point_curves"][name] = {"edges": src["edges"], "values": [v * scale for v in src["values"]],
                                         "direct_k": d["k"], "mln_min": src["mln_min"],
                                         "source": f"shape of WR{near[0]}_N{near[1]} scaled by {scale:.4f}"}
    path.write_text(json.dumps(payload, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the out/ directory of run_points.sh")
    ap.add_argument("--energy", choices=["13", "13p6"], required=True)
    ap.add_argument("--plots", default=None, help="directory for the pdf plots (default: <out>)")
    ap.add_argument("--run2-payload", default=str(Path(__file__).resolve().parent.parent.parent /
                                                 "data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_run2legacy_v4_2018.json"))
    ap.add_argument("--run2-inclusive", default=None, help="HNWR_Plotter script/PrintKFactor/result.txt")
    ap.add_argument("--payload", default=None, help="write the analyzer payload (k(m_lN) only, no averages) here")
    ap.add_argument("--sample-lo", default=None,
                    help="JSON {<energy>_WR<m>_N<n>: {xsec}} of the samples' own LO (GenXSecAnalyzer on the official "
                         "production): prints LO(card, this MG) / LO(sample) at the shared points")
    ap.add_argument("--lo-norm", default="none",
                    help="factor c applied to k(m_lN) and the payload so that LO(sample) x <k> = NLO: 'none', 'auto' "
                         "(the mean of LO(card) / LO(sample) over the shared --sample-lo points) or a number")
    ap.add_argument("--interpolate-to", default=None,
                    help="file with one WR<m>_N<n> (or MWR<m>_MN<n>) name per line: mass points to add to the payload's "
                         "direct_k table by interpolation (the private grids)")
    ap.add_argument("--gridpack-integrals", default=None,
                    help="JSON {<energy>_WR<m>_N<n>: {xsec, err}} of the official gridpacks' results.dat (gp_integrals.sh): "
                         "prints LO(card, this MG) / LO(gridpack) at the shared points")
    args = ap.parse_args()
    out = Path(args.out)
    points = [r for d in sorted(glob.glob(str(out / f"{args.energy}_WR*_N*"))) if (r := point_result(Path(d)))]
    if not points:
        raise SystemExit("no finished points")
    curve = combined_curve(points)
    # normalisation to the samples' own LO: the MG version that made the 13 TeV samples (2.6.5) gives a LO a few %
    # below the one used here (2.9.18, a different default-scale definition); c makes LO(sample) x <k> = sigma_NLO
    c, ratios = 1.0, []
    if args.sample_lo:
        smp = json.loads(Path(args.sample_lo).read_text())
        for p in points:
            g = smp.get(f"{args.energy}_WR{p['mwr']}_N{p['mn']}")
            if g and p["sigma_lo_official_card"]:
                ratios.append((p, p["sigma_lo_official_card"] / g["xsec"]))
    if args.lo_norm == "auto":
        if not ratios:
            raise SystemExit("--lo-norm auto needs --sample-lo with shared points")
        c = float(np.mean([r for _, r in ratios]))
    elif args.lo_norm != "none":
        c = float(args.lo_norm)
    if c != 1.0:
        for key in ("k", "k_born", "k_scale_min", "k_scale_max"):
            if key in curve:
                curve[key] = (np.asarray(curve[key], float) * c).tolist()
        for p in points:
            p["k_nlo_over_lo_sample"] = p["k_nlo_over_lo"] * c if p["k_nlo_over_lo"] else None
    curve["lo_norm_factor"] = c
    for p in points:
        p["k_integrated"] = average_k_from_curve(curve, p["mln_lo_edges"], p["mln_lo_counts"])
    summary = {"energy": args.energy, "curve": curve,
               "points": [{k: v for k, v in p.items() if not k.startswith("mln_")} for p in points]}
    (out / f"kfactor_{args.energy}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{'point':16s} {'sigma_NLO':>12s} {'sigma_Born':>12s} {'sigma_LO(card)':>14s} {'NLO/Born':>9s} {'NLO/LO':>8s} {'<k> integ':>10s}  scale env")
    for p in points:
        lo = p["sigma_lo_official_card"]
        print(f"WR{p['mwr']}_N{p['mn']:<9d} {p['sigma_nlo']:12.5g} {p['sigma_born']:12.5g} {lo if lo is None else f'{lo:14.5g}':>14} "
              f"{p['k_nlo_over_born']:9.4f} {p['k_nlo_over_lo'] if p['k_nlo_over_lo'] else float('nan'):8.4f} "
              f"{p['k_integrated'] if p['k_integrated'] else float('nan'):10.4f}  "
              f"[{p['sigma_nlo_scale_min'] / p['sigma_born']:.3f}, {p['sigma_nlo_scale_max'] / p['sigma_born']:.3f}]")
    plots = Path(args.plots) if args.plots else out
    plots.mkdir(parents=True, exist_ok=True)
    make_plots(args.energy, points, curve, plots, args.run2_payload if args.energy == "13" else None, args.run2_inclusive)
    if args.payload:
        interp = None
        if args.interpolate_to:
            interp = []
            for line in Path(args.interpolate_to).read_text().split():
                m = re.search(r"M?WR(\d+)_M?N(\d+)", line)
                if m:
                    interp.append((int(m.group(1)), int(m.group(2))))
        write_payload(curve, args.energy, Path(args.payload), points, interp)
    if ratios:
        print("\nLO with the official card in this MG version vs the samples' own LO (GenXSecAnalyzer):")
        for p, r in ratios:
            print(f"  WR{p['mwr']}_N{p['mn']:<6d} LO(card) {p['sigma_lo_official_card']:.5g}  sample {p['sigma_lo_official_card'] / r:.5g}  ratio {r:.4f}")
        print(f"  mean ratio {np.mean([r for _, r in ratios]):.4f} +- {np.std([r for _, r in ratios]):.4f} (rms over {len(ratios)} points)")
    print(f"LO normalisation factor c applied to k(m_lN) and <k>: {c:.4f}")
    if args.gridpack_integrals:
        gp = json.loads(Path(args.gridpack_integrals).read_text())
        print("\nLO with the official card in this MG version vs the official gridpack's own integral (results.dat):")
        for p in points:
            g = gp.get(f"{args.energy}_WR{p['mwr']}_N{p['mn']}")
            if g and p["sigma_lo_official_card"]:
                print(f"  WR{p['mwr']}_N{p['mn']:<6d} LO(card) {p['sigma_lo_official_card']:.5g}  gridpack {g['xsec']:.5g}  "
                      f"ratio {p['sigma_lo_official_card'] / g['xsec']:.4f}")
    print(f"k(m_lN) denominator: {curve['denominator']}; bins of {REBIN_GEV:.0f} GeV")
    print(f"wrote {out / f'kfactor_{args.energy}.json'} and plots in {plots}")


if __name__ == "__main__":
    main()
