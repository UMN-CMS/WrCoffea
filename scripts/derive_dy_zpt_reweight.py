#!/usr/bin/env python3
"""Derive the DY Z-pT (QCD NLO/LO) reweighting for a Run 3 era from generator-level spectra.

Follows the Run 2 recipe (SKFlat ``HNWR/Run2Legacy_v4``, ``Analyzers/src/GetZPtReweight.C``):
the generator-level Z of ``GenFinderForDY`` (``wrcoffea.dy_reweight.find_gen_z``), m(Z) >= 50 GeV,
every sample at 1 pb^-1 with the cross section the analysis uses for it, and

    ratio(pT_Z) = NLO(pT_Z) / LO(pT_Z)

in the Run 2 binning (28 bins, 0-8000 GeV).  The QCD scale (7-point envelope), PDF (Hessian) and
alpha_S variations of the numerator come from the NLO samples' LHE weights.  Only ee and mumu Z enter
(the Run 3 NLO inclusive samples are DYto2E / DYto2Mu); the ratio is flavour universal and is applied
to DY -> tautau as well, as in Run 2.

Samples (from the era's dataset configs, at their configured cross sections):

* LO = the madgraphMLM HT-binned DY of the ``dy_lo_ht`` config (what the analysis reweights) plus the
  LO inclusive sample below the first HT bin (``LHE_HT`` < 40 in Run 3), so the denominator is the full
  LO spectrum even though the analysis sample has no events there.
* NLO = the amcatnloFXFX inclusive sample below the first pT(ll) bin (``LHE_Vpt`` < 100) plus the
  pT(ll)-binned samples.  The Run 3 ``PTLL-X`` samples are open-ended (pT(ll) > X), so each one is
  restricted to [X, next X).

Each dataset's events are also kept per LHE slice, so the inclusive samples can be compared with the
binned ones where they overlap (the "stitch checks" in the payload provenance).

Two steps; ``fill`` caches one JSON per dataset in ``--workdir`` and ``combine`` reads the cache::

    python scripts/derive_dy_zpt_reweight.py fill    --era RunIII2024Summer24 --workdir logs/dy_zpt_spectra/RunIII2024Summer24
    python scripts/derive_dy_zpt_reweight.py combine --era RunIII2024Summer24 --workdir logs/dy_zpt_spectra/RunIII2024Summer24 \\
        --out data/reweights/Run3/2024/RunIII2024Summer24/dy_zpt_reweight_run3_<date>.json

``fill`` reads a spread of unskimmed NanoAOD files per dataset over XRootD (needs a grid proxy); the
skims cannot be used because their reco-level selection biases the generator spectrum.

The electroweak correction is not re-derived: the Run 2 (13 TeV) ``ZPtEWCorr`` block is copied from
``--ew-from`` and flagged in the provenance.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import math
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logger = logging.getLogger("derive_dy_zpt_reweight")

# Run 2 GetZPtReweight.C binning; the payload evaluates pT >= 8000 at 7999 (last bin).
PT_EDGES = [0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
            225, 250, 350, 500, 1000, 2000, 8000]
QCD_PT_MAX = 7999.0
MZ_MIN = 50.0

FLAVOURS = ("ee", "mumu", "tautau")
_FLAVOUR_PDG = {11: 0, 13: 1, 15: 2}

# LHEScaleWeight in NanoAOD: [i] = (muF, muR) as in the branch title; [4] is nominal.  The 7-point
# envelope drops the anti-correlated (muF, muR) = (2, 0.5) and (0.5, 2), i.e. indices 2 and 6.
N_SCALE = 9
SCALE_ENVELOPE = (0, 1, 3, 5, 7, 8)
# LHEPdfWeight for LHA 325300 (NNPDF31_nnlo_as_0118_mc_hessian_pdfas): 0 central, 1-100 Hessian,
# 101/102 alpha_S = 0.116 / 0.120.
N_PDF = 103
PDF_HESSIAN = range(1, 101)
PDF_ALPHAS_DOWN, PDF_ALPHAS_UP = 101, 102

GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_pdgId",
                "GenPart_status", "GenPart_statusFlags", "GenPart_genPartIdxMother"]

ROLES = ("lo_inc", "lo_ht", "nlo_inc", "nlo_ptll")
ROLE_CONFIG = {"lo_inc": "dy_lo_inc", "lo_ht": "dy_lo_ht", "nlo_inc": "dy_nlo_inc", "nlo_ptll": "dy_nlo_ptll"}
ROLE_SLICE_VAR = {"lo_inc": "LHE_HT", "lo_ht": "LHE_HT", "nlo_inc": "LHE_Vpt", "nlo_ptll": "LHE_Vpt"}
ROLE_IS_NLO = {"lo_inc": False, "lo_ht": False, "nlo_inc": True, "nlo_ptll": True}

REDIRECTORS = ["root://cmsxrootd.fnal.gov/", "root://cms-xrd-global.cern.ch/"]


# ---------------------------------------------------------------------------
# Dataset bookkeeping
# ---------------------------------------------------------------------------

def config_path(era: str, tag: str) -> Path:
    from wrcoffea.era_utils import get_era_details
    run, year, _ = get_era_details(era)
    return REPO / "data" / "configs" / run / year / era / f"{era}_mc_{tag}.json"


def _ht_low(name: str) -> float | None:
    m = re.search(r"Bin-HT-(\d+)", name)
    return float(m.group(1)) if m else None


def _ptll_low(name: str) -> float | None:
    m = re.search(r"PTLL-(\d+)", name)
    return float(m.group(1)) if m else None


def _is_dy(entry: dict) -> bool:
    return entry.get("physics_group") == "DYJets"


def select_datasets(era: str) -> dict[str, list[dict]]:
    """DY datasets per role with their cross sections, from the era's dataset configs.

    ``lo_inc`` and ``nlo_inc`` keep only their MLL-50 samples (the MLL-10to50 ones never reach
    m(Z) >= 50).
    """
    out: dict[str, list[dict]] = {}
    for role in ROLES:
        cfg = json.loads(config_path(era, ROLE_CONFIG[role]).read_text())
        rows = []
        for das, entry in cfg.items():
            if not _is_dy(entry):
                continue
            sample = das.split("/")[1]
            if role in ("lo_inc", "nlo_inc") and "MLL-50_" not in sample:
                continue
            rows.append({"das": das, "sample": sample, "xsec": float(entry["xsec"]), "role": role})
        if not rows:
            raise RuntimeError(f"{era}: no DY datasets for role '{role}' in {config_path(era, ROLE_CONFIG[role])}")
        out[role] = rows
    return out


def slice_edges(datasets: dict[str, list[dict]]) -> dict[str, list[float]]:
    """LHE slice edges per slice variable: the binned samples' lower edges, from 0 to infinity."""
    ht = sorted({_ht_low(d["sample"]) for d in datasets["lo_ht"]} - {None})
    ptll = sorted({_ptll_low(d["sample"]) for d in datasets["nlo_ptll"]} - {None})
    if not ht or not ptll:
        raise RuntimeError("could not parse the HT / PTLL bin edges from the sample names")
    return {"LHE_HT": [0.0, *ht, math.inf], "LHE_Vpt": [0.0, *ptll, math.inf]}


def kept_slice(d: dict, edges: list[float]) -> tuple[float, float]:
    """[lo, hi) of the slice variable this dataset contributes to the stitched spectrum."""
    if d["role"] in ("lo_inc", "nlo_inc"):
        return 0.0, edges[1]
    low = _ht_low(d["sample"]) if d["role"] == "lo_ht" else _ptll_low(d["sample"])
    i = edges.index(low)
    return low, edges[i + 1]


def pick_files(files: list, target_events: int, max_files: int) -> list:
    """Files spread evenly over the (sorted) dataset, enough for *target_events* (at most *max_files*)."""
    files = sorted(files, key=lambda f: f.lfn)
    if not files:
        return []
    mean = max(1.0, sum(f.events for f in files) / len(files))
    k = int(min(max_files, len(files), max(1, math.ceil(target_events / mean))))
    idx = np.unique(np.round(np.linspace(0, len(files) - 1, k)).astype(int))
    return [files[i] for i in idx]


# ---------------------------------------------------------------------------
# Filling (one task per file)
# ---------------------------------------------------------------------------

def _empty(n_slices: int, n_bins: int, with_lhe: bool) -> dict:
    h = {
        "sumw": np.zeros((len(FLAVOURS), n_slices, n_bins)),
        "sumw2": np.zeros((len(FLAVOURS), n_slices, n_bins)),
        "entries": np.zeros((len(FLAVOURS), n_slices, n_bins)),
    }
    if with_lhe:
        h["scale"] = np.zeros((n_slices, N_SCALE, n_bins))
        h["pdf"] = np.zeros((n_slices, N_PDF, n_bins))
    return h


def gen_z(arrays):
    """(pt, mass, found, flavour index) of the generator Z, using the analyzer's own finder."""
    import awkward as ak
    from coffea.nanoevents.methods import nanoaod

    from wrcoffea.dy_reweight import _FROM_HARD_PROCESS_BIT, find_gen_z

    gp = ak.zip({k[len("GenPart_"):]: arrays[k] for k in GEN_BRANCHES},
                with_name="GenParticle", behavior=nanoaod.behavior)
    pt, mass, found = find_gen_z(ak.zip({"GenPart": gp}, depth_limit=1))
    # Flavour of the pair: |pdgId| of the first candidate, with find_gen_z's candidate definition.
    pid = abs(gp.pdgId)
    from_hard = (gp.statusFlags & (1 << _FROM_HARD_PROCESS_BIT)) != 0
    cand = (((pid == 11) | (pid == 13)) & (gp.status == 1) & from_hard) | ((pid == 15) & (gp.status == 2) & from_hard)
    first = ak.fill_none(ak.firsts(pid[cand]), 0)
    flav = np.full(len(pt), -1, dtype=np.int64)
    for p, i in _FLAVOUR_PDG.items():
        flav[np.asarray(first) == p] = i
    return pt, mass, found, flav


def fill_file(lfn: str, slice_var: str, s_edges: list[float], with_lhe: bool,
              step: int = 100_000, timeout: int = 180) -> dict:
    """Generator Z-pT histograms of one file, per flavour and LHE slice (all events of the file)."""
    import uproot

    n_bins, n_slices = len(PT_EDGES) - 1, len(s_edges) - 1
    edges = np.asarray(PT_EDGES, dtype=np.float64)
    sedges = np.asarray(s_edges, dtype=np.float64)
    branches = GEN_BRANCHES + ["genWeight", slice_var] + (["LHEScaleWeight", "LHEPdfWeight"] if with_lhe else [])
    last_err = None
    for red in REDIRECTORS:
        h = _empty(n_slices, n_bins, with_lhe)
        stats = {"events": 0, "sumw_all": 0.0, "no_pair": 0, "mz_lt_50": 0, "bad_lhe_len": 0}
        try:
            with uproot.open(red + lfn, timeout=timeout) as f:
                runs_sumw = float(np.sum(f["Runs"]["genEventSumw"].array(library="np")))
                for arr in f["Events"].iterate(branches, step_size=step, library="ak"):
                    w = np.asarray(arr["genWeight"], dtype=np.float64)
                    stats["events"] += len(w)
                    stats["sumw_all"] += float(w.sum())
                    pt, mass, found, flav = gen_z(arr)
                    stats["no_pair"] += int(np.count_nonzero(~found))
                    stats["mz_lt_50"] += int(np.count_nonzero(found & (mass < MZ_MIN)))
                    sel = found & (mass >= MZ_MIN) & (flav >= 0)
                    ib = np.clip(np.searchsorted(edges, np.minimum(pt, QCD_PT_MAX), side="right") - 1, 0, n_bins - 1)
                    sv = np.asarray(arr[slice_var], dtype=np.float64)
                    isl = np.clip(np.searchsorted(sedges, sv, side="right") - 1, 0, n_slices - 1)
                    for name, val in (("sumw", w), ("sumw2", w * w), ("entries", np.ones_like(w))):
                        np.add.at(h[name], (flav[sel], isl[sel], ib[sel]), val[sel])
                    if with_lhe:
                        import awkward as ak
                        light = sel & (flav <= 1)
                        for key, n in (("scale", N_SCALE), ("pdf", N_PDF)):
                            branch = "LHEScaleWeight" if key == "scale" else "LHEPdfWeight"
                            lhe = arr[branch]
                            ok = np.asarray(ak.num(lhe) == n)
                            stats["bad_lhe_len"] += int(np.count_nonzero(light & ~ok))
                            use = light & ok
                            if not np.any(use):
                                continue
                            m = np.asarray(ak.to_numpy(lhe[use]), dtype=np.float64)       # (n_use, n)
                            wv = m * w[use][:, None]
                            np.add.at(h[key], (isl[use][:, None], np.arange(n)[None, :], ib[use][:, None]), wv)
            stats["runs_sumw"] = runs_sumw
            stats["redirector"] = red
            return {"lfn": lfn, "stats": stats, "hist": {k: v.tolist() for k, v in h.items()}}
        except Exception as e:  # noqa: BLE001 - try the next redirector, then report
            last_err = e
            logger.warning("%s via %s failed: %s", lfn, red, e, exc_info=logger.isEnabledFor(logging.DEBUG))
    raise RuntimeError(f"{lfn}: all redirectors failed: {last_err}")


def _cache_path(workdir: Path, d: dict) -> Path:
    return workdir / f"{d['sample']}.json"


def fill(era: str, workdir: Path, target_events: int, max_files: int, workers: int, force: bool,
         only: list[str] | None = None) -> None:
    from wrcoffea.skim.catalog import DBSClient

    datasets = select_datasets(era)
    sedges = slice_edges(datasets)
    workdir.mkdir(parents=True, exist_ok=True)
    dbs = DBSClient()
    jobs = []
    for role, rows in datasets.items():
        for d in rows:
            if only and not any(o in d["sample"] for o in only):
                continue
            out = _cache_path(workdir, d)
            if out.exists() and not force:
                logger.info("cached: %s", out.name)
                continue
            files = pick_files(dbs.files(d["das"]), target_events, max_files)
            jobs.append((d, files, out))
            logger.info("%-8s %-75s %d files, %d events", role, d["sample"], len(files), sum(f.events for f in files))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for d, files, out in jobs:
            var = ROLE_SLICE_VAR[d["role"]]
            for f in files:
                futs[pool.submit(fill_file, f.lfn, var, sedges[var], ROLE_IS_NLO[d["role"]])] = (d["sample"], f)
        results: dict[str, list] = {}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            sample, f = futs[fut]
            results.setdefault(sample, []).append((f, fut.result()))
            if i % 10 == 0 or i == len(futs):
                logger.info("%d/%d files done (%.0f s)", i, len(futs), time.time() - t0)
    for d, files, out in jobs:
        parts = results.get(d["sample"], [])
        var = ROLE_SLICE_VAR[d["role"]]
        agg = None
        stats = {}
        for _, r in parts:
            hh = {k: np.asarray(v) for k, v in r["hist"].items()}
            agg = hh if agg is None else {k: agg[k] + hh[k] for k in agg}
            for k, v in r["stats"].items():
                if isinstance(v, (int, float)):
                    stats[k] = stats.get(k, 0) + v
        payload = {
            "era": era, "role": d["role"], "das": d["das"], "sample": d["sample"], "xsec": d["xsec"],
            "slice_var": var, "slice_edges": [e if math.isfinite(e) else "inf" for e in sedges[var]],
            "kept_slice": [x if math.isfinite(x) else "inf" for x in kept_slice(d, sedges[var])],
            "pt_edges": PT_EDGES, "flavours": list(FLAVOURS), "mz_min": MZ_MIN,
            "files": [{"lfn": f.lfn, "events": f.events, "redirector": r["stats"]["redirector"]} for f, r in parts],
            "stats": stats, "filled_on": date.today().isoformat(),
            "hist": {k: v.tolist() for k, v in agg.items()},
        }
        out.write_text(json.dumps(payload))
        if stats.get("runs_sumw") and abs(stats["sumw_all"] / stats["runs_sumw"] - 1) > 1e-6:
            logger.warning("%s: summed genWeight %.6g != Runs genEventSumw %.6g", d["sample"],
                           stats["sumw_all"], stats["runs_sumw"])
        logger.info("wrote %s (%d events)", out.name, stats.get("events", 0))


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------

def _f(x):
    return math.inf if x == "inf" else float(x)


def load_cache(workdir: Path, era: str) -> list[dict]:
    rows = []
    for d in sum(select_datasets(era).values(), []):
        p = _cache_path(workdir, d)
        if not p.exists():
            raise FileNotFoundError(f"missing spectrum {p}; run 'fill' first")
        c = json.loads(p.read_text())
        if c["pt_edges"] != PT_EDGES:
            raise ValueError(f"{p.name}: pt binning differs from PT_EDGES")
        if abs(c["xsec"] - d["xsec"]) > 1e-9 * max(1.0, d["xsec"]):
            raise ValueError(f"{p.name}: cached xsec {c['xsec']} != config {d['xsec']}; re-run fill --force")
        c["hist"] = {k: np.asarray(v, dtype=np.float64) for k, v in c["hist"].items()}
        c["slice_edges"] = [_f(x) for x in c["slice_edges"]]
        c["kept_slice"] = [_f(x) for x in c["kept_slice"]]
        rows.append(c)
    return rows


def _norm(c: dict) -> float:
    """Per-pb normalization: xsec / sum of genWeight over every event read."""
    return c["xsec"] / c["stats"]["sumw_all"]


def _kept_mask(c: dict) -> np.ndarray:
    e = c["slice_edges"]
    lo, hi = c["kept_slice"]
    return np.array([(e[i] >= lo) and (e[i + 1] <= hi) for i in range(len(e) - 1)])


def spectrum(rows: list[dict], *, flavours=(0, 1), key="sumw", slices="kept") -> np.ndarray:
    """Stitched, per-pb spectrum summed over *rows* (nominal key arrays have a flavour axis)."""
    total = None
    for c in rows:
        h = c["hist"][key]
        mask = _kept_mask(c) if slices == "kept" else slices(c)
        n = _norm(c) ** (2 if key == "sumw2" else 1)
        if key in ("sumw", "sumw2", "entries"):
            part = h[list(flavours)][:, mask].sum(axis=(0, 1)) * (1 if key == "entries" else n)
        else:                                   # scale / pdf: (slice, variation, bin), ee + mumu only
            part = h[mask].sum(axis=0) * n
        total = part if total is None else total + part
    return total


def stitch_checks(rows: list[dict]) -> list[dict]:
    """Inclusive vs binned per-pb cross section (ee+mumu, m(Z) >= 50) in each overlapping slice.

    "Binned" is the sample(s) the stitch takes that slice from (both MLL bins of an HT bin; the one
    PTLL sample whose edge opens the slice), so overlapping open-ended samples are not added up.
    """
    out = []
    for inc_role, bin_role in (("lo_inc", "lo_ht"), ("nlo_inc", "nlo_ptll")):
        inc = [c for c in rows if c["role"] == inc_role]
        edges = inc[0]["slice_edges"]
        for i in range(1, len(edges) - 1):
            binned = [c for c in rows if c["role"] == bin_role and _kept_mask(c)[i]]
            def tot(cs, var="sumw"):
                v = 0.0
                for c in cs:
                    v += float(c["hist"][var][[0, 1], :, :][:, i].sum()) * _norm(c) ** (2 if var == "sumw2" else 1)
                return v
            x_inc, x_bin = tot(inc), tot(binned)
            e_inc, e_bin = math.sqrt(tot(inc, "sumw2")), math.sqrt(tot(binned, "sumw2"))
            r = x_inc / x_bin if x_bin > 0 else float("nan")
            err = r * math.sqrt((e_inc / x_inc) ** 2 + (e_bin / x_bin) ** 2) if x_inc > 0 and x_bin > 0 else float("nan")
            out.append({"slice_var": inc[0]["slice_var"], "slice": [edges[i], edges[i + 1] if math.isfinite(edges[i + 1]) else "inf"],
                        "inclusive_pb": x_inc, "binned_pb": x_bin, "inclusive_over_binned": r, "err": err})
    return out


def _fill_empty(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Bins without denominator take the value of the nearest filled bin."""
    out = values.copy()
    good = np.flatnonzero(ok)
    for i in np.flatnonzero(~ok):
        out[i] = values[good[np.argmin(np.abs(good - i))]]
    return out


def combine(era: str, workdir: Path, ew_from: Path) -> dict:
    rows = load_cache(workdir, era)
    lo = [c for c in rows if c["role"] in ("lo_inc", "lo_ht")]
    nlo = [c for c in rows if c["role"] in ("nlo_inc", "nlo_ptll")]
    lo_w, lo_w2 = spectrum(lo), spectrum(lo, key="sumw2")
    nlo_w, nlo_w2 = spectrum(nlo), spectrum(nlo, key="sumw2")
    scale = spectrum(nlo, key="scale")          # (9, bins)
    pdf = spectrum(nlo, key="pdf")              # (103, bins)

    ok = (lo_w > 0) & (nlo_w > 0)
    safe = np.where(ok, lo_w, 1.0)
    ratio = np.where(ok, nlo_w / safe, 1.0)
    stat = np.where(ok, ratio * np.sqrt(nlo_w2 / np.where(ok, nlo_w, 1.0) ** 2 + lo_w2 / safe ** 2), 0.0)
    scale_r = scale[list(SCALE_ENVELOPE)] / safe
    scale_up = np.where(ok, scale_r.max(axis=0), 1.0)
    scale_down = np.where(ok, scale_r.min(axis=0), 1.0)
    pdf_err = np.where(ok, np.sqrt(((pdf[list(PDF_HESSIAN)] - pdf[0]) ** 2).sum(axis=0)) / safe, 0.0)
    as_up = np.where(ok, pdf[PDF_ALPHAS_UP] / safe, 1.0)
    as_down = np.where(ok, pdf[PDF_ALPHAS_DOWN] / safe, 1.0)
    if not ok.all():
        logger.warning("%d bins without LO or NLO events take the nearest filled bin", int((~ok).sum()))
        ratio, scale_up, scale_down, as_up, as_down = (_fill_empty(a, ok) for a in (ratio, scale_up, scale_down, as_up, as_down))
        pdf_err, stat = _fill_empty(pdf_err, ok), _fill_empty(stat, ok)

    # per-flavour ratios as a check of flavour universality
    flav = {}
    for i, name in ((0, "ee"), (1, "mumu")):
        a, b = spectrum(nlo, flavours=(i,)), spectrum(lo, flavours=(i,))
        flav[name] = np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan).tolist()

    ew = json.loads(ew_from.read_text())["ew"]
    payload = {
        "provenance": {
            "era": era,
            "derived_on": date.today().isoformat(),
            "script": "scripts/derive_dy_zpt_reweight.py",
            "method": "Run 2 GetZPtReweight.C recipe: generator Z of GenFinderForDY (find_gen_z), m(Z) >= 50, "
                      "ee+mumu, per-pb spectra at the configured cross sections; ratio = NLO / LO",
            "lo": "dy_lo_ht HT-binned samples + dy_lo_inc MLL-50 below the first HT bin",
            "nlo": "dy_nlo_inc MLL-50 below the first PTLL bin + dy_nlo_ptll samples restricted to exclusive slices",
            "variations": "scale: 7-point envelope of LHEScaleWeight (indices 0,1,3,5,7,8) on NLO; pdf: Hessian "
                          "(LHEPdfWeight 1-100) on NLO; alphas: LHEPdfWeight 101/102 on NLO; all over nominal LO",
            "integrated": {"lo_pb": float(lo_w.sum()), "nlo_pb": float(nlo_w.sum()),
                           "nlo_over_lo": float(nlo_w.sum() / lo_w.sum())},
            "stitch_checks": stitch_checks(rows),
            "per_flavour_ratio": flav,
            "datasets": [{"sample": c["sample"], "role": c["role"], "xsec": c["xsec"], "kept_slice":
                          [x if math.isfinite(x) else "inf" for x in c["kept_slice"]],
                          "files": len(c["files"]), "events": c["stats"]["events"]} for c in rows],
            "ew_note": f"EW correction copied unchanged from {ew_from.name} (Run 2, 13 TeV ZPtEWCorr); not re-derived "
                       "for 13.6 TeV",
            "note": "first Run 3 derivation; the analyzer applies central x EW to DYJets whenever the era is mapped "
                    "in config.yaml dy_zpt_reweight",
        },
        "qcd": {
            "edges": [float(e) for e in PT_EDGES],
            "central": ratio.tolist(),
            "central_plain": ratio.tolist(),
            "stat_err": stat.tolist(),
            "pdf_err": pdf_err.tolist(),
            "scale_up": scale_up.tolist(),
            "scale_down": scale_down.tolist(),
            "alphas_up": as_up.tolist(),
            "alphas_down": as_down.tolist(),
            "pt_max": QCD_PT_MAX,
            "lo_pb_per_bin": lo_w.tolist(),
            "nlo_pb_per_bin": nlo_w.tolist(),
        },
        "ew": ew,
    }
    return payload


def plot(payload: dict, reference: Path | None, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = payload["qcd"]
    e = np.asarray(q["edges"])
    c = np.sqrt(np.maximum(e[:-1], 1.0) * e[1:])
    r = np.asarray(q["central"])
    fig, ax = plt.subplots(figsize=(9, 6))
    tot = np.sqrt(np.asarray(q["stat_err"]) ** 2 + np.asarray(q["pdf_err"]) ** 2)
    ax.fill_between(e, np.r_[np.asarray(q["scale_down"]), q["scale_down"][-1]],
                    np.r_[np.asarray(q["scale_up"]), q["scale_up"][-1]], step="post", color="#2a78d6", alpha=0.2,
                    label="scale envelope")
    ax.errorbar(c, r, yerr=tot, fmt="o", color="#2a78d6", ms=7, lw=2, label=f"{payload['provenance']['era']} NLO/LO (stat+PDF)")
    if reference is not None:
        ref = json.loads(reference.read_text())["qcd"]
        re_ = np.asarray(ref["edges"])
        rc = np.sqrt(np.maximum(re_[:-1], 1.0) * re_[1:])
        ax.errorbar(rc, ref["central"], yerr=ref["pdf_err"], fmt="s", color="#eb6834", ms=7, lw=2,
                    label="Run 2 2018 (HNWR)")
    ax.axhline(1, color="0.4", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, 8000)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"generator Z $p_T$ [GeV]")
    ax.set_ylabel("QCD NLO / LO")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fill", "combine"):
        s = sub.add_parser(name)
        s.add_argument("--era", required=True)
        s.add_argument("--workdir", type=Path, required=True, help="per-dataset spectrum cache")
        s.add_argument("-v", "--verbose", action="store_true")
    f = sub.choices["fill"]
    f.add_argument("--target-events", type=int, default=500_000, help="events per dataset (default 500k)")
    f.add_argument("--max-files", type=int, default=8)
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--force", action="store_true", help="refill datasets already cached")
    f.add_argument("--only", nargs="*", default=None, help="substrings of sample names to fill")
    c = sub.choices["combine"]
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--ew-from", type=Path,
                   default=REPO / "data/reweights/RunII/2018/RunIISummer20UL18/dy_zpt_reweight_run2legacy_v4_2018.json")
    c.add_argument("--plot", type=Path, default=None, help="PDF comparing the ratio with --ew-from's QCD ratio")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "fill":
        fill(args.era, args.workdir, args.target_events, args.max_files, args.workers, args.force, args.only)
        return 0
    payload = combine(args.era, args.workdir, args.ew_from)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    q, prov = payload["qcd"], payload["provenance"]
    print(f"integrated NLO/LO = {prov['integrated']['nlo_over_lo']:.4f}")
    for lo_e, hi_e, r, s, u, d in zip(q["edges"][:-1], q["edges"][1:], q["central"], q["stat_err"], q["scale_up"], q["scale_down"]):
        print(f"  [{lo_e:6.0f},{hi_e:6.0f})  {r:6.3f} +- {s:5.3f}  scale [{d:.3f}, {u:.3f}]")
    for chk in prov["stitch_checks"]:
        print(f"  stitch {chk['slice_var']} {chk['slice']}: inclusive/binned = {chk['inclusive_over_binned']:.3f} +- {chk['err']:.3f}")
    if args.plot:
        plot(payload, args.ew_from, args.plot)
        print(f"plot: {args.plot}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
