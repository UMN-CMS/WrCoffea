#!/usr/bin/env python3
"""Derive the DY Z-pT (QCD NLO/LO) reweighting of an era the way AN-19-083 did it (Sec. 5.1).

Run 2 (SKFlat ``HNWR/Run2Legacy_v4`` ``Analyzers/src/GetZPtReweight.C`` and HNWR_Plotter
``python/Get_ZPtReweight.py``) histogrammed the generator-level Z of ``GenFinderForDY`` with m(Z) >= 50 GeV
in the inclusive LO madgraphMLM sample (SKFlat ``DYJets_MG``) and in the inclusive NLO amcatnloFXFX sample
(``DYJets``), each at 1 pb^-1 with its cross section (both 6077.22 pb), and took

    ratio(pT_Z) = NLO(pT_Z) / LO(pT_Z)

in 28 bins from 0 to 8000 GeV; the QCD scale, PDF and alpha_S variations of the NLO sample's LHE weights give
the variations of the ratio.  This script repeats that with our samples and cross sections:

* LO  = the inclusive madgraphMLM ``M-50`` / ``MLL-50`` sample(s) of the era's ``dy_lo_inc`` config;
* NLO = the inclusive amcatnloFXFX ``M-50`` / ``MLL-50`` sample(s) of the era's ``dy_nlo_inc`` config;

both at their configured cross sections (the same inclusive NNLO value, so the per-pb ratio equals the ratio of
the unit-normalized shapes that Get_ZPtReweight.py divided).  Only ee and mumu Z enter (the Run 3 inclusive
samples are DYto2E / DYto2Mu); the ratio is applied to DY -> tautau as well, as in Run 2.  The HT-binned LO and
the pT(Z)-binned NLO samples are not used and nothing is stitched, as in Run 2; the weight is applied to the
HT-binned LO set, which is the same LO generator setup as the inclusive sample.

Two steps; ``fill`` caches one JSON per dataset in ``--workdir`` and ``combine`` reads the cache::

    python scripts/derive_dy_zpt_reweight.py fill    --era RunIISummer20UL17 --workdir ~/work/skim_scratch/dy_zpt_spectra/RunIISummer20UL17
    python scripts/derive_dy_zpt_reweight.py combine --era RunIISummer20UL17 --workdir ~/work/skim_scratch/dy_zpt_spectra/RunIISummer20UL17 \\
        --ew-from data/reweights/RunII/2017/RunIISummer20UL17/dy_zpt_reweight_run2legacy_v4_2017.json \\
        --out data/reweights/RunII/2017/RunIISummer20UL17/dy_zpt_reweight_<date>.json

``combine --plot <pdf>`` also draws the ratio next to the ``--ew-from`` one; ``plot`` redraws a written payload
without recomputing it::

    python scripts/derive_dy_zpt_reweight.py plot --payload <combine --out> --ew-from <reference> --plot <pdf>

``fill`` reads every file of both samples from unskimmed NanoAOD over XRootD (needs a grid proxy; about an hour
per UL era), as Run 2 ran over the full samples; the skims cannot be used because their reco-level selection
biases the generator spectrum.  ``--target-events N`` reads an even spread of about N events per dataset
instead, for a quick look (the bins above 1 TeV are then statistically empty).

The electroweak correction is not re-derived: the Run 2 (13 TeV) ``ZPtEWCorr`` block of the year is copied
from ``--ew-from`` and flagged in the provenance.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import math
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
# LHEPdfWeight of an NNPDF3.1 NNLO Hessian pdfas set (LHA 306000 in UL, 325300 in Run 3): 0 central,
# 1-100 Hessian, 101/102 alpha_S = 0.116 / 0.120.
N_PDF = 103
PDF_HESSIAN = range(1, 101)
PDF_ALPHAS_DOWN, PDF_ALPHAS_UP = 101, 102

GEN_BRANCHES = ["GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass", "GenPart_pdgId",
                "GenPart_status", "GenPart_statusFlags", "GenPart_genPartIdxMother"]

ROLES = ("lo", "nlo")
ROLE_CONFIG = {"lo": "dy_lo_inc", "nlo": "dy_nlo_inc"}
ROLE_IS_NLO = {"lo": False, "nlo": True}

REDIRECTORS = ["root://cmsxrootd.fnal.gov/", "root://cms-xrd-global.cern.ch/"]


# ---------------------------------------------------------------------------
# Dataset bookkeeping
# ---------------------------------------------------------------------------

def config_path(era: str, tag: str) -> Path:
    from wrcoffea.era_utils import get_era_details
    run, year, _ = get_era_details(era)
    return REPO / "data" / "configs" / run / year / era / f"{era}_mc_{tag}.json"


def _is_low_mass(sample: str) -> bool:
    """The m(ll) 10-50 companions of the inclusive samples, which never reach m(Z) >= 50."""
    return "MLL-10to50" in sample or "M-10to50" in sample


def _is_dy(entry: dict) -> bool:
    return entry.get("physics_group") == "DYJets"


def select_datasets(era: str) -> dict[str, list[dict]]:
    """The inclusive DY sample(s) per role with their cross sections, from the era's dataset configs.

    Only the ``M-50`` / ``MLL-50`` samples (the ``M-10to50`` companions never reach m(Z) >= 50); a second
    DAS path of the same primary dataset (UL ``_ext1``) is the same physics and is counted once.
    """
    out: dict[str, list[dict]] = {}
    for role in ROLES:
        cfg = json.loads(config_path(era, ROLE_CONFIG[role]).read_text())
        rows = []
        for das, entry in cfg.items():
            if not _is_dy(entry):
                continue
            sample = das.split("/")[1]
            if _is_low_mass(sample) or any(r["sample"] == sample for r in rows):
                continue
            rows.append({"das": das, "sample": sample, "xsec": float(entry["xsec"]), "role": role})
        if not rows:
            raise RuntimeError(f"{era}: no inclusive DY dataset for role '{role}' in {config_path(era, ROLE_CONFIG[role])}")
        out[role] = rows
    return out


def pick_files(files: list, target_events: int = 0, max_files: int = 0) -> list:
    """Every file of the (sorted) dataset, or, with *target_events* > 0, the smallest even spread whose files hold
    that many events (at most *max_files* when > 0).  Counted with the picked files' own event numbers, not the
    dataset mean: UL NanoAOD files range from a few thousand to a few million events, and a spread sized by the
    mean can land on tiny ones (UL18 inclusive NLO: one file of 59k events for a 500k target)."""
    files = sorted(files, key=lambda f: f.lfn)
    if not files:
        return []
    if target_events <= 0:
        return files
    if max_files <= 0:
        max_files = len(files)
    picked: list = []
    for k in range(1, min(max_files, len(files)) + 1):
        idx = np.unique(np.round(np.linspace(0, len(files) - 1, k)).astype(int))
        picked = [files[i] for i in idx]
        if sum(f.events for f in picked) >= target_events:
            break
    return picked


# ---------------------------------------------------------------------------
# Filling (one task per file)
# ---------------------------------------------------------------------------

def _empty(n_bins: int, with_lhe: bool) -> dict:
    h = {
        "sumw": np.zeros((len(FLAVOURS), n_bins)),
        "sumw2": np.zeros((len(FLAVOURS), n_bins)),
        "entries": np.zeros((len(FLAVOURS), n_bins)),
    }
    if with_lhe:
        h["scale"] = np.zeros((N_SCALE, n_bins))
        h["pdf"] = np.zeros((N_PDF, n_bins))
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


def fill_file(lfn: str, with_lhe: bool, step: int = 100_000, timeout: int = 180) -> dict:
    """Generator Z-pT histograms of one file, per flavour (all events of the file)."""
    import uproot

    n_bins = len(PT_EDGES) - 1
    edges = np.asarray(PT_EDGES, dtype=np.float64)
    branches = GEN_BRANCHES + ["genWeight"] + (["LHEScaleWeight", "LHEPdfWeight"] if with_lhe else [])
    last_err = None
    for red in REDIRECTORS:
        h = _empty(n_bins, with_lhe)
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
                    for name, val in (("sumw", w), ("sumw2", w * w), ("entries", np.ones_like(w))):
                        np.add.at(h[name], (flav[sel], ib[sel]), val[sel])
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
                            np.add.at(h[key], (np.arange(n)[None, :], ib[use][:, None]), wv)
            stats["runs_sumw"] = runs_sumw
            stats["redirector"] = red
            return {"lfn": lfn, "stats": stats, "hist": {k: v.tolist() for k, v in h.items()}}
        except Exception as e:  # noqa: BLE001 - try the next redirector, then report
            last_err = e
            logger.warning("%s via %s failed: %s", lfn, red, e, exc_info=logger.isEnabledFor(logging.DEBUG))
    raise RuntimeError(f"{lfn}: every redirector failed ({last_err})")


def _cache_path(workdir: Path, d: dict) -> Path:
    return workdir / f"{d['sample']}.json"


def fill(era: str, workdir: Path, target_events: int, max_files: int, workers: int, force: bool,
         only: list[str] | None = None) -> None:
    from wrcoffea.skim.catalog import DBSClient

    datasets = select_datasets(era)
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
            if not files:
                raise RuntimeError(f"{d['das']}: DBS lists no files")
            jobs.append((d, files, out))
            logger.info("%-4s %-75s %d files, %d events", role, d["sample"], len(files), sum(f.events for f in files))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for d, files, out in jobs:
            for f in files:
                futs[pool.submit(fill_file, f.lfn, ROLE_IS_NLO[d["role"]])] = (d["sample"], f)
        results: dict[str, list] = {}
        failed: dict[str, list] = {}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            sample, f = futs[fut]
            try:
                results.setdefault(sample, []).append((f, fut.result()))
            except Exception as e:  # noqa: BLE001 - one unreadable file must not lose the whole fill
                failed.setdefault(sample, []).append(f.lfn)
                logger.warning("skipping %s: %s", f.lfn, e)
            if i % 10 == 0 or i == len(futs):
                logger.info("%d/%d files done (%.0f s)", i, len(futs), time.time() - t0)
    for d, files, out in jobs:
        parts = results.get(d["sample"], [])
        agg = None
        stats: dict = {}
        for _, r in parts:
            hh = {k: np.asarray(v) for k, v in r["hist"].items()}
            agg = hh if agg is None else {k: agg[k] + hh[k] for k in agg}
            for k, v in r["stats"].items():
                if isinstance(v, (int, float)):
                    stats[k] = stats.get(k, 0) + v
        payload = {
            "era": era, "role": d["role"], "das": d["das"], "sample": d["sample"], "xsec": d["xsec"],
            "pt_edges": PT_EDGES, "flavours": list(FLAVOURS), "mz_min": MZ_MIN,
            "files": [{"lfn": f.lfn, "events": f.events, "redirector": r["stats"]["redirector"]} for f, r in parts],
            "failed_files": failed.get(d["sample"], []),
            "stats": stats, "filled_on": date.today().isoformat(),
            "hist": {k: v.tolist() for k, v in agg.items()},
        }
        if agg is None:
            raise RuntimeError(f"{d['sample']}: no file could be read")
        out.write_text(json.dumps(payload))
        if failed.get(d["sample"]):
            logger.warning("%s: %d of %d files skipped (listed in the cache as failed_files)", d["sample"],
                           len(failed[d["sample"]]), len(files))
        if stats.get("runs_sumw") and abs(stats["sumw_all"] / stats["runs_sumw"] - 1) > 1e-6:
            logger.warning("%s: summed genWeight %.6g != Runs genEventSumw %.6g", d["sample"],
                           stats["sumw_all"], stats["runs_sumw"])
        if stats.get("bad_lhe_len"):
            logger.warning("%s: %d events with an unexpected LHE weight count were left out of the variations",
                           d["sample"], stats["bad_lhe_len"])
        logger.info("wrote %s (%d events)", out.name, stats.get("events", 0))


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------

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
        if c["hist"]["sumw"].ndim != 2:
            raise ValueError(f"{p.name}: cache in the old per-slice layout; re-run fill --force")
        c["role"] = d["role"]
        rows.append(c)
    return rows


def _norm(c: dict) -> float:
    """Per-pb normalization: xsec / sum of genWeight over every event read."""
    return c["xsec"] / c["stats"]["sumw_all"]


def spectrum(rows: list[dict], *, flavours=(0, 1), key="sumw") -> np.ndarray:
    """Per-pb Z-pT spectrum summed over *rows* (ee + mumu by default); ``scale`` / ``pdf`` give (variation, bin)."""
    total = None
    for c in rows:
        h = c["hist"][key]
        n = _norm(c) ** (2 if key == "sumw2" else 1)
        if key in ("sumw", "sumw2", "entries"):
            part = h[list(flavours)].sum(axis=0) * (1 if key == "entries" else n)
        else:                                   # scale / pdf: (variation, bin), ee + mumu only
            part = h * n
        total = part if total is None else total + part
    return total


def _fill_empty(values: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Bins without denominator take the value of the nearest filled bin."""
    out = values.copy()
    good = np.flatnonzero(ok)
    for i in np.flatnonzero(~ok):
        out[i] = values[good[np.argmin(np.abs(good - i))]]
    return out


def qcd_block(lo: list[dict], nlo: list[dict]) -> tuple[dict, dict]:
    """The payload's ``qcd`` block (ratio and variations) and its checks, from the cached LO and NLO rows.

    Everything is per pb at the configured cross sections: the central ratio is NLO / LO, and each variation
    is the varied NLO spectrum over the nominal LO, not renormalized (as the Run 2 ratio files).
    """
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

    qcd = {
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
    }
    checks = {
        "integrated": {"lo_pb": float(lo_w.sum()), "nlo_pb": float(nlo_w.sum()),
                       "nlo_over_lo": float(nlo_w.sum() / lo_w.sum())},
        "per_flavour_ratio": flav,
        "entries": {"lo": spectrum(lo, key="entries").tolist(), "nlo": spectrum(nlo, key="entries").tolist()},
    }
    return qcd, checks


def merge_eras(rows_by_era: dict[str, list[dict]]) -> list[dict]:
    """Cached rows of several eras merged by (role, sample): the same primary dataset in two campaigns is more
    files of one sample (histograms and sums of weights added, one per-pb normalization), so every era counts with
    its events; samples with different names (2024's DYto2E / DYto2Mu) stay separate rows and add per pb."""
    merged: dict[tuple, dict] = {}
    for era, rows in rows_by_era.items():
        for c in rows:
            key = (c["role"], c["sample"])
            if key not in merged:
                m = dict(c)
                m["hist"] = {k: v.copy() for k, v in c["hist"].items()}
                m["stats"] = dict(c["stats"])
                m["files"] = list(c["files"])
                m["eras"] = [era]
                merged[key] = m
                continue
            m = merged[key]
            if abs(m["xsec"] - c["xsec"]) > 1e-9 * max(1.0, c["xsec"]):
                raise ValueError(f"{c['sample']}: cross section differs between eras ({m['xsec']} vs {c['xsec']} in {era})")
            for k in m["hist"]:
                m["hist"][k] = m["hist"][k] + c["hist"][k]
            for k, v in c["stats"].items():
                if isinstance(v, (int, float)):
                    m["stats"][k] = m["stats"].get(k, 0) + v
            m["files"] = m["files"] + list(c["files"])
            m["eras"].append(era)
    return list(merged.values())


def combine(eras: list[str], workdirs: list[Path], ew_from: Path, label: str | None = None) -> dict:
    """The payload of one era, or of several eras merged (``label`` names the combination, e.g. ``Run3``)."""
    if len(eras) != len(workdirs):
        raise ValueError("one --workdir per --era")
    rows = merge_eras({era: load_cache(w, era) for era, w in zip(eras, workdirs)})
    era = label or (eras[0] if len(eras) == 1 else "+".join(eras))
    lo = [c for c in rows if c["role"] == "lo"]
    nlo = [c for c in rows if c["role"] == "nlo"]
    xs_lo, xs_nlo = sum(c["xsec"] for c in lo), sum(c["xsec"] for c in nlo)
    if abs(xs_nlo / xs_lo - 1) > 1e-3:
        logger.warning("LO and NLO cross sections differ (%.2f vs %.2f pb): the per-pb ratio carries their "
                       "ratio %.4f on top of the shape", xs_lo, xs_nlo, xs_nlo / xs_lo)
    qcd, checks = qcd_block(lo, nlo)
    ew = json.loads(ew_from.read_text())["ew"]
    payload = {
        "provenance": {
            "era": era,
            "eras": list(eras),
            "derived_on": date.today().isoformat(),
            "script": "scripts/derive_dy_zpt_reweight.py",
            "method": "AN-19-083 Sec. 5.1 (SKFlat GetZPtReweight.C, HNWR_Plotter Get_ZPtReweight.py): generator Z "
                      "of GenFinderForDY (find_gen_z), m(Z) >= 50, ee+mumu, the inclusive LO madgraphMLM and the "
                      "inclusive NLO amcatnloFXFX samples at 1 pb^-1 with their configured cross sections; "
                      "ratio = NLO / LO per bin, no binned samples, no stitching",
            "lo": "inclusive madgraphMLM M-50 / MLL-50 sample(s) of the dy_lo_inc config",
            "nlo": "inclusive amcatnloFXFX M-50 / MLL-50 sample(s) of the dy_nlo_inc config",
            "variations": "scale: 7-point envelope of LHEScaleWeight (indices 0,1,3,5,7,8) on NLO; pdf: Hessian "
                          "(LHEPdfWeight 1-100) on NLO; alphas: LHEPdfWeight 101/102 on NLO; all over nominal LO, "
                          "not renormalized",
            "integrated": checks["integrated"],
            "per_flavour_ratio": checks["per_flavour_ratio"],
            "entries_per_bin": checks["entries"],
            "datasets": [{"sample": c["sample"], "role": c["role"], "xsec": c["xsec"], "eras": c.get("eras", [era]),
                          "files": len(c["files"]), "events": c["stats"]["events"]} for c in rows],
            "ew_note": f"EW correction copied unchanged from {ew_from.name} (Run 2 SKFlat ZPtEWCorr, 13 TeV); not "
                       "re-derived",
            "note": f"own derivation for {era}; the analyzer applies central x EW to DYJets whenever the era is "
                    "mapped in config.yaml dy_zpt_reweight",
        },
        "qcd": qcd,
        "ew": ew,
    }
    return payload


def plot(payload: dict, reference: Path | None, out: Path) -> None:
    """The QCD NLO/LO ratio against generator Z pT, with the reference (Run 2) ratio, in the CMS style of
    WR_Plotter (cms_label, fit_header, 10 x 10 in, 20 pt legend, Petroff colours with distinct markers)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    sys.path.insert(0, str(REPO / "WR_Plotter"))
    from wrplotter.config import era_label
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header

    blue, orange = "#5790fc", "#f89c20"          # Petroff 6
    hep.style.use("CMS")
    q = payload["qcd"]
    e = np.asarray(q["edges"])
    c = np.sqrt(np.maximum(e[:-1], 1.0) * e[1:])
    r = np.asarray(q["central"])
    fig, ax = plt.subplots(figsize=(10, 10))
    tot = np.sqrt(np.asarray(q["stat_err"]) ** 2 + np.asarray(q["pdf_err"]) ** 2)
    ax.fill_between(e, np.r_[np.asarray(q["scale_down"]), q["scale_down"][-1]],
                    np.r_[np.asarray(q["scale_up"]), q["scale_up"][-1]], step="post", facecolor=blue, alpha=0.25,
                    lw=0, label=r"Scale variations ($\mu_R$, $\mu_F$)")
    ax.errorbar(c, r, yerr=tot, fmt="o", color=blue, markersize=8, elinewidth=1.5,
                label="NLO / LO (stat. + PDF unc.)")
    if reference is not None:
        ref = json.loads(reference.read_text())["qcd"]
        re_ = np.asarray(ref["edges"])
        rc = np.sqrt(np.maximum(re_[:-1], 1.0) * re_[1:])
        ax.errorbar(rc, ref["central"], yerr=ref["pdf_err"], fmt="s", color=orange, markersize=8, elinewidth=1.5,
                    mfc="white", label="Run 2 (AN-19-083)")
    ax.axhline(1, color="0.4", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, 8000)
    ax.set_ylim(0, 2)
    ax.set_xlabel(r"Generator $p_{T}^{Z}$ [GeV]")
    ax.set_ylabel("QCD NLO / LO")
    era = str(payload["provenance"]["era"])
    run2 = era.startswith("RunIISummer")          # the UL eras; RunIII... and Run3... are 13.6 TeV
    cms_label(ax, None, data=False, rlabel="(13 TeV)" if run2 else "(13.6 TeV)",
              text=f"Drell-Yan, {era_label(era)}\nInclusive NLO over inclusive LO")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fill", "combine", "plot"):
        s = sub.add_parser(name)
        s.add_argument("-v", "--verbose", action="store_true")
    f = sub.choices["fill"]
    f.add_argument("--era", required=True)
    f.add_argument("--workdir", type=Path, required=True, help="per-dataset spectrum cache")
    f.add_argument("--target-events", type=int, default=0,
                   help="read an even spread of about this many events per dataset; 0 (default) reads every file")
    f.add_argument("--max-files", type=int, default=0, help="cap on the files of a spread; 0 (default) = no cap")
    f.add_argument("--workers", type=int, default=6)
    f.add_argument("--force", action="store_true", help="refill datasets already cached")
    f.add_argument("--only", nargs="*", default=None, help="substrings of sample names to fill")
    c = sub.choices["combine"]
    c.add_argument("--era", nargs="+", required=True, help="one era, or several to merge (e.g. the five Run 3 eras)")
    c.add_argument("--workdir", type=Path, nargs="+", required=True, help="one spectrum cache per --era, same order")
    c.add_argument("--label", default=None, help="name of the combination in the payload (default: the era)")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--ew-from", type=Path, required=True,
                   help="converted Run 2 payload of the year whose EW block is copied")
    c.add_argument("--plot", type=Path, default=None, help="PDF comparing the ratio with --ew-from's QCD ratio")
    pl = sub.choices["plot"]
    pl.add_argument("--payload", type=Path, required=True, help="a payload written by combine (--out)")
    pl.add_argument("--ew-from", type=Path, default=None, help="the reference payload drawn next to it")
    pl.add_argument("--plot", type=Path, required=True, help="output PDF")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "fill":
        fill(args.era, args.workdir, args.target_events, args.max_files, args.workers, args.force, args.only)
        return 0
    if args.cmd == "plot":                    # redraw a derived payload; nothing is recomputed
        plot(json.loads(args.payload.read_text()), args.ew_from, args.plot)
        print(f"plot: {args.plot}")
        return 0
    payload = combine(args.era, args.workdir, args.ew_from, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1))
    q, prov = payload["qcd"], payload["provenance"]
    print(f"integrated NLO/LO = {prov['integrated']['nlo_over_lo']:.4f} "
          f"(LO {prov['integrated']['lo_pb']:.1f} pb, NLO {prov['integrated']['nlo_pb']:.1f} pb)")
    ent = prov["entries_per_bin"]
    for i, (lo_e, hi_e) in enumerate(zip(q["edges"][:-1], q["edges"][1:])):
        print(f"  [{lo_e:6.0f},{hi_e:6.0f})  {q['central'][i]:6.3f} +- {q['stat_err'][i]:5.3f}  "
              f"scale [{q['scale_down'][i]:.3f}, {q['scale_up'][i]:.3f}]  events LO {ent['lo'][i]:.0f} NLO {ent['nlo'][i]:.0f}")
    if args.plot:
        plot(payload, args.ew_from, args.plot)
        print(f"plot: {args.plot}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
