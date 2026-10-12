"""Synthetic NanoAOD-like files for the skim tests."""

from __future__ import annotations

import numpy as np
import awkward as ak
import uproot


def make_events(n: int, seed: int = 0, *, run: int = 1, lumi_start: int = 1, events_per_lumi: int = 50,
                is_data: bool = False, extra_hlt: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> dict:
    """Random events with the collections the skim selection needs plus a few extras."""
    rng = np.random.default_rng(seed)

    def jagged(counts, gen):
        tot = int(counts.sum())
        return ak.unflatten(gen(tot), counts)

    n_e = rng.poisson(0.9, n)
    n_mu = rng.poisson(0.9, n)
    n_jet = rng.poisson(2.5, n)
    n_fat = rng.poisson(0.4, n)
    # pT spectra straddling the thresholds, with a few exact-threshold values.
    def pt_gen(scale):
        def g(k):
            v = rng.exponential(scale, k).astype(np.float32) + np.float32(20.0)
            if k:
                pick = rng.random(k) < 0.03
                v[pick] = rng.choice(np.array([45.0, 52.0, 32.0, 180.0], dtype=np.float32), pick.sum())
            return v
        return g
    eta_gen = lambda k: rng.uniform(-3.0, 3.0, k).astype(np.float32)
    phi_gen = lambda k: rng.uniform(-np.pi, np.pi, k).astype(np.float32)
    ev = {
        "run": np.full(n, run, dtype=np.uint32),
        "luminosityBlock": (lumi_start + np.arange(n) // events_per_lumi).astype(np.uint32),
        "event": (np.arange(n, dtype=np.uint64) + np.uint64(seed * 10_000_000)),
        "Electron_pt": jagged(n_e, pt_gen(40)), "Electron_eta": jagged(n_e, eta_gen), "Electron_phi": jagged(n_e, phi_gen),
        "Electron_mass": jagged(n_e, lambda k: np.zeros(k, np.float32)),
        "Electron_charge": jagged(n_e, lambda k: rng.choice(np.array([-1, 1], np.int32), k)),
        "Electron_cutBased_HEEP": jagged(n_e, lambda k: rng.random(k) < 0.7),
        "Muon_pt": jagged(n_mu, pt_gen(40)), "Muon_eta": jagged(n_mu, eta_gen), "Muon_phi": jagged(n_mu, phi_gen),
        "Muon_mass": jagged(n_mu, lambda k: np.full(k, 0.1057, np.float32)),
        "Muon_charge": jagged(n_mu, lambda k: rng.choice(np.array([-1, 1], np.int32), k)),
        "Muon_tunepRelPt": jagged(n_mu, lambda k: rng.uniform(0.8, 1.5, k).astype(np.float32)),
        "Muon_highPtId": jagged(n_mu, lambda k: rng.integers(0, 3, k).astype(np.uint8)),
        "Jet_pt": jagged(n_jet, pt_gen(50)), "Jet_eta": jagged(n_jet, eta_gen), "Jet_phi": jagged(n_jet, phi_gen),
        "Jet_mass": jagged(n_jet, lambda k: rng.uniform(5, 30, k).astype(np.float32)),
        "Jet_jetId": jagged(n_jet, lambda k: rng.integers(0, 7, k).astype(np.uint8)),
        "Jet_genJetIdx": jagged(n_jet, lambda k: rng.integers(-1, 5, k).astype(np.int16)),
        "FatJet_pt": jagged(n_fat, pt_gen(150)), "FatJet_eta": jagged(n_fat, eta_gen), "FatJet_phi": jagged(n_fat, phi_gen),
        "FatJet_mass": jagged(n_fat, lambda k: rng.uniform(30, 200, k).astype(np.float32)),
        "FatJet_msoftdrop": jagged(n_fat, lambda k: rng.uniform(0, 150, k).astype(np.float32)),
        "Tau_pt": jagged(rng.poisson(0.3, n), pt_gen(30)),
        "HLT_Mu50": rng.random(n) < 0.3,
        "HLT_Ele30_WPTight_Gsf": rng.random(n) < 0.3,
        "L1_SingleMu22": rng.random(n) < 0.5,
        "Flag_goodVertices": rng.random(n) < 0.99,
        "fixedGridRhoFastjetAll": rng.uniform(10, 40, n).astype(np.float32),
        "PSWeight": ak.unflatten(rng.uniform(0.9, 1.1, 4 * n).astype(np.float32), np.full(n, 4)),
        "SomethingNew_x": rng.uniform(0, 1, n).astype(np.float32),
    }
    if not is_data:
        ev["genWeight"] = rng.choice(np.array([-1.0, 1.0, 1.0, 1.0], np.float32), n) * np.float32(1234.5)
    for h in extra_hlt:
        ev[h] = rng.random(n) < 0.1
    for d in drop:
        ev.pop(d, None)
    return ev


def _collections(ev: dict) -> dict:
    """Group ``Coll_field`` arrays into zipped records so uproot writes NanoAOD-style counters."""
    out: dict = {}
    colls: dict[str, dict] = {}
    for k, v in ev.items():
        if "_" in k and isinstance(v, ak.Array) and v.ndim == 2 and k.split("_", 1)[0] in ("Electron", "Muon", "Jet", "FatJet", "Tau"):
            c, f = k.split("_", 1)
            colls.setdefault(c, {})[f] = v
        else:
            out[k] = v
    for c, fields in colls.items():
        out[c] = ak.zip(fields)
    return out


def _types(data: dict) -> dict:
    """mktree branch types for a dict of arrays (TTree, never RNTuple)."""
    types = {}
    for k, v in data.items():
        if isinstance(v, ak.Array):
            types[k] = str(v.type.content)          # "var * {...}" or "var * float32"
        else:
            types[k] = np.asarray(v).dtype
    return types


def _write_tree(f, name: str, data: dict) -> None:
    f.mktree(name, _types(data))
    f[name].extend(data)


def write_nano_file(path, n: int = 2000, seed: int = 0, *, is_data: bool = False, run: int = 1,
                    lumi_start: int = 1, sumw: float | None = None, extra_hlt: tuple[str, ...] = (),
                    drop: tuple[str, ...] = (), n_runs_entries: int = 1, compression=None,
                    runs_drop: tuple[str, ...] = ()) -> dict:
    """Write a synthetic NanoAOD-like file (TTrees); returns the event dict.

    *runs_drop* leaves Runs branches out (``("PSSumw",)`` mimics UL NanoAODv9)."""
    ev = make_events(n, seed, run=run, lumi_start=lumi_start, is_data=is_data, extra_hlt=extra_hlt, drop=drop)
    kw = {"compression": compression} if compression is not None else {}
    with uproot.recreate(str(path), **kw) as f:
        _write_tree(f, "Events", _collections(ev))
        if is_data:
            _write_tree(f, "Runs", {"run": np.full(n_runs_entries, run, np.uint32)})
            lumis = np.unique(ev["luminosityBlock"])
            _write_tree(f, "LuminosityBlocks", {"run": np.full(len(lumis), run, np.uint32),
                                                 "luminosityBlock": lumis.astype(np.uint32)})
        else:
            w = ev["genWeight"].astype(np.float64)
            total = float(w.sum()) if sumw is None else float(sumw)
            per = np.full(n_runs_entries, total / n_runs_entries)
            runs = {
                "run": np.full(n_runs_entries, run, np.uint32),
                "genEventCount": np.full(n_runs_entries, n // n_runs_entries, np.int64),
                "genEventSumw": per,
                "genEventSumw2": np.full(n_runs_entries, float((w * w).sum()) / n_runs_entries),
                "LHEScaleSumw": ak.Array([np.linspace(0.9, 1.1, 9).astype(np.float32)] * n_runs_entries),
                "PSSumw": ak.Array([np.array([1.0, 1.01, 0.99, 1.0], np.float32)] * n_runs_entries),
            }
            for name in runs_drop:
                runs.pop(name, None)
            _write_tree(f, "Runs", runs)
    return ev
