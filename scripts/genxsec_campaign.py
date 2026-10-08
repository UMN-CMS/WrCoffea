#!/usr/bin/env python3
"""Compute generator cross sections for every MC sample with GenXSecAnalyzer, and
inspect what the samples actually contain, so the config values can be set
without trusting dataset names.

For every unique (primary dataset, centre-of-mass energy) in the configs:

1. resolve the MiniAOD parent through DBS and download one file (two for
   aMC@NLO samples) to ``<work>/inputs`` with checksum verification;
2. run GenXSecAnalyzer on it inside the CMSSW container of the OrbStack VM
   (the container cannot read xrootd, hence the local copy) and parse the
   "after filter" cross section, its uncertainty, the matching and filter
   efficiencies and the negative-weight fraction;
3. read 20k events of the NanoAOD dataset over xrootd and record what the
   generator produced: LHE lepton flavours, LHE neutrinos/quarks/tops/bosons,
   and the W decay modes found in GenPart (how many leptonic W decays per
   event), so a "2L2Nu" or "DYto2E" label is checked against the events;
4. write ``genxsec_results.json`` and a review table ``genxsec_review.md``
   with the current config value, the generator value, the observed content,
   and a proposed factor (1/3 for single-flavour samples whose header carries
   the three-flavour cross section, a branching fraction for samples whose
   decays are restricted while the header is inclusive). Nothing is written
   to the configs: ``--apply`` does that after the table has been reviewed.

    python3 scripts/genxsec_campaign.py run --era RunIII2024Summer24 --work ~/work/skims_v2/_genxsec
    python3 scripts/genxsec_campaign.py review --work ~/work/skims_v2/_genxsec
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from wrcoffea.skim.catalog import DBSClient, is_das_path
from wrcoffea.skim.fetch import fetch_file
from wrcoffea.skim.plan import load_config_entries, ERA_SUBDIRS

# Release and container image per MiniAOD campaign prefix.
RELEASES = [
    ("RunIII2024Summer24MiniAODv6", "CMSSW_15_0_21", "el9"),
    ("Run3Summer22MiniAODv4", "CMSSW_13_0_13", "el8"),
    ("Run3Summer22EEMiniAODv4", "CMSSW_13_0_13", "el8"),
    ("Run3Summer23MiniAODv4", "CMSSW_13_0_13", "el8"),
    ("Run3Summer23BPixMiniAODv4", "CMSSW_13_0_13", "el8"),
    ("RunIISummer20UL", "CMSSW_10_6_30", "cc7"),
]
IMAGES = {"el9": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el9:x86_64",
          "el8": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el8:x86_64",
          "cc7": "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/cc7:x86_64"}

# PDG branching fractions used for the proposed factors (W -> l nu with l = e, mu, tau).
BR_W_LNU = 0.3258   # PDG 2024 lepton-universality fit (3 x 0.1086); the sum of the three measured values is 0.3272
BR = {"2L2Nu": BR_W_LNU ** 2, "LNu2Q": 2 * BR_W_LNU * (1 - BR_W_LNU), "4Q": (1 - BR_W_LNU) ** 2, "LNu": BR_W_LNU, "2Q": 1 - BR_W_LNU}

# Inclusive reference cross sections [pb] per centre-of-mass energy.  Two uses:
# (1) evidence for whether a generator header is inclusive (POWHEG hvq / ST_wtch print the undecayed
#     total even when the decays are restricted at LHE level) or already contains the decay
#     (MG5_aMC and POWHEG ST_tch_4f), by comparing the header with ref and with ref x BR;
# (2) the normalisation the config is scaled to with ``apply --normalize reference`` (k-factor = ref / header).
REFERENCES = {
    "ttbar": {"13p6TeV": (923.6, "NNLO+NNLL, TOP twiki TtbarNNLO, mt = 172.5 GeV"),
              "13TeV": (833.9, "NNLO+NNLL, TOP twiki TtbarNNLO, mt = 172.5 GeV")},
    "tW": {"13p6TeV": (43.95, "approx. NNLO per charge (87.9 pb for both), TOP twiki SingleTopNNLORef"),
           "13TeV": (35.85, "NLO+NNLL per charge (71.7 pb for both), TOP twiki SingleTopRefXsec")},
    "t-channel_top": {"13p6TeV": (145.0, "NNLO 5FS, TOP twiki SingleTopNNLORef"),
                      "13TeV": (136.02, "NLO 5FS, TOP twiki SingleTopRefXsec")},
    "t-channel_antitop": {"13p6TeV": (87.2, "NNLO 5FS, TOP twiki SingleTopNNLORef"),
                          "13TeV": (80.95, "NLO 5FS, TOP twiki SingleTopRefXsec")},
    "s-channel_top": {"13TeV": (6.35, "NLO, TOP twiki SingleTopRefXsec")},
    "s-channel_antitop": {"13TeV": (3.97, "NLO, TOP twiki SingleTopRefXsec")},
}
# Decay-channel tokens that distinguish sibling samples of one process (same header if it is inclusive).
CHANNEL_TOKENS = {"2L2Nu": "2L2Nu", "LNu2Q": "LNu2Q", "4Q": "4Q"}


def energy_of(das: str) -> str:
    return "13p6TeV" if "13p6TeV" in das or "Run3" in das or "RunIII" in das else "13TeV"


def era_of_miniaod(miniaod: str) -> str | None:
    """Config era of a MiniAOD dataset: ``.../Run3Summer23BPixMiniAODv4-.../MINIAODSIM`` -> ``Run3Summer23BPix``,
    ``RunIISummer20UL16MiniAODAPVv2`` -> ``RunIISummer20UL16APV``."""
    if not miniaod or miniaod.count("/") < 3:
        return None
    camp = miniaod.split("/")[2].split("-")[0]
    m = re.match(r"^(.*?)MiniAOD(APV)?v\d+$", camp)
    if not m:
        return None
    return m.group(1) + ("APV" if m.group(2) else "")


def sample_key(primary: str, energy: str, era: str) -> str:
    return f"{primary}|{energy}|{era}"


def migrate_results(results: dict) -> tuple[dict, int]:
    """Re-key records to ``primary|energy|era`` (era = the campaign whose MiniAOD was measured).

    Records written before 2026-10-06 were keyed by primary and energy only and could list eras
    they were never measured for; each record now stands for exactly the era of its MiniAOD.
    """
    out: dict = {}
    changed = 0
    for key, e in results.items():
        era = e.get("era") or era_of_miniaod(e.get("miniaod", "")) or (e.get("eras") or [None])[0]
        new_key = sample_key(e.get("primary", key.split("|")[0]), e.get("energy", ""), era) if era else key
        rec = dict(e)
        if era and (rec.get("era") != era or rec.get("eras") != [era]):
            rec["era"] = era
            rec["eras"] = [era]
            changed += 1
        if new_key != key:
            changed += 1
        if new_key in out and out[new_key].get("xsec_gen") is not None and rec.get("xsec_gen") is None:
            continue
        out[new_key] = rec
    return out, changed


def load_results(work: Path) -> tuple[Path, dict]:
    path = work / "genxsec_results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results, changed = migrate_results(results)
    if changed:
        path.write_text(json.dumps(results, indent=1) + "\n")
        print(f"migrated {changed} record field(s) to per-era keys in {path}")
    return path, results


def release_for(miniaod: str) -> tuple[str, str]:
    camp = miniaod.split("/")[2]
    for prefix, rel, img in RELEASES:
        if camp.startswith(prefix):
            return rel, img
    raise ValueError(f"no CMSSW release mapping for {camp}")


# ----------------------------------------------------------------------------
# content inspection (NanoAOD over xrootd)
# ----------------------------------------------------------------------------

def inspect_content(nano_das: str, dbs: DBSClient, n_events: int = 20_000) -> dict:
    import uproot
    import numpy as np
    import awkward as ak

    files = dbs.files(nano_das)
    if not files:
        return {"error": "no files"}
    url = "root://cmsxrootd.fnal.gov/" + files[0].lfn
    with uproot.open(url, timeout=120) as f:
        t = f["Events"]
        keys = set(t.keys())
        want = [k for k in ("LHEPart_pdgId", "LHEPart_status", "GenPart_pdgId", "GenPart_status", "GenPart_genPartIdxMother", "GenPart_statusFlags") if k in keys]
        a = t.arrays(want, entry_stop=n_events)
        n = len(a)
        out: dict = {"events_inspected": n, "has_lhe": "LHEPart_pdgId" in keys}
        if "LHEPart_pdgId" in keys:
            pid = a["LHEPart_pdgId"]
            st = a["LHEPart_status"] if "LHEPart_status" in keys else ak.ones_like(pid)
            fin = pid[st == 1]
            apid = abs(fin)
            per_event_leptons = ak.sum((apid == 11) | (apid == 13) | (apid == 15), axis=1)
            out["lhe_lepton_flavours"] = {fl: int(ak.sum(ak.flatten(apid == code))) for fl, code in (("e", 11), ("mu", 13), ("tau", 15))}
            out["lhe_mean_charged_leptons"] = float(ak.mean(per_event_leptons))
            out["lhe_mean_neutrinos"] = float(ak.mean(ak.sum((apid == 12) | (apid == 14) | (apid == 16), axis=1)))
            out["lhe_mean_quarks"] = float(ak.mean(ak.sum((apid >= 1) & (apid <= 5), axis=1)))
            out["lhe_tops"] = float(ak.mean(ak.sum(apid == 6, axis=1)))
            out["lhe_W"] = float(ak.mean(ak.sum(apid == 24, axis=1)))
            out["lhe_Z"] = float(ak.mean(ak.sum(apid == 23, axis=1)))
        if "GenPart_pdgId" in keys:
            gp = a["GenPart_pdgId"]; mom = a["GenPart_genPartIdxMother"]
            agp = abs(gp)
            # W decays: charged leptons / quarks whose mother is a W
            mother_pid = abs(gp[mom])  # mom == -1 wraps to the last particle; mask it
            valid = mom >= 0
            from_w = valid & (mother_pid == 24)
            lep_from_w = from_w & ((agp == 11) | (agp == 13) | (agp == 15))
            q_from_w = from_w & (agp >= 1) & (agp <= 5)
            n_lep_w = ak.sum(lep_from_w, axis=1)
            n_q_w = ak.sum(q_from_w, axis=1) // 2
            n_w_dec = n_lep_w + n_q_w
            has_w = n_w_dec > 0
            out["w_decays_mean"] = float(ak.mean(n_w_dec))
            if ak.any(has_w):
                frac = {}
                for nl in (0, 1, 2):
                    frac[f"{nl}_leptonic_W"] = float(ak.mean(n_lep_w[has_w] == nl))
                out["w_decay_fractions"] = frac
                tau_w = ak.sum(from_w & (agp == 15), axis=1)
                out["w_to_tau_fraction"] = float(ak.sum(tau_w) / max(ak.sum(n_lep_w), 1))
            lep_from_z = valid & (mother_pid == 23) & ((agp == 11) | (agp == 13) | (agp == 15))
            if ak.any(lep_from_z):
                out["z_lepton_flavours"] = {fl: int(ak.sum(ak.flatten(lep_from_z & (agp == code)))) for fl, code in (("e", 11), ("mu", 13), ("tau", 15))}
        return out


# ----------------------------------------------------------------------------
# GenXSecAnalyzer in the VM
# ----------------------------------------------------------------------------

def parse_genxsec_output(txt: str) -> dict:
    """The numbers GenXSecAnalyzer prints at the end of a cmsRun."""
    res: dict = {}
    m = re.search(r"After filter: final cross section = ([0-9.e+-]+) \+- ([0-9.e+-]+) pb", txt)
    if m:
        res["xsec_gen"], res["xsec_gen_unc"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Before matching: total cross section = ([0-9.e+-]+)", txt)
    if m:
        res["xsec_before_matching"] = float(m.group(1))
    m = re.search(r"Filter efficiency \(event-level\)= \(([0-9.e+-]+)\) / \(([0-9.e+-]+)\) = ([0-9.e+-]+)", txt)
    if m:
        res["filter_eff"] = float(m.group(3)); res["events_run"] = int(float(m.group(2)))
    m = re.search(r"final fraction of events with negative weights = ([0-9.e+-]+)", txt)
    if m:
        res["neg_weight_fraction"] = float(m.group(1))
    if "xsec_gen" not in res:
        res["error"] = txt[-1500:]
    return res


def run_genxsec(local_files: list[str], release: str, image: str, work: Path, vm: str = "combine") -> dict:
    script = work / "run_genxsec.sh"
    cmd = ["orb", "-m", vm, "bash", "-c",
           f"apptainer exec -B /cvmfs -B /Users/billy {IMAGES[image]} {script} {release} -1 " + " ".join(local_files)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
    res = parse_genxsec_output(proc.stdout + proc.stderr)
    res.update({"seconds": time.monotonic() - t0, "release": release, "rc": proc.returncode})
    return res


# ----------------------------------------------------------------------------
# campaign
# ----------------------------------------------------------------------------

def unique_samples(config_paths: list[Path]) -> dict[str, dict]:
    """key = primary|energy|era -> {primary, energy, era, eras=[era], das, xsec_config, physics_group, dataset}.

    Every era is measured on its own MiniAOD: campaigns of different years can share a primary
    dataset name and still differ in gridpack, PDF or generator version.
    """
    out: dict[str, dict] = {}
    for das, meta, cp in load_config_entries(config_paths, include_optional=True):
        if meta.get("datatype") != "mc" or not is_das_path(das):
            continue
        if meta.get("scan"):
            continue   # multi-model scan samples: the per-point cross sections come from the model config
        prim = das.split("/")[1]
        era = meta.get("era")
        key = sample_key(prim, energy_of(das), era)
        out.setdefault(key, {"primary": prim, "energy": energy_of(das), "era": era, "eras": [era], "das": das,
                             "xsec_config": meta.get("xsec"), "physics_group": meta.get("physics_group"),
                             "dataset": meta.get("dataset")})
    return out


def cmd_run(args) -> int:
    work = Path(args.work).expanduser()
    (work / "inputs").mkdir(parents=True, exist_ok=True)
    results_path, results = load_results(work)
    dbs = DBSClient()
    cfgs = []
    for era in args.era or sorted(ERA_SUBDIRS):
        sub = ERA_SUBDIRS[era]
        cfgs += [p for p in sorted((REPO / "data" / "configs" / sub).glob("*.json")) if "archived" not in p.parts]
    samples = unique_samples(cfgs)
    if args.dataset:
        samples = {k: v for k, v in samples.items() if any(s in k for s in args.dataset)}
    if args.skip_signal:
        samples = {k: v for k, v in samples.items() if v["physics_group"] != "Signal"}
    todo = [k for k in samples if args.force or "xsec_gen" not in results.get(k, {})]
    print(f"{len(samples)} unique samples, {len(todo)} to do")

    def prepare(key):
        s = samples[key]
        try:
            parents = [p["parent_dataset"] for p in dbs.get("datasetparents", dataset=s["das"])]
            mini = next((p for p in parents if p.endswith("MINIAODSIM")), None)
            if mini is None:
                return key, {"error": f"no MiniAOD parent for {s['das']}"}
            files = sorted(dbs.files(mini), key=lambda f: f.size)
            n_files = 2 if "amcatnlo" in s["primary"].lower() else 1
            picks = [files[len(files) // 2]] + ([files[len(files) // 3]] if n_files > 1 and len(files) > 1 else [])
            local = []
            for f in picks:
                dest = work / "inputs" / (s["primary"][:80] + "_" + os.path.basename(f.lfn))
                if not dest.exists():
                    r = fetch_file(f.lfn, str(dest), ["root://cmsxrootd.fnal.gov/" + f.lfn, "root://cms-xrd-global.cern.ch/" + f.lfn],
                                   adler32=f.adler32, size=f.size, streams=4, min_rate_bytes_s=3e5, base_timeout=600)
                    if not r.ok:
                        return key, {"error": f"download failed: {[a.category for a in r.attempts]}"}
                local.append(str(dest))
            content = inspect_content(s["das"], dbs)
            return key, {"miniaod": mini, "local_files": local, "content": content}
        except Exception as e:
            return key, {"error": f"{type(e).__name__}: {e}"}

    # Pipeline: a bounded prefetch of downloads runs ahead of the (serial) cmsRun jobs,
    # so results appear from the first sample on and scratch holds only a few files.
    from collections import deque
    import concurrent.futures as cf

    def analyze(key, prep):
        if "error" in prep:
            return prep
        rel, img = release_for(prep["miniaod"])
        res = run_genxsec(prep["local_files"], rel, img, work, vm=args.vm)
        res.update({k: v for k, v in prep.items() if k != "local_files"})
        return res

    pending = deque(todo)
    inflight: deque = deque()
    with ThreadPoolExecutor(max_workers=args.downloads) as pool:
        def submit_next():
            if pending:
                k = pending.popleft()
                inflight.append((k, pool.submit(prepare, k)))
        for _ in range(max(1, args.downloads + 1)):
            submit_next()
        i = 0
        while inflight:
            key, fut = inflight.popleft()
            _, prep = fut.result()
            submit_next()
            i += 1
            s = samples[key]
            res = analyze(key, prep)
            results[key] = {**s, **res, "done": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            results_path.write_text(json.dumps(results, indent=1) + "\n")
            xs = res.get("xsec_gen")
            print(f"[{i}/{len(todo)}] {s['primary'][:60]:60s} gen={xs if xs is None else f'{xs:.5g}'} config={s['xsec_config']} "
                  f"{'ERROR: ' + res['error'][:80] if 'error' in res else ''}", flush=True)
            if not args.keep_inputs:
                for pth in prep.get("local_files", []):
                    try:
                        os.remove(pth)
                    except FileNotFoundError:
                        pass
    return 0


def process_of(prim: str) -> tuple[str | None, str]:
    """(process, charge) guessed from the primary-dataset name: only used to pick a reference and to group siblings."""
    p = prim.lower()
    if p.startswith("tt") and not p.startswith(("ttl", "ttw", "ttz", "ttg", "tth", "tttt", "ttbb")):
        return "ttbar", "both"
    if p.startswith(("twminus", "st_tw_antitop", "tbarw")) or p.startswith("st_tw_top") or p.startswith("twplus"):
        return "tW", "top" if p.startswith(("twplus", "st_tw_top")) else "antitop"
    if "t-channel" in p:
        if "_top_" in p or p.startswith("tbbarq"):
            return "t-channel", "top"
        if "_antitop_" in p or p.startswith("tbarbq"):
            return "t-channel", "antitop"
        return "t-channel", "both"
    if "s-channel" in p:
        if p.startswith("tbbarto"):
            return "s-channel", "top"
        if p.startswith("tbarbto"):
            return "s-channel", "antitop"
        return "s-channel", "both"
    return None, ""


def reference_for(prim: str, energy: str) -> tuple[float, str] | None:
    proc, charge = process_of(prim)
    if proc is None:
        return None
    if proc in ("ttbar", "tW"):
        return REFERENCES[proc].get(energy)
    if charge == "both":
        top = REFERENCES.get(f"{proc}_top", {}).get(energy)
        anti = REFERENCES.get(f"{proc}_antitop", {}).get(energy)
        if top and anti:
            return top[0] + anti[0], f"top + antitop: {top[1]}"
        return None
    return REFERENCES.get(f"{proc}_{charge}", {}).get(energy)


def restricted_channel(content: dict) -> tuple[str | None, float]:
    """The decay channel every event falls into (> 97 %), as a BR key, else None."""
    wf = content.get("w_decay_fractions") or {}
    mean = content.get("w_decays_mean", 0.0) or 0.0
    if wf and mean >= 1.5:
        for ch, key in (("2L2Nu", "2_leptonic_W"), ("LNu2Q", "1_leptonic_W"), ("4Q", "0_leptonic_W")):
            if wf.get(key, 0) > 0.97:
                return ch, mean
    elif wf and 0.5 <= mean < 1.5:
        if wf.get("1_leptonic_W", 0) > 0.97:
            return "LNu", mean
        if wf.get("0_leptonic_W", 0) > 0.97:
            return "2Q", mean
    return None, mean


def sibling_headers(entry: dict, results: dict | None) -> list[tuple[str, float]]:
    """Generator values of samples that differ from *entry* only by the decay-channel token."""
    if not results:
        return []
    prim = entry["primary"]
    token = next((t for t in CHANNEL_TOKENS if t in prim), None)
    if token is None:
        return []
    stem = prim.replace(token, "@")
    out = []
    for e in results.values():
        if e is entry or e.get("energy") != entry.get("energy") or e.get("xsec_gen") is None:
            continue
        if e.get("era") != entry.get("era"):
            continue      # siblings must come from the same campaign
        t2 = next((t for t in CHANNEL_TOKENS if t in e["primary"]), None)
        if t2 and t2 != token and e["primary"].replace(t2, "@") == stem:
            out.append((e["primary"], e["xsec_gen"]))
    return out


def classify(entry: dict, results: dict | None = None) -> dict:
    """Decide how the generator value maps to a config cross section.

    Returns ``factor`` (flavour / branching-fraction multiplier, None when undecidable), ``kfactor``
    (reference / header-equivalent, 1 without a reference), the reference used, and the evidence.
    """
    c = entry.get("content") or {}
    gen = entry.get("xsec_gen")
    prim = entry["primary"]
    energy = entry.get("energy", "")
    out = {"factor": 1.0, "factor_reason": "generator value taken as is", "kfactor": 1.0,
           "reference": None, "reference_source": "", "header": "inclusive", "evidence": ""}
    if gen is None:
        out.update(factor=None, factor_reason="no generator value")
        return out
    fl = c.get("lhe_lepton_flavours") or {}
    present = [k for k, v in fl.items() if v > 0]
    low = prim.lower()
    # --- Drell-Yan: the header of a single-flavour MG5_aMC sample is the three-flavour cross section
    if low.startswith(("dyto2", "dyjetstoll", "dy")) and c.get("w_decays_mean", 0) < 0.5:
        if len(present) == 1:
            if "powheg" in low:
                out.update(factor=1.0, header="per flavour",
                           factor_reason="POWHEG header is the per-flavour cross section (the powheg mass bins "
                                         "sum to the FxFX three-flavour value / 3; see the review cross-check)")
            else:
                out.update(factor=1 / 3, header="three-flavour",
                           factor_reason="MG5_aMC header is the three-flavour cross section although the events "
                                         "hold one flavour (DYto2E-2Jets_MLL-50 header 6685 pb = three-flavour "
                                         "NLO 6688 pb; powheg per-flavour bins sum to 2244 pb)")
        if "madgraphmlm" in low:
            # LO sample: scale to the NLO inclusive value measured at the same energy (explicit, flat k-factor,
            # the Run 2 convention of LO x k), so every DY variant carries the same normalisation.
            k = dy_lo_kfactor(results, energy, entry.get("era"))
            if k is None:
                out.update(factor_reason=out["factor_reason"] + "; LO sample but no NLO and LO inclusive headers "
                                                                 "measured at this energy: k-factor left at 1 (CHECK)",
                           header=out["header"] + " (LO, k undecided)")
                out["undecided_k"] = True
            else:
                nlo = _dy_inclusive_headers(results, energy, "amcatnlo", entry.get("era"))
                out.update(kfactor=k[0], reference=sum(x for _, x in nlo) / len(nlo),
                           reference_source="NLO FxFX inclusive three-flavour header at this energy (GenXSec)",
                           evidence=f"LO -> NLO k-factor {k[0]:.4f} = {k[1]}")
        return out
    # --- restricted W decays: is the header inclusive or does it already include the branching fraction?
    channel, _ = restricted_channel(c)
    ref = reference_for(prim, energy)
    if ref:
        out.update(reference=ref[0], reference_source=ref[1])
    if channel is None:
        if ref:
            out.update(kfactor=ref[0] / gen, evidence=f"inclusive decays; header / reference = {gen / ref[0]:.3f}")
        return out
    br = BR[channel]
    sibs = sibling_headers(entry, results)
    if sibs and all(abs(x / gen - 1) < 0.02 for _, x in sibs):
        out.update(factor=br, header="inclusive",
                   factor_reason=f"all events are {channel} but the header is inclusive: the sibling samples "
                                 f"{', '.join(n.split('_')[0] for n, _ in sibs)} carry the same header value",
                   evidence="; ".join(f"{n.split('_')[0]} {x:.5g}" for n, x in sibs))
        if ref:
            out["kfactor"] = ref[0] / gen
        return out
    if ref:
        r_incl, r_dec = gen / ref[0], gen / (ref[0] * br)
        if 0.65 <= r_dec <= 1.2 and not (0.65 <= r_incl <= 1.2):
            out.update(factor=1.0, header="includes decay", kfactor=ref[0] * br / gen,
                       factor_reason=f"all events are {channel} and the header already includes BR = {br:.4f}: "
                                     f"header / (reference x BR) = {r_dec:.3f} while header / reference = {r_incl:.3f}",
                       evidence=f"reference {ref[0]} pb ({ref[1]})")
            return out
        if 0.65 <= r_incl <= 1.2 and not (0.65 <= r_dec <= 1.2):
            out.update(factor=br, header="inclusive", kfactor=ref[0] / gen,
                       factor_reason=f"all events are {channel} but the header is inclusive: header / reference = "
                                     f"{r_incl:.3f} while header / (reference x BR) = {r_dec:.3f}",
                       evidence=f"reference {ref[0]} pb ({ref[1]})")
            return out
        out.update(factor=None, factor_reason=f"all events are {channel}; header matches neither the reference "
                                              f"({r_incl:.3f}) nor reference x BR ({r_dec:.3f}): decide by hand")
        return out
    if "madgraph" in low or "amcatnlo" in low:
        out.update(factor=1.0, header="includes decay",
                   factor_reason=f"all events are {channel}; MG5_aMC generates the decay in the matrix element, "
                                 f"so the header includes the branching fraction (no reference to verify)")
        return out
    out.update(factor=None, factor_reason=f"all events are {channel}; POWHEG sample with restricted decays, no "
                                          f"sibling channel and no reference (hvq/ST_wtch headers are inclusive, "
                                          f"ST_tch_4f headers are not): decide by hand")
    return out


def _dy_inclusive_headers(results: dict, energy: str, generator: str, era: str | None = None) -> list[tuple[str, float]]:
    """Three-flavour-equivalent inclusive (M > 50 GeV) DY headers of one generator class at *energy* in *era*."""
    out = []
    for e in results.values():
        if e.get("energy") != energy or e.get("xsec_gen") is None:
            continue
        if e.get("era") != era:
            continue
        low = e["primary"].lower()
        if generator not in low:
            continue
        # 2024 "..._Bin-MLL-50_Tune...", 2022/23 "..._MLL-50_Tune..."; jet- and pT-binned siblings
        # ("_MLL-50_1J_", "_MLL-50_PTLL-100_", "_Bin-0J-MLL-50_") are not inclusive
        incl = (re.match(r"dyto2(e|mu|l)-[24]jets_(bin-)?mll-50_tune", low) is not None
                or re.match(r"dyjetstoll_m-50_tune", low) is not None)
        if incl:
            out.append((e["primary"], e["xsec_gen"]))   # MG5 headers are three-flavour whatever the sample's flavour
    return out


def dy_lo_kfactor(results: dict | None, energy: str, era: str | None = None) -> tuple[float, str] | None:
    """NLO / LO inclusive DY header ratio at *energy* in *era* (both MG5, both three-flavour headers), with its evidence."""
    if not results:
        return None
    nlo = _dy_inclusive_headers(results, energy, "amcatnlo", era)
    lo = _dy_inclusive_headers(results, energy, "madgraphmlm", era)
    if not nlo or not lo:
        return None
    k = (sum(x for _, x in nlo) / len(nlo)) / (sum(x for _, x in lo) / len(lo))
    ev = (f"NLO FxFX inclusive header {', '.join(f'{n.split(chr(95))[0]} {x:.5g}' for n, x in nlo)} / "
          f"LO MLM inclusive header {', '.join(f'{n.split(chr(95))[0]} {x:.5g}' for n, x in lo)}")
    return k, ev


def proposed_xsec(entry: dict, d: dict, normalize: str) -> float | None:
    if d.get("factor") is None or entry.get("xsec_gen") is None:
        return None
    k = d["kfactor"] if normalize == "reference" else 1.0
    return entry["xsec_gen"] * d["factor"] * k


def dy_crosscheck(results: dict) -> list[str]:
    """Sum of the POWHEG per-flavour mass bins (M > 50) against the FxFX inclusive header / 3."""
    lines = []
    for energy, era in sorted({(e.get("energy"), e.get("era") or "") for e in results.values()}):
        for fl in ("E", "Mu"):
            same = [e for e in results.values() if e.get("energy") == energy and (e.get("era") or "") == era
                    and e.get("xsec_gen") is not None]
            bins = [e for e in same if e["primary"].startswith(f"DYto2{fl}_Bin-MLL-") and "powheg" in e["primary"]
                    and "10to50" not in e["primary"]]
            fx = [e for e in same if e["primary"].startswith(f"DYto2{fl}-2Jets_Bin-MLL-50_")]
            if bins and fx:
                tot = sum(e["xsec_gen"] for e in bins)
                lines.append(f"- {era or energy} DYto2{fl}: sum of {len(bins)} POWHEG mass bins (M > 50 GeV) = {tot:.1f} pb; "
                             f"FxFX MLL-50 header / 3 = {fx[0]['xsec_gen'] / 3:.1f} pb (ratio {tot / (fx[0]['xsec_gen'] / 3):.3f})")
    return lines


def cmd_review(args) -> int:
    work = Path(args.work).expanduser()
    _, results = load_results(work)
    head = ("| sample | energy | header xsec [pb] | unc | LHE flavours (e/mu/tau) | W decays (0/1/2 leptonic) | header | "
            "factor | k-factor | proposed (gen) | proposed (ref) | config xsec | ratio | eras |")
    lines = [head, "|" + "---|" * 14]
    n_check = 0
    for key, e in sorted(results.items()):
        d = classify(e, results)
        xs = e.get("xsec_gen")
        prop_gen = proposed_xsec(e, d, "generator")
        prop_ref = proposed_xsec(e, d, "reference")
        cfg = e.get("xsec_config")
        prop = prop_ref if d.get("reference") else prop_gen
        ratio = (prop / cfg) if (prop and cfg) else None
        c = e.get("content") or {}
        fl = c.get("lhe_lepton_flavours") or {}
        wf = c.get("w_decay_fractions") or {}
        flag = ""
        if d.get("factor") is None or d.get("undecided_k") or ratio is None or abs(ratio - 1) > 0.05:
            flag = " **CHECK**"
            n_check += 1
        fmt = lambda v, g=5: "" if v is None else f"{v:.{g}g}"
        wdec = f"{wf.get('0_leptonic_W', 0):.2f}/{wf.get('1_leptonic_W', 0):.2f}/{wf.get('2_leptonic_W', 0):.2f}" if wf else "-"
        lines.append(f"| {e['primary'][:55]} | {e['energy']} | {fmt(xs)} | {fmt(e.get('xsec_gen_unc'), 3)} | "
                     f"{fl.get('e', '')}/{fl.get('mu', '')}/{fl.get('tau', '')} | {wdec} | {d['header']} | "
                     f"{fmt(d.get('factor'), 4)} | {fmt(d.get('kfactor'), 4)} | {fmt(prop_gen)} | {fmt(prop_ref)} | "
                     f"{cfg} | {fmt(ratio, 3)}{flag} | {','.join(e.get('eras', []))} |")
    reasons = ["", "## Decisions", ""]
    for key, e in sorted(results.items()):
        d = classify(e, results)
        reasons.append(f"- `{e['primary'][:70]}`: {d['factor_reason']}"
                       + (f"; reference {d['reference']} pb ({d['reference_source']})" if d.get("reference") else "")
                       + (f"; evidence: {d['evidence']}" if d.get("evidence") else ""))
    checks = dy_crosscheck(results)
    out = work / "genxsec_review.md"
    out.write_text(
        "# GenXSecAnalyzer review\n\n"
        "header: what the generator cross section is (inclusive, includes decay, per flavour, three-flavour). "
        "factor: flavour / branching-fraction multiplier from the header to the physical cross section of the sample. "
        "k-factor: reference / header-equivalent, applied by `apply --normalize reference` (the current config "
        "convention for ttbar and tW); `proposed (gen)` = header x factor, `proposed (ref)` = header x factor x k. "
        "ratio compares the config with proposed (ref) when a reference exists, else with proposed (gen). "
        "Rows marked CHECK differ from the current config by more than 5 % or could not be decided.\n\n"
        + "\n".join(lines) + "\n"
        + ("\n## Cross-checks\n\n" + "\n".join(checks) + "\n" if checks else "")
        + "\n".join(reasons) + "\n")
    print(f"wrote {out} ({len(lines) - 2} samples, {n_check} marked CHECK)")
    for line in checks:
        print(line)
    return 0


def _config_files(eras: list[str] | None) -> list[Path]:
    cfgs = []
    for era in eras or sorted(ERA_SUBDIRS):
        sub = ERA_SUBDIRS[era]
        cfgs += [p for p in sorted((REPO / "data" / "configs" / sub).glob("*.json")) if "archived" not in p.parts]
    return cfgs


def _sig(v: float, n: int = 5) -> float:
    return float(f"{v:.{n}g}")


def cmd_apply(args) -> int:
    """Write the decided cross sections into the config JSONs (dry run unless --write)."""
    work = Path(args.work).expanduser()
    _, results = load_results(work)
    today = time.strftime("%Y-%m-%d")
    rows = []
    changes: dict[Path, dict] = {}
    skipped = []
    for key, e in sorted(results.items()):
        if args.dataset and not any(s in key for s in args.dataset):
            continue
        if args.era and not args.any_era and not set(e.get("eras") or []) & set(args.era):
            continue          # measured for another campaign
        d = classify(e, results)
        new = proposed_xsec(e, d, args.normalize)
        if new is None:
            skipped.append((e["primary"], d["factor_reason"]))
            continue
        k = d["kfactor"] if args.normalize == "reference" else 1.0
        unc = e.get("xsec_gen_unc")
        source = {"generator": e["xsec_gen"], "generator_unc": unc, "header": d["header"],
                  "factor": _sig(d["factor"], 6), "factor_reason": d["factor_reason"],
                  "kfactor": _sig(k, 6), "reference": d.get("reference"), "reference_source": d.get("reference_source", ""),
                  "normalize": args.normalize, "release": e.get("release"), "events_run": e.get("events_run"),
                  "tool": "GenXSecAnalyzer", "date": today}
        hits = 0
        for cp in _config_files(args.era):
            cfg = json.loads(cp.read_text())
            for das, meta in cfg.items():
                if not isinstance(meta, dict) or not is_das_path(das):
                    continue
                if das.split("/")[1] != e["primary"] or energy_of(das) != e.get("energy"):
                    continue
                if not args.any_era and meta.get("era") not in (e.get("eras") or []):
                    continue   # the header was measured on another campaign's files: run GenXSec for that era
                hits += 1
                old = meta.get("xsec")
                upd = changes.setdefault(cp, {})
                upd[das] = {"xsec": _sig(new), "total_uncertainty": _sig(unc * d["factor"] * k, 4) if unc else meta.get("total_uncertainty"),
                            "xsec_source": source}
                rows.append((e["primary"], meta.get("era"), old, _sig(new), d["factor"], k, cp.name))
        if hits == 0:
            skipped.append((e["primary"], "no config entry found for this primary dataset and energy"))
    lines = ["| sample | era | config xsec | new xsec | change | factor | k-factor | file |", "|---|---|---|---|---|---|---|---|"]
    for prim, era, old, new, f, k, fn in rows:
        ch = "" if not old else f"{(new / old - 1) * 100:+.1f} %"
        lines.append(f"| {prim[:55]} | {era} | {old} | {new:.5g} | {ch} | {f:.4g} | {k:.4g} | {fn} |")
    plan = work / "genxsec_apply_plan.md"
    plan.write_text(f"# Cross-section update plan ({'written' if args.write else 'dry run'}, normalize = {args.normalize}, {today})\n\n"
                    + "\n".join(lines) + "\n\n## Skipped\n\n" + "\n".join(f"- `{p}`: {why}" for p, why in skipped) + "\n")
    n_big = sum(1 for _, _, old, new, _, _, _ in rows if old and abs(new / old - 1) > 0.05)
    print(f"{len(rows)} config entries would change ({n_big} by more than 5 %), {len(skipped)} samples skipped; plan: {plan}")
    for p, why in skipped:
        print(f"  skipped {p[:60]}: {why[:120]}")
    if not args.write:
        print("dry run: rerun with --write to modify the configs")
        return 0
    for cp, upd in changes.items():
        text = cp.read_text()
        cfg = json.loads(text)
        for das, fields in upd.items():
            cfg[das].update(fields)
        indent = len(text.splitlines()[1]) - len(text.splitlines()[1].lstrip()) if len(text.splitlines()) > 1 else 1
        cp.write_text(json.dumps(cfg, indent=indent or 1) + "\n")
        print(f"updated {cp.relative_to(REPO)}: {len(upd)} entries")
    print("remember: existing plans keep the old metadata until `bin/skim.py refresh-metadata --plan-dir ...` is run")
    return 0


# ----------------------------------------------------------------------------
# Condor mode: cmsRun streams the MiniAOD files at the LPC instead of downloading them here
# ----------------------------------------------------------------------------

SCRAM_ARCH = {"CMSSW_13_0_13": "el8_amd64_gcc11", "CMSSW_15_0_21": "el9_amd64_gcc12", "CMSSW_10_6_30": "slc7_amd64_gcc700"}
GENXSEC_CFG_URL = ("https://raw.githubusercontent.com/cms-sw/genproductions/master/Utilities/"
                   "calculateXSectionAndFilterEfficiency/genXsec_cfg.py")

JOB_SCRIPT = r"""#!/bin/bash
# GenXSecAnalyzer on MiniAOD files streamed over xrootd, one cmsRun per sample.
# Arguments: sample list (lines "<name> <comma-separated URLs>"), CMSSW release, SCRAM_ARCH.
LIST="$1"; REL="$2"; ARCH="$3"
echo "=== $(date -u) $(hostname) $LIST $REL $ARCH ==="
export SCRAM_ARCH="$ARCH"
source /cvmfs/cms.cern.ch/cmsset_default.sh
scram project CMSSW "$REL" > scram.log 2>&1 || { cat scram.log; exit 3; }
pushd "$REL/src" > /dev/null && eval "$(scramv1 runtime -sh)" && popd > /dev/null
mkdir -p out
while read -r name urls; do
  [ -z "$name" ] && continue
  echo "=== $(date -u) $name"
  timeout 5400 cmsRun genXsec_cfg.py inputFiles="$urls" maxEvents=-1 > "cmsrun_$name.log" 2>&1
  rc=$?
  { tail -400 "cmsrun_$name.log"; echo "cmsRun rc=$rc"; } > "out/$name.log"
  rm -f "cmsrun_$name.log"
done < "$LIST"
rm -rf "$REL"
echo "=== $(date -u) done ==="
"""

JOB_JDL = """universe                = vanilla
executable              = genxsec_job.sh
arguments               = "$(listfile) $(release) $(arch)"
+SingularityImage       = "$(image)"
+SingularityBindCVMFS   = True
use_x509userproxy       = true
x509userproxy           = {proxy}
request_cpus            = 1
request_memory          = 2500
request_disk            = 4000000
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = genXsec_cfg.py, lists/$(listfile)
transfer_output_files   = out
output                  = logs/$(listfile).out
error                   = logs/$(listfile).err
log                     = logs/condor.log
notification            = never
queue listfile,release,arch,image from jobs.txt
"""


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", key)


def prepare_sample(s: dict, dbs, *, inspect: bool = True) -> dict:
    """MiniAOD parent, the files to measure (median-size file; two for aMC@NLO) and the NanoAOD content."""
    parents = [p["parent_dataset"] for p in dbs.get("datasetparents", dataset=s["das"])]
    mini = next((p for p in parents if p.endswith("MINIAODSIM")), None)
    if mini is None:
        return {"error": f"no MiniAOD parent for {s['das']}"}
    files = sorted((f for f in dbs.files(mini) if f.valid), key=lambda f: f.size)
    if not files:
        return {"error": f"no valid files in {mini}"}
    n_files = 2 if "amcatnlo" in s["primary"].lower() else 1
    picks = [files[len(files) // 2]] + ([files[len(files) // 3]] if n_files > 1 and len(files) > 1 else [])
    rel, img = release_for(mini)
    out = {"miniaod": mini, "lfns": [f.lfn for f in picks], "release": rel, "image": img,
           "input_bytes": sum(f.size for f in picks)}
    if inspect:
        out["content"] = inspect_content(s["das"], dbs)
    return out


def write_condor_jobs(prepared: dict, jobs_dir: Path, *, per_job: int, proxy: str, cfg_text: str,
                      redirector: str = "root://cmsxrootd.fnal.gov/") -> int:
    """Lists of samples per job, grouped by release; returns the number of jobs."""
    (jobs_dir / "lists").mkdir(parents=True, exist_ok=True)
    (jobs_dir / "logs").mkdir(exist_ok=True)
    by_rel: dict[tuple[str, str], list[str]] = {}
    for key, p in sorted(prepared.items()):
        if "error" in p:
            continue
        by_rel.setdefault((p["release"], p["image"]), []).append(key)
    lines = []
    n = 0
    for (rel, img), keys in sorted(by_rel.items()):
        for i in range(0, len(keys), per_job):
            name = f"job_{n:04d}.txt"
            (jobs_dir / "lists" / name).write_text("".join(
                f"{safe_name(k)} {','.join(redirector + l for l in prepared[k]['lfns'])}\n" for k in keys[i:i + per_job]))
            lines.append(f"{name} {rel} {SCRAM_ARCH[rel]} {IMAGES[img]}\n")
            n += 1
    (jobs_dir / "jobs.txt").write_text("".join(lines))
    (jobs_dir / "genxsec_job.sh").write_text(JOB_SCRIPT)
    (jobs_dir / "genxsec_job.sh").chmod(0o755)
    (jobs_dir / "submit.jdl").write_text(JOB_JDL.format(proxy=proxy))
    (jobs_dir / "genXsec_cfg.py").write_text(cfg_text)
    return n


def _genxsec_cfg_text(work: Path) -> str:
    for cand in sorted(work.glob("CMSSW_*/src/genXsec_cfg.py")):
        return cand.read_text()
    import urllib.request
    with urllib.request.urlopen(GENXSEC_CFG_URL, timeout=60) as r:
        return r.read().decode()


def cmd_prepare_condor(args) -> int:
    work = Path(args.work).expanduser()
    jobs_dir = Path(args.jobs_dir).expanduser()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _, results = load_results(work)
    samples = unique_samples(_config_files(args.era))
    if args.dataset:
        samples = {k: v for k, v in samples.items() if any(x in k for x in args.dataset)}
    if args.skip_signal:
        samples = {k: v for k, v in samples.items() if v["physics_group"] != "Signal"}
    todo = [k for k in samples if args.force or results.get(k, {}).get("xsec_gen") is None]
    print(f"{len(samples)} samples in {', '.join(args.era)}, {len(todo)} to measure")
    dbs = DBSClient()
    prepared: dict[str, dict] = {}
    import concurrent.futures as cf

    def one(k):
        try:
            return k, {**samples[k], **prepare_sample(samples[k], dbs, inspect=not args.no_inspect)}
        except Exception as e:
            return k, {**samples[k], "error": f"{type(e).__name__}: {e}"}

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (k, p) in enumerate(ex.map(one, todo), 1):
            prepared[k] = p
            if "error" in p:
                print(f"  {k}: {p['error'][:150]}")
            if i % 50 == 0:
                print(f"  prepared {i}/{len(todo)}", flush=True)
    (jobs_dir / "samples.json").write_text(json.dumps(prepared, indent=1) + "\n")
    proxy = args.proxy
    n_jobs = write_condor_jobs(prepared, jobs_dir, per_job=args.per_job, proxy=proxy, cfg_text=_genxsec_cfg_text(work))
    n_err = sum("error" in p for p in prepared.values())
    print(f"{len(prepared) - n_err} samples in {n_jobs} job(s) written to {jobs_dir} ({n_err} could not be prepared); "
          f"copy the directory to the LPC and run condor_submit submit.jdl there")
    return 0 if not n_err else 1


def cmd_collect(args) -> int:
    """Parse the logs Condor brought back (``<jobs-dir>/out``) into genxsec_results.json."""
    work = Path(args.work).expanduser()
    jobs_dir = Path(args.jobs_dir).expanduser()
    results_path, results = load_results(work)
    prepared = json.loads((jobs_dir / "samples.json").read_text())
    done = missing = failed = 0
    for key, p in sorted(prepared.items()):
        if "error" in p:
            continue
        log = jobs_dir / "out" / f"{safe_name(key)}.log"
        if not log.exists():
            missing += 1
            continue
        res = parse_genxsec_output(log.read_text(errors="replace"))
        m = re.search(r"cmsRun rc=(-?\d+)", log.read_text(errors="replace"))
        res["rc"] = int(m.group(1)) if m else None
        rec = {k: v for k, v in p.items() if k not in ("lfns", "image")}
        rec.update(res)
        rec["files"] = p.get("lfns")
        rec["mode"] = "condor"
        rec["done"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if res.get("xsec_gen") is None:
            failed += 1
            if results.get(key, {}).get("xsec_gen") is not None:
                continue          # keep an earlier good measurement
        else:
            done += 1
        results[key] = rec
    results_path.write_text(json.dumps(results, indent=1) + "\n")
    print(f"collected {done} cross section(s), {failed} failed log(s), {missing} sample(s) without a log yet -> {results_path}")
    return 0 if not failed and not missing else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--work", default="~/work/skims_v2/_genxsec")
    p.add_argument("--era", nargs="+")
    p.add_argument("--dataset", nargs="+", help="substring filter on the sample key")
    p.add_argument("--skip-signal", action="store_true")
    p.add_argument("--downloads", type=int, default=1)
    p.add_argument("--cmsrun", type=int, default=2)
    p.add_argument("--vm", default="combine")
    p.add_argument("--keep-inputs", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("prepare-condor", help="find the MiniAOD files, inspect the content, write Condor jobs (LPC)")
    p.add_argument("--work", default="~/work/skims_v2/_genxsec")
    p.add_argument("--era", nargs="+", required=True)
    p.add_argument("--jobs-dir", required=True)
    p.add_argument("--dataset", nargs="+", help="substring filter on the sample key")
    p.add_argument("--skip-signal", action="store_true")
    p.add_argument("--force", action="store_true", help="measure samples that already have a result")
    p.add_argument("--per-job", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-inspect", action="store_true", help="skip the NanoAOD content inspection")
    p.add_argument("--proxy", default="/uscms/homes/b/bjackson/x509up_u24705", help="proxy path on the LPC")
    p.set_defaults(func=cmd_prepare_condor)
    p = sub.add_parser("collect", help="parse the logs of the Condor jobs into genxsec_results.json")
    p.add_argument("--work", default="~/work/skims_v2/_genxsec")
    p.add_argument("--jobs-dir", required=True)
    p.set_defaults(func=cmd_collect)
    p = sub.add_parser("review")
    p.add_argument("--work", default="~/work/skims_v2/_genxsec")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("apply", help="write the decided cross sections into the configs (dry run unless --write)")
    p.add_argument("--work", default="~/work/skims_v2/_genxsec")
    p.add_argument("--era", nargs="+", help="only configs of these eras (default: all)")
    p.add_argument("--any-era", action="store_true",
                   help="apply a result to every era with the same primary dataset and energy, not only the eras it was run for")
    p.add_argument("--dataset", nargs="+", help="substring filter on the sample key")
    p.add_argument("--normalize", choices=("reference", "generator"), default="reference",
                   help="scale to the inclusive reference where one exists (config convention for ttbar/tW) or keep the generator value")
    p.add_argument("--write", action="store_true", help="modify the config JSONs (default: dry run)")
    p.set_defaults(func=cmd_apply)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
