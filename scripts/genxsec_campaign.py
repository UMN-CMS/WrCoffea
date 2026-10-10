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
    ("Run3Winter24MiniAOD", "CMSSW_14_0_9", "el8"),        # 133X; only used for the inclusive LO W+jets reference sample
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
BR = {"2L2Nu": BR_W_LNU ** 2, "LNu2Q": 2 * BR_W_LNU * (1 - BR_W_LNU), "4Q": (1 - BR_W_LNU) ** 2, "LNu": BR_W_LNU, "2Q": 1 - BR_W_LNU,
      "NoFullHad": 1 - (1 - BR_W_LNU) ** 2}   # 2L2Nu + LNu2Q, e.g. UL ST_tW_*_NoFullyHadronicDecays

# Inclusive reference cross sections [pb] per centre-of-mass energy.  Two uses:
# (1) evidence for whether a generator header is inclusive (POWHEG hvq / ST_wtch print the undecayed
#     total even when the decays are restricted at LHE level) or already contains the decay
#     (MG5_aMC and POWHEG ST_tch_4f), by comparing the header with ref and with ref x BR;
# (2) the normalisation the config is scaled to with ``apply --normalize reference`` (k-factor = ref / header).
REFERENCES = {
    "ttbar": {"13p6TeV": (923.6, "NNLO+NNLL, TOP twiki TtbarNNLO, mt = 172.5 GeV"),
              "13TeV": (833.9, "NNLO+NNLL, TOP twiki TtbarNNLO, mt = 172.5 GeV")},
    "tW": {"13p6TeV": (43.95, "approx. NNLO = NLO+NNLL soft-gluon resummation, per charge (87.9 pb for both), TOP twiki SingleTopNNLORef"),
           "13TeV": (35.85, "NLO+NNLL per charge (71.7 pb for both), TOP twiki SingleTopRefXsec")},
    # t-channel NNLO: MCFM, 5FS, PDF4LHC21, mt = 172.5 GeV (Campbell, Neumann, Sullivan, JHEP 02 (2021) 040,
    # arXiv:2012.01574), as tabulated on the TOP twiki SingleTopNNLORef.
    "t-channel_top": {"13p6TeV": (145.0, "NNLO 5FS, TOP twiki SingleTopNNLORef"),
                      "13TeV": (134.2, "NNLO 5FS, TOP twiki SingleTopNNLORef (arXiv:2012.01574)")},
    "t-channel_antitop": {"13p6TeV": (87.2, "NNLO 5FS, TOP twiki SingleTopNNLORef"),
                          "13TeV": (80.0, "NNLO 5FS, TOP twiki SingleTopNNLORef (arXiv:2012.01574)")},
    # Inclusive diboson production (all decays, as in the pure-Pythia samples), from the NNLO predictions
    # quoted by the first 13.6 TeV measurements and by the CMS 13 TeV ones (each agrees with its measurement).
    "WW": {"13p6TeV": (125.8, "NNLO QCD x NLO EW, MATRIX v2.1.0, incl. gg->WW x 1.4, NNPDF3.0 "
                               "(CMS, arXiv:2406.05101; measured 125.7 +- 5.6 pb)"),
           "13TeV": (118.8, "NNLO QCD for qq (arXiv:1408.5243) incl. gg->WW x 1.4, NLO EW -1.2 % on qq "
                             "(CMS, arXiv:2009.00119; measured 117.6 +- 6.8 pb)")},
    "WZ": {"13p6TeV": (54.7, "NNLO QCD x NLO EW, MATRIX, W+Z + W-Z with 60 < m(Z) < 120 GeV "
                              "(CMS, arXiv:2412.02477; NLO QCD 50.4, NNLO QCD 55.0; measured 55.2 +- 1.9 pb)"),
           "13TeV": (50.7, "NNLO QCD x NLO EW, MATRIX v2.0.0, W+Z + W-Z with 60 < m(Z) < 120 GeV, NNPDF3.1 luxqed "
                            "(CMS, arXiv:2110.11231 Table 6; NNLO QCD 51.2; measured 50.6 +- 2.1 pb)")},
    "ZZ": {"13p6TeV": (16.7, "NNLO QCD + NLO EW for qq, NLO QCD for gg->ZZ, MATRIX, 66 < m(Z) < 116 GeV, NNPDF3.0 "
                              "(ATLAS, arXiv:2311.09715; measured 16.8 +- 1.1 pb)"),
           "13TeV": (16.5, "NNLO QCD x NLO EW for qq, NLO QCD for gg->ZZ, MATRIX v2.0.0, 60 < m(Z) < 120 GeV, "
                            "NNPDF3.1 luxqed (CMS, arXiv:2009.01186 Sec. 8; measured 17.4 +- 0.8 pb)")},
    "s-channel_top": {"13TeV": (6.35, "NLO, TOP twiki SingleTopRefXsec")},
    "s-channel_antitop": {"13TeV": (3.97, "NLO, TOP twiki SingleTopRefXsec")},
}
# Inclusive Drell-Yan (Z/gamma* -> ll, m(ll) > 50 GeV, three flavours) at NNLO [pb]: the normalisation of EVERY DY
# sample, LO and NLO (decided 2026-10-10).  A sample is scaled by k = reference / inclusive header of its own generator
# class in its own era (LO MLM, NLO FxFX, or the sum of the POWHEG per-flavour mass bins), so all DY variants share
# one total and k stays flat; the Z-pT reweight and the jet-pT reshape correct the shapes.  The 13.6 TeV value is the
# MATRIX NNLO QCD row (not the NNLO QCD x NLO EW one, 2091.7 pb); the 13 TeV value is the FEWZ NNLO number every CMS
# Run 2 analysis used.
DY_REFERENCES = {
    "13p6TeV": (6340.2, "NNLO QCD, MATRIX v2.1.0, pp -> e+e- with m(ll) > 50 GeV: 2113.4 pb per flavour "
                        "(+0.7/-1.2 % scale, 1.0 % PDF), x3 flavours = 6340.2 pb; CMS twiki MATRIXCrossSectionsat13p6TeV "
                        "r12 (2023-10-25, L. Markovic, R. Covarelli), saved as docs/xsec/CMS_MATRIXCrossSectionsat13p6TeV.pdf"),
    "13TeV": (6077.22, "NNLO, FEWZ 3.1.b2, Z/gamma* -> ll with m(ll) > 50 GeV: 2025.74 pb per flavour, 6077.22 pb for "
                       "three flavours; CMS twiki StandardModelCrossSectionsat13TeV r28 (2023-08-03), saved as "
                       "docs/xsec/CMS_StandardModelCrossSectionsat13TeV.pdf"),
}

# LO madgraphMLM W+jets: a flat k = 1.21 (inclusive NNLO / LO), the Run 2 HNWR convention for the HT-binned samples
# (AN-19-083 Tables 6-8; SKFlat CommonSampleInfo: 2016 LO x 1.21), applied in every era (decided 2026-10-08).
# Source: the k-factor column of the SummaryTable1G25ns twiki (docs/xsec/CMS_SummaryTable1G25ns.pdf): 1.21 on every
# HT-binned W+jets row, 1.23 on the HT-binned DY rows.
WJETS_LO_KFACTOR = (1.21, "LO x k(NNLO) = 1.21, Run 2 convention for LO MLM W+jets (inclusive NNLO / LO; "
                          "AN-19-083 Tables 6-8)")      # superseded 2026-10-10 by W_REFERENCES (kept for the record)

# Inclusive W -> l nu at NNLO, three flavours [pb]: the normalisation of the LO MLM W+jets samples (decided 2026-10-10,
# replacing the flat Run 2 factor 1.21): k = reference / inclusive LO MLM W+jets header of the same era (the
# `WtoLNu-4Jets` / `WJetsToLNu` inclusive sample, kept in the configs as an optional, unskimmed reference entry), the
# same construction as for DY.  The 1.21 was this ratio with the 2015 LO header (61526.7 / 50690).
W_REFERENCES = {
    "13p6TeV": (63650.1, "NNLO QCD, MATRIX v2.1.0, pp -> e- nubar (9045.8 pb) + e+ nu (12170.9 pb), no cuts, per flavour "
                         "21216.7 pb (+1.2/-1.3 % scale, 0.8 % PDF), x3 flavours = 63650.1 pb; CMS twiki "
                         "MATRIXCrossSectionsat13p6TeV r12 (2023-10-25), saved as docs/xsec/CMS_MATRIXCrossSectionsat13p6TeV.pdf"),
    "13TeV": (61526.7, "NNLO, FEWZ 3.1, W+ 11811.4 + W- 8677.3 = 20508.9 pb per flavour, x3 flavours = 61526.7 pb; CMS twiki "
                       "StandardModelCrossSectionsat13TeV r28 (2023-08-03), saved as docs/xsec/CMS_StandardModelCrossSectionsat13TeV.pdf"),
}

# Official WR signal in an era with a signal k-factor payload (config.yaml ``signal_kfactor``): LO x <k>, the Run 2
# convention (AN-19-083 Table 23).  <k> is the payload's per-point, per-channel average of k(m_lN), the same number
# the analyzer divides by in its k(m_lN) / <k> shape weight, so the normalisation is exactly LO x <k>.
OFFICIAL_SIGNAL_RE = re.compile(r"^WRtoN(El|Mu)to(?:El|Mu){2}JJ_MWR(\d+)_N(\d+)_")
REPO = Path(__file__).resolve().parent.parent
# UL campaigns share the 13 TeV signal gridpacks: a point unmeasured in one of them (MiniAOD on tape) takes the
# value measured on the same primary dataset in another.
UL_ERAS = ("RunIISummer20UL16APV", "RunIISummer20UL16", "RunIISummer20UL17", "RunIISummer20UL18")
RUN3_ERAS = ("Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix")
CAMPAIGN_FAMILIES = (UL_ERAS, RUN3_ERAS)      # eras that share one set of signal gridpacks: donors for unmeasured points


def campaign_family(era: str | None) -> tuple[str, ...] | None:
    return next((f for f in CAMPAIGN_FAMILIES if era in f), None)


SIGNAL_NORM = "direct"      # "direct": sigma_NLO / sigma_LO(sample) of the point (payload direct_k, docs/signal_kfactor.md
                            # option 2, decided 2026-10-10); "average": LO x <k> of the shared curve (Run 2)
SIGNAL_PAYLOAD = None       # --signal-payload: a payload used for every era instead of config.yaml's (dry runs before the
                            # config change)


def signal_average_k(era: str | None, primary: str, payload_path: str | None = None) -> tuple[float, str] | None:
    """(k, description) for an official WR signal sample in *era*, or None (no payload / not official signal):
    the payload's <k> of the point and channel (SIGNAL_NORM "average", the Run 2 LO x <k> convention) or its
    direct per-point sigma_NLO / sigma_LO ("direct"; docs/signal_kfactor.md)."""
    m = OFFICIAL_SIGNAL_RE.match(primary)
    if not m or not era:
        return None
    if payload_path is None and SIGNAL_PAYLOAD:
        payload_path = SIGNAL_PAYLOAD
    if payload_path is None:
        import yaml
        paths = (yaml.safe_load((REPO / "wrcoffea" / "config.yaml").read_text()).get("signal_kfactor") or {})
        payload_path = paths.get(era)
        if not payload_path:
            return None
    path = payload_path
    payload = json.loads((REPO / path).read_text() if not Path(path).is_absolute() else Path(path).read_text())
    name = f"WR{m.group(2)}_N{m.group(3)}"
    if SIGNAL_NORM == "direct":
        point = (payload.get("direct_k") or {}).get(name)
        if point is None or "k" not in point:
            raise KeyError(f"{primary}: no direct k for {name} in {path}")
        return float(point["k"]), f"{path} direct_k ({point.get('source', 'direct')})"
    point = payload["averages"].get(name)
    if point is None:
        raise KeyError(f"{primary}: no <k> for {name} in {path}")
    return float(point["ee" if m.group(1) == "El" else "mm"]), path
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
    import re
    m = re.match(r"(ww|wz|zz)_tune", p)          # inclusive dibosons only, not WWW/WWZ/WZZ/ZZZ
    if m:
        return m.group(1).upper(), "both"
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
    if proc in ("ttbar", "tW", "WW", "WZ", "ZZ"):
        return REFERENCES[proc].get(energy)
    if charge == "both":
        top = REFERENCES.get(f"{proc}_top", {}).get(energy)
        anti = REFERENCES.get(f"{proc}_antitop", {}).get(energy)
        if top and anti:
            return top[0] + anti[0], f"top + antitop: {top[1]}"
        return None
    return REFERENCES.get(f"{proc}_{charge}", {}).get(energy)


def restricted_channel(content: dict) -> tuple[str | None, float]:
    """The decay channel every event falls into (> 97 %), as a BR key, else None.

    Two-W samples whose fully hadronic decays were removed (> 97 % of the events have at least one
    leptonic W, both other channels populated) return "NoFullHad"."""
    wf = content.get("w_decay_fractions") or {}
    mean = content.get("w_decays_mean", 0.0) or 0.0
    if wf and mean >= 1.5:
        for ch, key in (("2L2Nu", "2_leptonic_W"), ("LNu2Q", "1_leptonic_W"), ("4Q", "0_leptonic_W")):
            if wf.get(key, 0) > 0.97:
                return ch, mean
        if wf.get("1_leptonic_W", 0) + wf.get("2_leptonic_W", 0) > 0.97:
            return "NoFullHad", mean
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
        # Every DY sample, LO and NLO, is scaled to the inclusive NNLO reference of its energy (DY_REFERENCES) by
        # k = reference / inclusive header of its own generator class in its own era (flat k, decided 2026-10-10).
        gen_class = ("madgraphmlm" if "madgraphmlm" in low else "powheg" if "powheg" in low
                     else "amcatnlo" if "amcatnlo" in low else None)
        k = dy_kfactor(results, energy, entry.get("era"), gen_class, present[0] if len(present) == 1 else None)
        if k is None:
            out.update(factor_reason=out["factor_reason"] + "; no NNLO DY reference at this energy or no inclusive "
                                                             "header of this generator class measured in this era: "
                                                             "k-factor left at 1 (CHECK)")
            out["undecided_k"] = True
        else:
            out.update(kfactor=k[0], reference=k[1], reference_source=k[2], evidence=k[3])
        return out
    # --- official WR signal: LO header x <k> where the era has a signal k-factor payload
    avg_k = signal_average_k(entry.get("era"), prim)
    if avg_k is not None:
        out.update(header="includes decay",
                   factor_reason="MG5_aMC generates the full WR -> l N -> l l q q chain; the header is the "
                                 "cross section of the sample",
                   kfactor=avg_k[0], reference=gen * avg_k[0],
                   reference_source=(f"LO x k(NLO): {'direct per-point sigma_NLO / sigma_LO' if SIGNAL_NORM == 'direct' else 'averaged signal k-factor <k>'}"
                                     f" = {avg_k[0]:.5f} (AN-19-083 App. A method; {avg_k[1]})"),
                   evidence=f"<k> {avg_k[0]:.5f}")
        return out
    # --- LO MLM W+jets: the header includes W -> l nu; scaled to the inclusive NNLO W cross section of the energy by
    #     k = reference / inclusive LO MLM header of the era (W_REFERENCES), like DY
    if re.match(r"(wjetstolnu|wtolnu-\djets)", low) and "madgraphmlm" in low:
        out.update(header="includes decay",
                   factor_reason="MG5_aMC generates W -> l nu in the matrix element, so the header includes the "
                                 "branching fraction")
        k = w_kfactor(results, energy, entry.get("era"))
        if k is None:
            out.update(factor_reason=out["factor_reason"] + "; no NNLO W reference at this energy or no inclusive LO "
                                                             "MLM W+jets header measured in this era: k-factor left at 1 "
                                                             "(CHECK)")
            out["undecided_k"] = True
        else:
            out.update(kfactor=k[0], reference=k[1], reference_source=k[2], evidence=k[3])
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


def _dy_powheg_mass_bins(results: dict, energy: str, era: str | None, flavour: str) -> list[tuple[str, float]]:
    """POWHEG DYto2<flavour> mass bins with m(ll) > 50 GeV at *energy* in *era* ("E"/"Mu"); their sum is the
    per-flavour inclusive cross section of that generator (the low-mass 10to50 bin is left out)."""
    return sorted((e["primary"], e["xsec_gen"]) for e in results.values()
                  if e.get("energy") == energy and e.get("era") == era and e.get("xsec_gen") is not None
                  and "powheg" in e["primary"].lower()
                  and re.match(rf"dyto2{flavour.lower()}_(bin-)?mll-", e["primary"].lower())
                  and "10to50" not in e["primary"])


def _w_inclusive_headers(results: dict, energy: str, era: str | None) -> list[tuple[str, float]]:
    """Inclusive LO MLM W+jets headers (``WtoLNu-4Jets_Tune...`` / ``WJetsToLNu_Tune...``, not HT-, jet- or
    m(l nu)-binned) at *energy* in *era*."""
    out = []
    for e in results.values():
        if e.get("energy") != energy or e.get("xsec_gen") is None or e.get("era") != era:
            continue
        low = e["primary"].lower()
        if "madgraphmlm" in low and re.match(r"(wtolnu-\djets|wjetstolnu)_tune", low):
            out.append((e["primary"], e["xsec_gen"]))
    return out


def w_kfactor(results: dict | None, energy: str, era: str | None) -> tuple[float, float, str, str] | None:
    """(k, reference, reference_source, evidence) scaling a LO MLM W+jets sample to the inclusive NNLO W cross
    section of *energy*: k = reference / inclusive LO MLM header of *era*.  None when either is missing."""
    ref = W_REFERENCES.get(energy)
    if not results or ref is None:
        return None
    heads = _w_inclusive_headers(results, energy, era)
    if not heads:
        return None
    incl = sum(x for _, x in heads) / len(heads)
    k = ref[0] / incl
    src = f"LO x k(NNLO): {ref[1]}; k = reference / LO MLM inclusive W+jets header of this era (GenXSec)"
    ev = (f"NNLO / LO MLM k-factor {k:.4f} = {ref[0]} / {incl:.1f} (LO MLM inclusive header "
          + ", ".join(f"{n.split(chr(95))[0]} {x:.5g}" for n, x in heads) + ")")
    return k, ref[0], src, ev


def dy_kfactor(results: dict | None, energy: str, era: str | None, generator: str | None,
               flavour: str | None = None) -> tuple[float, float, str, str] | None:
    """(k, reference, reference_source, evidence) scaling a DY sample of *generator* class ("madgraphmlm",
    "amcatnlo" or "powheg"; *flavour* "e"/"mu" for POWHEG) to the inclusive NNLO reference of *energy*:
    k = reference / inclusive header of the same generator class in *era*.  None when either is missing."""
    ref = DY_REFERENCES.get(energy)
    if not results or ref is None or generator is None:
        return None
    if generator == "powheg":
        fl = {"e": "E", "mu": "Mu"}.get(flavour or "")
        bins = _dy_powheg_mass_bins(results, energy, era, fl) if fl else []
        if not bins:
            return None
        tot = sum(x for _, x in bins)
        per_flavour = round(ref[0] / 3, 4)
        k = per_flavour / tot
        src = (f"NLO x k(NNLO): {ref[1]}; k = per-flavour reference {per_flavour} pb / sum of the POWHEG DYto2{fl} "
               f"mass bins with m(ll) > 50 GeV of this era (GenXSec)")
        ev = (f"NNLO / POWHEG k-factor {k:.4f} = {per_flavour} / {tot:.1f} (sum of {len(bins)} bins: "
              + ", ".join(f"{n.split('_Tune')[0].split('_')[-1]} {x:.5g}" for n, x in bins) + ")")
        return k, per_flavour, src, ev
    heads = _dy_inclusive_headers(results, energy, generator, era)
    if not heads:
        return None
    incl = sum(x for _, x in heads) / len(heads)
    k = ref[0] / incl
    label, cls = ("LO x k(NNLO)", "LO MLM") if generator == "madgraphmlm" else ("NLO x k(NNLO)", "NLO FxFX")
    src = f"{label}: {ref[1]}; k = reference / {cls} inclusive three-flavour header of this era (GenXSec)"
    ev = (f"NNLO / {cls} k-factor {k:.4f} = {ref[0]} / {incl:.1f} ({cls} inclusive header "
          + ", ".join(f"{n.split(chr(95))[0]} {x:.5g}" for n, x in heads) + ")")
    return k, ref[0], src, ev


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
            bins = _dy_powheg_mass_bins(results, energy, era or None, fl)
            fx = [x for n, x in _dy_inclusive_headers(results, energy, "amcatnlo", era or None)
                  if n.startswith(f"DYto2{fl}-") or n.startswith("DYto2L-")]
            if bins and fx:
                tot = sum(x for _, x in bins)
                incl = sum(fx) / len(fx)
                lines.append(f"- {era or energy} DYto2{fl}: sum of {len(bins)} POWHEG mass bins (M > 50 GeV) = {tot:.1f} pb; "
                             f"FxFX MLL-50 header / 3 = {incl / 3:.1f} pb (ratio {tot / (incl / 3):.3f})")
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


# ----------------------------------------------------------------------------
# Perturbative order of the cross section a config entry actually uses ("order" key, right after "xsec")
# ----------------------------------------------------------------------------

# Vocabulary (documented in docs/dataset_configs.md):
#   LO, NLO, NNLO                 the value is a generator or reference cross section at that order
#   NLO+NNLL, NNLO+NNLL           fixed order plus resummation (tW "approximate NNLO"; ttbar)
#   LO x k(NLO), LO x k(NNLO)     a LO generator value scaled by an inclusive NLO/LO or NNLO/LO k-factor
#   NLO x k(NNLO)                 an NLO generator value scaled by an inclusive NNLO/NLO k-factor (DY)
#   unknown                       no value, or a value whose provenance is not established
ORDERS = ("LO", "NLO", "NNLO", "NLO+NNLL", "NNLO+NNLL", "LO x k(NLO)", "LO x k(NNLO)", "NLO x k(NNLO)", "unknown")

# Run 2 UL entries whose committed values predate the GenXSecAnalyzer campaign (no xsec_source): the order of
# each value as documented in docs/configs/run2_ul_dataset_report.md (source column) and AN-19-083 Tables 6-8.
# First match wins; patterns are on the primary dataset name.
UL_ORDER_RULES = [
    (r"^DYJetsToLL_M-50_TuneCP5_13TeV-(madgraphMLM|amcatnloFXFX)", "NNLO"),   # 6077.22 pb, FEWZ NNLO
    (r"^DYJetsToLL_M-10to50_TuneCP5_13TeV-(madgraphMLM|amcatnloFXFX)", "NLO"),  # 18610 / 20490 pb
    (r"^DYJetsToLL_M-50_HT-", "LO"),                                          # XSDB bare LO
    (r"^DY\dJetsToLL_", "LO"),                                                # cmsdb GenXSecAnalyzer, madgraphMLM
    (r"^DYJetsToLL_(LHEFilterPtZ|[012]J)", "NLO"),                            # cmsdb GenXSecAnalyzer, amcatnloFXFX
    (r"^WJetsToLNu_HT-", "LO x k(NNLO)"),                                     # 2016 LO x 1.21 (AN-19-083)
    (r"^TTTo(2L2Nu|SemiLeptonic)_", "NNLO+NNLL"),                             # NNLO+NNLL x BR
    (r"^ST_tW_", "NLO+NNLL"),                                                 # approx. NNLO x BR (AN: 19.20, NNLO)
    (r"^ST_t-channel_", "NLO"),                                               # 136.02 / 80.95, 5FS NLO (Hathor)
    (r"^ST_s-channel_", "NLO"),                                               # 3.36, NLO
    (r"^WW_TuneCP5_13TeV-pythia8", "NNLO"),                                   # 118.7 pb, NNLO
    (r"^(WZ|ZZ)_TuneCP5_13TeV-pythia8", "NLO"),                               # 47.13 / 16.523 pb, MCFM NLO
    (r"^(WWW|WWZ|WZZ|ZZZ)_", "NLO"),                                          # amcatnlo XSDB values
    (r"^tt[ZW]Jets_", "LO"),                                                  # XSDB LO, madgraphMLM
    (r"^WRtoN", "LO x k(NLO)"),                                               # Run 2 AN signal: LO x <k>
]


def generator_order(primary: str) -> str:
    """Order of a generator cross section, from the generator named in the dataset."""
    p = primary.lower()
    if "amcatnlo" in p or "fxfx" in p or "powheg" in p:
        return "NLO"
    if "madgraph" in p or "pythia" in p:
        return "LO"
    return "unknown"


def order_from_source(primary: str, src: dict) -> str:
    """Order of the value written by `apply`, from its xsec_source record."""
    if src.get("status") == "not measured":
        return generator_order(primary)      # the kept earlier value is the generator value
    ref, k = src.get("reference"), src.get("kfactor") or 1.0
    if src.get("normalize") == "reference" and ref and abs(k - 1.0) > 1e-9:
        rs = (src.get("reference_source") or "").lower()
        if rs.startswith("lo x k(nnlo)"):     # LO MLM W+jets x 1.21; LO MLM DY x NNLO / LO inclusive
            return "LO x k(NNLO)"
        if rs.startswith("nlo x k(nnlo)"):    # NLO DY (FxFX, POWHEG) x NNLO / NLO inclusive
            return "NLO x k(NNLO)"
        if rs.startswith("lo x k(nlo)"):      # official WR signal x the Run 2 averaged k-factor
            return "LO x k(NLO)"
        if rs.startswith("nnlo+nnll"):
            return "NNLO+NNLL"
        if "approx. nnlo" in rs or rs.startswith("nlo+nnll"):
            return "NLO+NNLL"
        if rs.startswith("nnlo"):
            return "NNLO"
        if "nlo fxfx inclusive" in rs:      # LO MLM DY scaled by the NLO FxFX / LO MLM inclusive ratio
            return "LO x k(NLO)"
        if rs.startswith("nlo"):
            return "NLO"
        named = re.search(r"\b(nnlo\+nnll|nlo\+nnll|nnlo|nlo)\b", rs)   # e.g. "top + antitop: NLO, TOP twiki ..."
        if named:
            return named.group(1).upper()
        return "unknown"
    return generator_order(primary)


def xsec_order(das: str, meta: dict) -> str:
    """The "order" key of an MC config entry."""
    if meta.get("xsec") is None and not meta.get("scan"):
        return "unknown"
    primary = das.split("/")[1] if is_das_path(das) else das      # private samples are keyed by name
    src = meta.get("xsec_source")
    if isinstance(src, dict):
        return order_from_source(primary, src)
    import re
    for pattern, order in UL_ORDER_RULES:
        if re.search(pattern, primary):
            return order
    return "unknown"


def with_order(meta: dict, order: str) -> dict:
    """Copy of a config entry with "order" placed right after "xsec"."""
    out = {}
    for key, value in meta.items():
        if key == "order":
            continue
        out[key] = value
        if key == "xsec":
            out["order"] = order
    if "order" not in out:
        out["order"] = order
    return out


def cmd_annotate_order(args) -> int:
    """Write the "order" key of every MC entry of the active configs (dry run unless --write)."""
    import collections
    summary = collections.defaultdict(collections.Counter)
    unknown = collections.Counter()
    changed_files = 0
    for cp in _config_files(args.era):
        text = cp.read_text()
        cfg = json.loads(text)
        changed = False
        for das, meta in list(cfg.items()):
            if not isinstance(meta, dict) or meta.get("datatype") != "mc":
                continue
            order = xsec_order(das, meta)
            summary[meta.get("era")][order] += 1
            if order == "unknown":
                why = "no xsec" if meta.get("xsec") is None else "provenance not established"
                unknown[(meta.get("era"), why, das.split("/")[1].split("_Tune")[0][:40])] += 1
            if meta.get("order") != order or list(meta).index("order") != list(meta).index("xsec") + 1:
                cfg[das] = with_order(meta, order)
                changed = True
        if changed and args.write:
            indent = len(text.splitlines()[1]) - len(text.splitlines()[1].lstrip()) if len(text.splitlines()) > 1 else 1
            cp.write_text(json.dumps(cfg, indent=indent or 1) + "\n")
            changed_files += 1
    for era in sorted(summary):
        print(f"{era:22s} " + ", ".join(f"{o}: {n}" for o, n in sorted(summary[era].items())))
    if unknown:
        print("unknown:")
        for (era, why, name), n in sorted(unknown.items()):
            print(f"  {era:22s} {why:28s} {name} ({n})")
    print(f"{changed_files} config file(s) written" if args.write else "dry run: rerun with --write to modify the configs")
    return 0


def cmd_apply(args) -> int:
    """Write the decided cross sections into the config JSONs (dry run unless --write)."""
    global SIGNAL_NORM, SIGNAL_PAYLOAD
    SIGNAL_NORM = getattr(args, "signal_norm", "direct")
    SIGNAL_PAYLOAD = getattr(args, "signal_payload", None)
    work = Path(args.work).expanduser()
    _, results = load_results(work)
    today = time.strftime("%Y-%m-%d")
    rows = []
    changes: dict[Path, dict] = {}
    skipped = []
    # Official WR signal measured in an era of a campaign family, by primary dataset: donors for the points
    # unmeasured in a sibling era (same gridpacks).
    donors: dict[str, list[dict]] = {}
    for x in results.values():
        if x.get("xsec_gen") is not None and campaign_family(x.get("era")) and OFFICIAL_SIGNAL_RE.match(x.get("primary", "")):
            donors.setdefault(x["primary"], []).append(x)
    for key, e in sorted(results.items()):
        if args.dataset and not any(s in key for s in args.dataset):
            continue
        if args.era and not args.any_era and not set(e.get("eras") or []) & set(args.era):
            continue          # measured for another campaign
        copied_from = None
        fam = campaign_family(e.get("era"))
        if e.get("xsec_gen") is None and fam:
            # nearest sibling era first (UL18 takes UL17, then UL16, then UL16APV)
            cands = sorted((x for x in donors.get(e["primary"], []) if x["era"] != e["era"] and x["era"] in fam),
                           key=lambda x: abs(fam.index(x["era"]) - fam.index(e["era"])))
            if cands:
                copied_from = cands[0]["era"]
                e = {**cands[0], "era": e["era"], "eras": e.get("eras"), "das": e.get("das"), "miniaod": e.get("miniaod")}
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
        if copied_from:
            source["copied_from"] = (f"{copied_from}: the generator value was measured on the {copied_from} MiniAOD of the "
                                     f"same primary dataset (the {e['era']} MiniAOD could not be read; the UL campaigns "
                                     "share the 13 TeV gridpack)")
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
    # Official WR signal entries that no measurement reaches (private productions whose LO comes from the LHE
    # headers, points unmeasurable in every sibling era): the config's own LO value times the point's k.
    for cp in _config_files(args.era):
        cfg = json.loads(cp.read_text())
        upd = changes.setdefault(cp, {})
        for das, meta in cfg.items():
            if not isinstance(meta, dict) or das in upd or meta.get("datatype") != "mc":
                continue
            primary = das.split("/")[1] if is_das_path(das) else das
            if not OFFICIAL_SIGNAL_RE.match(primary) or (args.dataset and not any(x in primary for x in args.dataset)):
                continue
            if args.era and meta.get("era") not in args.era:
                continue
            src = meta.get("xsec_source") if isinstance(meta.get("xsec_source"), dict) else {}
            if src.get("tool") == "GenXSecAnalyzer" and src.get("status") != "not measured" and src.get("generator"):
                continue        # measured: handled above (or filtered out on purpose)
            base = how = None
            if src.get("header") == "LHE" and src.get("generator"):
                base, how = float(src["generator"]), "LHE header of the private production (LO)"
            elif src.get("status") == "not measured" and src.get("kept_value"):
                base, how = float(src["kept_value"]), "LO value kept when the MiniAOD could not be read in any era"
            elif meta.get("order") == "LO" and meta.get("xsec"):
                base, how = float(meta["xsec"]), "config LO value"
            if base is None:
                skipped.append((primary, "official signal without a LO value to scale"))
                continue
            try:
                kk = signal_average_k(meta.get("era"), primary)
            except KeyError as err:
                skipped.append((primary, str(err)))
                continue
            if kk is None:
                continue        # no payload for this era: the entry stays as it is
            new = base * kk[0]
            unc = src.get("generator_unc")
            upd[das] = {"xsec": _sig(new), "total_uncertainty": _sig(float(unc) * kk[0], 4) if unc else meta.get("total_uncertainty"),
                        "xsec_source": {"generator": base, "generator_unc": unc, "header": src.get("header", "LO"), "factor": 1.0,
                                        "factor_reason": how, "kfactor": _sig(kk[0], 6), "reference": _sig(new),
                                        "reference_source": f"LO x k(NLO): direct per-point sigma_NLO / sigma_LO = {kk[0]:.5f} "
                                                            f"(AN-19-083 App. A method; {kk[1]})",
                                        "normalize": args.normalize, "tool": src.get("tool", "config"), "date": today,
                                        "base_source": how}}
            rows.append((primary, meta.get("era"), meta.get("xsec"), _sig(new), 1.0, kk[0], cp.name))
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
            cfg[das] = with_order(cfg[das], xsec_order(das, cfg[das]))
        indent = len(text.splitlines()[1]) - len(text.splitlines()[1].lstrip()) if len(text.splitlines()) > 1 else 1
        cp.write_text(json.dumps(cfg, indent=indent or 1) + "\n")
        print(f"updated {cp.relative_to(REPO)}: {len(upd)} entries")
    print("remember: existing plans keep the old metadata until `bin/skim.py refresh-metadata --plan-dir ...` is run")
    return 0


# ----------------------------------------------------------------------------
# Condor mode: cmsRun streams the MiniAOD files at the LPC instead of downloading them here
# ----------------------------------------------------------------------------

SCRAM_ARCH = {"CMSSW_13_0_13": "el8_amd64_gcc11", "CMSSW_14_0_9": "el8_amd64_gcc12", "CMSSW_15_0_21": "el9_amd64_gcc12", "CMSSW_10_6_30": "slc7_amd64_gcc700"}
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
    p = sub.add_parser("annotate-order", help='write the "order" key (LO, NLO, ...) of every MC config entry (dry run unless --write)')
    p.add_argument("--era", nargs="+", default=None, help="only configs of these eras (default: all)")
    p.add_argument("--write", action="store_true", help="modify the config JSONs (default: dry run)")
    p.set_defaults(func=cmd_annotate_order)
    p = sub.add_parser("apply", help="write the decided cross sections into the configs (dry run unless --write)")
    p.add_argument("--work", default="~/work/skims_v2/_genxsec")
    p.add_argument("--era", nargs="+", help="only configs of these eras (default: all)")
    p.add_argument("--any-era", action="store_true",
                   help="apply a result to every era with the same primary dataset and energy, not only the eras it was run for")
    p.add_argument("--dataset", nargs="+", help="substring filter on the sample key")
    p.add_argument("--signal-payload", default=None,
                   help="signal k-factor payload JSON to use for every era instead of config.yaml's (dry runs before the config change)")
    p.add_argument("--signal-norm", choices=("direct", "average"), default="direct",
                   help="official WR signal: the point's direct NLO/LO (payload direct_k, default) or LO x <k> of the shared curve (Run 2)")
    p.add_argument("--normalize", choices=("reference", "generator"), default="reference",
                   help="scale to the inclusive reference where one exists (config convention for ttbar/tW) or keep the generator value")
    p.add_argument("--write", action="store_true", help="modify the config JSONs (default: dry run)")
    p.set_defaults(func=cmd_apply)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
