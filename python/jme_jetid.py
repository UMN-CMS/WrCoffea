#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# jme_jetid.py — JetID via correctionlib JSON (Run 3, Rereco2022CDE)
#
# - Targets AK4 PUPPI TightLeptonVeto by default.
# - Avoids zero-filling for integer inputs; synthesizes multiplicities from available Nano fields.
# - Designed to match NanoAOD Jet.jetId TLV bit for Run3Summer22 / Run3 Rereco2022CDE.
#
# Usage (example):
#   jid = JetIDManager().for_era("Run3Summer22")
#   mask = jid.ak4_tightlepveto(events.Jet)   # boolean mask per jet

import os
import numpy as np
import awkward as ak
from correctionlib import CorrectionSet


# ---------------------------
# small helpers (no truthiness on arrays)
# ---------------------------
def _unflatten_like(flat_np, like_awk):
    """Unflatten a flat numpy array back to jagged like 'like_awk' using ak.num counts."""
    counts = ak.num(like_awk)  # jets per event
    return ak.unflatten(flat_np, counts)

def _zeros_like_int(jets, dtype=np.int32, scalar=0):
    """Broadcast a scalar integer to the jagged shape of jets.pt."""
    return ak.zeros_like(jets.pt, dtype=dtype) + dtype(scalar)

def _getI_like(jets, name, default=None):
    """Return integer array for jets.<name> if present, else default (broadcast)."""
    if hasattr(jets, name):
        return ak.values_astype(getattr(jets, name), np.int32)
    if default is None:
        # Caller should provide a default for critical vars
        raise AttributeError(f"Missing required Jet field '{name}' and no default specified.")
    return _zeros_like_int(jets, np.int32, default)

def _getF_like(jets, name, default=None):
    """Return float array for jets.<name> if present, else default (broadcast)."""
    if hasattr(jets, name):
        return ak.values_astype(getattr(jets, name), np.float32)
    if default is None:
        raise AttributeError(f"Missing required Jet field '{name}' and no default specified.")
    return ak.zeros_like(jets.pt, dtype=np.float32) + np.float32(default)

def _synth_mults_from_available(jets):
    """
    Build (chMultiplicity, neMultiplicity, multiplicity) from fields present in your Nano:
      - You have: nConstituents, nElectrons, nMuons
      - You do NOT have: chHadMult, neutralHadronMultiplicity, photonMultiplicity
    Strategy:
      - multiplicity := nConstituents (total)
      - charged lower bound := nElectrons + nMuons
      - neutral := multiplicity - charged (never negative)
    """
    mult = _getI_like(jets, "nConstituents", default=0)

    n_el = _getI_like(jets, "nElectrons", default=0)
    n_mu = _getI_like(jets, "nMuons", default=0)

    chMult = n_el + n_mu
    # guard against accidental over-count (shouldn't happen, but safe)
    chMult = ak.where(chMult > mult, mult, chMult)
    neMult = mult - chMult

    # Cast to int32 explicitly
    chMult = ak.values_astype(chMult, np.int32)
    neMult = ak.values_astype(neMult, np.int32)
    mult   = ak.values_astype(mult,   np.int32)
    return chMult, neMult, mult


class JetIDProviderPinned:
    """
    Wraps a single CorrectionSet payload and evaluates JetID masks on Awkward jet collections.
    Default keys assume PUPPI JetID TightLeptonVeto for AK4 and Tight for AK8.
    """

    def __init__(self, path, ak4_key="AK4PUPPI_TightLeptonVeto", ak8_key="AK8PUPPI_Tight"):
        self.path = path
        self.cset = CorrectionSet.from_file(path)
        self.key_ak4 = ak4_key
        self.key_ak8 = ak8_key

        # Aliases for JSON variable names -> NanoAOD branches (ordered by preference).
        # Keep this minimal and aligned with what you actually have.
        self.alias = {
            # kinematics
            "pt":      ["pt"],
            "eta":     ["eta"],
            "abseta":  ["eta"],   # abs() handled in code

            # energy fractions (you have these)
            "neHEF":   ["neHEF", "nhEF"],
            "neEmEF":  ["neEmEF"],
            "chHEF":   ["chHEF"],
            "chEmEF":  ["chEmEF"],
            "muEF":    ["muEF"],

            # totals / counts (you have these)
            "nConstituents": ["nConstituents"],
            "nElectrons":    ["nElectrons"],
            "nMuons":        ["nMuons"],
        }

    # ---------------------------
    # core evaluator
    # ---------------------------
    def _eval(self, jets, corr_key: str, wp: str):
        corr = self.cset[corr_key]

        # discover inputs and whether a string WP is expected first
        raw_inputs = list(getattr(corr, "inputs", []))
        input_names = []
        expects_string_wp = False
        for ri in raw_inputs:
            if hasattr(ri, "type"):  # correctionlib.Variable
                if str(ri.type).lower() == "string":
                    expects_string_wp = True
                    continue
                input_names.append(ri.name)
            else:
                name = str(ri)
                if name.lower() in ("wp", "working_point"):
                    expects_string_wp = True
                    continue
                input_names.append(name)

        # precompute synthesized multiplicities once
        chMult_syn, neMult_syn, mult_syn = _synth_mults_from_available(jets)

        cols = []
        for name in input_names:
            lname = name.lower()

            # Handle special forms first
            if lname in ("abseta", "abs_eta", "abs(eta)"):
                cols.append(ak.abs(_getF_like(jets, "eta", default=0.0)))
                continue
            if lname == "eta":
                cols.append(_getF_like(jets, "eta", default=0.0))
                continue

            # Integer multiplicities expected by the JSON
            if lname == "chmultiplicity":
                cols.append(chMult_syn); continue
            if lname == "nemultiplicity":
                cols.append(neMult_syn); continue
            if lname == "multiplicity":
                cols.append(mult_syn); continue

            # Fractions and any remaining numeric inputs via alias
            aliases = self.alias.get(name, [name])
            arr = None
            for nm in aliases:
                if hasattr(jets, nm):
                    arr = getattr(jets, nm)
                    break
            if arr is None:
                # For critical inputs we prefer to raise rather than silently zero-fill
                raise AttributeError(f"[JetID] Missing required input '{name}' (aliases: {aliases})")
            # Ensure float for EF-like variables, int where appropriate; here we default to float
            # (correctionlib will accept either for EF thresholds)
            if lname.endswith("ef") or lname in ("pt", "eta"):
                arr = ak.values_astype(arr, np.float32)
            cols.append(arr)

        # flatten, evaluate, unflatten back to jagged, cast to bool
        flats = [ak.to_numpy(ak.flatten(c)) for c in cols]
        out = corr.evaluate(wp, *flats) if expects_string_wp else corr.evaluate(*flats)
        out = np.asarray(out, dtype=np.int8)
        mask = _unflatten_like(out, jets)
        return ak.values_astype(mask, np.bool_)

    # convenience wrappers
    def ak4_tightlepveto(self, jets):
        # WP string is ignored if JSON doesn't expect it (handled in _eval)
        return self._eval(jets, self.key_ak4, "TightLeptonVeto")

    def ak8_tight(self, fatjets):
        return self._eval(fatjets, self.key_ak8, "Tight")


class JetIDManager:
    """Per-era cached providers with pinned keys & paths. Update paths to your repo layout."""

    ERA_CFG = {
        # Run3Summer22 (Rereco2022CDE payload path)
        "Run3Summer22": {
            "path": "data/jsonpog/JME/Run3/Run3Summer22/jetid.json.gz",
            "ak4_key": "AK4PUPPI_TightLeptonVeto",
            "ak8_key": "AK8PUPPI_Tight",
        },
        # EE-era (if needed elsewhere)
        "Run3Summer22EE": {
            "path": "data/jsonpog/JME/Run3/Run3Summer22EE/jetid.json.gz",
            "ak4_key": "AK4PUPPI_TightLeptonVeto",
            "ak8_key": "AK8PUPPI_Tight",
        },
        # 2024 campaign (if/when you use it)
        "RunIII2024Summer24": {
            "path": "data/jsonpog/JME/Run3/RunIII2024Summer24/jetid.json.gz",
            "ak4_key": "AK4PUPPI_TightLeptonVeto",
            "ak8_key": "AK8PUPPI_Tight",
        },
    }

    def __init__(self):
        self._cache = {}

    def for_era(self, era: str) -> JetIDProviderPinned:
        cfg = self.ERA_CFG.get(era)
        if cfg is None:
            raise KeyError(f"[JetID] Unknown era '{era}'. Add it to JetIDManager.ERA_CFG.")
        if era not in self._cache:
            path = cfg["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"[JetID] Missing payload for {era}: {path}")
            prov = JetIDProviderPinned(path, ak4_key=cfg["ak4_key"], ak8_key=cfg["ak8_key"])
            self._cache[era] = prov
        return self._cache[era]


# ---------------------------
# Optional: tiny self-test helper (run only if executed directly)
# ---------------------------
if __name__ == "__main__":
    import sys
    print("This module provides JetID via correctionlib. Import it and call JetIDManager().for_era(...).")
    print("Example:")
    print("  jid = JetIDManager().for_era('Run3Summer22')")
    print("  mask = jid.ak4_tightlepveto(events.Jet)")
