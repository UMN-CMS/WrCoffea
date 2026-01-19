"""WR Coffea analysis processor.

This module implements the Coffea `ProcessorABC` for the WR→Nℓ→ℓℓjj analysis.

High-level flow per chunk:
    1) Apply lumi mask for data (golden JSON).
    2) Build tight/loose lepton collections + AK4/AK8 jet collections.
    3) Build resolved and (optionally) boosted PackedSelections.
    4) Build nominal weights (+ optional systematic variations).
    5) Fill histograms and cutflows.

Output conventions:
    - Canonical histogram naming: output dict key == numeric axis name == ROOT stem.
    - All physics histograms carry categorical axes: (process, region, syst).
    - `fill_cutflows()` writes a nested `output["cutflow"]` structure with both
        weighted and unweighted one-bin hists plus PackedSelection cutflow hists.

Notes for distributed execution (Dask/Condor):
    - correctionlib payloads are cached per worker process.
    - noisy but expected fallbacks (e.g. missing JetID inputs) are logged once per
        worker process via `_WARN_ONCE`.
"""

from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
import awkward as ak
import hist
import numpy as np
import os
import logging
from coffea.nanoevents.methods import vector
from typing import Callable

from python.analysis_config import JME_JSONS, LUMI_JSONS, LUMI_UNC, LUMIS

ak.behavior.update(vector.behavior)
logger = logging.getLogger(__name__)

# Cache correctionlib payloads per worker process (avoid re-reading JSON every chunk).
_CORRECTIONSET_CACHE = {}

# Warn-once cache (per worker process) to avoid log spam.
_WARN_ONCE: set[str] = set()


# --- Selection name constants (single source of truth for string keys) ---------
#
# These constants are used for:
#   - `PackedSelection.add()` names
#   - region definitions (`resolved_regions` / `boosted_regions`)
#   - cutflow bookkeeping in `fill_cutflows()`
SEL_MIN_TWO_AK4_JETS_PTETA = "min_two_ak4_jets_pteta"
SEL_MIN_TWO_AK4_JETS_ID = "min_two_ak4_jets_id"

SEL_TWO_PTETA_ELECTRONS = "two_pteta_electrons"
SEL_TWO_PTETA_MUONS = "two_pteta_muons"
SEL_TWO_PTETA_EM = "two_pteta_em"

SEL_TWO_ID_ELECTRONS = "two_id_electrons"
SEL_TWO_ID_MUONS = "two_id_muons"
SEL_TWO_ID_EM = "two_id_em"

SEL_E_TRIGGER = "e_trigger"
SEL_MU_TRIGGER = "mu_trigger"
SEL_EMU_TRIGGER = "emu_trigger"

SEL_DR_ALL_PAIRS_GT0P4 = "dr_all_pairs_gt0p4"
SEL_MLL_GT200 = "mll_gt200"
SEL_MLLJJ_GT800 = "mlljj_gt800"
SEL_MLL_GT400 = "mll_gt400"

# Resolved region selection keys
SEL_TWO_TIGHT_ELECTRONS = "two_tight_electrons"
SEL_TWO_TIGHT_MUONS = "two_tight_muons"
SEL_TWO_TIGHT_EM = "two_tight_em"
SEL_LEAD_TIGHT_PT60 = "lead_tight_lepton_pt60"
SEL_SUBLEAD_TIGHT_PT53 = "sublead_tight_pt53"
SEL_MIN_TWO_AK4_JETS = "min_two_ak4_jets"
SEL_60_MLL_150 = "60_mll_150"

# Boosted region selection keys
SEL_BOOSTEDTAG = "boostedtag"
SEL_LEAD_TIGHT_PT60_BOOSTED = "leadTightwithPt60"
SEL_DYCR_MASK = "DYCR_mask"
SEL_ATLEAST1AK8_DPHI_GT2 = "Atleast1AK8Jets & dPhi(J,tightLept)>2"
SEL_AK8JETS_WITH_LSF = "AK8JetswithLSF"
SEL_MUMU_DYCR = "mumu-dy_cr"
SEL_EE_DYCR = "ee-dy_cr"
SEL_MUMU_SR = "mumu_sr"
SEL_EE_SR = "ee_sr"
SEL_EMU_CR = "emu-cr"
SEL_MUE_CR = "mue-cr"


# --- Histogram schema (single source of truth) ---------------------------------
#
# Canonical naming choice:
#   - Histogram key in the output dict == numeric axis name == ROOT stem
#
# Each spec is: (name, bins, label, getter)
#   - For resolved: getter(tight_leptons, ak4_jets) -> values
#   - For boosted:  getter(tight_lepton, ak8_jet, loose_lepton) -> values

ResolvedGetter = Callable[[ak.Array, ak.Array], ak.Array]
BoostedGetter = Callable[[ak.Array, ak.Array, ak.Array], ak.Array]


RESOLVED_HIST_SPECS: list[tuple[str, tuple[int, float, float], str, ResolvedGetter]] = [
    ("pt_leading_lepton",           (200,   0, 2000), r"$p_{T}$ of the leading lepton [GeV]",           lambda L, J: L[:, 0].pt),
    ("eta_leading_lepton",          (60,   -3,    3), r"$\\eta$ of the leading lepton",                lambda L, J: L[:, 0].eta),
    ("phi_leading_lepton",          (80,   -4,    4), r"$\\phi$ of the leading lepton",                lambda L, J: L[:, 0].phi),
    ("pt_subleading_lepton",        (200,   0, 2000), r"$p_{T}$ of the subleading lepton [GeV]",        lambda L, J: L[:, 1].pt),
    ("eta_subleading_lepton",       (60,   -3,    3), r"$\\eta$ of the subleading lepton",             lambda L, J: L[:, 1].eta),
    ("phi_subleading_lepton",       (80,   -4,    4), r"$\\phi$ of the subleading lepton",             lambda L, J: L[:, 1].phi),
    ("pt_leading_jet",              (200,   0, 2000), r"$p_{T}$ of the leading jet [GeV]",              lambda L, J: J[:, 0].pt),
    ("eta_leading_jet",             (60,   -3,    3), r"$\\eta$ of the leading jet",                   lambda L, J: J[:, 0].eta),
    ("phi_leading_jet",             (80,   -4,    4), r"$\\phi$ of the leading jet",                   lambda L, J: J[:, 0].phi),
    ("pt_subleading_jet",           (200,   0, 2000), r"$p_{T}$ of the subleading jet [GeV]",           lambda L, J: J[:, 1].pt),
    ("eta_subleading_jet",          (60,   -3,    3), r"$\\eta$ of the subleading jet",                lambda L, J: J[:, 1].eta),
    ("phi_subleading_jet",          (80,   -4,    4), r"$\\phi$ of the subleading jet",                lambda L, J: J[:, 1].phi),
    ("mass_dilepton",               (5000,  0, 5000), r"$m_{\\ell\\ell}$ [GeV]",                     lambda L, J: (L[:, 0] + L[:, 1]).mass),
    ("pt_dilepton",                 (200,   0, 2000), r"$p_{T,\\ell\\ell}$ [GeV]",                   lambda L, J: (L[:, 0] + L[:, 1]).pt),
    ("mass_dijet",                  (500,   0, 5000), r"$m_{jj}$ [GeV]",                               lambda L, J: (J[:, 0] + J[:, 1]).mass),
    ("pt_dijet",                    (500,   0, 5000), r"$p_{T,jj}$ [GeV]",                             lambda L, J: (J[:, 0] + J[:, 1]).pt),
    ("mass_threeobject_leadlep",    (800,   0, 8000), r"$m_{\\ell jj}$ [GeV]",                        lambda L, J: (L[:, 0] + J[:, 0] + J[:, 1]).mass),
    ("pt_threeobject_leadlep",      (800,   0, 8000), r"$p_{T,\\ell jj}$ [GeV]",                      lambda L, J: (L[:, 0] + J[:, 0] + J[:, 1]).pt),
    ("mass_threeobject_subleadlep", (800,   0, 8000), r"$m_{\\ell jj}$ [GeV]",                        lambda L, J: (L[:, 1] + J[:, 0] + J[:, 1]).mass),
    ("pt_threeobject_subleadlep",   (800,   0, 8000), r"$p_{T,\\ell jj}$ [GeV]",                      lambda L, J: (L[:, 1] + J[:, 0] + J[:, 1]).pt),
    ("mass_fourobject",             (800,   0, 8000), r"$m_{\\ell\\ell jj}$ [GeV]",                 lambda L, J: (L[:, 0] + L[:, 1] + J[:, 0] + J[:, 1]).mass),
    ("pt_fourobject",               (800,   0, 8000), r"$p_{T,\\ell\\ell jj}$ [GeV]",               lambda L, J: (L[:, 0] + L[:, 1] + J[:, 0] + J[:, 1]).pt),
]


BOOSTED_HIST_SPECS: list[tuple[str, tuple[int, float, float], str, BoostedGetter]] = [
    ("pt_leading_lepton",               (200,   0, 2000), r"$p_{T}$ of the leading lepton [GeV]",  lambda lep, ak8, loose: lep.pt),
    ("eta_leading_lepton",              (60,   -3,    3), r"$\\eta$ of the leading lepton",       lambda lep, ak8, loose: lep.eta),
    ("phi_leading_lepton",              (80,   -4,    4), r"$\\phi$ of the leading lepton",       lambda lep, ak8, loose: lep.phi),
    ("pt_subleading_lepton",            (200,   0, 2000), r"$p_{T}$ of the subleading lepton [GeV]", lambda lep, ak8, loose: loose.pt),
    ("eta_subleading_lepton",           (60,   -3,    3), r"$\\eta$ of the subleading lepton",    lambda lep, ak8, loose: loose.eta),
    ("phi_subleading_lepton",           (80,   -4,    4), r"$\\phi$ of the subleading lepton",    lambda lep, ak8, loose: loose.phi),
    ("pt_leading_AK8Jets",              (200,   0, 2000), r"$p_{T}$ of the leading  AK8Jets [GeV]", lambda lep, ak8, loose: ak8.pt),
    ("eta_leading_AK8Jets",             (60,   -3,    3), r"$\\eta$ of theleading  AK8Jets",       lambda lep, ak8, loose: ak8.eta),
    ("phi_leading_AK8Jets",             (80,   -4,    4), r"$\\phi$ of theleading  AK8Jets",       lambda lep, ak8, loose: ak8.phi),
    ("mass_dilepton",                   (5000,  0, 5000), r"$m_{\\ell\\ell}$ [GeV]",             lambda lep, ak8, loose: (lep + loose).mass),
    ("pt_dilepton",                     (200,   0, 2000), r"$p_{T,\\ell\\ell}$ [GeV]",           lambda lep, ak8, loose: (lep + loose).pt),
    ("mass_twoobject",                  (800,   0, 8000), r"$m_{\\ell\\ell jj}$ [GeV]",          lambda lep, ak8, loose: (lep + ak8).mass),
    ("pt_twoobject",                    (800,   0, 8000), r"$p_{T,\\ell\\ell jj}$ [GeV]",        lambda lep, ak8, loose: (lep + ak8).pt),
    ("LSF_leading_AK8Jets",             (200,   0, 1.1),  r"LSF of leading AK8Jets",                lambda lep, ak8, loose: ak8.lsf3),
    ("dPhi_leading_tightlepton_AK8Jet", (80,   -4,    4), r"$d\\phi$ (leading Tight lepton, AK8 Jet)", lambda lep, ak8, loose: abs(ak8.delta_phi(lep))),
]


def _booking_specs() -> dict[str, tuple[tuple[int, float, float], str]]:
    """Return histogram booking metadata keyed by canonical histogram name."""
    specs: dict[str, tuple[tuple[int, float, float], str]] = {}
    for name, bins, label, _ in RESOLVED_HIST_SPECS:
        specs[name] = (bins, label)
    for name, bins, label, _ in BOOSTED_HIST_SPECS:
        specs[name] = (bins, label)
    # Misc always-available hists
    specs.setdefault("count", ((100, 0, 100), r"count"))
    return specs

class WrAnalysis(processor.ProcessorABC):
    """Main Coffea processor for WR analysis.

    Expected `events.metadata` keys (typical):
      - `era`: campaign key (e.g. Run3Summer22EE, RunIII2024Summer24)
      - `datatype`: "mc" or "data"
      - `physics_group`: high-level sample name (e.g. DYJets, EGamma, Signal)
      - `sample`: dataset identifier string
      - `xsec`, `genEventSumw` (MC only)

    Parameters
    - `mass_point`: optional signal mass label (carried through for naming).
    - `enabled_systs`: list of enabled systematic families. Currently supported:
      - `lumi`: add `LumiUp`/`LumiDown` variations as histogram `syst` axis values.
    """
    def __init__(self, mass_point, sf_file=None, enabled_systs=None):
        self._signal_sample = mass_point
        enabled = enabled_systs or []
        self._enabled_systs = {str(s).strip().lower() for s in enabled if str(s).strip()}
        booking = _booking_specs()
        self.make_output = lambda: {
            name: self.create_hist(name, bins, label)
            for name, (bins, label) in booking.items()
        }

    def create_hist(self, name, bins, label):
        """Create a single physics histogram with standard categorical axes."""
        return (
            hist.Hist.new
            .StrCat([], name="process", label="Process", growth=True)
            .StrCat([], name="region",  label="Analysis Region", growth=True)
            .StrCat([], name="syst",    label="Systematic", growth=True)
            .Reg(*bins, name=name, label=label)
            .Weight()
        )

    def apply_lumi_mask(self, events, mc_campaign, is_data):
        """Apply golden-JSON lumi mask for data.

        Returns filtered events. For MC (or if no JSON is configured), returns
        the input events unchanged.
        """
        if not is_data:
            return events

        json_path = os.environ.get("LUMI_JSON") or LUMI_JSONS.get(mc_campaign)
        if json_path:
            try:
                mask = LumiMask(json_path)
                events = events[mask(events.run, events.luminosityBlock)]
                if len(events) == 0:
                    key = f"lumi_mask_empty::{mc_campaign}"
                    if key not in _WARN_ONCE:
                        _WARN_ONCE.add(key)
                        logger.warning(f"All events removed by lumi mask for era '{mc_campaign}'.")
            except OSError as e:
                logger.warning(f"Failed to load lumi JSON '{json_path}': {e}")
        else:
            logger.warning(f"No lumi JSON found for era '{mc_campaign}'. Data left unmasked.")

        return events

    def select_leptons(self, events):
        """Build tight/loose lepton collections and bookkeeping masks.

        Returns
        - `tight_leps`: concatenated (e+μ) tight leptons sorted by pT
        - `loose_leps`: concatenated (e+μ) loose leptons (tight excluded) sorted by pT
        - `masks`: per-object boolean masks used later for cutflows

        Notes
        - Tight = (pT/eta) AND (ID).
        - Loose = (pT/eta) AND (loose-ID), with tight leptons excluded.
        """
        # Split pT/eta (kinematics) and ID components.
        ele_pteta_mask = (events.Electron.pt > 53) & (np.abs(events.Electron.eta) < 2.4)
        mu_pteta_mask  = (events.Muon.pt > 53)     & (np.abs(events.Muon.eta) < 2.4)

        ele_id_mask = events.Electron.cutBased_HEEP
        mu_id_mask  = (events.Muon.highPtId == 2) & (events.Muon.tkRelIso < 0.1)

        # --- TIGHT masks (unchanged behavior: pT/eta AND ID) ---
        electron_tight_mask = ele_pteta_mask & ele_id_mask
        muon_tight_mask     = mu_pteta_mask  & mu_id_mask

        # --- LOOSE base masks (as you had) ---
        electron_loose_base = ele_pteta_mask & (events.Electron.cutBased == 2)
        muon_loose_base     = mu_pteta_mask  & (events.Muon.highPtId == 2)

        # --- LOOSE-only masks (exclude tights) ---
        electron_loose_mask = electron_loose_base & ~electron_tight_mask
        muon_loose_mask     = muon_loose_base     & ~muon_tight_mask

        # --- Filter collections ---
        tight_electrons = events.Electron[electron_tight_mask]
        tight_muons     = events.Muon[muon_tight_mask]
        loose_electrons = events.Electron[electron_loose_mask]
        loose_muons     = events.Muon[muon_loose_mask]

        # --- Add flavor tags ---
        tight_electrons = ak.with_field(tight_electrons, "electron", "flavor")
        tight_muons     = ak.with_field(tight_muons,     "muon",     "flavor")
        loose_electrons = ak.with_field(loose_electrons, "electron", "flavor")
        loose_muons     = ak.with_field(loose_muons,     "muon",     "flavor")

        # --- Combine and sort by pT ---
        tight_leptons = ak.with_name(
            ak.concatenate([tight_electrons, tight_muons], axis=1),
            "PtEtaPhiMCandidate",
        )
        loose_leptons = ak.with_name(
            ak.concatenate([loose_electrons, loose_muons], axis=1),
            "PtEtaPhiMCandidate",
        )

        tight_leps = tight_leptons[ak.argsort(tight_leptons.pt, axis=1, ascending=False)]
        loose_leps = loose_leptons[ak.argsort(loose_leptons.pt, axis=1, ascending=False)]

        masks = {
            "ele_pteta": ele_pteta_mask,
            "mu_pteta": mu_pteta_mask,
            "ele_id": ele_id_mask,
            "mu_id": mu_id_mask,
        }

        return tight_leps, loose_leps, masks

    def select_jets(self, events, era, *, is_signal: bool = False):
        """Select AK4 and AK8 jets and return (collections, per-object masks).

        For `RunIII2024Summer24` backgrounds, AK4 PUPPI TightLeptonVeto is
        evaluated via correctionlib (JME JSON). If the required NanoAOD fields
        are missing, fall back to `Jet.isTightLeptonVeto` and warn once per worker.
        """
        # ---------- AK4 ----------
        ak4_pteta_mask = (events.Jet.pt > 40) & (np.abs(events.Jet.eta) < 2.4)
        if era == "RunIII2024Summer24" and (not is_signal):
            try:
                ak4_id_mask = self.jetid_mask_ak4puppi_tlv(events.Jet, era)
            except AttributeError as e:
                key = f"jetid_fallback::{era}"
                if key not in _WARN_ONCE:
                    _WARN_ONCE.add(key)
                    logger.warning(
                        "RunIII2024Summer24 puppi JetID requested but required jet fields are missing; "
                        "falling back to isTightLeptonVeto for this worker process. "
                        f"(example error: {e})"
                    )
                ak4_id_mask = events.Jet.isTightLeptonVeto
        else:
            ak4_id_mask = events.Jet.isTightLeptonVeto

        ak4_mask = ak4_pteta_mask & ak4_id_mask
        ak4_jets = events.Jet[ak4_mask]

        # ---------- AK8 ----------
        ak8_pteta_mask = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4)
        ak8_extra_sel = (events.FatJet.msoftdrop > 40)
        ak8_mask = ak8_pteta_mask & ak8_extra_sel
        ak8_jets = events.FatJet[ak8_mask]

        masks = {
            "ak4_pteta": ak4_pteta_mask,
            "ak4_id": ak4_id_mask,
            "ak8_pteta": ak8_pteta_mask,
        }

        return ak4_jets, ak8_jets, masks

    def jetid_mask_ak4puppi_tlv(self, jets, era):
        """
        Returns a jagged boolean mask aligned with `jets` for AK4 PUPPI TightLeptonVeto.
        """
        import correctionlib

        json_path = JME_JSONS[era]
        ceval = _CORRECTIONSET_CACHE.get(json_path)
        if ceval is None:
            ceval = correctionlib.CorrectionSet.from_file(json_path)
            _CORRECTIONSET_CACHE[json_path] = ceval

        out = ceval["AK4PUPPI_TightLeptonVeto"].evaluate(
                jets.eta,
                jets.chHEF,
                jets.neHEF,
                jets.chEmEF,
                jets.neEmEF,
                jets.muEF,
                jets.chMultiplicity,
                jets.neMultiplicity,
                jets.chMultiplicity + jets.neMultiplicity
        )
        return ak.values_astype(out, np.bool_)

    def build_trigger_masks(self, events, mc_campaign):
        """
        Returns (e_trig, mu_trig, emu_trig) per-event booleans.
        Handles missing HLT paths gracefully by defaulting to False.
        """
        import awkward as ak
        import numpy as np

        n = len(events)
        HLT = getattr(events, "HLT", None)

        def hlt(name):
            if HLT is not None and hasattr(HLT, name):
                return getattr(HLT, name)
            return ak.Array(np.zeros(n, dtype=bool))  # fallback

        # Electrons (common)
        e_trig = hlt("Ele32_WPTight_Gsf") | hlt("Photon200") | hlt("Ele115_CaloIdVT_GsfTrkIdT")

        # Muons by era
        if mc_campaign in ("RunIISummer20UL18", "Run2Autumn18"):
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100")
        elif mc_campaign in ("Run3Summer22", "Run3Summer23BPix", "Run3Summer22EE", "Run3Summer23", "RunIII2024Summer24"):
            mu_trig = hlt("Mu50") | hlt("HighPtTkMu100")
        else:
            mu_trig = hlt("Mu50") | hlt("OldMu100") | hlt("TkMu100") | hlt("HighPtTkMu100")

        emu_trig = mu_trig

        return e_trig, mu_trig, emu_trig

    def resolved_selections(self, tight_leptons, ak4_jets, *, lepton_masks=None, jet_masks=None, triggers=None):
        """
        Build PackedSelection for the resolved region.

        Baseline Criteria:
          - exactly 2 tight leptons
          - leading lepton pT > 60 GeV
          - subleading lepton pT > 53 GeV
          - at least two AK4 jets passing selection
          - ΔR > 0.4 between all pairs among {l1, l2, j1, j2}

        Resolved DY CR Criteria:
          - 60 < mll < 150 GeV
          - mlljj > 800 GeV  (built from l1, l2, j1, j2)

        Resolved SR Criteria:
          - mll > 400 GeV
          - mlljj > 800 GeV  (built from l1, l2, j1, j2)

                Plus flavor and trigger requirements.

                Implementation detail:
                    - Selections are built on padded leading objects (`ak.pad_none`) so
                        mass/ΔR expressions are defined even when objects are missing.
        """
        selections = PackedSelection()

        n_tight = ak.num(tight_leptons)
        n_ak4   = ak.num(ak4_jets)

        # flavor counts (per-event)
        n_tight_e  = ak.sum(tight_leptons.flavor == "electron", axis=1)
        n_tight_mu = ak.sum(tight_leptons.flavor == "muon",     axis=1)

        # flavor selections
        selections.add("two_tight_electrons", n_tight_e == 2)
        selections.add("two_tight_muons",     n_tight_mu == 2)
        selections.add("two_tight_em",        (n_tight_e == 1) & (n_tight_mu == 1))

        # Pad to always have 2 leptons/jets (None if missing)
        lpad = ak.pad_none(tight_leptons, 2)
        l1, l2 = lpad[:, 0], lpad[:, 1]

        jpad = ak.pad_none(ak4_jets, 2)
        j1, j2 = jpad[:, 0], jpad[:, 1]

        # Add criteria
        selections.add("two_tight_leptons",      n_tight == 2)
        selections.add("lead_tight_lepton_pt60", ak.fill_none(l1.pt > 60, False))
        selections.add("sublead_tight_pt53",     ak.fill_none(l2.pt > 53, False))
        selections.add("min_two_ak4_jets",       n_ak4 >= 2)

        # Invariant masses built from the cleaned leading pair
        mll   = (l1 + l2).mass
        mlljj = (l1 + l2 + j1 + j2).mass

        # mll selections
        selections.add("60_mll_150",   ak.fill_none((mll > 60) & (mll < 150), False))
        selections.add("mll_gt200",    ak.fill_none(mll > 200, False))
        selections.add("mll_gt400",    ak.fill_none(mll > 400, False))
        selections.add("mlljj_gt800",  ak.fill_none(mlljj > 800, False))

        # ΔR > 0.4 requirements among {l1, l2, j1, j2}.
        dr_ll   = ak.fill_none(l1.deltaR(l2) > 0.4, False)
        dr_l1j1 = ak.fill_none(l1.deltaR(j1) > 0.4, False)
        dr_l1j2 = ak.fill_none(l1.deltaR(j2) > 0.4, False)
        dr_l2j1 = ak.fill_none(l2.deltaR(j1) > 0.4, False)
        dr_l2j2 = ak.fill_none(l2.deltaR(j2) > 0.4, False)
        dr_jj   = ak.fill_none(j1.deltaR(j2) > 0.4, False)

        selections.add("dr_all_pairs_gt0p4", dr_ll & dr_l1j1 & dr_l1j2 & dr_l2j1 & dr_l2j2 & dr_jj)

        # Triggers
        if triggers is not None:
            e_trig, mu_trig, emu_trig = triggers
            selections.add("e_trigger",   e_trig)
            selections.add("mu_trigger",  mu_trig)
            selections.add("emu_trigger", emu_trig)

        if lepton_masks is None or jet_masks is None:
            raise ValueError("resolved_selections requires lepton_masks and jet_masks")

        # Jet-level counts per event
        n_ak4_pteta = ak.sum(jet_masks["ak4_pteta"], axis=1)
        n_ak4_id    = ak.sum(jet_masks["ak4_id"],    axis=1)
        selections.add("min_two_ak4_jets_pteta", n_ak4_pteta >= 2)
        selections.add("min_two_ak4_jets_id",    n_ak4_id    >= 2)

        # Lepton-level counts per event (separate by flavor)
        n_ele_pteta = ak.sum(lepton_masks["ele_pteta"], axis=1)
        n_mu_pteta  = ak.sum(lepton_masks["mu_pteta"],  axis=1)
        n_ele_id    = ak.sum(lepton_masks["ele_id"],    axis=1)
        n_mu_id     = ak.sum(lepton_masks["mu_id"],     axis=1)

        # pT/eta-only
        selections.add("two_pteta_electrons", n_ele_pteta == 2)
        selections.add("two_pteta_muons",     n_mu_pteta  == 2)
        selections.add("two_pteta_em",        (n_ele_pteta == 1) & (n_mu_pteta == 1))

        # ID-only
        selections.add("two_id_electrons", n_ele_id == 2)
        selections.add("two_id_muons",     n_mu_id  == 2)
        selections.add("two_id_em",        (n_ele_id == 1) & (n_mu_id == 1))

        return selections
    
    # --- Boosted helper functions -------------------------------------------------
    def remove_lepton(self,loose, tight):
        """Remove objects within ΔR<0.01 of the provided tight lepton."""
        match = (loose.deltaR(tight))
        keep_mask = match >= 0.01
        return loose[keep_mask]
    
    def ElectronCut(self, cut_bitmap, id_level=1):
        """Evaluate NanoAOD electron vid bitmap with an ID threshold.

        Awkward-array-safe version that:
          - ignores the isolation flag (cut index 7)
          - requires each considered cut to be >= `id_level`
        """
        nFlags = 10          # total number of bits (cuts)
        cut_size = 3         # 3 bits per cut
        ignore_flag = 7      # ignore the isolation flag
        mask_per_cut = (1 << cut_size) - 1  # 0b111 = 7
        # Define a helper applied per-electron.
        def passes(bitmap):
            for cut_nr in range(nFlags):
                if cut_nr == ignore_flag:
                    continue
                value = (bitmap >> (cut_nr * cut_size)) & mask_per_cut
                if value < id_level : 
                    return False
            return True
        # Apply per-electron check.
        mask = ak.Array([[passes(b) for b in event] for event in cut_bitmap])
        return mask
    
    def selectLooseElectrons(self, events):
        loose_noIso_mask = self.ElectronCut(events.Electron.vidNestedWPBitmap, id_level=2)
        loose_noIso_mask = ak.fill_none(loose_noIso_mask, False)

        # Ensure the HEEP flag is a true boolean array before bitwise ops.
        heep_flag = ak.fill_none(events.Electron.cutBased_HEEP, 0)
        heep_flag = heep_flag != 0

        loose_electrons = (
            (events.Electron.pt > 53)
            & (np.abs(events.Electron.eta) < 2.4)
            & (heep_flag | loose_noIso_mask)
        )
        return events.Electron[loose_electrons]

    def selectLooseMuons(self, events):
        loose_muons = (events.Muon.pt > 53) & (np.abs(events.Muon.eta) < 2.4) & (events.Muon.highPtId == 2)
        return events.Muon[loose_muons]
    
    def selectAK8Jets(self,events):
        """Baseline AK8 selection used in boosted categories."""
        ak8_jets = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4)  & (events.FatJet.msoftdrop > 40)
        return events.FatJet[ak8_jets]

    def selectAK8Jets_withLSF(self,events):
        """AK8 selection with LSF requirement used for boosted SR/CR definitions."""
        ak8_jets = (events.FatJet.pt > 200) & (np.abs(events.FatJet.eta) < 2.4)  & (events.FatJet.msoftdrop > 40) & (events.FatJet.lsf3 > 0.75)
        return events.FatJet[ak8_jets]
    
    def boosted_selections(self, events, era, triggers=None):
        """Build boosted PackedSelection and the objects needed for boosted histogram filling.

        Returns a tuple:
          (selections,
           tight_lep,
           ak8_cand_dy, dy_loose_lep,
           ak8_cand, of_candidate, sf_candidate)

        Notes
        - Boosted selections are skipped entirely upstream if `FatJet/lsf3` is missing.
        - The returned per-event objects are used directly by `fill_boosted_histograms()`.
        """

        selections = PackedSelection()

        # Object selections.
        looseElectrons = self.selectLooseElectrons(events)
        looseMuons = self.selectLooseMuons(events)
        AK8Jets = self.selectAK8Jets(events)
        AK8Jets_withLSF = self.selectAK8Jets_withLSF(events)
        is_signal = (getattr(events, "metadata", {}) or {}).get("physics_group") == "Signal"
        AK4Jets_inc, _, _ = self.select_jets(events, era, is_signal=is_signal)
        # define tight by querying loose
        tight_mask_e = (looseElectrons.cutBased_HEEP)
        tightElectrons_inc = looseElectrons[tight_mask_e]
        tightElectrons_inc = tightElectrons_inc[ak.argsort(tightElectrons_inc.pt, axis=1, ascending=False)]
        ## define loose 
        cut_bitmap = looseElectrons.vidNestedWPBitmap
        loose_e_mask = self.ElectronCut(cut_bitmap, id_level=2)
        looseElectrons = looseElectrons[loose_e_mask]
        ## -- tight muons --- ##
        tight_mask_mu = (looseMuons.tkRelIso < 0.1)
        tightMuons_inc = looseMuons[tight_mask_mu]
        tightMuons_inc = tightMuons_inc[ak.argsort(tightMuons_inc.pt, axis=1, ascending=False)]

        looseLeptons = ak.with_name(ak.concatenate((looseElectrons, looseMuons), axis=1), 'PtEtaPhiMCandidate')
        looseLeptons = looseLeptons[ak.argsort(looseLeptons.pt, axis=1, ascending=False)]
        tightLeptons_inc = ak.with_name(ak.concatenate((tightElectrons_inc, tightMuons_inc), axis=1), 'PtEtaPhiMCandidate')
        tightLeptons_inc = tightLeptons_inc[ak.argsort(tightLeptons_inc.pt, axis=1, ascending=False)]

        # Define a resolved-like tag and take boosted as the complement.
        has_two_leptons = ak.num(tightLeptons_inc) >= 2
        muons_padded = ak.pad_none(tightLeptons_inc, 2, axis=1)
        # Compute dr only for events with >=2 leptons.
        dr_l1l2 = ak.where(
            ak.num(tightLeptons_inc) >= 2,
            muons_padded[:,0].deltaR(muons_padded[:,1]),
            ak.full_like(ak.num(tightLeptons_inc), np.nan)
        )
        # For jets
        jpad = ak.pad_none(AK4Jets_inc, 2)
        j1, j2 = jpad[:, 0], jpad[:, 1] 
        dr_j1j2   = ak.fill_none(j1.deltaR(j2) > 0.4, False)

        has_two_jets = ak.num(AK4Jets_inc) >= 2

        # dr_jl_min: compute only if both jets and leptons exist.
        has_j_and_l = (ak.num(AK4Jets_inc) >= 1) & (ak.num(tightLeptons_inc) >= 1)
        dr_jl_min = ak.where( has_j_and_l, ak.min(AK4Jets_inc[:, :2].nearest(tightLeptons_inc).deltaR(AK4Jets_inc[:, :2]), axis=1), ak.full_like(has_j_and_l, np.nan))
        # Build all 2 leptons × 2 jets pairs.
        dr_lj = ak.cartesian({"lep": tightLeptons_inc[:,:2], "jet": AK4Jets_inc[:,:2]}, axis=1)
        dr_lj_vals = dr_lj["lep"].deltaR(dr_lj["jet"])
        # Condition: all l-j separations > 0.4.
        dr_lj_ok = ak.all(dr_lj_vals > 0.4, axis=1)
        resolved = (((ak.num(tightElectrons_inc)  + (ak.num(tightMuons_inc))) == 2) & (ak.num(AK4Jets_inc) >= 2) & (dr_l1l2 > 0.4) & (dr_j1j2 > 0.4) & (dr_jl_min >0.4)) #(dr_lj_ok))                                                                                                                                                                   
        boosted  = ~resolved
        selections.add("boostedtag",boosted)

        # Leading tight lepton pT > 60.
        tightLepton_padded = ak.pad_none(tightLeptons_inc,1,axis=1)
        tight_lep   = tightLepton_padded[:, 0]
        lead_pdgid  = ak.fill_none(abs(tight_lep.pdgId), 0)
        is_lead_mu  = lead_pdgid == 13
        is_lead_e   = lead_pdgid == 11
        is_tight_pt = tight_lep.pt > 60
        selections.add("leadTightwithPt60",is_tight_pt)
        # Remove this tight lepton from loose lepton selection.
        looseLeptons = self.remove_lepton(looseLeptons, tight_lep)

        # Same-flavor and other-flavor loose collections.
        sf_loose = looseLeptons[abs(looseLeptons.pdgId) == abs(tight_lep.pdgId)]
        of_loose = looseLeptons[abs(looseLeptons.pdgId) != abs(tight_lep.pdgId)]
        sf_loose = sf_loose[ak.argsort(sf_loose.pt, axis=1, ascending=False)]
        of_loose = of_loose[ak.argsort(of_loose.pt, axis=1, ascending=False)]

        # DY pair check.
        mll_pairs  = (tight_lep + sf_loose).mass
        mask_mll   = (mll_pairs > 60) & (mll_pairs < 150)
        has_dy_pair = ak.any(mask_mll, axis=1)
        # Pick the loose SF lepton candidate (first passing the DY window).
        DY_loose_lep  = ak.firsts(sf_loose[mask_mll])
        # AK8 jet candidate.
        AK8Jets = AK8Jets[ak.argsort(AK8Jets.pt, axis=1, ascending=False)]
        flag_ak8Jet = ak.num(AK8Jets)>=1
        AK8Jets = ak.pad_none(AK8Jets, 1, axis=1)
        dphi       = ak.fill_none(abs(AK8Jets.delta_phi(tight_lep)),0.0)
        has_ak8_dphi_gt2 = ak.any(dphi > 2, axis=1)
        ak8_mask   = dphi > 2.0
        AK8_cand_dy   = ak.firsts(AK8Jets[ak8_mask])
        # Require at least one AK8 jet with Δφ(jet, tight lepton) > 2.
        selections.add("Atleast1AK8Jets & dPhi(J,tightLept)>2", flag_ak8Jet & has_ak8_dphi_gt2 )
        
        # Case 1: DY CR.
        dr_dy  = AK8_cand_dy.deltaR(DY_loose_lep)
        mlj_dy = ak.where(dr_dy < 0.8,
                          (tight_lep + AK8_cand_dy).mass,
                          (tight_lep + DY_loose_lep + AK8_cand_dy).mass)
        mll_dy = (tight_lep + DY_loose_lep).mass
        pt_dilept_dy = (tight_lep + DY_loose_lep).pt
        pt_lj_dy = ak.where(dr_dy < 0.8,
                          (tight_lep + AK8_cand_dy).pt,
                          (tight_lep + DY_loose_lep + AK8_cand_dy).pt)
        sublead_pdgID = abs(DY_loose_lep.pdgId)
        is_sublead_mu = sublead_pdgID == 13
        is_sublead_e = sublead_pdgID ==11
        DYCR_mask = has_dy_pair & (mlj_dy > 800)
        selections.add("DYCR_mask", DYCR_mask)
        # Veto extra tight leptons for DY CR.
        extra_tight_mu = ak.sum(
            (looseMuons.tkRelIso < 0.1) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseMuons.deltaR(DY_loose_lep) > 0.01, True)),
            axis=1
        )
        extra_tight_el = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseElectrons.deltaR(DY_loose_lep) > 0.01, True)),
            axis=1
        )
        no_extra_tight_dyCR = (extra_tight_mu == 0) & (extra_tight_el == 0)

        # Case 2: SR (no DY, SF near AK8, no OF near AK8).
        flag_ak8Jet_lsf = ak.num(AK8Jets_withLSF)>=1
        AK8Jets_withLSF = ak.pad_none(AK8Jets_withLSF, 1, axis=1)
        dphi_lsf       = ak.fill_none(abs(AK8Jets_withLSF.delta_phi(tight_lep)),0.0)
        has_ak8_dphi_gt2_lsf = ak.any(dphi_lsf > 2, axis=1)
        ak8_mask_lsf   = dphi_lsf > 2.0
        AK8_cand   = ak.firsts(AK8Jets_withLSF[ak8_mask_lsf])
        selections.add("AK8JetswithLSF", flag_ak8Jet_lsf & has_ak8_dphi_gt2_lsf )
        dr_sf = AK8_cand.deltaR(sf_loose)
        mask_sf = dr_sf < 0.8
        # SF lepton candidate passing dR condition.
        sf_candidate = ak.firsts(sf_loose[mask_sf])
        sf_exist = ak.num(sf_loose[mask_sf])>=1
        dr_of = AK8_cand.deltaR(of_loose)
        mask_of = dr_of < 0.8
        # OF lepton candidate passing dR condition.
        of_candidate = ak.firsts(of_loose[mask_of])
        of_exist = ak.num(of_loose[mask_of])>=1

        is_sr = (~has_dy_pair) & (~ak.is_none(sf_candidate)) & ak.is_none(of_candidate) #sf_exist & (~of_exist) #(~ak.is_none(sf_candidate)) & ak.is_none(of_candidate)    
        sublead_pdgID = abs(sf_candidate.pdgId)
        is_sublead_mu_sr = sublead_pdgID == 13
        is_sublead_e_sr = sublead_pdgID ==11
        mlj_sr   = (tight_lep + AK8_cand).mass
        mll_sr = (tight_lep + sf_candidate).mass
        pt_dilept_sr  = (tight_lep + AK8_cand).pt
        pt_lj_sr = (tight_lep + sf_candidate).pt

        SR_mask = is_sr & (mlj_sr > 800) & (mll_sr > 200)
        selections.add("ee(mumu)SR", SR_mask)
        # -------- veto extra tight leptons for SR --------                                                                                                                 
        extra_tight_mu_sr = ak.sum(
            (looseMuons.tkRelIso < 0.1) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseMuons.deltaR(sf_candidate) > 0.01, True)),
            axis=1
        )
        extra_tight_el_sr = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseElectrons.deltaR(sf_candidate) > 0.01, True)),
            axis=1
        )
        no_extra_tight_sr = (extra_tight_mu_sr == 0) & (extra_tight_el_sr == 0)

        # Case 3: Flavor CR (no DY, OF near AK8).
        is_cr = (~has_dy_pair) & (~ak.is_none(of_candidate))  & ak.is_none(sf_candidate) #(~sf_exist) & of_exist #(~ak.is_none(of_candidate))  & ak.is_none(sf_candidate)  
        sublead_pdgID = abs(of_candidate.pdgId)
        is_sublead_mu_cr = sublead_pdgID == 13
        is_sublead_e_cr = sublead_pdgID ==11

        mlj_cr  = (tight_lep + AK8_cand).mass
        mll_cr = (tight_lep + of_candidate).mass
        pt_dilept_cr  = (tight_lep + AK8_cand).pt
        pt_lj_cr = (tight_lep + of_candidate).pt

        CR_mask = is_cr & (mlj_cr > 800) & (mll_cr > 200 )
        selections.add("e(mu) or mu(e)CR", CR_mask)
        # -------- veto extra tight leptons for flavor CR --------                                                                                                          
        extra_tight_mu_cr = ak.sum(
            (looseMuons.tkRelIso < 0.1) &
            (ak.fill_none(looseMuons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseMuons.deltaR(of_candidate) > 0.01, True)),
            axis=1
        )
        extra_tight_el_cr = ak.sum(
            (looseElectrons.cutBased_HEEP) &
            (ak.fill_none(looseElectrons.deltaR(tight_lep) > 0.01, True)) &
            (ak.fill_none(looseElectrons.deltaR(of_candidate) > 0.01, True)),
            axis=1
        )
        no_extra_tight_flav_cr = (extra_tight_mu_cr == 0) & (extra_tight_el_cr == 0)

        # Triggers.
        if triggers is not None:
            eTrig, muTrig, emu_trig = triggers
        
        # Region assignments.
        mumu_dy_cr = muTrig & DYCR_mask & is_lead_mu & no_extra_tight_dyCR
        ee_dy_cr   = eTrig & DYCR_mask & is_lead_e & no_extra_tight_dyCR
        emu_cr = eTrig  & CR_mask & is_lead_e & no_extra_tight_flav_cr # lead e, loose mu                                                                        
        mue_cr = muTrig & CR_mask & is_lead_mu & no_extra_tight_flav_cr # lead mu, loose e                                                                        
        ee_sr = eTrig & SR_mask & is_lead_e & no_extra_tight_sr
        mumu_sr = muTrig & SR_mask &  is_lead_mu & no_extra_tight_sr
        selections.add("mumu-dy_cr", mumu_dy_cr & is_sublead_mu)
        selections.add("ee-dy_cr",   ee_dy_cr & is_sublead_e)
        selections.add("mumu_sr", mumu_sr & is_sublead_mu_sr)
        selections.add("ee_sr",   ee_sr & is_sublead_e_sr)
        selections.add("emu-cr",     emu_cr & is_sublead_mu_cr)
        selections.add("mue-cr",     mue_cr & is_sublead_e_cr)        
        return selections, tight_lep, AK8_cand_dy,DY_loose_lep, AK8_cand,of_candidate, sf_candidate 

    def build_event_weights(self, events, metadata, is_mc):
        """
        Minimal weights:
          - MC: xsec/nevts normalization (+ optional DY UL18 scale) + lumi Up/Down
          - Data: unit weights
        NO genWeight, NO L1 prefire, NO pileup.
        """
        n = len(events)
        weights = Weights(n)

        if is_mc:
            # Cross-section normalization
            lumi = float(LUMIS[metadata.get("era")])
            xsec = float(metadata.get("xsec"))

            # IMPORTANT: use signed genEventSumw (do NOT abs) for NLO samples.
            sumw = float(metadata.get("genEventSumw"))
            if sumw == 0.0:
                raise ZeroDivisionError(
                    f"genEventSumw is zero for dataset '{metadata.get('sample')}'."
                )

            event_weight = events.genWeight * xsec * lumi * 1000.0 / sumw

            weights.add("event_weight", event_weight)

            syst_weights = {"Nominal": weights.weight()}

            # Optional lumi uncertainty (produces syst histograms only if enabled).
            if "lumi" in self._enabled_systs:
                era_key = metadata.get("era")
                delta = LUMI_UNC.get(era_key)
                if delta is None:
                    logger.warning(
                        f"No luminosity uncertainty defined for era '{era_key}'. "
                        "Skipping lumiUp/lumiDown systematics."
                    )
                else:
                    ones = np.ones(n, dtype=np.float32)
                    weights.add(
                        "lumi",
                        ones,
                        weightUp=ones * (1.0 + float(delta)),
                        weightDown=ones * (1.0 - float(delta)),
                    )
                    syst_weights = {
                        "Nominal": weights.weight(),
                        "LumiUp": weights.weight(modifier="lumiUp"),
                        "LumiDown": weights.weight(modifier="lumiDown"),
                    }
        
        else:  # is_data
            weights.add("data", np.ones(n, dtype=np.float32))
            syst_weights = { 
                "Nominal":  weights.weight(),
            }

        return weights, syst_weights

    def fill_resolved_histograms(self, output, region, cut, process_name, jets, leptons, weights, syst_weights):
        """Fill all resolved-region histograms for a given region selection mask."""
        leptons_cut = leptons[cut]
        jets_cut    = jets[cut]
        syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}

        for hist_name, _bins, _label, expr in RESOLVED_HIST_SPECS:
            vals = expr(leptons_cut, jets_cut)
            for syst_label, sw in syst_weights_cut.items():
                output[hist_name].fill(
                    process=process_name,
                    region=region,
                    syst=syst_label,
                    **{hist_name: vals},
                    weight=sw,
                )

    def fill_boosted_histograms(self, output, region, cut, process_name, leptons, ak8jets, looseleptons, weights, syst_weights):
        """Fill all boosted-region histograms for a given region selection mask."""
        syst_weights_cut = {k: v[cut] for k, v in syst_weights.items()}

        # Evaluate all base boosted quantities from the spec table.
        # Note: in boosted mode, the inputs are already per-event objects (not padded collections).
        value_map = {
            hist_name: expr(leptons, ak8jets, looseleptons)
            for hist_name, _bins, _label, expr in BOOSTED_HIST_SPECS
        }

        # Special-case DY CR: mass_twoobject / pt_twoobject switch depending on ΔR.
        if "boosted_dy_cr" in region:
            dr_dy = ak8jets.deltaR(looseleptons)
            value_map["mass_twoobject"] = ak.where(
                dr_dy < 0.8,
                (leptons + ak8jets).mass,
                (leptons + ak8jets + looseleptons).mass,
            )
            value_map["pt_twoobject"] = ak.where(
                dr_dy < 0.8,
                (leptons + ak8jets).pt,
                (leptons + ak8jets + looseleptons).pt,
            )

        for hist_name, vals_all in value_map.items():
            vals = vals_all[cut]
            for syst_label, sw in syst_weights_cut.items():
                output[hist_name].fill(
                    process=process_name,
                    region=region,
                    syst=syst_label,
                    **{hist_name: vals},
                    weight=sw,
                )

    def fill_cutflows(self, output, selections, weights):
        """
        Build flat and cumulative cutflows for ee, mumu, and em channels.
        Also store one-bin histograms for pT/eta-only vs ID-only (jets & leptons).

                Output layout (keys under `output["cutflow"]`):
                    - top-level: `no_cuts`, plus a few global one-bin counters
                    - per-flavor: `ee`, `mumu`, `em`
                        - one-bin counters for each step (weighted + `_unweighted`)
                        - `onecut` / `cumulative` (and unweighted variants) from PackedSelection.cutflow
        """
        output.setdefault("cutflow", {})

        required = [
            SEL_MIN_TWO_AK4_JETS_ID,
            SEL_MIN_TWO_AK4_JETS_PTETA,
            SEL_TWO_PTETA_ELECTRONS,
            SEL_TWO_PTETA_MUONS,
            SEL_TWO_PTETA_EM,
            SEL_TWO_ID_ELECTRONS,
            SEL_TWO_ID_MUONS,
            SEL_TWO_ID_EM,
            SEL_E_TRIGGER,
            SEL_MU_TRIGGER,
            SEL_EMU_TRIGGER,
            SEL_DR_ALL_PAIRS_GT0P4,
            SEL_MLL_GT200,
            SEL_MLLJJ_GT800,
            SEL_MLL_GT400,
        ]

        # Fail fast with a helpful message if resolved_selections does not define expected keys.
        missing = []
        for name in required:
            try:
                selections.all(name)
            except Exception:
                missing.append(name)
        if missing:
            available = getattr(selections, "names", None)
            raise KeyError(
                "fill_cutflows: missing selection keys: "
                f"{missing}. Available keys: {sorted(available) if available else 'unknown'}"
            )

        mask = selections.all

        m_j2_id = mask(SEL_MIN_TWO_AK4_JETS_ID)
        m_j2_pteta = mask(SEL_MIN_TWO_AK4_JETS_PTETA)

        m_two_pteta_e = mask(SEL_TWO_PTETA_ELECTRONS)
        m_two_pteta_mu = mask(SEL_TWO_PTETA_MUONS)
        m_two_pteta_em = mask(SEL_TWO_PTETA_EM)

        m_two_id_e = mask(SEL_TWO_ID_ELECTRONS)
        m_two_id_mu = mask(SEL_TWO_ID_MUONS)
        m_two_id_em = mask(SEL_TWO_ID_EM)

        m_e_trig = mask(SEL_E_TRIGGER)
        m_mu_trig = mask(SEL_MU_TRIGGER)
        m_em_trig = mask(SEL_EMU_TRIGGER)

        m_dr = mask(SEL_DR_ALL_PAIRS_GT0P4)
        m_mll200 = mask(SEL_MLL_GT200)
        m_mlljj8 = mask(SEL_MLLJJ_GT800)
        m_mll400 = mask(SEL_MLL_GT400)

        w = weights.weight()  # nominal weight

        def _onebin_hist(lastcut_name: str, mask, use_weights=True):
            """1 bin in [0,1), named by the last cut."""
            h = hist.Hist(
                hist.axis.Regular(1, 0, 1, name=lastcut_name, label=lastcut_name),
                storage=hist.storage.Weight(),
            )
            n_pass = int(ak.count_nonzero(mask))
            if use_weights:
                w_pass = ak.to_numpy(w[mask])
            else:
                # unweighted: count passing events
                w_pass = np.ones(n_pass, dtype="f8")
            if n_pass:
                coords = np.full(w_pass.size, 0.5, dtype=float)
                h.fill(**{lastcut_name: coords}, weight=w_pass)
            return h

        def _store_both_in(container: dict, name: str, mask):
            """Store weighted and unweighted variants into a given dict container."""
            container[name] = _onebin_hist(name, mask, use_weights=True)
            container[f"{name}_unweighted"] = _onebin_hist(name, mask, use_weights=False)

        # --- Top-level: no_cuts only
        n_events = len(w)
        mask_all = np.ones(n_events, dtype=bool)
        _store_both_in(output["cutflow"], "no_cuts", mask_all)
        _store_both_in(output["cutflow"], SEL_MIN_TWO_AK4_JETS_PTETA, m_j2_pteta)

        # --- Prepare flavor folders (+ jets folder for pt/eta vs ID)
        output["cutflow"].setdefault("ee", {})
        output["cutflow"].setdefault("mumu", {})
        output["cutflow"].setdefault("em", {})

        # --- Define cumulative chains per flavor (kept as-is)
        chains = {
            "ee":   [
                SEL_MIN_TWO_AK4_JETS_PTETA,
                SEL_MIN_TWO_AK4_JETS_ID,
                SEL_TWO_PTETA_ELECTRONS,
                SEL_TWO_ID_ELECTRONS,
                SEL_E_TRIGGER,
                SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLLJJ_GT800,
                SEL_MLL_GT200,
                SEL_MLL_GT400,
            ],
            "mumu": [
                SEL_MIN_TWO_AK4_JETS_PTETA,
                SEL_MIN_TWO_AK4_JETS_ID,
                SEL_TWO_PTETA_MUONS,
                SEL_TWO_ID_MUONS,
                SEL_MU_TRIGGER,
                SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLLJJ_GT800,
                SEL_MLL_GT200,
                SEL_MLL_GT400,
            ],
            "em":   [
                SEL_MIN_TWO_AK4_JETS_PTETA,
                SEL_MIN_TWO_AK4_JETS_ID,
                SEL_TWO_PTETA_EM,
                SEL_TWO_ID_EM,
                SEL_EMU_TRIGGER,
                SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLLJJ_GT800,
                SEL_MLL_GT200,
                SEL_MLL_GT400,
            ],
        }

        name2mask = {
            SEL_MIN_TWO_AK4_JETS_ID: m_j2_id,
            SEL_MIN_TWO_AK4_JETS_PTETA: m_j2_pteta,

            SEL_TWO_PTETA_ELECTRONS: m_two_pteta_e,
            SEL_TWO_PTETA_MUONS: m_two_pteta_mu,
            SEL_TWO_PTETA_EM: m_two_pteta_em,

            SEL_TWO_ID_ELECTRONS: m_two_id_e,
            SEL_TWO_ID_MUONS: m_two_id_mu,
            SEL_TWO_ID_EM: m_two_id_em,

            SEL_E_TRIGGER: m_e_trig,
            SEL_MU_TRIGGER: m_mu_trig,
            SEL_EMU_TRIGGER: m_em_trig,

            SEL_DR_ALL_PAIRS_GT0P4: m_dr,
            SEL_MLL_GT200: m_mll200,
            SEL_MLLJJ_GT800: m_mlljj8,
            SEL_MLL_GT400: m_mll400,
        }

        # --- Step-by-step cumulative counters per flavor
        for flavor, steps in chains.items():
            cumu = name2mask[steps[0]].copy()
            bucket = output["cutflow"][flavor]
            for step in steps:
                if step != steps[0]:
                    cumu = cumu & name2mask[step]
                _store_both_in(bucket, step, cumu)

        # --- Region cutflows: onecut + cumulative into flavor folders
        for flavor, order in chains.items():
            cf = selections.cutflow(*order, weights=weights)

            # weighted
            res = cf.yieldhist(weighted=True)
            h_onecut, h_cum = res[0], res[1]
            bucket = output["cutflow"][flavor]
            bucket["onecut"] = h_onecut
            bucket["cumulative"] = h_cum

            # unweighted
            res_unw = cf.yieldhist(weighted=False)
            h_onecut_unw, h_cum_unw = res_unw[0], res_unw[1]
            bucket["onecut_unweighted"] = h_onecut_unw
            bucket["cumulative_unweighted"] = h_cum_unw


    def process(self, events):
        """Run analysis for one NanoEvents chunk and return a dataset-nested output dict."""
        output = self.make_output()
        metadata = events.metadata

        mc_campaign = metadata.get("era")
        process_name = metadata.get("physics_group")
        dataset = metadata.get("sample")
    
        datatype = (metadata.get("datatype") or "").strip().lower()
        is_mc = datatype == "mc"
        is_data = not is_mc

        # Apply lumi mask (data only).
        events = self.apply_lumi_mask(events, mc_campaign, is_data)
        
        # Build physics objects.
        tight_leptons, loose_leptons, lepton_masks = self.select_leptons(events)
        ak4_jets, ak8_jets, jet_masks = self.select_jets(events, mc_campaign, is_signal=(process_name == "Signal"))
        # Construct trigger masks.
        triggers = self.build_trigger_masks(events, mc_campaign)

        # Resolved selections.
        resolved_selections = self.resolved_selections(
            tight_leptons,
            ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        boosted_payload = None
        # Boosted path needs FatJet (+ lsf3) branches; skip gracefully if absent.
        try:
            has_fatjet = hasattr(events, "FatJet")
            has_lsf3 = has_fatjet and ("lsf3" in getattr(events.FatJet, "fields", []))
            if has_fatjet and has_lsf3:
                boosted_payload = self.boosted_selections(events, mc_campaign, triggers=triggers)
            else:
                logger.info("Skipping boosted selections: missing FatJet/lsf3 branches")
        except Exception as e:
            logger.warning(f"Boosted selections failed; skipping boosted histograms: {e}")

        weights, syst_weights = self.build_event_weights(events, metadata, is_mc)

        # Define the resolved regions.
        resolved_regions = {
            'wr_ee_resolved_dy_cr': resolved_selections.all(
                SEL_TWO_TIGHT_ELECTRONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_E_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_60_MLL_150, SEL_MLLJJ_GT800,
            ),
            'wr_mumu_resolved_dy_cr': resolved_selections.all(
                SEL_TWO_TIGHT_MUONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_MU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_60_MLL_150, SEL_MLLJJ_GT800,
            ),
            'wr_resolved_flavor_cr': resolved_selections.all(
                SEL_TWO_TIGHT_EM, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_EMU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLL_GT400, SEL_MLLJJ_GT800,
            ),
            'wr_ee_resolved_sr': resolved_selections.all(
                SEL_TWO_TIGHT_ELECTRONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_E_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLL_GT400, SEL_MLLJJ_GT800,
            ),
            'wr_mumu_resolved_sr': resolved_selections.all(
                SEL_TWO_TIGHT_MUONS, SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
                SEL_MU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
                SEL_MLL_GT400, SEL_MLLJJ_GT800,
            ),
        }
        # Fill resolved histograms.
        for region, cuts in resolved_regions.items():
            self.fill_resolved_histograms(output, region, cuts, process_name, ak4_jets, tight_leptons, weights, syst_weights)

        # Fill resolved cutflows.
        self.fill_cutflows(output, resolved_selections, weights)

        if boosted_payload is not None:
            boosted_selection_list, tight_lep, AK8_cand_dy, DY_loose_lep, AK8_cand, of_candidate, sf_candidate = boosted_payload

            boosted_regions = {
                'wr_mumu_boosted_dy_cr': boosted_selection_list.all(
                    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
                    SEL_ATLEAST1AK8_DPHI_GT2, SEL_MUMU_DYCR,
                ),
                'wr_mumu_boosted_sr': boosted_selection_list.all(
                    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_MUMU_SR, SEL_AK8JETS_WITH_LSF,
                ),
                'wr_ee_boosted_dy_cr': boosted_selection_list.all(
                    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
                    SEL_ATLEAST1AK8_DPHI_GT2, SEL_EE_DYCR,
                ),
                'wr_ee_boosted_sr': boosted_selection_list.all(
                    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_EE_SR, SEL_AK8JETS_WITH_LSF,
                ),
                'wr_emu_boosted_flavor_cr': boosted_selection_list.all(
                    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_EMU_CR, SEL_AK8JETS_WITH_LSF,
                ),
                'wr_mue_boosted_flavor_cr': boosted_selection_list.all(
                    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_MUE_CR, SEL_AK8JETS_WITH_LSF,
                ),
            }
            for region, cuts in boosted_regions.items():
                if "dy_cr" in region:
                    self.fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand_dy, DY_loose_lep, weights, syst_weights)
                elif "flavor_cr" in region:
                    self.fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand, of_candidate, weights, syst_weights)
                else:
                    self.fill_boosted_histograms(output, region, cuts, process_name, tight_lep, AK8_cand, sf_candidate, weights, syst_weights)
                
        nested_output = {dataset: {**output}}
        
        return nested_output

    def postprocess(self, accumulator):
        return accumulator
