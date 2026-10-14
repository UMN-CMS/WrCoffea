"""Lightweight configuration for the Coffea WR analysis.

Loads physics parameters and per-era settings from ``config.yaml`` (shipped
with the package).  Edit the YAML file to change cuts or add new eras
without modifying Python code.

Selection name constants (``SEL_*``) remain here as Python constants since
they are code-level identifiers used for PackedSelection bookkeeping.
"""

import re
import logging
import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Load YAML config from the package directory.
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

#: Environment variable naming a YAML overlay merged on top of config.yaml:
#: each top-level mapping is updated key by key, anything else is replaced.
#: Meant for studies (e.g. a new reweighting payload) that must not change the
#: shared config other runs read; local Dask workers inherit the environment,
#: Condor jobs do not.
CONFIG_OVERLAY_ENV = "WRCOFFEA_CONFIG_OVERLAY"


def _merge_overlay(cfg: dict, overlay: dict) -> dict:
    out = dict(cfg)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


CONFIG_OVERLAY = os.environ.get(CONFIG_OVERLAY_ENV) or None
if CONFIG_OVERLAY:
    with open(CONFIG_OVERLAY, "r", encoding="utf-8") as _f:
        _cfg = _merge_overlay(_cfg, yaml.safe_load(_f))
    logging.getLogger(__name__).warning("config.yaml overlaid with %s (%s)", CONFIG_OVERLAY, CONFIG_OVERLAY_ENV)

# ---------------------------------------------------------------------------
# Resolve repo-relative data paths against the repo root, so lumi masks and
# correctionlib payloads load regardless of the current working directory.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_data_paths(node):
    """Recursively rewrite ``data/...`` path strings to absolute paths."""
    if isinstance(node, dict):
        return {k: _resolve_data_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_data_paths(v) for v in node]
    if isinstance(node, str) and node.startswith("data/"):
        return str(REPO_ROOT / node)
    return node


_cfg = _resolve_data_paths(_cfg)

# ---------------------------------------------------------------------------
# Expose config sections as module-level constants (preserves existing API).
# ---------------------------------------------------------------------------

LUMIS = _cfg["lumis"]
DEFAULT_MC_TAG = _cfg["default_mc_tag"]
DY_VARIANTS = _cfg["dy_variants"]

LUMI_JSONS = _cfg["lumi_jsons"]
JME_JSONS = _cfg["jme_jsons"]
JERC_JSONS = _cfg["jerc_jsons"]
JETVETO_JSONS = _cfg["jetveto_jsons"]
JETVETO_CORRECTION_NAMES = _cfg["jetveto_correction_names"]
JETVETO_MODES = _cfg.get("jetveto_mode") or {}
AK8_MSOFTDROP_MODES = _cfg.get("ak8_msoftdrop_mode") or {}

MUON_JSONS = _cfg["muon_jsons"]

PILEUP_JSONS = _cfg["pileup_jsons"]
PILEUP_CORRECTION_NAMES = _cfg["pileup_correction_names"]

ELECTRON_JSONS = _cfg["electron_jsons"]
ELECTRON_SF_ERA_KEYS = _cfg["electron_sf_era_keys"]
ELECTRON_HLT_ERA_KEYS = _cfg.get("electron_hlt_era_keys", {})
# era -> Path name evaluated in the electron TRIGGER payload.
ELECTRON_HLT_PATHS = _cfg["electron_hlt_paths"]
ELECTRON_RECO_CONFIG = _cfg["electron_reco_config"]

LUMI_UNC = _cfg["lumi_unc"]
# era -> era whose MC skims the data-only era reuses (see config.yaml mc_alias).
MC_ALIAS = _cfg.get("mc_alias") or {}
# era -> JSON payload for the DY Z-pT (QCD x EW) reweighting; eras absent here
# are not reweighted.
DY_ZPT_REWEIGHT_JSONS = _cfg.get("dy_zpt_reweight") or {}
# era -> JSON payload for the data-driven DY reshape vs jet pT.
DY_RESHAPE_JSONS = _cfg.get("dy_reshape") or {}
# era -> regex of the DY dataset names its Z-pT reweight and reshape were derived
# for; other DY samples get the identity.  Required for every era with a payload.
DY_REWEIGHT_DATASETS = _cfg.get("dy_reweight_datasets") or {}


def check_dy_reweight_datasets(zpt_jsons, reshape_jsons, datasets):
    """Raise unless every era with a DY Z-pT or reshape payload names, with a valid
    regular expression, the DY datasets it was derived for."""
    missing = sorted((set(zpt_jsons) | set(reshape_jsons)) - set(datasets))
    if missing:
        raise ValueError(
            f"config.yaml: eras {missing} have a dy_zpt_reweight or dy_reshape payload but no "
            "dy_reweight_datasets entry naming the DY datasets it was derived for."
        )
    for era, pattern in datasets.items():
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"config.yaml: dy_reweight_datasets[{era}] = {pattern!r}: {exc}") from exc


check_dy_reweight_datasets(DY_ZPT_REWEIGHT_JSONS, DY_RESHAPE_JSONS, DY_REWEIGHT_DATASETS)
# era -> JSON payload for the signal k(m_lN)/<k> reweighting.
SIGNAL_KFACTOR_JSONS = _cfg.get("signal_kfactor") or {}
# era -> JSON table {sample: LHA ID of its LHEPdfWeight} (scripts/pdf_sets.py);
# gates the ``pdf`` systematic family.
PDF_SETS = _cfg.get("pdf_sets") or {}
# era -> JSON payload of the high-pT muon momentum corrections (GE kappa map,
# MC smearing); eras listed get wrcoffea.muon_momentum, the others the TuneP
# placeholder.  The payloads (data/muon_highpt/) hold numbers from CMS-internal
# documents; docs/studies/muon_momentum.md lists their sources.
MUON_HIGHPT = _cfg.get("muon_highpt") or {}
# era -> correction name in the muon TRIGGER payload (default NUM_HLT_DEN_HighPtLooseRelIsoProbes).
MUON_HLT_CORRECTIONS = _cfg.get("muon_hlt_correction") or {}
# eras whose MC gets the NanoAOD L1 prefiring weight (Run 2 UL).
PREFIRING_ERAS = frozenset(_cfg.get("prefiring_eras") or [])
# era -> {eta_max, phi_min, phi_max}: HEM 15/16 electron veto window; eras
# absent here apply no veto.
HEM_VETO = _cfg.get("hem_veto") or {}
# era -> {electron: {value, err_up, err_down}, muon: {...}}: LSF scale factor
# keyed by the flavour of the lepton inside the AK8 jet.
LSF_SF = _cfg.get("lsf_sf") or {}
CUTS = _cfg["cuts"]

# ---------------------------------------------------------------------------
# Selection name constants (code-level identifiers, not user-configurable)
# ---------------------------------------------------------------------------

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
# No selected electron inside the HEM 15/16 window (always True for eras
# without a ``hem_veto`` entry).
SEL_HEM_VETO = "hem_veto"
# The boosted HEM 15/16 veto on its own, for the cutflows (the regions fold it into SEL_*_SR / SEL_*_CR):
# the leading tight lepton and the same-flavour (SR) or opposite-flavour (flavour CR) lepton in the jet.
SEL_HEM_VETO_BOOSTED_SR = "hem_veto_boosted_sr"
SEL_HEM_VETO_BOOSTED_CR = "hem_veto_boosted_cr"
SEL_MLL_GT200 = "mll_gt200"
SEL_MLLJJ_GT800 = "mlljj_gt800"
SEL_MLLJJ_LT800 = "mlljj_lt800"
SEL_MLL_GT400 = "mll_gt400"

# Resolved region selection keys
SEL_TWO_TIGHT_LEPTONS = "two_tight_leptons"
SEL_TWO_TIGHT_ELECTRONS = "two_tight_electrons"
SEL_TWO_TIGHT_MUONS = "two_tight_muons"
SEL_TWO_TIGHT_EM = "two_tight_em"
SEL_NO_THIRD_TIGHT_LEPTON = "no_third_tight_lepton"   # at most two tight leptons of any flavour
SEL_LEAD_TIGHT_PT60 = "lead_tight_lepton_pt60"
SEL_SUBLEAD_TIGHT_PT53 = "sublead_tight_pt53"
SEL_MIN_TWO_AK4_JETS = "min_two_ak4_jets"
SEL_60_MLL_150 = "60_mll_150"

# DY CR halves of the AN-19-083 CR1 -> CR2 reshape test (study ``dy-cr-split``).
# The resolved selections cut on the two tight leptons, the boosted ones on the
# DY candidate pair; each topology fills its own PackedSelection, and only when
# the study is requested.
DYCR_MLL_SPLIT = 100.0
SEL_MLL_60_100 = "mll_60_100"
SEL_MLL_100_150 = "mll_100_150"

# Boosted region selection keys
SEL_BOOSTEDTAG = "boosted_tag"
SEL_LEAD_TIGHT_PT60_BOOSTED = "lead_tight_lepton_pt60_boosted"
SEL_DYCR_MASK = "dy_cr_mask"
SEL_ATLEAST1AK8_DPHI_GT2 = "atleast_1ak8_dphi_gt2"
SEL_AK8JETS_WITH_LSF = "ak8_jets_with_lsf"
SEL_MUMU_DYCR = "mumu_dy_cr"
SEL_EE_DYCR = "ee_dy_cr"
SEL_MUMU_SR = "mumu_sr"
SEL_EE_SR = "ee_sr"
SEL_EMU_CR = "emu_cr"
SEL_MUE_CR = "mue_cr"
SEL_JET_VETO_MAP = "jet_veto_map"

# Study selections (regions.REGION_VARIANTS), filled only when a requested variant uses them.
# nolsf: the boosted SR / flavour-CR selections rebuilt on the leading AK8 jet with |dphi| > 2 and no LSF
# requirement (the DY CR jet), with their lepton-in-jet candidates and masses (AN-19-083 Figs. 12, 93-94).
SEL_EE_SR_NOLSF = "ee_sr_nolsf"
SEL_MUMU_SR_NOLSF = "mumu_sr_nolsf"
SEL_EMU_CR_NOLSF = "emu_cr_nolsf"
SEL_MUE_CR_NOLSF = "mue_cr_nolsf"
SEL_MLL_GT200_BOOSTED_NOLSF = "mll_gt200_boosted_nolsf"
SEL_MLJ_GT800_BOOSTED_NOLSF = "mlj_gt800_boosted_nolsf"

# Boosted cutflow intermediate selection keys
SEL_LEAD_IS_ELECTRON = "lead_is_electron"
SEL_LEAD_IS_MUON = "lead_is_muon"
SEL_NO_DY_PAIR = "no_dy_pair"
SEL_NO_EXTRA_TIGHT_SR = "no_extra_tight_sr"
SEL_NO_EXTRA_TIGHT_CR = "no_extra_tight_cr"
SEL_SF_LEPTON_IN_AK8 = "sf_lepton_in_ak8"
SEL_NO_OF_LEPTON_IN_AK8 = "no_of_lepton_in_ak8"
SEL_OF_LEPTON_IN_AK8 = "of_lepton_in_ak8"
SEL_NO_SF_LEPTON_IN_AK8 = "no_sf_lepton_in_ak8"
SEL_MLL_GT200_BOOSTED = "mll_gt200_boosted"
SEL_MLJ_GT800_BOOSTED = "mlj_gt800_boosted"
