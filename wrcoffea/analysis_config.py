"""Lightweight configuration for the Coffea WR analysis.

Loads physics parameters and per-era settings from ``config.yaml`` (shipped
with the package).  Edit the YAML file to change cuts or add new eras
without modifying Python code.

Selection name constants (``SEL_*``) remain here as Python constants since
they are code-level identifiers used for PackedSelection bookkeeping.
"""

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Load YAML config from the package directory.
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Expose config sections as module-level constants (preserves existing API).
# ---------------------------------------------------------------------------

LUMIS = _cfg["lumis"]
DEFAULT_MC_TAG = _cfg["default_mc_tag"]

LUMI_JSONS = _cfg["lumi_jsons"]
JME_JSONS = _cfg["jme_jsons"]

JETVETO_JSONS = _cfg["jetveto_jsons"]
JETVETO_CORRECTION_NAMES = _cfg["jetveto_correction_names"]

MUON_JSONS = _cfg["muon_jsons"]

PILEUP_JSONS = _cfg["pileup_jsons"]
PILEUP_CORRECTION_NAMES = _cfg["pileup_correction_names"]

ELECTRON_JSONS = _cfg["electron_jsons"]
ELECTRON_SF_ERA_KEYS = _cfg["electron_sf_era_keys"]
ELECTRON_RECO_CONFIG = _cfg["electron_reco_config"]

LUMI_UNC = _cfg["lumi_unc"]
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
SEL_MLL_GT200 = "mll_gt200"
SEL_MLLJJ_GT800 = "mlljj_gt800"
SEL_MLLJJ_LT800 = "mlljj_lt800"
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
