"""Lightweight configuration for the Coffea WR analysis.

Keep this module dependency-free so it can be shipped to Dask workers cheaply.
"""

# Integrated luminosities (fb^-1)
LUMIS = {
    "RunIISummer20UL18": 59.83,
    "Run3Summer22": 7.9804,
    "Run3Summer22EE": 26.6717,
    "Run3Summer23": 17.794,       # placeholder — update when 2023 lumi is finalized
    "Run3Summer23BPix": 9.451,    # placeholder — update when 2023 lumi is finalized
    "RunIII2024Summer24": 109.08,
}

# Default MC fileset tag per era.  All MC DY filesets contain every
# physics_group (tt_tW, Nonprompt, Other, …), so this tag only controls
# which DY variant is used when --dy is not explicitly given.
DEFAULT_MC_TAG = {
    "RunIISummer20UL18": "dy_lo_ht",
    "Run3Summer22": "dy_lo_inc",
    "Run3Summer22EE": "dy_lo_inc",
    "Run3Summer23": "dy_lo_inc",
    "Run3Summer23BPix": "dy_lo_inc",
    "RunIII2024Summer24": "dy_lo_inc",
}

# Eras where unskimmed signal filesets are not available (no DAS entries).
# For these eras, --unskimmed signal falls back to skimmed filesets.
SKIMMED_ONLY_SIGNAL = {"RunIISummer20UL18"}

# Golden JSON paths for data lumi masking
LUMI_JSONS = {
    "RunIISummer20UL18": "data/lumis/RunII/2018/RunIISummer20UL18/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt",
    "Run3Summer22": "data/lumis/Run3/2022/Cert_Collisions2022_355100_362760_Golden.txt",
    "Run3Summer22EE": "data/lumis/Run3/2022/Cert_Collisions2022_355100_362760_Golden.txt",
    "RunIII2024Summer24": "data/lumis/Run3/2024/RunIII2024Summer24/Cert_Collisions2024_378981_386951_Golden.txt",
}

# JSON POG payloads
JME_JSONS = {
    "RunIII2024Summer24": "data/jsonpog/JME/Run3/RunIII2024Summer24/jetid.json.gz",
}

JETVETO_JSONS = {
    "RunIISummer20UL18": "data/jsonpog/JME/RunII/RunIISummer20UL18/jetvetomaps.json.gz",
    "RunIII2024Summer24": "data/jsonpog/JME/Run3/RunIII2024Summer24/jetvetomaps.json.gz",
}

JETVETO_CORRECTION_NAMES = {
    "RunIISummer20UL18": "Summer19UL18_V1",
    "RunIII2024Summer24": "Summer24Prompt24_RunBCDEFGHI_V1",
}

MUON_JSONS = {
    "RunIISummer20UL18": "data/jsonpog/MUO/RunII/RunIISummer20UL18/muon_HighPt.json.gz",
    "RunIII2024Summer24": "data/jsonpog/MUO/Run3/RunIII2024Summer24/muon_HighPt.json",
}

PILEUP_JSONS = {
    "RunIISummer20UL18": "data/jsonpog/LUM/RunII/RunIISummer20UL18/puWeights.json.gz",
    "RunIII2024Summer24": "data/jsonpog/LUM/Run3/RunIII2024Summer24/puWeights_BCDEFGHI.json.gz",
}

# Correction name inside each pileup JSON (differs per era)
PILEUP_CORRECTION_NAMES = {
    "RunIISummer20UL18": "Collisions18_UltraLegacy_goldenJSON",
    "RunIII2024Summer24": "Collisions24_BCDEFGHI_goldenJSON",
}

ELECTRON_JSONS = {
    "RunIISummer20UL18": {
        "RECO": "data/jsonpog/EGM/RunII/RunIISummer20UL18/electron.json.gz",
    },
    "RunIII2024Summer24": {
        "RECO": "data/jsonpog/EGM/Run3/RunIII2024Summer24/electron.json.gz",
        "TRIGGER": "data/jsonpog/EGM/Run3/RunIII2024Summer24/electronHlt.json.gz",
    },
}

# correctionlib era key used inside EGM JSON payloads (differs from analysis era name)
ELECTRON_SF_ERA_KEYS = {
    "RunIISummer20UL18": "2018",
    "RunIII2024Summer24": "2024Prompt",
}

# Per-era electron Reco SF configuration (correction name and WP names differ between eras)
ELECTRON_RECO_CONFIG = {
    "RunIISummer20UL18": {
        "correction": "UL-Electron-ID-SF",
        "wp_low": "RecoBelow20",
        "wp_high": "RecoAbove20",
        "pt_split": 20.0,
    },
    "RunIII2024Summer24": {
        "correction": "Electron-ID-SF",
        "wp_low": "Reco20to75",
        "wp_high": "RecoAbove75",
        "pt_split": 75.0,
    },
}

# Systematic uncertainties: integrated luminosity fractional uncertainty
LUMI_UNC = {
    "RunIISummer20UL18": 0.025,  # 2.5% (UL2018)
    "Run3Summer22": 0.014,  # 1.4% (2022)
    "Run3Summer22EE": 0.014,  # 1.4% (2022EE)
    "Run3Summer23": 0.014,  # placeholder — update when 2023 lumi is finalized
    "Run3Summer23BPix": 0.014,  # placeholder — update when 2023 lumi is finalized
    "RunIII2024Summer24": 0.014,  # placeholder until 2024 lumi is finalized
}

# --- Selection name constants (single source of truth for string keys) ---------
#
# Used for PackedSelection.add() names, region definitions, and cutflow bookkeeping.
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

# --- Physics thresholds (single source of truth for analysis cuts) -------------
CUTS = {
    "lepton_pt_min": 53,
    "lepton_eta_max": 2.4,
    "muon_highPtId": 2,
    "muon_iso_max": 0.1,
    "ak4_pt_min": 40,
    "ak4_eta_max": 2.4,
    "ak8_pt_min": 200,
    "ak8_eta_max": 2.4,
    "ak8_msoftdrop_min": 40,
    "ak8_lsf3_min": 0.75,
    "lead_lepton_pt_min": 60,
    "sublead_lepton_pt_min": 53,
    "mll_dy_low": 60,
    "mll_dy_high": 150,
    "mll_sr_min": 200,
    "mll_sr_high_min": 400,
    "mlljj_min": 800,
    "dr_min": 0.4,
    "dphi_boosted_min": 2.0,
    "dr_loose_veto": 0.01,
    "dr_ak8_loose": 0.8,
    "jet_veto_pt_min": 15,
}
