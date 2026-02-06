"""Lightweight configuration for the Coffea WR analysis.

Keep this module dependency-free so it can be shipped to Dask workers cheaply.
"""

# Integrated luminosities (fb^-1)
LUMIS = {
    "RunIISummer20UL18": 59.83,
    "Run3Summer22": 7.9804,
    "Run3Summer22EE": 26.6717,
    "RunIII2024Summer24": 109.08,
}

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

MUON_JSONS = {
    "RunIII2024Summer24": "data/jsonpog/MUO/muon_HighPt.json",
}

ELECTRON_JSONS = {
    "RunIII2024Summer24": {
        "RECO": "data/jsonpog/EGM/electron.json.gz",
        "TRIGGER": "data/jsonpog/EGM/electronHlt.json.gz",
    },
}

# Systematic uncertainties: integrated luminosity fractional uncertainty
LUMI_UNC = {
    "RunIISummer20UL18": 0.025,  # 2.5% (UL2018)
    "Run3Summer22": 0.014,  # 1.4% (2022)
    "Run3Summer22EE": 0.014,  # 1.4% (2022EE)
    "RunIII2024Summer24": 0.014,  # placeholder until 2024 lumi is finalized
}
