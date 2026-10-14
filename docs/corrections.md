# Corrections and their payloads

What this page answers: which correction each era applies, from which payload, where the payload
comes from, and what is a placeholder. The mapping is `wrcoffea/config.yaml`; the files are under
`data/jsonpog/<POG>/<Run>/<era>/` (POG JSONs), `data/reweights/` (our own payloads),
`data/muon_highpt/` and `data/pdf_sets/`. How each correction enters the event weight is in
[analysis.md](analysis.md). Verified on 2026-10-12.

## Where POG payloads come from, and how to check them

The authority is the CAT metadata area on cvmfs,
`/cvmfs/cms-griddata.cern.ch/cat/metadata/<POG>/<campaign>/latest/` (readable at the LPC), with
one campaign per era (`Run3-22CDSep23-Summer22-NanoAODv12`, `Run3-22EFGSep23-Summer22EE-NanoAODv12`,
`Run3-23CSep23-Summer23-NanoAODv12`, `Run3-23DSep23-Summer23BPix-NanoAODv12`,
`Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15`, `Run3-25Prompt-Summer24-NanoAODv15`, and the
UL campaigns). The older `jsonpog-integration` mirror stopped in September 2025 and is stale. Compare
by the sha256 of the canonical JSON, not of the gzip file. Last full check, all ten eras:
2026-10-11; every file identical to CAT except the 2024 `EGM/electron.json.gz`, which is CAT's plus
a HEEP working point CAT does not ship (kept on purpose).

## Per era

Era columns: the four UL eras, then 2022, 2022EE, 2023, 2023BPix, 2024, 2025.

### Jets

| Era | JEC (MC / data) | JER | Jet veto map | Veto rule | Jet ID |
| --- | --- | --- | --- | --- | --- |
| UL16APV | Summer19UL16APV_V7 | Summer20UL16APV_JRV5 | Summer19UL16_V1 | drop the jet | `jetId` lepton-veto bit |
| UL16 | Summer19UL16_V7 | Summer20UL16_JRV5 | Summer19UL16_V1 | drop the jet | `jetId` lepton-veto bit |
| UL17 | Summer19UL17_V5 | Summer19UL17_JRV4 | Summer19UL17_V1 | drop the jet | `jetId` lepton-veto bit |
| UL18 | Summer19UL18_V5 | Summer19UL18_JRV3 | Summer19UL18_V1 | drop the jet | `jetId` lepton-veto bit |
| 2022 | Summer22_22Sep2023_V4 | Summer22_22Sep2023_JRV2 | Summer22_23Sep2023_RunCD_V1 | reject the event | JME repair of the v12 `jetId` |
| 2022EE | Summer22EE_22Sep2023_V4 | Summer22EE_22Sep2023_JRV2 | Summer22EE_23Sep2023_RunEFG_V1 | reject the event | JME repair of the v12 `jetId` |
| 2023 | Summer23Prompt23_V4 | Summer23Prompt23_RunCv1234_JRV3 | Summer23Prompt23_RunC_V1 | reject the event | JME repair of the v12 `jetId` |
| 2023BPix | Summer23BPixPrompt23_V4 | Summer23BPixPrompt23_RunD_JRV3 | Summer23BPixPrompt23_RunD_V1 | reject the event | JME repair of the v12 `jetId` |
| 2024 | Summer24Prompt24_V5 | Summer24Prompt24_JRV2 | Summer24Prompt24_RunBCDEFGHI_V1 | reject the event | `jetid.json.gz` |
| 2025 | Summer24Prompt25_V3 | Summer24Prompt25_JRV2 | Summer24Prompt25_RunCDEFG_V1 | reject the event | `jetid.json.gz` (NanoAODv15 has no `Jet_jetId`) |

Jet ID (JME Jet ID page, 2026-10-12): AK4 tight with lepton veto, AK8 tight. NanoAODv15 (2024,
2025) has no `jetId` and uses the CAT `jetid.json.gz`. NanoAODv12 (2022, 2023) stores a wrong
lepton-veto bit; the JME repair starts from the Tight bit, adds neHEF < 0.99 for 2.7 < |η| ≤ 3.0
and neEmEF < 0.4 above, and builds the lepton veto as muEF < 0.8 and chEmEF < 0.8 for |η| ≤ 2.7
(`ak4_tight_lepton_veto_from_bits` in `analyzer.py`; since 2026-10-12, before: the stored bit; 0.025 %
of the selected 2023 jets differ). UL (NanoAODv9) uses the stored bits.

AK4 is CHS in UL and Puppi in Run 3, AK8 is Puppi everywhere; the JES uncertainty source is `Total`
(`CMS_scale_j_Total`), the UL18 file also ships the regrouped sources. The UL JERC files are the
CAT repackaging (installed 2026-10-11; values identical to the old files except the UL17 JER
resolution, ≤ 3 %). The JME recommendation and the veto-map screenshots are referenced in
`docs/pog_recommendations/` (not tracked).

JER smearing on MC follows the JME hybrid method: a jet matched to a generator jet (ΔR < R/2, i.e.
0.2 for AK4 and 0.4 for AK8, and |pT − pT_gen| < 3 σ_JER pT) is scaled by its JER scale factor
towards the generator pT; any other jet is smeared stochastically by a Gaussian of width
σ_JER √(SF² − 1); no jet is smeared below an energy of 10 MeV. The JER up/down variations use the
same matching and random numbers as the nominal. The AK8 soft-drop mass follows the JME jet mass
recommendation in Run 3: the soft-drop subjets get the AK4 PUPPI JEC (production JEC undone with
`SubJet_rawFactor`; NanoAODv12 has no `SubJet_area`, so L1FastJet is skipped, which is exact for
PUPPI, where it is 1) and, on MC, the AK4 JER with gen subjets (`SubGenJetAK8`, NanoAOD's index in
v15, nearest in ΔR otherwise, matched within 0.2), and m_SD is recomputed from them; JES and JER
vary both subjets; JMS and JMR are the `jms` and `jmr` families. UL keeps NanoAOD's m_SD (no AK4
PUPPI payload). The random numbers of the stochastic JER and of the
Run 3 electron smearing are drawn per object from correctionlib's `hashprng` of (event, luminosity
block, φ, a salt per correction), so they are reproducible and independent between the objects of
an event (`object_normal_draws` in `wrcoffea/scale_factors.py`; since 2026-10-12).

### Pileup

`puWeights.json.gz` per era with the golden-JSON profile: `Collisions16_UltraLegacy_goldenJSON`,
`Collisions17_UltraLegacy_goldenJSON`, `Collisions18_UltraLegacy_goldenJSON`,
`Collisions2022_355100_357900_eraBCD_GoldenJson`, `Collisions2022_359022_362760_eraEFG_GoldenJson`,
`Collisions2023_366403_369802_eraBC_GoldenJson`, `Collisions2023_369803_370790_eraD_GoldenJson`,
`Collisions24_BCDEFGHI_goldenJSON` (`puWeights_BCDEFGHI.json.gz`), and for the 2024 MC used with 2025
data `Collisions25_goldenJSON` (`puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz`).

### Electrons

| Correction | UL16APV, UL16, UL17, UL18 | 2022, 2022EE, 2023, 2023BPix | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| reco SF | `electron.json.gz`, `UL-Electron-ID-SF`, RecoBelow20 / RecoAbove20 | `electron.json.gz`, `Electron-ID-SF`, Reco20to75 / RecoAbove75, clipped at 500 GeV | same | 2025 file |
| HEEP ID SF | flat per year (barrel / endcap): 2016 0.985 / 0.990, 2017 0.979 / 0.987, 2018 0.973 / 0.980 | flat central EGM values: 2022 pre-EE 1.007 / 1.019, post-EE 0.997 / 0.997, 2023 pre-BPix 1.006 / 0.988, post-BPix 1.009 / 0.988 (the 2022/2023 payloads have no HEEP working point) | the `HEEPID` working point of the 2024 payload, binned in pT, clipped at 1 TeV | 2025 file, HEEPID |
| trigger SF | the Run 2 HEEP trigger-OR efficiency per year, converted from the Run 2 analyzer (`electronHlt_WREGammaTrigger_run2legacy_v4.json.gz`, 2016 shared by both UL16 eras) | `electronHlt.json.gz`, `HLT_SF_Ele30_TightID` as a proxy for the OR | same | **placeholder**: the 2024 file |
| scale and smearing | none (UL NanoAOD is already corrected; no variation inputs) | `electronSS_EtDependent.json.gz` (scale on data, smearing on MC, with variations) | same | 2025 file |
| loose electron (boosted partner) | reco SF only; the loose-ID efficiency is in the LSF SF | same | same | same |

Above the last measured bin the EGM payloads hold a 1 ± 20 % placeholder; the analyzer clips reco
SFs to 500 GeV and HEEP ID SFs to 1 TeV instead of using it (open item in analysis.md).

### Muons

| Correction | UL16APV, UL16, UL17, UL18 | 2022, 2022EE, 2023, 2023BPix | 2024 | 2025 |
| --- | --- | --- | --- | --- |
| reco, ID, isolation SFs | `muon_HighPt.json.gz` (global-muon reco vs \|η\| and p, HighPt ID, loose tracker isolation) | same | same | 2025 file |
| trigger SF | the leading muon's SF from `muon_HighPt.json.gz` (`NUM_HLT_DEN_HighPtLooseRelIsoProbes`); two SFs are never multiplied | per-leg data/MC efficiency OR from `muon_Z.json.gz`, entry `NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose` (our exact OR with the HighPt ID and loose tk-iso denominator; variation since 2026-10-12: each muon's data efficiency moved by the POG per-muon SF uncertainty, `stat` and the fit variations `AltBkg`, `AltSig`, `massBin`, `massRange`, `tagIso` in quadrature, MC efficiency nominal, as the payload's efficiency-level `syst_*eff` are zero) | per-leg efficiency OR from `ScaleFactors_Muon_highPt_HLT_2024_schemaV2.json` | per-leg OR from the 2025 high-pT file |
| momentum | TuneP; **placeholder** scale uncertainty (5 %/TeV × pT), no resolution variation (`musmear` refused) | `data/muon_highpt/Run3/<year>/<era>.json` (since 2026-10-12): MuonScaRe below 200 GeV; above, the GE κ maps of the Muon POG Run 3 tables on data and the POG's MC smearing, barrel / endcap 0 / 5 % (2022), 5 / 10 % (2022EE, 2023), 0 / 15 % (2023BPix) | `data/muon_highpt/Run3/2024/RunIII2024Summer24.json`: MuonScaRe (`muon_scalesmearing.json.gz`) below 200 GeV, the generalized-endpoint correction on data and extra smearing on MC above ([studies/muon_momentum.md](studies/muon_momentum.md)) | placeholder as UL (no 2025 recipe yet) |
| loose muon (boosted partner) | reco and ID SFs, no isolation | same | same | same |

The reco SF's last bin is open-ended (p > 1.5 TeV; 2024 p > 0.9 TeV) with era-dependent central
values; see the open items in analysis.md. Every 2022, 2023 and 2024 muon file is the CAT release of
2026-06-18 (the 2022/2023 scale-smearing parameters were re-derived by the POG on 2026-04-28); the 2025
files come from the era's own CAT campaign, `Run3-25Prompt-Summer24-NanoAODv15` (2025 data with the
Summer24 MC), its release of 2026-04-28.

### Other event-level corrections

| Correction | Eras | Payload / value |
| --- | --- | --- |
| L1 prefiring | UL16APV, UL16, UL17, UL18 | NanoAOD `L1PreFiringWeight` Nom/Up/Dn (about −3 % on 2017 DY; the muon part only in 2018) |
| HEM 15/16 veto | UL18 | electrons with η < −1.25, −1.62 < φ < −0.82, data and MC, whole year |
| LSF SF (e / μ) | UL16APV, UL16 | 1.04 (+0.09 −0.08) / 1.01 ± 0.06 (AN-19-083 Table 60) |
| | UL17 | 1.02 ± 0.08 / 0.98 ± 0.07 |
| | UL18 | 1.05 (+0.07 −0.06) / 1.04 (+0.06 −0.05) |
| | 2022 to 2025 | **placeholder** 1.00 ± 0.10 per flavour until a Run 3 measurement exists |
| DY Z-pT reweight | all | `data/reweights/RunII/<year>/<era>/dy_zpt_reweight_ul_20261011.json` per UL era; `data/reweights/Run3/dy_zpt_reweight_run3_2022_2023_20261011.json` (2022, 2022EE, 2023, 2023BPix); `data/reweights/Run3/2024/RunIII2024Summer24/dy_zpt_reweight_run3_20261013.json` (2024, 2025; re-derived 2026-10-13, identical to the 20261011 file) |
| DY jet-pT reshape | all but 2025 | `dy_reshape_run2jetpt_20261011.json` per UL era; `data/reweights/Run3/2022/`, `Run3/2023/` per data-taking year (`_20261011`); 2024 `Run3/2024/RunIII2024Summer24/dy_reshape_run2jetpt_20261013.json` (from the 2026-10-13 rerun, within 1 % of the 20261011 file) |
| signal k-factor | all | `signal_kfactor_13tev_20261010.json` (UL), `signal_kfactor_13p6tev_20261010.json` (Run 3): per-point k(m(ℓN)) curves, [studies/signal_kfactor.md](studies/signal_kfactor.md) |
| PDF sets | all | `data/pdf_sets/<Run>/<year>/<era>.json` (`scripts/pdf_sets.py`): the LHA ID of each sample's `LHEPdfWeight` (NNPDF3.1 325300/325500 everywhere; UL also 306000 symmhessian+α_S and 320900 replicas); 2025 uses the 2024 table |

The DY payloads named `_20261011`, UL and Run 3, and the signal k-factor payloads of 2026-10-10
are derivations of this repository (`scripts/derive_dy_zpt_reweight.py`,
`scripts/derive_dy_reshape.py`, `scripts/signal_kfactor/`); the DY ones follow the recipe of
AN-19-083. The Run 3 reshapes were derived on 2026-10-12 from the 2026-10-11 production. The Run 2
payloads converted from the Run 2 analyzer (`scripts/convert_run2_*.py`) are kept next to the UL
ones for comparison.

## Placeholders at a glance

| What | Where | Until |
| --- | --- | --- |
| LSF SF 1.00 ± 0.10 | Run 3 | a Run 3 measurement |
| Electron trigger SF = Ele30 TightID proxy | Run 3 | a tag-and-probe measurement of the OR |
| Electron trigger SF = the 2024 file | 2025 | a 2025 EGM HLT payload |
| TuneP momentum with a 5 %/TeV × pT scale shift, no resolution variation | UL, 2025 | the high-pT recipe payloads of those eras |
| EGM 1 ± 20 % above 500 GeV (reco) / 1 TeV (HEEP), clipped away | Run 3 | a decision on the high-pT treatment |
| No electron scale/smearing variations | UL | a prescription (see analysis.md) |
