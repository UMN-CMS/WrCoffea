# Repository review — 2026-09-07

This records the repository review of the analyzer, corrections, systematic
templates, and their connection to the plotter. It includes the working-tree
changes reviewed alongside WrCoffea commit `56a7ed8807abe8d2ec409ec0545646bd311b5fe4`
and WR_Plotter commit `52b48edb7783b2c1a14241753497fd7a80c038f3`. The commits alone
do not reproduce the reviewed state: several improvements were still uncommitted.

The repository is usable for development and validation runs. The open findings
below must be resolved or explicitly justified for the affected channels and eras
before treating the yields and uncertainties as final. Priorities reflect
potential impact; this review did not measure the size of each physics effect.

## Open findings

### R1 — High: muon momentum treatment remains incomplete

**Scope:** Muon channels and mixed-flavor regions.

In `apply_muon_scale_smearing`, the presence of `tunepRelPt` selects the TuneP
branch and returns before the MuonScaRe code. That branch uses a hardcoded
5%/TeV scale variation and sets both resolution variations equal to nominal.
The era-dependent high-pT scale and resolution prescription discussed during
the review has not been implemented.

Using TuneP is not itself the finding. The missing calibration and uncertainty
treatment is the issue; the choice of a combined medium/high-pT approach or a
high-pT approach throughout remains a separate analysis decision.

**To close:** Agree and document the applicable prescription, implement the
required nominal treatment and uncertainties with validated era inputs, and
check the resulting yields and mass shapes. Producing `MuScale` and `MuSmear`
labels alone does not establish that their contents are physically complete.

**Evidence:** [scale_factors.py](../wrcoffea/scale_factors.py),
`apply_muon_scale_smearing`; [systematics.py](../wrcoffea/systematics.py),
the `muscale` and `musmear` entries.

### R2 — High: boosted LSF efficiency correction is missing

**Scope:** Boosted regions that require the LSF selection.

The analyzer requires `FatJet.lsf3 > 0.75`, but does not apply an associated
efficiency SF or uncertainty. The intended absorption of the loose-electron ID
efficiency into that correction is therefore also incomplete. The corrected
assignment of tight SFs to the lead and loose SFs to the selected partner is
already implemented.

**To close:** Obtain or derive an applicable LSF efficiency correction and
uncertainty, establish which regions and lepton flavors it covers, and integrate
it without double-counting the partner's efficiency factors.

**Evidence:** [analyzer.py](../wrcoffea/analyzer.py), `selectAK8Jets_withLSF` and
`_lepton_sfs_for_region`; [config.yaml](../wrcoffea/config.yaml),
`cuts.ak8_lsf3_min`.

### R3 — High: pre-2024 Run 3 HEEP SFs use the UL2018 fallback

**Scope:** Electron-containing regions in the 2022 and 2023 eras.

When the configured electron payload does not supply `HEEPID`, `electron_id_sf`
falls back to flat UL2018 barrel/endcap SFs and uncertainties. It logs a warning;
the fallback is not silent. The 2024 payload provides the actual HEEP working
point. Using the UL2018 values for UL18 is distinct from transferring those
values to Run 3.

**To close:** Replace the Run 3 fallback with applicable era inputs, or document
and validate the transfer and its uncertainty.

**Evidence:** [scale_factors.py](../wrcoffea/scale_factors.py), `electron_id_sf`.

### R4 — Medium: electron trigger corrections remain approximate

**Scope:** Regions selected by the electron triggers.

Run 3 uses the `HLT_SF_Ele30_TightID` efficiencies as a proxy for the full
electron trigger OR. UL18 has no electron trigger payload configured and returns
unity. The event-level OR efficiency calculation does not establish that the
single-path payload describes the full selected trigger combination.

**To close:** Validate or measure the correction for the actual trigger OR and
electron selection, including the boosted leading-electron treatment. Supply
the missing UL18 correction if that era is used for final results.

**Evidence:** [scale_factors.py](../wrcoffea/scale_factors.py),
`electron_trigger_sf`; [config.yaml](../wrcoffea/config.yaml), `electron_jsons`.

### R5 — Medium: plotter configuration omits three produced SF uncertainties

**Scope:** Stack-plot uncertainty bands in affected electron regions.

The analyzer registry produces `ElectronIdSf`, `ElectronTrigSf`, and
`LooseElectronRecoSf`, but the plotter's systematic list contains none of them.
Their templates are consequently omitted from the displayed uncertainty band
even when present in the ROOT file. This finding concerns uncertainty
propagation; the nominal SFs are applied by the analyzer.

**To close:** Add the corresponding plotter names (`electronidsf`,
`electrontrigsf`, `looseelectronrecosf`) and verify that their templates enter the
band. Keep the producer and consumer definitions synchronized, preferably using
the exported manifest or an explicit consistency check. The plotter also falls
back to nominal for missing templates, so it should distinguish a deliberate
omission from an unexpectedly missing applicable variation.

**Evidence:** [analyzer registry](../wrcoffea/systematics.py),
[plotter systematic list](../WR_Plotter/data/systematics.yaml), and
`collect_syst_variations` in
[make_stackplots.py](../WR_Plotter/bin/make_stackplots.py).

### R6 — Medium: required data-quality inputs can fail without stopping a run

**Scope:** Data certification and required event noise filters.

An unreadable golden JSON leaves data unmasked after a warning. Missing required
noise-filter branches are treated as passed. A job can therefore finish while
omitting a required event-quality selection.

**To close:** Make missing or unreadable required inputs fatal for production.
Keep genuinely optional, era-dependent flags explicitly separate, and verify
both the failure cases and normal filtering behavior.

**Evidence:** [analyzer.py](../wrcoffea/analyzer.py), `apply_lumi_mask`,
`_noise_flag`, and `apply_noise_filter`.

### R7 — Remaining uncertainty work: theory variations and luminosity inputs

**Scope:** Final uncertainty models for the affected simulated processes and eras.

The analyzer does not propagate PDF, QCD-scale, or parton-shower weights into
templates, although the skimmer preserves the relevant branches. The 2023 and
2024 luminosity uncertainties are explicitly marked as placeholders in the
configuration. The new coverage checks only enforce sources defined in the
registry; they cannot detect a physics uncertainty that has not been registered.

**To close:** Define the required process-dependent theory uncertainties and
their normalization treatment, propagate them through template production, and
replace the luminosity placeholders with the applicable inputs and correlations.

**Evidence:** [systematics.py](../wrcoffea/systematics.py), `FAMILIES` and source
definitions; [skimmer.py](../wrcoffea/skimmer.py), preserved LHE and PS branches;
[config.yaml](../wrcoffea/config.yaml), `lumi_unc`.

### R8 — Medium, UL18-specific: incomplete electron and jet variations

**Scope:** `RunIISummer20UL18` production.

Electron scale/smearing is unconfigured, and JES/JER shape variations are
explicitly disabled for UL18. Nominal JEC and JER corrections are implemented.
The missing electron trigger correction is covered by R4.

**To close:** Configure and validate the required UL18 electron momentum
treatment and jet variations before using those uncertainty templates in final
results. An explicit `--omit-systs` entry records missing coverage; it does not
supply the missing uncertainty.

**Evidence:** [config.yaml](../wrcoffea/config.yaml), UL18 `electron_jsons`;
[scale_factors.py](../wrcoffea/scale_factors.py),
`apply_electron_scale_smearing` and `apply_jet_corrections`;
[systematics.py](../wrcoffea/systematics.py), `_has_jet_variations`.

## Resolved findings and implemented improvements

| Item | Reviewed behavior |
| --- | --- |
| Jet-veto crash | The veto returns an event mask; the analyzer filters events once with that mask. |
| Unskimmed MC normalization after filtering | Signed generator weights are accumulated before event filtering, and histograms are normalized after accumulation. |
| Extra tight leptons in resolved same-flavor regions | Regions require exactly two total tight leptons. |
| Lepton efficiency SFs applied to data | SF application is guarded by `is_mc`. |
| Boosted lead/partner SF assignment | The lead receives tight SFs and the selected partner receives loose SFs, including when the partner also passes tight. |
| 2024 JER reused the BPix payload | AK4 and AK8 now use `Summer24Prompt24_JRV2`; JEC uses `Summer24Prompt24_V5`. |
| Missing systematic templates during export | The registry and coverage checks require applicable requested variations across chunks and configured regions. |
| Shape migrations between lepton channels | Shape variations rerun selections across regions, including migrations through extra-lepton vetoes. |

These bookkeeping improvements do not close R1–R8. In particular, complete
template coverage does not validate the underlying calibration.

## Intentional limitation

Run3Summer23BPix signal samples are being used as placeholders for 2024 because
2024 signal samples are not yet available to the analysis. The analyzer remaps
their jet ID while other corrections follow the configured 2024 era. Treat
results using these samples as provisional; the campaign mismatch remains
relevant when interpreting efficiencies and systematic studies.

## Repository structure and maintenance

The separation between configuration, regions, histogram filling, systematic
definitions, and export is useful. The shared systematic registry and coverage
manifest improve consistency between the analyzer and writer.

The main maintenance tasks are to reduce the size of `analyzer.py` and
`scale_factors.py`, remove obsolete commented implementations, update stale
docstrings, and avoid duplicated systematic definitions in the plotter. These
are lower priority than the physics and integration findings above. The new
registry and its tests were untracked at review time and must be included with
the related changes when committing a reproducible update.

## Validation recorded during the review

- The main repository test suite passed: **435 passed in 30.46 s**, using
  `.venv/bin/python -B -m pytest -p no:cacheprovider tests/ -q --tb=short` with
  bytecode writes disabled and the Numba cache directed outside the repository.
- Direct evaluations of the real 2024 AK4 and AK8 JER payloads returned finite,
  ordered nominal/up/down values at representative eta and pT points.
- Comparing the analyzer's SF source names with the plotter configuration
  confirmed the three omissions listed in R5.

These are results from the reviewed snapshot, not a test of later working-tree
changes. The review did not establish physics closure, a complete end-to-end
production validation, or final limit-setting readiness. The review itself made
no changes; this document and its README link were added afterward to record it.
