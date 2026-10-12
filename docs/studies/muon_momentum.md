# High-pT muon momentum corrections (Muon POG Run 3 recipe)

`wrcoffea/muon_momentum.py`, switched on per era by `config.yaml` `muon_highpt`
(`era -> payload JSON`): 2022, 2022EE, 2023, 2023BPix (since 2026-10-12) and 2024. Eras not listed (UL,
2025) keep the old TuneP placeholder in `scale_factors.apply_muon_scale_smearing`: a 5 %/TeV × pT scale
shift of both charges and no resolution variation (`musmear` is refused there).

## What it does

All muons use the TuneP transverse momentum, `Muon_pt * Muon_tunepRelPt`, and the TuneP
charge where NanoAOD has it. The regime is chosen on that pT before any correction:

| | data | MC | `muscale` | `musmear` |
| --- | --- | --- | --- | --- |
| pT below the threshold (200 GeV) | MuonScaRe scale | MuonScaRe scale + smearing | MuonScaRe stat. variation | MuonScaRe stat. variation |
| pT above the threshold | GE: q/pT -> q/pT - kappa(eta, phi) | p -> p (1 + N(0, sigma(p, abs(eta)) f)), f = sqrt((1 + x)^2 - 1) | q/pT -/+ kappa uncertainty | smearing fraction x +- syst |

- **Medium pT (MuonScaRe).**
  - Payload: the central Muon POG `muon_scalesmearing.json.gz`, the 2024 copy being the 2026-06-18 release
    (`/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15/`).
  - Code: the Muon POG kit, vendored in `wrcoffea/MuonScaRe.py` (gitlab `cms-muonPOG/muonscarekit`, master 1c7426b5).
  - The kit is calibrated on the PF pT. Below 200 GeV, TuneP takes the tracker-only fit for nearly all high-pT-ID muons
    and then equals the PF pT, so the kit is applied to the TuneP pT (checked on 2024 DY MC and data).
- **GE (generalized endpoint).** A misalignment adds a constant offset kappa to the measured curvature:
  q/pT(measured) = q/pT(true) + kappa, with kappa per (eta, phi) cell.
  - The correction removes it from data. MC carries no kappa.
  - Only the single-muon phi-modulation map ("step 1") is used. The dimuon eta step ("step 2") is not applied, as in
    EXO-25-021.
- **MC smearing above the threshold.**
  - The same Gaussian random number is used for the nominal and for both `musmear` variations, so the variations are
    coherent with the nominal.
  - Draws are deterministic: correctionlib hashprng of (event, luminosityBlock, phi).

## The payload

The kappa map, the resolution parametrisation and the smearing fractions are numbers from CMS-internal documents,
cited below by name and revision. The documents themselves are not in this repository.

- path: `data/muon_highpt/<run>/<year>/<era>.json`, named per era by `config.yaml` `muon_highpt`;
- `tests/test_muon_momentum.py` loads every configured payload through the analyzer's checks.

Anyone with CMS access can rebuild or check it from the sources below. Every number in the payload carries a `source`
or `note` field. To print it laid out like the source figures:

```bash
.venv/bin/python scripts/show_muon_highpt.py data/muon_highpt/Run3/2024/RunIII2024Summer24.json
```

The kappa map is printed as on the slides: eta along x, phi along y with +pi at the top, value +- uncertainty in each
cell. The resolution coefficients are printed as p0..p3, as in the fit boxes.

## Internal sources, 2024 (RunIII2024Summer24)

Documents:

- **[talk]** L. Thomas for EXO-25-021, "High pT muons", Muon POG meeting, 2026-03-22, 24 slides:
  indico event 1665380, contribution 7001589, `highptmuons.pdf`.
  - The [POG] page links it in "High pT: Momentum Scale". Its 2024/2025 link points to `#page=12`, but in the 24-slide
    file the GE maps are on slide 11; slide 12 compares with the medium-pT biases.
- **[AN]** AN-2025/107 v11 (2026-09-03), the EXO-25-021 analysis note. Page numbers below are PDF pages; the printed
  page is 2 lower.
- **[twiki]** TWiki EXO25021, as of 2026-10-09; page numbers refer to a print of that date.
- **[POG]** Muon POG "Muon correction recommendations" (muon-wiki.docs.cern.ch, CERN login).

| Payload entry | Taken from | How to check |
| --- | --- | --- |
| `ge_kappa.kappa`, `kappa_unc` | [talk] slide 11, "1. Single muon (phi modulation): results", panel "2024". Identical to [AN] Fig. 24, p. 38 | all 60 cells, value and uncertainty, against `show_muon_highpt.py`. The numbers are in the PDF text layer (`pdftotext -bbox-layout -f 11 -l 11`); each value has its "± unc" directly below it |
| `ge_kappa.eta_edges`, `phi_edges` | [talk] slide 11 axes | the slide 10 example fits land in the same cells |
| sign convention: kappa is the data bias, data corrected by q/pT - kappa | [talk] slide 9 (the bias as an additive constant on q/pT, injected in MC to match data); slide 10 (template shifted by kappa); slide 17 (applied to data); [POG] High pT: Momentum Scale (method description; Run 3: correct the pT with the GE values) | the forward 2022EFG cells of slide 11 agree in sign with the [POG] 2022 postEE table |
| step 2 not applied | [AN] l. 625-631, p. 37 (step 2 itself: [talk] slide 16, [AN] Fig. 27) | |
| `threshold_gev` | [POG] High pT notes; [AN] l. 501 | |
| `mc_smearing.sigma_barrel`, `sigma_endcap` | [AN] Fig. 18, p. 33: the fit boxes of the barrel-barrel and endcap-endcap panels | **The text formula under the figure (l. 553) misprints the endcap constant term by a factor 10 against the fit box; the payload uses the fit box.** |
| `mc_smearing.sigma_p_max_gev` | [AN] Fig. 18 x range (the fitted p_gen range) | our choice: sigma is held at its value at the end of the fitted range, not extrapolated |
| `mc_smearing.extra_smear` | [AN] l. 565-567, p. 33 (2024 data/MC Z-peak widths); [talk] slide 7 (MUO approval for 2024); [twiki] p. 3 (the MUO conveners asked for a 2024 derivation) | the [POG] page itself still lists 2024 high-pT resolution as "Work In Progress" |
| `mc_smearing.syst` | [POG] High pT: Momentum Resolution (systematic shift for all periods) | f(x) as defined on the [POG] page |
| `mc_smearing.eta_split` | [AN] Fig. 18 and [POG]: barrel / endcap boundary | |

## Our own choices (not from the sources)

- `musmear` up/down use the smearing fractions x + syst and x - syst with the same random number. The [POG] gives only
  the "+syst" form.
- `muscale` above the threshold shifts the MC q/pT by -/+ the per-cell kappa uncertainty (statistical only).
  EXO-25-021 instead uses a mass-scale nuisance ([AN] Table 17).
  - The [POG] also asks for a sanity check of signal muons above 500 GeV, shifting pT by the measured biases.
- A curvature shift that would flip the charge (pT above about 1/abs(kappa)) leaves that muon unchanged. In practice
  this affects only mismeasured TuneP tracks.

## 2022 and 2023 (Run3Summer22, Run3Summer22EE, Run3Summer23, Run3Summer23BPix)

Payloads `data/muon_highpt/Run3/2022/` and `Run3/2023/`, written 2026-10-12 from the [POG] page alone:

| Payload entry | Taken from |
| --- | --- |
| `ge_kappa.kappa`, `kappa_unc` | [POG] High pT: Momentum Scale, Run 3 tab, the four "kappa_b maps in (phi, eta)" tables (3 phi × 6 eta bins; value and the uncertainty in parentheses). The tables have phi rows and eta columns; the payload stores kappa[eta][phi]. |
| `mc_smearing.sigma_barrel`, `sigma_endcap` | [POG] High pT: Momentum Resolution, Run 3 tab, the era's two sigma(p, eta) formulas. |
| `mc_smearing.extra_smear` | Same section: the width factor per region, f = sqrt((1 + x)^2 - 1), so barrel / endcap x = 0 / 0.05 (2022), 0.05 / 0.10 (2022EE, 2023), 0 / 0.15 (2023BPix). A barrel x of 0 leaves the barrel MC unsmeared. |
| `mc_smearing.syst` | Same section: a 10 % shift for barrel and endcap, all periods. |

Our choices: sigma is held at its 2.5 TeV value above 2.5 TeV, as for 2024; MuonScaRe below 200 GeV uses the era's
central `muon_scalesmearing.json.gz`. [talk] slide 11 also shows newer, finer GE maps for 2022 and 2023 (the ones
used for 2024); the [POG] page's own 2022/2023 tables, its recommendation for those eras, are used instead.

## Other eras

- UL: the [POG] page has Run 2 GE tables and sigma(p) parametrisations, but below 200 GeV UL would need a
  Rochester-type correction, which is not in the repository.
- 2025 would need its own payload: EXO-25-021 uses 2024 MC with a larger smearing ([AN] l. 573-577), with no MUO
  approval statement for it; the [POG] lists the 2025 recipe as work in progress.
