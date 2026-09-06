"""Analysis region definitions, region variants, and the study catalogue.

Every region is a tuple of ``PackedSelection`` names that are AND-ed
together.  The nominal regions are listed once here and evaluated by the
analyzer.  A *region variant* derives extra regions from nominal ones by
dropping named cuts (cut relaxation only); a *study* is a named set of
variants exposed on the command line (``--study``).

Studies that change object reconstruction or add new selections still
need dedicated analyzer code; this registry does not cover them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from wrcoffea.analysis_config import (
    SEL_TWO_TIGHT_LEPTONS, SEL_TWO_TIGHT_ELECTRONS, SEL_TWO_TIGHT_MUONS, SEL_TWO_TIGHT_EM,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53, SEL_MIN_TWO_AK4_JETS,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_60_MLL_150, SEL_MLL_GT400, SEL_MLLJJ_GT800,
    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
    SEL_ATLEAST1AK8_DPHI_GT2, SEL_AK8JETS_WITH_LSF,
    SEL_MUMU_DYCR, SEL_EE_DYCR, SEL_MUMU_SR, SEL_EE_SR, SEL_EMU_CR, SEL_MUE_CR,
    SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED,
)

RegionCuts = tuple[str, ...]

# ---------------------------------------------------------------------------
# Nominal regions
# ---------------------------------------------------------------------------

_RESOLVED_EE_BASE: RegionCuts = (
    SEL_TWO_TIGHT_LEPTONS, SEL_TWO_TIGHT_ELECTRONS,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
    SEL_E_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
)
_RESOLVED_MUMU_BASE: RegionCuts = (
    SEL_TWO_TIGHT_LEPTONS, SEL_TWO_TIGHT_MUONS,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
    SEL_MU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
)
_RESOLVED_EM_BASE: RegionCuts = (
    SEL_TWO_TIGHT_LEPTONS, SEL_TWO_TIGHT_EM,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
    SEL_EMU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4,
)

RESOLVED_REGIONS: dict[str, RegionCuts] = {
    "wr_ee_resolved_dy_cr":   _RESOLVED_EE_BASE   + (SEL_60_MLL_150, SEL_MLLJJ_GT800),
    "wr_mumu_resolved_dy_cr": _RESOLVED_MUMU_BASE + (SEL_60_MLL_150, SEL_MLLJJ_GT800),
    "wr_resolved_flavor_cr":  _RESOLVED_EM_BASE   + (SEL_MLL_GT400, SEL_MLLJJ_GT800),
    "wr_ee_resolved_sr":      _RESOLVED_EE_BASE   + (SEL_MLL_GT400, SEL_MLLJJ_GT800),
    "wr_mumu_resolved_sr":    _RESOLVED_MUMU_BASE + (SEL_MLL_GT400, SEL_MLLJJ_GT800),
}

# The DY CRs keep their mass requirement inside SEL_DYCR_MASK because the DY
# candidate mass uses a different AK8 jet and may include the loose lepton.
# The SR / flavor-CR selections carry no mass cuts; those are the standalone
# SEL_MLL_GT200_BOOSTED / SEL_MLJ_GT800_BOOSTED selections.
_BOOSTED_COMMON: RegionCuts = (SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED)
_BOOSTED_MASS: RegionCuts = (SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED)

BOOSTED_REGIONS: dict[str, RegionCuts] = {
    "wr_mumu_boosted_dy_cr":     _BOOSTED_COMMON + (SEL_DYCR_MASK, SEL_ATLEAST1AK8_DPHI_GT2, SEL_MUMU_DYCR),
    "wr_mumu_boosted_sr":        _BOOSTED_COMMON + (SEL_MUMU_SR, SEL_AK8JETS_WITH_LSF) + _BOOSTED_MASS,
    "wr_ee_boosted_dy_cr":       _BOOSTED_COMMON + (SEL_DYCR_MASK, SEL_ATLEAST1AK8_DPHI_GT2, SEL_EE_DYCR),
    "wr_ee_boosted_sr":          _BOOSTED_COMMON + (SEL_EE_SR, SEL_AK8JETS_WITH_LSF) + _BOOSTED_MASS,
    "wr_emu_boosted_flavor_cr":  _BOOSTED_COMMON + (SEL_EMU_CR, SEL_AK8JETS_WITH_LSF) + _BOOSTED_MASS,
    "wr_mue_boosted_flavor_cr":  _BOOSTED_COMMON + (SEL_MUE_CR, SEL_AK8JETS_WITH_LSF) + _BOOSTED_MASS,
}

# ---------------------------------------------------------------------------
# Region variants (cut relaxation) and studies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionVariant:
    """Derive ``<region><suffix>`` from each nominal *region* by removing *drop* cuts."""
    suffix: str
    drop: frozenset[str]
    regions: tuple[str, ...]
    description: str = ""


REGION_VARIANTS: dict[str, RegionVariant] = {
    # SR and flavor CR without the reconstructed-mass requirement.  The mass
    # histograms span 0-8000 GeV, so one region yields both the m < 800 GeV
    # closure region and the m > 800 GeV SR as slices at plot time.
    "nomass": RegionVariant(
        suffix="_nomass",
        drop=frozenset({SEL_MLLJJ_GT800, SEL_MLJ_GT800_BOOSTED}),
        regions=(
            "wr_ee_resolved_sr", "wr_mumu_resolved_sr", "wr_resolved_flavor_cr",
            "wr_ee_boosted_sr", "wr_mumu_boosted_sr",
            "wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr",
        ),
        description="SR / flavor CR with the m(lljj) or m(lJ) > 800 GeV cut removed.",
    ),
}

# Study name -> variants it enables.  Exposed as ``--study`` on the CLI.
STUDIES: dict[str, tuple[str, ...]] = {
    # e-mu transfer factor R = SR / flavor CR, fitted separately below and
    # above 800 GeV (AN-19-083 Appendix J closure test).  Measured on ttbar,
    # so pair with ``--split-groups tt_tW``.
    "transfer-factor": ("nomass",),
}

_ALL_NOMINAL: dict[str, RegionCuts] = {**RESOLVED_REGIONS, **BOOSTED_REGIONS}


def resolve_studies(names: Iterable[str]) -> frozenset[str]:
    """Map study names to the set of region-variant names they enable."""
    requested = [str(n).strip() for n in (names or []) if str(n).strip()]
    unknown = sorted(set(requested) - set(STUDIES))
    if unknown:
        raise ValueError(f"Unknown study {unknown}; available: {sorted(STUDIES)}")
    return frozenset(v for n in requested for v in STUDIES[n])


def validate_variants(names: Iterable[str]) -> frozenset[str]:
    """Check variant names, their dropped cuts, and generated-name uniqueness."""
    requested = frozenset(str(n).strip() for n in (names or []) if str(n).strip())
    unknown = sorted(requested - set(REGION_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown region variant {unknown}; available: {sorted(REGION_VARIANTS)}")
    generated: set[str] = set()
    for name in sorted(requested):
        var = REGION_VARIANTS[name]
        for region in var.regions:
            cuts = _ALL_NOMINAL.get(region)
            if cuts is None:
                raise ValueError(f"Variant '{name}' targets unknown region '{region}'.")
            if not (var.drop & set(cuts)):
                raise ValueError(f"Variant '{name}' drops nothing from region '{region}'.")
            new_name = region + var.suffix
            if new_name in _ALL_NOMINAL or new_name in generated:
                raise ValueError(f"Variant '{name}' generates duplicate region name '{new_name}'.")
            generated.add(new_name)
    return requested


def build_regions(nominal: Mapping[str, RegionCuts], selections, variants: Iterable[str] = ()):
    """Evaluate nominal regions plus the requested variants on *selections*.

    Returns an ordered ``{region_name: boolean mask}`` dict.  Variant regions
    are appended after the nominal ones, in registry order.
    """
    out = {name: selections.all(*cuts) for name, cuts in nominal.items()}
    for vname in sorted(validate_variants(variants)):
        var = REGION_VARIANTS[vname]
        for region in var.regions:
            cuts = nominal.get(region)
            if cuts is None:
                continue  # belongs to the other topology
            kept = tuple(c for c in cuts if c not in var.drop)
            out[region + var.suffix] = selections.all(*kept)
    return out
