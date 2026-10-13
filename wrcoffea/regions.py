"""Analysis region definitions, region variants, the study catalogue, and blinding.

Every region is a tuple of ``PackedSelection`` names that are AND-ed
together.  The nominal regions are listed once here and evaluated by the
analyzer.  A *region variant* derives extra regions from nominal ones by
dropping named cuts and/or AND-ing extra ones; a *study* is a named set of
variants exposed on the command line (``--study``).

The blinding policy lives here as well: which regions are signal regions,
where blinded data cutflows stop, and which runs may be unblinded.

Studies that change object reconstruction or add new selections still
need dedicated analyzer code; this registry does not cover them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from wrcoffea.analysis_config import (
    SEL_TWO_TIGHT_ELECTRONS, SEL_TWO_TIGHT_MUONS, SEL_TWO_TIGHT_EM, SEL_NO_THIRD_TIGHT_LEPTON,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53, SEL_MIN_TWO_AK4_JETS,
    SEL_E_TRIGGER, SEL_MU_TRIGGER, SEL_EMU_TRIGGER,
    SEL_DR_ALL_PAIRS_GT0P4, SEL_HEM_VETO, SEL_60_MLL_150, SEL_MLL_GT200, SEL_MLL_GT400, SEL_MLLJJ_GT800,
    SEL_MLL_60_100, SEL_MLL_100_150,
    SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_DYCR_MASK,
    SEL_ATLEAST1AK8_DPHI_GT2, SEL_AK8JETS_WITH_LSF,
    SEL_MUMU_DYCR, SEL_EE_DYCR, SEL_MUMU_SR, SEL_EE_SR, SEL_EMU_CR, SEL_MUE_CR,
    SEL_MLL_GT200_BOOSTED, SEL_MLJ_GT800_BOOSTED,
)
from wrcoffea.era_utils import get_era_details

RegionCuts = tuple[str, ...]

# ---------------------------------------------------------------------------
# Nominal regions
# ---------------------------------------------------------------------------

_RESOLVED_EE_BASE: RegionCuts = (
    SEL_TWO_TIGHT_ELECTRONS, SEL_NO_THIRD_TIGHT_LEPTON,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
    SEL_E_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4, SEL_HEM_VETO,
)
_RESOLVED_MUMU_BASE: RegionCuts = (
    SEL_TWO_TIGHT_MUONS, SEL_NO_THIRD_TIGHT_LEPTON,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
    SEL_MU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4, SEL_HEM_VETO,
)
_RESOLVED_EM_BASE: RegionCuts = (
    SEL_TWO_TIGHT_EM, SEL_NO_THIRD_TIGHT_LEPTON,
    SEL_LEAD_TIGHT_PT60, SEL_SUBLEAD_TIGHT_PT53,
    SEL_EMU_TRIGGER, SEL_MIN_TWO_AK4_JETS, SEL_DR_ALL_PAIRS_GT0P4, SEL_HEM_VETO,
)

RESOLVED_REGIONS: dict[str, RegionCuts] = {
    "wr_ee_resolved_dy_cr":   _RESOLVED_EE_BASE   + (SEL_60_MLL_150, SEL_MLLJJ_GT800),
    "wr_mumu_resolved_dy_cr": _RESOLVED_MUMU_BASE + (SEL_60_MLL_150, SEL_MLLJJ_GT800),
    "wr_resolved_flavor_cr":  _RESOLVED_EM_BASE   + (SEL_MLL_GT400, SEL_MLLJJ_GT800),
    "wr_ee_resolved_sr":      _RESOLVED_EE_BASE   + (SEL_MLL_GT400, SEL_MLLJJ_GT800),
    "wr_mumu_resolved_sr":    _RESOLVED_MUMU_BASE + (SEL_MLL_GT400, SEL_MLLJJ_GT800),
}

# The HEM veto (SEL_HEM_VETO in the resolved bases) is folded into the
# per-region masks (SEL_MUMU_DYCR ...) for the boosted regions, because the
# sub-leading candidate it inspects differs per region.
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
    """Derive ``<region><suffix>`` from each nominal *region* by removing *drop* cuts
    and AND-ing *add* cuts.

    Added cuts are selections the analyzer only fills when a variant asks for them
    (see :func:`added_selections`).
    """
    suffix: str
    drop: frozenset[str]
    regions: tuple[str, ...]
    description: str = ""
    add: tuple[str, ...] = ()


_DY_CRS: tuple[str, ...] = ("wr_ee_resolved_dy_cr", "wr_mumu_resolved_dy_cr",
                             "wr_ee_boosted_dy_cr", "wr_mumu_boosted_dy_cr")

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
    # AN-19-083 Sec. 5.1 / Figs. 27-30: derive the DY reshape in DY CR1
    # (60 < m(ll) < 100) and test it in DY CR2 (100 < m(ll) < 150).
    "dycr_mll60to100": RegionVariant(
        suffix="_mll60to100",
        drop=frozenset(),
        add=(SEL_MLL_60_100,),
        regions=_DY_CRS,
        description="DY CR restricted to 60 < m(ll) < 100 GeV (AN-19-083 DY CR1).",
    ),
    "dycr_mll100to150": RegionVariant(
        suffix="_mll100to150",
        drop=frozenset(),
        add=(SEL_MLL_100_150,),
        regions=_DY_CRS,
        description="DY CR restricted to 100 < m(ll) < 150 GeV (AN-19-083 DY CR2).",
    ),
}

# Study name -> variants it enables.  Exposed as ``--study`` on the CLI.
STUDIES: dict[str, tuple[str, ...]] = {
    # e-mu transfer factor R = SR / flavor CR, fitted separately below and
    # above 800 GeV (AN-19-083 Appendix J closure test).  Measured on ttbar,
    # so pair with ``--split-groups tt_tW``.
    "transfer-factor": ("nomass",),
    # DY reshape CR1 -> CR2 closure (AN-19-083 Figs. 27-30).  Run data, the
    # non-DY backgrounds and DY with it, derive the reshape from the
    # _mll60to100 regions (scripts/derive_dy_reshape.py --region-suffix) and
    # compare the _mll100to150 regions before and after applying it.
    "dy-cr-split": ("dycr_mll60to100", "dycr_mll100to150"),
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
            if not var.drop and not var.add:
                raise ValueError(f"Variant '{name}' neither drops nor adds a cut.")
            if var.drop and not (var.drop & set(cuts)):
                raise ValueError(f"Variant '{name}' drops nothing from region '{region}'.")
            if set(var.add) & set(cuts):
                raise ValueError(f"Variant '{name}' adds a cut region '{region}' already has.")
            new_name = region + var.suffix
            if new_name in _ALL_NOMINAL or new_name in generated:
                raise ValueError(f"Variant '{name}' generates duplicate region name '{new_name}'.")
            generated.add(new_name)
    return requested


REGION_MODES = ("resolved", "boosted", "both")


def region_names(mode: str, variants: Iterable[str] = (), *, blind: bool = False) -> list[str]:
    """Names of every region a run in *mode* with *variants* must produce.

    Mirrors :func:`build_regions` without needing selections, so the writer
    can demand each region independently of what a chunk happened to fill.
    With *blind* the signal regions and their variants are left out, as data
    of a blinded run never builds them.
    """
    mode = (mode or "both").lower()
    if mode not in REGION_MODES:
        raise ValueError(f"Invalid region mode '{mode}'. Must be one of {REGION_MODES}.")
    topologies = []
    if mode in ("resolved", "both"):
        topologies.append(RESOLVED_REGIONS)
    if mode in ("boosted", "both"):
        topologies.append(BOOSTED_REGIONS)
    out: list[str] = []
    for nominal in topologies:
        out.extend(nominal)
        for vname in sorted(validate_variants(variants)):
            var = REGION_VARIANTS[vname]
            out.extend(region + var.suffix for region in var.regions if region in nominal)
    if blind:
        out = [name for name in out if not is_signal_region(name)]
    return out


def build_regions(nominal: Mapping[str, RegionCuts], selections, variants: Iterable[str] = (),
                  *, blind: bool = False):
    """Evaluate nominal regions plus the requested variants on *selections*.

    Returns an ordered ``{region_name: boolean mask}`` dict.  Variant regions
    are appended after the nominal ones, in registry order.  With *blind* the
    signal regions and their variants are not evaluated at all.
    """
    out = {name: selections.all(*cuts) for name, cuts in nominal.items()
           if not (blind and is_signal_region(name))}
    for vname in sorted(validate_variants(variants)):
        var = REGION_VARIANTS[vname]
        for region in var.regions:
            cuts = nominal.get(region)
            if cuts is None:
                continue  # belongs to the other topology
            name = region + var.suffix
            if blind and is_signal_region(name):
                continue
            kept = tuple(c for c in cuts if c not in var.drop) + var.add
            out[name] = selections.all(*kept)
    return out


def added_selections(variants: Iterable[str]) -> frozenset[str]:
    """Selection names the requested *variants* add, which the analyzer must then fill."""
    return frozenset(c for v in validate_variants(variants) for c in REGION_VARIANTS[v].add)


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------

# Data of a blinded run (the default) never builds these regions, nor any
# variant derived from them (e.g. ``wr_ee_resolved_sr_nomass``): no
# histograms, no --save-events trees, no coverage entry.  The flavor and DY
# control regions are never blinded, and MC is never blinded.
SIGNAL_REGIONS: frozenset[str] = frozenset({
    "wr_ee_resolved_sr", "wr_mumu_resolved_sr",
    "wr_ee_boosted_sr", "wr_mumu_boosted_sr",
})

# Cutflow kind -> flavor -> first step withheld from blinded data.  The chain
# is truncated before it (a shorter cut axis, not zeroed bins), so the
# steps closing in on the signal region are absent.  Flavors not listed
# (em, the flavor CR) keep their full chain.
BLINDED_CUTFLOW_STEPS: dict[str, dict[str, str]] = {
    "resolved": {"ee": SEL_MLL_GT200, "mumu": SEL_MLL_GT200},
    "boosted": {"ee": SEL_MLL_GT200_BOOSTED, "mumu": SEL_MLL_GT200_BOOSTED},
}

# Runs whose data may be unblinded (--unblind).  Every other run stays blind.
UNBLINDABLE_RUNS: frozenset[str] = frozenset({"RunII"})


def is_signal_region(name: str) -> bool:
    """True for a signal region or any region variant derived from one."""
    if name in SIGNAL_REGIONS:
        return True
    return any(name == region + var.suffix
               for var in REGION_VARIANTS.values()
               for region in var.regions if region in SIGNAL_REGIONS)


def blinded_region_names(mode: str, variants: Iterable[str] = ()) -> list[str]:
    """Signal regions (incl. variants) a blinded run in *mode* withholds from data."""
    return [name for name in region_names(mode, variants) if is_signal_region(name)]


def blind_cutflow_steps(kind: str, flavor: str, steps: Sequence[str]) -> list[str]:
    """Return a data cutflow chain truncated before its first blinded step.

    *kind* is ``"resolved"`` or ``"boosted"``.  Chains without an entry in
    :data:`BLINDED_CUTFLOW_STEPS` are returned unchanged.  A blinded chain
    that lacks its stop step raises, so a renamed or reordered chain can
    never silently stay unblinded.
    """
    steps = list(steps)
    stop = BLINDED_CUTFLOW_STEPS.get(kind, {}).get(flavor)
    if stop is None:
        return steps
    if stop not in steps:
        raise ValueError(f"Blinded {kind} cutflow '{flavor}' has no step '{stop}' to stop before: {steps}")
    return steps[:steps.index(stop)]


def check_unblind_allowed(era: str) -> None:
    """Raise unless the data of *era* may be unblinded (Run 2 eras only)."""
    run, _, _ = get_era_details(era)
    if run not in UNBLINDABLE_RUNS:
        raise ValueError(
            f"--unblind is not allowed for {era} ({run}): its data signal regions stay blind. "
            f"Only {', '.join(sorted(UNBLINDABLE_RUNS))} eras may be unblinded."
        )
