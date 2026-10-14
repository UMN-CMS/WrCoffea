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
    SEL_MLLJJ_LT800, SEL_MLJ_LT800_BOOSTED,
    SEL_EE_SR_NOLSF, SEL_MUMU_SR_NOLSF, SEL_EMU_CR_NOLSF, SEL_MUE_CR_NOLSF,
    SEL_MLL_GT200_BOOSTED_NOLSF, SEL_MLJ_GT800_BOOSTED_NOLSF,
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
    """Derive ``<region><suffix>`` from each nominal *region* by removing *drop* cuts,
    swapping each cut of *replace* (``(cut, replacement)`` pairs) for its replacement,
    and AND-ing *add* cuts.

    Added and replacement cuts are selections the analyzer only fills when a variant
    asks for them (see :func:`added_selections`).  *blinded* False marks a variant of a
    signal region that cannot overlap it (its selection excludes the SR), which blinded
    data still builds (see :func:`is_signal_region`).
    """
    suffix: str
    drop: frozenset[str]
    regions: tuple[str, ...]
    description: str = ""
    add: tuple[str, ...] = ()
    replace: tuple[tuple[str, str], ...] = ()
    blinded: bool = True


_DY_CRS: tuple[str, ...] = ("wr_ee_resolved_dy_cr", "wr_mumu_resolved_dy_cr",
                             "wr_ee_boosted_dy_cr", "wr_mumu_boosted_dy_cr")
_SR_AND_FLAVOR_CRS: tuple[str, ...] = (
    "wr_ee_resolved_sr", "wr_mumu_resolved_sr", "wr_resolved_flavor_cr",
    "wr_ee_boosted_sr", "wr_mumu_boosted_sr",
    "wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr",
)

REGION_VARIANTS: dict[str, RegionVariant] = {
    # SR and flavor CR without the reconstructed-mass requirement.  The mass
    # histograms span 0-12000 GeV, so one region yields both the m < 800 GeV
    # closure region and the m > 800 GeV SR as slices at plot time (simulation:
    # the SR variants stay blinded in data).
    "nomass": RegionVariant(
        suffix="_nomass",
        drop=frozenset({SEL_MLLJJ_GT800, SEL_MLJ_GT800_BOOSTED}),
        regions=_SR_AND_FLAVOR_CRS,
        description="SR / flavor CR with the m(lljj) or m(lJ) > 800 GeV cut removed.",
    ),
    # AN-19-083 App. J (Figs. 103-128), its low-mass region LowWRCR: the SR and flavor CR
    # selections with m(lljj) or m(lJ) < 800 GeV, where the data-driven tt prediction
    # (R x flavor-CR data) is compared with data.  Disjoint from the SR (m > 800 GeV), so
    # data fills it, as in Run 2.
    "lowmass": RegionVariant(
        suffix="_lowmass",
        drop=frozenset(),
        replace=((SEL_MLLJJ_GT800, SEL_MLLJJ_LT800), (SEL_MLJ_GT800_BOOSTED, SEL_MLJ_LT800_BOOSTED)),
        regions=_SR_AND_FLAVOR_CRS,
        blinded=False,
        description="SR / flavor CR with m(lljj) or m(lJ) < 800 GeV instead of > 800 GeV (AN-19-083 LowWRCR).",
    ),
    # AN-19-083 Figs. 12, 93-94: the boosted SR and flavor CRs with the jet taken without
    # the LSF requirement (the leading AK8 jet with |dphi| > 2, as in the DY CR), the
    # lepton-in-jet candidates and masses rebuilt on it.  The SR variants stay blinded.
    "nolsf": RegionVariant(
        suffix="_nolsf",
        drop=frozenset(),
        replace=(
            (SEL_EE_SR, SEL_EE_SR_NOLSF), (SEL_MUMU_SR, SEL_MUMU_SR_NOLSF),
            (SEL_EMU_CR, SEL_EMU_CR_NOLSF), (SEL_MUE_CR, SEL_MUE_CR_NOLSF),
            (SEL_AK8JETS_WITH_LSF, SEL_ATLEAST1AK8_DPHI_GT2),
            (SEL_MLL_GT200_BOOSTED, SEL_MLL_GT200_BOOSTED_NOLSF),
            (SEL_MLJ_GT800_BOOSTED, SEL_MLJ_GT800_BOOSTED_NOLSF),
        ),
        regions=("wr_ee_boosted_sr", "wr_mumu_boosted_sr", "wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr"),
        description="Boosted SR / flavor CR on the leading AK8 jet without the LSF requirement (AN-19-083 Fig. 12).",
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
    # above 800 GeV (AN-19-083 Appendix J closure test), and its closure in data
    # in the low-mass region.  Measured on ttbar, so pair with ``--split-groups tt_tW``.
    "transfer-factor": ("nomass", "lowmass"),
    # DY reshape CR1 -> CR2 closure (AN-19-083 Figs. 27-30).  Run data, the
    # non-DY backgrounds and DY with it, derive the reshape from the
    # _mll60to100 regions (scripts/derive_dy_reshape.py --region-suffix) and
    # compare the _mll100to150 regions before and after applying it.
    "dy-cr-split": ("dycr_mll60to100", "dycr_mll100to150"),
    # LSF3 of the boosted jet without the LSF requirement, data in the flavor CRs
    # (AN-19-083 Fig. 12) and simulation in the SRs (the S/sqrt(B) scan, Figs. 93-94).
    "no-lsf": ("nolsf",),
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
            if not var.drop and not var.add and not var.replace:
                raise ValueError(f"Variant '{name}' neither drops, replaces nor adds a cut.")
            if var.drop and not (var.drop & set(cuts)):
                raise ValueError(f"Variant '{name}' drops nothing from region '{region}'.")
            replaced = {old for old, _ in var.replace}
            if var.replace and not (replaced & set(cuts)):
                raise ValueError(f"Variant '{name}' replaces nothing in region '{region}'.")
            if set(var.add) & set(cuts):
                raise ValueError(f"Variant '{name}' adds a cut region '{region}' already has.")
            if {new for old, new in var.replace if old in cuts} & (set(cuts) - replaced):
                raise ValueError(f"Variant '{name}' replaces a cut of region '{region}' by one it already has.")
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
            out[name] = selections.all(*variant_cuts(var, cuts))
    return out


def variant_cuts(var: RegionVariant, cuts: RegionCuts) -> RegionCuts:
    """The cuts of *var*'s version of a region with *cuts*: drop, replace, then add."""
    swap = dict(var.replace)
    return tuple(swap.get(c, c) for c in cuts if c not in var.drop) + var.add


def added_selections(variants: Iterable[str]) -> frozenset[str]:
    """Selection names the requested *variants* add or swap in, which the analyzer must then fill."""
    out = set()
    for v in validate_variants(variants):
        var = REGION_VARIANTS[v]
        out.update(var.add)
        out.update(new for _, new in var.replace)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------

# Data of a blinded run (the default) never builds these regions, nor any
# variant derived from them (e.g. ``wr_ee_resolved_sr_nomass``): no
# histograms, no --save-events trees, no coverage entry.  The exception is a
# variant marked ``blinded=False`` because its selection excludes the SR (the
# m < 800 GeV ``_lowmass`` regions).  The flavor and DY control regions are
# never blinded, and MC is never blinded.
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
    """True for a signal region or any blinded region variant derived from one."""
    if name in SIGNAL_REGIONS:
        return True
    return any(name == region + var.suffix
               for var in REGION_VARIANTS.values() if var.blinded
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
