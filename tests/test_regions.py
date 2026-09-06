"""Tests for wrcoffea.regions: nominal tuples, variants, studies, and the 800 GeV boundary."""

import numpy as np
import pytest
from coffea.analysis_tools import PackedSelection

from wrcoffea import regions as R
from wrcoffea.analysis_config import SEL_MLLJJ_GT800, SEL_MLJ_GT800_BOOSTED
from wrcoffea.histograms import create_hist


def _selection_with(names, n=1, false=()):
    sel = PackedSelection()
    for name in names:
        sel.add(name, np.full(n, name not in false, dtype=bool))
    return sel


def _all_names(nominal):
    return sorted({c for cuts in nominal.values() for c in cuts})


def test_nominal_regions_are_unique_and_variant_names_do_not_collide():
    names = list(R.RESOLVED_REGIONS) + list(R.BOOSTED_REGIONS)
    assert len(names) == len(set(names))
    assert R.validate_variants(R.REGION_VARIANTS) == frozenset(R.REGION_VARIANTS)


def test_validate_variants_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown region variant"):
        R.validate_variants({"bogus"})


def test_validate_variants_rejects_variant_that_drops_nothing(monkeypatch):
    bad = R.RegionVariant(suffix="_x", drop=frozenset({"not_a_cut"}), regions=("wr_ee_resolved_sr",))
    monkeypatch.setitem(R.REGION_VARIANTS, "bad", bad)
    with pytest.raises(ValueError, match="drops nothing"):
        R.validate_variants({"bad"})


def test_validate_variants_rejects_duplicate_generated_names(monkeypatch):
    dup = R.RegionVariant(suffix="", drop=frozenset({SEL_MLLJJ_GT800}), regions=("wr_ee_resolved_sr",))
    monkeypatch.setitem(R.REGION_VARIANTS, "dup", dup)
    with pytest.raises(ValueError, match="duplicate region name"):
        R.validate_variants({"dup"})


def test_resolve_studies_maps_names_and_rejects_unknown():
    assert R.resolve_studies(["transfer-factor"]) == frozenset({"nomass"})
    assert R.resolve_studies([]) == frozenset()
    assert R.resolve_studies(["transfer-factor", "transfer-factor"]) == frozenset({"nomass"})
    with pytest.raises(ValueError, match="Unknown study"):
        R.resolve_studies(["no-such-study"])


@pytest.mark.parametrize("nominal,mass_cut", [
    (R.RESOLVED_REGIONS, SEL_MLLJJ_GT800),
    (R.BOOSTED_REGIONS, SEL_MLJ_GT800_BOOSTED),
])
def test_nomass_variant_targets_sr_and_flavor_cr_only(nominal, mass_cut):
    var = R.REGION_VARIANTS["nomass"]
    targets = [r for r in var.regions if r in nominal]
    assert targets, "variant must cover this topology"
    for name, cuts in nominal.items():
        if name in targets:
            assert mass_cut in cuts
            assert "dy_cr" not in name
        else:
            assert "dy_cr" in name


@pytest.mark.parametrize("nominal,mass_cut", [
    (R.RESOLVED_REGIONS, SEL_MLLJJ_GT800),
    (R.BOOSTED_REGIONS, SEL_MLJ_GT800_BOOSTED),
])
def test_event_failing_mass_cut_enters_nomass_but_not_nominal(nominal, mass_cut):
    """An event at (or below) 800 GeV fails the strict > 800 cut, so it is in
    the ``_nomass`` region and not in the nominal SR / flavor CR."""
    sel = _selection_with(_all_names(nominal), false={mass_cut})
    masks = R.build_regions(nominal, sel, {"nomass"})
    for region in R.REGION_VARIANTS["nomass"].regions:
        if region not in nominal:
            continue
        assert not masks[region][0], region
        assert masks[region + "_nomass"][0], region
    # Without variants no derived regions are produced.
    assert set(R.build_regions(nominal, sel)) == set(nominal)


def test_nomass_region_equals_nominal_when_mass_cut_passes():
    sel = _selection_with(_all_names(R.RESOLVED_REGIONS))
    masks = R.build_regions(R.RESOLVED_REGIONS, sel, {"nomass"})
    for region in ("wr_ee_resolved_sr", "wr_mumu_resolved_sr", "wr_resolved_flavor_cr"):
        assert masks[region][0] and masks[region + "_nomass"][0]


def test_mass_exactly_800_lands_in_high_slice_of_mass_histogram():
    """800 GeV is a bin edge of the mass axes, so a value of exactly 800 is
    counted on the high side when the _nomass histogram is sliced there."""
    for name in ("mass_fourobject", "mass_twoobject"):
        h = create_hist(name, (800, 0, 8000), "m")
        h.fill(process="p", region="r_nomass", syst="Nominal", **{name: [800.0]}, weight=[1.0])
        edges = h.axes[name].edges
        i800 = int(np.searchsorted(edges, 800.0))
        assert edges[i800] == 800.0
        vals = h[{"process": "p", "region": "r_nomass", "syst": "Nominal"}].values()
        assert vals[:i800].sum() == 0.0
        assert vals[i800:].sum() == 1.0
