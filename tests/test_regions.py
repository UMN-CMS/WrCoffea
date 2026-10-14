"""Tests for wrcoffea.regions: nominal tuples, variants, studies, and the 800 GeV boundary."""

import numpy as np
import pytest
from coffea.analysis_tools import PackedSelection

from wrcoffea import regions as R
from wrcoffea.analysis_config import SEL_MLLJJ_GT800, SEL_MLJ_GT800_BOOSTED
from wrcoffea.histograms import _booking_specs, create_hist


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
    assert R.resolve_studies(["transfer-factor"]) == frozenset({"nomass", "lowmass"})
    assert R.resolve_studies([]) == frozenset()
    assert R.resolve_studies(["transfer-factor", "transfer-factor"]) == frozenset({"nomass", "lowmass"})
    assert R.resolve_studies(["no-lsf"]) == frozenset({"nolsf"})
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
        bins, label = _booking_specs()[name]
        h = create_hist(name, bins, label)
        h.fill(process="p", region="r_nomass", syst="Nominal", **{name: [800.0]}, weight=[1.0])
        edges = h.axes[name].edges
        i800 = int(np.searchsorted(edges, 800.0))
        assert edges[i800] == 800.0
        vals = h[{"process": "p", "region": "r_nomass", "syst": "Nominal"}].values()
        assert vals[:i800].sum() == 0.0
        assert vals[i800:].sum() == 1.0


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------

SIGNAL_REGIONS = {"wr_ee_resolved_sr", "wr_mumu_resolved_sr", "wr_ee_boosted_sr", "wr_mumu_boosted_sr"}


def test_is_signal_region_covers_srs_and_their_variants_only():
    assert R.SIGNAL_REGIONS == SIGNAL_REGIONS
    for region in SIGNAL_REGIONS:
        assert R.is_signal_region(region)
        assert R.is_signal_region(region + "_nomass")
    for region in ("wr_ee_resolved_dy_cr", "wr_mumu_boosted_dy_cr", "wr_resolved_flavor_cr",
                   "wr_resolved_flavor_cr_nomass", "wr_emu_boosted_flavor_cr",
                   "wr_mue_boosted_flavor_cr_nomass", "wr_ee_resolved_sr_unknown", "wr_ee_resolved"):
        assert not R.is_signal_region(region), region


@pytest.mark.parametrize("mode", R.REGION_MODES)
@pytest.mark.parametrize("variants", [(), ("nomass",), ("lowmass", "nolsf")])
def test_region_names_blind_drops_only_signal_regions(mode, variants):
    full = R.region_names(mode, variants)
    blind = R.region_names(mode, variants, blind=True)
    assert blind == [name for name in full if not R.is_signal_region(name)]
    assert R.blinded_region_names(mode, variants) == [name for name in full if R.is_signal_region(name)]
    assert sorted(blind + R.blinded_region_names(mode, variants)) == sorted(full)
    if variants:
        # Variants of the flavor CRs stay; variants of the SRs go.
        flavor_cr_variants = [name for name in full if "_flavor_cr_" in name]
        assert flavor_cr_variants and set(flavor_cr_variants) <= set(blind)
        assert not any(name.endswith("_sr_nomass") for name in blind)


@pytest.mark.parametrize("nominal", [R.RESOLVED_REGIONS, R.BOOSTED_REGIONS])
def test_build_regions_blind_skips_signal_regions(nominal):
    sel = _selection_with(_all_names(nominal))
    full = R.build_regions(nominal, sel, {"nomass"})
    blind = R.build_regions(nominal, sel, {"nomass"}, blind=True)
    assert list(blind) == [name for name in full if not R.is_signal_region(name)]
    assert any(R.is_signal_region(name) for name in full)
    for name, mask in blind.items():
        np.testing.assert_array_equal(mask, full[name])
    # The writer's expectation mirrors what the analyzer builds.
    mode = "resolved" if nominal is R.RESOLVED_REGIONS else "boosted"
    assert list(blind) == R.region_names(mode, {"nomass"}, blind=True)


def test_blinded_cutflow_steps_are_the_mll_cuts():
    assert R.BLINDED_CUTFLOW_STEPS == {
        "resolved": {"ee": "mll_gt200", "mumu": "mll_gt200"},
        "boosted": {"ee": "mll_gt200_boosted", "mumu": "mll_gt200_boosted"},
    }


@pytest.mark.parametrize("kind, steps, kept", [
    ("resolved", ["dr_all_pairs_gt0p4", "mlljj_gt800", "mll_gt200", "mll_gt400"],
     ["dr_all_pairs_gt0p4", "mlljj_gt800"]),
    ("boosted", ["no_of_lepton_in_ak8", "mll_gt200_boosted", "mlj_gt800_boosted"],
     ["no_of_lepton_in_ak8"]),
])
def test_blind_cutflow_steps_truncates_ee_mumu_before_the_stop_step(kind, steps, kept):
    for flavor in ("ee", "mumu"):
        assert R.blind_cutflow_steps(kind, flavor, steps) == kept
    # em is the flavor CR and is never blinded.
    assert R.blind_cutflow_steps(kind, "em", steps) == steps
    # The input sequence is not modified and may be any iterable.
    assert R.blind_cutflow_steps(kind, "ee", tuple(steps)) == kept


@pytest.mark.parametrize("kind, flavor", [("resolved", "ee"), ("resolved", "mumu"),
                                          ("boosted", "ee"), ("boosted", "mumu")])
def test_blind_cutflow_steps_without_stop_step_raises(kind, flavor):
    with pytest.raises(ValueError, match="has no step"):
        R.blind_cutflow_steps(kind, flavor, ["no_cuts", "mlljj_gt800", "mll_gt400"])


@pytest.mark.parametrize("era", ["RunIISummer20UL16APV", "RunIISummer20UL16", "RunIISummer20UL17",
                                 "RunIISummer20UL18"])
def test_unblind_allowed_for_run2(era):
    R.check_unblind_allowed(era)


@pytest.mark.parametrize("era", ["Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix",
                                 "RunIII2024Summer24", "RunIII2025Summer25"])
def test_unblind_refused_for_run3(era):
    with pytest.raises(ValueError, match=f"--unblind is not allowed for {era} \\(Run3\\)"):
        R.check_unblind_allowed(era)


def test_unblind_refused_for_unknown_era():
    with pytest.raises(ValueError, match="Unsupported era"):
        R.check_unblind_allowed("Run4Summer30")


# ---------------------------------------------------------------------------
# DY CR m(ll) split (study dy-cr-split, AN-19-083 DY CR1 -> CR2 test)
# ---------------------------------------------------------------------------

def test_dy_cr_split_study_targets_the_four_dy_crs():
    variants = R.resolve_studies(["dy-cr-split"])
    assert variants == frozenset({"dycr_mll60to100", "dycr_mll100to150"})
    assert R.added_selections(variants) == frozenset({R.SEL_MLL_60_100, R.SEL_MLL_100_150})
    assert R.added_selections(R.resolve_studies(["transfer-factor"])) == frozenset({R.SEL_MLLJJ_LT800, R.SEL_MLJ_LT800_BOOSTED})
    names = R.region_names("both", variants)
    for cr in ("wr_ee_resolved_dy_cr", "wr_mumu_resolved_dy_cr", "wr_ee_boosted_dy_cr", "wr_mumu_boosted_dy_cr"):
        assert cr + "_mll60to100" in names and cr + "_mll100to150" in names
        assert not R.is_signal_region(cr + "_mll60to100")
    # blinded data still builds the split control regions
    assert set(R.region_names("both", variants, blind=True)) >= {n for n in names if "_mll" in n}


@pytest.mark.parametrize("nominal", [R.RESOLVED_REGIONS, R.BOOSTED_REGIONS])
def test_dy_cr_split_halves_partition_the_dy_cr(nominal):
    """Event 0 is in CR1, event 1 in CR2: each half keeps the nominal cuts and adds its own."""
    names = _all_names(nominal)
    sel = PackedSelection()
    for name in names:
        sel.add(name, np.array([True, True]))
    sel.add(R.SEL_MLL_60_100, np.array([True, False]))
    sel.add(R.SEL_MLL_100_150, np.array([False, True]))
    masks = R.build_regions(nominal, sel, R.resolve_studies(["dy-cr-split"]))
    for cr in (r for r in nominal if "dy_cr" in r):
        assert masks[cr].tolist() == [True, True]
        assert masks[cr + "_mll60to100"].tolist() == [True, False]
        assert masks[cr + "_mll100to150"].tolist() == [False, True]


def test_add_only_variant_must_add_something_new(monkeypatch):
    empty = R.RegionVariant(suffix="_e", drop=frozenset(), regions=("wr_ee_resolved_dy_cr",))
    monkeypatch.setitem(R.REGION_VARIANTS, "empty", empty)
    with pytest.raises(ValueError, match="neither drops, replaces nor adds"):
        R.validate_variants({"empty"})
    again = R.RegionVariant(suffix="_a", drop=frozenset(), add=(R.SEL_60_MLL_150,),
                            regions=("wr_ee_resolved_dy_cr",))
    monkeypatch.setitem(R.REGION_VARIANTS, "again", again)
    with pytest.raises(ValueError, match="already has"):
        R.validate_variants({"again"})


# ---------------------------------------------------------------------------
# Replacement variants: lowmass (AN-19-083 App. J) and nolsf (Fig. 12)
# ---------------------------------------------------------------------------

def test_lowmass_variant_swaps_the_mass_cut_in_both_topologies():
    var = R.REGION_VARIANTS["lowmass"]
    res = R.variant_cuts(var, R.RESOLVED_REGIONS["wr_ee_resolved_sr"])
    assert R.SEL_MLLJJ_LT800 in res and SEL_MLLJJ_GT800 not in res
    boo = R.variant_cuts(var, R.BOOSTED_REGIONS["wr_mue_boosted_flavor_cr"])
    assert R.SEL_MLJ_LT800_BOOSTED in boo and SEL_MLJ_GT800_BOOSTED not in boo
    assert len(res) == len(R.RESOLVED_REGIONS["wr_ee_resolved_sr"])


def test_lowmass_regions_are_built_by_blinded_data_and_the_sr_is_not():
    """m < 800 GeV excludes the SR, so blinded data builds every _lowmass region."""
    for region in ("wr_ee_resolved_sr", "wr_mumu_boosted_sr"):
        assert not R.is_signal_region(region + "_lowmass")
    blind = R.region_names("both", ["lowmass"], blind=True)
    assert all(r + "_lowmass" in blind for r in R.REGION_VARIANTS["lowmass"].regions)
    assert not set(R.SIGNAL_REGIONS) & set(blind)


def test_event_below_800_enters_lowmass_and_not_the_sr():
    names = _all_names(R.RESOLVED_REGIONS) + [R.SEL_MLLJJ_LT800]
    sel = _selection_with(names, false={SEL_MLLJJ_GT800})
    masks = R.build_regions(R.RESOLVED_REGIONS, sel, {"lowmass"})
    assert not masks["wr_ee_resolved_sr"][0]
    assert masks["wr_ee_resolved_sr_lowmass"][0]


def test_nolsf_variant_rebuilds_the_boosted_sr_and_flavor_crs():
    var = R.REGION_VARIANTS["nolsf"]
    for region, mask, new in (("wr_ee_boosted_sr", R.SEL_EE_SR, R.SEL_EE_SR_NOLSF),
                              ("wr_mue_boosted_flavor_cr", R.SEL_MUE_CR, R.SEL_MUE_CR_NOLSF)):
        cuts = R.variant_cuts(var, R.BOOSTED_REGIONS[region])
        assert new in cuts and mask not in cuts
        assert R.SEL_AK8JETS_WITH_LSF not in cuts and R.SEL_ATLEAST1AK8_DPHI_GT2 in cuts
        assert R.SEL_MLL_GT200_BOOSTED_NOLSF in cuts and R.SEL_MLJ_GT800_BOOSTED_NOLSF in cuts
        assert R.SEL_MLL_GT200_BOOSTED not in cuts and SEL_MLJ_GT800_BOOSTED not in cuts
    assert "wr_ee_boosted_dy_cr" not in var.regions


def test_nolsf_srs_stay_blind_and_its_flavor_crs_do_not():
    assert R.is_signal_region("wr_ee_boosted_sr_nolsf")
    assert R.is_signal_region("wr_mumu_boosted_sr_nolsf")
    assert not R.is_signal_region("wr_emu_boosted_flavor_cr_nolsf")
    assert not R.is_signal_region("wr_mue_boosted_flavor_cr_nolsf")


def test_added_selections_include_replacement_cuts():
    added = R.added_selections({"nolsf"})
    assert {R.SEL_EE_SR_NOLSF, R.SEL_MUMU_SR_NOLSF, R.SEL_EMU_CR_NOLSF, R.SEL_MUE_CR_NOLSF,
            R.SEL_MLL_GT200_BOOSTED_NOLSF, R.SEL_MLJ_GT800_BOOSTED_NOLSF} <= added


def test_replace_variant_must_replace_a_cut_of_each_region(monkeypatch):
    bad = R.RegionVariant(suffix="_x", drop=frozenset(), regions=("wr_ee_resolved_dy_cr",),
                          replace=((R.SEL_EE_SR, "something_else"),))
    monkeypatch.setitem(R.REGION_VARIANTS, "bad", bad)
    with pytest.raises(ValueError, match="replaces nothing"):
        R.validate_variants({"bad"})
