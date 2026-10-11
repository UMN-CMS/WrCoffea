"""Flat LSF scale factor (AN-19-083 Table 60) on the boosted SR / flavour CR."""

from __future__ import annotations

from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest

from wrcoffea.analysis_config import LSF_SF
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.regions import BOOSTED_REGIONS, RESOLVED_REGIONS
from wrcoffea.scale_factors import lsf_sf
from wrcoffea.systematics import NOMINAL, expected_labels, systematics_for, validate_requested

UL18 = "RunIISummer20UL18"
RUN3 = "Run3Summer22"           # Run 3: unit placeholder with a 10 % uncertainty (config.yaml lsf_sf, 2026-10-11)
UNMAPPED = "Run3Summer99"       # no analysis era is left without an entry; a made-up one stands in

# AN-19-083 Table 60, 2018 column.
E_JET = (1.05, 1.05 + 0.07, 1.05 - 0.06)
MU_JET = (1.04, 1.04 + 0.06, 1.04 - 0.05)


class TestValues:
    def test_config_holds_the_2018_table(self):
        assert UL18 in LSF_SF
        nom, up, down = lsf_sf(UL18, "electron", 3)
        assert (nom[0], up[0], down[0]) == pytest.approx(E_JET)
        nom, up, down = lsf_sf(UL18, "muon", 3)
        assert (nom[0], up[0], down[0]) == pytest.approx(MU_JET)
        assert nom.shape == (3,)

    @pytest.mark.parametrize("era,e,mu", [
        ("RunIISummer20UL16APV", (1.04, 0.09, 0.08), (1.01, 0.06, 0.06)),
        ("RunIISummer20UL16", (1.04, 0.09, 0.08), (1.01, 0.06, 0.06)),
        ("RunIISummer20UL17", (1.02, 0.08, 0.08), (0.98, 0.07, 0.07)),
        ("RunIISummer20UL18", (1.05, 0.07, 0.06), (1.04, 0.06, 0.05)),
    ])
    def test_config_holds_the_run2_table_per_year(self, era, e, mu):
        """AN-19-083 Table 60 (injected-lepton method): one row per year, both UL16 eras share 2016."""
        for flavour, (v, up_err, down_err) in (("electron", e), ("muon", mu)):
            nom, up, down = lsf_sf(era, flavour, 2)
            assert (nom[0], up[0], down[0]) == pytest.approx((v, v + up_err, v - down_err))
        assert set(LSF_SF) >= {"RunIISummer20UL16APV", "RunIISummer20UL16", "RunIISummer20UL17", "RunIISummer20UL18"}

    def test_run3_placeholder_is_unity_with_ten_percent(self):
        for era in ("Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix", "RunIII2024Summer24"):
            for flavour in ("electron", "muon"):
                nom, up, down = lsf_sf(era, flavour, 2)
                assert (nom[0], up[0], down[0]) == pytest.approx((1.0, 1.1, 0.9))

    def test_unconfigured_era_returns_none(self):
        assert lsf_sf(UNMAPPED, "electron", 2) is None

    def test_unknown_flavour_is_an_error(self):
        with pytest.raises(KeyError, match="flavour"):
            lsf_sf(UL18, "tau", 1)


class TestRegionMapping:
    @pytest.mark.parametrize("region, flavor", [
        ("wr_ee_boosted_sr", "electron"),
        ("wr_mumu_boosted_sr", "muon"),
        ("wr_emu_boosted_flavor_cr", "muon"),      # leading electron, muon in the jet
        ("wr_mue_boosted_flavor_cr", "electron"),  # leading muon, electron in the jet
        ("wr_ee_boosted_sr_nomass", "electron"),
        ("wr_ee_boosted_dy_cr", None),
        ("wr_mumu_boosted_dy_cr", None),
        ("wr_ee_resolved_sr", None),
        ("wr_resolved_flavor_cr", None),
    ])
    def test_jet_lepton_flavor(self, region, flavor):
        assert WrAnalysis._jet_lepton_flavor(region) == flavor

    def test_every_nominal_region_is_classified(self):
        for region in list(RESOLVED_REGIONS) + list(BOOSTED_REGIONS):
            expects_lsf = "boosted" in region and "dy_cr" not in region
            assert (WrAnalysis._jet_lepton_flavor(region) is not None) == expects_lsf, region


class TestRegistry:
    def test_family_and_availability(self):
        (src,) = systematics_for(UL18, ["lsf"])
        assert src.name == "LsfSf" and src.kind == "weight"
        assert src.available(UL18) and src.available(RUN3) and not src.available(UNMAPPED)
        with pytest.raises(ValueError, match="Cannot produce"):
            validate_requested(UNMAPPED, ["lsf"])

    def test_applies_to_boosted_sr_and_flavor_cr_only(self):
        for region in list(RESOLVED_REGIONS) + list(BOOSTED_REGIONS):
            labels = expected_labels(UL18, ["lsf"], region, is_mc=True)
            expects = "boosted" in region and "dy_cr" not in region
            assert (labels == ["LsfSfUp", "LsfSfDown"]) == expects, region
            assert expected_labels(UL18, ["lsf"], region, is_mc=False) == []


class TestAnalyzerWiring:
    N = 4

    @staticmethod
    def _identity_patches():
        ones = np.ones(TestAnalyzerWiring.N)
        triple = (ones, ones.copy(), ones.copy())
        return [
            patch("wrcoffea.analyzer.muon_sf", return_value={"reco": triple, "id": triple, "iso": triple}),
            patch("wrcoffea.analyzer.muon_sf_loose", return_value={"reco": triple, "id": triple}),
            patch("wrcoffea.analyzer.muon_trigger_sf", return_value=triple),
            patch("wrcoffea.analyzer.electron_reco_sf", return_value=triple),
            patch("wrcoffea.analyzer.electron_id_sf", return_value=triple),
            patch("wrcoffea.analyzer.electron_trigger_sf", return_value=triple),
        ]

    def _run(self, proc, region, era):
        dummy = ak.Array([[1.0]] * self.N)
        patches = self._identity_patches()
        for p in patches:
            p.start()
        try:
            return proc._lepton_sfs_for_region(
                region, {NOMINAL: np.ones(self.N)}, era, dummy, dummy,
                loose_muons=dummy, loose_electrons=dummy, is_mc=True,
            )
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.parametrize("region, expected", [
        ("wr_ee_boosted_sr", E_JET),
        ("wr_mumu_boosted_sr", MU_JET),
        ("wr_emu_boosted_flavor_cr", MU_JET),
        ("wr_mue_boosted_flavor_cr", E_JET),
    ])
    def test_ul18_boosted_regions_carry_the_factor(self, region, expected):
        proc = WrAnalysis(mass_point=None, enabled_systs=["lsf"], region="boosted")
        out = self._run(proc, region, UL18)
        np.testing.assert_allclose(out[NOMINAL], expected[0])
        np.testing.assert_allclose(out["LsfSfUp"], expected[1])
        np.testing.assert_allclose(out["LsfSfDown"], expected[2])

    def test_dy_cr_and_resolved_are_untouched(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=["lsf"], region="both")
        for region in ("wr_ee_boosted_dy_cr", "wr_mumu_resolved_sr"):
            out = self._run(proc, region, UL18)
            np.testing.assert_allclose(out[NOMINAL], 1.0)
            assert "LsfSfUp" not in out

    def test_run3_placeholder_leaves_the_nominal_and_adds_the_variation(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=["lsf", "sf"], region="boosted")
        out = self._run(proc, "wr_ee_boosted_sr", RUN3)
        np.testing.assert_allclose(out[NOMINAL], 1.0)
        np.testing.assert_allclose(out["LsfSfUp"], 1.1)
        np.testing.assert_allclose(out["LsfSfDown"], 0.9)

    def test_unmapped_era_has_no_factor(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=["sf"], region="boosted")
        out = self._run(proc, "wr_ee_boosted_sr", UNMAPPED)
        np.testing.assert_allclose(out[NOMINAL], 1.0)
        assert "LsfSfUp" not in out

    def test_nominal_applied_without_the_family(self):
        """The factor is a correction; the family only adds the variation."""
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="boosted")
        out = self._run(proc, "wr_ee_boosted_sr", UL18)
        np.testing.assert_allclose(out[NOMINAL], E_JET[0])
        assert set(out) == {NOMINAL}

    def test_lsf_family_does_not_produce_lepton_sf_variations(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=["lsf"], region="boosted")
        out = self._run(proc, "wr_ee_boosted_sr", UL18)
        assert set(out) == {NOMINAL, "LsfSfUp", "LsfSfDown"}
