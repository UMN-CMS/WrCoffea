"""Tests for wrcoffea.analysis_config — config consistency and YAML loading."""

from pathlib import Path

from wrcoffea.analysis_config import (
    LUMIS, LUMI_UNC, LUMI_JSONS,
    MUON_JSONS, ELECTRON_JSONS, ELECTRON_SF_ERA_KEYS,
    CUTS, _CONFIG_PATH,
)


class TestYamlLoading:
    def test_config_yaml_exists(self):
        assert _CONFIG_PATH.exists(), f"config.yaml not found at {_CONFIG_PATH}"

    def test_config_lives_in_package_dir(self):
        assert _CONFIG_PATH.parent == Path(__file__).resolve().parent.parent / "wrcoffea"

    def test_cuts_loaded_from_yaml(self):
        assert "lepton_pt_min" in CUTS
        assert "dr_min" in CUTS


class TestConfigConsistency:
    def test_lumi_unc_covers_all_lumi_eras(self):
        """Every era with a luminosity should have an uncertainty defined."""
        for era in LUMIS:
            assert era in LUMI_UNC, f"LUMI_UNC missing era '{era}'"

    def test_electron_sf_era_keys_covers_electron_jsons(self):
        """Every era in ELECTRON_JSONS should have a correctionlib key."""
        for era in ELECTRON_JSONS:
            assert era in ELECTRON_SF_ERA_KEYS, (
                f"ELECTRON_SF_ERA_KEYS missing era '{era}'"
            )

    def test_cuts_values_are_numeric(self):
        for key, val in CUTS.items():
            assert isinstance(val, (int, float)), f"CUTS['{key}'] is {type(val)}"

    def test_lepton_pt_thresholds_consistent(self):
        assert CUTS["sublead_lepton_pt_min"] <= CUTS["lead_lepton_pt_min"]
        assert CUTS["lepton_pt_min"] <= CUTS["sublead_lepton_pt_min"]

    def test_mll_thresholds_ordered(self):
        assert CUTS["mll_dy_low"] < CUTS["mll_dy_high"]
        assert CUTS["mll_dy_high"] < CUTS["mll_sr_min"]
        assert CUTS["mll_sr_min"] < CUTS["mll_sr_high_min"]
