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

    def test_muon_trigger_payloads_are_per_era(self):
        """Each era's muon TRIGGER payload must exist and live in that era's directory."""
        for era, payloads in MUON_JSONS.items():
            trig = payloads.get("TRIGGER")
            assert trig is not None, f"MUON_JSONS['{era}'] has no TRIGGER payload"
            assert Path(trig).exists(), f"missing muon TRIGGER payload: {trig}"
            assert Path(trig).parent.name == era, (
                f"{era} muon TRIGGER payload points at another era's file: {trig}"
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


# ---------------------------------------------------------------------------
# Config overlay (WRCOFFEA_CONFIG_OVERLAY)
# ---------------------------------------------------------------------------

def test_merge_overlay_updates_mappings_and_replaces_scalars():
    from wrcoffea.analysis_config import _merge_overlay
    cfg = {"dy_zpt_reweight": {"A": "a"}, "cuts": {"x": 1, "y": 2}, "flag": 1}
    out = _merge_overlay(cfg, {"dy_zpt_reweight": {"B": "b"}, "cuts": {"y": 3}, "flag": 2})
    assert out == {"dy_zpt_reweight": {"A": "a", "B": "b"}, "cuts": {"x": 1, "y": 3}, "flag": 2}
    assert cfg["dy_zpt_reweight"] == {"A": "a"}          # input untouched


def test_overlay_env_reaches_module_constants(tmp_path):
    import json
    import os
    import subprocess
    import sys
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("dy_zpt_reweight:\n  RunIII2024Summer24: data/reweights/x.json\n")
    code = ("import json, wrcoffea.analysis_config as a; "
            "print(json.dumps({'zpt': a.DY_ZPT_REWEIGHT_JSONS, 'overlay': a.CONFIG_OVERLAY}))")
    env = {**os.environ, "WRCOFFEA_CONFIG_OVERLAY": str(overlay)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["overlay"] == str(overlay)
    assert got["zpt"]["RunIII2024Summer24"].endswith("/data/reweights/x.json")   # resolved like config.yaml
    assert "RunIISummer20UL18" in got["zpt"]                                     # base entries kept
