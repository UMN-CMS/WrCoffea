"""Tests for wrcoffea.save_hists — ROOT naming and histogram summing."""

import pytest

from wrcoffea.save_hists import _normalize_syst_name, _folder_and_hist_names


class TestNormalizeSystName:
    def test_nominal_passthrough(self):
        # Nominal is handled upstream, but if called directly:
        assert _normalize_syst_name("Nominal") == "nominal"

    def test_camelcase(self):
        assert _normalize_syst_name("LumiUp") == "lumiup"

    def test_mixed_with_special_chars(self):
        assert _normalize_syst_name("RenFact_Scale-Up!") == "renfactscaleup"


class TestFolderAndHistNames:
    def test_nominal(self):
        folder, hname = _folder_and_hist_names("resolved_dy_cr", "Nominal", "pt_leading_lepton")
        assert folder == "resolved_dy_cr"
        assert hname == "pt_leading_lepton_resolved_dy_cr"

    def test_systematic(self):
        folder, hname = _folder_and_hist_names("resolved_dy_cr", "LumiUp", "pt_leading_lepton")
        assert folder == "syst_lumiup_resolved_dy_cr"
        assert hname == "pt_leading_lepton_syst_lumiup_resolved_dy_cr"

    def test_systematic_down(self):
        folder, hname = _folder_and_hist_names("wr_ee_resolved_sr", "LumiDown", "mass_dilepton")
        assert folder == "syst_lumidown_wr_ee_resolved_sr"
        assert hname == "mass_dilepton_syst_lumidown_wr_ee_resolved_sr"
