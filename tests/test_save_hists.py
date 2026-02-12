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


# ---------------------------------------------------------------------------
# Tests for histogram summing and ROOT I/O
# ---------------------------------------------------------------------------


import hist
import numpy as np
import uproot
from unittest.mock import MagicMock

from wrcoffea.save_hists import (
    sum_hists,
    split_hists_with_syst,
    _sum_cutflow_hists,
    save_histograms,
)


class TestSumHists:
    """Test histogram summing across datasets."""

    def _create_test_hist(self, name="test"):
        """Helper to create a test histogram with standard axes."""
        return (
            hist.Hist.new
            .StrCat([], name="process", growth=True)
            .StrCat([], name="region", growth=True)
            .StrCat([], name="syst", growth=True)
            .Reg(100, 0, 5000, name=name)
            .Weight()
        )

    def test_single_dataset(self):
        """Test summing with a single dataset returns copy."""
        h1 = self._create_test_hist("mass_dilepton")
        h1.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)

        my_hists = {"dataset1": {"mass_dilepton": h1}}
        summed = sum_hists(my_hists)

        assert "mass_dilepton" in summed
        assert summed["mass_dilepton"].sum().value == pytest.approx(1.0)

    def test_multiple_datasets_sum_correctly(self):
        """Test that histograms are summed across datasets."""
        h1 = self._create_test_hist("mass_dilepton")
        h1.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=2.0)

        h2 = self._create_test_hist("mass_dilepton")
        h2.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=3.0)

        my_hists = {
            "dataset1": {"mass_dilepton": h1},
            "dataset2": {"mass_dilepton": h2},
        }
        summed = sum_hists(my_hists)

        # Should sum weights: 2.0 + 3.0 = 5.0
        assert summed["mass_dilepton"].sum().value == pytest.approx(5.0, rel=1e-5)

    def test_different_processes_preserved(self):
        """Test that different process categories are preserved."""
        h1 = self._create_test_hist("pt_leading_lepton")
        h1.fill(process="DYJets", region="sr", syst="Nominal", pt_leading_lepton=100.0, weight=1.0)

        h2 = self._create_test_hist("pt_leading_lepton")
        h2.fill(process="tt_tW", region="sr", syst="Nominal", pt_leading_lepton=120.0, weight=2.0)

        my_hists = {
            "dataset1": {"pt_leading_lepton": h1},
            "dataset2": {"pt_leading_lepton": h2},
        }
        summed = sum_hists(my_hists)

        # Both processes should be present
        assert "DYJets" in summed["pt_leading_lepton"].axes["process"]
        assert "tt_tW" in summed["pt_leading_lepton"].axes["process"]

    def test_empty_histograms_raises(self):
        """Test that empty histogram dict raises ValueError."""
        with pytest.raises(ValueError, match="No histogram data provided"):
            sum_hists({})

    def test_multiple_histograms_all_summed(self):
        """Test that all histograms in datasets are summed."""
        h1_mass = self._create_test_hist("mass_dilepton")
        h1_mass.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)

        h1_pt = self._create_test_hist("pt_leading_lepton")
        h1_pt.fill(process="DYJets", region="sr", syst="Nominal", pt_leading_lepton=100.0, weight=2.0)

        h2_mass = self._create_test_hist("mass_dilepton")
        h2_mass.fill(process="tt_tW", region="sr", syst="Nominal", mass_dilepton=60.0, weight=3.0)

        h2_pt = self._create_test_hist("pt_leading_lepton")
        h2_pt.fill(process="tt_tW", region="sr", syst="Nominal", pt_leading_lepton=110.0, weight=4.0)

        my_hists = {
            "dataset1": {"mass_dilepton": h1_mass, "pt_leading_lepton": h1_pt},
            "dataset2": {"mass_dilepton": h2_mass, "pt_leading_lepton": h2_pt},
        }
        summed = sum_hists(my_hists)

        # Both histograms should be present
        assert "mass_dilepton" in summed
        assert "pt_leading_lepton" in summed

        # Verify sums
        assert summed["mass_dilepton"].sum().value == pytest.approx(4.0, rel=1e-5)  # 1+3
        assert summed["pt_leading_lepton"].sum().value == pytest.approx(6.0, rel=1e-5)  # 2+4

    def test_non_hist_values_skipped(self):
        """Test that non-Hist values (like cutflow dicts) are skipped."""
        h1 = self._create_test_hist("mass_dilepton")
        h1.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)

        my_hists = {
            "dataset1": {
                "mass_dilepton": h1,
                "cutflow": {"ee": {}, "mumu": {}},  # Not a Hist
                "_sumw": 12345.0,  # Not a Hist
            }
        }
        summed = sum_hists(my_hists)

        # Only mass_dilepton should be in summed (cutflow and _sumw skipped)
        assert "mass_dilepton" in summed
        assert "cutflow" not in summed
        assert "_sumw" not in summed


class TestSplitHistsWithSyst:
    """Test histogram splitting by region and systematic."""

    def _create_test_hist(self, name="test"):
        """Helper to create a test histogram with standard axes."""
        return (
            hist.Hist.new
            .StrCat([], name="process", growth=True)
            .StrCat([], name="region", growth=True)
            .StrCat([], name="syst", growth=True)
            .Reg(100, 0, 5000, name=name)
            .Weight()
        )

    def test_split_single_region_single_syst(self):
        """Test splitting with one region and one systematic."""
        h = self._create_test_hist("mass_dilepton")
        h.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)

        summed_hists = {"mass_dilepton": h}
        split = split_hists_with_syst(summed_hists, sum_over_process=True)

        # Should have one entry: (region, syst, hist_name)
        assert ("sr", "Nominal", "mass_dilepton") in split

    def test_split_multiple_regions(self):
        """Test splitting with multiple regions."""
        h = self._create_test_hist("pt_leading_lepton")
        h.fill(process="DYJets", region="sr", syst="Nominal", pt_leading_lepton=100.0, weight=1.0)
        h.fill(process="DYJets", region="cr", syst="Nominal", pt_leading_lepton=120.0, weight=2.0)

        summed_hists = {"pt_leading_lepton": h}
        split = split_hists_with_syst(summed_hists, sum_over_process=True)

        # Should have two entries (one per region)
        assert ("sr", "Nominal", "pt_leading_lepton") in split
        assert ("cr", "Nominal", "pt_leading_lepton") in split

    def test_split_multiple_systematics(self):
        """Test splitting with multiple systematics."""
        h = self._create_test_hist("mass_fourobject")
        h.fill(process="Signal", region="sr", syst="Nominal", mass_fourobject=2000.0, weight=1.0)
        h.fill(process="Signal", region="sr", syst="LumiUp", mass_fourobject=2000.0, weight=1.2)
        h.fill(process="Signal", region="sr", syst="LumiDown", mass_fourobject=2000.0, weight=0.8)

        summed_hists = {"mass_fourobject": h}
        split = split_hists_with_syst(summed_hists, sum_over_process=True)

        # Should have three entries (one per systematic)
        assert ("sr", "Nominal", "mass_fourobject") in split
        assert ("sr", "LumiUp", "mass_fourobject") in split
        assert ("sr", "LumiDown", "mass_fourobject") in split

    def test_sum_over_process_removes_process_axis(self):
        """Test that sum_over_process=True projects out the process axis."""
        h = self._create_test_hist("mass_dilepton")
        h.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)
        h.fill(process="tt_tW", region="sr", syst="Nominal", mass_dilepton=60.0, weight=2.0)

        summed_hists = {"mass_dilepton": h}
        split = split_hists_with_syst(summed_hists, sum_over_process=True)

        h_split = split[("sr", "Nominal", "mass_dilepton")]
        # Process axis should be removed
        assert "process" not in [ax.name for ax in h_split.axes]

    def test_sum_over_process_false_preserves_process_axis(self):
        """Test that sum_over_process=False keeps the process axis."""
        h = self._create_test_hist("mass_dilepton")
        h.fill(process="DYJets", region="sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)

        summed_hists = {"mass_dilepton": h}
        split = split_hists_with_syst(summed_hists, sum_over_process=False)

        h_split = split[("sr", "Nominal", "mass_dilepton")]
        # Process axis should still be present
        assert "process" in [ax.name for ax in h_split.axes]

    def test_missing_region_axis_skipped(self):
        """Test that histograms without region axis are skipped with error log."""
        # Create a histogram without region axis
        h = hist.Hist.new.Reg(100, 0, 100, name="test").Weight()
        h.fill(test=50.0, weight=1.0)

        summed_hists = {"test": h}
        split = split_hists_with_syst(summed_hists, sum_over_process=True)

        # Should be empty (histogram skipped)
        assert len(split) == 0


class TestSumCutflowHists:
    """Test cutflow histogram summing across datasets."""

    def test_simple_cutflow_sum(self):
        """Test summing simple cutflow histograms."""
        h1 = hist.Hist(hist.axis.StrCategory(["cut1", "cut2"], name="cut"), storage=hist.storage.Weight())
        h1.fill(cut="cut1", weight=1.0)
        h1.fill(cut="cut2", weight=2.0)

        h2 = hist.Hist(hist.axis.StrCategory(["cut1", "cut2"], name="cut"), storage=hist.storage.Weight())
        h2.fill(cut="cut1", weight=3.0)
        h2.fill(cut="cut2", weight=4.0)

        my_hists = {
            "dataset1": {"cutflow": {"ee": {"onecut": h1}}},
            "dataset2": {"cutflow": {"ee": {"onecut": h2}}},
        }

        summed = _sum_cutflow_hists(my_hists)

        # Should sum: cut1 = 1+3=4, cut2 = 2+4=6
        h_sum = summed["ee"]["onecut"]
        assert h_sum[{"cut": "cut1"}].value == pytest.approx(4.0, rel=1e-5)
        assert h_sum[{"cut": "cut2"}].value == pytest.approx(6.0, rel=1e-5)

    def test_nested_cutflow_structure(self):
        """Test summing with nested cutflow structure (ee, mumu, em)."""
        h_ee = hist.Hist(hist.axis.StrCategory(["cut1"], name="cut"), storage=hist.storage.Weight())
        h_ee.fill(cut="cut1", weight=1.0)

        h_mumu = hist.Hist(hist.axis.StrCategory(["cut1"], name="cut"), storage=hist.storage.Weight())
        h_mumu.fill(cut="cut1", weight=2.0)

        my_hists = {
            "dataset1": {
                "cutflow": {
                    "ee": {"onecut": h_ee},
                    "mumu": {"onecut": h_mumu},
                }
            }
        }

        summed = _sum_cutflow_hists(my_hists)

        assert "ee" in summed
        assert "mumu" in summed
        assert summed["ee"]["onecut"][{"cut": "cut1"}].value == pytest.approx(1.0)
        assert summed["mumu"]["onecut"][{"cut": "cut1"}].value == pytest.approx(2.0)

    def test_empty_cutflow_returns_empty(self):
        """Test that datasets without cutflow return empty dict."""
        my_hists = {
            "dataset1": {"mass_dilepton": "some_hist"},  # No cutflow
        }

        summed = _sum_cutflow_hists(my_hists)

        assert summed == {}


class TestSaveHistogramsIntegration:
    """Integration tests for save_histograms() ROOT file I/O."""

    def _create_mock_args(self, **overrides):
        """Create mock args object for save_histograms."""
        args = MagicMock()
        args.era = "RunIISummer20UL18"
        args.sample = "DYJets"
        args.mass = None
        args.dir = None
        args.name = None
        # Apply overrides
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def _create_test_hist(self, name="test"):
        """Helper to create a test histogram."""
        return (
            hist.Hist.new
            .StrCat([], name="process", growth=True)
            .StrCat([], name="region", growth=True)
            .StrCat([], name="syst", growth=True)
            .Reg(100, 0, 5000, name=name)
            .Weight()
        )

    def test_save_histograms_creates_root_file(self, tmp_path, monkeypatch):
        """Test that save_histograms creates a ROOT file."""
        # Patch the working directory to use tmp_path
        monkeypatch.chdir(tmp_path)

        h = self._create_test_hist("mass_dilepton")
        h.fill(process="DYJets", region="wr_mumu_resolved_sr", syst="Nominal", mass_dilepton=50.0, weight=1.0)

        my_hists = {"dataset1": {"mass_dilepton": h}}
        args = self._create_mock_args()

        save_histograms(my_hists, args)

        # Verify ROOT file was created
        expected_path = tmp_path / "WR_Plotter" / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18" / "WRAnalyzer_DYJets.root"
        assert expected_path.exists()

    def test_save_histograms_signal_naming(self, tmp_path, monkeypatch):
        """Test that Signal samples get signal_{mass}.root naming."""
        monkeypatch.chdir(tmp_path)

        h = self._create_test_hist("mass_dilepton")
        h.fill(process="Signal", region="wr_ee_resolved_sr", syst="Nominal", mass_dilepton=2000.0, weight=1.0)

        my_hists = {"dataset1": {"mass_dilepton": h}}
        args = self._create_mock_args(sample="Signal", mass="WR3000_N1500")

        save_histograms(my_hists, args)

        # Verify signal naming
        expected_path = tmp_path / "WR_Plotter" / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18" / "WRAnalyzer_signal_WR3000_N1500.root"
        assert expected_path.exists()

    def test_save_histograms_root_file_contents(self, tmp_path, monkeypatch):
        """Test that ROOT file contains correctly named histograms in folders."""
        monkeypatch.chdir(tmp_path)

        h = self._create_test_hist("mass_dilepton")
        h.fill(process="DYJets", region="wr_mumu_resolved_sr", syst="Nominal", mass_dilepton=300.0, weight=1.5)

        my_hists = {"dataset1": {"mass_dilepton": h}}
        args = self._create_mock_args()

        save_histograms(my_hists, args)

        # Read back the ROOT file
        expected_path = tmp_path / "WR_Plotter" / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18" / "WRAnalyzer_DYJets.root"
        with uproot.open(expected_path) as f:
            # Check folder structure: /region/hist_name_region
            assert "wr_mumu_resolved_sr/mass_dilepton_wr_mumu_resolved_sr" in f

    def test_save_histograms_systematic_folders(self, tmp_path, monkeypatch):
        """Test that systematic variations go into syst_{name}_{region} folders."""
        monkeypatch.chdir(tmp_path)

        h = self._create_test_hist("pt_leading_lepton")
        h.fill(process="Signal", region="wr_ee_resolved_sr", syst="Nominal", pt_leading_lepton=100.0, weight=1.0)
        h.fill(process="Signal", region="wr_ee_resolved_sr", syst="LumiUp", pt_leading_lepton=100.0, weight=1.2)

        my_hists = {"dataset1": {"pt_leading_lepton": h}}
        args = self._create_mock_args(sample="Signal", mass="WR3000_N1500")

        save_histograms(my_hists, args)

        expected_path = tmp_path / "WR_Plotter" / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18" / "WRAnalyzer_signal_WR3000_N1500.root"
        with uproot.open(expected_path) as f:
            # Nominal in standard folder
            assert "wr_ee_resolved_sr/pt_leading_lepton_wr_ee_resolved_sr" in f
            # Systematic in syst folder
            assert "syst_lumiup_wr_ee_resolved_sr/pt_leading_lepton_syst_lumiup_wr_ee_resolved_sr" in f
