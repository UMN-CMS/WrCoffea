"""Tests for wrcoffea.cli_utils — mass normalization, filtering, fileset path building."""

import pytest

from wrcoffea.cli_utils import (
    normalize_mass_point,
    signal_sample_matches_mass,
    filter_by_process,
    list_eras,
    list_samples,
    build_fileset_path,
)


# ---------------------------------------------------------------------------
# normalize_mass_point
# ---------------------------------------------------------------------------

class TestNormalizeMassPoint:
    def test_none_passthrough(self):
        assert normalize_mass_point(None) is None

    def test_canonical_unchanged(self):
        assert normalize_mass_point("WR2000_N100") == "WR2000_N100"

    def test_legacy_MWR_MN_converted(self):
        assert normalize_mass_point("MWR2000_MN100") == "WR2000_N100"

    def test_whitespace_stripped(self):
        assert normalize_mass_point("  WR4000_N2100  ") == "WR4000_N2100"

    def test_unknown_format_passthrough(self):
        assert normalize_mass_point("something_else") == "something_else"


# ---------------------------------------------------------------------------
# signal_sample_matches_mass
# ---------------------------------------------------------------------------

class TestSignalSampleMatchesMass:
    def test_canonical_match(self):
        sample = "WRtoNLtoLLJJ_WR2000_N1100_TuneCP5"
        assert signal_sample_matches_mass(sample, "WR2000_N1100") is True

    def test_legacy_MWR_MN_match(self):
        sample = "WRtoNLtoLLJJ_MWR2000_MN100_TuneCP5"
        assert signal_sample_matches_mass(sample, "WR2000_N100") is True

    def test_legacy_MWR_N_match(self):
        sample = "WRtoNLtoLLJJ_MWR600_N100_TuneCP5"
        assert signal_sample_matches_mass(sample, "WR600_N100") is True

    def test_no_match(self):
        sample = "WRtoNLtoLLJJ_WR4000_N2100_TuneCP5"
        assert signal_sample_matches_mass(sample, "WR2000_N100") is False

    def test_empty_sample(self):
        assert signal_sample_matches_mass("", "WR2000_N100") is False


# ---------------------------------------------------------------------------
# filter_by_process
# ---------------------------------------------------------------------------

def _mock_fileset():
    """Return a small mock fileset dict for testing."""
    return {
        "DY_50to100": {
            "files": {"/path/dy.root": "Events"},
            "metadata": {"physics_group": "DYJets", "sample": "DYJets_50to100"},
        },
        "TTbar": {
            "files": {"/path/tt.root": "Events"},
            "metadata": {"physics_group": "tt_tW", "sample": "TTbar"},
        },
        "Signal_WR2000": {
            "files": {"/path/sig.root": "Events"},
            "metadata": {"physics_group": "Signal", "sample": "WRtoNLtoLLJJ_WR2000_N1100"},
        },
        "Signal_WR4000": {
            "files": {"/path/sig2.root": "Events"},
            "metadata": {"physics_group": "Signal", "sample": "WRtoNLtoLLJJ_WR4000_N2100"},
        },
    }


class TestFilterByProcess:
    def test_filter_background(self):
        result = filter_by_process(_mock_fileset(), "DYJets")
        assert list(result.keys()) == ["DY_50to100"]

    def test_filter_signal_with_mass(self):
        result = filter_by_process(_mock_fileset(), "Signal", mass="WR2000_N1100")
        assert list(result.keys()) == ["Signal_WR2000"]

    def test_filter_signal_no_mass_raises(self):
        with pytest.raises(ValueError, match="requires a mass point"):
            filter_by_process(_mock_fileset(), "Signal")

    def test_filter_no_match_returns_empty(self):
        result = filter_by_process(_mock_fileset(), "Nonprompt")
        assert result == {}


# ---------------------------------------------------------------------------
# list_eras / list_samples
# ---------------------------------------------------------------------------

class TestListHelpers:
    def test_list_eras_nonempty(self):
        eras = list_eras()
        assert len(eras) > 0
        assert "RunIII2024Summer24" in eras

    def test_list_samples_nonempty(self):
        samples = list_samples()
        assert "Signal" in samples
        assert "DYJets" in samples


# ---------------------------------------------------------------------------
# build_fileset_path
# ---------------------------------------------------------------------------

class TestBuildFilesetPath:
    def test_mc_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_mc_fileset.json"
        assert "Run3/2024/RunIII2024Summer24" in str(path)

    def test_data_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="EGamma", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_data_fileset.json"

    def test_signal_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="Signal", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_signal_fileset.json"

    def test_dy_lo_inclusive(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy="LO_inclusive")
        assert "dy_lo_inc" in path.name

    def test_dy_nlo_mll(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy="NLO_mll_binned")
        assert "dy_nlo_mll" in path.name

    def test_dy_lo_ht(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy="LO_HT")
        assert "dy_lo_ht" in path.name

    def test_unskimmed_mc_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=True, dy=None)
        assert path.name == "RunIII2024Summer24_mc_fileset.json"
        assert "unskimmed" in path.parts

    def test_unskimmed_data_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="Muon", unskimmed=True, dy=None)
        assert path.name == "RunIII2024Summer24_data_fileset.json"
        assert "unskimmed" in path.parts

    def test_unskimmed_signal_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="Signal", unskimmed=True, dy=None)
        assert path.name == "RunIII2024Summer24_signal_fileset.json"
        assert "unskimmed" in path.parts

    def test_unskimmed_dy_lo_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=True, dy="LO_inclusive")
        assert "dy_lo_inc" in path.name
        assert "unskimmed" in path.parts

    def test_skimmed_path_no_unskimmed_dir(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy=None)
        assert "unskimmed" not in path.parts
