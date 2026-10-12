"""Tests for wrcoffea.cli_utils — mass normalization, filtering, fileset path building."""

import pytest
from pathlib import Path

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
        "TTto2L2Nu": {
            "files": {"/path/tt.root": "Events"},
            "metadata": {"physics_group": "tt_tW", "sample": "TTto2L2Nu_TuneCP5"},
        },
        "TWminusto2L2Nu": {
            "files": {"/path/tw1.root": "Events"},
            "metadata": {"physics_group": "tt_tW", "sample": "TWminusto2L2Nu_TuneCP5"},
        },
        "TbarWplusto2L2Nu": {
            "files": {"/path/tw2.root": "Events"},
            "metadata": {"physics_group": "tt_tW", "sample": "TbarWplusto2L2Nu_TuneCP5"},
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

    def test_filter_tt_tW_returns_all_top(self):
        result = filter_by_process(_mock_fileset(), "tt_tW")
        assert set(result.keys()) == {"TTto2L2Nu", "TWminusto2L2Nu", "TbarWplusto2L2Nu"}

    def test_filter_TTbar_returns_dileptonic_only(self):
        result = filter_by_process(_mock_fileset(), "TTbar")
        assert list(result.keys()) == ["TTto2L2Nu"]

    def test_filter_tW_returns_single_top_only(self):
        result = filter_by_process(_mock_fileset(), "tW")
        assert set(result.keys()) == {"TWminusto2L2Nu", "TbarWplusto2L2Nu"}

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

    def test_list_samples_includes_subgroups(self):
        samples = list_samples()
        assert "TTbar" in samples
        assert "tW" in samples


# ---------------------------------------------------------------------------
# build_fileset_path
# ---------------------------------------------------------------------------

class TestBuildFilesetPath:
    def test_mc_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_mc_dy_lo_ht_fileset.json"
        assert "Run3/2024/RunIII2024Summer24" in str(path)
        assert "skimmed" in path.parts

    def test_data_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="EGamma", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_data_fileset.json"

    def test_signal_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="Signal", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_signal_fileset.json"

    def test_dy_lo_inc(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy="lo_inc")
        assert "dy_lo_inc" in path.name

    def test_dy_nlo_inc(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy="nlo_inc")
        assert "dy_nlo_inc" in path.name

    def test_unskimmed_mc_path(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=True, dy=None)
        assert path.name == "RunIII2024Summer24_mc_dy_lo_ht_fileset.json"
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
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=True, dy="lo_inc")
        assert "dy_lo_inc" in path.name
        assert "unskimmed" in path.parts

    def test_skimmed_path_has_skimmed_dir(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="DYJets", unskimmed=False, dy=None)
        assert "skimmed" in path.parts
        assert "unskimmed" not in path.parts

    def test_non_dy_mc_uses_era_default(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="tt_tW", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_mc_dy_lo_ht_fileset.json"

    def test_TTbar_subgroup_uses_parent_fileset(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="TTbar", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_mc_dy_lo_ht_fileset.json"

    def test_tW_subgroup_uses_parent_fileset(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="tW", unskimmed=False, dy=None)
        assert path.name == "RunIII2024Summer24_mc_dy_lo_ht_fileset.json"

    def test_ul18_default_mc_tag(self):
        path = build_fileset_path(era="RunIISummer20UL18", sample="DYJets", unskimmed=False, dy=None)
        assert path.name == "RunIISummer20UL18_mc_dy_lo_ht_fileset.json"

    def test_dy_lo_ht_ul18(self):
        path = build_fileset_path(era="RunIISummer20UL18", sample="DYJets", unskimmed=False, dy="lo_ht")
        assert path.name == "RunIISummer20UL18_mc_dy_lo_ht_fileset.json"

    def test_ul18_unskimmed_signal_stays_unskimmed(self):
        # The old UL18 Signal unskimmed->skimmed fallback was removed:
        # build_fileset_path now honors the unskimmed flag for every era/sample.
        path = build_fileset_path(era="RunIISummer20UL18", sample="Signal", unskimmed=True, dy=None)
        assert path.name == "RunIISummer20UL18_signal_fileset.json"
        assert "unskimmed" in path.parts
        assert path.parts[-2] == "unskimmed"

    def test_2024_unskimmed_signal_stays_unskimmed(self):
        path = build_fileset_path(era="RunIII2024Summer24", sample="Signal", unskimmed=True, dy=None)
        assert path.name == "RunIII2024Summer24_signal_fileset.json"
        assert "unskimmed" in path.parts

    def test_ul18_unskimmed_mc_stays_unskimmed(self):
        path = build_fileset_path(era="RunIISummer20UL18", sample="DYJets", unskimmed=True, dy=None)
        assert path.name == "RunIISummer20UL18_mc_dy_lo_ht_fileset.json"
        assert "unskimmed" in path.parts


# ---------------------------------------------------------------------------
# select_signal_points (run_analysis --signal-points / --mwr-range)
# ---------------------------------------------------------------------------

class TestSelectSignalPoints:
    ERA = "RunIII2024Summer24"

    def _csv_points(self):
        from pathlib import Path
        from wrcoffea.cli_utils import load_masses_from_csv
        return load_masses_from_csv(Path(f"data/signal_points/{self.ERA}_mass_points.csv"))

    def test_default_is_the_default_grid(self):
        from wrcoffea.cli_utils import select_default_signal_points, select_signal_points
        assert select_signal_points(self.ERA) == select_default_signal_points(self.ERA)
        assert select_signal_points(self.ERA, ["default"]) == select_default_signal_points(self.ERA)

    def test_all_is_every_csv_point(self):
        from wrcoffea.cli_utils import select_signal_points
        assert select_signal_points(self.ERA, ["all"]) == self._csv_points()

    def test_mwr_range_filters_on_wr(self):
        from wrcoffea.cli_utils import select_signal_points
        pts = select_signal_points(self.ERA, ["all"], [2000, 4000])
        wrs = [int(p.split("_")[0][2:]) for p in pts]
        assert pts and all(2000 <= wr <= 4000 for wr in wrs)
        assert len(pts) == sum(1 for p in self._csv_points() if 2000 <= int(p.split("_")[0][2:]) <= 4000)

    def test_explicit_points_are_normalized_and_deduplicated(self):
        from wrcoffea.cli_utils import select_signal_points
        tag = self._csv_points()[0]
        wr, n = tag[2:].split("_N")
        assert select_signal_points(self.ERA, [tag, f"MWR{wr}_MN{n}"]) == [tag]

    def test_unknown_point_raises(self):
        from wrcoffea.cli_utils import select_signal_points
        with pytest.raises(ValueError, match="not in"):
            select_signal_points(self.ERA, ["WR123_N45"])

    def test_empty_selection_raises(self):
        from wrcoffea.cli_utils import select_signal_points
        with pytest.raises(ValueError, match="No signal points selected"):
            select_signal_points(self.ERA, ["all"], [0, 1])

    def test_keywords_cannot_be_mixed_with_points(self):
        from wrcoffea.cli_utils import select_signal_points
        with pytest.raises(ValueError, match="cannot be combined"):
            select_signal_points(self.ERA, ["all", self._csv_points()[0]])


@pytest.mark.parametrize("era", ["RunIISummer20UL18", "RunIISummer20UL17", "RunIISummer20UL16", "RunIISummer20UL16APV"])
def test_ul_signal_points_select_the_official_samples(era):
    """The UL eras run the official UL samples: every point of the era's CSV selects its e and mu sample from the
    default fileset, and no pre-legacy scan model is in it (those live in <era>_signal_scan_fileset.json).
    Eras whose skims_v2 signal fileset is not installed yet are skipped."""
    import json
    from pathlib import Path
    from wrcoffea.cli_utils import build_fileset_path, filter_by_process, load_masses_from_csv
    path = build_fileset_path(era=era, sample="Signal", unskimmed=False, dy=None)
    if not path.exists() or not Path(f"data/signal_points/{era}_mass_points.csv").exists():
        pytest.skip(f"{era}: signal fileset or mass-point CSV not installed yet")
    fileset = json.loads(path.read_text())
    points = load_masses_from_csv(Path(f"data/signal_points/{era}_mass_points.csv"))
    assert 2 * len(points) == len(fileset)
    for mass in points:
        selected = filter_by_process(fileset, "Signal", mass=mass)
        flavours = sorted(d["metadata"]["sample"].split("_")[0] for d in selected.values())
        assert flavours == ["WRtoNEltoElElJJ", "WRtoNMutoMuMuJJ"], mass
    assert not any(d["metadata"].get("scan_source") for d in fileset.values())


class TestMassPointsOnlyForSignalRuns:
    """The mass-point CSV is read only when Signal is involved (a data-only era has none)."""

    def test_needs_mass_points(self):
        from wrcoffea.cli_utils import needs_mass_points
        assert needs_mass_points("Signal") and needs_mass_points("signal") and needs_mass_points("mc") and needs_mass_points("all")
        assert not any(needs_mass_points(s) for s in ("EGamma", "Muon", "DYJets", "tt_tW", "Nonprompt", "Other", "data", "bkg"))

    def test_data_run_of_an_era_without_csv(self):
        from wrcoffea.cli_utils import mass_choices_for
        assert not Path("data/signal_points/RunIII2025Summer25_mass_points.csv").exists()
        assert mass_choices_for("RunIII2025Summer25", "EGamma") == []
        assert mass_choices_for("RunIII2025Summer25", "data") == []
        with pytest.raises(FileNotFoundError):
            mass_choices_for("RunIII2025Summer25", "Signal")

    def test_signal_run_reads_the_csv(self):
        from wrcoffea.cli_utils import mass_choices_for
        points = mass_choices_for("RunIII2024Summer24", "Signal")
        assert "WR4000_N2100" in points
        assert mass_choices_for("RunIII2024Summer24", "all") == points
