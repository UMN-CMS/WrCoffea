"""Tests for wrcoffea.fileset_validation — schema and selection checks."""

import pytest

from wrcoffea.fileset_validation import validate_fileset_schema, validate_selection


def _valid_fileset():
    return {
        "ds1": {
            "files": {"/path/file.root": "Events"},
            "metadata": {"sample": "DYJets_50to100", "physics_group": "DYJets"},
        },
    }


class TestValidateFilesetSchema:
    def test_valid(self):
        validate_fileset_schema(_valid_fileset())

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_fileset_schema({})

    def test_non_dict_raises(self):
        with pytest.raises(ValueError, match="dict-like"):
            validate_fileset_schema([1, 2, 3])

    def test_missing_files_raises(self):
        bad = {"ds1": {"metadata": {"sample": "x"}}}
        with pytest.raises(ValueError, match="files"):
            validate_fileset_schema(bad)

    def test_missing_metadata_raises(self):
        bad = {"ds1": {"files": {"/path.root": "Events"}}}
        with pytest.raises(ValueError, match="metadata"):
            validate_fileset_schema(bad)

    def test_missing_sample_raises(self):
        bad = {"ds1": {"files": {"/path.root": "Events"}, "metadata": {"era": "x"}}}
        with pytest.raises(ValueError, match="sample"):
            validate_fileset_schema(bad)


class TestValidateSelection:
    def test_nonempty_passes(self):
        validate_selection(
            _valid_fileset(),
            desired_process="DYJets",
            mass=None,
        )

    def test_empty_non_signal_raises(self):
        with pytest.raises(ValueError, match="0 datasets"):
            validate_selection(
                {},
                desired_process="DYJets",
                mass=None,
                preprocessed_fileset=_valid_fileset(),
            )

    def test_empty_signal_raises_with_mass_hint(self):
        with pytest.raises(ValueError, match="WR9999_N1"):
            validate_selection(
                {},
                desired_process="Signal",
                mass="WR9999_N1",
                preprocessed_fileset=_valid_fileset(),
            )
