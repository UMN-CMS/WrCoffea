"""Tests for wrcoffea.das_utils."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from wrcoffea.das_utils import (
    validate_das_path,
    primary_dataset_from_das_path,
    check_dasgoclient,
    check_grid_proxy,
    query_das_files,
    das_files_to_urls,
    infer_output_dir,
    REDIRECTOR,
)


# ---------------------------------------------------------------------------
# validate_das_path
# ---------------------------------------------------------------------------

class TestValidateDasPath:
    def test_valid_nanoaodsim(self):
        p, c, t = validate_das_path(
            "/TTto2L2Nu_TuneCP5_13p6TeV/Run3Summer24NanoAODv15-150X/NANOAODSIM"
        )
        assert p == "TTto2L2Nu_TuneCP5_13p6TeV"
        assert c == "Run3Summer24NanoAODv15-150X"
        assert t == "NANOAODSIM"

    def test_valid_nanoaod(self):
        p, c, t = validate_das_path(
            "/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD"
        )
        assert p == "EGamma0"
        assert c == "Run2024C-MINIv6NANOv15-v1"
        assert t == "NANOAOD"

    def test_missing_leading_slash(self):
        with pytest.raises(ValueError, match="must start with '/'"):
            validate_das_path("TTto2L2Nu/Run3Summer24/NANOAODSIM")

    def test_wrong_number_of_components(self):
        with pytest.raises(ValueError, match="3 components"):
            validate_das_path("/TTto2L2Nu/NANOAODSIM")

    def test_bad_datatier(self):
        with pytest.raises(ValueError, match="NANOAOD or NANOAODSIM"):
            validate_das_path("/TTto2L2Nu/Run3Summer24/MINIAOD")

    def test_trailing_slash_stripped(self):
        p, c, t = validate_das_path(
            "/TTto2L2Nu/Run3Summer24/NANOAODSIM/"
        )
        assert t == "NANOAODSIM"


# ---------------------------------------------------------------------------
# primary_dataset_from_das_path
# ---------------------------------------------------------------------------

class TestPrimaryDataset:
    def test_extracts_first_component(self):
        assert primary_dataset_from_das_path(
            "/DYto2E-4Jets_Bin-MLL-50/Run3Summer24NanoAODv15-150X/NANOAODSIM"
        ) == "DYto2E-4Jets_Bin-MLL-50"


# ---------------------------------------------------------------------------
# check_dasgoclient
# ---------------------------------------------------------------------------

class TestCheckDasgoclient:
    @patch("shutil.which", return_value="/usr/bin/dasgoclient")
    def test_found_in_path(self, mock_which):
        assert check_dasgoclient() == "/usr/bin/dasgoclient"

    @patch("shutil.which", return_value=None)
    @patch("wrcoffea.das_utils.Path")
    def test_found_at_cvmfs(self, mock_path_cls, mock_which):
        mock_path_cls.return_value.exists.return_value = True
        assert check_dasgoclient() == "/cvmfs/cms.cern.ch/common/dasgoclient"

    @patch("shutil.which", return_value=None)
    @patch("wrcoffea.das_utils.Path")
    def test_not_found(self, mock_path_cls, mock_which):
        mock_path_cls.return_value.exists.return_value = False
        with pytest.raises(FileNotFoundError, match="dasgoclient not found"):
            check_dasgoclient()


# ---------------------------------------------------------------------------
# check_grid_proxy
# ---------------------------------------------------------------------------

class TestCheckGridProxy:
    @patch("subprocess.run")
    def test_valid_proxy(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="7200\n")
        check_grid_proxy()  # should not raise

    @patch("subprocess.run")
    def test_no_proxy(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
        with pytest.raises(RuntimeError, match="No valid grid proxy"):
            check_grid_proxy()

    @patch("subprocess.run")
    def test_expired_proxy(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="30\n")
        with pytest.raises(RuntimeError, match="expires in 30s"):
            check_grid_proxy()


# ---------------------------------------------------------------------------
# query_das_files
# ---------------------------------------------------------------------------

class TestQueryDasFiles:
    @patch("wrcoffea.das_utils.check_dasgoclient", return_value="/usr/bin/dasgoclient")
    @patch("subprocess.run")
    def test_returns_sorted_lfns(self, mock_run, mock_check):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/store/mc/b.root\n/store/mc/a.root\n",
        )
        result = query_das_files("/DS/Campaign/NANOAODSIM")
        assert result == ["/store/mc/a.root", "/store/mc/b.root"]

    @patch("wrcoffea.das_utils.check_dasgoclient", return_value="/usr/bin/dasgoclient")
    @patch("subprocess.run")
    def test_failure_raises(self, mock_run, mock_check):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error"
        )
        with pytest.raises(RuntimeError, match="dasgoclient failed"):
            query_das_files("/DS/Campaign/NANOAODSIM")

    @patch("wrcoffea.das_utils.check_dasgoclient", return_value="/usr/bin/dasgoclient")
    @patch("subprocess.run")
    def test_empty_raises(self, mock_run, mock_check):
        mock_run.return_value = MagicMock(returncode=0, stdout="\n")
        with pytest.raises(RuntimeError, match="No files returned"):
            query_das_files("/DS/Campaign/NANOAODSIM")


# ---------------------------------------------------------------------------
# das_files_to_urls
# ---------------------------------------------------------------------------

class TestDasFilesToUrls:
    def test_prepends_redirector(self):
        lfns = ["/store/mc/a.root", "/store/mc/b.root"]
        urls = das_files_to_urls(lfns)
        assert urls == [
            f"{REDIRECTOR}/store/mc/a.root",
            f"{REDIRECTOR}/store/mc/b.root",
        ]

    def test_empty_list(self):
        assert das_files_to_urls([]) == []


# ---------------------------------------------------------------------------
# infer_output_dir
# ---------------------------------------------------------------------------

class TestInferOutputDir:
    def test_returns_data_skims_primary_ds(self):
        result = infer_output_dir(
            "/TTto2L2Nu_TuneCP5/Run3Summer24/NANOAODSIM"
        )
        # Campaign "Run3Summer24" has no "NanoAOD" substring → fallback base
        assert result == Path("data/skims/files/TTto2L2Nu_TuneCP5")

    def test_data_dataset(self):
        result = infer_output_dir("/EGamma0/Run2024C-v1/NANOAOD")
        assert result == Path("data/skims/files/EGamma0")
