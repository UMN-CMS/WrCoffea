"""Unit tests for wrcoffea.xrootd_fallback."""

from __future__ import annotations

import pytest

from wrcoffea.xrootd_fallback import (
    build_xrootd_url,
    extract_lfn_from_url,
    extract_root_url_from_error,
    resolve_url_with_redirectors,
)


def test_extract_lfn_from_url():
    assert (
        extract_lfn_from_url("root://cmsxrootd.fnal.gov//store/mc/a.root")
        == "/store/mc/a.root"
    )
    assert extract_lfn_from_url("/store/mc/a.root") == "/store/mc/a.root"
    assert extract_lfn_from_url("file:///tmp/a.root") is None


def test_build_xrootd_url_keeps_double_slash_before_store():
    assert (
        build_xrootd_url("root://cmsxrootd.fnal.gov/", "/store/mc/a.root")
        == "root://cmsxrootd.fnal.gov//store/mc/a.root"
    )


def test_extract_root_url_from_error():
    exc = ValueError(
        "not a ROOT file: first four bytes are b'\\x00\\x00\\x00\\x00'\n"
        "in file root://cmsxrootd.fnal.gov//store/mc/a.root"
    )
    assert (
        extract_root_url_from_error(exc)
        == "root://cmsxrootd.fnal.gov//store/mc/a.root"
    )


def test_extract_root_url_from_error_bare_lfn():
    exc = OSError(
        "expected Chunk of length 403, received 0 bytes from FSSpecSource\n"
        "for file path /store/mc/Run3Summer22NanoAODv12/sample/NANOAODSIM/"
        "130X_mcRun3_2022_realistic_v5-v2/40000/c517dbcb.root"
    )
    assert (
        extract_root_url_from_error(exc)
        == "/store/mc/Run3Summer22NanoAODv12/sample/NANOAODSIM/"
        "130X_mcRun3_2022_realistic_v5-v2/40000/c517dbcb.root"
    )


def test_resolve_url_with_redirectors_first_fails_second_succeeds():
    attempts = []

    def _probe(url, timeout):
        attempts.append((url, timeout))
        if url.startswith("root://r1/"):
            raise OSError("bad replica")

    result = resolve_url_with_redirectors(
        "/store/mc/a.root",
        timeout=5,
        retries_per_redirector=2,
        sleep_seconds=0,
        redirectors=["root://r1/", "root://r2/"],
        probe=_probe,
    )

    assert result.success is True
    assert result.redirector == "root://r2/"
    assert result.resolved_url == "root://r2//store/mc/a.root"
    assert result.total_attempts == 3
    assert "root://r1/" in result.failures_by_redirector


def test_resolve_url_with_redirectors_all_fail():
    def _probe(url, timeout):
        raise RuntimeError(f"cannot open {url}")

    result = resolve_url_with_redirectors(
        "root://r1//store/mc/a.root",
        timeout=5,
        retries_per_redirector=1,
        sleep_seconds=0,
        redirectors=["root://r1/", "root://r2/"],
        probe=_probe,
    )

    assert result.success is False
    assert result.resolved_url is None
    summary = result.failure_summary()
    assert "root://r1/" in summary
    assert "root://r2/" in summary


def test_resolve_url_with_redirectors_rejects_non_xrootd_input():
    with pytest.raises(ValueError, match="Not an XRootD URL or LFN"):
        resolve_url_with_redirectors("not-a-url", probe=lambda *_: None)
