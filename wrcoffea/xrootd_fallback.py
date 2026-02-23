"""Helpers for XRootD redirector fallback during remote file access."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable


REDIRECTORS = [
    "root://cmsxrootd.fnal.gov/",
    "root://cms-xrd-global.cern.ch/",
    "root://xrootd-cms.infn.it/",
]

DEFAULT_TIMEOUT_SECONDS = 10
DEFAULT_RETRIES_PER_REDIRECTOR = 10
DEFAULT_RETRY_SLEEP_SECONDS = 10.0

_ROOT_URL_RE = re.compile(r"(root://[^\s'\"\\]+\.root)")
_LFN_RE = re.compile(r"(/store/[^\s'\"\\]+\.root)")


@dataclass
class RedirectorProbeResult:
    """Result of probing one LFN across one or more redirectors."""

    success: bool
    lfn: str
    resolved_url: str | None
    redirector: str | None
    total_attempts: int
    failures_by_redirector: dict[str, str] = field(default_factory=dict)

    def failure_summary(self) -> str:
        """Return a compact per-redirector error summary."""
        if self.success:
            return ""
        if not self.failures_by_redirector:
            return "no redirector failures recorded"
        return "; ".join(
            f"{redirector}: {msg}"
            for redirector, msg in self.failures_by_redirector.items()
        )


def extract_lfn_from_url(url: str) -> str | None:
    """Extract ``/store/...`` LFN from an XRootD URL or bare LFN, or ``None``."""
    if not isinstance(url, str):
        return None
    if url.startswith("/store/"):
        return url
    if not url.startswith("root://"):
        return None
    idx = url.find("//store/")
    if idx >= 0:
        return url[idx + 1:]
    return None


def build_xrootd_url(redirector: str, lfn: str) -> str:
    """Build ``root://<redirector>//store/...`` URL from redirector + LFN."""
    if not lfn.startswith("/"):
        lfn = "/" + lfn
    return f"{redirector.rstrip('/')}/{lfn}"


def extract_root_url_from_error(exc: Exception) -> str | None:
    """Best-effort extraction of a failing ROOT URL or bare LFN from an exception."""
    if exc is None:
        return None
    for text in (str(exc), repr(exc)):
        if not text:
            continue
        match = _ROOT_URL_RE.search(text)
        if match:
            return match.group(1)
        match = _LFN_RE.search(text)
        if match:
            return match.group(1)
    return None


def _probe_root_url(url: str, timeout: int) -> None:
    """Open a ROOT file URL to validate readability."""
    import uproot

    with uproot.open({url: None}, timeout=timeout):
        pass


def resolve_url_with_redirectors(
    url_or_lfn: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries_per_redirector: int = DEFAULT_RETRIES_PER_REDIRECTOR,
    sleep_seconds: float = DEFAULT_RETRY_SLEEP_SECONDS,
    redirectors: list[str] | None = None,
    probe: Callable[[str, int], None] | None = None,
) -> RedirectorProbeResult:
    """Resolve one LFN by probing redirectors until one succeeds."""
    if retries_per_redirector < 1:
        raise ValueError("retries_per_redirector must be >= 1")
    if timeout < 1:
        raise ValueError("timeout must be >= 1")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    lfn = extract_lfn_from_url(url_or_lfn)
    if lfn is None:
        if isinstance(url_or_lfn, str) and url_or_lfn.startswith("/store/"):
            lfn = url_or_lfn
        else:
            raise ValueError(f"Not an XRootD URL or LFN: {url_or_lfn!r}")

    redirector_list = redirectors or REDIRECTORS
    failures: dict[str, str] = {}
    total_attempts = 0
    probe_fn = probe or _probe_root_url

    for redirector in redirector_list:
        url = build_xrootd_url(redirector, lfn)
        last_error = None
        for attempt in range(1, retries_per_redirector + 1):
            total_attempts += 1
            try:
                probe_fn(url, timeout)
                return RedirectorProbeResult(
                    success=True,
                    lfn=lfn,
                    resolved_url=url,
                    redirector=redirector,
                    total_attempts=total_attempts,
                    failures_by_redirector=failures,
                )
            except Exception as exc:  # pragma: no cover - exercised in tests via fake probe
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries_per_redirector and sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        if last_error is not None:
            failures[redirector] = last_error

    return RedirectorProbeResult(
        success=False,
        lfn=lfn,
        resolved_url=None,
        redirector=None,
        total_attempts=total_attempts,
        failures_by_redirector=failures,
    )
