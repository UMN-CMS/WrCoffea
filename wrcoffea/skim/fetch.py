"""Download one NanoAOD file to local scratch with checksum verification.

``xrdcp`` is used (it is on every CMS node and in Homebrew), with the DBS
adler32 passed as ``--cksum adler32:<value>`` so a corrupt transfer fails the
copy instead of producing a broken skim.  Sources are tried in the order
:func:`wrcoffea.skim.catalog.xrootd_sources` returns them: the redirectors
(FNAL, global, INFN), then every disk replica Rucio knows about, as HTTPS
WebDAV with the grid proxy.  Each
attempt is logged with its classification so a file that fails everywhere
can be reported with evidence.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
import subprocess
import threading
import time
import zlib
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# (category, retry same source?) keyed by substrings of xrdcp's stderr.
_ERROR_PATTERNS: list[tuple[str, str, bool]] = [
    ("checksum error", "checksum", False),
    ("cksum", "checksum", False),
    ("no such file", "missing", False),
    ("[3011]", "missing", False),
    ("not found", "missing", False),
    ("permission denied", "permission", False),
    ("[3010]", "permission", False),
    ("operation expired", "timeout", True),
    ("timed out", "timeout", True),
    ("timeout", "timeout", True),
    ("[3005]", "timeout", True),
    ("connection", "network", True),
    ("socket", "network", True),
    ("redirect limit", "redirect", True),
    ("no servers are available", "unavailable", True),
    ("unable to", "network", True),
]


@dataclass
class FetchAttempt:
    url: str
    rc: int
    seconds: float
    category: str
    message: str


@dataclass
class FetchResult:
    lfn: str
    dest: str
    ok: bool = False
    size: int = 0
    adler32: str = ""
    source: str = ""
    attempts: list[FetchAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_xrdcp_error(rc: int, stderr: str) -> tuple[str, bool]:
    """Map an xrdcp failure to (category, retryable-on-same-source)."""
    if rc == 53:
        return "checksum", False
    msg = (stderr or "").lower()
    for pat, cat, retry in _ERROR_PATTERNS:
        if pat in msg:
            return cat, retry
    return "unknown", True


def adler32_of(path: str, chunk: int = 16 << 20) -> str:
    """adler32 of a local file as 8 lowercase hex digits (DBS convention)."""
    a = 1
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            a = zlib.adler32(b, a)
    return f"{a & 0xFFFFFFFF:08x}"


def xrdcp_available() -> bool:
    return shutil.which("xrdcp") is not None


# Cooperative cancellation: a worker that is told to stop sets STOP and calls
# abort_active() so that transfers running in prefetch threads end at once
# instead of finishing a multi-GB download first.
STOP = threading.Event()
_ACTIVE: set = set()
_ACTIVE_LOCK = threading.Lock()


class FetchAborted(RuntimeError):
    """Raised by :func:`fetch_file` when :data:`STOP` is set."""


def abort_active() -> int:
    """Kill every transfer subprocess started by this process; returns how many."""
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    n = 0
    for proc in procs:
        try:
            proc.kill()
            n += 1
        except Exception:
            pass
    return n


# A transfer whose output file has not grown for this long is killed and retried from the next
# source: a redirector can hand out a server that accepts the connection and then sends nothing,
# and waiting for the size-based deadline would cost up to half an hour per file.
STALL_TIMEOUT = 180.0
STALL_RC = -15
_POLL_SECONDS = 5.0


def _run_transfer(cmd: list[str], timeout: float, *, progress_path: str | None = None,
                  stall_timeout: float | None = None, poll: float | None = None) -> tuple[int, str]:
    """Run a transfer subprocess, registered in _ACTIVE while it lives.

    Raises ``subprocess.TimeoutExpired`` at the hard deadline; returns ``(STALL_RC, message)``
    when *progress_path* has not changed size for *stall_timeout* seconds.
    """
    t0 = time.monotonic()
    last_size, last_change = -1, t0
    poll = poll or _POLL_SECONDS
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        with _ACTIVE_LOCK:
            _ACTIVE.add(proc)
        try:
            while True:
                try:
                    _, err = proc.communicate(timeout=poll)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if now - t0 > timeout:
                    proc.kill()
                    proc.wait()      # not communicate(): a grandchild holding the pipes would block it
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if progress_path and stall_timeout:
                    try:
                        size = os.path.getsize(progress_path)
                    except OSError:
                        size = -1
                    if size != last_size:
                        last_size, last_change = size, now
                    elif now - last_change > stall_timeout:
                        proc.kill()
                        proc.wait()
                        return STALL_RC, f"no progress for {stall_timeout:.0f}s at {max(size, 0) / 1e6:.0f} MB (stalled)"
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(proc)
    return proc.returncode, (err or "").strip()


def _timeout_for(size: int, min_rate: float, base: float, cap: float) -> float:
    return min(cap, base + (size / max(min_rate, 1.0)))


def xrdcp(url: str, dest: str, *, adler32: str | None, streams: int = 4,
          timeout: float = 1800.0, stall_timeout: float | None = STALL_TIMEOUT) -> FetchAttempt:
    """Run one xrdcp into ``dest`` (written as ``dest + '.part'`` then renamed)."""
    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    cmd = ["xrdcp", "-f", "--nopbar", "--streams", str(max(1, streams))]
    if adler32:
        cmd += ["--cksum", f"adler32:{adler32}"]
    cmd += [url, tmp]
    t0 = time.monotonic()
    try:
        rc, err = _run_transfer(cmd, timeout, progress_path=tmp, stall_timeout=stall_timeout)
    except subprocess.TimeoutExpired:
        rc, err = -9, f"killed after {timeout:.0f}s (local deadline)"
    if rc < 0 and STOP.is_set():
        err = "transfer aborted (stop requested)"
    seconds = time.monotonic() - t0
    if rc == 0 and os.path.exists(tmp):
        os.replace(tmp, dest)
        return FetchAttempt(url=url, rc=0, seconds=seconds, category="ok", message="")
    if os.path.exists(tmp):
        os.remove(tmp)
    cat, _ = classify_xrdcp_error(rc, err)
    if rc == -9:
        cat = "timeout"
    elif rc == STALL_RC:
        cat = "stall"
    return FetchAttempt(url=url, rc=rc, seconds=seconds, category=cat, message=err[-400:])


def https_get(url: str, dest: str, *, adler32: str | None, timeout: float = 1800.0,
              chunk: int = 8 << 20) -> FetchAttempt:
    """Stream an ``https://`` (WebDAV) replica to ``dest`` with the grid proxy."""
    import urllib.error
    import urllib.request
    from wrcoffea.skim.catalog import ssl_context

    tmp = dest + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    t0 = time.monotonic()
    a = 1
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ssl_context(), timeout=timeout) as resp, open(tmp, "wb") as out:
            while True:
                b = resp.read(chunk)
                if not b:
                    break
                out.write(b)
                a = zlib.adler32(b, a)
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"download exceeded {timeout:.0f}s")
                if STOP.is_set():
                    raise FetchAborted(f"{url}: stop requested")
    except urllib.error.HTTPError as e:
        cat = "missing" if e.code == 404 else ("permission" if e.code in (401, 403) else "network")
        if os.path.exists(tmp):
            os.remove(tmp)
        return FetchAttempt(url=url, rc=e.code, seconds=time.monotonic() - t0, category=cat, message=str(e)[:400])
    except FetchAborted:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    except Exception as e:  # network, TLS, timeout
        if os.path.exists(tmp):
            os.remove(tmp)
        cat = "timeout" if isinstance(e, TimeoutError) or "timed out" in str(e).lower() else "network"
        return FetchAttempt(url=url, rc=-1, seconds=time.monotonic() - t0, category=cat, message=f"{type(e).__name__}: {e}"[:400])
    seconds = time.monotonic() - t0
    got = f"{a & 0xFFFFFFFF:08x}"
    if adler32 and got != adler32.lower():
        os.remove(tmp)
        return FetchAttempt(url=url, rc=53, seconds=seconds, category="checksum",
                            message=f"adler32 {got} != expected {adler32}")
    os.replace(tmp, dest)
    return FetchAttempt(url=url, rc=0, seconds=seconds, category="ok", message="")


def fetch_file(lfn: str, dest: str, sources: list[str], *, adler32: str | None = None,
               size: int = 0, streams: int = 4, attempts_per_source: int = 2,
               min_rate_bytes_s: float = 2e6, base_timeout: float = 180.0,
               max_timeout: float = 3 * 3600.0, backoff: tuple[float, float] = (5.0, 60.0),
               verify_local: bool = True, stall_timeout: float | None = STALL_TIMEOUT) -> FetchResult:
    """Fetch *lfn* into *dest* trying *sources* in order.

    Returns a :class:`FetchResult`; ``ok`` is False only when every source was
    exhausted.  The local file is additionally re-checksummed when
    *verify_local* is set and an expected adler32 is known, so a server that
    ignores the checksum request still cannot hand us a corrupt file.
    """
    result = FetchResult(lfn=lfn, dest=dest)
    if not xrdcp_available() and any(not u.startswith(("https://", "http://")) for u in sources):
        raise RuntimeError("xrdcp not found on PATH; install the XRootD client")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    timeout = _timeout_for(size, min_rate_bytes_s, base_timeout, max_timeout)
    for url in sources:
        for attempt in range(1, attempts_per_source + 1):
            if STOP.is_set():
                raise FetchAborted(f"{lfn}: stop requested")
            if url.startswith(("https://", "http://")):
                att = https_get(url, dest, adler32=adler32, timeout=timeout)
            else:
                att = xrdcp(url, dest, adler32=adler32, streams=streams, timeout=timeout,
                            stall_timeout=stall_timeout)
            result.attempts.append(att)
            if att.rc == 0:
                ok = True
                if verify_local and adler32:
                    try:
                        local = adler32_of(dest)
                    except FileNotFoundError:
                        local = None
                    if local != adler32.lower():
                        att.category = "checksum" if local is not None else "vanished"
                        att.message = (f"local adler32 {local} != expected {adler32}" if local is not None
                                       else "downloaded file disappeared before verification")
                        att.rc = 53
                        if os.path.exists(dest):
                            os.remove(dest)
                        ok = False
                if ok:
                    result.ok = True
                    result.size = os.path.getsize(dest)
                    result.adler32 = adler32 or (adler32_of(dest) if verify_local else "")
                    result.source = url
                    logger.info("fetched %s from %s in %.0fs (%.1f MB/s)", os.path.basename(lfn),
                                url.split("//")[1].split("/")[0], att.seconds,
                                result.size / 1e6 / max(att.seconds, 1e-3))
                    return result
            logger.warning("fetch attempt %d/%d via %s failed [%s]: %s", attempt, attempts_per_source,
                           url.split("//")[1].split("/")[0] if "//" in url else url, att.category,
                           att.message[-160:])
            _, retry_same = classify_xrdcp_error(att.rc, att.message)
            if att.category == "timeout":
                retry_same = True
            if att.category == "stall":
                retry_same = False     # the redirector may hand out the same dead server: go to the next source
            if not retry_same or attempt == attempts_per_source:
                break
            lo, hi = backoff
            time.sleep(min(hi, lo * (2 ** (attempt - 1))) + random.uniform(0, 1))
    return result
