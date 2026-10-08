"""DBS and Rucio access for the skim campaign.

Everything here talks to CMS services over HTTPS with the user's grid proxy, so
it works on a laptop without ``dasgoclient``:

- :class:`DBSClient` lists datasets, files (with adler32 checksums and event
  counts), run numbers and dataset summaries from the DBS REST API.
- :class:`RucioClient` resolves the replicas of a logical file name so a file
  that fails through a redirector can be fetched from a specific site.
- :func:`xrootd_sources` orders the candidate URLs for one file.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DBS_URL = "https://cmsweb.cern.ch/dbs/prod/global/DBSReader/"
RUCIO_AUTH_URL = "https://cms-rucio-auth.cern.ch/auth/x509_proxy"
RUCIO_URL = "https://cms-rucio.cern.ch/"

# Redirectors tried in this order before any site-specific replica.
REDIRECTORS = [
    "root://cmsxrootd.fnal.gov/",
    "root://cms-xrd-global.cern.ch/",
    "root://xrootd-cms.infn.it/",
]

# Site preference for direct replica access (prefix match on the RSE name).
# US sites first: the laptop and LPC both sit on the US side of the WAN.
PREFERRED_RSE_PREFIXES = (
    "T1_US_FNAL_Disk", "T2_US_Wisconsin", "T2_US_Purdue", "T2_US_Nebraska",
    "T2_US_Caltech", "T2_US_UCSD", "T2_US_MIT", "T2_US_Florida", "T2_US_Vanderbilt",
    "T2_CH_CERN", "T1_", "T2_", "T3_",
)


def proxy_path() -> str:
    """Return the grid proxy file: ``$X509_USER_PROXY``, else the usual defaults."""
    env = os.environ.get("X509_USER_PROXY")
    if env:
        return env
    home = os.path.expanduser("~/.x509up_proxy")
    if os.path.exists(home):
        return home
    return f"/tmp/x509up_u{os.getuid()}"


def cert_dir() -> str | None:
    """Return the CA certificate directory, or None for the system default."""
    for cand in (os.environ.get("X509_CERT_DIR"),
                 os.path.expanduser("~/.grid-security/certificates"),
                 "/etc/grid-security/certificates"):
        if cand and os.path.isdir(cand):
            return cand
    return None


def ssl_context() -> ssl.SSLContext:
    """SSL context authenticating with the grid proxy (cert and key in one file)."""
    ctx = ssl.create_default_context(capath=cert_dir())
    proxy = proxy_path()
    if not os.path.exists(proxy):
        raise FileNotFoundError(
            f"Grid proxy not found at {proxy}. Run voms-proxy-init (or ~/bin/renew-proxy.sh) first."
        )
    ctx.load_cert_chain(proxy, proxy)
    return ctx


def is_das_path(key: str) -> bool:
    """True for ``/<primary>/<campaign>/<tier>`` strings."""
    parts = key.strip("/").split("/")
    return key.startswith("/") and len(parts) == 3 and parts[2] in ("NANOAOD", "NANOAODSIM")


@dataclass(frozen=True)
class FileRecord:
    """One NanoAOD file as DBS knows it."""
    lfn: str
    size: int
    events: int
    adler32: str
    valid: bool
    dataset: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FileRecord":
        return cls(lfn=d["lfn"], size=int(d["size"]), events=int(d["events"]),
                   adler32=str(d["adler32"]), valid=bool(d.get("valid", True)),
                   dataset=d.get("dataset", ""))


class _HttpClient:
    def __init__(self, ctx: ssl.SSLContext | None = None, *, retries: int = 4,
                 timeout: float = 120.0, backoff: float = 2.0):
        self._ctx = ctx
        self.retries = retries
        self.timeout = timeout
        self.backoff = backoff

    @property
    def ctx(self) -> ssl.SSLContext:
        if self._ctx is None:
            self._ctx = ssl_context()
        return self._ctx

    def request(self, url: str, headers: dict | None = None) -> tuple[bytes, dict]:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(url, headers=headers or {})
                with urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout) as r:
                    return r.read(), {k.lower(): v for k, v in r.headers.items()}
            except urllib.error.HTTPError as e:
                # 4xx is a bad request, not transient; 5xx and network errors are retried.
                if 400 <= e.code < 500:
                    raise
                last = e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
            if attempt < self.retries:
                sleep = self.backoff ** (attempt - 1)
                logger.warning("HTTP attempt %d/%d failed for %s: %s (retry in %.0fs)",
                               attempt, self.retries, url.split("?")[0], last, sleep)
                time.sleep(sleep)
        raise RuntimeError(f"HTTP request failed after {self.retries} attempts: {url}: {last}")


class DBSClient(_HttpClient):
    """Thin DBS3 REST reader (global instance)."""

    def get(self, endpoint: str, **params) -> list:
        url = DBS_URL + endpoint + "?" + urllib.parse.urlencode(params)
        body, _ = self.request(url)
        return json.loads(body)

    def datasets(self, pattern: str, *, access: str = "VALID") -> list[str]:
        """Dataset names matching a DAS-style wildcard pattern."""
        rows = self.get("datasets", dataset=pattern, dataset_access_type=access)
        return sorted(r["dataset"] for r in rows)

    def dataset_status(self, dataset: str) -> str | None:
        rows = self.get("datasets", dataset=dataset, dataset_access_type="*", detail=1)
        return rows[0]["dataset_access_type"] if rows else None

    def files(self, dataset: str, *, valid_only: bool = True) -> list[FileRecord]:
        """All files of a dataset with sizes, event counts and adler32 checksums."""
        params = {"dataset": dataset, "detail": 1}
        if valid_only:
            params["validFileOnly"] = 1
        rows = self.get("files", **params)
        out = [
            FileRecord(lfn=r["logical_file_name"], size=int(r["file_size"]),
                       events=int(r["event_count"]), adler32=str(r["adler32"]).lower(),
                       valid=bool(r["is_file_valid"]), dataset=dataset)
            for r in rows
        ]
        return sorted(out, key=lambda f: f.lfn)

    def summary(self, dataset: str) -> dict:
        rows = self.get("filesummaries", dataset=dataset, validFileOnly=1)
        if not rows:
            return {"num_file": 0, "num_event": 0, "file_size": 0}
        r = rows[0]
        return {"num_file": int(r["num_file"]), "num_event": int(r["num_event"]),
                "file_size": int(r["file_size"]), "num_lumi": int(r.get("num_lumi", 0))}

    def runs(self, dataset: str) -> list[int]:
        rows = self.get("runs", dataset=dataset)
        runs: list[int] = []
        for r in rows:
            v = r.get("run_num")
            if isinstance(v, list):
                runs.extend(int(x) for x in v)
            elif v is not None:
                runs.append(int(v))
        return sorted(set(runs))


class RucioClient(_HttpClient):
    """Replica lookup through the CMS Rucio REST API with x509 proxy auth."""

    def __init__(self, account: str, ctx: ssl.SSLContext | None = None, **kw):
        super().__init__(ctx, **kw)
        self.account = account
        self._token: str | None = None
        self._token_time = 0.0

    def token(self) -> str:
        # Tokens last an hour; refresh after 45 minutes.
        if self._token is None or time.time() - self._token_time > 45 * 60:
            _, headers = self.request(RUCIO_AUTH_URL, headers={"X-Rucio-Account": self.account})
            tok = headers.get("x-rucio-auth-token")
            if not tok:
                raise RuntimeError(f"Rucio authentication returned no token for account {self.account!r}")
            self._token, self._token_time = tok, time.time()
        return self._token

    def replicas(self, lfn: str) -> list[dict]:
        """Return ``[{rse, pfn, state}, ...]`` for one LFN (disk and tape alike).

        No scheme filter is passed: CMS RSEs advertise WAN access as ``davs``
        and a ``schemes=root`` request returns no PFNs at all.
        """
        url = RUCIO_URL + "replicas/cms/" + urllib.parse.quote(lfn, safe="")
        body, _ = self.request(url, headers={"X-Rucio-Auth-Token": self.token(),
                                             "Accept": "application/x-json-stream"})
        out: list[dict] = []
        for line in body.decode().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            states = d.get("states") or {}
            for rse, pfns in (d.get("rses") or {}).items():
                for pfn in pfns:
                    out.append({"rse": rse, "pfn": pfn, "state": states.get(rse, "UNKNOWN")})
        return out


def _rse_rank(rse: str) -> int:
    for i, prefix in enumerate(PREFERRED_RSE_PREFIXES):
        if rse.startswith(prefix):
            return i
    return len(PREFERRED_RSE_PREFIXES)


def xrootd_sources(lfn: str, rucio: RucioClient | None = None, *,
                   redirectors: list[str] | None = None,
                   include_tape: bool = False) -> list[str]:
    """Ordered candidate URLs for one file.

    Order: the first redirector (it finds the nearest healthy replica on its
    own), the remaining redirectors, then every disk replica Rucio knows
    about in site-preference order.  CMS Rucio publishes WAN replicas as
    ``davs://`` WebDAV URLs; they are returned as ``https://`` so
    :func:`wrcoffea.skim.fetch.fetch_file` can stream them with the proxy
    when every XRootD route failed.  Tape-only replicas are listed last and
    only when *include_tape* is set (reading them triggers a stage-in).
    """
    reds = list(redirectors or REDIRECTORS)
    urls: list[str] = [red.rstrip("/") + "/" + lfn for red in reds]
    if rucio is None:
        return urls
    try:
        reps = rucio.replicas(lfn)
    except Exception as e:  # replica lookup is best effort
        logger.warning("Rucio replica lookup failed for %s: %s", lfn, e)
        reps = []
    disk = [r for r in reps if r["state"] == "AVAILABLE" and "Tape" not in r["rse"]]
    tape = [r for r in reps if r["state"] == "AVAILABLE" and "Tape" in r["rse"]]
    for r in sorted(disk, key=lambda r: (_rse_rank(r["rse"]), r["rse"])):
        url = r["pfn"].replace("davs://", "https://", 1)
        if url not in urls:
            urls.append(url)
    if include_tape:
        for r in tape:
            url = r["pfn"].replace("davs://", "https://", 1)
            if url not in urls:
                urls.append(url)
    return urls
