"""DAS query utilities for the WrCoffea skimming pipeline.

Provides functions for validating DAS dataset paths, querying dasgoclient
for file lists, and converting logical file names to XRootD URLs.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Map analysis era name → subdirectory under data/skims/ (and data/configs/, etc.)
ERA_SUBDIRS = {
    "RunIISummer20UL18": "RunII/2018/RunIISummer20UL18",
    "Run3Summer22": "Run3/2022/Run3Summer22",
    "Run3Summer22EE": "Run3/2022/Run3Summer22EE",
    "Run3Summer23": "Run3/2023/Run3Summer23",
    "Run3Summer23BPix": "Run3/2023/Run3Summer23BPix",
    "RunIII2024Summer24": "Run3/2024/RunIII2024Summer24",
}

REDIRECTOR = "root://cmsxrootd.fnal.gov/"
DASGOCLIENT_PATH = "/cvmfs/cms.cern.ch/common/dasgoclient"
SCRATCH_ROOT = Path(
    f"/uscmst1b_scratch/lpc1/3DayLifetime/{os.environ.get('USER', 'unknown')}/skims"
)


def validate_das_path(das_path: str) -> tuple[str, str, str]:
    """Parse and validate a DAS dataset path.

    DAS paths have format: /<primary_dataset>/<campaign>/<datatier>

    Returns
    -------
    (primary_dataset, campaign, datatier)

    Raises
    ------
    ValueError
        If the path does not have the expected format.
    """
    if not das_path.startswith("/"):
        raise ValueError(
            f"DAS path must start with '/': {das_path!r}"
        )
    parts = das_path.strip("/").split("/")
    if len(parts) != 3:
        raise ValueError(
            f"DAS path must have 3 components (/<primary>/<campaign>/<tier>): {das_path!r}"
        )
    primary_dataset, campaign, datatier = parts
    if datatier not in ("NANOAOD", "NANOAODSIM"):
        raise ValueError(
            f"Expected datatier NANOAOD or NANOAODSIM, got: {datatier!r}"
        )
    return primary_dataset, campaign, datatier


def primary_dataset_from_das_path(das_path: str) -> str:
    """Extract the primary dataset name from a DAS path."""
    return validate_das_path(das_path)[0]


def era_from_campaign(campaign: str) -> str | None:
    """Extract the analysis era name from a DAS campaign string.

    The campaign looks like ``RunIII2024Summer24NanoAODv15-150X_...``.
    We strip the ``NanoAOD...`` suffix to get the era prefix, then match
    against known eras in :data:`ERA_SUBDIRS`.

    Returns ``None`` if no known era matches.
    """
    m = re.match(r"^(.+?)NanoAOD", campaign)
    if not m:
        return None
    prefix = m.group(1)
    if prefix in ERA_SUBDIRS:
        return prefix
    return None


def check_dasgoclient() -> str:
    """Verify dasgoclient is available.

    Returns the path to dasgoclient.

    Raises
    ------
    FileNotFoundError
        If dasgoclient cannot be found.
    """
    path = shutil.which("dasgoclient")
    if path:
        return path
    if Path(DASGOCLIENT_PATH).exists():
        return DASGOCLIENT_PATH
    raise FileNotFoundError(
        "dasgoclient not found. Ensure CMS software environment is set up "
        "(source /cvmfs/cms.cern.ch/cmsset_default.sh) or add dasgoclient to PATH."
    )


def check_grid_proxy() -> None:
    """Verify a valid grid proxy exists.

    Raises
    ------
    RuntimeError
        If no valid proxy or proxy is about to expire.
    """
    result = subprocess.run(
        ["voms-proxy-info", "--timeleft"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "No valid grid proxy found. Run: voms-proxy-init -voms cms"
        )
    try:
        timeleft = int(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            f"Could not parse proxy timeleft: {result.stdout.strip()!r}"
        )
    if timeleft < 60:
        raise RuntimeError(
            f"Grid proxy expires in {timeleft}s. Renew with: voms-proxy-init -voms cms"
        )


def query_das_files(das_path: str) -> list[str]:
    """Query DAS for the logical file names of a dataset.

    Parameters
    ----------
    das_path : str
        Full DAS dataset path, e.g.
        ``/TTto2L2Nu_.../Run3Summer24NanoAODv15-.../NANOAODSIM``

    Returns
    -------
    list[str]
        Sorted list of logical file names (LFNs), e.g.
        ``/store/mc/.../file.root``

    Raises
    ------
    RuntimeError
        If dasgoclient returns a non-zero exit code or no files.
    """
    dasgoclient = check_dasgoclient()
    cmd = [dasgoclient, "-query", f"file dataset={das_path}"]
    logger.info("Querying DAS: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(
            f"dasgoclient failed for {das_path}: {result.stderr.strip()}"
        )
    files = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not files:
        raise RuntimeError(f"No files returned by DAS for {das_path}")
    logger.info("Found %d files", len(files))
    return files


def das_files_to_urls(lfns: list[str]) -> list[str]:
    """Prepend XRootD redirector to each logical file name."""
    return [f"{REDIRECTOR}{lfn}" for lfn in lfns]


def era_from_das_path(das_path: str) -> str | None:
    """Return the era name (e.g. ``'Run3Summer22'``) from a DAS path, or ``None``."""
    _, campaign, _ = validate_das_path(das_path)
    return era_from_campaign(campaign)


def infer_category(primary_ds: str) -> str:
    """Return ``'signals'``, ``'data'``, or ``'backgrounds'`` for Wisconsin upload paths."""
    if "WR" in primary_ds:
        return "signals"
    if "EGamma" in primary_ds or "Muon" in primary_ds:
        return "data"
    return "backgrounds"


def base_dir_for_era(era: str, *, scratch: bool = False) -> Path:
    """Return the era-level base directory for a known era name.

    Unlike :func:`infer_base_dir`, this takes the era name directly
    (e.g. from config metadata) rather than inferring it from a DAS path.
    This is necessary for data datasets whose campaign strings don't
    contain ``NanoAOD``.

    Raises
    ------
    ValueError
        If *era* is not in :data:`ERA_SUBDIRS`.
    """
    if era not in ERA_SUBDIRS:
        raise ValueError(f"Unknown era {era!r}. Known: {list(ERA_SUBDIRS)}")
    root = SCRATCH_ROOT if scratch else Path("data/skims")
    return root / ERA_SUBDIRS[era]


def infer_base_dir(das_path: str, *, scratch: bool = False) -> Path:
    """Derive the era-level base directory from a DAS path.

    Returns ``data/skims/<run>/<year>/<era>/``
    (e.g. ``data/skims/Run3/2024/RunIII2024Summer24/``).

    When *scratch* is True, returns the equivalent path under
    ``/uscmst1b_scratch/lpc1/3DayLifetime/$USER/skims/`` instead.

    Falls back to the root skims directory if the campaign
    does not match any known era.
    """
    _, campaign, _ = validate_das_path(das_path)
    era = era_from_campaign(campaign)
    root = SCRATCH_ROOT if scratch else Path("data/skims")
    if era is not None:
        return root / ERA_SUBDIRS[era]
    return root


def infer_output_dir(das_path: str, *, scratch: bool = False) -> Path:
    """Derive default output directory from a DAS path.

    Returns ``data/skims/<run>/<year>/<era>/files/<primary_dataset>/``
    (e.g. ``data/skims/Run3/2024/RunIII2024Summer24/files/TTto2L2Nu_.../``).

    When *scratch* is True, returns the equivalent path under the
    3DayLifetime scratch area.
    """
    primary_ds = primary_dataset_from_das_path(das_path)
    return infer_base_dir(das_path, scratch=scratch) / "files" / primary_ds
