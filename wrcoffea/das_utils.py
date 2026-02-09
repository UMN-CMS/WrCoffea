"""DAS query utilities for the WrCoffea skimming pipeline.

Provides functions for validating DAS dataset paths, querying dasgoclient
for file lists, and converting logical file names to XRootD URLs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REDIRECTOR = "root://cmsxrootd.fnal.gov/"
DASGOCLIENT_PATH = "/cvmfs/cms.cern.ch/common/dasgoclient"


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


def infer_output_dir(das_path: str) -> Path:
    """Derive default output directory from a DAS path.

    Returns ``data/skims/<primary_dataset>/``.
    """
    primary_ds = primary_dataset_from_das_path(das_path)
    return Path("data/skims") / primary_ds
