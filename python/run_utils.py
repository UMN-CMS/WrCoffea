from __future__ import annotations

import csv
import logging
import re
from collections.abc import Mapping
from pathlib import Path

from python.fileset_validation import validate_fileset_schema, validate_selection
from python.preprocess_utils import get_era_details, load_json

logger = logging.getLogger(__name__)

_MASS_RE_WR_N = re.compile(r"^WR(?P<wr>\d+)_N(?P<n>\d+)$")
_MASS_RE_MWR_MN = re.compile(r"^MWR(?P<wr>\d+)_MN(?P<n>\d+)$")


def normalize_mass_point(mass: str | None) -> str | None:
    """Normalize user-provided mass points to the repo's canonical format.

    Canonical: WR<wr>_N<n>
    Legacy accepted: MWR<wr>_MN<n> (auto-converted)
    """

    if mass is None:
        return None

    mass = mass.strip()
    if _MASS_RE_WR_N.match(mass):
        return mass

    m = _MASS_RE_MWR_MN.match(mass)
    if m:
        converted = f"WR{m.group('wr')}_N{m.group('n')}"
        logger.warning(
            "Interpreting legacy mass '%s' as '%s' (canonical WR/N format).",
            mass,
            converted,
        )
        return converted

    return mass


def signal_sample_matches_mass(sample_name: str, mass_wr_n: str) -> bool:
    """True if a signal dataset/sample string corresponds to the mass point.

    Keep matching flexible because dataset names vary across campaigns.

    Examples observed in this repo:
      - ..._MWR2000_N1100_...
      - ..._MWR600_MN100_...
    """

    if not sample_name:
        return False

    m = _MASS_RE_WR_N.match(mass_wr_n)
    if not m:
        return mass_wr_n in sample_name

    wr = m.group("wr")
    n = m.group("n")
    needles = (
        mass_wr_n,
        f"MWR{wr}_N{n}",
        f"MWR{wr}_MN{n}",
    )
    return any(x in sample_name for x in needles)


def load_masses_from_csv(file_path: Path) -> list[str]:
    mass_choices: list[str] = []
    try:
        with open(file_path, mode="r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)
            for row in csv_reader:
                if len(row) >= 2:
                    wr_mass = row[0].strip()
                    n_mass = row[1].strip()
                    mass_choices.append(f"WR{wr_mass}_N{n_mass}")
    except FileNotFoundError:
        logger.error("Mass CSV file not found at: %s", file_path)
        raise
    except Exception:
        logger.exception("Error loading mass CSV: %s", file_path)
        raise
    return mass_choices


def filter_by_process(fileset: Mapping, desired_process: str, *, mass: str | None = None) -> dict:
    if desired_process == "Signal":
        if mass is None:
            raise ValueError("Signal filtering requires a mass point.")
        return {
            ds: data
            for ds, data in fileset.items()
            if signal_sample_matches_mass(
                (data.get("metadata") or {}).get("sample", ""),
                mass,
            )
        }

    return {
        ds: data
        for ds, data in fileset.items()
        if (data.get("metadata") or {}).get("physics_group") == desired_process
    }


def build_fileset_path(*, era: str, sample: str, unskimmed: bool) -> Path:
    run, year, era_name = get_era_details(era)
    subdir = "unskimmed" if unskimmed else "skimmed"

    if sample in ["EGamma", "Muon"]:
        filename = f"{era_name}_{sample}_fileset.json" if unskimmed else f"{era_name}_data_skimmed_fileset.json"
    elif sample == "Signal":
        filename = f"{era_name}_{sample}_fileset.json" if unskimmed else f"{era_name}_signal_skimmed_fileset.json"
    else:
        filename = f"{era_name}_{sample}_fileset.json" if unskimmed else f"{era_name}_mc_skimmed_fileset.json"

    return Path("data/jsons") / run / year / era_name / subdir / filename


def load_and_select_fileset(
    *,
    filepath: Path,
    desired_process: str,
    mass: str | None,
) -> dict:
    if not filepath.exists():
        raise FileNotFoundError(
            f"Fileset JSON not found: {filepath}. "
            "Create filesets first (see docs/filesets.md), or check --unskimmed and era/sample names."
        )

    preprocessed_fileset = load_json(str(filepath))
    validate_fileset_schema(preprocessed_fileset, filepath=str(filepath))

    filtered_fileset = filter_by_process(preprocessed_fileset, desired_process, mass=mass)
    validate_selection(
        filtered_fileset,
        desired_process=desired_process,
        mass=mass,
        preprocessed_fileset=preprocessed_fileset,
    )

    return filtered_fileset
