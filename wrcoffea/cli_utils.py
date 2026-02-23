from __future__ import annotations

import csv
import logging
import re
from collections.abc import Mapping
from pathlib import Path

from wrcoffea.fileset_validation import validate_fileset_schema, validate_selection
from wrcoffea.era_utils import ERA_MAPPING, get_era_details, load_json
from wrcoffea.analysis_config import DEFAULT_MC_TAG, DY_VARIANTS

logger = logging.getLogger(__name__)

COMPOSITE_SAMPLES: dict[str, list[str]] = {
    "all":    ["EGamma", "Muon", "DYJets", "tt_tW", "Nonprompt", "Other", "Signal"],
    "data":   ["EGamma", "Muon"],
    "bkg":    ["DYJets", "tt_tW", "Nonprompt", "Other"],
    "signal": ["Signal"],
    "mc":     ["DYJets", "tt_tW", "Nonprompt", "Other", "Signal"],
}

# Sub-samples within tt_tW, filtered by dataset name pattern.
# Use "TTto2L2Nu" (dileptonic) specifically — semileptonic ttbar
# (e.g. TTtoLNu2Q) belongs in Nonprompt, not here.
_PHYSICS_SUBGROUPS: dict[str, tuple[str, list[str]]] = {
    "TTbar": ("tt_tW", ["TTto2L2Nu"]),
    "tW":    ("tt_tW", ["TWminus", "TbarWplus"]),
}

# WR masses to scan when selecting default signal points.
# For each WR value, the min, median, and max N are picked from the era's CSV.
# Edit this list to change which signal points are included in composite runs.
SIGNAL_WR_GRID = [2000, 4000, 6000]

_MASS_RE_WR_N = re.compile(r"^WR(?P<wr>\d+)_N(?P<n>\d+)$")
_MASS_RE_MWR_MN = re.compile(r"^MWR(?P<wr>\d+)_MN(?P<n>\d+)$")


def list_eras() -> list[str]:
    """Return supported era strings in the repo's preferred order."""

    # Dict insertion order is intentional here (curated in era_utils).
    return list(ERA_MAPPING.keys())


def list_samples() -> list[str]:
    """Return supported top-level sample choices for the CLI."""

    return ["DYJets", "tt_tW", "TTbar", "tW", "Nonprompt", "Other", "EGamma", "Muon", "Signal"]


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
    return any(
        re.search(re.escape(x) + r"(?!\d)", sample_name)
        for x in needles
    )


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


def select_default_signal_points(era: str) -> list[str]:
    """Return the default grid of signal mass points for an era.

    For each WR value in ``SIGNAL_WR_GRID``, picks (min, median, max) N
    from the era's mass-point CSV.  Falls back to 9 evenly spaced points
    if none of the target WR values exist.

    Edit ``SIGNAL_WR_GRID`` to change which WR values are scanned.
    """
    csv_path = Path(f"data/signal_points/{era}_mass_points.csv")
    all_masses = load_masses_from_csv(csv_path)

    # Parse into {wr: [n, ...]} mapping.
    by_wr: dict[int, list[int]] = {}
    for mp in all_masses:
        m = _MASS_RE_WR_N.match(mp)
        if m:
            by_wr.setdefault(int(m.group("wr")), []).append(int(m.group("n")))

    selected: list[str] = []
    for wr in SIGNAL_WR_GRID:
        ns = sorted(by_wr.get(wr, []))
        if not ns:
            continue
        picks = [ns[0]]
        if len(ns) > 2:
            picks.append(ns[len(ns) // 2])
        if len(ns) > 1:
            picks.append(ns[-1])
        seen: set[int] = set()
        for n in picks:
            if n not in seen:
                seen.add(n)
                selected.append(f"WR{wr}_N{n}")

    if not selected:
        # Fallback: 9 evenly spaced points from the full list.
        if not all_masses:
            return []
        k = min(9, len(all_masses))
        if k == 1:
            return [all_masses[0]]
        idxs = [round(i * (len(all_masses) - 1) / (k - 1)) for i in range(k)]
        seen_str: set[str] = set()
        for ix in idxs:
            mp = all_masses[ix]
            if mp not in seen_str:
                seen_str.add(mp)
                selected.append(mp)

    return selected


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

    # Sub-sample filtering (e.g., TTbar or tW within tt_tW).
    if desired_process in _PHYSICS_SUBGROUPS:
        parent_group, prefixes = _PHYSICS_SUBGROUPS[desired_process]
        return {
            ds: data
            for ds, data in fileset.items()
            if (data.get("metadata") or {}).get("physics_group") == parent_group
            and any(
                (data.get("metadata") or {}).get("sample", "").startswith(p)
                for p in prefixes
            )
        }

    return {
        ds: data
        for ds, data in fileset.items()
        if (data.get("metadata") or {}).get("physics_group") == desired_process
    }


def build_fileset_path(*, era: str, sample: str, unskimmed: bool, dy: str) -> Path:
    run, year, era_name = get_era_details(era)

    # Resolve sub-samples (TTbar, tW) to their parent group for fileset lookup.
    effective_sample = _PHYSICS_SUBGROUPS[sample][0] if sample in _PHYSICS_SUBGROUPS else sample

    if effective_sample in ["EGamma", "Muon"]:
        filename = f"{era_name}_data_fileset.json"
    elif effective_sample == "Signal":
        filename = f"{era_name}_signal_fileset.json"
    elif dy is not None:
        filename = f"{era_name}_mc_dy_{dy}_fileset.json"
    else:
        tag = DEFAULT_MC_TAG.get(era_name)
        if tag is None:
            raise ValueError(
                f"No default MC fileset tag for era '{era_name}'. "
                "Add it to DEFAULT_MC_TAG in analysis_config.py."
            )
        filename = f"{era_name}_mc_{tag}_fileset.json"

    base = Path("data/filesets") / run / year / era_name
    if unskimmed:
        base = base / "unskimmed"
    else:
        base = base / "skimmed"
    return base / filename


def load_and_select_fileset(
    *,
    filepath: Path,
    desired_process: str,
    mass: str | None,
    maxfiles: int | None = None,
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

    if maxfiles is not None:
        from coffea.dataset_tools import max_files
        filtered_fileset = max_files(filtered_fileset, maxfiles)

    return filtered_fileset


def load_composite_fileset(
    *,
    era: str,
    composite_mode: str,
    unskimmed: bool,
    dy: str | None,
    signal_points: list[str] | None = None,
    maxfiles: int | None = None,
) -> dict:
    """Merge fileset JSONs for a composite mode (all/data/bkg) into one dict.

    When the composite mode includes "Signal", ``signal_points`` controls
    which mass points are loaded.  If ``None``, defaults are selected via
    :func:`select_default_signal_points`.
    """
    groups = COMPOSITE_SAMPLES[composite_mode]
    combined: dict = {}
    # Cache loaded filesets by filepath to avoid redundant I/O and log spam.
    _fileset_cache: dict[Path, dict] = {}

    def _get_fileset(fp: Path) -> dict:
        if fp not in _fileset_cache:
            if not fp.exists():
                raise FileNotFoundError(
                    f"Fileset JSON not found: {fp}. "
                    "Create filesets first (see docs/filesets.md), or check --unskimmed and era/sample names."
                )
            raw = load_json(str(fp))
            validate_fileset_schema(raw, filepath=str(fp))
            _fileset_cache[fp] = raw
        return _fileset_cache[fp]

    for group in groups:
        if group == "Signal":
            pts = signal_points if signal_points is not None else select_default_signal_points(era)
            if not pts:
                logger.warning("No signal points selected for era %s; skipping Signal.", era)
                continue
            filepath = build_fileset_path(era=era, sample="Signal", unskimmed=unskimmed, dy=dy)
            raw = _get_fileset(filepath)
            for mass in pts:
                filtered = filter_by_process(raw, "Signal", mass=mass)
                validate_selection(filtered, desired_process="Signal", mass=mass, preprocessed_fileset=raw)
                if maxfiles is not None:
                    from coffea.dataset_tools import max_files
                    filtered = max_files(filtered, maxfiles)
                combined.update(filtered)
        else:
            filepath = build_fileset_path(era=era, sample=group, unskimmed=unskimmed, dy=dy)
            raw = _get_fileset(filepath)
            filtered = filter_by_process(raw, group)
            validate_selection(filtered, desired_process=group, mass=None, preprocessed_fileset=raw)
            if maxfiles is not None:
                from coffea.dataset_tools import max_files
                filtered = max_files(filtered, maxfiles)
            combined.update(filtered)
    return combined


def build_sample_to_group_map(fileset: dict, *, signal_points: list[str] | None = None) -> dict[str, str]:
    """Map processor output keys (metadata['sample']) to physics_group.

    For signal datasets, the group is ``"Signal:<mass_point>"`` so that
    each mass point gets its own output ROOT file.
    """
    mapping: dict[str, str] = {}
    for ds, info in fileset.items():
        md = info.get("metadata", {})
        sample = md.get("sample", ds)
        group = md.get("physics_group", "Unknown")

        if group == "Signal" and signal_points:
            for mp in signal_points:
                if signal_sample_matches_mass(sample, mp):
                    group = f"Signal:{mp}"
                    break

        mapping[sample] = group
    return mapping
