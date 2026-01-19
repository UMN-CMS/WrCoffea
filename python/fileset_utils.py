from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_config_path(config_path: Path) -> tuple[Path, str, str, str]:
    """Parse a config path under data/configs/<Run>/<Year>/<Era>/...json.

    Returns (data_root, run, year, era).
    """

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg_dir = config_path.parent

    cur = cfg_dir
    while cur.name != "configs":
        if cur.parent == cur:
            raise ValueError(
                f"Could not locate a parent 'configs' directory for {config_path}. "
                "Expected a path like data/configs/<Run>/<Year>/<Era>/...json."
            )
        cur = cur.parent

    data_root = cur.parent
    rel = cfg_dir.relative_to(data_root / "configs")

    if len(rel.parts) < 3:
        raise ValueError(
            f"Config path {config_path} is too shallow under data/configs; "
            "expected data/configs/<Run>/<Year>/<Era>/..."
        )

    run, year, era = rel.parts[:3]
    return data_root, run, year, era


def sample_from_config_filename(config_path: Path, *, era: str) -> str:
    """Derive the sample tag from an era-prefixed config filename.

    Example: data/configs/.../Run3Summer22_mc_lo_dy.json -> mc_lo_dy
    """

    stem = config_path.stem
    prefix = f"{era}_"
    if not stem.startswith(prefix):
        logger.warning("Filename '%s' doesn't start with '%s'", stem, prefix)
        return stem
    return stem[len(prefix) :]


def normalize_skimmed_sample(sample: str) -> str:
    """Keep historical behavior for skimmed fileset naming.

    The skimmed fileset script collapses any 'mc_*' config to a single 'mc'
    output name (e.g. mc_lo_dy -> mc).
    """

    return "mc" if "mc" in sample else sample


def output_dir(*, data_root: Path, run: str, year: str, era: str) -> Path:
    return data_root / "filesets" / run / year / era 


def write_fileset_json(path: Path, fileset: Mapping, *, indent: int = 2, sort_keys: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(fileset, f, indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=str)


def rename_dataset_key_to_sample(fileset: dict) -> dict:
    """Rename metadata.dataset -> metadata.sample if present."""

    for entry in fileset.values():
        md = entry.get("metadata", {})
        if "dataset" in md:
            md["sample"] = md.pop("dataset")
    return fileset
