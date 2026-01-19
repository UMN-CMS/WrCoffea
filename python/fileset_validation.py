from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)


def _short_list(items: list[str], *, limit: int = 8) -> str:
    if not items:
        return "(none)"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit} more)"


def validate_fileset_schema(fileset: object, *, filepath: str | None = None) -> None:
    """Validate that the fileset matches what `bin/run_analysis.py` expects.

    Expected structure:
      {dataset_key: {"files": {path: "Events", ...}, "metadata": {...}}, ...}
    """
    where = f" ({filepath})" if filepath else ""

    if not isinstance(fileset, Mapping):
        raise ValueError(f"Fileset must be a JSON object (dict-like){where}.")

    if not fileset:
        raise ValueError(f"Fileset is empty{where}.")

    # Validate a few entries (full validation can be expensive on huge filesets)
    checked = 0
    for ds_key, ds_val in fileset.items():
        checked += 1
        if not isinstance(ds_key, str):
            raise ValueError(f"Fileset dataset key must be a string{where}.")
        if not isinstance(ds_val, Mapping):
            raise ValueError(f"Fileset['{ds_key}'] must be an object{where}.")

        files = ds_val.get("files")
        md = ds_val.get("metadata")
        if not isinstance(files, Mapping):
            raise ValueError(f"Fileset['{ds_key}']['files'] must be an object mapping file→treename{where}.")
        if not isinstance(md, Mapping):
            raise ValueError(f"Fileset['{ds_key}']['metadata'] must be an object{where}.")

        # Common required metadata fields for this repo
        if "sample" not in md:
            raise ValueError(f"Fileset['{ds_key}']['metadata']['sample'] is missing{where}.")

        if checked >= 10:
            break


def validate_selection(
    filtered_fileset: Mapping,
    *,
    desired_process: str,
    mass: str | None,
    preprocessed_fileset: Mapping | None = None,
) -> None:
    """Fail fast if filtering produced no datasets, with actionable hints."""

    if filtered_fileset:
        return

    # Provide helpful hints based on what we know.
    if desired_process == "Signal":
        candidates: list[str] = []
        if preprocessed_fileset:
            for ds in preprocessed_fileset.values():
                md = ds.get("metadata") or {}
                s = md.get("sample")
                if isinstance(s, str):
                    candidates.append(s)
        raise ValueError(
            "Signal selection matched 0 datasets. "
            f"Mass was '{mass}'. "
            "Check that your signal fileset contains this mass point in metadata.sample. "
            f"Available sample strings (subset): {_short_list(sorted(set(candidates)))}"
        )

    # Non-signal: show available physics groups if present
    groups: list[str] = []
    if preprocessed_fileset:
        for ds in preprocessed_fileset.values():
            md = ds.get("metadata") or {}
            g = md.get("physics_group")
            if isinstance(g, str):
                groups.append(g)

    raise ValueError(
        f"Selection matched 0 datasets for sample '{desired_process}'. "
        f"Available physics_group values (subset): {_short_list(sorted(set(groups)))}"
    )
