"""Sum per-dataset histograms, check systematic coverage, and write ROOT files.

Systematic bookkeeping
----------------------
Each dataset payload produced by ``WrAnalysis.process`` carries, next to its
histograms, three coverage records (see ``wrcoffea.systematics``):

- ``n_chunks``: number of processed chunks,
- ``syst_coverage[region][label]``: number of chunks in which every requested
  histogram for that (region, label) was filled,
- ``dataset_meta``: ``{"era": {...}, "datatype": {...}, "physics_group": {...},
  "syst_config": {...}}`` where ``syst_config`` is the canonical production
  configuration (families, omitted sources, observables) of the run.

At save time every dataset gets a status per (region, label):

======================  =====================================================
status                  contribution to the group's varied histogram
======================  =====================================================
evaluated               its own varied histogram (a legitimately empty one
                        stays empty)
unaffected              its nominal histogram (the registry says the source
                        does not apply to this dataset/region)
omitted                 nothing; recorded in the manifest, never presented
                        as complete coverage (the source was not requested
                        for this dataset)
unavailable             error: the source applies but the dataset did not
                        (fully) evaluate it, or its inputs are missing
======================  =====================================================

Expected labels come from the registry and the recorded production
configuration, never from what happened to be produced, so a variation that
is missing everywhere is still an error.  Chunks and datasets with different
production configurations are never merged.

Normalization sources (``lumi``) are materialised here by scaling the nominal.
A JSON manifest next to each ROOT file records what was requested, produced,
unaffected, and omitted per region.

Blinding is part of the production configuration: data of a blinded run must
not have filled any signal region (``wrcoffea.regions``), and the manifest's
``blinding`` entry says what a data file withholds.

Cutflows and the MC event-weight distributions are plain nested Hists,
summed over the datasets of a file and written next to the histograms
(``cutflow_resolved/``, ``cutflow_boosted/``, ``event_weights/``).

Legacy payloads without coverage records are summed and written as-is.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import uproot
from hist import Hist

from wrcoffea.era_utils import get_era_details
from wrcoffea.histograms import all_histogram_names
from wrcoffea.regions import BLINDED_CUTFLOW_STEPS, blinded_region_names, is_signal_region, region_names
from wrcoffea.systematics import (
    NOMINAL,
    NORMALIZATION,
    describe_syst_observables,
    required_labels,
    normalization_factor,
    normalize_families,
    normalize_omitted,
    parse_production_config,
    producible,
    resolve_syst_observables,
    source_of_label,
    systematics_for,
)

logger = logging.getLogger(__name__)

COVERAGE_KEYS = ("syst_coverage", "n_chunks", "dataset_meta")

STATUS_EVALUATED = "evaluated"
STATUS_UNAFFECTED = "unaffected"
STATUS_OMITTED = "omitted"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ABSENT = "absent"   # the dataset never filled this region at all

# ROOT directory of each cutflow kind (wrcoffea.regions.BLINDED_CUTFLOW_STEPS keys).
CUTFLOW_DIRS = {"resolved": "cutflow_resolved", "boosted": "cutflow_boosted"}
# ROOT directory of the MC event-weight distributions (<dir>/<region>/positive|negative).
EVENT_WEIGHTS_DIR = "event_weights"


class SystematicCoverageError(RuntimeError):
    """A requested systematic variation is missing or incomplete."""


# ---------------------------------------------------------------------------
# ROOT naming (unchanged)
# ---------------------------------------------------------------------------

def _normalize_syst_name(syst: str) -> str:
    """
    Turn 'RenFactScaleUp' -> 'renfactscaleup' (root-key friendly, lowercase).
    """
    return "".join(ch.lower() for ch in syst if ch.isalnum())


def _folder_and_hist_names(region: str, syst: str, hist_stem: str):
    if syst == NOMINAL:
        folder = f"{region}"
        hname = f"{hist_stem}_{region}"
    else:
        norm = _normalize_syst_name(syst)
        folder = f"syst_{norm}_{region}"
        hname = f"{hist_stem}_syst_{norm}_{region}"
    return folder, hname


# ---------------------------------------------------------------------------
# Cutflows and event-weight distributions (nested Hist dicts)
# ---------------------------------------------------------------------------

def _sum_cutflow_hists(my_hists, cutflow_keys=("cutflow", "__cutflow__")):
    """
    Sum ALL cutflow histograms across datasets, recursively.

    Supports structures like:
      cutflow = {
        "ee":   {"onecut": Hist, "cumulative": Hist, ...},
        "mumu": {...},
        "em":   {...},
      }

    Any nested Hist dict works the same way, e.g. the MC event-weight
    distributions ``event_weights = {region: {"positive": Hist, "negative": Hist}}``.

    Args:
        my_hists: Nested dataset histogram payload map.
        cutflow_keys: Ordered tuple of keys to look for in each dataset payload.
            The first present dict is used (legacy fallback supported).
    """
    def _merge(dst, src):
        # src can be Hist or dict
        if isinstance(src, Hist):
            if dst is None:
                return src.copy()
            if isinstance(dst, Hist):
                dst += src
                return dst
            # dst is dict -> inconsistent types; treat as error or ignore
            raise TypeError("Cutflow key has mixed types (Hist vs dict) across datasets.")
        elif isinstance(src, dict):
            if dst is None or isinstance(dst, Hist):
                dst = {}  # (re)start a dict
            for k, v in src.items():
                dst[k] = _merge(dst.get(k), v)
            return dst
        else:
            # unknown type: ignore
            return dst

    out = {}
    for dataset_payload in my_hists.values():
        cfmap = None
        for k in cutflow_keys:
            candidate = dataset_payload.get(k)
            if isinstance(candidate, dict):
                cfmap = candidate
                break
        if not isinstance(cfmap, dict):
            continue
        for k, v in cfmap.items():
            out[k] = _merge(out.get(k), v)
    return out


def _save_cutflows(root_file, cutflow_summed: Dict[str, dict], prefix: str):
    """
    Recursively write all cutflow histograms (or any nested Hist dict, such as
    the event-weight distributions) as ``<prefix>/<key>/.../<name>``, avoiding duplicate writes.
    If a given path is encountered twice, the second write is skipped to prevent ROOT ;2 cycles.
    """
    seen = set()

    def _recurse(prefix, obj):
        if isinstance(obj, Hist):
            if prefix not in seen:
                root_file[prefix] = obj
                seen.add(prefix)
            return
        if isinstance(obj, dict):
            for name, child in obj.items():
                _recurse(f"{prefix}/{name}", child)

    _recurse(prefix, cutflow_summed)


def _save_event_lists(root_file, my_hists):
    """Write the --save-events per-event lists as TTrees ``events_<region>``
    (branches ``mass`` and ``weight``), concatenated over the datasets of
    this output file.  No-op when the run was made without --save-events."""
    merged: Dict[str, Dict[str, list]] = {}
    for payload in my_hists.values():
        lists = payload.get("event_lists") if isinstance(payload, dict) else None
        if not lists:
            continue
        for region, cols in lists.items():
            dst = merged.setdefault(region, {"mass": [], "weight": []})
            for key in ("mass", "weight"):
                acc = cols.get(key)
                if acc is not None:
                    dst[key].append(np.asarray(getattr(acc, "value", acc), dtype=np.float64))
    for region, cols in sorted(merged.items()):
        mass = np.concatenate(cols["mass"]) if cols["mass"] else np.zeros(0)
        weight = np.concatenate(cols["weight"]) if cols["weight"] else np.zeros(0)
        if len(mass) != len(weight):
            raise RuntimeError(f"event list for {region}: {len(mass)} masses vs {len(weight)} weights")
        # a classic TTree (not an RNTuple), readable by every ROOT in use
        tree = root_file.mktree(f"events_{region}", {"mass": np.float64, "weight": np.float64})
        if len(mass):
            tree.extend({"mass": mass, "weight": weight})
        logger.info("  event list %s: %d events, sum of weights %.3f", region, len(mass), weight.sum())


# ---------------------------------------------------------------------------
# Coverage records
# ---------------------------------------------------------------------------

@dataclass
class DatasetCoverage:
    """Coverage bookkeeping for one dataset, as accumulated over its chunks."""

    name: str
    has_info: bool
    n_chunks: int = 0
    counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    era: str | None = None
    is_mc: bool | None = None
    physics_group: str | None = None
    config_text: str | None = None
    config: dict = field(default_factory=dict)

    def regions(self) -> list[str]:
        return sorted(self.counts)

    def count(self, region: str, label: str) -> int:
        return int(self.counts.get(region, {}).get(label, 0))

    @property
    def families(self) -> frozenset[str]:
        return self.config.get("families", frozenset())

    @property
    def omitted(self) -> frozenset[str]:
        return self.config.get("omitted", frozenset())

    @property
    def syst_observables(self):
        return self.config.get("observables")

    @property
    def blinded(self) -> bool:
        """True for data of a blinded run: it withholds every signal region."""
        return bool(self.config.get("blind", False)) and self.is_mc is False

    def expected_regions(self) -> list[str]:
        """Regions the recorded configuration requires this dataset to have filled.

        Data of a blinded run must not have filled the signal regions; MC
        always requires every region.
        """
        return region_names(self.config.get("region", "both"), self.config.get("variants", ()),
                            blind=self.blinded)

    def expected(self, region: str) -> list[str]:
        """Labels the registry requires this dataset to have produced in *region*.

        Input availability is deliberately ignored here: an unavailable
        source that was not omitted must surface as missing.
        """
        return required_labels(self.era, self.families, region, is_mc=bool(self.is_mc),
                               omitted=self.omitted)


def _single_value(values, what, dataset):
    values = set(values or ())
    if len(values) != 1:
        raise SystematicCoverageError(
            f"Dataset '{dataset}' accumulated {len(values)} distinct values for {what}: "
            f"{sorted(map(str, values))}; expected exactly one."
        )
    return next(iter(values))


def dataset_coverage(name: str, payload: dict) -> DatasetCoverage:
    """Read the coverage records of one dataset payload."""
    if not all(key in payload for key in COVERAGE_KEYS):
        return DatasetCoverage(name=name, has_info=False)
    meta = payload["dataset_meta"]
    config_text = _single_value(meta.get("syst_config"), "systematic production config", name)
    return DatasetCoverage(
        name=name,
        has_info=True,
        n_chunks=int(payload["n_chunks"]),
        counts={r: dict(labels) for r, labels in payload["syst_coverage"].items()},
        era=_single_value(meta.get("era"), "era", name),
        is_mc=_single_value(meta.get("datatype"), "datatype", name) == "mc",
        physics_group=_single_value(meta.get("physics_group"), "physics_group", name),
        config_text=config_text,
        config=parse_production_config(config_text),
    )


def check_dataset_coverage(cov: DatasetCoverage, families: Iterable[str] | None = None) -> None:
    """Fail when a dataset did not produce exactly what the registry expects.

    - every region it filled must have Nominal in every chunk,
    - every label the registry expects there (given the dataset's era, its
      recorded families and omissions) must be complete in every chunk,
      including labels with a count of zero,
    - every label it produced must be one the registry expects.
    *families*, when given, must match the recorded production configuration.
    """
    if not cov.has_info:
        return
    if cov.n_chunks < 1:
        raise SystematicCoverageError(f"Dataset '{cov.name}' recorded no processed chunks.")
    if families is not None:
        requested = normalize_families(families)
        if requested != cov.families:
            raise SystematicCoverageError(
                f"Dataset '{cov.name}' was produced with systematic families "
                f"{sorted(cov.families)} but the export asks for {sorted(requested)}."
            )
    required_regions = cov.expected_regions()
    missing_regions = sorted(set(required_regions) - set(cov.counts))
    if missing_regions:
        raise SystematicCoverageError(
            f"Dataset '{cov.name}' was configured for regions '{cov.config.get('region')}' but "
            f"never filled {missing_regions}."
        )
    extra_regions = sorted(set(cov.counts) - set(required_regions))
    if cov.blinded and any(is_signal_region(r) for r in extra_regions):
        raise SystematicCoverageError(
            f"Dataset '{cov.name}' is data of a blinded run but filled signal regions "
            f"{[r for r in extra_regions if is_signal_region(r)]}."
        )
    if extra_regions:
        raise SystematicCoverageError(
            f"Dataset '{cov.name}' filled regions {extra_regions} that its configuration "
            "does not include."
        )
    for region in cov.regions():
        nominal = cov.count(region, NOMINAL)
        if nominal != cov.n_chunks:
            raise SystematicCoverageError(
                f"Dataset '{cov.name}', region '{region}': Nominal filled in {nominal} of "
                f"{cov.n_chunks} chunks."
            )
        expected = set(cov.expected(region))
        produced = {label for label in cov.counts[region] if label != NOMINAL}
        unexpected = sorted(produced - expected)
        if unexpected:
            raise SystematicCoverageError(
                f"Dataset '{cov.name}', region '{region}': labels {unexpected} were produced but "
                "the systematic registry does not expect them there."
            )
        for label in sorted(expected):
            count = cov.count(region, label)
            if count != cov.n_chunks:
                raise SystematicCoverageError(
                    f"Dataset '{cov.name}', region '{region}': systematic '{label}' completed in "
                    f"{count} of {cov.n_chunks} chunks."
                )


def coverage_status(cov: DatasetCoverage, region: str, label: str,
                    families: Iterable[str] | None = None) -> str:
    """Status of (region, label) for one dataset.  See module docstring."""
    if region not in cov.counts:
        return STATUS_ABSENT
    if label == NOMINAL:
        return STATUS_EVALUATED
    count = cov.count(region, label)
    if count > 0 and count == cov.n_chunks:
        return STATUS_EVALUATED
    src = source_of_label(label, families if families is not None else cov.families)
    if src is None:
        return STATUS_UNAVAILABLE
    if not src.applies(region=region, is_mc=bool(cov.is_mc)):
        return STATUS_UNAFFECTED
    if src.name in cov.omitted or src.family not in cov.families:
        return STATUS_OMITTED
    return STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# Histogram surgery helpers
# ---------------------------------------------------------------------------

def _axis_index(h: Hist, name: str) -> int:
    return list(h.axes.name).index(name)


def _ensure_syst_label(h: Hist, process: str, region: str, label: str) -> None:
    """Add *label* to the growth ``syst`` axis without changing any content."""
    if label in h.axes["syst"]:
        return
    kwargs = {"process": process, "region": region, "syst": label}
    for ax in h.axes:
        if ax.name not in kwargs:
            kwargs[ax.name] = np.array([np.nan])
    h.fill(weight=np.array([0.0]), **kwargs)


def _slice_index(h: Hist, region: str, label: str):
    idx = [slice(None)] * h.ndim
    idx[_axis_index(h, "region")] = h.axes["region"].index(region)
    idx[_axis_index(h, "syst")] = h.axes["syst"].index(label)
    return tuple(idx)


def copy_nominal_into(h: Hist, region: str, label: str, *, scale: float = 1.0) -> None:
    """Set ``h[region, label] = h[region, Nominal] * scale`` (variance scales by scale**2)."""
    if region not in h.axes["region"] or NOMINAL not in h.axes["syst"]:
        return
    for process in h.axes["process"]:
        _ensure_syst_label(h, process, region, label)
    view = h.view(flow=True)
    src = view[_slice_index(h, region, NOMINAL)]
    dst = view[_slice_index(h, region, label)]
    dst["value"] = src["value"] * scale
    dst["variance"] = src["variance"] * scale * scale


# ---------------------------------------------------------------------------
# Summing
# ---------------------------------------------------------------------------

def _hist_items(payload: dict):
    return {k: v for k, v in payload.items() if isinstance(v, Hist)}


def sum_hists(my_hists):
    """Blind sum of every histogram across datasets (no coverage logic)."""
    if not my_hists:
        raise ValueError("No histogram data provided.")

    original_histograms = list(my_hists.values())[0]
    sum_histograms = {
        key: Hist(*original_histograms[key].axes,
                  storage=original_histograms[key].storage_type())
        for key in original_histograms
        if isinstance(original_histograms[key], Hist)
    }

    for dataset_info in my_hists.values():
        for key, value in dataset_info.items():
            if isinstance(value, Hist):
                hist_name = key
                hist_data = value
                if hist_name in sum_histograms:
                    sum_histograms[hist_name] += hist_data
                else:
                    sum_histograms[hist_name] = hist_data.copy()

    return sum_histograms


@dataclass
class MergeReport:
    """What the coverage-aware merge decided, for the manifest."""

    coverage_known: bool
    datasets: Dict[str, DatasetCoverage]
    # Production configuration shared by every dataset (tracked payloads).
    config: dict = field(default_factory=dict)
    # region -> sorted labels evaluated by at least one dataset (excl. Nominal)
    produced: Dict[str, list[str]] = field(default_factory=dict)
    # region -> label -> datasets whose nominal was substituted
    unaffected: Dict[str, Dict[str, list[str]]] = field(default_factory=dict)
    # region -> label -> datasets that did not request the source
    omitted: Dict[str, Dict[str, list[str]]] = field(default_factory=dict)
    # region -> labels materialised from nominal at export
    normalization: Dict[str, list[str]] = field(default_factory=dict)

    def pairs(self) -> set[tuple[str, str]]:
        """(region, label) pairs that carry real content and should be written."""
        out = set()
        for region, labels in self.produced.items():
            out.add((region, NOMINAL))
            out.update((region, label) for label in labels)
        for region, labels in self.normalization.items():
            out.update((region, label) for label in labels)
        return out


def sum_hists_with_coverage(my_hists, era: str, families: Iterable[str]):
    """Coverage-aware sum across datasets.

    Applies the status table from the module docstring before adding:
    unaffected contributions are replaced by their nominal, unavailable ones
    raise, omitted ones are skipped and recorded.  Returns
    ``(summed_hists, MergeReport)``.
    """
    if not my_hists:
        raise ValueError("No histogram data provided.")
    fams = normalize_families(families) if families is not None else None
    covs = {name: dataset_coverage(name, payload) for name, payload in my_hists.items()}
    for cov in covs.values():
        check_dataset_coverage(cov, fams)

    if not any(cov.has_info for cov in covs.values()):
        logger.warning("No systematic coverage records found; summing histograms blindly.")
        summed = sum_hists(my_hists)
        report = MergeReport(coverage_known=False, datasets=covs)
        for h in summed.values():
            for region in h.axes["region"]:
                labels = [s for s in h.axes["syst"] if s != NOMINAL]
                report.produced.setdefault(region, [])
                report.produced[region] = sorted(set(report.produced[region]) | set(labels))
        return summed, report

    missing = [name for name, cov in covs.items() if not cov.has_info]
    if missing:
        raise SystematicCoverageError(
            f"Datasets {missing} carry no systematic coverage records while others do; "
            "cannot merge them safely."
        )
    configs = {cov.config_text for cov in covs.values()}
    if len(configs) != 1:
        raise SystematicCoverageError(
            "Datasets were produced with different systematic configurations and cannot be "
            f"merged: {sorted(configs)}"
        )
    shared_config = next(iter(covs.values())).config

    # Labels evaluated by at least one dataset, per region.
    produced: Dict[str, set[str]] = {}
    for cov in covs.values():
        for region in cov.regions():
            produced.setdefault(region, set()).update(
                label for label in cov.counts[region] if label != NOMINAL
            )

    report = MergeReport(coverage_known=True, datasets=covs, config=shared_config,
                         produced={r: sorted(v) for r, v in produced.items()})

    # Decide substitutions before touching any histogram.
    substitutions: Dict[str, list[tuple[str, str]]] = {name: [] for name in covs}
    for region, labels in produced.items():
        for label in sorted(labels):
            for name, cov in covs.items():
                status = coverage_status(cov, region, label)
                if status in (STATUS_EVALUATED, STATUS_ABSENT):
                    continue
                if status == STATUS_UNAFFECTED:
                    substitutions[name].append((region, label))
                    report.unaffected.setdefault(region, {}).setdefault(label, []).append(name)
                elif status == STATUS_OMITTED:
                    report.omitted.setdefault(region, {}).setdefault(label, []).append(name)
                else:
                    raise SystematicCoverageError(
                        f"Dataset '{name}' must provide systematic '{label}' in region '{region}' "
                        f"but completed it in {cov.count(region, label)} of {cov.n_chunks} chunks "
                        "(inputs unavailable or evaluation incomplete)."
                    )

    summed: Dict[str, Hist] = {}
    for name, payload in my_hists.items():
        for key, h in _hist_items(payload).items():
            h = h.copy()
            for region, label in substitutions[name]:
                copy_nominal_into(h, region, label)
            if key in summed:
                summed[key] += h
            else:
                summed[key] = h
    return summed, report


def materialize_normalization(summed: Dict[str, Hist], report: MergeReport, era: str,
                              families: Iterable[str], syst_observables, *, is_mc: bool) -> None:
    """Write normalization variations (e.g. LumiUp/Down) as scaled nominals."""
    if not is_mc:
        return
    sources = producible(era, families, omitted=report.config.get("omitted", ()),
                         kinds=(NORMALIZATION,))
    if not sources:
        return
    for region in report.produced:
        labels = []
        for src in sources:
            if not src.applies(region=region, is_mc=True):
                continue
            factor = normalization_factor(src, era)
            for direction, sign in (("Up", 1.0), ("Down", -1.0)):
                label = src.label(direction)
                for key, h in summed.items():
                    if syst_observables is not None and key not in syst_observables:
                        continue
                    copy_nominal_into(h, region, label, scale=1.0 + sign * factor)
                labels.append(label)
        if labels:
            report.normalization[region] = labels


def split_hists_with_syst(summed_hists, *, sum_over_process=True, pairs=None):
    """Slice ``{hist_name: Hist}`` into ``{(region, syst, hist_name): Hist}``.

    *pairs* restricts the output to the given ``(region, syst)`` combinations;
    ``None`` writes every combination present on the axes (legacy behavior,
    which also emits empty templates for combinations never filled).
    """
    out = {}
    for hist_name, h in summed_hists.items():
        try:
            region_ax = h.axes["region"]
            syst_ax = h.axes["syst"]
        except KeyError as e:
            logger.error("Missing expected axis in histogram '%s': %s", hist_name, e)
            continue

        has_proc = any(ax.name == "process" for ax in h.axes)

        regions = [region_ax.value(i) for i in range(region_ax.size)]
        systs = [syst_ax.value(i) for i in range(syst_ax.size)]

        for reg in regions:
            for sy in systs:
                if pairs is not None and (reg, sy) not in pairs:
                    continue
                hh = h[{region_ax.name: reg, syst_ax.name: sy}]

                if sum_over_process and has_proc and any(ax.name == "process" for ax in hh.axes):
                    idxs_to_keep = [i for i, ax in enumerate(hh.axes) if ax.name != "process"]
                    if len(idxs_to_keep) < hh.ndim:
                        hh = hh.project(*idxs_to_keep)

                out[(reg, sy, hist_name)] = hh
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def blinding_record(report: MergeReport) -> dict:
    """The manifest's ``blinding`` entry: what this file's data withholds.

    ``applied`` is true only when every dataset of the file is data and the
    recorded run configuration was blinded; MC files, unblinded data, and
    legacy payloads without coverage records report ``applied: false``.
    """
    covs = list(report.datasets.values())
    is_data = bool(covs) and all(cov.has_info and cov.is_mc is False for cov in covs)
    if not (report.coverage_known and is_data and report.config.get("blind", False)):
        return {"applied": False, "signal_regions": [], "cutflow_truncated_before": {}}
    mode = report.config.get("region", "both")
    return {
        "applied": True,
        "signal_regions": sorted(blinded_region_names(mode, report.config.get("variants", ()))),
        # Only the cutflow kinds this run produced (all of them for --region both).
        "cutflow_truncated_before": {
            CUTFLOW_DIRS[kind]: dict(stops)
            for kind, stops in BLINDED_CUTFLOW_STEPS.items() if mode in (kind, "both")
        },
    }


def build_manifest(report: MergeReport, *, era: str, sample, families, syst_observables,
                   output_file: Path, omitted=()) -> dict:
    fams = sorted(normalize_families(families))
    omitted = normalize_omitted(omitted)
    sources = systematics_for(era, fams)
    regions = sorted(set(report.produced) | set(report.normalization))
    per_region = {}
    for region in regions:
        produced = list(report.produced.get(region, []))
        normalization = list(report.normalization.get(region, []))
        # Labels that physically cannot change this region.
        unaffected = sorted({
            label for src in sources for label in src.labels()
            if src.kind != NORMALIZATION and not src.applies(region=region, is_mc=True)
        } | set(report.unaffected.get(region, {})))
        # Labels deliberately left out although they would apply here.
        omitted_labels = sorted(
            label for src in sources for label in src.labels()
            if src.name in omitted and src.applies(region=region, is_mc=True)
        )
        per_region[region] = {
            "produced": produced,
            "normalization": normalization,
            "unaffected": unaffected,
            "omitted": omitted_labels,
            "unaffected_datasets": report.unaffected.get(region, {}),
            "omitted_datasets": report.omitted.get(region, {}),
        }
    return {
        "root_file": output_file.name,
        "era": era,
        "sample": sample,
        "requested_families": fams,
        "omitted_sources": sorted(omitted),
        "sources": [
            {"name": src.name, "family": src.family, "kind": src.kind,
             "labels": list(src.labels()), "note": src.note,
             "available": src.available(era),
             "omitted": src.name in omitted}
            for src in sources
        ],
        "syst_observables": describe_syst_observables(syst_observables),
        "coverage_known": report.coverage_known,
        "blinding": blinding_record(report),
        "datasets": {
            name: {
                "n_chunks": cov.n_chunks,
                "datatype": None if cov.is_mc is None else ("mc" if cov.is_mc else "data"),
                "physics_group": cov.physics_group,
                "regions": cov.regions(),
            }
            for name, cov in report.datasets.items()
        },
        "regions": per_region,
    }


def _families_from_args(args):
    value = getattr(args, "systs", None)
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else []


def _omitted_from_args(args):
    value = getattr(args, "omit_systs", None)
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else []


def _syst_observables_from_args(args):
    value = getattr(args, "syst_observables", None)
    if value is not None and not isinstance(value, (list, tuple, set, frozenset)):
        value = None
    return resolve_syst_observables(value, all_histogram_names())


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def save_histograms(histograms, args):
    run, year, era = get_era_details(args.era)
    sample = args.sample
    hnwr_mass = args.mass

    working_dir = Path("WR_Plotter")

    if getattr(args, 'dir', None):
        output_dir = working_dir / 'rootfiles' / run / year / era / args.dir
    else:
        output_dir = working_dir / 'rootfiles' / run / year / era

    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'name', None):
        filename_prefix = f"WRAnalyzer_{args.name}"
    else:
        filename_prefix = f"WRAnalyzer"

    if sample == "Signal":
        output_file = output_dir / f"{filename_prefix}_signal_{hnwr_mass}.root"
    else:
        output_file = output_dir / f"{filename_prefix}_{sample}.root"

    families = _families_from_args(args)
    omitted = _omitted_from_args(args)
    syst_observables = _syst_observables_from_args(args)

    summed_hist, report = sum_hists_with_coverage(histograms, era, families)
    if report.coverage_known:
        # The recorded production configuration is authoritative.
        families = sorted(report.config["families"])
        omitted = sorted(report.config["omitted"])
        syst_observables = report.config["observables"]
        group_is_mc = any(cov.is_mc for cov in report.datasets.values())
        materialize_normalization(summed_hist, report, era, families, syst_observables,
                                  is_mc=group_is_mc)
        pairs = report.pairs()
    else:
        pairs = None
    split_histograms_dict = split_hists_with_syst(summed_hist, sum_over_process=True, pairs=pairs)

    cutflow_resolved_summed = _sum_cutflow_hists(
        histograms,
        cutflow_keys=("cutflow", "__cutflow__"),
    )
    cutflow_boosted_summed = _sum_cutflow_hists(
        histograms,
        cutflow_keys=("cutflow_boosted", "__cutflow_boosted__"),
    )
    # MC only (data payloads carry none): event counts vs log10|w| per region,
    # outside the systematics bookkeeping.
    event_weights_summed = _sum_cutflow_hists(histograms, cutflow_keys=("event_weights",))

    with uproot.recreate(output_file) as root_file:
        for (region, syst, hist_name), hist_obj in split_histograms_dict.items():
            # Use the canonical histogram key for stable ROOT naming.
            # This intentionally decouples output naming from the internal axis name.
            var_stem = hist_name
            folder, hname = _folder_and_hist_names(region, syst, var_stem)
            root_file[f"/{folder}/{hname}"] = hist_obj

        _save_cutflows(root_file, cutflow_resolved_summed, f"/{CUTFLOW_DIRS['resolved']}")
        _save_cutflows(root_file, cutflow_boosted_summed, f"/{CUTFLOW_DIRS['boosted']}")
        _save_cutflows(root_file, event_weights_summed, f"/{EVENT_WEIGHTS_DIR}")
        _save_event_lists(root_file, histograms)

    manifest = build_manifest(
        report, era=era, sample=(f"signal_{hnwr_mass}" if sample == "Signal" else sample),
        families=families, syst_observables=syst_observables, output_file=output_file,
        omitted=omitted,
    )
    manifest_file = output_file.with_name(output_file.stem + "_systematics.json")
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    logger.info(f"Histograms saved to {output_file}.")
    logger.info(f"Systematics manifest saved to {manifest_file}.")
    return output_file


def save_histograms_by_group(histograms, args, sample_to_group):
    """Save one ROOT file per physics_group from a composite run.

    Signal groups use the format ``"Signal:<mass_point>"`` so each mass
    point gets its own ROOT file (e.g. ``WRAnalyzer_signal_WR2000_N1900.root``).
    """
    from types import SimpleNamespace

    grouped: Dict[str, dict] = {}
    for sample_key, data in histograms.items():
        group = sample_to_group.get(sample_key, "Unknown")
        grouped.setdefault(group, {})[sample_key] = data

    for group, group_hists in grouped.items():
        group_args = SimpleNamespace(**vars(args))
        if group.startswith("Signal:"):
            group_args.sample = "Signal"
            group_args.mass = group.split(":", 1)[1]
        else:
            group_args.sample = group
            group_args.mass = None
        save_histograms(group_hists, group_args)
