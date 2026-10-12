"""Theory-weight sums the ``Runs`` tree lacks, accumulated from the events.

The MC ``Runs`` tree of NanoAOD stores, per original file, ``genEventSumw`` and
the ratio-valued sums ``LHEScaleSumw`` / ``LHEPdfSumw`` / ``PSSumw`` =
``sum(genWeight * w_i) / genEventSumw`` over *all* generated events.  The
analyzer normalizes every theory variation with them (``wrcoffea/analyzer.py``,
``_theory_variations``), so a sample whose ``Runs`` tree lacks one cannot
produce that family.  UL NanoAODv9 has no ``PSSumw`` at all (the branch came
with later NanoAOD versions), which is why the UL skims of 2026-10 could not
give the ``ps`` family.

The skimmer reads every event of every input before the selection, so it can
accumulate the same sums at no cost: :class:`EventWeightSums` takes each
chunk's ``genWeight`` and per-event weights, and
:func:`add_derived_runs_columns` adds the missing ``Runs`` branches to the
input's table, one identical row per ``Runs`` entry (the ``genEventSumw``-
weighted collapse of :mod:`wrcoffea.skim.runs` then gives the right mean over
inputs).  From there on the collapse, the part sidecar, the upload metadata and
the fileset treat them like native branches.  A branch the ``Runs`` tree
already has is never recomputed, and a derived one is listed in the input's
sidecar record (``derived_runs_branches``).

The ratio uses the sum of ``genWeight`` over the events actually read as the
denominator.  For a fully read file that is ``genEventSumw``; for a partial
read it is still the unbiased mean over the events seen.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import awkward as ak
import numpy as np

# Runs branch -> the per-event weight branch it is the mean of
RATIO_FROM_EVENTS = {
    "PSSumw": "PSWeight",
    "LHEScaleSumw": "LHEScaleWeight",
    "LHEPdfSumw": "LHEPdfWeight",
}


def derivable(runs_branches: Iterable[str], event_branches: Iterable[str]) -> list[str]:
    """The ``Runs`` branches missing from *runs_branches* that the events can supply.

    Needs ``genWeight`` (MC) and the per-event weight branch; the order is that
    of :data:`RATIO_FROM_EVENTS`.
    """
    runs = set(runs_branches)
    events = set(event_branches)
    if "genWeight" not in events:
        return []
    return [name for name, ev in RATIO_FROM_EVENTS.items() if name not in runs and ev in events]


class EventWeightSums:
    """``sum(genWeight * w_i)`` and ``sum(genWeight)`` over the events added."""

    def __init__(self, names: Sequence[str]):
        unknown = [n for n in names if n not in RATIO_FROM_EVENTS]
        if unknown:
            raise ValueError(f"not derivable from the events: {unknown}")
        self.names = list(names)
        self.sumw = 0.0
        self.n_events = 0
        self._sums: dict[str, np.ndarray | None] = {n: None for n in self.names}

    @property
    def branches(self) -> list[str]:
        """The event branches to read."""
        return ["genWeight"] + [RATIO_FROM_EVENTS[n] for n in self.names]

    def add(self, arrays) -> None:
        """Accumulate one chunk (a dict or record array holding :attr:`branches`), all its events."""
        gw = np.asarray(ak.to_numpy(arrays["genWeight"]), dtype=np.float64)
        if len(gw) == 0:
            return
        self.sumw += float(gw.sum())
        self.n_events += len(gw)
        for name in self.names:
            ev = RATIO_FROM_EVENTS[name]
            w = arrays[ev]
            counts = np.asarray(ak.to_numpy(ak.num(w, axis=1)), dtype=np.int64)
            k = int(counts[0])
            if np.any(counts != k):
                raise ValueError(f"{ev} has {sorted(set(counts.tolist()))} entries per event within one input; "
                                 f"cannot derive {name}")
            prev = self._sums[name]
            if prev is None:
                self._sums[name] = prev = np.zeros(k, dtype=np.float64)
            elif len(prev) != k:
                raise ValueError(f"{ev} has {k} entries per event after {len(prev)} earlier in the same input; "
                                 f"cannot derive {name}")
            if k == 0:
                continue
            mat = np.asarray(ak.to_numpy(ak.flatten(w, axis=1)), dtype=np.float64).reshape(len(gw), k)
            prev += (mat * gw[:, None]).sum(axis=0)

    def ratios(self) -> dict[str, np.ndarray]:
        """``{Runs branch: sum(genWeight * w_i) / sum(genWeight)}``.

        Without events (or with a vanishing weight sum) every vector is empty,
        which the ``Runs`` collapse reads as "no such weights in this input".
        """
        out: dict[str, np.ndarray] = {}
        for name in self.names:
            s = self._sums[name]
            if s is None or self.sumw == 0.0:
                out[name] = np.zeros(0, dtype=np.float64)
            else:
                out[name] = s / self.sumw
        return out


def add_derived_runs_columns(runs_table: Mapping[str, ak.Array], sums: EventWeightSums, *,
                             dtype=np.float32) -> dict[str, ak.Array]:
    """*runs_table* plus the derived ratio branches, every row holding the input's ratios.

    Branches the table already has are left as they are.  The new columns are
    jagged (``var * float32``, like NanoAOD's), so the writer declares them as
    such and uproot adds their ``n<branch>`` counters itself.
    """
    out = dict(runs_table)
    if not out:
        return out
    n_rows = len(next(iter(out.values())))
    for name, ratio in sums.ratios().items():
        if name in out:
            continue
        flat = np.tile(np.asarray(ratio, dtype=dtype), n_rows)
        out[name] = ak.unflatten(flat, np.full(n_rows, len(ratio), dtype=np.int64))
    return out
