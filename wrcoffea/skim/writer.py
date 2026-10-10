"""Rolling writer for skim parts.

A :class:`PartWriter` receives the selected events of one input file after
another and writes them, in input order, into ``<dataset>_gNNNNpNN.root``
files of bounded size (group index, then part index within the group).  Every part has the same :class:`~wrcoffea.skim.schema.OutputSchema`
(branches an input lacks are filled with defaults and recorded), a collapsed
``Runs`` tree, for data a merged ``LuminosityBlocks`` tree, and a JSON sidecar
with full provenance: which inputs it holds, in which entry ranges, how many
events each input had and passed, the event-id digests, checksums and the
campaign settings.  Parts are written as ``.partial`` files and renamed only
when complete, so an interrupted job never leaves a finished-looking part.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from wrcoffea.skim.fetch import adler32_of
from wrcoffea.skim.runs import collapse_runs, collapse_scan_runs, concat_runs, is_scan_runs, merge_lumiblocks, pad_scan_runs
from wrcoffea.skim.schema import OutputSchema, BranchSpec

logger = logging.getLogger(__name__)

SIDECAR_VERSION = 1


def compression_from_spec(spec: str):
    """``'zstd:3'`` -> ``uproot.ZSTD(3)``; also zlib, lz4, lzma, none."""
    algo, _, level = spec.lower().partition(":")
    level = int(level) if level else {"zstd": 3, "zlib": 1, "lz4": 4, "lzma": 9}.get(algo, 1)
    table = {"zstd": uproot.ZSTD, "zlib": uproot.ZLIB, "lz4": uproot.LZ4, "lzma": uproot.LZMA}
    if algo in ("none", "off", "0"):
        return None
    if algo not in table:
        raise ValueError(f"unknown compression {spec!r}; use zstd|zlib|lz4|lzma[:level]")
    return table[algo](level)


def empty_jagged(n: int, dtype) -> ak.Array:
    """A jagged array of *n* empty lists with inner *dtype*."""
    return ak.unflatten(np.zeros(0, dtype=np.dtype(dtype)), np.zeros(n, dtype=np.int64))


def _zeros_like_jagged(template: ak.Array, dtype) -> ak.Array:
    counts = ak.to_numpy(ak.num(template, axis=1))
    return ak.unflatten(np.zeros(int(counts.sum()), dtype=np.dtype(dtype)), counts)


@dataclass
class InputRecord:
    """Provenance of one input file inside a part."""
    lfn: str
    dataset: str = ""
    size: int = 0
    adler32: str = ""
    events_dbs: int = 0
    n_in: int = 0
    n_pass: int = 0
    entry_start: int = 0
    entry_stop: int = 0
    ids_digest: str = ""
    recount_digest: str = ""
    filled_branches: list[str] = field(default_factory=list)
    extra_branches: list[str] = field(default_factory=list)
    gen_sumw: float | None = None
    gen_count: int | None = None
    n_lumis: int | None = None
    source: str = ""
    fetch_seconds: float = 0.0
    skim_seconds: float = 0.0
    # Scan samples: generated-event sums per GenModel_* flag, before the skim.
    scan_counts: dict | None = None
    scan_sumw: dict | None = None
    scan_sumw2: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PartRecord:
    path: str
    index: int
    n_events: int
    size: int
    adler32: str
    inputs: list[InputRecord]
    runs: dict
    n_lumis: int | None
    schema_digest: str
    sidecar: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["inputs"] = [i.to_dict() for i in self.inputs]
        return d


class PartWriter:
    """Write selected events into size-bounded parts with one shared schema."""

    def __init__(self, out_dir: str | Path, dataset_name: str, schema: OutputSchema, *,
                 is_data: bool, compression: str = "zstd:3", flush_events: int = 50_000,
                 part_max_bytes: float = 2.0e9, first_index: int = 1,
                 campaign: dict | None = None, tree_name: str = "Events",
                 name_fmt: str = "{base}_part{index:04d}.root", name_fields: dict | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.schema = schema
        self.is_data = is_data
        self.compression_spec = compression
        self.compression = compression_from_spec(compression)
        self.flush_events = int(flush_events)
        self.part_max_bytes = float(part_max_bytes)
        self.next_index = int(first_index)
        self.campaign = dict(campaign or {})
        self.tree_name = tree_name
        self.name_fmt = name_fmt
        self.name_fields = dict(name_fields or {})
        self._by_name = schema.by_name()
        self._collections = schema.collections()
        self._types = schema.uproot_types()

        self._file = None
        self._tree = None
        self._part_path: Path | None = None
        self._buffer: list[dict[str, ak.Array]] = []
        self._buffered = 0
        self._part_events = 0
        self._inputs: list[InputRecord] = []
        self._runs_tables: list[dict] = []
        self._lumi_tables: list[dict] = []
        self._current: InputRecord | None = None
        self.parts: list[PartRecord] = []

    # ------------------------------------------------------------------ parts
    def _open_part(self) -> None:
        name = self.name_fmt.format(base=self.dataset_name, index=self.next_index, **self.name_fields)
        self._part_path = self.out_dir / name
        tmp = self._part_path.with_suffix(".root.partial")
        if tmp.exists():
            tmp.unlink()
        self._file = uproot.recreate(str(tmp), compression=self.compression)
        self._tree = self._file.mktree(self.tree_name, self._types)
        self._part_events = 0
        self._inputs = []
        self._runs_tables = []
        self._lumi_tables = []
        logger.info("opened %s", name)

    @property
    def part_open(self) -> bool:
        return self._file is not None

    def _part_bytes(self) -> int:
        if self._part_path is None:
            return 0
        tmp = self._part_path.with_suffix(".root.partial")
        return tmp.stat().st_size if tmp.exists() else 0

    # ----------------------------------------------------------------- inputs
    def begin_input(self, record: InputRecord, available_branches: list[str] | None = None) -> None:
        """Start a new input file; opens a part if none is open."""
        if not self.part_open:
            self._open_part()
        self._current = record
        record.entry_start = self._part_events
        if available_branches is not None:
            avail = set(available_branches)
            record.filled_branches = [n for n in self.schema.names if n not in avail]
            record.extra_branches = sorted(
                n for n in avail
                if n not in self._by_name and not (n.startswith("n") and len(n) > 1 and n[1].isupper() and "_" not in n)
            )

    def add_chunk(self, arrays, n_events: int | None = None) -> None:
        """Append already-selected events (dict or awkward record array of input branches)."""
        if self._current is None:
            raise RuntimeError("add_chunk called before begin_input")
        complete = self._complete(arrays)
        n = n_events if n_events is not None else len(next(iter(complete.values()))) if complete else 0
        if n == 0:
            return
        self._buffer.append(complete)
        self._buffered += n
        self._part_events += n
        if self._buffered >= self.flush_events:
            self._flush()

    def end_input(self, runs_table: dict | None, lumi_table: dict | None) -> InputRecord:
        """Finish the current input: record its Runs / LuminosityBlocks rows and maybe roll the part."""
        rec = self._current
        if rec is None:
            raise RuntimeError("end_input without begin_input")
        rec.entry_stop = self._part_events
        rec.n_pass = rec.entry_stop - rec.entry_start
        if runs_table:
            self._runs_tables.append(runs_table)
            for name in ("genEventSumw", "genEventSumw_"):
                if name in runs_table:
                    rec.gen_sumw = float(ak.sum(runs_table[name]))
                    break
            for name in ("genEventCount", "genEventCount_"):
                if name in runs_table:
                    rec.gen_count = int(ak.sum(runs_table[name]))
                    break
        if lumi_table:
            self._lumi_tables.append(lumi_table)
            rec.n_lumis = len(lumi_table["run"]) if "run" in lumi_table else 0
        self._inputs.append(rec)
        self._current = None
        # Roll at input boundaries so an input's events never straddle parts.
        self._flush()
        if self._part_bytes() >= self.part_max_bytes:
            self.close_part()
        return rec

    # ---------------------------------------------------------------- buffers
    def _complete(self, arrays) -> dict[str, ak.Array]:
        """Return ``{branch: array}`` for every schema branch, filling and casting."""
        def get(name):
            if isinstance(arrays, ak.Array):
                return arrays[name] if name in arrays.fields else None
            return arrays.get(name)

        n = None
        for spec in self.schema.branches:
            a = get(spec.name)
            if a is not None:
                n = len(a)
                break
        if n is None:
            raise ValueError("chunk contains no schema branch")

        out: dict[str, ak.Array] = {}
        # Jagged collection fields: template counts from any present field.
        for coll, specs in self._collections.items():
            template = None
            for s in specs:
                a = get(s.name)
                if a is not None:
                    template = a
                    break
            for s in specs:
                a = get(s.name)
                if a is None:
                    a = _zeros_like_jagged(template, s.dtype) if template is not None else empty_jagged(n, s.dtype)
                out[s.name] = self._cast(a, s)
        for s in self.schema.jagged_flat():
            a = get(s.name)
            out[s.name] = self._cast(a if a is not None else empty_jagged(n, s.dtype), s)
        for s in self.schema.flat():
            a = get(s.name)
            if a is None:
                a = ak.Array(np.zeros(n, dtype=np.dtype(s.dtype)))
            out[s.name] = self._cast(a, s)
        return out

    @staticmethod
    def _cast(a: ak.Array, spec: BranchSpec) -> ak.Array:
        want = np.dtype(spec.dtype)
        if spec.jagged:
            have = ak.to_numpy(ak.flatten(a, axis=1)).dtype if len(a) else None
        else:
            have = ak.to_numpy(a).dtype
        if have is not None and have.newbyteorder("=") != want:
            a = ak.values_astype(a, want)
        return a

    def _flush(self) -> None:
        if not self._buffer or self._tree is None:
            return
        chunks = self._buffer
        self._buffer = []
        self._buffered = 0
        merged = {name: (chunks[0][name] if len(chunks) == 1 else ak.concatenate([c[name] for c in chunks]))
                  for name in self.schema.names}
        payload: dict = {}
        for coll, specs in self._collections.items():
            payload[coll] = ak.zip({s.name.split("_", 1)[1]: merged[s.name] for s in specs})
        for s in self.schema.jagged_flat():
            payload[s.name] = merged[s.name]
        for s in self.schema.flat():
            payload[s.name] = merged[s.name]
        self._tree.extend(payload)

    # ---------------------------------------------------------------- closing
    def _write_runs(self) -> dict:
        if not self._runs_tables:
            return {}
        if any(is_scan_runs(t) for t in self._runs_tables):
            # multi-model scan inputs: per-model branches that differ from file to file
            collapsed = collapse_scan_runs(concat_runs(pad_scan_runs(self._runs_tables)))
        else:
            collapsed = collapse_runs(concat_runs(self._runs_tables))
        types = {}
        for name, arr in collapsed.items():
            if arr.ndim > 1:
                inner = ak.to_numpy(ak.flatten(arr, axis=None)).dtype if len(arr) else np.float64
                types[name] = f"var * {np.dtype(inner).newbyteorder('=').name}"
            else:
                types[name] = np.dtype(ak.to_numpy(arr).dtype).newbyteorder("=")
        self._file.mktree("Runs", types)
        self._file["Runs"].extend(collapsed)
        summary = {}
        for name in ("genEventSumw", "genEventSumw_", "genEventSumw2", "genEventSumw2_", "genEventCount", "genEventCount_"):
            if name in collapsed:
                summary[name] = float(collapsed[name][0]) if "Sumw" in name else int(collapsed[name][0])
        for name in ("LHEScaleSumw", "LHEPdfSumw", "PSSumw", "LHEReweightingSumw"):
            if name in collapsed:
                summary[name] = [float(x) for x in ak.to_list(collapsed[name][0])]
        if "run" in collapsed:
            summary["runs"] = [int(x) for x in ak.to_list(collapsed["run"])]
        return summary

    def _write_lumis(self) -> int | None:
        if not self._lumi_tables:
            return None
        merged = merge_lumiblocks(self._lumi_tables)
        types = {name: np.dtype(ak.to_numpy(arr).dtype).newbyteorder("=") for name, arr in merged.items() if arr.ndim == 1}
        self._file.mktree("LuminosityBlocks", types)
        self._file["LuminosityBlocks"].extend({n: merged[n] for n in types})
        return int(len(merged["run"])) if "run" in merged else 0

    def close_part(self) -> PartRecord | None:
        """Finish the open part: flush, write Runs/LuminosityBlocks, rename, write the sidecar."""
        if not self.part_open:
            return None
        self._flush()
        runs_summary = self._write_runs()
        n_lumis = self._write_lumis()
        self._file.close()
        self._file = None
        self._tree = None
        tmp = self._part_path.with_suffix(".root.partial")
        os.replace(tmp, self._part_path)
        size = self._part_path.stat().st_size
        digest = adler32_of(str(self._part_path))
        sidecar = self._part_path.with_suffix(".json")
        rec = PartRecord(path=str(self._part_path), index=self.next_index, n_events=self._part_events,
                         size=size, adler32=digest, inputs=list(self._inputs), runs=runs_summary,
                         n_lumis=n_lumis, schema_digest=self.schema.digest(), sidecar=str(sidecar))
        part_doc = rec.to_dict()
        # Stored relative to the sidecar's own directory so the output tree can be moved or copied
        # (Condor scratch -> EOS -> laptop) as a whole; read_sidecar() makes them absolute again.
        part_doc["path"] = self._part_path.name
        part_doc["sidecar"] = sidecar.name
        payload = {
            "sidecar_version": SIDECAR_VERSION,
            "dataset": self.dataset_name,
            "is_data": self.is_data,
            "part": part_doc,
            "schema": {"digest": self.schema.digest(), "n_branches": len(self.schema.branches),
                       "policy_version": self.schema.policy_version},
            "compression": self.compression_spec,
            "flush_events": self.flush_events,
            "campaign": self.campaign,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
        }
        tmp_json = sidecar.with_suffix(".json.partial")
        tmp_json.write_text(json.dumps(payload, indent=1) + "\n")
        os.replace(tmp_json, sidecar)
        logger.info("closed %s: %d events, %.1f MB, %d inputs", self._part_path.name, rec.n_events,
                    size / 1e6, len(rec.inputs))
        self.parts.append(rec)
        self.next_index += 1
        self._part_path = None
        self._inputs = []
        self._runs_tables = []
        self._lumi_tables = []
        self._part_events = 0
        return rec

    def close(self) -> list[PartRecord]:
        if self._current is not None:
            raise RuntimeError("close() called with an input still open")
        self.close_part()
        return self.parts

    def abort(self) -> None:
        """Discard the open part (on failure) so no partial output survives."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._tree = None
        if self._part_path is not None:
            tmp = self._part_path.with_suffix(".root.partial")
            if tmp.exists():
                tmp.unlink()
        self._buffer = []
        self._buffered = 0
        self._current = None


def read_sidecar(path: str | Path) -> dict:
    """Load a part sidecar; relative part/sidecar names are resolved against the sidecar's directory."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    base = os.path.dirname(os.path.abspath(str(path)))
    part = doc.get("part") or {}
    for key in ("path", "sidecar"):
        v = part.get(key)
        if v and not os.path.isabs(v):
            part[key] = os.path.join(base, v)
    return doc
