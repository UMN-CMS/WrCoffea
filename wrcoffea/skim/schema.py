"""Branch policy and output schema for the skims.

The policy (``branches.yaml``) says which NanoAOD collections and flat
branches are written.  :func:`build_schema` applies it to the branch names
and types found in sample input files of one dataset and returns an
:class:`OutputSchema`: the exact, ordered list of output branches with their
dtypes, grouped by collection.  Every part of a dataset is written with the
same schema, so parts can be merged or read together regardless of which
input files produced them; an input file that lacks a schema branch has it
filled with a default value (and the sidecar records that), and extra
branches in an input are dropped (also recorded).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_POLICY_PATH = Path(__file__).resolve().parent / "branches.yaml"


def load_policy(path: str | Path | None = None) -> dict:
    with open(path or _POLICY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def collection_of(branch: str) -> str | None:
    """NanoAOD collection a branch belongs to, or None for a flat branch.

    ``Jet_pt`` -> ``Jet``; ``nJet`` -> ``Jet``; ``HLT_Mu50`` -> ``HLT``;
    ``run`` -> None.
    """
    if "_" in branch:
        return branch.split("_", 1)[0]
    if branch.startswith("n") and len(branch) > 1 and branch[1].isupper():
        return branch[1:]
    return None


def is_counter(branch: str) -> bool:
    return "_" not in branch and branch.startswith("n") and len(branch) > 1 and branch[1].isupper()


def dtype_name(dtype) -> str:
    """Canonical, byte-order-free dtype name (``float32``, ``bool``, ...)."""
    return np.dtype(dtype).newbyteorder("=").name


@dataclass
class BranchSpec:
    name: str
    dtype: str          # canonical numpy dtype name
    jagged: bool        # True for per-object (var *) branches
    collection: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "dtype": self.dtype, "jagged": self.jagged, "collection": self.collection}

    @classmethod
    def from_dict(cls, d: dict) -> "BranchSpec":
        return cls(d["name"], d["dtype"], bool(d["jagged"]), d.get("collection"))


@dataclass
class OutputSchema:
    """Ordered output branches, their types, and the policy decisions behind them."""
    branches: list[BranchSpec]
    dropped: list[str] = field(default_factory=list)      # input branches the policy drops
    unknown_kept: list[str] = field(default_factory=list)  # kept only because unknown_policy == keep
    policy_version: int = 0
    synthetic: list[str] = field(default_factory=list)     # required triggers absent from every sampled file

    @property
    def names(self) -> list[str]:
        return [b.name for b in self.branches]

    def by_name(self) -> dict[str, BranchSpec]:
        return {b.name: b for b in self.branches}

    def collections(self) -> dict[str, list[BranchSpec]]:
        """Jagged branches grouped by collection, in schema order."""
        out: dict[str, list[BranchSpec]] = {}
        for b in self.branches:
            if b.jagged and b.collection is not None:
                out.setdefault(b.collection, []).append(b)
        return out

    def flat(self) -> list[BranchSpec]:
        return [b for b in self.branches if not b.jagged]

    def jagged_flat(self) -> list[BranchSpec]:
        """Jagged branches that are not collection fields (LHEScaleWeight, PSWeight, ...)."""
        return [b for b in self.branches if b.jagged and b.collection is None]

    def digest(self) -> str:
        """Stable hash of names and types; stored in sidecars and the plan."""
        payload = json.dumps([(b.name, b.dtype, b.jagged) for b in self.branches]).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {"policy_version": self.policy_version, "digest": self.digest(),
                "branches": [b.to_dict() for b in self.branches],
                "dropped": self.dropped, "unknown_kept": self.unknown_kept, "synthetic": self.synthetic}

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSchema":
        return cls([BranchSpec.from_dict(b) for b in d["branches"]], list(d.get("dropped", [])),
                   list(d.get("unknown_kept", [])), int(d.get("policy_version", 0)),
                   list(d.get("synthetic", [])))

    def uproot_types(self) -> dict:
        """``mktree`` branch types: records per collection, dtypes for the rest."""
        types: dict = {}
        for coll, specs in self.collections().items():
            inner = ", ".join(f"{s.name.split('_', 1)[1]}: {s.dtype}" for s in specs)
            types[coll] = f"var * {{{inner}}}"
        for s in self.jagged_flat():
            types[s.name] = f"var * {s.dtype}"
        for s in self.flat():
            types[s.name] = np.dtype(s.dtype)
        return types


def branch_types_of_tree(tree) -> dict[str, tuple[str, bool]]:
    """``{branch: (dtype name, jagged)}`` for every interpretable branch of an uproot TTree.

    Counter branches (``nJet``) are skipped: the writer regenerates them.
    Branches uproot cannot interpret as a flat or jagged numeric array
    (nested objects) are skipped too.
    """
    out: dict[str, tuple[str, bool]] = {}
    for name in tree.keys():
        if is_counter(name):
            continue
        interp = tree[name].interpretation
        content = getattr(interp, "content", None)
        if content is not None and hasattr(content, "to_dtype"):
            dt = content.to_dtype
            if dt.names is not None:
                continue
            out[name] = (dtype_name(dt), True)
        elif hasattr(interp, "to_dtype"):
            dt = interp.to_dtype
            if dt.names is not None:
                continue
            out[name] = (dtype_name(dt), False)
    return out


def _trigger_wanted(name: str, policy: dict) -> bool:
    trig = policy.get("triggers", {})
    if name in trig.get("required", []):
        return True
    return any(fnmatch.fnmatchcase(name, pat) for pat in trig.get("patterns", []))


def _decide(name: str, policy: dict) -> str:
    """'keep', 'drop' or 'unknown' for a non-trigger branch."""
    coll = collection_of(name)
    keep_c = set(policy.get("keep_collections", []))
    drop_c = set(policy.get("drop_collections", []))
    keep_f = set(policy.get("keep_flat", []))
    if name in keep_f:
        return "keep"
    if coll is None:
        # Flat branches without a collection prefix (SoftActivityJetHT,
        # genTtbarId, ...) match the lists by their full name.
        if name in drop_c:
            return "drop"
        if name in keep_c:
            return "keep"
        return "unknown"
    if coll in keep_c:
        return "keep"
    if coll in drop_c:
        return "drop"
    return "unknown"


def build_schema(sampled: Iterable[Mapping[str, tuple[str, bool]]], policy: dict | None = None,
                 *, is_data: bool | None = None) -> OutputSchema:
    """Build the output schema from the branch tables of one or more sample files.

    *sampled* yields ``{branch: (dtype, jagged)}`` tables (see
    :func:`branch_types_of_tree`).  The union of the sampled branches is
    taken, so a dataset whose files differ in their trigger menu gets every
    matching path; the required trigger list is always present as ``bool``.
    """
    policy = policy or load_policy()
    union: dict[str, tuple[str, bool]] = {}
    for table in sampled:
        for name, (dt, jag) in table.items():
            if name in union and union[name] != (dt, jag):
                prev = union[name]
                # Prefer the wider type so values from either file fit.
                if np.dtype(dt).itemsize > np.dtype(prev[0]).itemsize:
                    union[name] = (dt, jag)
            else:
                union.setdefault(name, (dt, jag))
    synthetic: list[str] = []
    for req in policy.get("triggers", {}).get("required", []):
        if req not in union:
            synthetic.append(req)
            union[req] = ("bool", False)

    kept: list[BranchSpec] = []
    dropped: list[str] = []
    unknown_kept: list[str] = []
    unknown_policy = policy.get("unknown_policy", "keep")
    for name in sorted(union):
        dt, jag = union[name]
        coll = collection_of(name)
        if coll == "HLT":
            if _trigger_wanted(name, policy):
                kept.append(BranchSpec(name, dt, jag, None))
            else:
                dropped.append(name)
            continue
        if coll in ("L1", "DST"):
            dropped.append(name)
            continue
        decision = _decide(name, policy)
        if decision == "unknown":
            if unknown_policy == "keep":
                unknown_kept.append(name)
                decision = "keep"
            else:
                decision = "drop"
        if decision == "keep":
            # Collection fields are jagged; a flat branch sharing a prefix
            # (e.g. "PV_npvs" is flat) is still grouped under its collection
            # name only when jagged.
            kept.append(BranchSpec(name, dt, jag, coll if (jag and coll is not None and "_" in name) else None))
        else:
            dropped.append(name)

    # Order: flat branches first (event ids, weights, flags, triggers), then
    # jagged non-collection branches, then collections in name order.
    def _key(b: BranchSpec):
        if not b.jagged:
            return (0, b.name)
        if b.collection is None:
            return (1, b.name)
        return (2, b.collection, b.name)

    kept.sort(key=_key)
    # The event identifiers always come first.
    ids = {"run": 0, "luminosityBlock": 1, "event": 2}
    kept.sort(key=lambda b: (ids.get(b.name, 3),) + _key(b))
    return OutputSchema(kept, dropped, unknown_kept, int(policy.get("version", 0)), synthetic)
