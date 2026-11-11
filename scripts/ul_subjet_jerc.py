#!/usr/bin/env python3
"""AK4 PUPPI JEC and JER of the Run 2 UL eras as correctionlib files, for the AK8 soft-drop subjets.

The JME jet mass recommendation corrects both soft-drop subjets with the AK4 PUPPI JEC and JER.  The UL
correctionlib files hold AK4 CHS and AK8 PUPPI only.  The AK4 PUPPI corrections of the same tags are text
files in the JME databases (github.com/cms-jet/JECDatabase and JRDatabase).  This script reads them at fixed
revisions and writes ``data/jsonpog/JME/RunII/<era>/subjet_jerc.json.gz`` with the names of the official
files (``<tag>_<level>_AK4PFPuppi``):

- MC tag: L1FastJet, L2Relative, L3Absolute, L2L3Residual, every JES uncertainty source (``Total`` among
  them), the pT resolution, the JER scale factor and its uncertainty;
- DATA tag: the same levels; L1FastJet and L2L3Residual take the run number, as in the official files,
  with the run ranges of the era's official AK4 CHS file.

    .venv/bin/python scripts/ul_subjet_jerc.py --all            # write the four files
    .venv/bin/python scripts/ul_subjet_jerc.py --all --check    # closure: convert the AK4 CHS text files the
                                                                # same way and compare with the official files

``config.yaml`` (``jerc_jsons: <era>: JERC_SUBJET``) points the analyzer at the files.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# Revisions of the JME text-file databases the payloads are built from.
JEC_DB = ("cms-jet/JECDatabase", "e825b8378a63df0cc4fb4c65dfe6088cbbb54a09")
JR_DB = ("cms-jet/JRDatabase", "083bf1262e243b611749558ef36b5cbbdfdd517a")

# era -> JEC tag, data-taking periods of the tag (in run order), JER tag.  The tags are those of the era's
# official files (config.yaml jerc_jsons).
ERAS = {
    "RunIISummer20UL16APV": ("Summer19UL16APV_V7", ("RunBCD", "RunEF"), "Summer20UL16APV_JRV5"),
    "RunIISummer20UL16": ("Summer19UL16_V7", ("RunFGH",), "Summer20UL16_JRV5"),
    "RunIISummer20UL17": ("Summer19UL17_V5", ("RunB", "RunC", "RunD", "RunE", "RunF"), "Summer19UL17_JRV4"),
    "RunIISummer20UL18": ("Summer19UL18_V5", ("RunA", "RunB", "RunC", "RunD"), "Summer19UL18_JRV3"),
}
YEAR = {"RunIISummer20UL16APV": "2016", "RunIISummer20UL16": "2016", "RunIISummer20UL17": "2017",
        "RunIISummer20UL18": "2018"}

JEC_LEVELS = ("L1FastJet", "L2Relative", "L3Absolute", "L2L3Residual")
RUN_DEPENDENT = ("L1FastJet", "L2L3Residual")   # data levels with a run binning in the official files
SUBJET_ALGO = "AK4PFPuppi"
REFERENCE_ALGO = "AK4PFchs"

# Linear interpolation between two pT points of an uncertainty table, as in the official files.
INTERPOLATION = {"nodetype": "formula", "expression": "[0]*max(min(x,[3]),[2])+[1]", "parser": "TFormula",
                 "variables": ["JetPt"]}
DESCRIPTIONS = {
    "JetPt": "pT of the jet before specific correction (for JER: after all corrections applied)",
    "JetA": "area of the jet",
    "JetEta": "pseudorapidity of the jet",
    "Rho": "energy density rho (as measure of PU)",
    "run": "run number",
}


# ---------------------------------------------------------------------------------------------------------
# Text files
# ---------------------------------------------------------------------------------------------------------
class TextFiles:
    """Files of the two databases at their fixed revisions, with git symbolic links followed."""

    def __init__(self, cache: Path | None):
        self.cache = cache

    def get(self, db: tuple[str, str], directory: str, name: str) -> str:
        repo, rev = db
        path = f"textFiles/{directory}/{name}"
        for _ in range(6):
            text = self._fetch(repo, rev, path)
            target = text.strip()
            if "{" in text or "\n" in target or not target.endswith(".txt"):
                return text
            # A git symbolic link: its content is the path of the file it points to.
            parts = (path.rsplit("/", 1)[0] + "/" + target).split("/")
            resolved: list[str] = []
            for part in parts:
                if part == "..":
                    resolved.pop()
                elif part != ".":
                    resolved.append(part)
            path = "/".join(resolved)
        raise RuntimeError(f"{repo}: too many symbolic links from {directory}/{name}")

    def _fetch(self, repo: str, rev: str, path: str) -> str:
        local = self.cache / repo.replace("/", "_") / rev[:12] / path if self.cache else None
        if local is not None and local.exists():
            return local.read_text()
        url = f"https://raw.githubusercontent.com/{repo}/{rev}/{path}"
        with urllib.request.urlopen(url, timeout=120) as response:
            text = response.read().decode()
        if local is not None:
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(text)
        return text


@dataclass
class Table:
    """One table of a JME text file: its header and its records."""

    section: str | None
    bin_vars: list[str]
    par_vars: list[str]
    formula: str
    level: str
    records: list[tuple[list[tuple[float, float]], list[float]]]   # (bin ranges, parameters)


def _single(value: str) -> float:
    """The number as CMSSW holds it (single precision), written back at its shortest form."""
    return float(str(np.float32(value)))


def parse_tables(text: str) -> list[Table]:
    tables: list[Table] = []
    section = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if line.startswith("{"):
            tok = line.strip("{}").split()
            nbin = int(tok[0])
            bin_vars = tok[1:1 + nbin]
            npar = int(tok[1 + nbin])
            par_vars = tok[2 + nbin:2 + nbin + npar]
            rest = tok[2 + nbin + npar:]
            if len(rest) >= 2 and rest[-2] == "Correction":
                formula, level = "".join(rest[:-2]), rest[-1]
            elif rest and rest[-1] == "Resolution":
                formula, level = "".join(rest[:-1]), "Resolution"
            else:
                raise ValueError(f"Header not understood: {line}")
            tables.append(Table(section, bin_vars, par_vars, formula, level, []))
            continue
        table = tables[-1]
        tok = line.split()
        nbin = len(table.bin_vars)
        ranges = [(_single(tok[2 * i]), _single(tok[2 * i + 1])) for i in range(nbin)]
        n = int(tok[2 * nbin])
        params = [_single(t) for t in tok[2 * nbin + 1:]]
        if len(params) != n:
            raise ValueError(f"Record announces {n} parameters and holds {len(params)}: {line}")
        table.records.append((ranges, params))
    return tables


# ---------------------------------------------------------------------------------------------------------
# correctionlib nodes
# ---------------------------------------------------------------------------------------------------------
_TMATH = (("TMath::Log10", "log10"), ("TMath::Log", "log"), ("TMath::Max", "max"), ("TMath::Min", "min"),
          ("TMath::Power", "pow"), ("TMath::Sqrt", "sqrt"), ("TMath::Abs", "abs"), ("TMath::Exp", "exp"))


def _is_table_of_points(table: Table) -> bool:
    return table.formula in ('""', "None")


def _formula(table: Table, params: list[float]) -> dict:
    """The record's formula with each variable clamped to its validity range.  The parameters are the record's:
    the ranges of the variables first, then the formula's own, whose indices move up accordingly."""
    npar = len(table.par_vars)
    expr = table.formula
    for old, new in _TMATH:
        expr = expr.replace(old, new)
    if "TMath" in expr:
        raise ValueError(f"Function not translated in: {expr}")
    expr = re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) + 2 * npar}]", expr)
    for i, var in enumerate("xyzt"[:npar]):
        expr = re.sub(rf"(?<![A-Za-z_]){var}(?![A-Za-z_])", f"max(min({var},[{2 * i + 1}]),[{2 * i}])", expr)
    return {"nodetype": "formula", "expression": expr, "parser": "TFormula", "parameters": params,
            "variables": list(table.par_vars)}


def _points(table: Table, params: list[float]) -> dict:
    """An uncertainty given at pT points (pT, up, down): linear in pT between them, constant outside.  A
    single point (the JER scale-factor uncertainty) is a constant: its down value, as in the official files
    (up and down differ by 1e-4 in a few bins)."""
    (var,) = table.par_vars
    edges, up, down = params[0::3], params[1::3], params[2::3]
    if len(edges) == 1:
        return {"nodetype": "binning", "input": var, "edges": [0.0, edges[0]], "content": [down[0]], "flow": "clamp"}
    content = []
    for i in range(len(edges) - 1):
        width = edges[i + 1] - edges[i]
        content.append({"nodetype": "formularef", "index": 0,
                        "parameters": [(up[i + 1] - up[i]) / width,
                                       (up[i] * edges[i + 1] - up[i + 1] * edges[i]) / width,
                                       edges[0], edges[-1]]})
    return {"nodetype": "binning", "input": var, "edges": edges, "content": content, "flow": "clamp"}


def _binned(table: Table, records: list, depth: int) -> dict:
    """Nested binning over the table's bin variables, down to one formula (or point table) per record."""
    if depth == len(table.bin_vars):
        (record,) = records
        return _points(table, record[1]) if _is_table_of_points(table) else _formula(table, record[1])
    groups: list[tuple[tuple[float, float], list]] = []
    for record in records:
        rng = record[0][depth]
        if groups and groups[-1][0] == rng:
            groups[-1][1].append(record)
        else:
            groups.append((rng, [record]))
    edges = [groups[0][0][0]]
    for (low, high), _ in groups:
        if low != edges[-1]:
            raise ValueError(f"{table.bin_vars[depth]} bins are not contiguous at {low}")
        edges.append(high)
    return {"nodetype": "binning", "input": table.bin_vars[depth], "edges": edges,
            "content": [_binned(table, group, depth + 1) for _, group in groups],
            # Outside the table, as in the official files: 1, and -999 for a JES uncertainty source.
            "flow": -999.0 if table.level == "JECSource" else 1.0}


def _inputs(*names: str) -> list[dict]:
    ordered = [n for n in ("run",) if n in names] + sorted(set(names) - {"run"})
    return [{"name": n, "type": "real", "description": DESCRIPTIONS[n]} for n in ordered]


def correction(name: str, description: str, table: Table, run_edges: list[float] | None = None,
               period_tables: list[Table] | None = None) -> dict:
    """One correction from one table, or from one table per data-taking period under a run binning."""
    names = list(table.bin_vars) + list(table.par_vars)
    if period_tables is None:
        data = _binned(table, table.records, 0)
    else:
        names.append("run")
        data = {"nodetype": "binning", "input": "run", "edges": run_edges,
                "content": [_binned(t, t.records, 0) for t in period_tables], "flow": "error"}
    out = {"name": name, "description": description, "version": 1, "inputs": _inputs(*names),
           "output": {"name": "correction", "type": "real"}, "data": data}
    if _is_table_of_points(table):
        out["generic_formulas"] = [INTERPOLATION]
    return out


# ---------------------------------------------------------------------------------------------------------
# One era
# ---------------------------------------------------------------------------------------------------------
def official_file(era: str) -> Path:
    return REPO / "data" / "jsonpog" / "JME" / "RunII" / era / "jet_jerc.json.gz"


def official_run_edges(era: str, jec_tag: str) -> list[float]:
    """Run ranges of the data-taking periods: the run binning of the official AK4 CHS residual."""
    with gzip.open(official_file(era)) as f:
        corrections = json.load(f)["corrections"]
    name = f"{jec_tag}_DATA_L2L3Residual_{REFERENCE_ALGO}"
    (residual,) = [c for c in corrections if c["name"] == name]
    if residual["data"].get("input") != "run":
        raise ValueError(f"{name} has no run binning")
    return residual["data"]["edges"]


def build(era: str, algo: str, files: TextFiles) -> dict:
    jec_tag, periods, jer_tag = ERAS[era]
    base, version = jec_tag.rsplit("_", 1)
    source = f"{JEC_DB[0]} at {JEC_DB[1][:12]}"
    corrections = []

    def one_table(db, directory, level):
        (table,) = parse_tables(files.get(db, directory, f"{directory}_{level}_{algo}.txt"))
        return table

    mc_dir = f"{jec_tag}_MC"
    for level in JEC_LEVELS:
        corrections.append(correction(f"{mc_dir}_{level}_{algo}", f"{level} for {algo} jets, from {mc_dir} ({source})",
                                      one_table(JEC_DB, mc_dir, level)))
    for table in parse_tables(files.get(JEC_DB, mc_dir, f"{mc_dir}_UncertaintySources_{algo}.txt")):
        corrections.append(correction(f"{mc_dir}_{table.section}_{algo}",
                                      f"{table.section} uncertainty for {algo} jets, from {mc_dir} ({source})", table))

    run_edges = official_run_edges(era, jec_tag)
    if len(run_edges) != len(periods) + 1:
        raise ValueError(f"{era}: {len(periods)} periods but {len(run_edges) - 1} run ranges in the official file")
    data_dirs = [f"{base}_{period}_{version}_DATA" for period in periods]
    for level in JEC_LEVELS:
        texts = [files.get(JEC_DB, d, f"{d}_{level}_{algo}.txt") for d in data_dirs]
        tables = [parse_tables(t)[0] for t in texts]
        name = f"{jec_tag}_DATA_{level}_{algo}"
        if level in RUN_DEPENDENT:
            corrections.append(correction(name, f"{level} for {algo} jets, from {', '.join(data_dirs)} ({source})",
                                          tables[0], run_edges=run_edges, period_tables=tables))
        elif all(t.records == tables[0].records and t.formula == tables[0].formula for t in tables):
            corrections.append(correction(name, f"{level} for {algo} jets, from {data_dirs[0]} ({source}); "
                                          "the same in every data-taking period", tables[0]))
        else:
            raise ValueError(f"{name} differs between the data-taking periods and has no run binning")

    jr_dir = f"{jer_tag}_MC"
    jr_source = f"{JR_DB[0]} at {JR_DB[1][:12]}"
    for level, file_level in (("PtResolution", "PtResolution"), ("ScaleFactor", "SF"), ("SFUncertainty", "SFUncertainty")):
        corrections.append(correction(f"{jr_dir}_{level}_{algo}", f"{level} for {algo} jets, from {jr_dir} ({jr_source})",
                                      one_table(JR_DB, jr_dir, file_level)))
    return {
        "schema_version": 2,
        "description": f"{algo} JEC ({jec_tag}) and JER ({jer_tag}) of {era}, converted from the JME text files "
                       f"({source}; {jr_source}) by scripts/ul_subjet_jerc.py",
        "corrections": corrections,
    }


def validated(cset: dict) -> str:
    from correctionlib import schemav2
    return schemav2.CorrectionSet.model_validate(cset).model_dump_json(exclude_unset=True)


# ---------------------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------------------
def _random_inputs(corr, rng, n, run_edges):
    values = {
        "JetEta": rng.uniform(-5.3, 5.3, n),
        "JetPt": np.exp(rng.uniform(np.log(4.0), np.log(7000.0), n)),
        "Rho": rng.uniform(0.0, 80.0, n),
        "JetA": rng.uniform(0.1, 1.0, n),
        "run": np.floor(rng.uniform(run_edges[0], run_edges[-1], n)),
    }
    return [values[i.name] for i in corr.inputs]


def check_against_official(era: str, files: TextFiles, n: int = 200_000) -> bool:
    """Convert the AK4 CHS text files and compare every correction with the official file's."""
    import correctionlib
    mine = correctionlib.CorrectionSet.from_string(validated(build(era, REFERENCE_ALGO, files)))
    official = correctionlib.CorrectionSet.from_file(str(official_file(era)))
    run_edges = official_run_edges(era, ERAS[era][0])
    rng = np.random.default_rng(97)
    worst: dict[str, tuple[float, str]] = {}
    missing = []
    for name in mine:
        if name not in official:
            missing.append(name)
            continue
        a, b = mine[name], official[name]
        if sorted(i.name for i in a.inputs) != sorted(i.name for i in b.inputs):
            print(f"  {name}: inputs {[i.name for i in a.inputs]} against {[i.name for i in b.inputs]}")
            worst[name] = (np.inf, name)
            continue
        args = _random_inputs(a, rng, n, run_edges)
        by_name = dict(zip([i.name for i in a.inputs], args))
        if "ScaleFactor" in name or "SFUncertainty" in name:
            by_name["JetPt"] = np.clip(by_name["JetPt"], 10.0, 2000.0)   # the range both versions share
        va = a.evaluate(*[by_name[i.name] for i in a.inputs])
        vb = b.evaluate(*[by_name[i.name] for i in b.inputs])
        diff = float(np.max(np.abs(va - vb)))
        kind = ("uncertainty source" if not re.search(r"_(L\d\w+|PtResolution|ScaleFactor|SFUncertainty)_", name)
                else re.search(r"_(L\d\w+|PtResolution|ScaleFactor|SFUncertainty)_", name).group(1)
                + (" (DATA)" if "_DATA_" in name else ""))
        if diff > worst.get(kind, (-1.0, ""))[0]:
            worst[kind] = (diff, name)
    ok = True
    for kind, (diff, name) in sorted(worst.items()):
        good = diff <= 1e-6
        ok &= good
        print(f"  {kind:28s} max |converted - official| = {diff:.2e}  {'ok' if good else 'DIFFERENT'}  ({name})")
    official_names = [n for n in official if n.endswith(REFERENCE_ALGO)]
    extra = sorted(set(official_names) - set(mine))
    print(f"  {len(list(mine)) - len(missing)} corrections compared; not in the official file: {missing or 'none'}; "
          f"official ones not converted: {len(extra)} "
          f"({sorted({re.sub(r'.*_(MC|DATA)_', '', e).split('_AK4')[0].split('_')[0] for e in extra})})")
    return ok and not missing


def check_subjet_payload(era: str, text: str) -> None:
    """What the analyzer relies on: the PUPPI L1FastJet and L3Absolute are 1."""
    import correctionlib
    cset = correctionlib.CorrectionSet.from_string(text)
    jec_tag = ERAS[era][0]
    run_edges = official_run_edges(era, jec_tag)
    rng = np.random.default_rng(1)
    for kind in ("MC", "DATA"):
        for level in ("L1FastJet", "L3Absolute"):
            corr = cset[f"{jec_tag}_{kind}_{level}_{SUBJET_ALGO}"]
            values = corr.evaluate(*_random_inputs(corr, rng, 20_000, run_edges))
            if not np.allclose(values, 1.0):
                raise RuntimeError(f"{corr.name} is not 1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--era", choices=sorted(ERAS))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Convert the AK4 CHS text files and compare with the official files; write nothing.")
    parser.add_argument("--cache", type=Path, default=None, help="Directory to keep the downloaded text files in.")
    args = parser.parse_args()

    files = TextFiles(args.cache)
    status = 0
    for era in sorted(ERAS) if args.all else [args.era]:
        if args.check:
            print(f"{era}: AK4 CHS text files against {official_file(era).relative_to(REPO)}")
            if not check_against_official(era, files):
                status = 1
            continue
        text = validated(build(era, SUBJET_ALGO, files))
        check_subjet_payload(era, text)
        out = REPO / "data" / "jsonpog" / "JME" / "RunII" / era / "subjet_jerc.json.gz"
        with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            f.write(text.encode())
        n = len(json.loads(text)["corrections"])
        print(f"{era}: {n} corrections -> {out.relative_to(REPO)} ({out.stat().st_size / 1e3:.0f} kB)")
    return status


if __name__ == "__main__":
    sys.exit(main())
