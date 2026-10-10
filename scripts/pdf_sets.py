#!/usr/bin/env python3
"""Which LHAPDF set each sample's LHEPdfWeight holds, per era.

The ``pdf`` systematic family needs the member layout of each sample's PDF set
(Hessian eigenvectors, alpha_s members).  NanoAOD names the set only in the
branch description of ``LHEPdfWeight`` ("... for LHA IDs 325300 - 325402"),
which the skims do not keep, so this script reads one ORIGINAL NanoAOD file
per sample:

- the ``files`` of the sample's entry in the era's dataset configs
  (``data/configs/<Run>/<Year>/<Era>/*.json``) for private samples (the 2024
  signal),
- otherwise a file DBS lists for the sample's ``das_name`` (the repo's DBS
  client, grid proxy as for the skims), opened through an XRootD redirector.
  Up to ``--tries`` files are tried before giving up on a sample.

Samples whose fileset metadata has an empty ``LHEPdfSumw`` (pure Pythia 8
dibosons) are recorded with ``lhaid: null`` without opening anything.

    .venv/bin/python scripts/pdf_sets.py --era RunIII2024Summer24

writes ``data/pdf_sets/<Run>/<Year>/<Era>.json`` (``config.yaml`` ``pdf_sets``
points the analyzer at it).  Every set found must be described in
``wrcoffea.pdf_weights.KNOWN_SETS``; the script says so otherwise.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

LHA_RE = re.compile(r"LHA IDs\s+(\d+)\s*-\s*(\d+)")


def era_dirs(era: str) -> tuple[str, str]:
    from wrcoffea.era_utils import get_era_details
    run, year, _ = get_era_details(era)
    return run, year


def mc_samples(fileset_dir: Path, run: str, year: str, era: str) -> dict[str, dict]:
    """{sample: metadata} of every MC and signal entry of the era's skimmed filesets."""
    out = {}
    pattern = str(fileset_dir / run / year / era / "skimmed" / f"{era}_*_fileset.json")
    for path in sorted(glob.glob(pattern)):
        if path.endswith("_data_fileset.json"):
            continue
        for key, entry in json.load(open(path)).items():
            md = entry.get("metadata") or {}
            if (md.get("datatype") or "").lower() != "mc":
                continue
            out.setdefault(md.get("sample") or key, md)
    return out


def private_files(config_dir: Path, run: str, year: str, era: str) -> dict[str, list[str]]:
    """{sample: files} of the config entries that list their own files (private samples).

    The unskimmed filesets are not used: the 2024 one still lists the 23BPix
    placeholder signal.
    """
    out = {}
    for path in sorted(glob.glob(str(config_dir / run / year / era / f"{era}_*.json"))):
        for key, entry in json.load(open(path)).items():
            if isinstance(entry, dict) and entry.get("files"):
                out[entry.get("dataset") or key] = list(entry["files"])
    return out


def das_files(das_name: str, n: int, das_command: str | None = None) -> list[str]:
    """Up to *n* valid LFNs of *das_name* (DBS REST by default, or a dasgoclient command)."""
    if das_command:
        cmd = shlex.split(das_command) + ["-query", f"file dataset={das_name}"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        lfns = [ln.strip() for ln in res.stdout.splitlines() if ln.strip().startswith("/store/")]
    else:
        from wrcoffea.skim.catalog import DBSClient
        lfns = [rec.lfn for rec in DBSClient().files(das_name)]
    if not lfns:
        raise RuntimeError(f"no valid file listed for {das_name}")
    return lfns[:n]


def read_lhaid(url: str) -> tuple[int, int, int]:
    """(first LHA ID, last LHA ID, entries per event) from LHEPdfWeight's description."""
    import uproot
    with uproot.open(url, timeout=120) as f:
        tree = f["Events"]
        if "LHEPdfWeight" not in tree:
            raise RuntimeError(f"{url} has no LHEPdfWeight")
        title = tree["LHEPdfWeight"].title
        m = LHA_RE.search(title)
        if not m:
            raise RuntimeError(f"cannot parse the LHA IDs from '{title}' ({url})")
        n = tree["nLHEPdfWeight"].array(entry_stop=1)[0] if "nLHEPdfWeight" in tree else -1
        return int(m.group(1)), int(m.group(2)), int(n)


def scan(args) -> int:
    from wrcoffea.pdf_weights import KNOWN_SETS

    run, year = era_dirs(args.era)
    fileset_dir = Path(args.fileset_dir)
    samples = mc_samples(fileset_dir, run, year, args.era)
    if not samples:
        print(f"no MC filesets found for {args.era} under {fileset_dir}", file=sys.stderr)
        return 1
    originals = private_files(Path(args.config_dir), run, year, args.era)

    def one(item):
        sample, md = item
        n_meta = len(md.get("LHEPdfSumw") or [])
        if n_meta == 0:
            return sample, {"lhaid": None, "n": 0, "source": "no LHEPdfSumw in the fileset metadata"}
        urls = originals.get(sample, [])[:args.tries]
        if not urls:
            das_name = md.get("das_name") or ""
            if not das_name.startswith("/"):
                raise RuntimeError(f"{sample}: private sample without files in its config entry")
            urls = [args.redirector + lfn for lfn in das_files(das_name, args.tries, args.das_command)]
        errors = []
        for url in urls * (args.tries if len(urls) == 1 else 1):
            try:
                first, last, n = read_lhaid(url)
                break
            except Exception as exc:  # timeouts at busy sites: try the next file
                errors.append(f"{url}: {exc}")
        else:
            raise RuntimeError("; ".join(errors)[-600:])
        if last - first + 1 != n_meta:
            raise RuntimeError(f"{sample}: LHA IDs {first}-{last} but LHEPdfSumw has {n_meta} entries")
        return sample, {"lhaid": first, "lhaid_last": last, "n": n_meta, "source": url}

    results, errors = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, item): item[0] for item in sorted(samples.items())}
        for fut, sample in futures.items():
            try:
                name, rec = fut.result()
                results[name] = rec
                print(f"{rec['lhaid']!s:>7}  n={rec['n']:<4} {name}", flush=True)
            except Exception as exc:  # report all failures, write nothing
                errors.append(f"{sample}: {exc}")
    if errors:
        print("\nFAILED:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 1

    sets = sorted({rec["lhaid"] for rec in results.values() if rec["lhaid"] is not None})
    unknown = [s for s in sets if s not in KNOWN_SETS]
    if unknown:
        print(f"\nLHA sets {unknown} are not in wrcoffea.pdf_weights.KNOWN_SETS: add their member "
              "layout (from the set's .info file) first.", file=sys.stderr)
    out = Path(args.out) if args.out else REPO / "data" / "pdf_sets" / run / year / f"{args.era}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "era": args.era,
        "written": dt.date.today().isoformat(),
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "note": "First LHA ID of each sample's LHEPdfWeight (read from an original NanoAOD file's "
                "branch description); null = no PDF weights. Written by scripts/pdf_sets.py.",
        "sets": {str(s): (KNOWN_SETS[s].name if s in KNOWN_SETS else "UNKNOWN") for s in sets},
        "samples": dict(sorted(results.items())),
    }
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\n{len(results)} samples, sets {sets} -> {out}")
    return 2 if unknown else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--era", required=True)
    ap.add_argument("--fileset-dir", default=str(REPO / "data" / "filesets"))
    ap.add_argument("--config-dir", default=str(REPO / "data" / "configs"))
    ap.add_argument("--das-command", default=None,
                    help="use this dasgoclient invocation instead of the DBS REST client")
    ap.add_argument("--tries", type=int, default=3, help="files (or attempts) per sample")
    ap.add_argument("--redirector", default="root://cmsxrootd.fnal.gov/")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", help="output JSON (default data/pdf_sets/<Run>/<Year>/<Era>.json)")
    return scan(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
