#!/usr/bin/env python3
"""Convert the Run 2 HNWR signal NLO/LO k-factor inputs to JSON.

The Run 2 WR analysis (SKFlatAnalyzer ``HNWR/Run2Legacy_v4``, AN-19-083
Appendix A) reweights every signal event by ``k(m_lN) / <k>``:

* ``HNWRGeneralData/kfactor.root`` (``kfactor_all``): the NLO/LO ratio as a
  function of the generator-level mass of the primary lepton plus N system
  (200 GeV bins, 0-12000 GeV); ``m_lN < 200`` is evaluated at 201.
* ``<year>/HNWRKFactor/AveragedKFactor{,_WROfficial}.txt``: per mass point
  the average of ``k(m_lN)`` over the generated sample, for the ee and mumu
  channels (``WR1000_N100  <k_ee>  <k_mm>``).  The sample cross sections
  quoted in the AN (and copied into our signal configs) are LO x <k>, so the
  event weight must divide the average out to stay a pure shape correction.

The official-sample table is preferred; mass points only present in the
private-sample table are taken from there and marked as such.

Example::

    python scripts/convert_run2_signal_kfactor.py \
        --skflat-data /path/to/SKFlatAnalyzer/data/Run2Legacy_v4 --year 2018 \
        --out data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_run2legacy_v4_2018.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import uproot

SOURCE_REPO = "https://github.com/jedori0228/SKFlatAnalyzer"
SOURCE_BRANCH = "HNWR/Run2Legacy_v4"
MLN_MIN = 201.0   # HNWRAnalyzer::GetHNWRKFactor: if(mlN<200.) mlN = 200.+1.


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_table(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        out[parts[0]] = (float(parts[1]), float(parts[2]))
    return out


def convert(skflat_data: Path, year: int) -> dict:
    hist_file = skflat_data / "HNWRGeneralData" / "kfactor.root"
    with uproot.open(hist_file) as f:
        h = f["kfactor_all"]
        values, edges = h.to_numpy()
        errors = h.errors()

    official_file = skflat_data / str(year) / "HNWRKFactor" / "AveragedKFactor_WROfficial.txt"
    private_file = skflat_data / str(year) / "HNWRKFactor" / "AveragedKFactor.txt"
    official = _read_table(official_file)
    private = _read_table(private_file)

    averages = {}
    for key, (k_ee, k_mm) in private.items():
        averages[key] = {"ee": k_ee, "mm": k_mm, "source": "private"}
    for key, (k_ee, k_mm) in official.items():
        averages[key] = {"ee": k_ee, "mm": k_mm, "source": "official"}

    return {
        "provenance": {
            "source_repo": SOURCE_REPO,
            "source_branch": SOURCE_BRANCH,
            "source_commit": _git_commit(skflat_data),
            "source_files": [str(p.relative_to(skflat_data.parent)) for p in (hist_file, official_file, private_file)],
            "year": year,
            "converted_on": date.today().isoformat(),
            "analyzer_reference": "HNWRAnalyzer::GetHNWRKFactor, weight = k(m_lN)/avg (AN-19-083 App. A)",
            "note": (
                "NLO/LO k-factor of WR production vs the generator-level m(lN); the per-mass-point "
                "average divides out because the signal cross sections used for normalization are "
                "already LO x <k>.  'source' says which Run 2 sample set the average was computed on."
            ),
        },
        "variable": "gen_mln",
        "kfactor": {
            "edges": edges.tolist(),
            "values": values.tolist(),
            "errors": errors.tolist(),
            "mln_min": MLN_MIN,
        },
        "averages": dict(sorted(averages.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skflat-data", type=Path, required=True, help="SKFlatAnalyzer data/Run2Legacy_v4 directory")
    parser.add_argument("--year", type=int, default=2018, choices=(2016, 2017, 2018))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = convert(args.skflat_data, args.year)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    n_off = sum(1 for v in payload["averages"].values() if v["source"] == "official")
    print(f"Wrote {args.out}: {len(payload['kfactor']['values'])} k-factor bins, "
          f"{len(payload['averages'])} mass points ({n_off} official, {len(payload['averages']) - n_off} private only)")


if __name__ == "__main__":
    main()
