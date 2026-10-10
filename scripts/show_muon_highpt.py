#!/usr/bin/env python3
"""Print a high-pT muon payload (``config.yaml`` ``muon_highpt``) laid out like its sources.

The payloads hold CMS-internal numbers and are not in git (``data/internal/``).
This prints them in the orientation of the source figures so that anyone with
access to the documents can check every number by eye:

- the GE kappa map like the EXO-25-021 maps (eta along x from -2.4, phi along
  y with +pi at the top), value +- uncertainty in each cell;
- the MC resolution coefficients p0..p3 like the fit boxes of the resolution
  figure, with sigma(p) at a few momenta;
- the smearing fractions and every ``source`` / ``note`` field.

Where each number comes from is listed in docs/muon_momentum.md.

    .venv/bin/python scripts/show_muon_highpt.py data/internal/muon_highpt_RunIII2024Summer24.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def _phi_label(lo: float, hi: float) -> str:
    def f(x):
        k = round(3 * x / math.pi)
        return {0: "0", 1: "pi/3", -1: "-pi/3", 3: "pi", -3: "-pi"}.get(k, f"{k}pi/3")
    return f"phi {f(lo)}..{f(hi)}"


def show(path: str) -> None:
    cfg = json.load(open(path))
    print(f"{path}\n  era {cfg.get('era')}, written {cfg.get('written')}, threshold {cfg['threshold_gev']} GeV")
    ge = cfg["ge_kappa"]
    eta, phi = ge["eta_edges"], ge["phi_edges"]
    print(f"\nGE kappa [{ge['units']}]  (rows: phi, +pi at the top; columns: eta)")
    print(f"  source: {ge['source']}")
    head = "".join(f"{f'{eta[i]:+.1f}..{eta[i + 1]:+.1f}':>15s}" for i in range(len(eta) - 1))
    print(f"{'':16s}{head}")
    for j in reversed(range(len(phi) - 1)):
        cells = "".join(f"{ge['kappa'][i][j]:>+8.3f}+-{ge['kappa_unc'][i][j]:.3f}" for i in range(len(eta) - 1))
        print(f"  {_phi_label(phi[j], phi[j + 1]):14s}{cells}")
    m = cfg["mc_smearing"]
    print("\nMC smearing: p -> p (1 + N(0, sigma(p, |eta|) f)), f = sqrt((1 + x)^2 - 1)")
    for key in ("sigma_source", "sigma_note", "sigma_p_max_note", "extra_smear_source", "syst_source"):
        if key in m:
            print(f"  {key}: {m[key]}")
    pmax = float(m.get("sigma_p_max_gev", float("inf")))
    for region, key in (("barrel |eta| < %s" % m["eta_split"], "sigma_barrel"),
                        ("endcap |eta| > %s" % m["eta_split"], "sigma_endcap")):
        c = m[key]
        sig = lambda p: sum(ci * min(p, pmax) ** i for i, ci in enumerate(c))  # noqa: E731
        print(f"  {region}: p0 {c[0]:.6g}  p1 {c[1]:.6g}  p2 {c[2]:.6g}  p3 {c[3]:.6g}   sigma at "
              + ", ".join(f"{p / 1000:g} TeV {sig(p):.4f}" for p in (200, 1000, 2500, 4000)))
    print(f"  extra smearing x = {m['extra_smear']} (f = {math.sqrt((1 + m['extra_smear']) ** 2 - 1):.3f}), "
          f"MuSmear x +- {m['syst']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("payload", nargs="+")
    for path in ap.parse_args(argv).payload:
        show(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
