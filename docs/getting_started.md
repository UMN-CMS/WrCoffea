# Getting started

What this page answers: how to set up the environment on a laptop or at the FNAL LPC, get a grid
proxy, run one sample and the tests. Verified on 2026-10-12.

## Clone

```bash
git clone --recursive -b billy_dev git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
git submodule update --init --recursive      # if the clone was made without --recursive
```

The analyzer is on branch `billy_dev`, the plotter submodule on `billyDev`. The default branch
`main` is far behind and has the old documentation, so clone with `-b billy_dev`. Run every command
from the repository root: `data/...` and `WR_Plotter/rootfiles/...` are resolved relative to it.

## Environment

`pyproject.toml` requires Python 3.11 or later and pins coffea 2025.12.0, the version of the
`coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12` container. Use Python 3.12, as the container
does, so laptop and LPC runs match. Python 3.14 does not work (no correctionlib wheel).

**Laptop (local runs, plotting, derivations).**

```bash
python3.12 -m venv .venv                    # or: uv venv --python 3.12 .venv
.venv/bin/pip install --upgrade pip setuptools
.venv/bin/pip install -e ".[xrootd,test]"
```

`[xrootd]` builds the XRootD bindings needed to open `root://` URLs (a corrupted cached wheel
shows as `ERROR: Wheel 'xrootd' ... is invalid`: `pip cache remove xrootd` and retry); `[test]`
adds pytest. Everything in `WR_Plotter/bin/` runs with this same environment. The fits in
`WR_Plotter/signal_fitting/` need ROOT and Combine; see `WR_Plotter/signal_fitting/README.md`.

**FNAL LPC.** The system Python is 3.9; use the CVMFS one for a local environment:

```bash
/cvmfs/sft.cern.ch/lcg/releases/Python/3.12.11-531c6/x86_64-el9-gcc13-opt/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate && pip install -e .     # XRootD comes from the system site-packages
```

To run the analyzer on Condor, work inside the pinned coffea container, which creates its own
`.env`:

```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh && bash bootstrap.sh   # once
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12                                                 # every session
pip install -e .                                                                                            # once per container tag
```

Never use the `:latest` tag: it can lag behind and mismatch the pinned versions. Details and the
Condor flags: [condor.md](condor.md). The skim campaign's Condor jobs do not use this container:
they run in their own image on the worker nodes ([skimming.md](skimming.md)). Which machine does
what: the laptop runs the analyzer, the plots, the fits and the derivations; the LPC runs the
skim campaigns and large Condor productions.

Check the environment:

```bash
.venv/bin/python -c "import coffea, sys; print(coffea.__version__, sys.version.split()[0])"   # 2025.12.0 3.12.x
.venv/bin/python -m pytest tests -q                                                           # 836 tests, offline, about two minutes
```

## Grid proxy

The skims stream from Wisconsin and DBS queries need a VOMS proxy. That needs a CMS grid
certificate (`usercert.pem` and `userkey.pem` in `.globus/` in your home directory) registered in
the CMS VO. On a laptop you also need `voms-proxy-init` and the CA certificates, with
`X509_CERT_DIR` pointing at them.

```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

The tools read `X509_USER_PROXY` (the proxy file) and `X509_CERT_DIR` (the CA certificates) when
they are set, which is the way to keep a proxy at a fixed path on a laptop.

## First run

```bash
.venv/bin/python bin/run_analysis.py RunIII2024Summer24 DYJets --maxfiles 1 --maxchunks 1 --dir smoke   # a minute
.venv/bin/python bin/run_analysis.py RunIII2024Summer24 DYJets --dir mytag                              # the full sample
cd WR_Plotter && ../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir mytag --local-plots
```

The smoke run writes `WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/smoke/WRAnalyzer_DYJets.root`
and its manifest. Its normalization is meaningless (one file, the full sum of weights): never read
data/MC agreement from a smoke run; delete the `smoke` directory afterwards. Everything else about
running: [run_analysis.md](run_analysis.md).

## Where to read next

- What the analysis does: [analysis.md](analysis.md).
- Which corrections each era applies: [corrections.md](corrections.md).
- Configs, cross sections and filesets: [inputs.md](inputs.md).
- A new era: [adding_a_new_era.md](adding_a_new_era.md).
