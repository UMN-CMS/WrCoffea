# Getting Started

## Clone the Repository

Clone with submodules to include the WR_Plotter:
```bash
git clone --recursive git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
```

If you already cloned without `--recursive`, initialize the submodule:
```bash
git submodule update --init --recursive
```

## Environment Setup

There are two ways to set up your environment depending on whether you need Condor submission. Both use **Python 3.12** and **coffea 2025.12.0** to ensure local and Condor environments match.

### Option A: Local runs (no Condor)

Requires Python 3.12 (available on FNAL LPC via CVMFS, since the system Python is 3.9). Create and activate a virtual environment, then install the package:
```bash
/cvmfs/sft.cern.ch/lcg/releases/Python/3.12.11-531c6/x86_64-el9-gcc13-opt/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install -e .
```

> **Note:** The `--system-site-packages` flag is required so the venv can access XRootD Python bindings from CVMFS. If you already have a `.venv` without `wrcoffea` installed, activate it and run `pip install -e .`.

> **Troubleshooting XRootD:** The `xrootd` package is built from source during installation. If `pip install -e .` fails with `ERROR: Wheel 'xrootd' ... is invalid`, a corrupted wheel is cached from a previous attempt. Clear it and retry:
> ```bash
> pip cache remove xrootd
> pip install -e .
> ```

### Option B: Condor runs at FNAL LPC (recommended for production)

Set up the lpcjobqueue Apptainer environment (one-time):
```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

Enter the container using a **pinned tag** (required before each Condor session):
```bash
./shell coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12
```

> **Important:** Always use a pinned container tag instead of `:latest`. The `:latest` tag may lag behind and ship older coffea versions, causing version mismatches between the container's system packages and `pip install -e .` dependencies.

On first launch, the `.env` virtual environment is created automatically. Install the analysis package once:
```bash
pip install -e .
```

This is a one-time step — the `.env` venv persists between sessions, and editable mode picks up code changes automatically. You only need to re-run `pip install -e .` if:
- You switch to a different container tag (which recreates `.env`)
- You change dependencies in `pyproject.toml`

To leave the container, type `exit`.

### Verifying your environment

After setup, confirm that versions match between local and container environments:
```bash
python -c "import coffea; print(coffea.__version__)"   # should print 2025.12.0
python -c "import sys; print(sys.version)"              # should print 3.12.x
```

### Dependency Versions

**Important:** Python package versions in `pyproject.toml` are pinned to **exactly match** the `coffeateam/coffea-dask-almalinux8:2025.12.0-py3.12` container for reproducibility.

If you switch to a different container version:
1. Check installed versions in the new container:
   ```bash
   apptainer exec /cvmfs/unpacked.cern.ch/.../new-container python -c "import coffea; print(coffea.__version__)"
   ```
2. Update `pyproject.toml` to match the new container versions
3. Recreate your local `.venv`:
   ```bash
   rm -rf .venv
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

This ensures your local development environment matches the production container exactly.

## Grid Proxy

Authenticate for accessing grid resources (required for both local and Condor runs):
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```