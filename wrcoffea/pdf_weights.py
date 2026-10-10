"""PDF and alpha_s uncertainties from the generator's LHEPdfWeight.

The members are the output
--------------------------
A Hessian PDF uncertainty is a quadrature over ~100 eigenvector shifts, and it
must be formed at the binning where it is used: the members keep the dataset's
total yield, so each eigenvector moves events between bins, and a per-bin
quadrature formed on fine bins and summed afterwards counts both sides of
every shift as "up" (tested on 2024 signal: a few-fold overestimate in the wide
last fit bins).

So the ``pdf`` family produces no Up/Down labels.  The analyzer fills, per
dataset, every member of the fit masses (``output["pdf_members"]``, same bins
as the analysis histograms), and the writer sums them over the datasets that
share a PDF set and stores one TH2 per (region, set):
``pdfmembers_<region>/<variable>_lha<first LHA ID>`` (x: the variable, y: the
member index, 0 = the set's central member).  The set layouts go into the
manifest (``pdf_sets``).  WR_Plotter (``wrplotter/pdf.py``) and the datacards
combine the members at their own binning:

- ``symmhessian``: delta = sqrt(sum_i (h_i - h_0)^2) over the eigenvectors;
  ``replicas``: delta = standard deviation of the replicas;
- alpha_s: (h_up - h_down) / 2 of the alpha_s members, scaled to
  ``ALPHAS_UNCERTAINTY`` (PDF4LHC 0.0015; the NNPDF3.1 members are
  0.116 / 0.120, so the scale is 0.75).

Conventions
-----------
- Member weights are ``LHEPdfWeight[i] / LHEPdfSumw[i]``: divided by the
  dataset-level weighted mean (fileset metadata, scripts/gen_event_sumw.py),
  so every member keeps the dataset's total yield.  Only acceptance and shape
  move, as for the QCD-scale and parton-shower weights.
- The PDF set of each sample comes from the per-era table written by
  ``scripts/pdf_sets.py`` (config.yaml ``pdf_sets``); the skims dropped the
  NanoAOD branch descriptions.
- Samples without LHEPdfWeight (pure Pythia 8 WW/WZ/ZZ) contribute no members
  (no PDF uncertainty defined; warned at fill time).  Events without weights
  inside a dataset that has them enter every member with the nominal weight.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import awkward as ak
import hist
import numpy as np

from wrcoffea.analysis_config import PDF_SETS, REPO_ROOT

logger = logging.getLogger(__name__)

PDF_SOURCE = "PDF"
ALPHAS_SOURCE = "AlphaS"

# Histograms that carry the PDF members: the fit masses, at their analysis binning.
PDF_OBSERVABLES = frozenset({"mass_fourobject", "mass_twoobject"})

# PDF4LHC recommendation for the alpha_s(MZ) uncertainty.
ALPHAS_UNCERTAINTY = 0.0015

MEMBER_AXIS = "member"
NO_PDF = "none"
MEMBERS_KEY = "pdf_members"
CHUNKS_KEY = "pdf_member_chunks"
ROOT_FOLDER = "pdfmembers"


@dataclass(frozen=True)
class PdfSet:
    """Member layout of one LHAPDF set (from its .info file)."""

    lhaid: int
    name: str
    error_type: str                       # "symmhessian" or "replicas"
    eigenvectors: tuple[int, int]         # first, last member index (inclusive)
    alphas_members: tuple[int, int] | None = None   # (down, up) member indices
    alphas_values: tuple[float, float] | None = None  # alpha_s(MZ) of (down, up)

    @property
    def n_members(self) -> int:
        last = self.alphas_members[1] if self.alphas_members else self.eigenvectors[1]
        return max(last, self.eigenvectors[1]) + 1

    def alphas_scale(self) -> float:
        if not self.alphas_values:
            raise ValueError(f"PDF set {self.name} has no alpha_s members")
        half_spread = (self.alphas_values[1] - self.alphas_values[0]) / 2.0
        return ALPHAS_UNCERTAINTY / half_spread

    def layout(self) -> dict:
        """JSON description for the manifest (what downstream combination needs)."""
        out = {"name": self.name, "error_type": self.error_type, "n_members": self.n_members,
               "central": 0, "eigenvectors": list(self.eigenvectors)}
        if self.alphas_members:
            out.update(alphas_members={"down": self.alphas_members[0], "up": self.alphas_members[1]},
                       alphas_values=list(self.alphas_values), alphas_scale=self.alphas_scale())
        return out


# Sets found in our samples' LHEPdfWeight.  Layouts copied from the .info files
# under /cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/ (checked 2026-10-10).
KNOWN_SETS: dict[int, PdfSet] = {
    # "mem=0 => average on replicas; mem=1-100 => PDF eig.; mem=101 => alphas(MZ)=0.116;
    #  mem=102 => alphas(MZ)=0.120", ErrorType symmhessian+as
    325300: PdfSet(325300, "NNPDF31_nnlo_as_0118_mc_hessian_pdfas", "symmhessian", (1, 100),
                   alphas_members=(101, 102), alphas_values=(0.116, 0.120)),
    # 4-flavour scheme (4FS single top, 4F tribosons), ErrorType symmhessian, 101 members
    325500: PdfSet(325500, "NNPDF31_nnlo_as_0118_nf_4_mc_hessian", "symmhessian", (1, 100)),
}


def pdf_set(lhaid: int) -> PdfSet:
    try:
        return KNOWN_SETS[int(lhaid)]
    except KeyError:
        raise KeyError(
            f"Unknown LHAPDF set {lhaid}: add its member layout to wrcoffea.pdf_weights.KNOWN_SETS "
            "(from the set's .info file) before using its weights."
        ) from None


@lru_cache(maxsize=None)
def _sample_table(era: str) -> dict:
    path = PDF_SETS.get(era)
    if not path:
        raise KeyError(f"No pdf_sets table configured for era '{era}' (config.yaml pdf_sets)")
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        return json.load(f)["samples"]


def sample_lhaid(era: str, sample: str) -> int | None:
    """First LHA ID of *sample*'s LHEPdfWeight in *era*, or None (no PDF weights)."""
    table = _sample_table(era)
    if sample not in table:
        raise KeyError(
            f"Sample '{sample}' is missing from the {era} pdf_sets table; rerun "
            f"scripts/pdf_sets.py --era {era}"
        )
    lhaid = table[sample].get("lhaid")
    return None if lhaid is None else int(lhaid)


class PdfMembers:
    """Per-event member weights of one chunk, computed on demand for a subset.

    *raw* is the jagged ``LHEPdfWeight`` of the chunk, *ratios* the dataset
    means ``LHEPdfSumw``.  Events without weights get 1 for every member.
    """

    def __init__(self, raw, ratios, pdfset: PdfSet):
        self.pdfset = pdfset
        self.ratios = np.asarray(ratios, dtype=np.float64)
        self.n = len(self.ratios)
        if self.n != pdfset.n_members:
            raise ValueError(
                f"LHEPdfSumw has {self.n} entries but set {pdfset.name} has {pdfset.n_members} members"
            )
        counts = ak.to_numpy(ak.num(raw))
        if len(counts) and not np.all((counts == self.n) | (counts == 0)):
            raise ValueError(
                f"LHEPdfWeight has {sorted(set(counts.tolist()))} entries per event, expected {self.n}"
            )
        self._raw = raw
        self.n_missing = int(np.sum(counts == 0))

    def weights(self, cut=None) -> np.ndarray:
        """(n_events, n_members) member factors for the events selected by *cut*."""
        raw = self._raw if cut is None else self._raw[cut]
        w = ak.to_numpy(ak.fill_none(ak.pad_none(raw, self.n, clip=True), 1.0)).astype(np.float64)
        return w / self.ratios


def create_member_hist(observable: str, bins) -> hist.Hist:
    """(region, member, observable) histogram; *bins* = the observable's (n, lo, hi)."""
    return hist.Hist(
        hist.axis.StrCategory([], name="region", growth=True),
        hist.axis.IntCategory([], name=MEMBER_AXIS, growth=True),
        hist.axis.Regular(*bins, name=observable),
        storage=hist.storage.Double(),
    )


def fill_members(store: dict, observable: str, bins, region: str, values, nominal_weight,
                 members: np.ndarray):
    """Fill every member of *observable* in *region* (one entry per event and member)."""
    h = store.get(observable)
    if h is None:
        h = store[observable] = create_member_hist(observable, bins)
    vals = np.asarray(ak.to_numpy(values), dtype=np.float64)
    nom = np.asarray(nominal_weight, dtype=np.float64)
    n_ev, n_mem = members.shape
    if n_ev == 0:
        # Book every member so a dataset with an empty region still has them.
        h.fill(region=region, member=np.arange(n_mem), **{observable: np.full(n_mem, np.nan)},
               weight=np.zeros(n_mem))
        return
    h.fill(
        region=region,
        member=np.tile(np.arange(n_mem), n_ev),
        **{observable: np.repeat(vals, n_mem)},
        weight=(nom[:, None] * members).ravel(),
    )


def member_array(member_hist: hist.Hist, region: str, n_members: int) -> np.ndarray:
    """(n_members, n_bins + 2) contents of one region, flow bins included."""
    if region not in member_hist.axes["region"]:
        raise KeyError(f"no PDF members filled for region {region}")
    ridx = member_hist.axes["region"].index(region)
    axis = member_hist.axes[MEMBER_AXIS]
    view = member_hist.view(flow=True)
    rows = []
    for i in range(n_members):
        if i not in axis:
            raise KeyError(f"PDF member {i} missing for region {region}")
        rows.append(np.asarray(view[ridx, axis.index(i), :], dtype=np.float64))
    return np.stack(rows)


def hessian_shifts(members: np.ndarray, pdfset: PdfSet | dict) -> dict[str, np.ndarray]:
    """Absolute per-bin shifts {"PDF": delta, "AlphaS": signed delta} of member arrays.

    *members* is (n_members, n_bins) at the binning the shifts are wanted at
    (rebin the members first, never the shifts).  AlphaS is absent for sets
    without alpha_s members.  Accepts a PdfSet or its manifest ``layout()``.
    """
    layout = pdfset.layout() if isinstance(pdfset, PdfSet) else pdfset
    m = np.asarray(members, dtype=np.float64)
    h0 = m[layout.get("central", 0)]
    first, last = layout["eigenvectors"]
    eig = m[first:last + 1]
    if layout["error_type"] == "symmhessian":
        delta = np.sqrt(np.sum((eig - h0) ** 2, axis=0))
    elif layout["error_type"] == "replicas":
        delta = np.std(eig, axis=0, ddof=1)
    else:
        raise ValueError(f"Unsupported PDF error type {layout['error_type']!r}")
    out = {PDF_SOURCE: delta}
    if layout.get("alphas_members"):
        a = layout["alphas_members"]
        out[ALPHAS_SOURCE] = layout["alphas_scale"] * (m[a["up"]] - m[a["down"]]) / 2.0
    return out


def collect_members(my_hists: dict) -> tuple[dict, dict]:
    """Pop the per-dataset members and sum them per PDF set.

    Returns ``({lhaid: {observable: Hist}}, {dataset: lhaid or "none"})``.
    A dataset that recorded a PDF set must have filled members in every
    processed chunk.
    """
    by_set: dict[str, dict[str, hist.Hist]] = {}
    datasets: dict[str, str] = {}
    for name, payload in my_hists.items():
        store = payload.pop(MEMBERS_KEY, None)
        member_chunks = payload.pop(CHUNKS_KEY, 0)
        lhaids = (payload.get("dataset_meta") or {}).get("pdf_lhaid")
        if not lhaids:
            if store:
                raise ValueError(f"Dataset '{name}' carries PDF members but no pdf_lhaid record")
            continue
        if len(lhaids) != 1:
            raise ValueError(f"Dataset '{name}' accumulated PDF sets {sorted(lhaids)}; expected one")
        lhaid = next(iter(lhaids))
        datasets[name] = lhaid
        if lhaid == NO_PDF:
            continue
        n_chunks = payload.get("n_chunks")
        if not store or (n_chunks is not None and member_chunks != n_chunks):
            raise ValueError(
                f"Dataset '{name}' (PDF set {lhaid}) filled PDF members in {member_chunks} of "
                f"{n_chunks} chunks")
        target = by_set.setdefault(str(lhaid), {})
        for observable, h in store.items():
            target[observable] = h.copy() if observable not in target else target[observable] + h
    return by_set, datasets


def member_th2(member_hist: hist.Hist, region: str, n_members: int) -> hist.Hist:
    """Writable 2D histogram (observable x member index) of one region, flow included."""
    obs_axis = member_hist.axes[-1]
    out = hist.Hist(
        hist.axis.Regular(obs_axis.size, obs_axis.edges[0], obs_axis.edges[-1], name=obs_axis.name),
        hist.axis.Regular(n_members, -0.5, n_members - 0.5, name=MEMBER_AXIS, flow=False),
        storage=hist.storage.Double(),
    )
    out.view(flow=True)[...] = member_array(member_hist, region, n_members).T
    return out


def root_key(region: str, observable: str, lhaid) -> str:
    return f"{ROOT_FOLDER}_{region}/{observable}_lha{lhaid}"


def write_members(root_file, by_set: dict) -> dict:
    """Write the summed members; returns the manifest's ``pdf_sets`` layouts."""
    layouts = {}
    for lhaid, store in sorted(by_set.items()):
        pdfset = pdf_set(int(lhaid))
        layouts[str(lhaid)] = pdfset.layout()
        for observable, h in sorted(store.items()):
            for region in h.axes["region"]:
                root_file[root_key(region, observable, lhaid)] = member_th2(h, region, pdfset.n_members)
    return layouts
