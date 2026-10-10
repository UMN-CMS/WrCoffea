"""Registry of systematic variations for the WR analysis.

One place that says, for every systematic source, what it is called, how it
is produced, and where it applies.  The analyzer, the histogram writer, and
the CLI all read from here so that "requested", "produced", "unaffected", and
"omitted" mean the same thing everywhere.

Vocabulary
----------
family
    The name a user passes to ``--systs`` (``pileup``, ``sf``, ``jes`` ...).
    One family can expand to several sources (``sf`` expands to one source
    per lepton scale-factor component; ``jes`` to one source per configured
    JES uncertainty).
source (``Systematic``)
    One Up/Down pair.  ``labels()`` gives the two values written on the
    histogram ``syst`` axis, e.g. ``("PileupUp", "PileupDown")``.
kind
    ``weight``         - same selection, different per-event weight.
    ``shape``          - kinematics are shifted and the selection re-run.
    ``normalization``  - flat scaling; never filled at event level, the
                         export scales the nominal histogram instead.
    ``members``        - generator weight sets combined downstream (PDF
                         eigenvectors): no Up/Down labels; the analyzer fills
                         per-member histograms and the writer stores them
                         (``wrcoffea.pdf_weights``).

Applicability versus availability
----------------------------------
``Systematic.applies(region=..., is_mc=...)`` is physics: can this source
change this region for this kind of dataset at all?  Data never has
variations, and muon scale factors cannot touch an ``ee`` region.  A source
that does not apply is *explicitly unaffected* there and contributes its
nominal to any varied total.

``Systematic.available(era)`` is bookkeeping: are the correction inputs for
this source configured for this era?  A source that applies but is not
available is *unavailable*.  It never silently becomes a nominal copy; the
user either drops the family or records the omission with ``--omit-systs``.

``Systematic.expected(era=..., region=..., is_mc=...)`` is both, and is what
the analyzer produces and the writer demands.

Coverage bookkeeping (what the analyzer records per chunk)
-----------------------------------------------------------
``output[dataset]["syst_coverage"][region][label]`` counts the chunks in
which every requested histogram for that (region, label) was filled, and
``output[dataset]["n_chunks"]`` counts processed chunks.  Both accumulate
across chunks by addition, so at save time a label is complete for a
dataset only when its count equals ``n_chunks``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable

from wrcoffea.analysis_config import (
    DY_RESHAPE_JSONS,
    DY_ZPT_REWEIGHT_JSONS,
    ELECTRON_JSONS,
    JERC_JSONS,
    LSF_SF,
    LUMI_UNC,
    MUON_JSONS,
    PDF_SETS,
    PILEUP_JSONS,
)

WEIGHT = "weight"
SHAPE = "shape"
NORMALIZATION = "normalization"
MEMBERS = "members"
KINDS = (WEIGHT, SHAPE, NORMALIZATION, MEMBERS)

NOMINAL = "Nominal"
DIRECTIONS = ("Up", "Down")

# By default every histogram receives every enabled variation.
# ``--syst-observables fit`` restricts the variations to the fit masses
# (a memory/throughput option), and an explicit list restricts them further.
FIT_OBSERVABLES: tuple[str, ...] = ("mass_dilepton", "mass_fourobject", "mass_twoobject")
ALL_OBSERVABLES = "all"
FIT_KEYWORD = "fit"

# Public CLI families, in the order they are documented.
FAMILIES: tuple[str, ...] = (
    "lumi", "pileup", "sf", "jer", "jes", "elescale", "elesmear", "muscale", "musmear",
    "dyzpt", "lsf", "dyreshape", "scale", "ps", "pdf",
)

# The JES source that keeps the historical bare "JES" label.
_JES_TOTAL_SOURCE = "CMS_scale_j_Total"


# ---------------------------------------------------------------------------
# Region predicates.  These mirror the branching in
# ``WrAnalysis._lepton_sfs_for_region`` / ``_region_trigger``; the registry
# test checks that the two stay in step.
# ---------------------------------------------------------------------------

def _uses_muon(region: str) -> bool:
    return any(tag in region for tag in ("mumu", "flavor_cr", "mue", "emu"))


def _uses_electron(region: str) -> bool:
    return any(tag in region for tag in ("ee", "flavor_cr", "mue", "emu"))


def _trigger(region: str) -> str:
    return "e" if ("ee" in region or "emu" in region) else "mu"


def _is_boosted(region: str) -> bool:
    return "boosted" in region


def _uses_lsf(region: str) -> bool:
    # The LSF requirement (and its scale factor) enters the boosted SR and
    # flavour CR; the boosted DY CR uses AK8 jets without it.
    return _is_boosted(region) and "dy_cr" not in region


def _all_regions(region: str) -> bool:
    return True


def _all_eras(era: str) -> bool:
    return True


def _has_muon_payload(key: str) -> Callable[[str], bool]:
    return lambda era: key in (MUON_JSONS.get(era) or {})


def _has_electron_payload(key: str) -> Callable[[str], bool]:
    return lambda era: key in (ELECTRON_JSONS.get(era) or {})


_has_electron_scale_smearing = _has_electron_payload("ScaleNSmear")


def _jerc_blocks(era: str) -> list[dict]:
    cfg = JERC_JSONS.get(era) or {}
    return [cfg.get(key) or {} for key in ("JERC_AK4", "JERC_AK8")]


def _has_jer_variations(era: str) -> bool:
    # JER Up/Down need the resolution scale-factor tag on both jet collections.
    blocks = _jerc_blocks(era)
    return era in JERC_JSONS and all(b.get("JR_tag_MC") for b in blocks)


def _has_jes_variations(era: str) -> bool:
    # JES sources come from ``jes_unc_mapping``; AK4 and AK8 must agree.
    blocks = _jerc_blocks(era)
    mappings = [dict(b.get("jes_unc_mapping") or {}) for b in blocks]
    return era in JERC_JSONS and bool(mappings[0]) and mappings[0] == mappings[1]


def _jes_sources(era: str) -> dict[str, str]:
    cfg = JERC_JSONS.get(era, {}).get("JERC_AK4", {})
    return dict(cfg.get("jes_unc_mapping") or {})


# ---------------------------------------------------------------------------
# Source definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Systematic:
    """One systematic source producing an Up/Down pair."""

    name: str
    family: str
    kind: str
    regions: Callable[[str], bool] = _all_regions
    # Are the correction inputs configured for this era?
    available_in: Callable[[str], bool] = _all_eras
    objects: tuple[str, ...] = ()
    # For ``shape`` sources: the ``<field_source>`` in ``pt_<field_source>_<dir>``.
    field_source: str | None = None
    # Shape sources whose swap also replaces the object mass (jets).
    varies_mass: bool = False
    note: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown systematic kind '{self.kind}' for {self.name}")
        if self.kind == SHAPE and (not self.objects or not self.field_source):
            raise ValueError(f"Shape systematic {self.name} needs objects and field_source")

    def labels(self) -> tuple[str, ...]:
        if self.kind == MEMBERS:
            return ()                     # stored as members, combined downstream
        return tuple(f"{self.name}{d}" for d in DIRECTIONS)

    def label(self, direction: str) -> str:
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        return f"{self.name}{direction}"

    def applies(self, *, region: str, is_mc: bool) -> bool:
        """Physical applicability: can this source change *region* for MC/data?"""
        return bool(is_mc and self.regions(region))

    def available(self, era: str) -> bool:
        """Are the correction inputs for this source configured for *era*?"""
        return bool(self.available_in(era))

    def expected(self, *, era: str, region: str, is_mc: bool) -> bool:
        return self.applies(region=region, is_mc=is_mc) and self.available(era)

    def variation_field(self, direction: str) -> str:
        """NanoAOD-side field carrying the shifted pT for a shape source."""
        if self.kind != SHAPE:
            raise ValueError(f"{self.name} is not a shape systematic")
        return f"pt_{self.field_source}_{direction.lower()}"

    def required_fields(self, direction: str) -> tuple[str, ...]:
        """Fields a non-empty affected collection must carry for *direction*."""
        pt_field = self.variation_field(direction)
        if self.varies_mass:
            return (pt_field, pt_field.replace("pt_", "mass_", 1))
        return (pt_field,)


# ---------------------------------------------------------------------------
# Family expansion
# ---------------------------------------------------------------------------

def _sf_sources() -> list[Systematic]:
    """One source per lepton scale-factor component.

    Names follow the analyzer convention ``<Object><Component>Sf`` so the
    labels match what ``_lepton_sfs_for_region`` produces.
    """
    muon_reco = _has_muon_payload("RECO")
    muon_trig = _has_muon_payload("TRIGGER")
    ele_reco = _has_electron_payload("RECO")
    ele_trig_payload = _has_electron_payload("TRIGGER")

    def mu(region):
        return _uses_muon(region)

    def mu_trig(region):
        return _uses_muon(region) and _trigger(region) == "mu"

    def loose_mu(region):
        return _is_boosted(region) and _uses_muon(region)

    def ele(region):
        return _uses_electron(region)

    def ele_trig(region):
        return _uses_electron(region) and _trigger(region) == "e"

    def loose_ele(region):
        return _is_boosted(region) and _uses_electron(region)

    return [
        Systematic("MuonRecoSf", "sf", WEIGHT, regions=mu, available_in=muon_reco, objects=("Muon",)),
        Systematic("MuonIdSf", "sf", WEIGHT, regions=mu, available_in=muon_reco, objects=("Muon",)),
        Systematic("MuonIsoSf", "sf", WEIGHT, regions=mu, available_in=muon_reco, objects=("Muon",)),
        Systematic("MuonTrigSf", "sf", WEIGHT, regions=mu_trig, available_in=muon_trig, objects=("Muon",)),
        Systematic("LooseMuonRecoSf", "sf", WEIGHT, regions=loose_mu, available_in=muon_reco, objects=("Muon",)),
        Systematic("LooseMuonIdSf", "sf", WEIGHT, regions=loose_mu, available_in=muon_reco, objects=("Muon",)),
        Systematic("LooseElectronRecoSf", "sf", WEIGHT, regions=loose_ele, available_in=ele_reco, objects=("Electron",)),
        Systematic("ElectronRecoSf", "sf", WEIGHT, regions=ele, available_in=ele_reco, objects=("Electron",)),
        Systematic("ElectronIdSf", "sf", WEIGHT, regions=ele, available_in=ele_reco, objects=("Electron",)),
        Systematic("ElectronTrigSf", "sf", WEIGHT, regions=ele_trig, available_in=ele_trig_payload, objects=("Electron",)),
    ]


def _jes_sources_for(era: str) -> list[Systematic]:
    out = []
    for cat_name in _jes_sources(era):
        name = "JES" if cat_name == _JES_TOTAL_SOURCE else f"JES_{cat_name}"
        out.append(Systematic(
            name, "jes", SHAPE, available_in=_has_jes_variations,
            objects=("Jet", "FatJet"), field_source=cat_name, varies_mass=True,
        ))
    return out


def _dyzpt_sources() -> list[Systematic]:
    """QCD and EW uncertainties of the DY Z-pT reweighting (Run 2 port).

    The weight is only non-trivial for ``DYJets``; every other MC dataset
    produces these labels as nominal copies so that group merging treats
    them as evaluated-and-unchanged.
    """
    from wrcoffea.dy_reweight import SOURCE_NAMES

    def available(era):
        return era in DY_ZPT_REWEIGHT_JSONS

    return [
        Systematic(name, "dyzpt", WEIGHT, available_in=available,
                   note="Non-trivial for DYJets only; identity for other datasets.")
        for name in SOURCE_NAMES
    ]


def _dyreshape_sources() -> list[Systematic]:
    """Stat and ee-vs-mumu uncertainties of the data-driven DY reshape."""
    from wrcoffea.dy_reshape import SOURCE_NAMES

    def available(era):
        return era in DY_RESHAPE_JSONS

    return [
        Systematic(name, "dyreshape", WEIGHT, available_in=available,
                   note="Non-trivial for DYJets only; identity for other datasets.")
        for name in SOURCE_NAMES
    ]


_MUON_TUNEP_NOTE = (
    "TuneP branch uses a flat 5%/TeV scale placeholder and smear == nominal; "
    "replace with the Muon POG generalized-endpoint prescription before use in a fit."
)

_THEORY_NOTE = (
    "Per-event generator weight divided by the dataset-level ratio from the config "
    "(scripts/gen_event_sumw.py), so the total yield is unchanged and only acceptance "
    "and shape move. A sample generated without the branch (pure-Pythia 8 WW/WZ/ZZ have no "
    "LHEScaleWeight) gets the nominal weight with a warning, and so do single events without "
    "the weights (one file each in 2022 ZZZ and WR MWR6000_N1500 22EE, see docs/skimming.md)."
)

_PDF_NOTE = (
    "LHEPdfWeight members of mass_fourobject / mass_twoobject, divided by the dataset mean (only "
    "acceptance and shape move), summed per PDF set and written as pdfmembers_<region>/<var>_lha<id> "
    "(TH2: variable x member); WR_Plotter and the datacards form the Hessian quadrature (and the "
    "alpha_s shift, scaled to 0.0015) at their own binning (wrcoffea/pdf_weights.py)."
)

_STATIC_SOURCES: dict[str, list[Systematic]] = {
    "lumi": [Systematic("Lumi", "lumi", NORMALIZATION, available_in=lambda e: e in LUMI_UNC)],
    # QCD renormalization / factorization scales (LHEScaleWeight, 9-weight NanoAOD order)
    "scale": [Systematic("ScaleMuR", "scale", WEIGHT, note=_THEORY_NOTE),
              Systematic("ScaleMuF", "scale", WEIGHT, note=_THEORY_NOTE),
              Systematic("ScaleMuRF", "scale", WEIGHT, note=_THEORY_NOTE)],
    # parton-shower ISR / FSR (PSWeight: [ISR x2, FSR x2, ISR x0.5, FSR x0.5])
    "ps": [Systematic("PSISR", "ps", WEIGHT, note=_THEORY_NOTE),
           Systematic("PSFSR", "ps", WEIGHT, note=_THEORY_NOTE)],
    # PDF eigenvectors and alpha_s (LHEPdfWeight; per-sample set from config.yaml pdf_sets)
    "pdf": [Systematic("PDF", "pdf", MEMBERS, available_in=lambda e: e in PDF_SETS, note=_PDF_NOTE),
            Systematic("AlphaS", "pdf", MEMBERS, available_in=lambda e: e in PDF_SETS, note=_PDF_NOTE)],
    "pileup": [Systematic("Pileup", "pileup", WEIGHT, available_in=lambda e: e in PILEUP_JSONS)],
    "sf": _sf_sources(),
    "jer": [Systematic("JER", "jer", SHAPE, available_in=_has_jer_variations,
                       objects=("Jet", "FatJet"), field_source="jer", varies_mass=True)],
    "elescale": [Systematic("EleScale", "elescale", SHAPE, available_in=_has_electron_scale_smearing,
                            objects=("Electron",), field_source="scale")],
    "elesmear": [Systematic("EleSmear", "elesmear", SHAPE, available_in=_has_electron_scale_smearing,
                            objects=("Electron",), field_source="smear")],
    "muscale": [Systematic("MuScale", "muscale", SHAPE, objects=("Muon",),
                           field_source="scale", note=_MUON_TUNEP_NOTE)],
    "musmear": [Systematic("MuSmear", "musmear", SHAPE, objects=("Muon",),
                           field_source="smear", note=_MUON_TUNEP_NOTE)],
    "dyzpt": _dyzpt_sources(),
    "lsf": [Systematic("LsfSf", "lsf", WEIGHT, regions=_uses_lsf, available_in=lambda e: e in LSF_SF,
                       note="Flat per-year value by lepton-in-jet flavour (AN-19-083 Table 60).")],
    "dyreshape": _dyreshape_sources(),
}


def normalize_families(families: Iterable[str] | None) -> frozenset[str]:
    """Lower-case, strip, drop empties, and reject unknown family names."""
    out = set()
    for fam in families or ():
        fam = str(fam).strip().lower()
        if not fam:
            continue
        if fam not in FAMILIES:
            raise ValueError(f"Unknown systematic family '{fam}'. Choices: {', '.join(FAMILIES)}")
        out.add(fam)
    return frozenset(out)


def sources_for_family(family: str, era: str) -> list[Systematic]:
    """All sources a family defines for *era* (before era applicability)."""
    if family == "jes":
        return _jes_sources_for(era)
    return list(_STATIC_SOURCES[family])


def systematics_for(era: str, families: Iterable[str] | None, *, kinds: Iterable[str] | None = None) -> list[Systematic]:
    """Expand *families* into their sources for *era* (no availability filter).

    *era* only matters for families whose source list is configured per era
    (``jes``).  Use :func:`producible` to keep only the sources whose inputs
    exist and that were not explicitly omitted.
    """
    fams = normalize_families(families)
    wanted_kinds = set(kinds) if kinds is not None else set(KINDS)
    out: list[Systematic] = []
    for fam in FAMILIES:
        if fam not in fams:
            continue
        out.extend(src for src in sources_for_family(fam, era) if src.kind in wanted_kinds)
    return out


def all_sources(families: Iterable[str] | None = None) -> list[Systematic]:
    """Every source of *families* (default: all) across every configured era."""
    fams = normalize_families(families if families is not None else FAMILIES)
    out: list[Systematic] = []
    seen: set[str] = set()
    for fam in FAMILIES:
        if fam not in fams:
            continue
        eras = list(JERC_JSONS) if fam == "jes" else [""]
        for era in eras:
            for src in sources_for_family(fam, era):
                if src.name not in seen:
                    seen.add(src.name)
                    out.append(src)
    return out


def normalize_omitted(omitted: Iterable[str] | None) -> frozenset[str]:
    """Validate ``--omit-systs`` names against the known source names."""
    known = {src.name for src in all_sources()}
    out = set()
    for name in omitted or ():
        name = str(name).strip()
        if not name:
            continue
        if name not in known:
            raise ValueError(
                f"Unknown systematic source '{name}' in --omit-systs. Known sources: "
                f"{', '.join(sorted(known))}"
            )
        out.add(name)
    return frozenset(out)


def producible(era: str, families: Iterable[str] | None, *, omitted: Iterable[str] | None = None,
               kinds: Iterable[str] | None = None) -> list[Systematic]:
    """Sources of *families* that have inputs for *era* and were not omitted."""
    skip = normalize_omitted(omitted)
    return [
        src for src in systematics_for(era, families, kinds=kinds)
        if src.available(era) and src.name not in skip
    ]


def validate_requested(era: str, families: Iterable[str] | None,
                       omitted: Iterable[str] | None = None) -> list[Systematic]:
    """Fail loudly when a requested source cannot be produced for *era*.

    A family with no sources at all for the era (``jes`` without a
    ``jes_unc_mapping``) and a source whose inputs are missing (the electron
    trigger payload for UL18) are both errors.  The second kind can be turned
    into a recorded omission by naming the source in *omitted*.
    """
    fams = normalize_families(families)
    skip = normalize_omitted(omitted)
    missing_families, unavailable = [], []
    for fam in sorted(fams):
        candidates = sources_for_family(fam, era)
        if not candidates:
            missing_families.append(fam)
            continue
        for src in candidates:
            if src.name in skip:
                continue
            if not src.available(era):
                unavailable.append(src.name)
    problems = []
    if missing_families:
        problems.append(
            f"families {missing_families} define no sources for era '{era}'"
        )
    if unavailable:
        problems.append(
            f"sources {unavailable} have no correction inputs configured for era '{era}' "
            f"(drop the family, add the payloads to config.yaml, or record the omission with "
            f"--omit-systs {' '.join(unavailable)})"
        )
    if problems:
        raise ValueError("Cannot produce the requested systematics: " + "; ".join(problems) + ".")
    return producible(era, fams, omitted=skip)


def expected_labels(era: str, families: Iterable[str] | None, region: str, *, is_mc: bool,
                    omitted: Iterable[str] | None = None,
                    kinds: Iterable[str] | None = None) -> list[str]:
    """Axis labels the analyzer must produce for (era, region) beyond Nominal."""
    out = []
    for src in producible(era, families, omitted=omitted, kinds=kinds):
        if src.applies(region=region, is_mc=is_mc):
            out.extend(src.labels())
    return out


def required_labels(era: str, families: Iterable[str] | None, region: str, *, is_mc: bool,
                    omitted: Iterable[str] | None = None) -> list[str]:
    """Labels a dataset is *required* to have produced: applies and not omitted.

    Unlike :func:`expected_labels` this ignores input availability on
    purpose: a source that applies but has no inputs must show up as missing
    at save time unless the user recorded the omission.
    """
    skip = normalize_omitted(omitted)
    out = []
    for src in systematics_for(era, families, kinds=(WEIGHT, SHAPE)):
        if src.name in skip or not src.applies(region=region, is_mc=is_mc):
            continue
        out.extend(src.labels())
    return out


def source_of_label(label: str, families: Iterable[str] | None = None) -> Systematic | None:
    """Resolve an axis label to its source (any era), or None if unknown."""
    for src in all_sources(families if families is not None else FAMILIES):
        if label in src.labels():
            return src
    return None


def normalization_factor(src: Systematic, era: str) -> float:
    """Fractional Up shift for a normalization source (Down uses -factor)."""
    if src.kind != NORMALIZATION:
        raise ValueError(f"{src.name} is not a normalization systematic")
    if src.family == "lumi":
        return float(LUMI_UNC[era])
    raise ValueError(f"No normalization factor defined for {src.name}")


# ---------------------------------------------------------------------------
# Production configuration
# ---------------------------------------------------------------------------

def production_config(families: Iterable[str] | None, omitted: Iterable[str] | None,
                      syst_observables: frozenset[str] | None, *, region: str = "both",
                      variants: Iterable[str] = (), blind: bool = False) -> str:
    """Canonical string describing what a processing run was asked to produce.

    Stored in every chunk's payload so that chunks and datasets produced with
    different settings are never merged, and so the writer can demand every
    region and variation the run was configured for.  *blind* is run-level
    (one run may process data and MC): it says that data withheld its signal
    regions (``wrcoffea.regions``); MC ignores it.
    """
    return json.dumps({
        "blind": bool(blind),
        "families": sorted(normalize_families(families)),
        "omitted": sorted(normalize_omitted(omitted)),
        "observables": describe_syst_observables(syst_observables),
        "region": (region or "both").lower(),
        "variants": sorted(str(v) for v in (variants or ())),
    }, sort_keys=True)


def parse_production_config(text: str) -> dict:
    cfg = json.loads(text)
    return {
        # Payloads written before blinding existed were never blinded.
        "blind": bool(cfg.get("blind", False)),
        "families": frozenset(cfg.get("families", ())),
        "omitted": frozenset(cfg.get("omitted", ())),
        "observables": (None if cfg.get("observables") == ALL_OBSERVABLES
                        else frozenset(cfg.get("observables", ()))),
        "region": cfg.get("region", "both"),
        "variants": frozenset(cfg.get("variants", ())),
    }


def resolve_syst_observables(names: Iterable[str] | None, known: Iterable[str]) -> frozenset[str] | None:
    """Turn a ``--syst-observables`` value into a frozenset, or None for all.

    ``None`` / empty input and the literal ``"all"`` select every observable
    (represented as ``None``); the literal ``"fit"`` selects
    :data:`FIT_OBSERVABLES` (restricted to the names that are actually
    booked); otherwise an explicit list of histogram names.
    """
    known = set(known)
    names = [str(n).strip() for n in (names or ()) if str(n).strip()]
    if not names or any(n.lower() == ALL_OBSERVABLES for n in names):
        if len(names) > 1:
            raise ValueError("--syst-observables: 'all' cannot be combined with explicit names")
        return None
    if any(n.lower() == FIT_KEYWORD for n in names):
        if len(names) > 1:
            raise ValueError("--syst-observables: 'fit' cannot be combined with explicit names")
        return frozenset(n for n in FIT_OBSERVABLES if n in known)
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"--syst-observables: unknown observable(s) {unknown}")
    return frozenset(names)


def describe_syst_observables(cov: frozenset[str] | None) -> list[str] | str:
    return ALL_OBSERVABLES if cov is None else sorted(cov)
