"""Tests for the PDF / alpha_s systematics (wrcoffea.pdf_weights and its analyzer hooks).

- member weights are divided by the dataset means, events without weights get 1;
- the members are summed per PDF set and written as TH2s, never combined early;
- the combination (symmetric-Hessian quadrature, alpha_s scaled to 0.0015)
  done on the rebinned members is exact for the integral, unlike a per-bin
  quadrature summed afterwards;
- the pdf family has no Up/Down labels; samples without weights carry no members.
"""

import logging

import awkward as ak
import hist
import numpy as np
import pytest
import uproot

import wrcoffea.analyzer as analyzer_mod
import wrcoffea.systematics as systematics_mod
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.pdf_weights import (
    CHUNKS_KEY,
    KNOWN_SETS,
    MEMBERS_KEY,
    NO_PDF,
    PdfMembers,
    collect_members,
    fill_members,
    hessian_shifts,
    member_array,
    root_key,
    write_members,
)
from wrcoffea.systematics import expected_labels, sources_for_family

NNPDF31 = KNOWN_SETS[325300]       # 1 + 100 eigenvectors + 2 alpha_s
NNPDF31_NF4 = KNOWN_SETS[325500]   # 1 + 100 eigenvectors
ERA = "RunIII2024Summer24"
BINS = (800, 0.0, 8000.0)


@pytest.fixture(autouse=True)
def _reset_warnings():
    analyzer_mod._WARN_ONCE_ANALYZER.clear()
    yield
    analyzer_mod._WARN_ONCE_ANALYZER.clear()


# ---------------------------------------------------------------------------
# Member weights
# ---------------------------------------------------------------------------

def test_member_weights_are_divided_by_the_dataset_means_and_missing_events_get_one():
    rng = np.random.default_rng(1)
    full = rng.uniform(0.9, 1.1, size=(3, 103))
    raw = ak.Array([full[0].tolist(), [], full[1].tolist(), full[2].tolist()])
    ratios = rng.uniform(0.95, 1.05, size=103)
    members = PdfMembers(raw, ratios, NNPDF31)
    assert members.n_missing == 1
    w = members.weights()
    np.testing.assert_allclose(w[0], full[0] / ratios)
    np.testing.assert_allclose(w[1], 1.0 / ratios)       # no weights: factor 1 before the mean
    cut = np.array([False, False, True, True])
    np.testing.assert_allclose(members.weights(cut), full[1:] / ratios)


def test_member_count_must_match_the_set_and_every_event():
    with pytest.raises(ValueError, match="has 101 members"):
        PdfMembers(ak.Array([[1.0] * 103]), [1.0] * 103, NNPDF31_NF4)
    with pytest.raises(ValueError, match="entries per event, expected 103"):
        PdfMembers(ak.Array([[1.0] * 103, [1.0] * 101]), [1.0] * 103, NNPDF31)


# ---------------------------------------------------------------------------
# Combination rule
# ---------------------------------------------------------------------------

def test_symmetric_hessian_quadrature_and_alphas_scaling():
    rng = np.random.default_rng(2)
    h0 = rng.uniform(10, 20, size=12)
    eig = h0 + rng.normal(0, 0.3, size=(100, 12))
    as_down, as_up = h0 * 0.98, h0 * 1.03
    shifts = hessian_shifts(np.vstack([h0, eig, as_down, as_up]), NNPDF31)
    np.testing.assert_allclose(shifts["PDF"], np.sqrt(((eig - h0) ** 2).sum(axis=0)))
    # (0.120 - 0.116) / 2 = 0.002 -> scaled to 0.0015: factor 0.75 on (h_up - h_down) / 2
    np.testing.assert_allclose(shifts["AlphaS"], 0.75 * (as_up - as_down) / 2)
    # the manifest layout gives the same result
    np.testing.assert_allclose(hessian_shifts(np.vstack([h0, eig, as_down, as_up]), NNPDF31.layout())["PDF"],
                               shifts["PDF"])


def test_set_without_alphas_members_has_no_alphas_shift():
    assert set(hessian_shifts(np.ones((101, 3)), NNPDF31_NF4)) == {"PDF"}


def test_rebinning_members_before_the_quadrature_is_what_keeps_integrals_right():
    """Yield-preserving eigenvectors move events between bins: the quadrature of the
    rebinned members is exact, the sum of fine-bin quadratures is not."""
    h0 = np.array([10.0, 10.0])
    eig = np.tile(h0, (100, 1))
    eig[0] = [11.0, 9.0]                                  # one eigenvector, migration only
    members = np.vstack([h0, eig, h0, h0])
    fine = hessian_shifts(members, NNPDF31)["PDF"].sum()
    merged = hessian_shifts(members.sum(axis=1, keepdims=True), NNPDF31)["PDF"][0]
    assert fine == pytest.approx(2.0) and merged == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Filling, collecting, writing
# ---------------------------------------------------------------------------

def _payload(lhaid="325300", n_members=103, seed=3, n_chunks=1):
    rng = np.random.default_rng(seed)
    n = 3000
    mass = rng.uniform(0, 3000, size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    members = rng.normal(1.0, 0.02, size=(n, n_members))
    store = {}
    fill_members(store, "mass_fourobject", BINS, "r", mass, w, members)
    payload = {MEMBERS_KEY: store, CHUNKS_KEY: n_chunks, "n_chunks": n_chunks,
               "dataset_meta": {"pdf_lhaid": {lhaid}}}
    return payload, mass, w, members


def test_fill_members_books_every_member_with_the_nominal_times_the_factor():
    payload, mass, w, members = _payload()
    arr = member_array(payload[MEMBERS_KEY]["mass_fourobject"], "r", 103)
    assert arr.shape == (103, 802)
    np.testing.assert_allclose(arr.sum(axis=1), (w[:, None] * members).sum(axis=0))
    store = {}
    fill_members(store, "mass_fourobject", BINS, "empty", np.array([]), np.array([]), np.zeros((0, 103)))
    assert member_array(store["mass_fourobject"], "empty", 103).sum() == 0.0


def test_collect_sums_datasets_of_the_same_set_and_skips_samples_without_weights():
    a, *_ = _payload(seed=4)
    b, *_ = _payload(seed=5)
    c = {"n_chunks": 1, "dataset_meta": {"pdf_lhaid": {NO_PDF}}}
    expected = (member_array(a[MEMBERS_KEY]["mass_fourobject"], "r", 103)
                + member_array(b[MEMBERS_KEY]["mass_fourobject"], "r", 103))
    by_set, datasets = collect_members({"a": a, "b": b, "c": c})
    assert datasets == {"a": "325300", "b": "325300", "c": NO_PDF}
    assert MEMBERS_KEY not in a and CHUNKS_KEY not in a
    np.testing.assert_allclose(member_array(by_set["325300"]["mass_fourobject"], "r", 103), expected)


def test_members_missing_in_some_chunks_are_an_error():
    a, *_ = _payload(n_chunks=2)
    a[CHUNKS_KEY] = 1
    with pytest.raises(ValueError, match="in 1 of 2 chunks"):
        collect_members({"a": a})


def test_written_th2_round_trips(tmp_path):
    a, *_ = _payload()
    by_set, _ = collect_members({"a": a})
    expected = member_array(by_set["325300"]["mass_fourobject"], "r", 103)
    path = tmp_path / "out.root"
    with uproot.recreate(path) as f:
        layouts = write_members(f, by_set)
    assert layouts["325300"]["alphas_scale"] == pytest.approx(0.75)
    with uproot.open(path) as f:
        h = f[root_key("r", "mass_fourobject", "325300")]
        vals = h.values(flow=True)
    assert vals.shape == (802, 105)                       # ROOT adds flow bins on the member axis too
    np.testing.assert_allclose(vals[:, 1:-1].T, expected)


# ---------------------------------------------------------------------------
# Registry and analyzer hooks
# ---------------------------------------------------------------------------

def test_pdf_family_has_no_up_down_labels():
    assert all(src.labels() == () for src in sources_for_family("pdf", ERA))
    assert expected_labels(ERA, ["pdf"], "wr_ee_resolved_sr", is_mc=True) == []


@pytest.fixture
def pdf_era(monkeypatch):
    """Make the pdf family available for ERA with an in-memory sample table."""
    monkeypatch.setitem(systematics_mod.PDF_SETS, ERA, "unused.json")
    table = {"DY": 325300, "TBbarQ": 325500, "WW": None}
    monkeypatch.setattr(analyzer_mod, "sample_lhaid", lambda era, sample: table[sample])
    return WrAnalysis(mass_point=None, enabled_systs=["pdf"], region="both")


def _events(rows):
    return ak.zip({"LHEPdfWeight": ak.Array(rows)}, depth_limit=1)


def test_analyzer_builds_members_for_a_sample_with_weights(pdf_era):
    rows = [[1.0] * 103] * 3
    lhaid, members = pdf_era._pdf_members(_events(rows), {"sample": "DY", "LHEPdfSumw": [1.0] * 103},
                                          ERA, True)
    assert lhaid == "325300" and members.weights().shape == (3, 103)
    assert pdf_era._pdf_members(_events(rows), {"sample": "DY"}, ERA, False) == (None, None)


def test_sample_without_weights_is_none_with_a_warning(pdf_era, caplog):
    events = ak.zip({"genWeight": ak.Array([1.0, 1.0])}, depth_limit=1)
    with caplog.at_level(logging.WARNING, logger="wrcoffea.analyzer"):
        assert pdf_era._pdf_members(events, {"sample": "WW"}, ERA, True) == (NO_PDF, None)
    assert any("has no LHEPdfWeight" in r.getMessage() for r in caplog.records)


def test_set_without_alphas_warns_and_missing_sums_raise(pdf_era, caplog):
    rows = [[1.0] * 101] * 2
    with caplog.at_level(logging.WARNING, logger="wrcoffea.analyzer"):
        lhaid, _ = pdf_era._pdf_members(_events(rows), {"sample": "TBbarQ", "LHEPdfSumw": [1.0] * 101},
                                        ERA, True)
    assert lhaid == "325500"
    assert any("no alpha_s members" in r.getMessage() for r in caplog.records)
    with pytest.raises(ValueError, match="no LHEPdfSumw"):
        pdf_era._pdf_members(_events(rows), {"sample": "TBbarQ"}, ERA, True)
