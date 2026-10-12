"""
Tests for scale factor evaluation functions with mocked correctionlib.

Critical functionality:
- Pileup weight evaluation
- Muon SF components (RECO, ID, ISO, trigger)
- Electron SF components (Reco, ID, trigger)
- Jet veto map evaluation (per-event boolean masks)
- Pt/eta clipping to valid bin edges
- Era fallback handling

The ``correctionlib`` library is imported lazily (inside each function) via
``import correctionlib``, so we intercept it by patching ``sys.modules``
rather than a module attribute.  The module-level ``_CORRECTIONSET_CACHE``
and ``_WARN_ONCE`` caches are cleared between every test to prevent
cross-test leakage.
"""

import sys

import pytest
import numpy as np
import awkward as ak
from unittest.mock import patch, MagicMock
from coffea.nanoevents.methods import base

import wrcoffea.scale_factors as sf


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

TEST_ERA = "RunIII2024Summer24"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_sf_cache():
    """Clear the correctionlib payload cache and warn-once set between tests."""
    sf._CORRECTIONSET_CACHE.clear()
    sf._WARN_ONCE.clear()
    yield
    sf._CORRECTIONSET_CACHE.clear()
    sf._WARN_ONCE.clear()


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_events(n_events, nTrueInt_values=None):
    """Build a mock *events* object with a ``Pileup.nTrueInt`` attribute.

    Parameters
    ----------
    n_events : int
        Number of events.
    nTrueInt_values : array-like, optional
        Explicit nTrueInt values.  Defaults to linspace(10, 50, n_events).
    """
    if nTrueInt_values is None:
        nTrueInt_values = np.linspace(10, 50, n_events)

    class MockPileup:
        pass

    class MockEvents:
        pass

    ev = MockEvents()
    type(ev).__len__ = lambda self: n_events
    pu = MockPileup()
    pu.nTrueInt = ak.Array(nTrueInt_values)
    ev.Pileup = pu
    return ev


def _make_events_with_jets(jet_pt, jet_eta, jet_phi, jet_chEmEF=None, jet_neEmEF=None):
    """Build an awkward *events* array whose ``Jet`` field is a proper
    jagged record, supporting ``events.Jet``, ``len(events)``, and boolean
    event indexing ``events[mask]``.

    Parameters
    ----------
    jet_pt, jet_eta, jet_phi : list of lists
        Jagged arrays of jet kinematics (one inner list per event).
    jet_chEmEF, jet_neEmEF : list of lists, optional
        Charged/neutral EM energy fractions; default to zeros (which pass
        the jet-veto EM-fraction preselection chEmEF + neEmEF < 0.9).
    """
    pt = ak.Array(jet_pt)
    if jet_chEmEF is None:
        jet_chEmEF = ak.zeros_like(pt)
    if jet_neEmEF is None:
        jet_neEmEF = ak.zeros_like(pt)

    jets = ak.zip({
        "pt": pt,
        "eta": ak.Array(jet_eta),
        "phi": ak.Array(jet_phi),
        "chEmEF": ak.Array(jet_chEmEF),
        "neEmEF": ak.Array(jet_neEmEF),
    })
    return ak.zip({"Jet": jets}, depth_limit=1)


def _all_true_ak4_id_mask(events):
    """Per-jet boolean ID mask, shaped like the analyzer's ``ak4_id_mask``
    (e.g. ``events.Jet.jetId >= 6``) but passing every jet."""
    return events.Jet.pt > -1.0


def _make_tight_leptons(pt_lists, eta_lists, deltaEtaSC_lists=None):
    """Build an awkward-array record with ``.pt``, ``.eta`` (and optional
    ``.deltaEtaSC``) attributes, matching the layout expected by the SF
    functions.

    Parameters
    ----------
    pt_lists, eta_lists : list of lists
        Jagged arrays of lepton kinematics (one inner list per event).
    deltaEtaSC_lists : list of lists, optional
        If given, the result will also carry a ``.deltaEtaSC`` field.
    """
    fields = {"pt": ak.Array(pt_lists), "eta": ak.Array(eta_lists)}
    if deltaEtaSC_lists is not None:
        fields["deltaEtaSC"] = ak.Array(deltaEtaSC_lists)
    return ak.zip(fields)


def _mock_correction_evaluate_factory(value=0.98):
    """Return a callable suitable for ``mock_correction.evaluate`` that
    returns a numpy array whose length matches the first array-like arg.
    """
    def mock_evaluate(*args):
        for a in args:
            if hasattr(a, "__len__") and not isinstance(a, str):
                return np.full(len(a), value, dtype=np.float64)
        return value
    return mock_evaluate


def _make_mock_correctionlib(evaluate_fn=None):
    """Build a fully wired mock ``correctionlib`` module.

    Returns ``(mock_module, mock_correction)`` where ``mock_module`` is
    suitable for insertion into ``sys.modules["correctionlib"]`` and
    ``mock_correction`` exposes the ``.evaluate`` mock for assertion.
    """
    if evaluate_fn is None:
        evaluate_fn = _mock_correction_evaluate_factory(0.98)

    mock_module = MagicMock()
    mock_cset = MagicMock()
    mock_correction = MagicMock()
    mock_correction.evaluate = MagicMock(side_effect=evaluate_fn)
    mock_cset.__getitem__ = MagicMock(return_value=mock_correction)
    mock_module.CorrectionSet.from_file.return_value = mock_cset
    return mock_module, mock_correction


def _patch_correctionlib(evaluate_fn=None):
    """Return a ``patch.dict`` context manager that replaces
    ``sys.modules["correctionlib"]`` with a mock.

    Usage::

        mock_mod, mock_corr = _make_mock_correctionlib(...)
        with patch.dict(sys.modules, {"correctionlib": mock_mod}):
            sf.pileup_weight(events, era)

    This is a convenience wrapper that calls ``_make_mock_correctionlib``
    and returns ``(ctx_manager, mock_module, mock_correction)``.
    """
    mock_module, mock_correction = _make_mock_correctionlib(evaluate_fn)
    ctx = patch.dict(sys.modules, {"correctionlib": mock_module})
    return ctx, mock_module, mock_correction


# ---------------------------------------------------------------------------
# Tests: Pileup weights
# ---------------------------------------------------------------------------

class TestPileupWeights:
    """Test pileup reweighting."""

    def test_pileup_weight_nominal(self):
        """Test pileup weight returns (nominal, up, down) tuple."""
        n_events = 100
        events = _make_events(n_events)

        def evaluate_fn(*args):
            nTrueInt_arr, variation = args
            n = len(nTrueInt_arr)
            if variation == "nominal":
                return np.ones(n, dtype=np.float64)
            elif variation == "up":
                return np.full(n, 1.05, dtype=np.float64)
            elif variation == "down":
                return np.full(n, 0.95, dtype=np.float64)
            return np.ones(n, dtype=np.float64)

        ctx, mock_mod, _ = _patch_correctionlib(evaluate_fn)
        with ctx:
            weights = sf.pileup_weight(events, TEST_ERA)

        assert isinstance(weights, tuple), "Should return tuple of (nominal, up, down)"
        assert len(weights) == 3, "Should have 3 components"

        nominal, up, down = weights
        assert len(nominal) == n_events
        assert len(up) == n_events
        assert len(down) == n_events

        np.testing.assert_allclose(nominal, 1.0)
        np.testing.assert_allclose(up, 1.05)
        np.testing.assert_allclose(down, 0.95)

    def test_pileup_weight_passes_nTrueInt(self):
        """Verify that events.Pileup.nTrueInt is what gets passed to evaluate."""
        n_events = 4
        nTrueInt_values = [20.0, 30.0, 40.0, 50.0]
        events = _make_events(n_events, nTrueInt_values=nTrueInt_values)

        received_args = []

        def evaluate_fn(*args):
            received_args.append(args)
            return np.ones(len(args[0]), dtype=np.float64)

        ctx, mock_mod, _ = _patch_correctionlib(evaluate_fn)
        with ctx:
            sf.pileup_weight(events, TEST_ERA)

        assert len(received_args) == 3
        for args in received_args:
            np.testing.assert_array_almost_equal(args[0], nTrueInt_values)

    def test_pileup_weight_unconfigured_era(self):
        """When the era is not in PILEUP_JSONS, return unity weights."""
        n_events = 10
        events = _make_events(n_events)

        nom, up, down = sf.pileup_weight(events, "FakeEra9999")

        np.testing.assert_array_equal(nom, np.ones(n_events))
        np.testing.assert_array_equal(up, np.ones(n_events))
        np.testing.assert_array_equal(down, np.ones(n_events))


# ---------------------------------------------------------------------------
# Tests: Muon scale factors
# ---------------------------------------------------------------------------

class TestMuonScaleFactors:
    """Test muon scale factor evaluation."""

    def test_muon_sf_returns_dict(self):
        """muon_sf should return a dict with 'reco', 'id', 'iso' keys."""
        n_events = 10
        tight_muons = _make_tight_leptons(
            [[50.0, 40.0]] * n_events,
            [[1.0, -1.0]] * n_events,
        )

        ctx, _, _ = _patch_correctionlib()
        with ctx:
            result = sf.muon_sf(tight_muons, TEST_ERA)

        assert isinstance(result, dict), "muon_sf should return a dict"
        assert set(result.keys()) == {"reco", "id", "iso"}

        for key in ("reco", "id", "iso"):
            component = result[key]
            assert isinstance(component, tuple) and len(component) == 3, \
                f"'{key}' component should be (nom, up, down) tuple"
            nom, up, down = component
            assert len(nom) == n_events, f"'{key}' nom length should match n_events"
            assert len(up) == n_events
            assert len(down) == n_events

    def test_muon_sf_jagged_flattening(self):
        """Varying muon multiplicity: per-event products are computed correctly."""
        # Event 0: 1 muon, Event 1: 2 muons, Event 2: 0 muons, Event 3: 1 muon
        tight_muons = _make_tight_leptons(
            [[50.0], [60.0, 45.0], [], [70.0]],
            [[1.0], [1.2, -1.5], [], [0.5]],
        )

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.98))
        with ctx:
            result = sf.muon_sf(tight_muons, TEST_ERA)

        reco_nom, reco_up, reco_down = result["reco"]
        assert len(reco_nom) == 4, "Should return per-event SF (4 events)"

        # Event with 0 muons should get SF = 1.0.
        assert reco_nom[2] == pytest.approx(1.0), "Empty event should get SF=1.0"

        # Event with 2 muons should get product: 0.98 * 0.98 = 0.9604.
        assert reco_nom[1] == pytest.approx(0.98 * 0.98, rel=1e-5)

    def test_muon_trigger_sf(self):
        """muon_trigger_sf should return (nominal, up, down) tuple."""
        n_events = 30
        tight_muons = _make_tight_leptons(
            [[55.0, 45.0]] * n_events,
            [[1.0, -1.2]] * n_events,
        )

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.95))
        with ctx:
            result = sf.muon_trigger_sf(tight_muons, TEST_ERA)

        assert isinstance(result, tuple) and len(result) == 3
        nom, up, down = result
        assert len(nom) == n_events
        assert len(up) == n_events
        assert len(down) == n_events

    def test_muon_trigger_sf_empty_muons(self):
        """Events with zero muons should get trigger SF = 1.0."""
        tight_muons = _make_tight_leptons(
            [[], [], []],
            [[], [], []],
        )

        ctx, _, _ = _patch_correctionlib()
        with ctx:
            nom, up, down = sf.muon_trigger_sf(tight_muons, TEST_ERA)

        assert len(nom) == 3
        np.testing.assert_array_equal(nom, np.ones(3))

    def test_muon_sf_unconfigured_era(self):
        """When era is not in MUON_JSONS, muon_sf returns dict with unity SFs."""
        tight_muons = _make_tight_leptons([[50.0]], [[1.0]])

        result = sf.muon_sf(tight_muons, "FakeEra9999")

        assert isinstance(result, dict)
        for key in ("reco", "id", "iso"):
            nom, up, down = result[key]
            np.testing.assert_array_equal(nom, np.ones(1))

    def test_muon_trigger_sf_unconfigured_era(self):
        """When era is not in MUON_JSONS, muon_trigger_sf returns unity."""
        tight_muons = _make_tight_leptons([[50.0]], [[1.0]])

        nom, up, down = sf.muon_trigger_sf(tight_muons, "FakeEra9999")

        np.testing.assert_array_equal(nom, np.ones(1))

    @staticmethod
    def _patch_trigger_cset(corrections):
        """Patch correctionlib with a CorrectionSet exposing exactly *corrections*."""
        mock_cset = MagicMock()
        # _has_correction() iterates the set, so provide __iter__ (not keys()).
        mock_cset.__iter__ = MagicMock(side_effect=lambda: iter(list(corrections)))
        mock_cset.__getitem__ = MagicMock(side_effect=lambda k: corrections[k])
        mock_module = MagicMock()
        mock_module.CorrectionSet.from_file.return_value = mock_cset
        return patch.dict(sys.modules, {"correctionlib": mock_module})

    @staticmethod
    def _make_corr(evaluate_fn, first_input="abseta"):
        corr = MagicMock()
        corr.evaluate = MagicMock(side_effect=evaluate_fn)
        inp = MagicMock()
        inp.name = first_input
        corr.inputs = [inp]
        return corr

    def test_muon_trigger_sf_efficiency_payload(self):
        """Payloads with DATAeff/MCeff use the dilepton OR efficiency formula."""
        tight_muons = _make_tight_leptons([[100.0, 80.0], [100.0]], [[0.5, -0.5], [0.5]])
        corrections = {
            "NUM_HLT_DEN_HighPtLooseRelIsoProbes_DATAeff": self._make_corr(
                _mock_correction_evaluate_factory(0.9)),
            "NUM_HLT_DEN_HighPtLooseRelIsoProbes_MCeff": self._make_corr(
                _mock_correction_evaluate_factory(0.8)),
            "NUM_HLT_DEN_HighPtLooseRelIsoProbes": self._make_corr(
                _mock_correction_evaluate_factory(0.5)),  # must NOT be used
        }

        with self._patch_trigger_cset(corrections):
            nom, up, down = sf.muon_trigger_sf(tight_muons, "RunIII2024Summer24")

        # two muons: (1 - 0.1*0.1) / (1 - 0.2*0.2); one muon: 0.9 / 0.8
        assert nom[0] == pytest.approx(0.99 / 0.96, rel=1e-9)
        assert nom[1] == pytest.approx(0.9 / 0.8, rel=1e-9)
        assert not corrections["NUM_HLT_DEN_HighPtLooseRelIsoProbes"].evaluate.called

    def test_muon_trigger_sf_efficiency_payload_variation_is_anticorrelated(self):
        """Up = data efficiency up with MC efficiency down (the Run 2 convention), down the reverse."""
        tight_muons = _make_tight_leptons([[100.0], [100.0, 80.0]], [[0.5], [0.5, -0.5]])

        def eff(nominal, err):
            def evaluate(eta, pt, variation):
                v = {"nominal": nominal, "systup": nominal + err, "systdown": nominal - err}[variation]
                return np.full(len(np.atleast_1d(pt)), v, dtype=np.float64)
            return evaluate

        corrections = {
            "NUM_HLT_DEN_HighPtLooseRelIsoProbes_DATAeff": self._make_corr(eff(0.90, 0.02)),
            "NUM_HLT_DEN_HighPtLooseRelIsoProbes_MCeff": self._make_corr(eff(0.80, 0.02)),
        }
        with self._patch_trigger_cset(corrections):
            nom, up, down = sf.muon_trigger_sf(tight_muons, "RunIII2024Summer24")
        assert nom[0] == pytest.approx(0.90 / 0.80)
        assert up[0] == pytest.approx(0.92 / 0.78) and down[0] == pytest.approx(0.88 / 0.82)
        # two muons: the OR of the shifted per-leg efficiencies
        assert up[1] == pytest.approx((1 - 0.08 ** 2) / (1 - 0.22 ** 2))
        assert down[1] == pytest.approx((1 - 0.12 ** 2) / (1 - 0.18 ** 2))

    def test_muon_trigger_sf_muonz_or_efficiencies(self):
        """2022-2025 (since 2026-10-11): the POG muon_Z entry of our trigger OR carries the data and MC efficiencies
        as scale_factors values, and the per-leg OR formula is used (real payload, no mocks)."""
        import correctionlib
        from wrcoffea.analysis_config import MUON_HLT_CORRECTIONS, MUON_JSONS
        era = "Run3Summer22"
        name = MUON_HLT_CORRECTIONS[era]
        assert name == "NUM_Mu50_or_CascadeMu100_or_HighPtTkMu100_DEN_CutBasedIdGlobalHighPt_and_TkIsoLoose"
        corr = correctionlib.CorrectionSet.from_file(MUON_JSONS[era]["TRIGGER"])[name]
        tight_muons = _make_tight_leptons([[60.0, 55.0], [300.0]], [[0.3, -2.0], [1.0]])
        nom, up, down = sf.muon_trigger_sf(tight_muons, era)

        def eff(val, eta, pt):
            return corr.evaluate(eta, pt, val)
        e_d = 1 - (1 - eff("nominal_DATAeff", 0.3, 60.0)) * (1 - eff("nominal_DATAeff", -2.0, 55.0))
        e_m = 1 - (1 - eff("nominal_MCeff", 0.3, 60.0)) * (1 - eff("nominal_MCeff", -2.0, 55.0))
        assert nom[0] == pytest.approx(e_d / e_m, rel=1e-9)
        assert nom[1] == pytest.approx(eff("nominal_DATAeff", 1.0, 300.0) / eff("nominal_MCeff", 1.0, 300.0), rel=1e-9)
        assert 0.97 < nom[0] < 1.01
        # variations as Run 2: each leg's data efficiency moved by its stat (+) syst error one way while the MC
        # efficiency moves the other way, so the errors add in the ratio
        def shifted(kind, sign, eta, pt):
            err = np.hypot(eff(f"stat_{kind}eff", eta, pt), eff(f"syst_{kind}eff", eta, pt))
            return min(max(eff(f"nominal_{kind}eff", eta, pt) + sign * err, 0.0), 1.0)
        def event(kind, sign):
            return 1 - (1 - shifted(kind, sign, 0.3, 60.0)) * (1 - shifted(kind, sign, -2.0, 55.0))
        assert up[0] == pytest.approx(event("DATA", +1) / event("MC", -1), rel=1e-9)
        assert down[0] == pytest.approx(event("DATA", -1) / event("MC", +1), rel=1e-9)
        assert up[0] > nom[0] > down[0]
        # the same-direction shift would nearly cancel: the Run 2 convention is at least as large as either error
        rel_d = abs(event("DATA", +1) / event("DATA", 0) - 1)
        rel_m = abs(event("MC", -1) / event("MC", 0) - 1)
        assert up[0] / nom[0] - 1 >= max(rel_d, rel_m) * 0.999
        # the leading-muon SF of the same entry would differ: the OR formula is in use
        assert nom[0] != pytest.approx(corr.evaluate(0.3, 60.0, "nominal"), rel=1e-4)

    def test_muon_trigger_sf_leading_muon_fallback(self):
        """Payloads with only a per-lepton SF apply the leading-pT muon's SF once."""
        tight_muons = _make_tight_leptons(
            [[80.0, 100.0], [60.0], []],
            [[0.5, -1.5], [0.5], []],
        )

        def sf_from_pt(eta, pt, variation):
            base = np.asarray(pt, dtype=np.float64) / 100.0
            if variation == "systup":
                return base + 0.01
            if variation == "systdown":
                return base - 0.01
            return base

        corr = self._make_corr(sf_from_pt, first_input="abseta")
        corrections = {"NUM_HLT_DEN_HighPtLooseRelIsoProbes": corr}

        with self._patch_trigger_cset(corrections):
            nom, up, down = sf.muon_trigger_sf(tight_muons, "RunIISummer20UL18")   # muon_HighPt: SF only

        # Leading muon (pt=100) is the second entry in event 0: SF = 1.00,
        # not the product (0.8) and not the first-listed muon (0.8).
        assert nom[0] == pytest.approx(1.00)
        assert nom[1] == pytest.approx(0.60)
        assert nom[2] == pytest.approx(1.0)
        assert up[0] == pytest.approx(1.01)
        assert down[1] == pytest.approx(0.59)

        # abseta payload: eta must be passed as |eta|.
        eta_arg = corr.evaluate.call_args_list[0][0][0]
        assert np.all(np.asarray(eta_arg) >= 0.0)

    def test_muon_trigger_sf_signed_eta_payload(self):
        """POG-style payloads binned in signed eta receive signed eta."""
        tight_muons = _make_tight_leptons([[70.0]], [[-1.5]])
        corr = self._make_corr(_mock_correction_evaluate_factory(0.97), first_input="eta")
        corrections = {"NUM_HLT_DEN_HighPtLooseRelIsoProbes": corr}

        with self._patch_trigger_cset(corrections):
            nom, _, _ = sf.muon_trigger_sf(tight_muons, "RunIISummer20UL18")

        assert nom[0] == pytest.approx(0.97)
        eta_arg = corr.evaluate.call_args_list[0][0][0]
        assert np.asarray(eta_arg)[0] == pytest.approx(-1.5)


# ---------------------------------------------------------------------------
# Tests: Electron scale factors
# ---------------------------------------------------------------------------

class TestElectronScaleFactors:
    """Test electron scale factor evaluation."""

    def test_electron_reco_sf(self):
        """electron_reco_sf should return (nominal, up, down) tuple."""
        n_events = 20
        tight_electrons = _make_tight_leptons(
            [[55.0, 45.0]] * n_events,
            [[1.0, -1.0]] * n_events,
            deltaEtaSC_lists=[[0.01, -0.01]] * n_events,
        )

        ctx, _, _ = _patch_correctionlib()
        with ctx:
            result = sf.electron_reco_sf(tight_electrons, TEST_ERA)

        assert isinstance(result, tuple) and len(result) == 3
        nom, up, down = result
        assert len(nom) == n_events
        assert len(up) == n_events
        assert len(down) == n_events

    def test_electron_trigger_sf(self):
        """electron_trigger_sf should return (nominal, up, down) tuple."""
        n_events = 15
        tight_electrons = _make_tight_leptons(
            [[60.0, 50.0]] * n_events,
            [[1.5, -1.2]] * n_events,
            deltaEtaSC_lists=[[0.01, -0.01]] * n_events,
        )

        # electron_trigger_sf uses two corrections keyed by name:
        #   "Electron-HLT-DataEff" and "Electron-HLT-McEff"
        mock_module = MagicMock()
        mock_cset = MagicMock()

        def make_eff_mock(eff_value):
            m = MagicMock()
            def evaluate_fn(*args):
                # args: (era_key, variation, trigger_path, eta, pt)
                n = len(args[3])
                return np.full(n, eff_value, dtype=np.float64)
            m.evaluate = MagicMock(side_effect=evaluate_fn)
            return m

        data_eff = make_eff_mock(0.90)
        mc_eff = make_eff_mock(0.88)

        def getitem_fn(key):
            if "DataEff" in key:
                return data_eff
            elif "McEff" in key:
                return mc_eff
            return MagicMock()

        mock_cset.__getitem__ = MagicMock(side_effect=getitem_fn)
        mock_module.CorrectionSet.from_file.return_value = mock_cset

        with patch.dict(sys.modules, {"correctionlib": mock_module}):
            result = sf.electron_trigger_sf(tight_electrons, TEST_ERA)

        assert isinstance(result, tuple) and len(result) == 3
        nom, up, down = result
        assert len(nom) == n_events
        assert len(up) == n_events
        assert len(down) == n_events
        # With eff_data=0.90, eff_mc=0.88, SF should be > 1.0 for each event.
        assert np.all(nom > 0.5), "Trigger SF should be reasonable"

    def test_electron_id_sf_json_payload(self):
        """electron_id_sf evaluates HEEPID from the official EGamma JSON.

        The RunIII2024Summer24 electron JSON provides a HEEPID working point;
        up/down variations must bracket the nominal value.
        """
        n_events = 5
        tight_electrons = _make_tight_leptons(
            [[50.0]] * n_events,
            [[0.5]] * n_events,
            deltaEtaSC_lists=[[0.0]] * n_events,
        )

        result = sf.electron_id_sf(tight_electrons, TEST_ERA)

        assert isinstance(result, tuple) and len(result) == 3
        nom, up, down = result
        assert len(nom) == n_events
        # Barrel HEEPID SF from the 2024 EGamma JSON.
        np.testing.assert_allclose(nom, 0.945079, atol=1e-4)
        assert np.all(up > nom)
        assert np.all(down < nom)

    def test_electron_id_sf_barrel_vs_endcap(self):
        """Barrel and endcap electrons should get different HEEP ID SFs."""
        tight_electrons = _make_tight_leptons(
            [[50.0], [50.0]],
            [[0.5], [2.0]],
            deltaEtaSC_lists=[[0.0], [0.0]],
        )

        nom, up, down = sf.electron_id_sf(tight_electrons, TEST_ERA)

        # 2024 EGamma JSON: barrel 0.945079, endcap 0.961394.
        assert nom[0] == pytest.approx(0.945079, abs=1e-4)
        assert nom[1] == pytest.approx(0.961394, abs=1e-4)
        assert nom[0] != nom[1]

    def test_electron_id_sf_flat_fallback_era(self):
        """Eras whose JSON lacks the HEEPID WP fall back to flat UL2018 SFs."""
        tight_electrons = _make_tight_leptons(
            [[50.0], [50.0]],
            [[0.5], [2.0]],
            deltaEtaSC_lists=[[0.0], [0.0]],
        )

        nom, up, down = sf.electron_id_sf(tight_electrons, "RunIISummer20UL18")

        # Flat UL2018 HEEP V7.0 values: barrel 0.973, endcap 0.980.
        assert nom[0] == pytest.approx(0.973, abs=1e-4)
        assert nom[1] == pytest.approx(0.980, abs=1e-4)
        assert np.all(up > nom)
        assert np.all(down < nom)

    def test_electron_id_sf_unconfigured_era(self):
        """When era is not in ELECTRON_JSONS, return unity."""
        tight_electrons = _make_tight_leptons([[50.0]], [[1.0]])

        nom, up, down = sf.electron_id_sf(tight_electrons, "FakeEra9999")

        np.testing.assert_array_equal(nom, np.ones(1))

    def test_electron_reco_sf_unconfigured_era(self):
        """When era is not in ELECTRON_JSONS, electron_reco_sf returns unity."""
        tight_electrons = _make_tight_leptons([[50.0]], [[1.0]])

        nom, up, down = sf.electron_reco_sf(tight_electrons, "FakeEra9999")

        np.testing.assert_array_equal(nom, np.ones(1))

    def test_electron_trigger_sf_unconfigured_era(self):
        """When era is not in ELECTRON_JSONS, electron_trigger_sf returns unity."""
        tight_electrons = _make_tight_leptons([[50.0]], [[1.0]])

        nom, up, down = sf.electron_trigger_sf(tight_electrons, "FakeEra9999")

        np.testing.assert_array_equal(nom, np.ones(1))


# ---------------------------------------------------------------------------
# Tests: Jet veto map
# ---------------------------------------------------------------------------

class TestJetVetoMap:
    """Jet veto evaluation always returns a NumPy boolean mask per event."""

    def test_jet_veto_event_mask(self):
        """All events pass when none of their jets is vetoed."""
        n_events = 20
        events = _make_events_with_jets(
            jet_pt=[[100.0, 80.0]] * n_events,
            jet_eta=[[2.0, -2.2]] * n_events,
            jet_phi=[[1.5, -1.0]] * n_events,
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        # All jets pass (veto value = 0 means not vetoed).
        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.0))
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, np.ones(n_events, dtype=bool))

    def test_jet_veto_some_vetoed(self):
        """The mask filters events without losing their fields or metadata."""
        events = _make_events_with_jets(
            jet_pt=[[100.0], [200.0, 80.0], [150.0]],
            jet_eta=[[2.0], [1.0, 3.0], [1.0]],
            jet_phi=[[1.5], [0.5, -1.0], [0.5]],
        )
        events = ak.with_field(events, ak.Array([101, 102, 103]), "event")
        events = ak.with_name(events, "NanoEvents", behavior=base.behavior)
        events = ak.with_parameter(events, "metadata", {"era": TEST_ERA, "sample": "test"})
        ak4_id_mask = _all_true_ak4_id_mask(events)

        def evaluate_fn(*args):
            # args: ("jetvetomap", eta_array, phi_array)
            eta = args[1]
            n = len(eta)
            result = np.zeros(n, dtype=np.float64)
            result[np.abs(eta) > 2.5] = 1.0
            return result

        ctx, _, _ = _patch_correctionlib(evaluate_fn)
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        # One vetoed jet rejects its whole event, even if another jet passes.
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, [True, False, True])
        filtered = events[result]
        assert filtered.fields == events.fields
        assert filtered.metadata == events.metadata
        assert ak.to_list(filtered.event) == [101, 103]
        assert ak.to_list(filtered.Jet.pt) == [[100.0], [150.0]]
        assert len(events) == 3

    def test_jet_veto_events_without_jets(self):
        """Jet-less events pass even when another event has qualifying jets."""
        events = _make_events_with_jets(
            jet_pt=[[], [100.0], []],
            jet_eta=[[], [2.0], []],
            jet_phi=[[], [1.5], []],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.0))
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        np.testing.assert_array_equal(result, [True, True, True])
        assert ak.to_list(events[result].Jet.pt) == [[], [100.0], []]

    def test_jet_veto_empty_chunk(self):
        """Empty input returns a typed empty mask without loading a payload."""
        events = _make_events_with_jets([[100.0]], [[2.0]], [[1.5]])[:0]
        ctx, mock_module, _ = _patch_correctionlib()
        with ctx:
            result = sf.jet_veto_event_mask(events, _all_true_ak4_id_mask(events), TEST_ERA)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert result.shape == (0,)
        assert events[result].fields == events.fields
        mock_module.CorrectionSet.from_file.assert_not_called()

    def test_jet_veto_all_events_vetoed(self):
        """Rejecting every event still returns a mask aligned with the input."""
        events = _make_events_with_jets([[100.0], [200.0]], [[2.0], [3.0]], [[1.5], [-1.0]])
        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(1.0))
        with ctx:
            result = sf.jet_veto_event_mask(events, _all_true_ak4_id_mask(events), TEST_ERA)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, [False, False])
        assert len(events[result]) == 0
        assert events[result].fields == events.fields

    @pytest.mark.parametrize("jet_pt", [[[5.0], [10.0]], [[], []]])
    def test_jet_veto_no_preselected_jets_returns_bool_mask(self, jet_pt):
        """A chunk with no qualifying jets passes without evaluating the map."""
        events = _make_events_with_jets(
            jet_pt=jet_pt,
            jet_eta=[[2.0] * len(row) for row in jet_pt],
            jet_phi=[[1.5] * len(row) for row in jet_pt],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        ctx, _, mock_correction = _patch_correctionlib(_mock_correction_evaluate_factory(0.0))
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, np.ones(2, dtype=bool))
        assert ak.to_list(events[result].Jet.pt) == jet_pt
        mock_correction.evaluate.assert_not_called()

    def test_jet_veto_low_pt_jets_excluded(self):
        """Jets below the veto map pT threshold should not be evaluated."""
        events = _make_events_with_jets(
            jet_pt=[[100.0, 5.0]],  # 5 GeV < jet_veto_pt_min
            jet_eta=[[2.0, 3.5]],   # Second would be vetoed if evaluated
            jet_phi=[[1.5, -1.0]],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        calls = []
        def evaluate_fn(*args):
            eta = args[1]
            calls.append(len(eta))
            return np.zeros(len(eta), dtype=np.float64)

        ctx, _, _ = _patch_correctionlib(evaluate_fn)
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        np.testing.assert_array_equal(result, [True])
        # Only 1 jet should have been evaluated (the one above threshold).
        assert calls[0] == 1

    def test_jet_veto_ak4_id_mask_respected(self):
        """Jets failing the per-jet ID mask are excluded from evaluation, so
        a would-be-vetoed jet that fails ID cannot veto its event."""
        events = _make_events_with_jets(
            jet_pt=[[100.0, 90.0]],
            jet_eta=[[2.0, 3.0]],  # second jet would be vetoed if evaluated
            jet_phi=[[1.5, -1.0]],
        )
        # Only the first jet passes the ID mask.
        ak4_id_mask = ak.Array([[True, False]])

        def evaluate_fn(*args):
            eta = args[1]
            result = np.zeros(len(eta), dtype=np.float64)
            result[np.abs(eta) > 2.5] = 1.0
            return result

        ctx, _, _ = _patch_correctionlib(evaluate_fn)
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        # The event survives because the bad-eta jet failed ID preselection.
        np.testing.assert_array_equal(result, [True])

    def test_jet_veto_unconfigured_era(self):
        """An unconfigured era returns an all-True mask."""
        events = _make_events_with_jets(
            jet_pt=[[100.0]],
            jet_eta=[[2.0]],
            jet_phi=[[1.5]],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        result = sf.jet_veto_event_mask(events, ak4_id_mask, "FakeEra9999")

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, np.ones(1, dtype=bool))

    def test_jet_veto_jet_mask_drops_only_vetoed_jets(self):
        """Run 2 rule: vetoed jets are dropped and every event is kept, while
        the Run 3 rule on the same events rejects the event instead."""
        events = _make_events_with_jets(
            jet_pt=[[100.0], [200.0, 80.0], [150.0, 10.0]],
            jet_eta=[[2.0], [1.0, 3.0], [1.0, 3.0]],
            jet_phi=[[1.5], [0.5, -1.0], [0.5, -1.0]],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        def evaluate_fn(*args):
            eta = args[1]
            result = np.zeros(len(eta), dtype=np.float64)
            result[np.abs(eta) > 2.5] = 1.0
            return result

        ctx, _, _ = _patch_correctionlib(evaluate_fn)
        with ctx:
            keep = sf.jet_veto_jet_mask(events, ak4_id_mask, TEST_ERA)
            event_mask = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        # The 10 GeV jet at eta 3 fails the minimal selection, so it is not vetoed.
        assert ak.to_list(keep) == [[True], [True, False], [True, True]]
        np.testing.assert_array_equal(event_mask, [True, False, True])

    def test_jet_veto_jet_mask_spares_jets_near_pf_muons(self):
        """A jet with a PF muon within dR < 0.2 is not vetoed; a non-PF muon does not count."""
        events = _make_events_with_jets(
            jet_pt=[[100.0, 90.0], [100.0]],
            jet_eta=[[3.0, 3.2], [3.0]],
            jet_phi=[[1.0, -1.0], [1.0]],
        )
        muons = ak.zip({
            "eta": ak.Array([[3.05], [3.05]]),
            "phi": ak.Array([[1.05], [1.05]]),
            "isPFcand": ak.Array([[True], [False]]),
        })
        events = ak.with_field(events, muons, "Muon")
        ak4_id_mask = _all_true_ak4_id_mask(events)

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(1.0))
        with ctx:
            keep = sf.jet_veto_jet_mask(events, ak4_id_mask, TEST_ERA)

        assert ak.to_list(keep) == [[True, False], [False]]

    def test_jet_veto_jet_mask_unconfigured_era_keeps_all_jets(self):
        """Without a map every jet is kept, including in jet-less events."""
        events = _make_events_with_jets(
            jet_pt=[[100.0, 50.0], []],
            jet_eta=[[3.0, 1.0], []],
            jet_phi=[[1.0, 0.0], []],
        )
        keep = sf.jet_veto_jet_mask(events, _all_true_ak4_id_mask(events), "FakeEra9999")
        assert ak.to_list(keep) == [[True, True], []]

    def test_jet_veto_mode(self, monkeypatch):
        """Unlisted eras use the Run 3 event rule; an unknown mode is an error."""
        monkeypatch.setattr(sf, "JETVETO_MODES", {"RunIISummer20UL18": "jet", "BadEra": "jets"})
        assert sf.jet_veto_mode("RunIISummer20UL18") == "jet"
        assert sf.jet_veto_mode(TEST_ERA) == "event"
        with pytest.raises(ValueError, match="jetveto_mode"):
            sf.jet_veto_mode("BadEra")


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases in SF evaluation."""

    def test_empty_muon_collection(self):
        """muon_sf with all-empty events should return SF = 1.0 everywhere."""
        tight_muons = _make_tight_leptons(
            [[], [], []],
            [[], [], []],
        )

        ctx, _, _ = _patch_correctionlib()
        with ctx:
            result = sf.muon_sf(tight_muons, TEST_ERA)

        for key in ("reco", "id", "iso"):
            nom, up, down = result[key]
            assert len(nom) == 3
            np.testing.assert_array_equal(nom, np.ones(3))

    def test_single_muon_sf(self):
        """Single muon per event: per-event SF equals per-muon SF."""
        tight_muons = _make_tight_leptons(
            [[55.0]],
            [[1.2]],
        )

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.97))
        with ctx:
            result = sf.muon_sf(tight_muons, TEST_ERA)

        nom, _, _ = result["reco"]
        assert len(nom) == 1
        assert nom[0] == pytest.approx(0.97, rel=1e-4)

    def test_electron_reco_sf_missing_deltaEtaSC(self):
        """When deltaEtaSC is missing, electron_reco_sf should still work
        (falls back to deltaEtaSC=0).
        """
        n_events = 5
        tight_electrons = _make_tight_leptons(
            [[50.0]] * n_events,
            [[1.0]] * n_events,
        )

        ctx, _, _ = _patch_correctionlib()
        with ctx:
            result = sf.electron_reco_sf(tight_electrons, TEST_ERA)

        assert isinstance(result, tuple) and len(result) == 3
        nom, up, down = result
        assert len(nom) == n_events

    def test_electron_id_sf_missing_deltaEtaSC(self):
        """electron_id_sf without deltaEtaSC should fall back gracefully."""
        tight_electrons = _make_tight_leptons(
            [[50.0], [50.0]],
            [[0.5], [2.0]],
        )

        result = sf.electron_id_sf(tight_electrons, TEST_ERA)

        assert isinstance(result, tuple) and len(result) == 3
        nom, _, _ = result
        assert len(nom) == 2

    def test_cache_is_used_across_calls(self):
        """After the first call loads a CorrectionSet, the second call should
        reuse the cache and NOT call ``from_file`` again.
        """
        n_events = 5
        tight_muons = _make_tight_leptons(
            [[50.0]] * n_events,
            [[1.0]] * n_events,
        )

        mock_module, _ = _make_mock_correctionlib()
        with patch.dict(sys.modules, {"correctionlib": mock_module}):
            sf.muon_sf(tight_muons, TEST_ERA)
            sf.muon_sf(tight_muons, TEST_ERA)

        # from_file should only be called once (first call caches).
        assert mock_module.CorrectionSet.from_file.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Era fallback
# ---------------------------------------------------------------------------

class TestEraFallback:
    """Test graceful handling of unconfigured eras."""

    def test_all_sf_functions_return_unity_for_unknown_era(self):
        """Every SF function should return unity when the era is not configured."""
        n_events = 3
        events = _make_events(n_events)
        tight_leptons = _make_tight_leptons(
            [[50.0]] * n_events,
            [[1.0]] * n_events,
        )
        events_with_jets = _make_events_with_jets(
            jet_pt=[[100.0]] * n_events,
            jet_eta=[[2.0]] * n_events,
            jet_phi=[[1.5]] * n_events,
        )

        fake_era = "FakeEra9999"

        # pileup_weight
        nom, up, down = sf.pileup_weight(events, fake_era)
        np.testing.assert_array_equal(nom, np.ones(n_events))

        # muon_sf
        result = sf.muon_sf(tight_leptons, fake_era)
        for key in ("reco", "id", "iso"):
            np.testing.assert_array_equal(result[key][0], np.ones(n_events))

        # muon_trigger_sf
        nom, _, _ = sf.muon_trigger_sf(tight_leptons, fake_era)
        np.testing.assert_array_equal(nom, np.ones(n_events))

        # electron_id_sf
        nom, _, _ = sf.electron_id_sf(tight_leptons, fake_era)
        np.testing.assert_array_equal(nom, np.ones(n_events))

        # electron_reco_sf
        nom, _, _ = sf.electron_reco_sf(tight_leptons, fake_era)
        np.testing.assert_array_equal(nom, np.ones(n_events))

        # electron_trigger_sf
        nom, _, _ = sf.electron_trigger_sf(tight_leptons, fake_era)
        np.testing.assert_array_equal(nom, np.ones(n_events))

        # jet_veto_event_mask: an unconfigured era passes all events.
        ak4_id_mask = _all_true_ak4_id_mask(events_with_jets)
        mask = sf.jet_veto_event_mask(events_with_jets, ak4_id_mask, fake_era)
        np.testing.assert_array_equal(mask, np.ones(n_events, dtype=bool))


# ---------------------------------------------------------------------------
# Tests: CorrectionSet membership helper
# ---------------------------------------------------------------------------

class TestHasCorrection:
    """_has_correction must not rely on the Mapping ``in`` operator, which
    correctionlib breaks by raising IndexError from __getitem__."""

    @pytest.fixture
    def real_cset(self):
        correctionlib = pytest.importorskip("correctionlib")
        from wrcoffea.analysis_config import PILEUP_JSONS, PILEUP_CORRECTION_NAMES
        era = "Run3Summer22"
        return (
            correctionlib.CorrectionSet.from_file(PILEUP_JSONS[era]),
            PILEUP_CORRECTION_NAMES[era],
        )

    def test_in_operator_raises_on_real_correctionset(self, real_cset):
        cset, _ = real_cset
        with pytest.raises(IndexError):
            "definitely_not_a_correction" in cset
        with pytest.raises(IndexError):
            "definitely_not_a_correction" in cset.keys()

    def test_has_correction_present_and_absent(self, real_cset):
        cset, present = real_cset
        assert sf._has_correction(cset, present) is True
        assert sf._has_correction(cset, "definitely_not_a_correction") is False

    def test_has_correction_with_mock_correctionset(self):
        mock_cset = MagicMock()
        mock_cset.__iter__ = MagicMock(side_effect=lambda: iter(["a", "b"]))
        assert sf._has_correction(mock_cset, "a")
        assert not sf._has_correction(mock_cset, "c")


# ---------------------------------------------------------------------------
# Tests: jet correction shape variations are only computed on request
# ---------------------------------------------------------------------------

def _make_jec_events(jet_counts=(2, 1, 3, 2)):
    """Minimal jagged MC NanoAOD-like events for apply_jet_corrections (no SubJet)."""
    rng = np.random.default_rng(1)
    n_events = len(jet_counts)

    def jagged(low, high):
        return ak.Array([rng.uniform(low, high, n).tolist() for n in jet_counts])

    def coll(idx_name):
        return ak.zip({
            "pt": jagged(60, 400),
            "eta": jagged(-2.0, 2.0),
            "phi": jagged(-3.0, 3.0),
            "mass": jagged(5, 40),
            "area": ak.Array([[0.5] * n for n in jet_counts]),
            "rawFactor": ak.Array([[0.05] * n for n in jet_counts]),
            # Match jet i to gen jet i; the gen collections below have 3 entries.
            idx_name: ak.Array([list(range(n)) for n in jet_counts]),
        })

    jets, fatjets = coll("genJetIdx"), coll("genJetAK8Idx")

    def gen_of(c):
        # Gen jet i sits on jet i (dR = 0) with a pT within the JER resolution.
        return ak.zip({"pt": c.pt * 0.97, "eta": c.eta, "phi": c.phi})

    return ak.zip({
        "Jet": jets,
        "FatJet": fatjets,
        "GenJet": gen_of(jets),
        "GenJetAK8": gen_of(fatjets),
        "Rho": ak.zip({"fixedGridRhoFastjetAll": ak.Array([20.0] * n_events)}),
        "event": ak.Array(list(range(n_events))),
        "luminosityBlock": ak.Array([1] * n_events),
        # The 2024 data residual correction is binned in run number.
        "run": ak.Array([380000] * n_events),
    }, depth_limit=1)


class TestJetVariationGating:
    """apply_jet_corrections attaches only the requested variation fields."""

    ERA = "RunIII2024Summer24"

    @pytest.fixture(autouse=True)
    def _real_correctionlib(self):
        pytest.importorskip("correctionlib")

    @staticmethod
    def _var_fields(coll):
        return {f for f in coll.fields if f.startswith(("pt_", "mass_"))}

    def test_no_variations_by_default(self):
        out = sf.apply_jet_corrections(_make_jec_events(), self.ERA, True)
        assert self._var_fields(out.Jet) == set()
        assert self._var_fields(out.FatJet) == set()

    def test_jer_only(self):
        out = sf.apply_jet_corrections(_make_jec_events(), self.ERA, True, variations={"jer"})
        expected = {"pt_jer_up", "pt_jer_down", "mass_jer_up", "mass_jer_down"}
        assert self._var_fields(out.Jet) == expected
        assert self._var_fields(out.FatJet) == expected

    def test_jes_only(self):
        out = sf.apply_jet_corrections(_make_jec_events(), self.ERA, True, variations={"jes"})
        expected = {"pt_CMS_scale_j_Total_up", "pt_CMS_scale_j_Total_down",
                    "mass_CMS_scale_j_Total_up", "mass_CMS_scale_j_Total_down"}
        assert self._var_fields(out.Jet) == expected
        assert self._var_fields(out.FatJet) == expected
        # JES up must move pt above nominal and down below it.
        assert ak.all(out.Jet.pt_CMS_scale_j_Total_up > out.Jet.pt)
        assert ak.all(out.Jet.pt_CMS_scale_j_Total_down < out.Jet.pt)

    def test_data_never_gets_variations(self):
        out = sf.apply_jet_corrections(_make_jec_events(), self.ERA, False, variations={"jer", "jes"})
        assert self._var_fields(out.Jet) == set()


# ---------------------------------------------------------------------------
# Tests: AK8 soft-drop mass from the uncorrected subjets (ak8_msoftdrop_mode)
# ---------------------------------------------------------------------------

def _make_subjet_events():
    """One event with three AK8 jets: two subjets, one subjet, none."""
    events = _make_jec_events(jet_counts=(3,))
    fatjets = ak.with_field(events.FatJet, ak.Array([[0, 2, -1]]), "subJetIdx1")
    fatjets = ak.with_field(fatjets, ak.Array([[1, -1, -1]]), "subJetIdx2")
    fatjets = ak.with_field(fatjets, ak.Array([[55.0, 66.0, 77.0]]), "msoftdrop")
    subjets = ak.zip({
        "pt": ak.Array([[120.0, 80.0, 60.0]]),
        "eta": ak.Array([[0.3, 0.6, -1.0]]),
        "phi": ak.Array([[0.1, 0.5, 2.0]]),
        "mass": ak.Array([[10.0, 8.0, 12.0]]),
        "rawFactor": ak.Array([[0.1, 0.2, 0.25]]),
    })
    events = ak.with_field(events, fatjets, "FatJet")
    return ak.with_field(events, subjets, "SubJet")


class TestRawSoftdropMass:
    """ak8_msoftdrop_mode "raw" cuts on the uncorrected-subjet mass, as the Run 2 analysis did."""

    ERA = "RunIII2024Summer24"

    @pytest.fixture(autouse=True)
    def _real_correctionlib(self):
        pytest.importorskip("correctionlib")

    def test_mode_lookup(self, monkeypatch):
        """Unlisted eras cut on the corrected mass; an unknown mode is an error."""
        monkeypatch.setattr(sf, "AK8_MSOFTDROP_MODES", {"RunIISummer20UL18": "raw", "BadEra": "uncorrected"})
        assert sf.ak8_msoftdrop_mode("RunIISummer20UL18") == "raw"
        assert sf.ak8_msoftdrop_mode(TEST_ERA) == "corrected"
        with pytest.raises(ValueError, match="ak8_msoftdrop_mode"):
            sf.ak8_msoftdrop_mode("BadEra")

    def test_raw_mass_sums_uncorrected_subjets(self):
        vector = pytest.importorskip("vector")
        events = _make_subjet_events()
        two = (vector.obj(pt=108.0, eta=0.3, phi=0.1, mass=9.0)
               + vector.obj(pt=64.0, eta=0.6, phi=0.5, mass=6.4)).mass
        got = ak.to_list(sf._raw_softdrop_mass(events.FatJet, events.SubJet))
        assert got[0] == pytest.approx([two, 9.0, 0.0], rel=1e-5)

    def test_corrected_mode_keeps_nanoaod_msoftdrop(self, monkeypatch):
        monkeypatch.setattr(sf, "AK8_MSOFTDROP_MODES", {})
        out = sf.apply_jet_corrections(_make_subjet_events(), self.ERA, False)
        assert ak.to_list(out.FatJet.msoftdrop) == [[55.0, 66.0, 77.0]]

    def test_raw_mode_replaces_msoftdrop(self, monkeypatch):
        monkeypatch.setattr(sf, "AK8_MSOFTDROP_MODES", {self.ERA: "raw"})
        events = _make_subjet_events()
        expected = ak.to_list(sf._raw_softdrop_mass(events.FatJet, events.SubJet))
        out = sf.apply_jet_corrections(events, self.ERA, False)
        assert ak.to_list(out.FatJet.msoftdrop)[0] == pytest.approx(expected[0])

    def test_raw_mode_needs_subjets(self, monkeypatch):
        monkeypatch.setattr(sf, "AK8_MSOFTDROP_MODES", {self.ERA: "raw"})
        with pytest.raises(RuntimeError, match="rawFactor"):
            sf.apply_jet_corrections(_make_jec_events(), self.ERA, False)


# ---------------------------------------------------------------------------
# Tests: JER scale-factor payload schemas
# ---------------------------------------------------------------------------

class TestJerScaleFactorSchemas:
    """_jer_scale_factor must handle both the 'systematic'-input schema and the
    Summer24 JRV2 schema (nominal + separate SFUncertainty correction)."""

    @pytest.fixture(autouse=True)
    def _real_correctionlib(self):
        pytest.importorskip("correctionlib")

    @staticmethod
    def _eval(era):
        import correctionlib
        from wrcoffea.analysis_config import JERC_JSONS
        cfg = JERC_JSONS[era]["JERC_AK4"]
        cs = correctionlib.CorrectionSet.from_file(cfg["json"])
        eta, pt = ak.Array([[0.5, 2.0]]), ak.Array([[100.0, 500.0]])
        return {v: ak.to_numpy(ak.flatten(sf._jer_scale_factor(cs, cfg["JR_tag_MC"], cfg["algo"], eta, pt, v)))
                for v in ("nom", "up", "down")}

    @pytest.mark.parametrize("era", ["RunIII2024Summer24", "Run3Summer23BPix", "RunIISummer20UL18"])
    def test_variations_bracket_nominal(self, era):
        vals = self._eval(era)
        assert np.all(vals["up"] > vals["nom"]), era
        assert np.all(vals["down"] < vals["nom"]), era
        assert np.all(vals["nom"] > 0.9), era

    def test_jrv2_uses_sfuncertainty_symmetrically(self):
        vals = self._eval("RunIII2024Summer24")
        np.testing.assert_allclose(vals["up"] - vals["nom"], vals["nom"] - vals["down"], rtol=1e-9)


# ---------------------------------------------------------------------------
# Tests: jet corrections without a rho branch
# ---------------------------------------------------------------------------

class TestJetCorrectionsWithoutRho:
    """Inputs without rho (pre-fix UL18 skims) must never yield silent null JER templates."""

    ERA = "RunIII2024Summer24"

    @pytest.fixture(autouse=True)
    def _real_correctionlib(self):
        pytest.importorskip("correctionlib")

    @staticmethod
    def _events_without_rho():
        ev = _make_jec_events()
        return ak.zip({f: ev[f] for f in ev.fields if f != "Rho"}, depth_limit=1)

    def test_nominal_run_warns_and_leaves_kinematics_unchanged(self, caplog):
        import logging
        ev = self._events_without_rho()
        with caplog.at_level(logging.WARNING, logger="wrcoffea.scale_factors"):
            out = sf.apply_jet_corrections(ev, self.ERA, True)
        assert "no rho branch" in caplog.text
        assert ak.all(out.Jet.pt == ev.Jet.pt)

    def test_jer_request_is_refused(self):
        with pytest.raises(RuntimeError, match="no rho"):
            sf.apply_jet_corrections(self._events_without_rho(), self.ERA, True, variations={"jer"})

    def test_jes_alone_still_runs(self):
        out = sf.apply_jet_corrections(self._events_without_rho(), self.ERA, True, variations={"jes"})
        assert "pt_CMS_scale_j_Total_up" in out.Jet.fields


@pytest.mark.parametrize("era,barrel,endcap", [
    ("RunIISummer20UL16APV", 0.985, 0.990),
    ("RunIISummer20UL16", 0.985, 0.990),
    ("RunIISummer20UL17", 0.979, 0.987),
    ("RunIISummer20UL18", 0.973, 0.980),
    ("Run3Summer22", 1.007, 1.019),
    ("Run3Summer22EE", 0.997, 0.997),
    ("Run3Summer23", 1.006, 0.988),
    ("Run3Summer23BPix", 1.009, 0.988),
])
def test_heep_flat_sf_per_year(era, barrel, endcap):
    """The Run 2 UL and the 2022/2023 electron payloads have no HEEPID working point: each era takes its own flat
    HEEP scale factors (EGM twikis EgammaUL2016To2018 and EgammSFandSSRun3), barrel and endcap."""
    tight_electrons = _make_tight_leptons([[100.0], [100.0]], [[0.5], [2.0]], deltaEtaSC_lists=[[0.0], [0.0]])
    nom, up, down = sf.electron_id_sf(tight_electrons, era)      # one electron per event: barrel, then endcap
    assert nom[0] == pytest.approx(barrel, abs=1e-4)
    assert nom[1] == pytest.approx(endcap, abs=1e-4)
    assert np.all(up > nom) and np.all(down < nom)


# ---------------------------------------------------------------------------
# Tests: every era's lepton SF payloads evaluate (both eta signs)
# ---------------------------------------------------------------------------

from wrcoffea.analysis_config import ELECTRON_JSONS as _ELECTRON_JSONS, MUON_JSONS as _MUON_JSONS


@pytest.mark.parametrize("era", sorted(_MUON_JSONS))
def test_muon_sfs_evaluate_in_every_era(era):
    """The 2025 muon_HighPt payload is binned in |eta|, the others in signed eta: both must evaluate."""
    pytest.importorskip("correctionlib")
    muons = _make_tight_leptons([[300.0, 150.0], [800.0], []], [[-1.2, 0.8], [-2.1], []])
    for out in (sf.muon_sf(muons, era), sf.muon_sf_loose(muons, era)):
        for nom, up, down in out.values():
            assert len(nom) == 3 and np.all(np.isfinite(nom)) and np.all(nom > 0), era
    nom, up, down = sf.muon_trigger_sf(muons, era)
    assert np.all(np.isfinite(nom)) and np.all(nom > 0), era


@pytest.mark.parametrize("era", sorted(e for e, payloads in _ELECTRON_JSONS.items() if "TRIGGER" in payloads))
def test_electron_trigger_sf_evaluates_in_every_era(era):
    """Every era's trigger payload knows the era key it is evaluated with (2025 borrows the 2024 file)."""
    pytest.importorskip("correctionlib")
    electrons = _make_tight_leptons([[300.0, 150.0], [800.0], []], [[-1.2, 0.8], [-2.1], []],
                                    [[0.01, -0.01], [0.0], []])
    nom, up, down = sf.electron_trigger_sf(electrons, era)
    assert len(nom) == 3 and np.all(np.isfinite(nom)) and np.all(nom > 0), era
    assert nom[2] == 1.0


# ---------------------------------------------------------------------------
# Tests: per-object random numbers (electron smearing, stochastic JER)
# ---------------------------------------------------------------------------

def _events_with_objects(phis_per_event):
    n = len(phis_per_event)
    return ak.zip({
        "event": ak.Array(np.arange(1, n + 1, dtype=np.int64) * 7919),
        "luminosityBlock": ak.Array(np.full(n, 42, dtype=np.int64)),
    }), ak.zip({"phi": ak.Array(phis_per_event)})


class TestObjectNormalDraws:
    """The draws of two objects of one event must be independent (they were a hash of the event number
    and the float bits of phi, which gave objects with |phi| in the same power-of-two range nearly
    the same draw)."""

    def test_two_objects_in_one_event_are_uncorrelated(self):
        pytest.importorskip("correctionlib")
        rng = np.random.default_rng(3)
        n = 20000
        # |phi| in [1, 2) for both objects: the worst case of the old hash.
        phi1 = rng.uniform(1.0, 2.0, n) * rng.choice([-1, 1], n)
        phi2 = rng.uniform(1.0, 2.0, n) * rng.choice([-1, 1], n)
        events, objs = _events_with_objects(np.stack([phi1, phi2], axis=1).tolist())
        z = sf.object_normal_draws(events, objs, salt=11).reshape(n, 2)
        assert abs(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) < 0.05
        assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05

    def test_deterministic_and_salted(self):
        pytest.importorskip("correctionlib")
        events, objs = _events_with_objects([[0.3, -2.9], [1.1]])
        a = sf.object_normal_draws(events, objs, salt=11)
        np.testing.assert_array_equal(a, sf.object_normal_draws(events, objs, salt=11))
        assert not np.allclose(a, sf.object_normal_draws(events, objs, salt=12))


# ---------------------------------------------------------------------------
# Tests: JER hybrid smearing follows the JME matching criteria
# ---------------------------------------------------------------------------

class TestJerMatching:
    ERA = "RunIII2024Summer24"

    @pytest.fixture(autouse=True)
    def _real_correctionlib(self):
        pytest.importorskip("correctionlib")

    def _ceval(self):
        import correctionlib
        from wrcoffea.analysis_config import JERC_JSONS
        cfg = JERC_JSONS[self.ERA]["JERC_AK4"]
        return correctionlib.CorrectionSet.from_file(cfg["json"]), cfg["JR_tag_MC"], cfg["algo"]

    def _smear(self, pt, genpt, matched, phi=(0.7,)):
        ceval, tag, algo = self._ceval()
        n = len(pt)
        jets = ak.Array([[{"eta": 0.5, "phi": phi[i % len(phi)]}] for i in range(n)])
        events = ak.zip({"event": ak.Array(np.arange(1, n + 1, dtype=np.int64)),
                         "luminosityBlock": ak.Array(np.ones(n, dtype=np.int64)),
                         "Rho": ak.zip({"fixedGridRhoFastjetAll": ak.Array(np.full(n, 20.0))})}, depth_limit=1)
        wrap = lambda v: ak.Array([[x] for x in v])
        out, _ = sf._get_smeared_kinematics(jets, wrap(pt), wrap(pt), wrap(genpt), wrap(matched),
                                            events, ceval, tag, algo, sys_var="nom")
        return np.asarray(ak.flatten(out))

    def test_far_gen_pt_is_smeared_stochastically(self):
        """A jet whose gen pT is far outside 3 sigma_JER is not scaled towards it (it was: 50 -> negative)."""
        matched_far = self._smear([50.0], [500.0], [True])
        unmatched = self._smear([50.0], [0.0], [False])
        np.testing.assert_allclose(matched_far, unmatched)
        assert matched_far[0] > 0

    def test_close_gen_pt_uses_the_scaling_method(self):
        out = self._smear([100.0], [98.0], [True])
        ceval, tag, algo = self._ceval()
        sf_nom = float(np.asarray(ak.flatten(sf._jer_scale_factor(ceval, tag, algo, ak.Array([[0.5]]),
                                                                   ak.Array([[100.0]]), "nom")))[0])
        np.testing.assert_allclose(out[0], 100.0 * (1 + (sf_nom - 1) * 2.0 / 100.0), rtol=1e-9)

    def test_gen_match_needs_delta_r_below_half_the_cone(self):
        jets = ak.Array([[{"eta": 0.5, "phi": 0.7}, {"eta": 0.5, "phi": 0.7}]])
        gen = ak.Array([[{"pt": 90.0, "eta": 0.55, "phi": 0.72}, {"pt": 80.0, "eta": 0.5, "phi": 1.0}]])
        genpt, matched = sf._gen_jet_match(jets, ak.Array([[0, 1]]), gen, max_dr=0.2)
        assert ak.to_list(matched) == [[True, False]]
        assert ak.to_list(genpt) == [[90.0, 0.0]]

    def test_gen_match_with_no_gen_jets(self):
        jets = ak.Array([[{"eta": 0.5, "phi": 0.7}], []])
        gen = ak.Array([[], []], with_name=None)
        gen = ak.zip({"pt": ak.Array([[], []]), "eta": ak.Array([[], []]), "phi": ak.Array([[], []])})
        genpt, matched = sf._gen_jet_match(jets, ak.Array([[0], []]), gen, max_dr=0.2)
        assert ak.to_list(matched) == [[False], []]
