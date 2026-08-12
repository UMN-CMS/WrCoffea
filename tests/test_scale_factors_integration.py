"""
Tests for scale factor evaluation functions with mocked correctionlib.

Critical functionality:
- Pileup weight evaluation
- Muon SF components (RECO, ID, ISO, trigger)
- Electron SF components (Reco, ID, trigger)
- Jet veto map evaluation (event filtering on the main path)
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
    event indexing ``events[mask]`` (jet_veto_event_mask filters events on
    its main path).

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
    """Test jet veto map evaluation.

    Current contract of ``jet_veto_event_mask(events, ak4_id_mask, era)``:

    - Main path (era configured AND >= 1 preselected jet): returns the
      FILTERED events array (``events[~any_vetoed]``).
    - Early exits (era not configured, or zero preselected jets): returns
      an all-True boolean mask of length ``len(events)`` instead.

    NOTE: this dual return-type contract (filtered events vs. boolean mask)
    is a known, reviewed inconsistency; these tests pin the current behavior.
    """

    def test_jet_veto_event_mask(self):
        """Main path with no vetoed jets returns all events (filtered array)."""
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

        # Main path returns filtered events, not a boolean mask.
        assert isinstance(result, ak.Array)
        assert len(result) == n_events
        assert ak.to_list(result.Jet.pt) == ak.to_list(events.Jet.pt)

    def test_jet_veto_some_vetoed(self):
        """Events containing a vetoed jet are removed from the returned events."""
        # 3 events, each with 1 jet.
        events = _make_events_with_jets(
            jet_pt=[[100.0], [200.0], [150.0]],
            jet_eta=[[2.0], [3.0], [1.0]],
            jet_phi=[[1.5], [-1.0], [0.5]],
        )
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

        # eta=3.0 event is vetoed; eta=2.0 and eta=1.0 events survive.
        assert len(result) == 2
        assert ak.to_list(result.Jet.eta) == [[2.0], [1.0]]
        assert ak.to_list(result.Jet.pt) == [[100.0], [150.0]]

    def test_jet_veto_empty_events(self):
        """Events with no jets should survive the veto (main path is taken
        because another event has a preselected jet)."""
        events = _make_events_with_jets(
            jet_pt=[[], [100.0], []],
            jet_eta=[[], [2.0], []],
            jet_phi=[[], [1.5], []],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.0))
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        # All 3 events survive, including the jet-less ones.
        assert len(result) == 3
        assert ak.to_list(result.Jet.pt) == [[], [100.0], []]

    def test_jet_veto_no_preselected_jets_returns_bool_mask(self):
        """Early exit: when ZERO jets pass preselection, the function returns
        an all-True boolean mask rather than filtered events (known reviewed
        contract inconsistency with the main path)."""
        events = _make_events_with_jets(
            jet_pt=[[5.0], [10.0]],  # all below jet_veto_pt_min = 15
            jet_eta=[[2.0], [1.0]],
            jet_phi=[[1.5], [0.5]],
        )
        ak4_id_mask = _all_true_ak4_id_mask(events)

        ctx, _, _ = _patch_correctionlib(_mock_correction_evaluate_factory(0.0))
        with ctx:
            result = sf.jet_veto_event_mask(events, ak4_id_mask, TEST_ERA)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        np.testing.assert_array_equal(result, np.ones(2, dtype=bool))

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

        # Main path: event survives (its evaluated jet is not vetoed).
        assert len(result) == 1
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
        assert len(result) == 1

    def test_jet_veto_unconfigured_era(self):
        """Early exit: when era is not in JETVETO_JSONS, an all-True boolean
        mask is returned (not filtered events -- known reviewed contract
        inconsistency with the main path)."""
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

        # jet_veto_event_mask: unconfigured era takes the early exit, which
        # returns an all-True boolean mask (NOT filtered events as on the
        # main path -- known reviewed contract inconsistency).
        ak4_id_mask = _all_true_ak4_id_mask(events_with_jets)
        mask = sf.jet_veto_event_mask(events_with_jets, ak4_id_mask, fake_era)
        np.testing.assert_array_equal(mask, np.ones(n_events, dtype=bool))
