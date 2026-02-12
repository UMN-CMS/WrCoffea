"""
Tests for event selection logic in WrAnalysis.resolved_selections()

Critical functionality:
- Flavor counting (ee, mumu, emu)
- Leading/subleading lepton pT cuts
- Jet multiplicity cuts
- mll and mlljj invariant mass windows
- Delta-R separation requirements
- Lepton/jet mask propagation
- Trigger propagation
- Edge cases (empty events, single lepton, no jets)
"""

import pytest
import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
from coffea.analysis_tools import PackedSelection

from wrcoffea.analyzer import WrAnalysis
from wrcoffea.analysis_config import CUTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use vector.behavior which provides PtEtaPhiMLorentzVector with Lorentz
# vector arithmetic (addition, deltaR, .mass, etc.).  The analyzer's
# ``_to_candidate()`` creates arrays with ``with_name="PtEtaPhiMCandidate"``
# but in unit tests we use ``PtEtaPhiMLorentzVector`` directly -- the
# resolved_selections method only needs the Lorentz-vector operations and the
# extra ``.flavor`` / ``.charge`` fields, not full NanoAOD Candidate behavior.
_BEHAVIOR = dict(vector.behavior)


def _make_leptons(pts, etas, phis, masses, charges, flavors):
    """Build a jagged PtEtaPhiMCandidate array with flavor and charge fields.

    Each argument is a list-of-lists (one inner list per event).
    ``flavors`` entries must be ``"electron"`` or ``"muon"``.
    """
    return ak.zip(
        {
            "pt": pts,
            "eta": etas,
            "phi": phis,
            "mass": masses,
            "charge": charges,
            "flavor": flavors,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=_BEHAVIOR,
    )


def _make_jets(pts, etas, phis, masses):
    """Build a jagged PtEtaPhiMCandidate array for AK4 jets."""
    return ak.zip(
        {"pt": pts, "eta": etas, "phi": phis, "mass": masses},
        with_name="PtEtaPhiMLorentzVector",
        behavior=_BEHAVIOR,
    )


def _ee_masks(n_events=1, n_ele=2, n_mu=0, n_jets=2,
              ele_pteta=True, mu_pteta=True, ele_id=True, mu_id=True,
              ak4_pteta=True, ak4_id=True):
    """Build lepton_masks and jet_masks dicts for simple test scenarios.

    Each mask is a jagged boolean array whose inner lists have the specified
    lengths, filled with a constant value.
    """
    lepton_masks = {
        "ele_pteta": ak.Array([[ele_pteta] * n_ele] * n_events),
        "mu_pteta":  ak.Array([[mu_pteta] * n_mu] * n_events),
        "ele_id":    ak.Array([[ele_id] * n_ele] * n_events),
        "mu_id":     ak.Array([[mu_id] * n_mu] * n_events),
    }
    jet_masks = {
        "ak4_pteta": ak.Array([[ak4_pteta] * n_jets] * n_events),
        "ak4_id":    ak.Array([[ak4_id] * n_jets] * n_events),
    }
    return lepton_masks, jet_masks


def _default_triggers(n_events=1, e=True, mu=True, emu=True):
    """Build a (e_trig, mu_trig, emu_trig) tuple of per-event booleans."""
    return (
        ak.Array([e] * n_events),
        ak.Array([mu] * n_events),
        ak.Array([emu] * n_events),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    """Create a WrAnalysis instance suitable for unit tests."""
    return WrAnalysis(mass_point=None, enabled_systs=[], region="both")


# ---------------------------------------------------------------------------
# Resolved selections
# ---------------------------------------------------------------------------

class TestResolvedSelections:
    """Test resolved region selection logic."""

    # ----- mll window tests ------------------------------------------------

    def test_sr_mll_threshold(self, analyzer):
        """Test SR requires mll > 200 GeV (DY CR: 60-150 GeV).

        We vary the lepton momenta so that the invariant mass ``(l1+l2).mass``
        falls in different windows while keeping everything else passing.
        """
        # Use widely-separated etas/phis so delta-R is always large.
        # We control mll by varying lepton pT while keeping eta/phi fixed.
        # For two back-to-back massless leptons with pT1=pT2=pT at eta=0,
        # phi=0 vs phi=pi:  mll ~ 2*pT.
        n = 10
        # Target mll values (approx 2*pt for massless back-to-back leptons)
        target_pt = [25, 50, 60, 90, 110, 150, 250, 500, 40, 125]
        pts_lead    = [[float(p), float(p)] for p in target_pt]
        etas        = [[0.0, 0.0]] * n
        phis        = [[0.0, 3.14159]] * n  # back-to-back
        masses      = [[0.0, 0.0]] * n
        charges     = [[1, -1]] * n
        flavors     = [["electron", "electron"]] * n

        tight_leptons = _make_leptons(pts_lead, etas, phis, masses, charges, flavors)

        # Jets well-separated from leptons
        jet_pts   = [[200.0, 150.0]] * n
        jet_etas  = [[2.0, -2.0]] * n
        jet_phis  = [[1.5, -1.5]] * n
        jet_masses = [[10.0, 10.0]] * n
        ak4_jets = _make_jets(jet_pts, jet_etas, jet_phis, jet_masses)

        lepton_masks, jet_masks = _ee_masks(n_events=n)
        triggers = _default_triggers(n)

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        # Compute the actual mll values from the leptons
        lpad = ak.pad_none(tight_leptons, 2)
        mll = (lpad[:, 0] + lpad[:, 1]).mass
        mll_np = ak.to_numpy(mll)

        # SR: mll > 200
        sr_mask = selections.all("mll_gt200")
        expected_sr = mll_np > CUTS["mll_sr_min"]
        np.testing.assert_array_equal(
            ak.to_numpy(sr_mask), expected_sr,
            err_msg="SR mll > 200 GeV cut not applied correctly",
        )

        # DY CR: 60 < mll < 150
        dy_cr_mask = selections.all("60_mll_150")
        expected_dy = (mll_np > CUTS["mll_dy_low"]) & (mll_np < CUTS["mll_dy_high"])
        np.testing.assert_array_equal(
            ak.to_numpy(dy_cr_mask), expected_dy,
            err_msg="DY CR 60-150 GeV window not applied correctly",
        )

    def test_mlljj_threshold(self, analyzer):
        """Test SR requires mlljj > 800 GeV.

        We hold leptons and one jet fixed and vary the second jet pT
        to sweep the four-object mass across 800 GeV.
        """
        n = 6
        # Fixed high-pT leptons (back-to-back, mll ~ 2*300 = 600 GeV)
        pts   = [[300.0, 300.0]] * n
        etas  = [[0.0, 0.0]] * n
        phis  = [[0.0, 3.14159]] * n
        masses = [[0.0, 0.0]] * n
        charges = [[1, -1]] * n
        flavors = [["electron", "electron"]] * n
        tight_leptons = _make_leptons(pts, etas, phis, masses, charges, flavors)

        # Vary second jet pT to control mlljj
        j2_pts = [50, 100, 150, 200, 300, 500]
        jet_pts   = [[200.0, float(p)] for p in j2_pts]
        jet_etas  = [[2.0, -2.0]] * n
        jet_phis  = [[1.5, -1.5]] * n
        jet_masses = [[10.0, 10.0]] * n
        ak4_jets = _make_jets(jet_pts, jet_etas, jet_phis, jet_masses)

        lepton_masks, jet_masks = _ee_masks(n_events=n)
        triggers = _default_triggers(n)

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        # Compute the actual mlljj
        lpad = ak.pad_none(tight_leptons, 2)
        jpad = ak.pad_none(ak4_jets, 2)
        mlljj = (lpad[:, 0] + lpad[:, 1] + jpad[:, 0] + jpad[:, 1]).mass
        mlljj_np = ak.to_numpy(mlljj)

        sr_mask = selections.all("mlljj_gt800")
        expected = mlljj_np > CUTS["mlljj_min"]
        np.testing.assert_array_equal(
            ak.to_numpy(sr_mask), expected,
            err_msg="mlljj > 800 GeV cut not applied correctly",
        )

    # ----- lepton pT cut tests --------------------------------------------

    def test_dilepton_pt_thresholds(self, analyzer):
        """Test leading/subleading lepton pT cuts (lead > 60, sublead > 53).

        Each sub-case is a single event with a different pT pair.
        """
        configs = [
            # (lead_pt, sublead_pt, expect_lead_pass, expect_sublead_pass)
            (65.0, 55.0, True,  True),
            (61.0, 54.0, True,  True),
            (70.0, 52.0, True,  False),
            (59.0, 55.0, False, True),
            (80.0, 60.0, True,  True),
            (55.0, 50.0, False, False),
        ]

        for lead_pt, sub_pt, expect_lead, expect_sub in configs:
            tight_leptons = _make_leptons(
                pts=[[lead_pt, sub_pt]],
                etas=[[0.0, 0.0]],
                phis=[[0.0, 3.14159]],
                masses=[[0.0, 0.0]],
                charges=[[1, -1]],
                flavors=[["electron", "electron"]],
            )
            ak4_jets = _make_jets(
                pts=[[200.0, 150.0]],
                etas=[[2.0, -2.0]],
                phis=[[1.5, -1.5]],
                masses=[[10.0, 10.0]],
            )
            lepton_masks, jet_masks = _ee_masks()
            triggers = _default_triggers()

            selections = analyzer.resolved_selections(
                tight_leptons, ak4_jets,
                lepton_masks=lepton_masks,
                jet_masks=jet_masks,
                triggers=triggers,
            )

            lead_cut = ak.to_numpy(selections.all("lead_tight_lepton_pt60"))[0]
            sub_cut  = ak.to_numpy(selections.all("sublead_tight_pt53"))[0]

            assert lead_cut == expect_lead, (
                f"lead_pt={lead_pt}: expected lead_tight_lepton_pt60={expect_lead}, got {lead_cut}"
            )
            assert sub_cut == expect_sub, (
                f"sub_pt={sub_pt}: expected sublead_tight_pt53={expect_sub}, got {sub_cut}"
            )

    # ----- flavor separation tests -----------------------------------------

    def test_flavor_separation_ee_mumu(self, analyzer):
        """Test ee and mumu channels are mutually exclusive."""
        # --- ee events (2 events) ---
        ee_leptons = _make_leptons(
            pts=[[65.0, 55.0], [70.0, 60.0]],
            etas=[[0.0, 0.0], [0.0, 0.0]],
            phis=[[0.0, 3.14159], [0.0, 3.14159]],
            masses=[[0.0, 0.0], [0.0, 0.0]],
            charges=[[1, -1], [1, -1]],
            flavors=[["electron", "electron"], ["electron", "electron"]],
        )
        ee_jets = _make_jets(
            pts=[[200.0, 150.0], [200.0, 150.0]],
            etas=[[2.0, -2.0], [2.0, -2.0]],
            phis=[[1.5, -1.5], [1.5, -1.5]],
            masses=[[10.0, 10.0], [10.0, 10.0]],
        )
        ee_lep_masks, ee_jet_masks = _ee_masks(n_events=2)
        ee_triggers = _default_triggers(2)

        sel_ee = analyzer.resolved_selections(
            ee_leptons, ee_jets,
            lepton_masks=ee_lep_masks,
            jet_masks=ee_jet_masks,
            triggers=ee_triggers,
        )

        # --- mumu events (2 events) ---
        mumu_leptons = _make_leptons(
            pts=[[65.0, 55.0], [70.0, 60.0]],
            etas=[[0.0, 0.0], [0.0, 0.0]],
            phis=[[0.0, 3.14159], [0.0, 3.14159]],
            masses=[[0.0, 0.0], [0.0, 0.0]],
            charges=[[1, -1], [1, -1]],
            flavors=[["muon", "muon"], ["muon", "muon"]],
        )
        mumu_lep_masks = {
            "ele_pteta": ak.Array([[], []]),
            "mu_pteta":  ak.Array([[True, True], [True, True]]),
            "ele_id":    ak.Array([[], []]),
            "mu_id":     ak.Array([[True, True], [True, True]]),
        }
        mumu_jet_masks = {
            "ak4_pteta": ak.Array([[True, True], [True, True]]),
            "ak4_id":    ak.Array([[True, True], [True, True]]),
        }
        mumu_triggers = _default_triggers(2)

        sel_mumu = analyzer.resolved_selections(
            mumu_leptons, mumu_jets := ee_jets,  # reuse same jets
            lepton_masks=mumu_lep_masks,
            jet_masks=mumu_jet_masks,
            triggers=mumu_triggers,
        )

        # ee events: tight flavor counts
        assert ak.all(sel_ee.all("two_tight_electrons"))
        assert not ak.any(sel_ee.all("two_tight_muons"))

        # mumu events: tight flavor counts
        assert ak.all(sel_mumu.all("two_tight_muons"))
        assert not ak.any(sel_mumu.all("two_tight_electrons"))

        # pteta-mask-based flavor
        assert ak.all(sel_ee.all("two_pteta_electrons"))
        assert not ak.any(sel_mumu.all("two_pteta_electrons"))
        assert ak.all(sel_mumu.all("two_pteta_muons"))
        assert not ak.any(sel_ee.all("two_pteta_muons"))

    def test_emu_flavor_cr(self, analyzer):
        """Test emu flavor control region (one electron + one muon)."""
        emu_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "muon"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        lepton_masks = {
            "ele_pteta": ak.Array([[True]]),
            "mu_pteta":  ak.Array([[True]]),
            "ele_id":    ak.Array([[True]]),
            "mu_id":     ak.Array([[True]]),
        }
        jet_masks = {
            "ak4_pteta": ak.Array([[True, True]]),
            "ak4_id":    ak.Array([[True, True]]),
        }
        triggers = _default_triggers()

        selections = analyzer.resolved_selections(
            emu_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        assert ak.to_numpy(selections.all("two_tight_em"))[0], \
            "emu event should pass two_tight_em"
        assert ak.to_numpy(selections.all("two_pteta_em"))[0], \
            "emu event should pass two_pteta_em"
        assert not ak.to_numpy(selections.all("two_tight_electrons"))[0]
        assert not ak.to_numpy(selections.all("two_tight_muons"))[0]

    # ----- jet multiplicity tests ------------------------------------------

    def test_min_two_jets(self, analyzer):
        """Test min_two_ak4_jets requires >= 2 jets."""
        for n_jets, should_pass in [(0, False), (1, False), (2, True), (3, True)]:
            tight_leptons = _make_leptons(
                pts=[[65.0, 55.0]],
                etas=[[0.0, 0.0]],
                phis=[[0.0, 3.14159]],
                masses=[[0.0, 0.0]],
                charges=[[1, -1]],
                flavors=[["electron", "electron"]],
            )
            jet_pts  = [[200.0] * n_jets] if n_jets > 0 else [[]]
            jet_etas = [[2.0 * ((-1)**i) for i in range(n_jets)]] if n_jets > 0 else [[]]
            jet_phis = [[1.5 * ((-1)**i) for i in range(n_jets)]] if n_jets > 0 else [[]]
            jet_masses = [[10.0] * n_jets] if n_jets > 0 else [[]]
            ak4_jets = _make_jets(jet_pts, jet_etas, jet_phis, jet_masses)

            lepton_masks, jet_masks = _ee_masks(n_jets=n_jets)
            triggers = _default_triggers()

            selections = analyzer.resolved_selections(
                tight_leptons, ak4_jets,
                lepton_masks=lepton_masks,
                jet_masks=jet_masks,
                triggers=triggers,
            )

            result = ak.to_numpy(selections.all("min_two_ak4_jets"))[0]
            assert result == should_pass, (
                f"n_jets={n_jets}: expected min_two_ak4_jets={should_pass}, got {result}"
            )

    # ----- delta-R tests ---------------------------------------------------

    def test_dr_all_pairs(self, analyzer):
        """Test delta-R > 0.4 requirement among {l1, l2, j1, j2}.

        We place all four objects at well-separated angles (should pass),
        then test a case where two objects are close (should fail).
        """
        # Well-separated case: all pairs have large delta-R
        tight_leptons = _make_leptons(
            pts=[[100.0, 80.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 2.0]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[1.5, -1.5]],
            phis=[[-2.0, -0.5]],
            masses=[[10.0, 10.0]],
        )
        lepton_masks, jet_masks = _ee_masks()
        triggers = _default_triggers()

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )
        assert ak.to_numpy(selections.all("dr_all_pairs_gt0p4"))[0], \
            "Well-separated objects should pass delta-R cut"

        # Close-pair case: l1 and j1 at nearly the same position
        tight_leptons_close = _make_leptons(
            pts=[[100.0, 80.0]],
            etas=[[0.0, 1.0]],
            phis=[[0.0, 2.0]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets_close = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[0.0, -1.5]],   # j1 at same eta as l1
            phis=[[0.1, -1.5]],   # j1 at very close phi to l1
            masses=[[10.0, 10.0]],
        )

        selections_close = analyzer.resolved_selections(
            tight_leptons_close, ak4_jets_close,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )
        assert not ak.to_numpy(selections_close.all("dr_all_pairs_gt0p4"))[0], \
            "Close pair (l1, j1) should fail delta-R cut"

    # ----- trigger propagation tests ---------------------------------------

    def test_trigger_propagation(self, analyzer):
        """Test that trigger booleans are stored in selections."""
        tight_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        lepton_masks, jet_masks = _ee_masks()

        # e_trig=True, mu_trig=False, emu_trig=True
        triggers = (ak.Array([True]), ak.Array([False]), ak.Array([True]))

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        assert ak.to_numpy(selections.all("e_trigger"))[0] == True
        assert ak.to_numpy(selections.all("mu_trigger"))[0] == False
        assert ak.to_numpy(selections.all("emu_trigger"))[0] == True

    # ----- lepton/jet mask propagation tests -------------------------------

    def test_lepton_mask_propagation(self, analyzer):
        """Test that lepton masks are properly counted and stored."""
        tight_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        triggers = _default_triggers()

        # Only 1 electron passes ID (should fail two_id_electrons)
        lepton_masks = {
            "ele_pteta": ak.Array([[True, True]]),
            "mu_pteta":  ak.Array([[]]),
            "ele_id":    ak.Array([[True, False]]),  # only 1 passes ID
            "mu_id":     ak.Array([[]]),
        }
        jet_masks = {
            "ak4_pteta": ak.Array([[True, True]]),
            "ak4_id":    ak.Array([[True, True]]),
        }

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        assert ak.to_numpy(selections.all("two_pteta_electrons"))[0] == True, \
            "2 electrons pass pteta"
        assert ak.to_numpy(selections.all("two_id_electrons"))[0] == False, \
            "Only 1 electron passes ID, should fail two_id_electrons"

    def test_jet_mask_propagation(self, analyzer):
        """Test that jet masks are properly counted and stored."""
        tight_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        triggers = _default_triggers()

        # Only 1 jet passes ID (should fail min_two_ak4_jets_id)
        lepton_masks, _ = _ee_masks()
        jet_masks = {
            "ak4_pteta": ak.Array([[True, True]]),
            "ak4_id":    ak.Array([[True, False]]),
        }

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        assert ak.to_numpy(selections.all("min_two_ak4_jets_pteta"))[0] == True
        assert ak.to_numpy(selections.all("min_two_ak4_jets_id"))[0] == False, \
            "Only 1 jet passes ID, should fail min_two_ak4_jets_id"

    # ----- return type test ------------------------------------------------

    def test_returns_packed_selection(self, analyzer):
        """resolved_selections returns a single PackedSelection object."""
        tight_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        lepton_masks, jet_masks = _ee_masks()
        triggers = _default_triggers()

        result = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        assert isinstance(result, PackedSelection), \
            f"Expected PackedSelection, got {type(result)}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_lepton_collections(self, analyzer):
        """Test handling of events with no leptons.

        We build a 2-event batch where one event has leptons (to anchor
        the column types) and one is empty, then slice to the empty one.
        This avoids awkward type-inference issues on fully-empty arrays.
        """
        two_evt = _make_leptons(
            pts=[[65.0, 55.0], []],
            etas=[[0.0, 0.0], []],
            phis=[[0.0, 3.14], []],
            masses=[[0.0, 0.0], []],
            charges=[[1, -1], []],
            flavors=[["electron", "electron"], []],
        )
        tight_leptons = two_evt[1:]  # keep only the empty event
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        lepton_masks = {
            "ele_pteta": ak.Array([[]]),
            "mu_pteta":  ak.Array([[]]),
            "ele_id":    ak.Array([[]]),
            "mu_id":     ak.Array([[]]),
        }
        jet_masks = {
            "ak4_pteta": ak.Array([[True, True]]),
            "ak4_id":    ak.Array([[True, True]]),
        }
        triggers = _default_triggers()

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        # No leptons: dilepton selections should all be False
        assert not ak.any(selections.all("two_tight_leptons"))
        assert not ak.any(selections.all("two_tight_electrons"))
        assert not ak.any(selections.all("two_tight_muons"))

    def test_single_lepton_event(self, analyzer):
        """Test handling of events with only one lepton (should fail 2-lepton cuts)."""
        tight_leptons = _make_leptons(
            pts=[[70.0]],
            etas=[[0.0]],
            phis=[[0.0]],
            masses=[[0.0]],
            charges=[[1]],
            flavors=[["electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )
        lepton_masks = {
            "ele_pteta": ak.Array([[True]]),
            "mu_pteta":  ak.Array([[]]),
            "ele_id":    ak.Array([[True]]),
            "mu_id":     ak.Array([[]]),
        }
        jet_masks = {
            "ak4_pteta": ak.Array([[True, True]]),
            "ak4_id":    ak.Array([[True, True]]),
        }
        triggers = _default_triggers()

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        assert not ak.any(selections.all("two_tight_leptons")), \
            "1 lepton should fail two_tight_leptons"
        assert not ak.any(selections.all("two_tight_electrons")), \
            "1 electron should fail two_tight_electrons"

    def test_missing_jets(self, analyzer):
        """Test handling of events with no jets."""
        tight_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[]],
            etas=[[]],
            phis=[[]],
            masses=[[]],
        )
        lepton_masks, _ = _ee_masks(n_jets=0)
        jet_masks = {
            "ak4_pteta": ak.Array([[]]),
            "ak4_id":    ak.Array([[]]),
        }
        triggers = _default_triggers()

        selections = analyzer.resolved_selections(
            tight_leptons, ak4_jets,
            lepton_masks=lepton_masks,
            jet_masks=jet_masks,
            triggers=triggers,
        )

        # Dilepton cuts should still pass
        assert ak.to_numpy(selections.all("two_tight_electrons"))[0], \
            "2 electrons present, should pass flavor count"
        assert ak.to_numpy(selections.all("two_tight_leptons"))[0], \
            "2 leptons present, should pass count"

        # Jet-related cuts should fail
        assert not ak.to_numpy(selections.all("min_two_ak4_jets"))[0], \
            "No jets, should fail min_two_ak4_jets"
        assert not ak.to_numpy(selections.all("min_two_ak4_jets_pteta"))[0], \
            "No jets, should fail min_two_ak4_jets_pteta"

    def test_lepton_masks_required(self, analyzer):
        """resolved_selections raises ValueError when masks are None."""
        tight_leptons = _make_leptons(
            pts=[[65.0, 55.0]],
            etas=[[0.0, 0.0]],
            phis=[[0.0, 3.14159]],
            masses=[[0.0, 0.0]],
            charges=[[1, -1]],
            flavors=[["electron", "electron"]],
        )
        ak4_jets = _make_jets(
            pts=[[200.0, 150.0]],
            etas=[[2.0, -2.0]],
            phis=[[1.5, -1.5]],
            masses=[[10.0, 10.0]],
        )

        with pytest.raises(ValueError, match="lepton_masks and jet_masks"):
            analyzer.resolved_selections(tight_leptons, ak4_jets)
