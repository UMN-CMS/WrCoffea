"""Unit tests for the wrcoffea.skimmer module."""

from unittest.mock import MagicMock, patch

import awkward as ak
import numpy as np
import pytest

import wrcoffea.skimmer as skimmer
from wrcoffea.analysis_config import CUTS
from wrcoffea.skimmer import (
    KEPT_BRANCHES,
    SKIM_CUTS,
    SUMMED_RUNS_BRANCHES,
    SkimResult,
    read_runs_tree,
)


# ---------------------------------------------------------------------------
# Skim cuts vs analysis cuts
# ---------------------------------------------------------------------------


class TestSkimCuts:
    """Skim cuts must always be looser than (or equal to) analysis cuts."""

    def test_lepton_pt(self):
        assert SKIM_CUTS["lepton_pt_min"] <= CUTS["lepton_pt_min"]

    def test_lepton_eta(self):
        # Analysis cuts split lepton eta into electron (2.5) and muon (2.4);
        # the skim keeps a single flavor-agnostic cut, so it must be looser
        # than (or equal to) the loosest analysis-level lepton eta cut.
        assert SKIM_CUTS["lepton_eta_max"] >= max(
            CUTS["electron_eta_max"], CUTS["muon_eta_max"]
        )

    def test_ak4_pt(self):
        assert SKIM_CUTS["ak4_pt_min"] <= CUTS["ak4_pt_min"]

    def test_ak4_eta(self):
        assert SKIM_CUTS["ak4_eta_max"] >= CUTS["ak4_eta_max"]

    def test_ak8_pt(self):
        assert SKIM_CUTS["ak8_pt_min"] <= CUTS["ak8_pt_min"]

    def test_lead_lepton_pt(self):
        assert SKIM_CUTS["lead_lepton_pt_min"] <= CUTS["lead_lepton_pt_min"]

    def test_sublead_lepton_pt(self):
        assert SKIM_CUTS["sublead_lepton_pt_min"] <= CUTS["sublead_lepton_pt_min"]


# ---------------------------------------------------------------------------
# Kept branches
# ---------------------------------------------------------------------------


class TestKeptBranches:
    """Verify that critical branches are in the kept set."""

    @pytest.mark.parametrize(
        "branch",
        [
            "genWeight", "HLT", "Muon", "Electron", "Jet", "FatJet",
            "MET", "Pileup", "PV", "Flag", "Generator", "GenPart",
            "event", "run", "luminosityBlock",
        ],
    )
    def test_essential_branches_kept(self, branch):
        assert branch in KEPT_BRANCHES


# ---------------------------------------------------------------------------
# read_runs_tree
# ---------------------------------------------------------------------------


class TestReadRunsTree:
    """Test Runs tree collapsing (the genEventSumw inflation fix)."""

    class _MockAsDtype:
        """Mock for uproot AsDtype interpretation."""
        def __init__(self, dtype):
            self._dtype = np.dtype(dtype)

        @property
        def to_dtype(self):
            return self._dtype

    class _MockAsJagged:
        """Mock for uproot AsJagged interpretation."""
        def __init__(self, content_dtype):
            self.content = TestReadRunsTree._MockAsDtype(content_dtype)

    def _mock_uproot_open(self, branch_data, branch_dtypes):
        """Create a mock for uproot.open with proper branch interpretations.

        branch_data : dict of branch_name -> list/array values
        branch_dtypes : dict of branch_name -> (dtype_str, is_jagged)
        """
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: key == "Runs"

        mock_runs_tree = MagicMock()
        mock_runs_tree.keys.return_value = list(branch_data.keys())

        # Mock branch interpretations
        interps = {}
        for bname, (dtype_str, is_jagged) in branch_dtypes.items():
            if is_jagged:
                interps[bname] = TestReadRunsTree._MockAsJagged(dtype_str)
            else:
                interps[bname] = TestReadRunsTree._MockAsDtype(dtype_str)

        def branch_getitem(key):
            mock_branch = MagicMock()
            mock_branch.interpretation = interps[key]
            return mock_branch

        # MagicMock passes (self, key) to assigned dunder methods
        mock_runs_tree.__getitem__ = lambda _self, key: branch_getitem(key)

        # Return awkward array from .arrays()
        ak_data = ak.Array({k: v for k, v in branch_data.items()})
        mock_runs_tree.arrays.return_value = ak_data

        mock_file.__getitem__ = lambda _self, key: mock_runs_tree

        cm = MagicMock()
        cm.__enter__ = lambda self: mock_file
        cm.__exit__ = lambda self, *a: None
        return cm

    def test_single_entry_unchanged(self):
        branch_data = {
            "genEventSumw_": [1000.0],
            "run": [np.uint32(1)],
        }
        branch_dtypes = {
            "genEventSumw_": ("float64", False),
            "run": ("uint32", False),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(branch_data, branch_dtypes)):
            result = read_runs_tree("fake.root")

        assert result is not None
        data, types = result
        assert float(data["genEventSumw_"][0]) == pytest.approx(1000.0)
        assert int(data["run"][0]) == 1

    def test_multi_entry_sumw_summed(self):
        branch_data = {
            "genEventSumw_": [100.0, 200.0, 300.0],
            "genEventCount_": [10, 20, 30],
            "run": [np.uint32(1), np.uint32(1), np.uint32(1)],
        }
        branch_dtypes = {
            "genEventSumw_": ("float64", False),
            "genEventCount_": ("int64", False),
            "run": ("uint32", False),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(branch_data, branch_dtypes)):
            result = read_runs_tree("fake.root")

        assert result is not None
        data, types = result
        assert float(data["genEventSumw_"][0]) == pytest.approx(600.0)
        assert int(data["genEventCount_"][0]) == 60
        assert len(data["run"]) == 1

    def test_ratio_branches_take_the_sumw_weighted_mean(self):
        branch_data = {
            "genEventSumw_": [100.0, 200.0],
            "nLHEScaleSumw": [np.int32(3), np.int32(3)],
            "LHEScaleSumw": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "run": [np.uint32(1), np.uint32(1)],
        }
        branch_dtypes = {
            "genEventSumw_": ("float64", False),
            "nLHEScaleSumw": ("int32", False),
            "LHEScaleSumw": ("float64", True),
            "run": ("uint32", False),
        }
        with patch("wrcoffea.skimmer.uproot.open", return_value=self._mock_uproot_open(branch_data, branch_dtypes)):
            result = read_runs_tree("fake.root")

        assert result is not None
        data, types = result
        # LHEScaleSumw is a ratio to genEventSumw, so collapsing two entries
        # takes the sumw-weighted mean: (100*[1,2,3] + 200*[4,5,6]) / 300.
        collapsed = ak.to_list(data["LHEScaleSumw"])
        assert collapsed == [[pytest.approx(3.0), pytest.approx(4.0), pytest.approx(5.0)]]
        # nLHEScaleSumw is not in SUMMED_RUNS_BRANCHES, just take first
        assert int(data["nLHEScaleSumw"][0]) == 3
        # Branch types should include jagged spec
        assert types["LHEScaleSumw"] == "var * float64"

    def test_missing_runs_returns_none(self):
        mock_file = MagicMock()
        mock_file.__contains__ = lambda self, key: False
        cm = MagicMock()
        cm.__enter__ = lambda self: mock_file
        cm.__exit__ = lambda self, *a: None
        with patch("wrcoffea.skimmer.uproot.open", return_value=cm):
            result = read_runs_tree("fake.root")

        assert result is None



# ---------------------------------------------------------------------------
# SkimResult
# ---------------------------------------------------------------------------


class TestSkimResult:
    def test_construction(self):
        r = SkimResult(
            src_path="a.root", dest_path="b.root",
            n_events_before=1000, n_events_after=500,
            file_size_bytes=1024, efficiency=50.0,
        )
        assert r.n_events_before == 1000
        assert r.efficiency == 50.0
        assert r.status == "success"
        assert r.attempts == 1
        assert r.redirector is None


# ---------------------------------------------------------------------------
# Retry/error classification
# ---------------------------------------------------------------------------


class TestRetryClassification:
    def test_classify_network_error_retryable(self):
        cat, retryable = skimmer._classify_skim_error(RuntimeError("operation timed out on xrootd socket"))
        assert cat == "network_error"
        assert retryable is True

    def test_classify_corrupt_error_nonretryable(self):
        cat, retryable = skimmer._classify_skim_error(RuntimeError("received 0 bytes from server"))
        assert cat == "corrupt_file"
        assert retryable is False

    def test_classify_corrupt_pattern_with_network_signature_is_retryable(self):
        # A transport error surfacing mid-read matches broad corrupt patterns
        # (e.g. "basket"), but a strong network signature takes precedence and
        # keeps the failure retryable as network_error.
        cat, retryable = skimmer._classify_skim_error(
            RuntimeError("failed to read basket: operation timed out")
        )
        assert cat == "network_error"
        assert retryable is True

    def test_classify_corrupt_pattern_without_network_signature_stays_corrupt(self):
        cat, retryable = skimmer._classify_skim_error(
            RuntimeError("zlib decompression failure in basket 12")
        )
        assert cat == "corrupt_file"
        assert retryable is False

    def test_classify_empty_error_nonretryable(self):
        cat, retryable = skimmer._classify_skim_error(RuntimeError("branch not in fields"))
        assert cat == "empty_input"
        assert retryable is False

    def test_retry_backoff_exponential_with_cap(self):
        with patch.object(skimmer.random, "uniform", return_value=0.0):
            assert skimmer._retry_sleep_seconds(1) == pytest.approx(1.0)
            assert skimmer._retry_sleep_seconds(2) == pytest.approx(2.0)
            assert skimmer._retry_sleep_seconds(3) == pytest.approx(4.0)
            assert skimmer._retry_sleep_seconds(6) == pytest.approx(30.0)


class TestSkimSingleFileRetryFlow:
    def test_records_attempts_and_redirector(self):
        calls = []

        def _side_effect(_src, _dest, progress=None):
            calls.append(_src)
            if len(calls) == 1:
                raise RuntimeError("timeout contacting xrootd")
            return SkimResult(
                src_path=_src,
                dest_path=_dest,
                n_events_before=10,
                n_events_after=5,
                file_size_bytes=100,
                efficiency=50.0,
            )

        with patch.object(skimmer, "REDIRECTORS", ["root://r1/", "root://r2/"]), \
             patch.object(skimmer, "MAX_RETRIES_PER_REDIRECTOR", 2), \
             patch.object(skimmer, "_skim_impl", side_effect=_side_effect), \
             patch.object(skimmer, "_retry_sleep_seconds", return_value=0.0), \
             patch.object(skimmer.time, "sleep"):
            result = skimmer.skim_single_file(
                "root://host//store/mc/f.root",
                "out.root",
            )

        assert result.attempts == 2
        assert result.redirector == "root://r1/"

    def test_nonretryable_error_raises_immediately(self):
        with patch.object(skimmer, "REDIRECTORS", ["root://r1/"]), \
             patch.object(skimmer, "MAX_RETRIES_PER_REDIRECTOR", 3), \
             patch.object(skimmer, "_skim_impl", side_effect=RuntimeError("received 0 bytes")), \
             patch.object(skimmer.time, "sleep") as sleep_mock:
            with pytest.raises(RuntimeError, match=r"Non-retryable \[corrupt_file\]"):
                skimmer.skim_single_file("root://host//store/mc/f.root", "out.root")

        sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# collapse_runs_entries (shared by the skim and the merge)
# ---------------------------------------------------------------------------

from wrcoffea.skimmer import collapse_runs_entries, _kept_branch_names, RATIO_RUNS_JAGGED


class TestCollapseRunsEntries:
    def test_ratio_branches_are_weighted_by_sumw_and_counts_summed(self):
        out = collapse_runs_entries({
            "genEventSumw": ak.Array([100.0, 300.0]),
            "genEventCount": ak.Array([10, 30]),
            "LHEScaleSumw": ak.Array([[1.0, 2.0], [2.0, 4.0]]),
            "PSSumw": ak.Array([[0.5], [1.0]]),
            "run": ak.Array([1, 1]),
        })
        assert float(out["genEventSumw"][0]) == 400.0
        assert int(out["genEventCount"][0]) == 40
        # (100*1 + 300*2)/400, (100*2 + 300*4)/400
        assert ak.to_list(out["LHEScaleSumw"]) == [[pytest.approx(1.75), pytest.approx(3.5)]]
        assert ak.to_list(out["PSSumw"]) == [[pytest.approx(0.875)]]
        assert len(out["run"]) == 1

    def test_single_entry_is_unchanged(self):
        out = collapse_runs_entries({
            "genEventSumw": ak.Array([50.0]),
            "LHEPdfSumw": ak.Array([[0.9, 1.1, 1.0]]),
        })
        assert ak.to_list(out["LHEPdfSumw"]) == [[pytest.approx(0.9), pytest.approx(1.1), pytest.approx(1.0)]]

    def test_no_sumw_falls_back_to_plain_mean(self):
        out = collapse_runs_entries({"LHEScaleSumw": ak.Array([[1.0, 3.0], [3.0, 5.0]])})
        assert ak.to_list(out["LHEScaleSumw"]) == [[pytest.approx(2.0), pytest.approx(4.0)]]

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="different lengths"):
            collapse_runs_entries({
                "genEventSumw": ak.Array([1.0, 1.0]),
                "LHEScaleSumw": ak.Array([[1.0, 2.0], [1.0]]),
            })

    def test_ratio_branch_set_covers_the_nanoaod_names(self):
        assert RATIO_RUNS_JAGGED == {"LHEScaleSumw", "LHEPdfSumw", "PSSumw"}


class TestKeptBranchNames:
    """Branch-name mapping, including the flat pre-v10 rho and the MET variants."""

    NANOAOD_V9 = [
        "run", "event", "fixedGridRhoFastjetAll", "fixedGridRhoFastjetCentralCalo",
        "nJet", "Jet_pt", "nMuon", "Muon_pt", "MET_pt", "PuppiMET_pt", "RawPuppiMET_pt",
        "L1PreFiringWeight_Nom", "nTrigObj", "TrigObj_pt", "SoftActivityJetHT", "nPhoton", "Photon_pt",
    ]
    NANOAOD_V15 = ["Rho_fixedGridRhoFastjetAll", "PFMET_pt", "RawPFMET_pt", "PuppiMET_pt", "nTrigObj", "TrigObj_pt"]

    def test_flat_rho_is_kept_for_nanoaod_v9(self):
        kept = set(_kept_branch_names(self.NANOAOD_V9))
        assert {"fixedGridRhoFastjetAll", "fixedGridRhoFastjetCentralCalo"} <= kept

    def test_rho_collection_is_kept_for_nanoaod_v15(self):
        assert "Rho_fixedGridRhoFastjetAll" in _kept_branch_names(self.NANOAOD_V15)

    def test_met_variants_prefiring_and_trigger_objects_are_kept(self):
        kept = set(_kept_branch_names(self.NANOAOD_V9 + self.NANOAOD_V15))
        assert {"MET_pt", "PuppiMET_pt", "RawPuppiMET_pt", "PFMET_pt", "RawPFMET_pt",
                "L1PreFiringWeight_Nom", "nTrigObj", "TrigObj_pt"} <= kept

    def test_unwanted_collections_are_still_dropped(self):
        kept = set(_kept_branch_names(self.NANOAOD_V9))
        assert not {"SoftActivityJetHT", "nPhoton", "Photon_pt"} & kept


# ---------------------------------------------------------------------------
# apply_skim_selection: muon pT uses max(PF, TuneP)
# ---------------------------------------------------------------------------

from wrcoffea.skimmer import apply_skim_selection


def _skim_events(muon_pt, muon_tunep_rel, electron_pt=(), jet_pt=(80.0, 60.0)):
    """One event with the given muons/electrons/jets (eta = 0 everywhere)."""
    def coll(pts, extra=None):
        # Full four-vector fields: once the vector library's behaviours are
        # registered (scale_factors imports them), the PtEtaPhiMCandidate
        # name the skimmer assigns needs pt/eta/phi/mass to be present.
        n = len(pts)
        fields = {"pt": ak.Array([list(pts)]), "eta": ak.Array([[0.0] * n]),
                  "phi": ak.Array([[0.0] * n]), "mass": ak.Array([[0.1] * n]), "charge": ak.Array([[1] * n])}
        if extra:
            fields.update({k: ak.Array([list(v)]) for k, v in extra.items()})
        return ak.zip(fields)
    return ak.zip({
        "Muon": coll(muon_pt, {"tunepRelPt": muon_tunep_rel}),
        "Electron": coll(electron_pt),
        "Jet": coll(jet_pt),
        "FatJet": coll([]),
    }, depth_limit=1)


class TestSkimSelectionMuonMomentum:
    def test_muon_below_pf_threshold_but_above_on_tunep_is_kept(self):
        # PF pT 40 (fails 45) but TuneP = 40 * 1.5 = 60 (passes); partner at 60.
        events = _skim_events(muon_pt=[60.0, 40.0], muon_tunep_rel=[1.0, 1.5])
        _, mask, _ = apply_skim_selection(events)
        assert ak.to_list(mask) == [True]

    def test_muon_failing_both_is_dropped(self):
        events = _skim_events(muon_pt=[60.0, 40.0], muon_tunep_rel=[1.0, 1.05])
        _, mask, _ = apply_skim_selection(events)
        assert ak.to_list(mask) == [False]

    def test_leading_lepton_ranking_uses_the_larger_momentum(self):
        # Only the TuneP value of the second muon exceeds the 52 GeV lead cut.
        events = _skim_events(muon_pt=[46.0, 45.5], muon_tunep_rel=[1.0, 1.2])
        _, mask, _ = apply_skim_selection(events)
        assert ak.to_list(mask) == [True]

    def test_without_tunep_branch_pf_pt_is_used(self):
        events = _skim_events(muon_pt=[60.0, 40.0], muon_tunep_rel=[1.0, 1.5])
        muon = events.Muon
        events = ak.zip({"Muon": ak.zip({f: muon[f] for f in muon.fields if f != "tunepRelPt"}),
                         "Electron": events.Electron, "Jet": events.Jet, "FatJet": events.FatJet}, depth_limit=1)
        _, mask, _ = apply_skim_selection(events)
        assert ak.to_list(mask) == [False]


# ---------------------------------------------------------------------------
# End-to-end skim of a synthetic NanoAOD file
# ---------------------------------------------------------------------------

import os
import uproot
from wrcoffea import skimmer as skimmer_module
from wrcoffea.skimmer import _skim_impl, XROOTD_REQUEST_TIMEOUT, XROOTD_CONNECTION_TIMEOUT


def _write_nano_file(path, n_events=6):
    """A minimal NanoAODv9-like file: three events pass the skim, three do not."""
    rng = np.random.default_rng(3)
    # events 0-2: two 60 GeV muons + two 50 GeV jets (pass); 3-5: one muon only (fail)
    mu_pt = [[60.0, 55.0]] * 3 + [[60.0]] * 3
    jet_pt = [[50.0, 45.0]] * 6
    def jag(vals, dtype=np.float32):
        return ak.values_astype(ak.Array(vals), dtype)
    def zeros_like(vals, dtype=np.float32):
        return ak.values_astype(ak.Array([[0.0] * len(v) for v in vals]), dtype)
    events = {
        "run": np.full(n_events, 1, dtype=np.uint32),
        "event": np.arange(n_events, dtype=np.uint64),
        "luminosityBlock": np.full(n_events, 1, dtype=np.uint32),
        "genWeight": np.ones(n_events, dtype=np.float32),
        "fixedGridRhoFastjetAll": np.full(n_events, 20.0, dtype=np.float32),
        "SoftActivityJetHT": np.zeros(n_events, dtype=np.float32),
        "Muon_pt": jag(mu_pt), "Muon_eta": zeros_like(mu_pt), "Muon_phi": zeros_like(mu_pt),
        "Muon_mass": zeros_like(mu_pt), "Muon_charge": zeros_like(mu_pt, np.int32),
        "Muon_tunepRelPt": ak.values_astype(ak.Array([[1.0] * len(v) for v in mu_pt]), np.float32),
        "Electron_pt": jag([[]] * n_events), "Electron_eta": jag([[]] * n_events), "Electron_phi": jag([[]] * n_events),
        "Electron_mass": jag([[]] * n_events), "Electron_charge": jag([[]] * n_events, np.int32),
        "Jet_pt": jag(jet_pt), "Jet_eta": zeros_like(jet_pt), "Jet_phi": zeros_like(jet_pt), "Jet_mass": zeros_like(jet_pt),
        "FatJet_pt": jag([[]] * n_events), "FatJet_eta": jag([[]] * n_events), "FatJet_phi": jag([[]] * n_events),
        "FatJet_mass": jag([[]] * n_events),
    }
    def spec(v):
        # Jagged branches must be given as "var * <dtype>" strings: that is
        # what the skimmer's uproot shared-counter patch keys on.
        if isinstance(v, ak.Array):
            return "var * " + str(ak.to_numpy(ak.flatten(v, axis=None)).dtype)
        return v.dtype
    with uproot.recreate(path) as f:
        f.mktree("Events", {k: spec(v) for k, v in events.items()},
                 counter_name=lambda counted: "n" + counted.split("_")[0])
        f["Events"].extend(events)
        f.mktree("Runs", {"run": np.uint32, "genEventSumw": np.float64, "genEventCount": np.int64,
                          "LHEScaleSumw": "var * float32"})
        f["Runs"].extend({"run": np.array([1, 1], dtype=np.uint32),
                          "genEventSumw": np.array([100.0, 300.0]), "genEventCount": np.array([100, 300]),
                          "LHEScaleSumw": ak.Array([np.array([1.0, 2.0], np.float32), np.array([2.0, 4.0], np.float32)])})


class TestSkimEndToEnd:
    def test_skim_writes_selected_events_rho_and_collapsed_runs(self, tmp_path):
        src = tmp_path / "nano.root"; dest = tmp_path / "out" / "nano_skim.root"
        _write_nano_file(src)
        result = _skim_impl(str(src), str(dest))
        assert result.status == "success"
        assert (result.n_events_before, result.n_events_after) == (6, 3)
        assert dest.exists() and not (tmp_path / "out" / "nano_skim.root.partial").exists()
        with uproot.open(dest) as f:
            keys = f["Events"].keys()
            assert "fixedGridRhoFastjetAll" in keys and "Muon_tunepRelPt" in keys
            assert "SoftActivityJetHT" not in keys
            assert f["Events"].num_entries == 3
            runs = f["Runs"].arrays(library="ak")
            assert float(runs["genEventSumw"][0]) == 400.0
            assert ak.to_list(runs["LHEScaleSumw"]) == [[pytest.approx(1.75, rel=1e-6), pytest.approx(3.5, rel=1e-6)]]

    def test_failed_skim_leaves_no_finished_looking_output(self, tmp_path, monkeypatch):
        src = tmp_path / "nano.root"; dest = tmp_path / "out" / "nano_skim.root"
        _write_nano_file(src)
        def boom(*a, **k):
            raise RuntimeError("simulated read failure")
        monkeypatch.setattr(skimmer_module, "read_runs_tree", boom)
        with pytest.raises(RuntimeError, match="simulated"):
            _skim_impl(str(src), str(dest))
        assert not dest.exists()


class TestXrootdTimeouts:
    def test_request_timeout_is_not_seconds_scale(self):
        # The streaming pass issues one vector read per 50k events x ~1000
        # branches; a 10 s request deadline made routine reads expire.
        assert XROOTD_REQUEST_TIMEOUT >= 60
        assert XROOTD_CONNECTION_TIMEOUT >= 10
