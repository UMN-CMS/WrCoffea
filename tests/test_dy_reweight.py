"""DY Z-pT reweighting (Run 2 HNWRAnalyzer port).

Checks the generator-level Z finder against the GenFinderForDY rules, the
histogram lookup (including the Run 2 pT clamps) against the converted
payload, the identity behaviour for non-DY datasets, and the wiring into
``build_event_weights`` and the systematic registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import vector

from wrcoffea.analysis_config import DY_ZPT_REWEIGHT_JSONS, LUMIS
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.dy_reweight import (
    DY_PROCESS_GROUP,
    SOURCE_NAMES,
    dy_zpt_weight,
    evaluate,
    find_gen_z,
    identity_weights,
    load_payload,
)
from wrcoffea.systematics import NOMINAL, expected_labels, systematics_for, validate_requested

UL18 = "RunIISummer20UL18"
RUN3 = "Run3Summer22"
UNMAPPED = "Run3Summer99"       # no era is left without a payload; a made-up one stands in for "unconfigured"

FROM_HARD = 1 << 8


def _genpart(rows):
    """Build a GenPart-like jagged collection with Lorentz-vector behaviour.

    Each row is a list of ``(pt, eta, phi, mass, pdgId, status, statusFlags)``.
    """
    def col(i, dtype):
        return ak.Array([[r[i] for r in row] for row in rows]) if rows else ak.Array([])
    return ak.zip(
        {
            "pt": ak.values_astype(col(0, float), np.float64),
            "eta": ak.values_astype(col(1, float), np.float64),
            "phi": ak.values_astype(col(2, float), np.float64),
            "mass": ak.values_astype(col(3, float), np.float64),
            "pdgId": ak.values_astype(col(4, int), np.int32),
            "status": ak.values_astype(col(5, int), np.int32),
            "statusFlags": ak.values_astype(col(6, int), np.int32),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )


class _Events:
    def __init__(self, genpart, n=None):
        self.GenPart = genpart
        self._n = len(genpart) if n is None else n
        # NanoAOD L1 prefiring record (unit weights): build_event_weights reads it for the Run 2 UL eras
        from types import SimpleNamespace
        ones = ak.Array([1.0] * self._n)
        self.L1PreFiringWeight = SimpleNamespace(Nom=ones, Up=ones, Dn=ones)

    def __len__(self):
        return self._n


# ---------------------------------------------------------------------------
# Gen Z finder
# ---------------------------------------------------------------------------

class TestFindGenZ:
    def test_sfos_hard_process_pair(self):
        # Back-to-back muons: pT of the pair is 0, mass ~ 2*pt (massless, eta 0).
        gp = _genpart([[
            (21.0, 0.0, 0.0, 0.0, 21, 21, 0),           # initial-state gluon, ignored
            (50.0, 0.0, 0.0, 0.105, 13, 1, FROM_HARD),
            (50.0, 0.0, np.pi, 0.105, -13, 1, FROM_HARD),
        ]])
        pt, mass, found = find_gen_z(_Events(gp))
        assert found.tolist() == [True]
        assert pt[0] == pytest.approx(0.0, abs=1e-6)
        assert mass[0] == pytest.approx(100.0, rel=1e-3)

    def test_boosted_pair_pt(self):
        gp = _genpart([[
            (80.0, 0.5, 0.1, 0.0005, 11, 1, FROM_HARD),
            (40.0, -0.2, 0.3, 0.0005, -11, 1, FROM_HARD),
        ]])
        pt, _, found = find_gen_z(_Events(gp))
        v = gp[0, 0] + gp[0, 1]
        assert found[0] and pt[0] == pytest.approx(float(v.pt))

    def test_taus_use_status_two(self):
        gp = _genpart([[
            (60.0, 0.1, 0.0, 1.777, 15, 23, FROM_HARD),    # hard copy, not status 2 -> skipped
            (58.0, 0.1, 0.0, 1.777, 15, 2, FROM_HARD),
            (58.0, -0.1, 3.0, 1.777, -15, 2, FROM_HARD),
        ]])
        _, _, found = find_gen_z(_Events(gp))
        assert found.tolist() == [True]

    def test_non_hard_process_leptons_ignored(self):
        gp = _genpart([[
            (50.0, 0.0, 0.0, 0.105, 13, 1, 0),             # e.g. from a hadron decay
            (50.0, 0.0, 3.0, 0.105, -13, 1, 0),
        ]])
        pt, mass, found = find_gen_z(_Events(gp))
        assert found.tolist() == [False] and pt[0] == 0.0 and mass[0] == 0.0

    def test_first_two_candidates_must_be_sfos(self):
        gp = _genpart([
            [(50.0, 0.0, 0.0, 0.105, 13, 1, FROM_HARD), (50.0, 0.0, 3.0, 0.0005, -11, 1, FROM_HARD)],  # OS, OF
            [(50.0, 0.0, 0.0, 0.105, 13, 1, FROM_HARD), (50.0, 0.0, 3.0, 0.105, 13, 1, FROM_HARD)],    # SS
            [(50.0, 0.0, 0.0, 0.105, 13, 1, FROM_HARD)],                                                # one lepton
            [],
        ])
        _, _, found = find_gen_z(_Events(gp))
        assert found.tolist() == [False, False, False, False]


# ---------------------------------------------------------------------------
# Lookup against the converted payload
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def payload():
    return load_payload(DY_ZPT_REWEIGHT_JSONS[UL18])


@pytest.fixture(scope="module")
def raw_payload():
    with open(DY_ZPT_REWEIGHT_JSONS[UL18], encoding="utf-8") as fh:
        return json.load(fh)


class TestEvaluate:
    def test_payload_provenance_and_shape(self, raw_payload):
        """The mapped UL18 payload is our own AN-19-083-style derivation (since 2026-10-11) in the Run 2 bins; the
        converted Run 2 file next to it keeps the SKFlat provenance."""
        prov = raw_payload["provenance"]
        assert prov["era"] == "RunIISummer20UL18" and "AN-19-083" in prov["method"]
        assert raw_payload["qcd"]["edges"][0] == 0.0 and raw_payload["qcd"]["edges"][-1] == 8000.0
        assert len(raw_payload["qcd"]["edges"]) == 29                                   # GetZPtReweight.C binning
        assert raw_payload["ew"]["edges"][0] == 30.0 and raw_payload["ew"]["edges"][-1] == 6500.0
        run2 = json.loads(Path("data/reweights/RunII/2018/RunIISummer20UL18/dy_zpt_reweight_run2legacy_v4_2018.json").read_text())
        assert run2["provenance"]["source_branch"] == "HNWR/Run2Legacy_v4"
        assert run2["provenance"]["source_commit"] == "06f274c754cff941c5e406b8f8d50bfe650e20a0"
        assert run2["qcd"]["edges"] == raw_payload["qcd"]["edges"] and run2["ew"]["edges"] == raw_payload["ew"]["edges"]

    def test_nominal_is_qcd_times_ew(self, payload):
        pt = np.array([55.0])          # QCD bin [50,60) = 11, EW bin [50,60) = 2
        out = evaluate(pt, np.array([True]), payload)
        expected = payload["qcd"]["central"][11] * payload["ew"]["central"][2]
        assert out["nominal"][0] == pytest.approx(expected)
        assert out["applied"]

    def test_bin_edges_follow_root_findbin(self, payload):
        # A value on a bin edge belongs to the upper bin.
        out = evaluate(np.array([50.0, 49.999]), np.array([True, True]), payload)
        assert out["nominal"][0] == pytest.approx(payload["qcd"]["central"][11] * payload["ew"]["central"][2])
        assert out["nominal"][1] == pytest.approx(payload["qcd"]["central"][10] * payload["ew"]["central"][1])

    def test_run2_clamps(self, payload):
        # ptZ >= 8000 -> 7999 (last QCD bin); ptZ >= 6500 -> 6499 (last EW bin);
        # ptZ < 30 -> 31 (first EW bin) while the QCD lookup uses the true pT.
        out = evaluate(np.array([9000.0, 10.0]), np.array([True, True]), payload)
        assert out["nominal"][0] == pytest.approx(payload["qcd"]["central"][-1] * payload["ew"]["central"][-1])
        assert out["nominal"][1] == pytest.approx(payload["qcd"]["central"][5] * payload["ew"]["central"][0])

    def test_variations(self, payload):
        pt = np.array([300.0])
        iq = 23  # [250,350)
        ie = 15  # [300,350)
        out = evaluate(pt, np.array([True]), payload)
        q, e = payload["qcd"], payload["ew"]
        v = out["variations"]
        assert set(v) == set(SOURCE_NAMES)
        assert v["DYZptQcdScale"][0][0] == pytest.approx(q["scale_up"][iq] * e["central"][ie])
        assert v["DYZptQcdScale"][1][0] == pytest.approx(q["scale_down"][iq] * e["central"][ie])
        assert v["DYZptQcdPdf"][0][0] == pytest.approx((q["central"][iq] + q["pdf_err"][iq]) * e["central"][ie])
        assert v["DYZptQcdAlphaS"][1][0] == pytest.approx(q["alphas_down"][iq] * e["central"][ie])
        assert v["DYZptEw1"][0][0] == pytest.approx(q["central"][iq] * (e["central"][ie] + e["e1_err"][ie]))
        assert v["DYZptEw3"][1][0] == pytest.approx(q["central"][iq] * (e["central"][ie] - e["e3_err"][ie]))
        # The EW correction (the 1705.04664 table, copied into every payload) is a suppression at 300 GeV;
        # the QCD ratio there depends on the derivation (Run 2's file: 0.97; our UL18 one: ~1.3).
        assert e["central"][ie] < 1.0
        assert out["nominal"][0] == pytest.approx(q["central"][iq] * e["central"][ie])

    def test_missing_gen_z_keeps_weight_one(self, payload):
        out = evaluate(np.array([300.0, 300.0]), np.array([True, False]), payload)
        assert out["nominal"][1] == 1.0
        for up, down in out["variations"].values():
            assert up[1] == 1.0 and down[1] == 1.0


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

class TestDyZptWeight:
    def _dy_events(self, n=3):
        rows = [[(60.0 + 10 * i, 0.3, 0.0, 0.105, 13, 1, FROM_HARD),
                 (40.0, -0.3, 2.5, 0.105, -13, 1, FROM_HARD)] for i in range(n)]
        return _Events(_genpart(rows))

    def test_identity_for_non_dy_and_unconfigured_era(self):
        """Non-DY samples and eras without a payload get the identity; every analysis era has a payload since
        2026-10-11 (Run 3 included), so the unconfigured case is a made-up era."""
        ev = self._dy_events()
        for era, is_dy in ((UL18, False), (RUN3, False), (UNMAPPED, True), (UNMAPPED, False)):
            out = dy_zpt_weight(ev, era, is_dy=is_dy)
            assert not out["applied"]
            np.testing.assert_array_equal(out["nominal"], 1.0)
            assert set(out["variations"]) == set(SOURCE_NAMES)
        assert dy_zpt_weight(ev, RUN3, is_dy=True)["applied"]

    def test_applied_for_dy_on_ul18(self):
        ev = self._dy_events()
        out = dy_zpt_weight(ev, UL18, is_dy=True)
        assert out["applied"] and len(out["nominal"]) == 3
        assert not np.allclose(out["nominal"], 1.0)

    def test_dy_without_genpart_is_an_error(self):
        class NoGen:
            def __len__(self):
                return 2
        with pytest.raises(RuntimeError, match="GenPart"):
            dy_zpt_weight(NoGen(), UL18, is_dy=True)

    def test_identity_helper(self):
        out = identity_weights(4)
        assert out["nominal"].shape == (4,) and not out["applied"]


# ---------------------------------------------------------------------------
# Registry and analyzer wiring
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_family_expands_to_six_weight_sources(self):
        sources = systematics_for(UL18, ["dyzpt"])
        assert [s.name for s in sources] == list(SOURCE_NAMES)
        assert all(s.kind == "weight" for s in sources)
        assert all(s.available(UL18) and s.available(RUN3) and not s.available(UNMAPPED) for s in sources)

    def test_unmapped_era_cannot_produce_it(self):
        with pytest.raises(ValueError, match="Cannot produce"):
            validate_requested(UNMAPPED, ["dyzpt"])
        assert expected_labels(UNMAPPED, ["dyzpt"], "wr_ee_resolved_sr", is_mc=True) == []
        assert expected_labels(RUN3, ["dyzpt"], "wr_ee_resolved_sr", is_mc=True) != []       # mapped since 2026-10-11

    def test_labels_expected_in_every_region_for_ul18(self):
        labels = expected_labels(UL18, ["dyzpt"], "wr_mumu_boosted_sr", is_mc=True)
        assert "DYZptQcdScaleUp" in labels and "DYZptEw3Down" in labels
        assert expected_labels(UL18, ["dyzpt"], "wr_mumu_boosted_sr", is_mc=False) == []


class TestAnalyzerWiring:
    N = 5

    def _events(self):
        rows = [[(100.0 + 20 * i, 0.2, 0.0, 0.0005, 11, 1, FROM_HARD),
                 (50.0, -0.4, 2.8, 0.0005, -11, 1, FROM_HARD)] for i in range(self.N)]
        ev = _Events(_genpart(rows))
        ev.genWeight = ak.Array(np.ones(self.N))
        return ev

    # A dataset of the sample the UL18 payload was derived for (dy_reweight_datasets).
    LO_HT = "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8"
    NLO = "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8"

    def _meta(self, group, sample=LO_HT):
        return {"era": UL18, "xsec": 10.0, "genEventSumw": 100.0, "sample": sample, "physics_group": group}

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_dy_nominal_includes_reweight_and_variations(self, mock_pu):
        ones = np.ones(self.N)
        mock_pu.return_value = (ones, ones, ones)
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyzpt"], region="resolved")
        ev = self._events()
        weights, syst = proc.build_event_weights(ev, self._meta(DY_PROCESS_GROUP), is_mc=True)
        expected = dy_zpt_weight(ev, UL18, is_dy=True)
        base = 10.0 * LUMIS[UL18] * 1000.0 / 100.0      # the era's luminosity from config.yaml (brilcalc, 59.52 since 2026-10-10)
        np.testing.assert_allclose(syst[NOMINAL], base * expected["nominal"], rtol=1e-9)
        for name in SOURCE_NAMES:
            up, down = expected["variations"][name]
            np.testing.assert_allclose(syst[f"{name}Up"], base * up, rtol=1e-9)
            np.testing.assert_allclose(syst[f"{name}Down"], base * down, rtol=1e-9)

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_non_dy_gets_identity_labels(self, mock_pu):
        ones = np.ones(self.N)
        mock_pu.return_value = (ones, ones, ones)
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyzpt"], region="resolved")
        _, syst = proc.build_event_weights(self._events(), self._meta("tt_tW"), is_mc=True)
        for name in SOURCE_NAMES:
            np.testing.assert_array_equal(syst[f"{name}Up"], syst[NOMINAL])
            np.testing.assert_array_equal(syst[f"{name}Down"], syst[NOMINAL])

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_disabled_and_unrequested(self, mock_pu):
        ones = np.ones(self.N)
        mock_pu.return_value = (ones, ones, ones)
        ev = self._events()
        off = WrAnalysis(mass_point=None, enabled_systs=[], region="resolved", dy_zpt_reweight=False)
        _, syst_off = off.build_event_weights(ev, self._meta(DY_PROCESS_GROUP), is_mc=True)
        np.testing.assert_allclose(syst_off[NOMINAL], 10.0 * LUMIS[UL18] * 1000.0 / 100.0)
        assert set(syst_off) == {NOMINAL}
        on = WrAnalysis(mass_point=None, enabled_systs=[], region="resolved")
        _, syst_on = on.build_event_weights(ev, self._meta(DY_PROCESS_GROUP), is_mc=True)
        assert set(syst_on) == {NOMINAL}
        assert not np.allclose(syst_on[NOMINAL], syst_off[NOMINAL])



class TestDySampleGuard:
    """The DY Z-pT reweight and reshape reach only the DY sample they were derived for."""

    LO_HT = TestAnalyzerWiring.LO_HT
    NLO = TestAnalyzerWiring.NLO

    @pytest.fixture(autouse=True)
    def _fresh_warnings(self):
        from wrcoffea import analyzer
        stale = {k for k in analyzer._WARN_ONCE_ANALYZER if k.startswith("dy_reweight_not_for_sample")}
        analyzer._WARN_ONCE_ANALYZER.difference_update(stale)
        yield

    @pytest.mark.parametrize("group, sample, era, expected", [
        (DY_PROCESS_GROUP, LO_HT, UL18, True),
        (DY_PROCESS_GROUP, NLO, UL18, False),
        (DY_PROCESS_GROUP, "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8", UL18, False),  # LO inclusive
        ("tt_tW", LO_HT, UL18, False),
        (DY_PROCESS_GROUP, "DYto2L-4Jets_Bin-HT-100to400-MLL-50to120_TuneCP5_13p6TeV_madgraphMLM-pythia8",
         "RunIII2024Summer24", True),
        (DY_PROCESS_GROUP, "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8",
         "RunIII2024Summer24", False),
        (DY_PROCESS_GROUP, LO_HT, "EraWithoutPattern", False),
    ])
    def test_target(self, group, sample, era, expected):
        meta = {"era": era, "physics_group": group, "sample": sample}
        assert WrAnalysis._dy_reweight_target(meta) is expected

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_nlo_dy_keeps_its_weight(self, mock_pu, caplog):
        wiring = TestAnalyzerWiring()
        ones = np.ones(wiring.N)
        mock_pu.return_value = (ones, ones, ones)
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyzpt"], region="resolved")
        with caplog.at_level("WARNING"):
            _, syst = proc.build_event_weights(wiring._events(), wiring._meta(DY_PROCESS_GROUP, self.NLO),
                                               is_mc=True)
        np.testing.assert_allclose(syst[NOMINAL], 10.0 * LUMIS[UL18] * 1000.0 / 100.0)
        for name in SOURCE_NAMES:
            np.testing.assert_array_equal(syst[f"{name}Up"], syst[NOMINAL])
        assert "not the sample" in caplog.text

    @pytest.mark.parametrize("sample, status", [(LO_HT, "applied"), (NLO, "not_for_sample")])
    def test_status_recorded(self, sample, status):
        from tests.test_analyzer import _SF_PATCHES, _make_integration_events
        from contextlib import ExitStack
        events = _make_integration_events(n_events=3, metadata={"sample": sample})
        proc = WrAnalysis(mass_point=None, region="resolved")
        with ExitStack() as stack:
            for target, replacement in _SF_PATCHES.items():
                stack.enter_context(patch(target, new=replacement))
            output = proc.process(events)[sample]
        assert output["dataset_meta"]["dy_zpt_reweight"] == {status}

    def test_payload_requires_dataset_pattern(self):
        from wrcoffea.analysis_config import check_dy_reweight_datasets
        with pytest.raises(ValueError, match="EraX"):
            check_dy_reweight_datasets({"EraX": "zpt.json"}, {}, {})
        with pytest.raises(ValueError, match="EraY"):
            check_dy_reweight_datasets({}, {"EraY": "reshape.json"}, {"EraX": "^DY"})
        with pytest.raises(ValueError, match="dy_reweight_datasets"):
            check_dy_reweight_datasets({"EraX": "zpt.json"}, {}, {"EraX": "(unclosed"})
        check_dy_reweight_datasets({"EraX": "zpt.json"}, {}, {"EraX": "^DY", "EraZ": "^DY"})
