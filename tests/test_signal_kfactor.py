"""Signal k(m_lN)/<k> reweighting (Run 2 HNWRAnalyzer port, AN-19-083 App. A)."""

from __future__ import annotations

import json
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
from coffea.nanoevents.methods import vector

from wrcoffea.analysis_config import LUMIS, SIGNAL_KFACTOR_JSONS
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.signal_kfactor import (
    SIGNAL_PROCESS_GROUP,
    evaluate,
    find_gen_ln,
    load_payload,
    mass_point_key,
    signal_kfactor_weight,
)
from wrcoffea.systematics import NOMINAL

UL18 = "RunIISummer20UL18"
RUN3 = "Run3Summer22"

# (pt, eta, phi, mass, pdgId, status, genPartIdxMother)
QUARK = (0.0, 0.0, 0.0, 0.0, 2, 21, -1)


def _genpart(rows):
    def col(i):
        return ak.Array([[r[i] for r in row] for row in rows])
    return ak.zip(
        {
            "pt": ak.values_astype(col(0), np.float64),
            "eta": ak.values_astype(col(1), np.float64),
            "phi": ak.values_astype(col(2), np.float64),
            "mass": ak.values_astype(col(3), np.float64),
            "pdgId": ak.values_astype(col(4), np.int32),
            "status": ak.values_astype(col(5), np.int32),
            "genPartIdxMother": ak.values_astype(col(6), np.int32),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )


class _Events:
    def __init__(self, gp):
        self.GenPart = gp
        # NanoAOD L1 prefiring record (unit weights): build_event_weights reads it for the Run 2 UL eras
        from types import SimpleNamespace
        ones = ak.Array([1.0] * len(gp))
        self.L1PreFiringWeight = SimpleNamespace(Nom=ones, Up=ones, Dn=ones)

    def __len__(self):
        return len(self.GenPart)


def _on_shell_event(m_wr=1000.0, n_mass=100.0, n_pid=9900012, extra_lep_first=False):
    """WR (mother quark) -> lepton + N, with the pair at rest so m(lN) is exact."""
    # Two-body decay at rest: lepton massless, N massive.
    p = (m_wr**2 - n_mass**2) / (2 * m_wr)
    lep_pid = 11 if n_pid == 9900012 else 13
    rows = [
        QUARK,                                            # 0
        (0.0, 0.0, 0.0, m_wr, 9900024, 22, 0),            # 1: WR from the quark
        (p, 0.0, 0.0, 0.0, -lep_pid, 23, 1),              # 2: primary lepton, mother WR
        (p, 0.0, np.pi, n_mass, n_pid, 22, 1),            # 3: N
        (50.0, 1.0, 1.0, 0.0, lep_pid, 1, 3),             # 4: secondary lepton from N
    ]
    if extra_lep_first:
        # A lepton earlier in the record whose mother is not the WR must be ignored.
        rows.insert(1, (30.0, 0.5, 0.5, 0.0, 13, 1, 0))
        rows = [rows[0], rows[1]] + [(r[0], r[1], r[2], r[3], r[4], r[5], r[6] + (1 if r[6] >= 1 else 0)) for r in rows[2:]]
    return rows


class TestGenFinder:
    def test_on_shell_mass_and_channel(self):
        m, ch, found = find_gen_ln(_Events(_genpart([_on_shell_event(1000.0, 100.0)])))
        assert found.tolist() == [True] and ch.tolist() == [9900012]
        assert m[0] == pytest.approx(1000.0, rel=1e-6)

    def test_primary_lepton_needs_wr_mother_when_wr_present(self):
        m, ch, found = find_gen_ln(_Events(_genpart([_on_shell_event(1000.0, 100.0, extra_lep_first=True)])))
        assert found.tolist() == [True]
        assert m[0] == pytest.approx(1000.0, rel=1e-6)

    def test_off_shell_takes_first_lepton(self):
        # No WR in the record: the first electron/muon is the primary lepton.
        p = (1500.0**2 - 400.0**2) / (2 * 1500.0)
        rows = [QUARK, (p, 0.0, 0.0, 0.0, -13, 23, 0), (p, 0.0, np.pi, 400.0, 9900014, 22, 0), (20.0, 0.0, 1.0, 0.0, 13, 1, 2)]
        m, ch, found = find_gen_ln(_Events(_genpart([rows])))
        assert found.tolist() == [True] and ch.tolist() == [9900014]
        assert m[0] == pytest.approx(1500.0, rel=1e-6)

    def test_missing_n_is_not_found(self):
        rows = [QUARK, (100.0, 0.0, 0.0, 0.0, 11, 1, 0)]
        m, ch, found = find_gen_ln(_Events(_genpart([rows, []])))
        assert found.tolist() == [False, False] and m.tolist() == [0.0, 0.0] and ch.tolist() == [0, 0]


RUN2_PAYLOAD = "data/reweights/RunII/2018/RunIISummer20UL18/signal_kfactor_run2legacy_v4_2018.json"


@pytest.fixture(scope="module")
def payload():
    # the Run 2 payload (kept for reference); the configured UL18 payload is tested in TestConfiguredPayload
    return load_payload(RUN2_PAYLOAD)


class TestConfiguredPayload:
    def test_per_point_curves_and_direct_k(self):
        raw = json.load(open(SIGNAL_KFACTOR_JSONS[UL18], encoding="utf-8"))
        pl = load_payload(SIGNAL_KFACTOR_JSONS[UL18])
        assert raw["direct_k"]["WR2000_N100"]["source"] == "direct" and 1.2 < raw["direct_k"]["WR2000_N100"]["k"] < 1.4
        assert "WR2000_N100" in pl["point_curves"] and "WR1000_N100" in pl["averages"]
        # the 490 private points (interpolated) and the 210 official ones all have an average and a curve
        for name in ("WR600_N100", "WR3400_N2600", "WR8000_N7900"):
            assert name in pl["point_curves"] and name in raw["direct_k"]
            assert pl["averages"][name].get("ee") or pl["averages"][name].get("mm")
        # a point's own curve integrates to its direct k: the sample average is within 2 % of it
        a = pl["averages"]["WR2000_N100"]
        assert a["mm"] == pytest.approx(raw["direct_k"]["WR2000_N100"]["k"], rel=0.02)

    def test_single_channel_average(self):
        pl = {"edges": np.array([0.0, 1000.0, 2000.0]), "values": np.array([1.2, 1.4]), "mln_min": 0.0,
              "averages": {"WR3200_N100": {"ee": 1.3}}, "point_curves": {}}
        w = evaluate(np.array([500.0, 1500.0]), np.array([9900012, 9900012]), np.array([True, True]), pl, "WR3200_N100")
        assert w.tolist() == pytest.approx([1.2 / 1.3, 1.4 / 1.3])


class TestEvaluate:
    def test_payload(self, payload):
        raw = json.load(open(RUN2_PAYLOAD, encoding="utf-8"))
        assert raw["provenance"]["source_commit"] == "06f274c754cff941c5e406b8f8d50bfe650e20a0"
        assert payload["edges"][0] == 0.0 and payload["edges"][-1] == 12000.0 and payload["edges"][1] == 200.0
        assert payload["averages"]["WR1000_N100"]["ee"] == pytest.approx(1.21258986224)
        assert payload["averages"]["WR1000_N100"]["source"] == "official"
        assert payload["averages"]["WR600_N100"]["source"] == "private"
        # AN Table 23: k = 1.21 for WR1000 (rounded), 1.18 for WR200.
        assert round(payload["averages"]["WR1000_N100"]["ee"], 2) == 1.21
        assert round(payload["averages"]["WR200_N100"]["ee"], 2) == 1.18

    def test_shape_weight_is_k_over_average(self, payload):
        m = np.array([1000.0, 5000.0])
        w = evaluate(m, np.array([9900012, 9900014]), np.array([True, True]), payload, "WR1000_N100")
        assert w[0] == pytest.approx(payload["values"][5] / payload["averages"]["WR1000_N100"]["ee"])
        assert w[1] == pytest.approx(payload["values"][25] / payload["averages"]["WR1000_N100"]["mm"])

    def test_low_mass_clamp_and_missing(self, payload):
        w = evaluate(np.array([50.0, 3000.0]), np.array([9900012, 9900012]), np.array([True, False]),
                     payload, "WR1000_N100")
        assert w[0] == pytest.approx(payload["values"][1] / payload["averages"]["WR1000_N100"]["ee"])
        assert w[1] == 1.0

    def test_unknown_mass_point(self, payload):
        with pytest.raises(KeyError, match="WR123_N45"):
            evaluate(np.array([1000.0]), np.array([9900012]), np.array([True]), payload, "WR123_N45")

    def test_point_curve_takes_precedence_over_the_shared_one(self, tmp_path):
        # a payload with a per-point curve for WR2000_N100 only: that point uses it, WR1000_N100 the shared curve
        raw = {"variable": "gen_mln",
               "kfactor": {"edges": [0.0, 1000.0, 2000.0, 3000.0], "values": [1.1, 1.2, 1.3], "mln_min": 0.0},
               "averages": {"WR1000_N100": {"ee": 1.2, "mm": 1.2}, "WR2000_N100": {"ee": 1.5, "mm": 1.5}},
               "point_curves": {"WR2000_N100": {"edges": [0.0, 1000.0, 2000.0, 3000.0], "values": [1.4, 1.5, 1.6],
                                                "mln_min": 1000.0}}}
        f = tmp_path / "payload.json"
        f.write_text(json.dumps(raw))
        pl = load_payload(str(f))
        assert set(pl["point_curves"]) == {"WR2000_N100"}
        ch, ok = np.array([9900012, 9900012]), np.array([True, True])
        w_shared = evaluate(np.array([1500.0, 50.0]), ch, ok, pl, "WR1000_N100")
        assert w_shared.tolist() == pytest.approx([1.2 / 1.2, 1.1 / 1.2])          # bin 1; clamp to the shared mln_min
        w_point = evaluate(np.array([1500.0, 50.0]), ch, ok, pl, "WR2000_N100")
        assert w_point.tolist() == pytest.approx([1.5 / 1.5, 1.5 / 1.5])           # bin 1; clamp to the point's mln_min


class TestMassPointKey:
    @pytest.mark.parametrize("name, key", [
        ("WRtoNLtoLLJJ_MWR1000_MN100", "WR1000_N100"),
        ("WR4000_N2100", "WR4000_N2100"),
        ("Official_FullSim_WR3000_N1500", "WR3000_N1500"),
        ("DYJetsToLL_M-50", None),
        (None, None),
    ])
    def test_parse(self, name, key):
        assert mass_point_key(name) == key


class TestGating:
    def _ev(self):
        return _Events(_genpart([_on_shell_event(1000.0, 100.0)] * 2))

    def test_identity_when_not_signal_or_unconfigured(self):
        # every real era has a payload since 2026-10-10; an era absent from the map gets the identity
        for era, is_sig in ((UL18, False), ("NoSuchEra", True)):
            out = signal_kfactor_weight(self._ev(), era, mass_point="WR1000_N100", is_signal=is_sig)
            assert not out["applied"] and np.all(out["nominal"] == 1.0)

    def test_applied(self):
        out = signal_kfactor_weight(self._ev(), UL18, mass_point="WR1000_N100", is_signal=True)
        assert out["applied"] and not np.allclose(out["nominal"], 1.0)

    def test_missing_mass_point_is_an_error(self):
        with pytest.raises(ValueError, match="mass point"):
            signal_kfactor_weight(self._ev(), UL18, mass_point=None, is_signal=True)

    def test_missing_genpart_is_an_error(self):
        class NoGen:
            def __len__(self):
                return 1
        with pytest.raises(RuntimeError, match="GenPart"):
            signal_kfactor_weight(NoGen(), UL18, mass_point="WR1000_N100", is_signal=True)


class TestAnalyzerWiring:
    N = 3

    def _events(self):
        ev = _Events(_genpart([_on_shell_event(1000.0, 100.0)] * self.N))
        ev.genWeight = ak.Array(np.ones(self.N))
        return ev

    def _meta(self, group, dataset="WRtoNLtoLLJJ_MWR1000_MN100"):
        return {"era": UL18, "xsec": 7.72343, "genEventSumw": 100.0, "sample": dataset,
                "dataset": dataset, "physics_group": group}

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_signal_nominal_includes_kfactor(self, mock_pu):
        ones = np.ones(self.N)
        mock_pu.return_value = (ones, ones, ones)
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="resolved")
        ev = self._events()
        _, syst = proc.build_event_weights(ev, self._meta(SIGNAL_PROCESS_GROUP), is_mc=True)
        expected = signal_kfactor_weight(ev, UL18, mass_point="WR1000_N100", is_signal=True)["nominal"]
        base = 7.72343 * LUMIS["RunIISummer20UL18"] * 1000.0 / 100.0
        np.testing.assert_allclose(syst[NOMINAL], base * expected, rtol=1e-9)
        assert set(syst) == {NOMINAL}

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_mass_point_from_cli_when_name_has_none(self, mock_pu):
        ones = np.ones(self.N)
        mock_pu.return_value = (ones, ones, ones)
        proc = WrAnalysis(mass_point="WR1000_N100", enabled_systs=[], region="resolved")
        _, syst = proc.build_event_weights(self._events(), self._meta(SIGNAL_PROCESS_GROUP, dataset="signal"), is_mc=True)
        assert not np.allclose(syst[NOMINAL], 7.72343 * LUMIS["RunIISummer20UL18"] * 1000.0 / 100.0)

    @patch("wrcoffea.analyzer.pileup_weight")
    def test_background_and_disabled_are_untouched(self, mock_pu):
        ones = np.ones(self.N)
        mock_pu.return_value = (ones, ones, ones)
        base = 7.72343 * LUMIS["RunIISummer20UL18"] * 1000.0 / 100.0
        bkg = WrAnalysis(mass_point=None, enabled_systs=[], region="resolved")
        _, syst = bkg.build_event_weights(self._events(), self._meta("tt_tW"), is_mc=True)
        np.testing.assert_allclose(syst[NOMINAL], base)
        off = WrAnalysis(mass_point=None, enabled_systs=[], region="resolved", signal_kfactor=False)
        _, syst_off = off.build_event_weights(self._events(), self._meta(SIGNAL_PROCESS_GROUP), is_mc=True)
        np.testing.assert_allclose(syst_off[NOMINAL], base)
