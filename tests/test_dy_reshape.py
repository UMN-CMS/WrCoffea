"""Data-driven DY reshape vs jet pT: derivation script, lookup, and analyzer wiring."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import awkward as ak
import hist
import numpy as np
import pytest
import uproot

from wrcoffea import dy_reshape as mod
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.dy_reshape import SOURCE_NAMES, dy_reshape_weight, evaluate, load_payload
from wrcoffea.systematics import NOMINAL, expected_labels, systematics_for, validate_requested

UL18 = "RunIISummer20UL18"
RUN3 = "Run3Summer22"
UNMAPPED = "RunXSummer99"   # no era maps a reshape payload under this name
ON = {"dy_zpt_reweight": True, "dy_reshape": True}   # both off by default since 2026-10-12
REPO = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("derive_dy_reshape", REPO / "scripts" / "derive_dy_reshape.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# Synthetic production: one coarse bin per kind carries all the yield
# ---------------------------------------------------------------------------

def _fine_hist(fill_at, weight, n_entries):
    h = hist.Hist.new.Reg(200, 0, 2000).Weight()
    h.fill(np.full(n_entries, fill_at), weight=np.full(n_entries, weight))
    return h


def _make_rootfiles(tmp_path):
    """Resolved: everything at 150 GeV (bin [100,200)); boosted: at 500 GeV (bin [400,600))."""
    spec = {
        # file: {channel: (data or MC yields per kind)}
        "WRAnalyzer_EGamma.root": {"ee": (1.0, 1000), "mumu": (0.0, 0)},
        "WRAnalyzer_Muon.root": {"ee": (0.0, 0), "mumu": (1.0, 2000)},
        "WRAnalyzer_DYJets_lo_ht_zpt.root": {"ee": (0.5, 1600), "mumu": (0.5, 3400)},   # 800 / 1700
        "WRAnalyzer_DYJets.root": {"ee": (1.0, 777), "mumu": (1.0, 777)},   # the default DY: ignored
        "WRAnalyzer_tt_tW.root": {"ee": (1.0, 100), "mumu": (1.0, 100)},
        "WRAnalyzer_Other.root": {"ee": (1.0, 50), "mumu": (1.0, 60)},
        "WRAnalyzer_signal_WR2000_N1000.root": {"ee": (1.0, 999), "mumu": (1.0, 999)},  # must be ignored
    }
    regions = {
        "resolved": ("pt_leading_jet", 150.0, {"ee": "wr_ee_resolved_dy_cr", "mumu": "wr_mumu_resolved_dy_cr"}),
        "boosted": ("pt_leading_AK8Jets", 500.0, {"ee": "wr_ee_boosted_dy_cr", "mumu": "wr_mumu_boosted_dy_cr"}),
    }
    for fname, chans in spec.items():
        with uproot.recreate(tmp_path / fname) as f:
            for var, at, regs in regions.values():
                for ch, region in regs.items():
                    w, n = chans[ch]
                    f[f"{region}/{var}_{region}"] = _fine_hist(at, w, n) if n else _fine_hist(at, 1.0, 0)
    return tmp_path


class TestDerivation:
    def test_ratio_combines_channels_and_ignores_signal(self, tmp_path):
        rootfiles = _make_rootfiles(tmp_path)
        s = _script()
        assert [p.name for p in s.background_files(rootfiles)] == ["WRAnalyzer_Other.root", "WRAnalyzer_tt_tW.root"]
        out = s.derive(rootfiles, UL18, note="synthetic")
        res = out["resolved"]
        assert res["edges"] == s.RESOLVED_EDGES
        i = 2  # [100, 200)
        # ee: (1000 - 150)/800 ; mumu: (2000 - 160)/1700 ; combined: (3000 - 310)/2500
        assert res["ee_ratio"][i] == pytest.approx(850 / 800)
        assert res["mumu_ratio"][i] == pytest.approx(1840 / 1700)
        assert res["ratio"][i] == pytest.approx(2690 / 2500)
        assert res["channel_err"][i] == pytest.approx(abs(850 / 800 - 1840 / 1700))
        # Stat error: data and MC weights are 1, DY weight 0.5 -> var_dy = 0.25 * N.
        num, dy = 2690.0, 2500.0
        var_dy = 0.25 * 5000
        expected = np.sqrt((3000 + 310) / dy**2 + num**2 * var_dy / dy**4)
        assert res["stat_err"][i] == pytest.approx(expected)
        # Empty bins are 1 +- 0 with no channel term.
        for j in (0, 1, 3):
            assert res["ratio"][j] == 1.0 and res["stat_err"][j] == 0.0 and res["channel_err"][j] == 0.0
        boo = out["boosted"]
        assert boo["edges"] == s.BOOSTED_EDGES and boo["ratio"][2] == pytest.approx(2690 / 2500)
        assert out["requires_dy_zpt_reweight"] is True
        assert "WRAnalyzer_signal_WR2000_N1000.root" not in out["provenance"]["background_files"]

    def test_summed_productions_give_the_same_ratio_with_smaller_errors(self, tmp_path):
        """A data-taking year = its sub-eras' productions summed before the ratio."""
        a, b = _make_rootfiles(tmp_path / "a"), _make_rootfiles(tmp_path / "b")
        s = _script()
        one = s.derive(a, "Run3Summer22")
        two = s.derive([a, b], "2022", dy_rootfiles=[a, b])
        assert two["resolved"]["ratio"][2] == pytest.approx(one["resolved"]["ratio"][2])
        assert two["resolved"]["stat_err"][2] == pytest.approx(one["resolved"]["stat_err"][2] / np.sqrt(2))
        assert two["resolved"]["yields"]["data"][2] == pytest.approx(2 * one["resolved"]["yields"]["data"][2])
        assert two["provenance"]["rootfiles"] == [str(a), str(b)] and one["provenance"]["rootfiles"] == str(a)
        (b / "WRAnalyzer_Other.root").unlink()
        with pytest.raises(ValueError, match="different non-DY background files"):
            s.derive([a, b], "2022")

    @pytest.mark.parametrize("reweight, ok", [("zpt", True), ("zpt+reshape", False), ("none", False)])
    def test_dy_record_checked(self, tmp_path, reweight, ok):
        rootfiles = _make_rootfiles(tmp_path)
        (rootfiles / "WRAnalyzer_DYJets_lo_ht_zpt_systematics.json").write_text(
            json.dumps({"dy": {"variant": "lo_ht", "reweight": reweight}}))
        s = _script()
        if ok:
            assert s.derive(rootfiles, UL18)["resolved"]["ratio"][2] == pytest.approx(2690 / 2500)
        else:
            with pytest.raises(ValueError, match="--dy-reweight zpt"):
                s.derive(rootfiles, UL18)

    def test_old_layout_dy_file(self, tmp_path, capsys):
        """Productions before 2026-10-12: the DY file of another directory, named WRAnalyzer_DYJets.root."""
        rootfiles = _make_rootfiles(tmp_path / "prod")
        old = tmp_path / "prod_noreshape"
        old.mkdir()
        (rootfiles / "WRAnalyzer_DYJets_lo_ht_zpt.root").rename(old / "WRAnalyzer_DYJets.root")
        out = _script().derive(rootfiles, UL18, dy_rootfiles=old, dy_file="WRAnalyzer_DYJets.root")
        assert out["resolved"]["ratio"][2] == pytest.approx(2690 / 2500)
        assert out["provenance"]["dy_file"] == "WRAnalyzer_DYJets.root"
        assert "no DY record" in capsys.readouterr().err

    def test_cli_writes_payload(self, tmp_path):
        rootfiles = _make_rootfiles(tmp_path / "prod")
        out = tmp_path / "dy_reshape.json"
        _script().main(["--era", UL18, "--rootfiles", str(rootfiles), "--out", str(out)])
        payload = load_payload(str(out))
        assert set(payload) == {"resolved", "boosted", "requires_dy_zpt_reweight"}
        assert payload["resolved"]["ratio"][2] == pytest.approx(2690 / 2500)


# ---------------------------------------------------------------------------
# Lookup and analyzer wiring on a synthetic payload
# ---------------------------------------------------------------------------

def _write_payload(path: Path):
    def block(edges, ratio):
        n = len(edges) - 1
        return {"variable": "x", "edges": edges, "ratio": ratio, "stat_err": [0.1] * n,
                "ee_ratio": [r + 0.05 for r in ratio], "mumu_ratio": [r - 0.05 for r in ratio],
                "channel_err": [0.1] * n}
    payload = {
        "requires_dy_zpt_reweight": True,
        "resolved": block([0.0, 40.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0],
                          [1.0, 1.2, 1.1, 1.0, 0.9, 0.85, 0.8, 0.7]),
        "boosted": block([0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0],
                         [1.0, 0.95, 0.9, 1.0, 0.6, 0.95]),
    }
    path.write_text(json.dumps(payload))
    return str(path)


@pytest.fixture
def configured(tmp_path):
    path = _write_payload(tmp_path / "reshape.json")
    mod._PAYLOAD_CACHE.clear()
    with patch("wrcoffea.dy_reshape.DY_RESHAPE_JSONS", {UL18: path}), \
         patch("wrcoffea.systematics.DY_RESHAPE_JSONS", {UL18: path}), \
         patch("wrcoffea.analyzer.DY_RESHAPE_JSONS", {UL18: path}):
        yield path
    mod._PAYLOAD_CACHE.clear()


class TestLookup:
    def test_bins_and_clamp(self, configured):
        block = load_payload(configured)["resolved"]
        out = evaluate(np.array([50.0, 150.0, 1999.0, 2500.0, 0.0]), block)
        assert out["nominal"].tolist() == pytest.approx([1.2, 1.1, 0.7, 0.7, 1.0])
        up, down = out["variations"]["DYReshapeStat"]
        assert up[1] == pytest.approx(1.2) and down[1] == pytest.approx(1.0)
        up, down = out["variations"]["DYReshapeChannel"]
        assert up[0] == pytest.approx(1.3) and down[0] == pytest.approx(1.1)

    def test_gating(self, configured):
        pt = np.array([150.0])
        assert dy_reshape_weight(pt, RUN3, "resolved", is_dy=True) is None
        ident = dy_reshape_weight(pt, UL18, "resolved", is_dy=False)
        assert not ident["applied"] and ident["nominal"].tolist() == [1.0]
        assert dy_reshape_weight(pt, UL18, "boosted", is_dy=True)["nominal"].tolist() == [1.0]
        with pytest.raises(ValueError, match="kind"):
            dy_reshape_weight(pt, UL18, "somewhere", is_dy=True)

    def test_unconfigured_era(self):
        assert dy_reshape_weight(np.array([1.0]), UNMAPPED, "resolved", is_dy=True) is None

    def test_ul18_payload_is_configured(self):
        from wrcoffea.analysis_config import DY_RESHAPE_JSONS
        block = load_payload(DY_RESHAPE_JSONS[UL18])["resolved"]
        assert block["edges"].tolist() == [0.0, 40.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 2000.0]
        assert block["ratio"][0] == 1.0 and block["ratio"][1] > block["ratio"][-1]


class TestRegistry:
    def test_family(self, configured):
        names = [s.name for s in systematics_for(UL18, ["dyreshape"])]
        assert names == list(SOURCE_NAMES)
        assert {s.name for s in validate_requested(UL18, ["dyreshape"])} == set(SOURCE_NAMES)
        with pytest.raises(ValueError, match="Cannot produce"):
            validate_requested(RUN3, ["dyreshape"])
        assert "DYReshapeStatUp" in expected_labels(UL18, ["dyreshape"], "wr_mumu_boosted_sr", is_mc=True)

    def test_unavailable_without_payload(self):
        # every analysis era maps a payload now (Run 3 ahead of its derivation); a made-up era stands in
        with pytest.raises(ValueError, match="Cannot produce"):
            validate_requested(UNMAPPED, ["dyreshape"])


class TestAnalyzerWiring:
    N = 3

    def _run(self, proc, kind, pt, group, syst=None):
        base = {NOMINAL: np.full(self.N, 2.0)} if syst is None else syst
        # Only the DY sample the payload was derived for is reshaped (_dy_reweight_target).
        return proc._dy_reshape_for_region(base, kind, np.asarray(pt, dtype=float), UL18,
                                           group == "DYJets", True)

    def test_dy_gets_ratio_and_variations(self, configured):
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyreshape"], region="both", **ON)
        out = self._run(proc, "resolved", [50.0, 150.0, 2500.0], "DYJets")
        np.testing.assert_allclose(out[NOMINAL], [2.4, 2.2, 1.4])
        np.testing.assert_allclose(out["DYReshapeStatUp"], [2.6, 2.4, 1.6])
        np.testing.assert_allclose(out["DYReshapeChannelDown"], [2.2, 2.0, 1.2])
        out = self._run(proc, "boosted", [500.0, 900.0, 3000.0], "DYJets")
        np.testing.assert_allclose(out[NOMINAL], [1.8, 1.2, 1.9])

    def test_non_dy_identity_with_labels(self, configured):
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyreshape"], region="both", **ON)
        out = self._run(proc, "resolved", [150.0] * self.N, "tt_tW")
        for name in SOURCE_NAMES:
            np.testing.assert_array_equal(out[f"{name}Up"], out[NOMINAL])
        np.testing.assert_allclose(out[NOMINAL], 2.0)

    def test_existing_sf_variations_are_scaled_too(self, configured):
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="both", **ON)
        base = {NOMINAL: np.ones(self.N), "MuonIdSfUp": np.full(self.N, 1.5)}
        out = self._run(proc, "resolved", [150.0] * self.N, "DYJets", syst=base)
        np.testing.assert_allclose(out["MuonIdSfUp"], 1.65)
        assert set(out) == {NOMINAL, "MuonIdSfUp"}

    def test_disabled_and_coupled_to_zpt(self, configured):
        base = {NOMINAL: np.ones(self.N)}
        for proc in (WrAnalysis(mass_point=None, region="both"),  # off by default since 2026-10-12
                     WrAnalysis(mass_point=None, region="both", dy_zpt_reweight=True, dy_reshape=False),
                     WrAnalysis(mass_point=None, region="both", dy_zpt_reweight=False, dy_reshape=True)):
            assert not proc._dy_reshape
            assert self._run(proc, "resolved", [150.0] * self.N, "DYJets", syst=base) is base

    def test_unconfigured_era_untouched(self):
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyreshape"], region="both", **ON)
        base = {NOMINAL: np.ones(self.N)}
        assert proc._dy_reshape_for_region(base, "resolved", np.ones(self.N), UNMAPPED, True, True) is base

    def test_other_dy_sample_gets_identity(self, configured):
        proc = WrAnalysis(mass_point=None, enabled_systs=["dyreshape"], region="both", **ON)
        base = {NOMINAL: np.full(self.N, 2.0)}
        out = proc._dy_reshape_for_region(base, "resolved", np.full(self.N, 150.0), UL18, False, True)
        np.testing.assert_allclose(out[NOMINAL], 2.0)
        for name in SOURCE_NAMES:
            np.testing.assert_array_equal(out[f"{name}Up"], out[NOMINAL])

    def test_dyreshape_family_needs_the_reshape(self):
        with pytest.raises(ValueError, match="dyreshape"):
            WrAnalysis(mass_point=None, enabled_systs=["dyreshape"], region="both", dy_zpt_reweight=True)

    def test_group_name_instead_of_flag_refused(self, configured):
        proc = WrAnalysis(mass_point=None, enabled_systs=[], region="both")
        with pytest.raises(TypeError, match="must be a bool"):
            proc._dy_reshape_for_region({NOMINAL: np.ones(self.N)}, "resolved", np.ones(self.N),
                                        UL18, "DYJets", True)
