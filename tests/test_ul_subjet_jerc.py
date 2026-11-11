"""scripts/ul_subjet_jerc.py: the JME text files of the AK4 PUPPI corrections as correctionlib payloads."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

correctionlib = pytest.importorskip("correctionlib")

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ul_subjet_jerc", REPO / "scripts" / "ul_subjet_jerc.py")
conv = importlib.util.module_from_spec(spec)
sys.modules["ul_subjet_jerc"] = conv
spec.loader.exec_module(conv)


def _evaluator(name, table, **kwargs):
    cset = {"schema_version": 2, "corrections": [conv.correction(name, "test", table, **kwargs)]}
    return correctionlib.CorrectionSet.from_string(conv.validated(cset))[name]


FORMULA_TABLE = """
# a comment
{1 JetEta 1 JetPt [0]+[1]*TMath::Log(x) Correction L2Relative}
 -2.5  0.0   4   10  1000   1.0  0.1
  0.0  2.5   4   20   500   2.0  0.2
"""


def test_formula_is_clamped_to_its_pt_range_and_one_outside_the_table():
    (table,) = conv.parse_tables(FORMULA_TABLE)
    assert table.bin_vars == ["JetEta"] and table.par_vars == ["JetPt"] and table.level == "L2Relative"
    corr = _evaluator("c", table)
    assert [i.name for i in corr.inputs] == ["JetEta", "JetPt"]
    assert corr.evaluate(-1.0, 100.0) == pytest.approx(1.0 + 0.1 * np.log(100.0))
    assert corr.evaluate(1.0, 100.0) == pytest.approx(2.0 + 0.2 * np.log(100.0))
    assert corr.evaluate(-1.0, 1.0) == pytest.approx(1.0 + 0.1 * np.log(10.0))        # pT below its range
    assert corr.evaluate(1.0, 9000.0) == pytest.approx(2.0 + 0.2 * np.log(500.0))     # and above
    assert corr.evaluate(3.0, 100.0) == 1.0                                           # eta outside the table


def test_two_bin_variables_and_formula_variables_keep_their_names():
    text = """{2 JetEta Rho 1 JetPt sqrt([0]*abs([0])/(x*x)+[1]*[1]) Resolution}
    -1 1  0 10   4  15 3000  2.0 0.05
    -1 1 10 90   4  15 3000  4.0 0.06
    """
    (table,) = conv.parse_tables(text)
    corr = _evaluator("r", table)
    assert [i.name for i in corr.inputs] == ["JetEta", "JetPt", "Rho"]
    assert corr.evaluate(0.0, 100.0, 5.0) == pytest.approx(np.sqrt(4.0 / 1e4 + 0.05 ** 2))
    assert corr.evaluate(0.0, 100.0, 50.0) == pytest.approx(np.sqrt(16.0 / 1e4 + 0.06 ** 2))
    # exp, max and x are told apart when the variable is clamped
    (other,) = conv.parse_tables("{1 JetEta 1 JetPt max(0.5,exp(-x/[0])) Correction L2Relative}\n -5 5 3 10 100 50.0\n")
    assert _evaluator("e", other).evaluate(0.0, 20.0) == pytest.approx(np.exp(-20.0 / 50.0))


UNCERTAINTY_SOURCES = """
#Uncertainty sources
[AbsoluteStat]
{1 JetEta 1 JetPt "" Correction JECSource}
-2.0 0.0 6 10.0 0.02 0.02 100.0 0.01 0.01
0.0 2.0 6 10.0 0.04 0.04 100.0 0.02 0.02
[Total]
{1 JetEta 1 JetPt "" Correction JECSource}
-2.0 2.0 9 10.0 0.05 0.05 100.0 0.03 0.03 1000.0 0.03 0.03
"""


def test_uncertainty_sources_interpolate_in_pt():
    stat, total = conv.parse_tables(UNCERTAINTY_SOURCES)
    assert (stat.section, total.section) == ("AbsoluteStat", "Total")
    corr = _evaluator("u", stat)
    assert corr.evaluate(-1.0, 55.0) == pytest.approx(0.015)
    assert corr.evaluate(1.0, 10.0) == pytest.approx(0.04)
    assert corr.evaluate(1.0, 5.0) == pytest.approx(0.04) and corr.evaluate(1.0, 5000.0) == pytest.approx(0.02)
    assert corr.evaluate(3.0, 55.0) == -999.0
    assert _evaluator("t", total).evaluate(0.0, 500.0) == pytest.approx(0.03)


def test_data_periods_are_binned_in_run():
    tables = [conv.parse_tables(f"{{1 JetEta 1 JetPt [0] Correction L2Relative}}\n -5 5 3 10 6500 {value}\n")[0]
              for value in (1.01, 1.02)]
    corr = _evaluator("d", tables[0], run_edges=[100.0, 200.0, 300.0], period_tables=tables)
    assert [i.name for i in corr.inputs] == ["run", "JetEta", "JetPt"]
    assert corr.evaluate(150.0, 0.0, 50.0) == pytest.approx(1.01)
    assert corr.evaluate(250.0, 0.0, 50.0) == pytest.approx(1.02)
    with pytest.raises(Exception):
        corr.evaluate(400.0, 0.0, 50.0)


def test_symbolic_links_are_followed():
    store = {
        "textFiles/V3/V3_SF_AK4PFchs.txt": "V3_SF_AK4PFPuppi.txt",
        "textFiles/V3/V3_SF_AK4PFPuppi.txt": "../V2/V2_SF_AK4PFPuppi.txt\n",
        "textFiles/V2/V2_SF_AK4PFPuppi.txt": "{1 JetEta 1 JetPt [0] Correction L2Relative}\n -5 5 3 8 6500 1.1\n",
    }

    class Files(conv.TextFiles):
        def _fetch(self, repo, rev, path):
            return store[path]

    text = Files(None).get(("repo", "rev"), "V3", "V3_SF_AK4PFchs.txt")
    assert text == store["textFiles/V2/V2_SF_AK4PFPuppi.txt"]


@pytest.mark.parametrize("era", sorted(conv.ERAS))
def test_payload_agrees_with_the_official_file_where_jme_uses_one_text_file(era):
    """JME's AK4 PUPPI uncertainty sources and JER scale factors are its AK4 CHS text files, so the payload
    equals the official AK4 CHS file there; its L1FastJet and L3Absolute are 1."""
    jec_tag, _, jer_tag = conv.ERAS[era]
    mine = correctionlib.CorrectionSet.from_file(str(REPO / "data/jsonpog/JME/RunII" / era / "subjet_jerc.json.gz"))
    official = correctionlib.CorrectionSet.from_file(str(conv.official_file(era)))
    rng = np.random.default_rng(3)
    eta, pt = rng.uniform(-5.0, 5.0, 5000), np.exp(rng.uniform(np.log(10.0), np.log(2000.0), 5000))
    for name in (f"{jec_tag}_MC_Total", f"{jec_tag}_MC_AbsoluteStat", f"{jer_tag}_MC_ScaleFactor",
                 f"{jer_tag}_MC_SFUncertainty"):
        a, b = mine[f"{name}_AK4PFPuppi"], official[f"{name}_AK4PFchs"]
        assert [i.name for i in a.inputs] == [i.name for i in b.inputs] == ["JetEta", "JetPt"]
        np.testing.assert_allclose(a.evaluate(eta, pt), b.evaluate(eta, pt), rtol=0, atol=1e-9)
    rho, area = rng.uniform(0.0, 60.0, 5000), rng.uniform(0.2, 0.8, 5000)
    assert np.all(mine[f"{jec_tag}_MC_L1FastJet_AK4PFPuppi"].evaluate(area, eta, pt, rho) == 1.0)
    assert np.all(mine[f"{jec_tag}_MC_L3Absolute_AK4PFPuppi"].evaluate(eta, pt) == 1.0)
    l2 = mine[f"{jec_tag}_MC_L2Relative_AK4PFPuppi"].evaluate(eta, pt)
    assert np.all((l2 > 0.5) & (l2 < 3.0)) and np.std(l2) > 0.01
