"""UL18 electron trigger SF from the ported Run 2 HEEP trigger efficiencies.

The payload is a correctionlib set with the Run 3 EGM HLT signature built by
``scripts/convert_run2_electron_trigger_sf.py``.  These tests pin the lookup
to the numbers in the Run 2 ``WREGammaTriggerEff.root`` histograms, the Run 2
pT clamp, and the OR-formula combination in ``electron_trigger_sf``.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
import correctionlib

from wrcoffea.analysis_config import ELECTRON_HLT_PATHS, ELECTRON_JSONS
from wrcoffea.scale_factors import electron_trigger_sf

UL18 = "RunIISummer20UL18"
PATH = "WREGammaTrigger"

# Values read off the Run 2 histograms (barrel row 0 < sceta < 1.444,
# endcap row 1.566 < sceta < 2.5).
BARREL_PT100 = {"data": 0.905, "data_err": 0.0092, "mc": 0.94}
BARREL_PT300_500 = {"data": 0.999, "mc": 1.0}   # bin [300, 500): the Run 2 clamp target for pT >= 500
ENDCAP_PT300_2000 = {"data": 0.998, "mc": 1.0}


@pytest.fixture(scope="module")
def cset():
    return correctionlib.CorrectionSet.from_file(ELECTRON_JSONS[UL18]["TRIGGER"])


class TestPayload:
    def test_signature_matches_run3_payloads(self, cset):
        for name in ("Electron-HLT-DataEff", "Electron-HLT-McEff", "Electron-HLT-SF"):
            assert [i.name for i in cset[name].inputs] == ["year", "ValType", "Path", "eta", "pt"]
        assert ELECTRON_HLT_PATHS[UL18] == PATH

    def test_barrel_values(self, cset):
        d, m = cset["Electron-HLT-DataEff"], cset["Electron-HLT-McEff"]
        assert d.evaluate("2018", "nom", PATH, 0.5, 100.0) == pytest.approx(BARREL_PT100["data"])
        assert d.evaluate("2018", "up", PATH, 0.5, 100.0) == pytest.approx(BARREL_PT100["data"] + BARREL_PT100["data_err"], abs=2e-4)
        assert d.evaluate("2018", "down", PATH, 0.5, 100.0) == pytest.approx(BARREL_PT100["data"] - BARREL_PT100["data_err"], abs=2e-4)
        assert m.evaluate("2018", "nom", PATH, 0.5, 100.0) == pytest.approx(BARREL_PT100["mc"])
        # MC efficiencies carry no error in the Run 2 file.
        assert m.evaluate("2018", "up", PATH, 0.5, 100.0) == pytest.approx(BARREL_PT100["mc"])

    def test_run2_pt_clamp(self, cset):
        """pT >= 500 is evaluated in the [300, 500) barrel bin, as HNWRAnalyzer did (pt -> 499)."""
        d = cset["Electron-HLT-DataEff"]
        assert d.evaluate("2018", "nom", PATH, 0.5, 1000.0) == pytest.approx(BARREL_PT300_500["data"])
        assert d.evaluate("2018", "nom", PATH, 0.5, 499.0) == pytest.approx(BARREL_PT300_500["data"])
        assert d.evaluate("2018", "nom", PATH, 2.0, 1000.0) == pytest.approx(ENDCAP_PT300_2000["data"])
        # Below 50 GeV the first bin is used.
        assert d.evaluate("2018", "nom", PATH, 0.5, 30.0) == d.evaluate("2018", "nom", PATH, 0.5, 52.0)

    def test_crack_rows_are_empty(self, cset):
        assert cset["Electron-HLT-DataEff"].evaluate("2018", "nom", PATH, 1.5, 100.0) == 0.0


class TestEventScaleFactor:
    @staticmethod
    def _electrons(rows):
        return ak.zip({
            "pt": ak.Array([[r[0] for r in row] for row in rows]),
            "eta": ak.Array([[r[1] for r in row] for row in rows]),
            "deltaEtaSC": ak.Array([[0.0] * len(row) for row in rows]),
        })

    def test_single_electron_is_data_over_mc(self):
        nom, up, down = electron_trigger_sf(self._electrons([[(100.0, 0.5)]]), UL18)
        assert nom[0] == pytest.approx(BARREL_PT100["data"] / BARREL_PT100["mc"])
        assert up[0] > nom[0] > down[0]

    def test_two_electrons_use_or_formula(self):
        nom, _, _ = electron_trigger_sf(self._electrons([[(100.0, 0.5), (1000.0, 2.0)]]), UL18)
        eff_data = 1 - (1 - BARREL_PT100["data"]) * (1 - ENDCAP_PT300_2000["data"])
        eff_mc = 1 - (1 - BARREL_PT100["mc"]) * (1 - ENDCAP_PT300_2000["mc"])
        assert nom[0] == pytest.approx(eff_data / eff_mc)

    def test_no_electrons_is_unity(self):
        nom, up, down = electron_trigger_sf(self._electrons([[], [(100.0, 0.5)]]), UL18)
        assert nom[0] == up[0] == down[0] == 1.0
        assert nom[1] != 1.0
