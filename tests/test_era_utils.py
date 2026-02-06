"""Tests for wrcoffea.era_utils — era mapping and detail retrieval."""

import pytest

from wrcoffea.era_utils import ERA_MAPPING, get_era_details


class TestGetEraDetails:
    def test_run2_ul18(self):
        run, year, era = get_era_details("RunIISummer20UL18")
        assert run == "RunII"
        assert year == "2018"
        assert era == "RunIISummer20UL18"

    def test_run3_2024(self):
        run, year, era = get_era_details("RunIII2024Summer24")
        assert run == "Run3"
        assert year == "2024"
        assert era == "RunIII2024Summer24"

    def test_run3_22ee(self):
        run, year, era = get_era_details("Run3Summer22EE")
        assert run == "Run3"
        assert year == "2022"

    def test_unsupported_era_raises(self):
        with pytest.raises(ValueError, match="Unsupported era"):
            get_era_details("RunIV2030")

    def test_all_mapped_eras_resolve(self):
        for era_key in ERA_MAPPING:
            run, year, era = get_era_details(era_key)
            assert run
            assert year
            assert era == era_key
