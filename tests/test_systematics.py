"""Systematic registry, chunk coverage, and coverage-aware merging.

Acceptance checks agreed for the bookkeeping contract:

1. An evaluated ``10 -> 11`` plus an explicitly unaffected ``20`` gives a
   group nominal of 30 and Up of 31.
2. A label missing in one chunk fails even when another chunk produced it.
3. An evaluated zero-yield variation stays zero.
4. Chunk boundaries do not change contents, variances, or coverage, including
   a chunk with no affected objects (its nominal survives under the label).
5. Reduced and ``all`` observable coverage agree on their shared histograms.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import awkward as ak
import numpy as np
import pytest
import uproot
from coffea.processor import accumulate

from tests.test_analyzer import _apply_sf_patches, _make_integration_events
from wrcoffea.analysis_config import ELECTRON_JSONS, LUMI_UNC
from wrcoffea.analyzer import WrAnalysis
from wrcoffea.histograms import all_histogram_names, create_hist
from wrcoffea.regions import BOOSTED_REGIONS, RESOLVED_REGIONS, is_signal_region, region_names
from wrcoffea.save_hists import (
    STATUS_EVALUATED,
    STATUS_UNAFFECTED,
    STATUS_UNAVAILABLE,
    SystematicCoverageError,
    blinding_record,
    build_manifest,
    check_dataset_coverage,
    coverage_status,
    dataset_coverage,
    save_histograms,
    sum_hists,
    sum_hists_with_coverage,
)
from wrcoffea.systematics import (
    FAMILIES,
    FIT_OBSERVABLES,
    NOMINAL,
    NORMALIZATION,
    SHAPE,
    WEIGHT,
    expected_labels,
    parse_production_config,
    production_config,
    required_labels,
    resolve_syst_observables,
    systematics_for,
    validate_requested,
)

RUN3 = "Run3Summer22"
UL18 = "RunIISummer20UL18"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_every_family_expands_for_run3(self):
        sources = systematics_for(RUN3, FAMILIES)
        labels = {label for src in sources for label in src.labels()}
        for expected in ("PileupUp", "PileupDown", "JERUp", "JESUp", "EleScaleUp",
                         "EleSmearDown", "MuScaleUp", "MuSmearDown", "MuonIdSfUp",
                         "ElectronTrigSfDown", "LumiUp"):
            assert expected in labels
        kinds = {src.name: src.kind for src in sources}
        assert kinds["Lumi"] == NORMALIZATION
        assert kinds["Pileup"] == WEIGHT
        assert kinds["JES"] == SHAPE
        assert kinds["JES"] == SHAPE and next(s for s in sources if s.name == "JES").field_source == "CMS_scale_j_Total"

    def test_jes_sources_follow_config(self):
        (jes,) = [s for s in systematics_for(RUN3, ["jes"])]
        assert jes.variation_field("Up") == "pt_CMS_scale_j_Total_up"
        assert jes.objects == ("Jet", "FatJet")

    @pytest.mark.parametrize("family", ["elescale", "elesmear"])
    def test_ul18_cannot_produce_electron_shapes(self, family):
        with pytest.raises(ValueError, match="Cannot produce"):
            validate_requested(UL18, [family])

    @pytest.mark.parametrize("era", [UL18, "RunIII2025Summer25"])
    def test_musmear_needs_a_high_pt_muon_payload(self, era):
        """Without a muon_highpt payload the smearing variation would be a copy of the nominal: refused."""
        with pytest.raises(ValueError, match="Cannot produce"):
            validate_requested(era, ["musmear"])
        assert {s.name for s in validate_requested(era, ["muscale"])} == {"MuScale"}

    @pytest.mark.parametrize("era", ["Run3Summer22", "Run3Summer23BPix", "RunIII2024Summer24"])
    def test_musmear_in_the_high_pt_muon_eras(self, era):
        assert {s.name for s in validate_requested(era, ["musmear"])} == {"MuSmear"}

    def test_ul18_produces_jet_shapes(self):
        """UL18 has JRV2 and Total in its JERC payloads; jer/jes are available."""
        names = {s.name for s in validate_requested(UL18, ["jer", "jes"])}
        assert names == {"JER", "JES"}
        (jes,) = systematics_for(UL18, ["jes"])
        assert jes.available(UL18) and jes.variation_field("Up") == "pt_CMS_scale_j_Total_up"

    @staticmethod
    def _without_electron_trigger_payload():
        """Patch the registry's view of the electron payloads: no TRIGGER anywhere."""
        stripped = {era: {k: v for k, v in (cfg or {}).items() if k != "TRIGGER"}
                    for era, cfg in ELECTRON_JSONS.items()}
        return patch("wrcoffea.systematics.ELECTRON_JSONS", stripped)

    def test_ul18_sf_family_is_complete(self):
        """UL18 carries every lepton SF payload (the trigger one is the Run 2 HEEP port)."""
        names = {s.name for s in validate_requested(UL18, ["sf"])}
        assert {"ElectronTrigSf", "MuonTrigSf", "ElectronIdSf"} <= names

    def test_unavailable_component_must_be_omitted_explicitly(self):
        """A source without a payload for the era needs a recorded omission."""
        with self._without_electron_trigger_payload():
            with pytest.raises(ValueError, match="ElectronTrigSf.*--omit-systs ElectronTrigSf"):
                validate_requested(UL18, ["sf"])
            sources = validate_requested(UL18, ["lumi", "pileup", "sf", "muscale"],
                                         omitted=["ElectronTrigSf"])
            names = {s.name for s in sources}
            assert "ElectronTrigSf" not in names and "MuonTrigSf" in names
            with pytest.raises(ValueError, match="Unknown systematic source"):
                validate_requested(UL18, ["sf"], omitted=["NoSuchSf"])

    def test_required_labels_ignore_availability(self):
        """Required = applies and not omitted; availability is not a free pass."""
        with self._without_electron_trigger_payload():
            req = required_labels(UL18, ["sf"], "wr_ee_resolved_sr", is_mc=True)
            assert "ElectronTrigSfUp" in req
            assert "ElectronTrigSfUp" not in required_labels(UL18, ["sf"], "wr_ee_resolved_sr",
                                                              is_mc=True, omitted=["ElectronTrigSf"])
            assert "ElectronTrigSfUp" not in expected_labels(UL18, ["sf"], "wr_ee_resolved_sr", is_mc=True)
            assert "MuonIdSfUp" not in req

    def test_applies_is_physics_and_available_is_inputs(self):
        (jes,) = systematics_for(RUN3, ["jes"])
        assert jes.applies(region="wr_ee_resolved_sr", is_mc=True)
        assert not jes.applies(region="wr_ee_resolved_sr", is_mc=False)
        assert jes.available(RUN3) and jes.available(UL18)
        assert jes.required_fields("Up") == ("pt_CMS_scale_j_Total_up", "mass_CMS_scale_j_Total_up")

    def test_jet_mass_sources_vary_the_softdrop_mass_only(self):
        (jms,) = systematics_for(RUN3, ["jms"])
        (jmr,) = systematics_for(UL18, ["jmr"])
        assert jms.objects == ("FatJet",) and jms.required_fields("Down") == ("msoftdrop_jms_down",)
        assert jmr.variation_field("Up") == "msoftdrop_jmr_up" and jmr.available(UL18)
        assert jms.applies(region="wr_ee_resolved_sr", is_mc=True)   # the shape pass refills every region

    def test_shape_swap_takes_the_softdrop_mass(self):
        from types import SimpleNamespace
        from wrcoffea.analyzer import WrAnalysis
        fat = ak.zip({"pt": [[300.0]], "mass": [[90.0]], "msoftdrop": [[50.0]],
                      "pt_jer_up": [[310.0]], "mass_jer_up": [[93.0]], "msoftdrop_jer_up": [[52.0]],
                      "msoftdrop_jms_up": [[50.5]]})
        events = SimpleNamespace(FatJet=fat)
        (jer,) = systematics_for(RUN3, ["jer"])
        (jms,) = systematics_for(RUN3, ["jms"])
        out = WrAnalysis._apply_shape_variation(events, jer, "Up").FatJet
        assert ak.to_list(out.pt) == [[310.0]] and ak.to_list(out.msoftdrop) == [[52.0]]
        out = WrAnalysis._apply_shape_variation(events, jms, "Up").FatJet
        assert ak.to_list(out.pt) == [[300.0]] and ak.to_list(out.msoftdrop) == [[50.5]]

    def test_unknown_family_rejected(self):
        with pytest.raises(ValueError, match="Unknown systematic family"):
            systematics_for(RUN3, ["btag"])

    def test_data_never_has_variations(self):
        for region in list(RESOLVED_REGIONS) + list(BOOSTED_REGIONS):
            assert expected_labels(RUN3, FAMILIES, region, is_mc=False) == []

    def test_muon_placeholder_is_flagged(self):
        (src,) = systematics_for(RUN3, ["muscale"])
        assert "placeholder" in src.note

    def test_resolve_syst_observables(self):
        known = all_histogram_names()
        assert resolve_syst_observables(None, known) is None
        assert resolve_syst_observables([], known) is None
        assert resolve_syst_observables(["all"], known) is None
        assert resolve_syst_observables(["fit"], known) == frozenset(FIT_OBSERVABLES)
        assert resolve_syst_observables(["mass_dilepton"], known) == frozenset({"mass_dilepton"})
        with pytest.raises(ValueError, match="unknown observable"):
            resolve_syst_observables(["not_a_hist"], known)
        with pytest.raises(ValueError, match="cannot be combined"):
            resolve_syst_observables(["all", "mass_dilepton"], known)

    def test_sf_expectations_match_analyzer_labels(self):
        """Registry region rules must reproduce what _lepton_sfs_for_region emits."""
        n = 4
        ones = np.ones(n)
        triple = (ones, ones * 1.01, ones * 0.99)
        proc = WrAnalysis(mass_point=None, enabled_systs=["sf"], region="both")
        regions = list(RESOLVED_REGIONS) + list(BOOSTED_REGIONS)
        regions += [f"{r}_nomass" for r in ("wr_ee_resolved_sr", "wr_resolved_flavor_cr")]
        dummy = ak.Array([[1.0]] * n)
        with patch("wrcoffea.analyzer.muon_sf", return_value={"reco": triple, "id": triple, "iso": triple}), \
             patch("wrcoffea.analyzer.muon_sf_loose", return_value={"reco": triple, "id": triple}), \
             patch("wrcoffea.analyzer.muon_trigger_sf", return_value=triple), \
             patch("wrcoffea.analyzer.electron_reco_sf", return_value=triple), \
             patch("wrcoffea.analyzer.electron_id_sf", return_value=triple), \
             patch("wrcoffea.analyzer.electron_trigger_sf", return_value=triple):
            for region in regions:
                boosted = "boosted" in region
                result = proc._lepton_sfs_for_region(
                    region, {NOMINAL: ones.copy()}, RUN3, dummy, dummy,
                    loose_muons=dummy if boosted else None,
                    loose_electrons=dummy if boosted else None,
                    is_mc=True,
                )
                produced = set(result) - {NOMINAL}
                expected = set(expected_labels(RUN3, ["sf"], region, is_mc=True))
                assert produced == expected, region


# ---------------------------------------------------------------------------
# Synthetic payload helpers
# ---------------------------------------------------------------------------

def _hist(name="mass_dilepton"):
    return create_hist(name, (4, 0, 4), name)


def _payload(h, coverage, *, n_chunks=1, era=RUN3, datatype="mc", group="DYJets", name="mass_dilepton",
             families=("jes",), omitted=(), observables=None, region="resolved", variants=(),
             blind=False):
    """Synthetic dataset payload.

    Regions the configuration requires but *coverage* does not mention are
    filled in as Nominal-complete so the tests can focus on one region.  With
    *blind* (a blinded run), data leaves out the signal regions.
    """
    is_mc = datatype == "mc"
    full = {}
    for r in region_names(region, variants, blind=blind and not is_mc):
        full[r] = {NOMINAL: n_chunks}
        full[r].update({l: n_chunks for l in required_labels(era, families, r, is_mc=is_mc, omitted=omitted)})
    for r, labels in coverage.items():
        # The region under test carries exactly what the test says.
        full[r] = dict(labels)
    return {
        name: h,
        "syst_coverage": full,
        "n_chunks": n_chunks,
        "dataset_meta": {
            "era": {era}, "datatype": {datatype}, "physics_group": {group},
            "syst_config": {production_config(families, omitted, observables,
                                              region=region, variants=variants, blind=blind)},
        },
    }


def _fill(h, region, label, weight, process="DYJets"):
    h.fill(process=process, region=region, syst=label, mass_dilepton=1.0, weight=weight)


def _yield(summed, region, label, name="mass_dilepton"):
    h = summed[name]
    if label not in h.axes["syst"] or region not in h.axes["region"]:
        return 0.0
    return h[{"region": region, "syst": label}].sum(flow=True).value


# ---------------------------------------------------------------------------
# Coverage status and merge rule
# ---------------------------------------------------------------------------

class TestMergeRule:
    REGION = "wr_ee_resolved_sr"

    def _evaluated_a(self, up=11.0, down=9.0):
        a = _hist()
        _fill(a, self.REGION, NOMINAL, 10.0)
        _fill(a, self.REGION, "JESUp", up)
        _fill(a, self.REGION, "JESDown", down)
        return _payload(a, {self.REGION: {NOMINAL: 1, "JESUp": 1, "JESDown": 1}})

    def test_unaffected_dataset_contributes_nominal(self):
        """10 -> 11 plus an explicitly unaffected 20 gives nominal 30, Up 31."""
        region = self.REGION
        b = _hist()
        _fill(b, region, NOMINAL, 20.0, process="Observed")
        # Variations never apply to data, so the data dataset is explicitly
        # unaffected and contributes its nominal to every variation.
        hists = {
            "A": self._evaluated_a(),
            "B": _payload(b, {region: {NOMINAL: 1}}, datatype="data", group="Observed"),
        }
        summed, report = sum_hists_with_coverage(hists, RUN3, ["jes"])
        assert _yield(summed, region, NOMINAL) == pytest.approx(30.0)
        assert _yield(summed, region, "JESUp") == pytest.approx(31.0)
        assert _yield(summed, region, "JESDown") == pytest.approx(29.0)
        assert report.unaffected[region]["JESUp"] == ["B"]
        assert coverage_status(dataset_coverage("B", hists["B"]), region, "JESUp") == STATUS_UNAFFECTED
        assert coverage_status(dataset_coverage("A", hists["A"]), region, "JESUp") == STATUS_EVALUATED
        # Inputs are not mutated by the substitution.
        assert "JESUp" not in b.axes["syst"]

    def test_entirely_missing_variation_fails(self):
        """A requested source nobody produced is an error, not silent coverage."""
        a = _hist()
        _fill(a, self.REGION, NOMINAL, 10.0)
        payload = _payload(a, {self.REGION: {NOMINAL: 1}})
        with pytest.raises(SystematicCoverageError, match="'JES(Up|Down)' completed in 0 of 1"):
            check_dataset_coverage(dataset_coverage("A", payload), ["jes"])
        with pytest.raises(SystematicCoverageError):
            sum_hists_with_coverage({"A": payload}, RUN3, ["jes"])

    def test_missing_down_direction_fails(self):
        a = _hist()
        _fill(a, self.REGION, NOMINAL, 10.0)
        _fill(a, self.REGION, "JESUp", 11.0)
        payload = _payload(a, {self.REGION: {NOMINAL: 1, "JESUp": 1}})
        with pytest.raises(SystematicCoverageError, match="'JESDown' completed in 0 of 1"):
            sum_hists_with_coverage({"A": payload}, RUN3, ["jes"])

    def test_label_missing_in_one_chunk_fails(self):
        region = self.REGION
        a = _hist()
        _fill(a, region, NOMINAL, 10.0)
        _fill(a, region, "JESUp", 5.0)
        _fill(a, region, "JESDown", 5.0)
        payload = _payload(a, {region: {NOMINAL: 2, "JESUp": 1, "JESDown": 2}}, n_chunks=2)
        with pytest.raises(SystematicCoverageError, match="'JESUp' completed in 1 of 2 chunks"):
            check_dataset_coverage(dataset_coverage("A", payload), ["jes"])
        with pytest.raises(SystematicCoverageError):
            sum_hists_with_coverage({"A": payload}, RUN3, ["jes"])

    def test_nominal_missing_in_one_chunk_fails(self):
        payload = _payload(_hist(), {self.REGION: {NOMINAL: 1}}, n_chunks=2, families=())
        with pytest.raises(SystematicCoverageError, match="Nominal filled in 1 of 2"):
            check_dataset_coverage(dataset_coverage("A", payload), [])

    def test_required_but_never_evaluated_dataset_fails(self):
        region = self.REGION
        b = _hist()
        _fill(b, region, NOMINAL, 20.0)
        hists = {"A": self._evaluated_a(), "B": _payload(b, {region: {NOMINAL: 1}})}
        assert coverage_status(dataset_coverage("B", hists["B"]), region, "JESUp") == STATUS_UNAVAILABLE
        with pytest.raises(SystematicCoverageError, match="'B'.*'JES(Up|Down)' completed in 0 of 1"):
            sum_hists_with_coverage(hists, RUN3, ["jes"])

    def test_unavailable_inputs_are_not_unaffected(self):
        """A UL18 dataset cannot evaluate JES; that is an error, not a nominal copy."""
        region = self.REGION
        b = _hist()
        _fill(b, region, NOMINAL, 20.0)
        hists = {"A": self._evaluated_a(), "B": _payload(b, {region: {NOMINAL: 1}}, era=UL18)}
        assert coverage_status(dataset_coverage("B", hists["B"]), region, "JESUp") == STATUS_UNAVAILABLE
        with pytest.raises(SystematicCoverageError):
            sum_hists_with_coverage(hists, RUN3, ["jes"])

    def test_omitted_source_is_recorded_not_required(self):
        region = "wr_ee_resolved_sr"
        a = _hist()
        _fill(a, region, NOMINAL, 10.0)
        cov = {region: {NOMINAL: 1, **{l: 1 for l in ("ElectronRecoSfUp", "ElectronRecoSfDown",
                                                       "ElectronIdSfUp", "ElectronIdSfDown")}}}
        for label in list(cov[region])[1:]:
            _fill(a, region, label, 1.0)
        payload = _payload(a, cov, era=UL18, families=("sf",), omitted=("ElectronTrigSf",))
        check_dataset_coverage(dataset_coverage("A", payload), ["sf"])
        assert coverage_status(dataset_coverage("A", payload), region, "ElectronTrigSfUp") == "omitted"
        # Without the recorded omission the same payload is incomplete.
        bad = _payload(a, cov, era=UL18, families=("sf",))
        with pytest.raises(SystematicCoverageError, match="'ElectronTrigSf(Up|Down)' completed in 0 of 1"):
            check_dataset_coverage(dataset_coverage("A", bad), ["sf"])

    def test_unexpected_label_fails(self):
        region = self.REGION
        a = _hist()
        _fill(a, region, NOMINAL, 1.0)
        payload = _payload(a, {region: {NOMINAL: 1, "MuonIdSfUp": 1}}, families=("sf",))
        with pytest.raises(SystematicCoverageError, match="does not expect them"):
            check_dataset_coverage(dataset_coverage("A", payload), ["sf"])

    def test_families_must_match_production(self):
        with pytest.raises(SystematicCoverageError, match="produced with systematic families"):
            check_dataset_coverage(dataset_coverage("A", self._evaluated_a()), ["pileup"])

    def test_evaluated_zero_yield_stays_zero(self):
        region = self.REGION
        a = _hist()
        _fill(a, region, NOMINAL, 10.0)
        # Evaluated but nothing selected: labels exist, content is zero.
        for label in ("JESUp", "JESDown"):
            a.fill(process="DYJets", region=region, syst=label, mass_dilepton=np.array([]), weight=np.array([]))
        summed, report = sum_hists_with_coverage(
            {"A": _payload(a, {region: {NOMINAL: 1, "JESUp": 1, "JESDown": 1}})}, RUN3, ["jes"],
        )
        assert _yield(summed, region, NOMINAL) == pytest.approx(10.0)
        assert _yield(summed, region, "JESUp") == 0.0
        assert report.produced[region] == ["JESDown", "JESUp"]
        assert report.unaffected == {}

    def test_data_dataset_is_unaffected_by_everything(self):
        region = self.REGION
        mc = _hist()
        _fill(mc, region, NOMINAL, 1.0)
        _fill(mc, region, "PileupUp", 2.0)
        _fill(mc, region, "PileupDown", 0.5)
        data = _hist()
        _fill(data, region, NOMINAL, 7.0, process="Observed")
        hists = {
            "MC": _payload(mc, {region: {NOMINAL: 1, "PileupUp": 1, "PileupDown": 1}}, families=("pileup",)),
            "Data": _payload(data, {region: {NOMINAL: 1}}, datatype="data", group="Observed", families=("pileup",)),
        }
        summed, report = sum_hists_with_coverage(hists, RUN3, ["pileup"])
        assert _yield(summed, region, "PileupUp") == pytest.approx(9.0)
        assert report.unaffected[region]["PileupUp"] == ["Data"]

    def test_missing_region_fails(self):
        """A region the run was configured for must be present in every chunk."""
        a = _hist()
        _fill(a, self.REGION, NOMINAL, 1.0)
        payload = _payload(a, {self.REGION: {NOMINAL: 1}}, families=(), region="both")
        for region in list(BOOSTED_REGIONS):
            payload["syst_coverage"].pop(region)
        with pytest.raises(SystematicCoverageError, match="never filled \\['wr_ee_boosted_dy_cr'"):
            check_dataset_coverage(dataset_coverage("A", payload), [])
        # A resolved-only configuration does not ask for them.
        payload = _payload(a, {self.REGION: {NOMINAL: 1}}, families=(), region="resolved")
        check_dataset_coverage(dataset_coverage("A", payload), [])
        assert set(payload["syst_coverage"]) == set(RESOLVED_REGIONS)
        # Variants are part of the requirement too.
        payload = _payload(a, {self.REGION: {NOMINAL: 1}}, families=(), region="resolved", variants=("nomass",))
        assert "wr_ee_resolved_sr_nomass" in payload["syst_coverage"]
        payload["syst_coverage"].pop("wr_ee_resolved_sr_nomass")
        with pytest.raises(SystematicCoverageError, match="never filled"):
            check_dataset_coverage(dataset_coverage("A", payload), [])

    def test_region_outside_configuration_fails(self):
        a = _hist()
        payload = _payload(a, {"wr_ee_boosted_sr": {NOMINAL: 1}}, families=(), region="resolved")
        with pytest.raises(SystematicCoverageError, match="does not include"):
            check_dataset_coverage(dataset_coverage("A", payload), [])

    def test_different_production_configs_cannot_merge(self):
        region = self.REGION
        b = self._evaluated_a()
        b["dataset_meta"]["syst_config"] = {production_config(["jes"], [], frozenset({"mass_dilepton"}),
                                                              region="resolved")}
        with pytest.raises(SystematicCoverageError, match="different systematic configurations"):
            sum_hists_with_coverage({"A": self._evaluated_a(), "B": b}, RUN3, ["jes"])

    def test_legacy_payload_without_coverage_sums_blindly(self):
        region = self.REGION
        a = _hist()
        _fill(a, region, NOMINAL, 3.0)
        summed, report = sum_hists_with_coverage({"A": {"mass_dilepton": a}}, RUN3, [])
        assert not report.coverage_known
        assert _yield(summed, region, NOMINAL) == pytest.approx(_yield(sum_hists({"A": {"mass_dilepton": a}}), region, NOMINAL))

    def test_mixed_legacy_and_tracked_payloads_rejected(self):
        region = self.REGION
        a = _hist()
        _fill(a, region, NOMINAL, 3.0)
        hists = {"A": _payload(a.copy(), {region: {NOMINAL: 1}}, families=()), "B": {"mass_dilepton": a}}
        with pytest.raises(SystematicCoverageError, match="no systematic coverage records"):
            sum_hists_with_coverage(hists, RUN3, [])


# ---------------------------------------------------------------------------
# Blinding: production config, coverage, and the manifest record
# ---------------------------------------------------------------------------

def _data_payload(**kwargs):
    kwargs = {"families": (), "region": "both", "variants": ("nomass",), **kwargs}
    return _payload(_hist(), {}, datatype="data", group="Observed", **kwargs)


class TestBlinding:
    # The record the plotter reads from the manifest of a blinded data file.
    BLINDED_RECORD = {
        "applied": True,
        "signal_regions": [
            "wr_ee_boosted_sr", "wr_ee_boosted_sr_nomass",
            "wr_ee_resolved_sr", "wr_ee_resolved_sr_nomass",
            "wr_mumu_boosted_sr", "wr_mumu_boosted_sr_nomass",
            "wr_mumu_resolved_sr", "wr_mumu_resolved_sr_nomass",
        ],
        "cutflow_truncated_before": {
            "cutflow_resolved": {"ee": "mll_gt200", "mumu": "mll_gt200"},
            "cutflow_boosted": {"ee": "mll_gt200_boosted", "mumu": "mll_gt200_boosted"},
        },
    }
    NOT_APPLIED = {"applied": False, "signal_regions": [], "cutflow_truncated_before": {}}

    @pytest.mark.parametrize("blind", [False, True])
    def test_production_config_round_trip(self, blind):
        text = production_config(["jes"], ["ElectronTrigSf"], frozenset({"mass_dilepton"}),
                                 region="boosted", variants=("nomass",), blind=blind)
        assert json.loads(text)["blind"] is blind
        assert parse_production_config(text) == {
            "blind": blind,
            "families": frozenset({"jes"}),
            "omitted": frozenset({"ElectronTrigSf"}),
            "observables": frozenset({"mass_dilepton"}),
            "region": "boosted",
            "variants": frozenset({"nomass"}),
        }

    def test_production_config_defaults_to_unblinded(self):
        assert parse_production_config(production_config([], [], None))["blind"] is False
        # Different blinding never merges: the config strings differ.
        assert production_config([], [], None, blind=True) != production_config([], [], None)

    def test_legacy_config_without_blind_is_unblinded(self):
        legacy = json.loads(production_config(["jes"], [], None, region="resolved"))
        del legacy["blind"]
        assert parse_production_config(json.dumps(legacy))["blind"] is False
        payload = _data_payload(families=("jes",), region="resolved", variants=())
        payload["dataset_meta"]["syst_config"] = {json.dumps(legacy, sort_keys=True)}
        cov = dataset_coverage("Data", payload)
        assert not cov.blinded
        assert "wr_ee_resolved_sr" in cov.expected_regions()
        check_dataset_coverage(cov, ["jes"])

    def test_blinded_data_must_not_have_signal_regions(self):
        cov = dataset_coverage("Data", _data_payload(blind=True))
        assert cov.blinded
        assert cov.expected_regions() == region_names("both", ("nomass",), blind=True)
        assert set(cov.expected_regions()) == {
            r for r in region_names("both", ("nomass",)) if not is_signal_region(r)}
        assert set(cov.regions()) == set(cov.expected_regions())
        check_dataset_coverage(cov, [])

    @pytest.mark.parametrize("region", ["wr_mumu_resolved_sr", "wr_ee_boosted_sr_nomass"])
    def test_blinded_data_that_filled_a_signal_region_fails(self, region):
        payload = _data_payload(blind=True)
        payload["syst_coverage"][region] = {NOMINAL: 1}
        with pytest.raises(SystematicCoverageError,
                           match=f"blinded run but filled signal regions \\['{region}'\\]"):
            check_dataset_coverage(dataset_coverage("Data", payload), [])

    def test_unblinded_data_needs_every_region(self):
        payload = _data_payload()
        cov = dataset_coverage("Data", payload)
        assert not cov.blinded
        assert cov.expected_regions() == region_names("both", ("nomass",))
        check_dataset_coverage(cov, [])
        del payload["syst_coverage"]["wr_ee_resolved_sr_nomass"]
        with pytest.raises(SystematicCoverageError, match="never filled \\['wr_ee_resolved_sr_nomass'\\]"):
            check_dataset_coverage(dataset_coverage("Data", payload), [])

    def test_mc_of_a_blinded_run_needs_every_region(self):
        payload = _payload(_hist(), {}, families=(), region="both", variants=("nomass",), blind=True)
        cov = dataset_coverage("MC", payload)
        assert cov.config["blind"] and not cov.blinded
        assert cov.expected_regions() == region_names("both", ("nomass",))
        check_dataset_coverage(cov, [])
        del payload["syst_coverage"]["wr_mumu_resolved_sr"]
        with pytest.raises(SystematicCoverageError, match="never filled \\['wr_mumu_resolved_sr'\\]"):
            check_dataset_coverage(dataset_coverage("MC", payload), [])

    def test_blinding_record_for_blinded_data(self):
        _, report = sum_hists_with_coverage({"Data": _data_payload(blind=True)}, RUN3, [])
        assert blinding_record(report) == self.BLINDED_RECORD
        manifest = build_manifest(report, era=RUN3, sample="Muon", families=[], syst_observables=None,
                                  output_file=Path("WRAnalyzer_Muon.root"))
        assert manifest["blinding"] == self.BLINDED_RECORD

    @pytest.mark.parametrize("region, kinds", [("resolved", ["cutflow_resolved"]),
                                               ("boosted", ["cutflow_boosted"])])
    def test_blinding_record_follows_region_mode(self, region, kinds):
        _, report = sum_hists_with_coverage(
            {"Data": _data_payload(region=region, variants=(), blind=True)}, RUN3, [])
        record = blinding_record(report)
        assert record["applied"] is True
        assert record["signal_regions"] == [f"wr_{f}_{region}_sr" for f in ("ee", "mumu")]
        assert sorted(record["cutflow_truncated_before"]) == kinds

    def test_blinding_record_not_applied_to_mc_or_unblinded_data(self):
        mc = _payload(_hist(), {}, families=(), region="both", variants=("nomass",), blind=True)
        _, report = sum_hists_with_coverage({"MC": mc}, RUN3, [])
        assert blinding_record(report) == self.NOT_APPLIED
        _, report = sum_hists_with_coverage({"Data": _data_payload(era=UL18)}, UL18, [])
        assert blinding_record(report) == self.NOT_APPLIED

    def test_blinding_record_not_applied_to_legacy_payloads(self):
        h = _hist()
        _fill(h, "wr_ee_resolved_sr", NOMINAL, 3.0, process="Observed")
        _, report = sum_hists_with_coverage({"Data": {"mass_dilepton": h}}, RUN3, [])
        assert not report.coverage_known
        assert blinding_record(report) == self.NOT_APPLIED
        # A tracked payload from before blinding existed (no "blind" entry).
        payload = _data_payload()
        legacy = json.loads(next(iter(payload["dataset_meta"]["syst_config"])))
        del legacy["blind"]
        payload["dataset_meta"]["syst_config"] = {json.dumps(legacy, sort_keys=True)}
        _, report = sum_hists_with_coverage({"Data": payload}, RUN3, [])
        assert report.coverage_known
        assert blinding_record(report) == self.NOT_APPLIED


# ---------------------------------------------------------------------------
# End-to-end through the analyzer
# ---------------------------------------------------------------------------

def _mumu_chunk(n=4, *, with_variation_fields=True):
    """MC chunk of mumu resolved-SR events (era UL18)."""
    ev = _make_integration_events(n_events=n, n_muons=2, n_electrons=0, n_jets=2)
    ev.genWeight = ak.Array([1.0, 2.0, 3.0, 4.0][:n])
    mu = ev.Muon
    mu = ak.with_field(mu, ak.Array([[300.0, 250.0]] * n), "pt")
    mu = ak.with_field(mu, ak.zeros_like(mu.eta), "eta")
    mu = ak.with_field(mu, ak.Array([[0.0, np.pi]] * n), "phi")
    if with_variation_fields:
        mu = ak.with_field(mu, mu.pt * 1.02, "pt_scale_up")
        mu = ak.with_field(mu, mu.pt * 0.98, "pt_scale_down")
    ev.Muon = mu
    _set_sr_jets(ev, n)
    return ev


def _ee_chunk(n=3, *, with_variation_fields=False):
    """MC chunk of ee resolved-SR events with no muons at all."""
    ev = _make_integration_events(n_events=n, n_muons=0, n_electrons=2, n_jets=2)
    ev.genWeight = ak.Array([1.0] * n)
    el = ev.Electron
    el = ak.with_field(el, ak.Array([[300.0, 250.0]] * n), "pt")
    el = ak.with_field(el, ak.zeros_like(el.eta), "eta")
    el = ak.with_field(el, ak.Array([[0.0, np.pi]] * n), "phi")
    ev.Electron = el
    if with_variation_fields:
        mu = ev.Muon
        mu = ak.with_field(mu, mu.pt, "pt_scale_up")
        mu = ak.with_field(mu, mu.pt, "pt_scale_down")
        ev.Muon = mu
    _set_sr_jets(ev, n)
    return ev


def _set_sr_jets(ev, n):
    ev.Jet = ak.with_field(ev.Jet, ak.full_like(ev.Jet.pt, 300.0), "pt")
    ev.Jet = ak.with_field(ev.Jet, ak.zeros_like(ev.Jet.eta), "eta")
    ev.Jet = ak.with_field(ev.Jet, ak.Array([[np.pi / 2, -np.pi / 2]] * n), "phi")


def _concat_events(a, b):
    """Concatenate two mock event chunks collection by collection."""
    out = type(a)()
    for name, value in vars(a).items():
        other = getattr(b, name)
        if name == "metadata":
            setattr(out, name, value)
        elif isinstance(value, ak.Array):
            setattr(out, name, ak.concatenate([value, other]))
        else:
            obj = type(value)()
            for fname, fvalue in vars(value).items():
                setattr(obj, fname, ak.concatenate([fvalue, getattr(other, fname)]))
            setattr(out, name, obj)
    return out


def _pairs(h):
    return {(r, s) for r in h.axes["region"] for s in h.axes["syst"]}


def _assert_hists_equal(h1, h2):
    assert _pairs(h1) == _pairs(h2)
    for region, syst in _pairs(h1):
        v1 = h1[{"region": region, "syst": syst}].view(flow=True)
        v2 = h2[{"region": region, "syst": syst}].view(flow=True)
        np.testing.assert_allclose(v1["value"], v2["value"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(v1["variance"], v2["variance"], rtol=1e-12, atol=0)


class TestAnalyzerCoverage:
    FAMILIES = ["pileup", "sf", "muscale"]
    OMIT = ["ElectronTrigSf"]   # exercises the omission bookkeeping

    def _proc(self, syst_observables=None, region="resolved", omit=None):
        return WrAnalysis(mass_point=None, region=region, enabled_systs=self.FAMILIES,
                          syst_observables=syst_observables,
                          omit_systs=self.OMIT if omit is None else omit)

    @_apply_sf_patches
    def test_chunk_records_complete_coverage(self, *_):
        out = self._proc().process(_mumu_chunk())["DYJetsToLL_M-50"]
        assert out["n_chunks"] == 1
        meta = out["dataset_meta"]
        assert (meta["era"], meta["datatype"], meta["physics_group"]) == ({UL18}, {"mc"}, {"DYJets"})
        # Blinding is run-level (recorded for MC too); the processor default is blind.
        assert meta["syst_config"] == {production_config(self.FAMILIES, self.OMIT, None, region="resolved",
                                                          blind=True)}
        assert set(out["syst_coverage"]) == set(RESOLVED_REGIONS)
        for region in RESOLVED_REGIONS:
            labels = set(out["syst_coverage"][region])
            expected = set(expected_labels(UL18, self.FAMILIES, region, is_mc=True, omitted=self.OMIT))
            assert labels == {NOMINAL} | expected, region
        check_dataset_coverage(dataset_coverage("DYJetsToLL_M-50", out), self.FAMILIES)

    @_apply_sf_patches
    def test_omitted_pileup_is_not_filled(self, *_):
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=["pileup"],
                          omit_systs=["Pileup"])
        out = proc.process(_mumu_chunk())["DYJetsToLL_M-50"]
        assert "PileupUp" not in set(out["mass_dilepton"].axes["syst"])
        assert set(out["syst_coverage"]["wr_mumu_resolved_sr"]) == {NOMINAL}
        check_dataset_coverage(dataset_coverage("d", out), ["pileup"])
        assert coverage_status(dataset_coverage("d", out), "wr_mumu_resolved_sr", "PileupUp") == "omitted"

    @_apply_sf_patches
    def test_boosted_regions_missing_from_a_chunk_fail_at_processing(self, *_):
        ev = _mumu_chunk()
        ev.FatJet = ak.without_field(ev.FatJet, "lsf3")
        with pytest.raises(RuntimeError, match="no 'lsf3' field"):
            self._proc(region="both").process(ev)

    @_apply_sf_patches
    def test_unavailable_component_without_omission_is_caught_at_save(self, *_):
        """An identity SF from a missing payload never becomes a produced variation."""
        with TestRegistry._without_electron_trigger_payload():
            out = self._proc(omit=[]).process(_mumu_chunk())["DYJetsToLL_M-50"]
        assert "ElectronTrigSfUp" not in out["syst_coverage"]["wr_ee_resolved_sr"]
        assert "ElectronTrigSfUp" not in set(out["mass_dilepton"].axes["syst"])
        with pytest.raises(SystematicCoverageError, match="'ElectronTrigSf(Up|Down)' completed in 0 of 1"):
            check_dataset_coverage(dataset_coverage("d", out), self.FAMILIES)

    @_apply_sf_patches
    def test_jes_requires_shifted_fields_on_every_populated_collection(self, *_):
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=["jes"])
        # AK4 shifted, populated AK8 not shifted.
        ev = _make_integration_events(n_events=2, n_muons=2, n_jets=2, n_fatjets=1,
                                      metadata={"era": RUN3})
        for d in ("up", "down"):
            ev.Jet = ak.with_field(ev.Jet, ev.Jet.pt * 1.1, f"pt_CMS_scale_j_Total_{d}")
            ev.Jet = ak.with_field(ev.Jet, ev.Jet.mass * 1.1, f"mass_CMS_scale_j_Total_{d}")
        with pytest.raises(RuntimeError, match="FatJet lacks"):
            proc.process(ev)
        # Mass variation missing on AK4.
        ev = _make_integration_events(n_events=2, n_muons=2, n_jets=2, n_fatjets=0,
                                      metadata={"era": RUN3})
        for d in ("up", "down"):
            ev.Jet = ak.with_field(ev.Jet, ev.Jet.pt * 1.1, f"pt_CMS_scale_j_Total_{d}")
        with pytest.raises(RuntimeError, match="Jet lacks.*mass_CMS_scale_j_Total"):
            proc.process(ev)
        # Empty AK8 collection needs nothing.
        ev = _make_integration_events(n_events=2, n_muons=2, n_jets=2, n_fatjets=0,
                                      metadata={"era": RUN3})
        for d in ("up", "down"):
            ev.Jet = ak.with_field(ev.Jet, ev.Jet.pt * 1.1, f"pt_CMS_scale_j_Total_{d}")
            ev.Jet = ak.with_field(ev.Jet, ev.Jet.mass * 1.1, f"mass_CMS_scale_j_Total_{d}")
        out = proc.process(ev)["DYJetsToLL_M-50"]
        assert out["syst_coverage"]["wr_mumu_resolved_sr"]["JESUp"] == 1

    @_apply_sf_patches
    def test_chunks_with_different_observable_coverage_cannot_merge(self, *_):
        out_fit = self._proc(syst_observables=["fit"]).process(_mumu_chunk())
        out_all = self._proc(syst_observables=None).process(_mumu_chunk())
        merged = accumulate([out_fit, out_all])
        with pytest.raises(SystematicCoverageError, match="systematic production config"):
            dataset_coverage("d", merged["DYJetsToLL_M-50"])

    @_apply_sf_patches
    def test_lepton_shape_variation_covers_every_region(self, *_):
        """No flavor filter: the ee regions are evaluated under the muon shift too."""
        out = self._proc().process(_mumu_chunk())["DYJetsToLL_M-50"]
        assert out["syst_coverage"]["wr_ee_resolved_sr"]["MuScaleUp"] == 1
        assert out["syst_coverage"]["wr_mumu_resolved_sr"]["MuScaleDown"] == 1

    @_apply_sf_patches
    def test_missing_variation_fields_with_objects_present_raises(self, *_):
        with pytest.raises(RuntimeError, match="Muon lacks \\['pt_scale_up'\\]"):
            self._proc().process(_mumu_chunk(with_variation_fields=False))

    @_apply_sf_patches
    def test_chunk_boundaries_do_not_change_results(self, *_):
        """Two chunks (one with no muons) accumulate to the single-chunk result."""
        a, b = _mumu_chunk(), _ee_chunk()
        both = _concat_events(_mumu_chunk(), _ee_chunk(with_variation_fields=True))

        out_a = self._proc().process(a)
        out_b = self._proc().process(b)
        merged = accumulate([out_a, out_b])["DYJetsToLL_M-50"]
        single = self._proc().process(both)["DYJetsToLL_M-50"]

        assert merged["n_chunks"] == 2 and single["n_chunks"] == 1
        # The muon-less chunk counted MuScale as covered by filling its nominal.
        assert out_b["DYJetsToLL_M-50"]["syst_coverage"]["wr_ee_resolved_sr"]["MuScaleUp"] == 1
        assert merged["syst_coverage"] == {
            r: {l: 2 for l in labels} for r, labels in single["syst_coverage"].items()
        }
        for key in ("mass_dilepton", "mass_fourobject", "pt_leading_lepton"):
            _assert_hists_equal(merged[key], single[key])

        # The ee nominal survives under the muon-shift label.
        ee = merged["mass_dilepton"][{"region": "wr_ee_resolved_sr", "syst": "MuScaleUp"}].sum(flow=True).value
        ee_nom = merged["mass_dilepton"][{"region": "wr_ee_resolved_sr", "syst": NOMINAL}].sum(flow=True).value
        assert ee_nom > 0 and ee == pytest.approx(ee_nom)

        for cov in (dataset_coverage("d", merged), dataset_coverage("d", single)):
            check_dataset_coverage(cov, self.FAMILIES)

    @_apply_sf_patches
    def test_reduced_and_all_coverage_agree_on_shared_histograms(self, *_):
        reduced = self._proc(syst_observables=["fit"]).process(_mumu_chunk())["DYJetsToLL_M-50"]
        everything = self._proc(syst_observables=None).process(_mumu_chunk())["DYJetsToLL_M-50"]

        for name in FIT_OBSERVABLES:
            if name in reduced:
                _assert_hists_equal(reduced[name], everything[name])
        assert set(reduced["pt_leading_lepton"].axes["syst"]) == {NOMINAL}
        assert "PileupUp" in everything["pt_leading_lepton"].axes["syst"]
        assert reduced["syst_coverage"] == everything["syst_coverage"]


# ---------------------------------------------------------------------------
# Writer and manifest
# ---------------------------------------------------------------------------

class TestWriter:
    @_apply_sf_patches
    def test_writes_produced_pairs_normalization_and_manifest(self, tmp_path, monkeypatch, *_):
        monkeypatch.chdir(tmp_path)
        families = ["lumi", "pileup", "sf"]
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=families,
                          syst_observables=["fit"], omit_systs=["ElectronTrigSf"])
        hists = proc.process(_mumu_chunk())
        args = SimpleNamespace(era=UL18, sample="DYJets", mass=None, dir=None, name=None,
                               systs=families, syst_observables=["fit"], omit_systs=["ElectronTrigSf"])
        root_path = save_histograms(hists, args)
        manifest = json.loads(root_path.with_name(root_path.stem + "_systematics.json").read_text())

        with uproot.open(root_path) as f:
            keys = {k.split(";")[0] for k in f.keys()}
            nominal = f["wr_mumu_resolved_sr/mass_dilepton_wr_mumu_resolved_sr"].values(flow=True)
            up = f["syst_lumiup_wr_mumu_resolved_sr/mass_dilepton_syst_lumiup_wr_mumu_resolved_sr"].values(flow=True)
            down = f["syst_lumidown_wr_mumu_resolved_sr/mass_dilepton_syst_lumidown_wr_mumu_resolved_sr"].values(flow=True)
            up_var = f["syst_lumiup_wr_mumu_resolved_sr/mass_dilepton_syst_lumiup_wr_mumu_resolved_sr"].variances(flow=True)
            nom_var = f["wr_mumu_resolved_sr/mass_dilepton_wr_mumu_resolved_sr"].variances(flow=True)

        delta = float(LUMI_UNC[UL18])
        assert nominal.sum() > 0
        np.testing.assert_allclose(up, nominal * (1 + delta))
        np.testing.assert_allclose(down, nominal * (1 - delta))
        np.testing.assert_allclose(up_var, nom_var * (1 + delta) ** 2)

        assert "syst_pileupup_wr_mumu_resolved_sr/mass_dilepton_syst_pileupup_wr_mumu_resolved_sr" in keys
        assert "syst_muonidsfup_wr_mumu_resolved_sr/mass_dilepton_syst_muonidsfup_wr_mumu_resolved_sr" in keys
        # Unaffected: muon SFs do not apply to the ee region, so no empty template.
        assert not any(k.startswith("syst_muonidsfup_wr_ee_resolved_sr") for k in keys)
        # Coverage restriction: only fit observables carry variations.
        assert not any("pt_leading_lepton_syst_" in k for k in keys)
        assert "wr_mumu_resolved_sr/pt_leading_lepton_wr_mumu_resolved_sr" in keys

        assert manifest["requested_families"] == sorted(families)
        assert manifest["syst_observables"] == sorted(n for n in FIT_OBSERVABLES if n in all_histogram_names())
        assert manifest["coverage_known"] is True
        assert manifest["datasets"]["DYJetsToLL_M-50"]["n_chunks"] == 1
        ee = manifest["regions"]["wr_ee_resolved_sr"]
        assert "MuonIdSfUp" in ee["unaffected"]
        assert "PileupUp" in ee["produced"]
        assert ee["normalization"] == ["LumiUp", "LumiDown"]
        mumu = manifest["regions"]["wr_mumu_resolved_sr"]
        assert "MuonIdSfUp" in mumu["produced"]
        assert "ElectronIdSfUp" in mumu["unaffected"]
        assert manifest["omitted_sources"] == ["ElectronTrigSf"]
        assert ee["omitted"] == ["ElectronTrigSfDown", "ElectronTrigSfUp"]
        assert mumu["omitted"] == []
        assert "ElectronTrigSfUp" not in ee["produced"]

    @_apply_sf_patches
    def test_writer_refuses_incomplete_dataset(self, tmp_path, monkeypatch, *_):
        monkeypatch.chdir(tmp_path)
        proc = WrAnalysis(mass_point=None, region="resolved", enabled_systs=["pileup"])
        hists = proc.process(_mumu_chunk())
        payload = hists["DYJetsToLL_M-50"]
        payload["n_chunks"] = 2   # pretend a second chunk produced nothing
        args = SimpleNamespace(era=UL18, sample="DYJets", mass=None, dir=None, name=None,
                               systs=["pileup"], syst_observables=None, omit_systs=[])
        with pytest.raises(SystematicCoverageError):
            save_histograms(hists, args)
