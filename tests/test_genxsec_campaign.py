"""Decision logic of scripts/genxsec_campaign.py (header -> config cross section)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("genxsec_campaign", REPO / "scripts" / "genxsec_campaign.py")
gx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gx)


def _entry(primary, gen, *, energy="13p6TeV", flavours=None, wmean=0.0, wfrac=None, unc=1.0):
    content = {"lhe_lepton_flavours": flavours or {"e": 0, "mu": 0, "tau": 0}, "w_decays_mean": wmean}
    if wfrac:
        content["w_decay_fractions"] = wfrac
    return {"primary": primary, "energy": energy, "xsec_gen": gen, "xsec_gen_unc": unc, "content": content,
            "eras": ["RunIII2024Summer24"], "done": True}


DILEP = {"0_leptonic_W": 0.0, "1_leptonic_W": 0.0, "2_leptonic_W": 1.0}
SEMI = {"0_leptonic_W": 0.0, "1_leptonic_W": 1.0, "2_leptonic_W": 0.0}
ONE_LEP = {"0_leptonic_W": 0.0, "1_leptonic_W": 1.0, "2_leptonic_W": 0.0}
NATURAL = {"0_leptonic_W": 0.45, "1_leptonic_W": 0.44, "2_leptonic_W": 0.11}


def test_ttbar_siblings_prove_inclusive_header():
    r = {"a": _entry("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", 762.1, wmean=2.0, wfrac=DILEP),
         "b": _entry("TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8", 762.1, wmean=2.0, wfrac=SEMI)}
    d = gx.classify(r["a"], r)
    assert d["header"] == "inclusive"
    assert d["factor"] == pytest.approx(gx.BR["2L2Nu"])
    assert d["reference"] == 923.6
    assert d["kfactor"] == pytest.approx(923.6 / 762.1)
    assert gx.proposed_xsec(r["a"], d, "generator") == pytest.approx(762.1 * 0.3258 ** 2)
    assert gx.proposed_xsec(r["a"], d, "reference") == pytest.approx(923.6 * 0.3258 ** 2)
    d2 = gx.classify(r["b"], r)
    assert d2["factor"] == pytest.approx(2 * 0.3258 * (1 - 0.3258))


def test_tw_without_sibling_uses_reference():
    e = _entry("TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", 35.99, wmean=2.0, wfrac=DILEP)
    d = gx.classify(e, {"x": e})
    assert d["header"] == "inclusive" and d["factor"] == pytest.approx(gx.BR["2L2Nu"])
    assert d["kfactor"] == pytest.approx(43.95 / 35.99)


def test_tchannel_header_already_includes_decay():
    e = _entry("TBbarQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8", 38.6, wmean=1.0, wfrac=ONE_LEP)
    d = gx.classify(e, {"x": e})
    assert d["header"] == "includes decay" and d["factor"] == 1.0
    assert d["kfactor"] == pytest.approx(145.0 * 0.3258 / 38.6)
    anti = _entry("TbarBQtoLNu-t-channel-4FS_TuneCP5_13p6TeV_powheg-madspin-pythia8", 23.34, wmean=1.0, wfrac=ONE_LEP)
    assert gx.classify(anti, {"x": anti})["reference"] == 87.2


def test_inclusive_decays_keep_header():
    e = _entry("WW_TuneCP5_13p6TeV_pythia8", 80.18, wmean=2.0, wfrac=NATURAL)
    d = gx.classify(e, {"x": e})
    assert d["factor"] == 1.0 and d["kfactor"] == 1.0 and d["reference"] is None
    tt = _entry("TT_TuneCP5_13TeV-powheg-pythia8", 700.0, energy="13TeV", wmean=2.0, wfrac=NATURAL)
    d = gx.classify(tt, {"x": tt})
    assert d["factor"] == 1.0 and d["kfactor"] == pytest.approx(833.9 / 700.0)


def test_dy_flavour_rules():
    pw = _entry("DYto2E_Bin-MLL-50to120_TuneCP5_13p6TeV_powheg-pythia8", 2219.0, flavours={"e": 40000, "mu": 0, "tau": 0})
    mg = _entry("DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", 6685.0, flavours={"e": 40000, "mu": 0, "tau": 0})
    lo = _entry("DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8", 5442.0, flavours={"e": 0, "mu": 40000, "tau": 0})
    three = _entry("DYto2L-4Jets_Bin-HT-100to400-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8", 3.365, flavours={"e": 13570, "mu": 13266, "tau": 13164})
    ul = _entry("DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8", 5398.0, energy="13TeV", flavours={"e": 1, "mu": 1, "tau": 1})
    assert gx.classify(pw)["factor"] == 1.0 and gx.classify(pw)["header"] == "per flavour"
    assert gx.classify(mg)["factor"] == pytest.approx(1 / 3) and gx.classify(mg)["header"] == "three-flavour"
    assert gx.classify(lo)["factor"] == pytest.approx(1 / 3)
    assert gx.classify(three)["factor"] == 1.0
    assert gx.classify(ul)["factor"] == 1.0
    lines = gx.dy_crosscheck({"a": pw, "b": mg, "c": _entry("DYto2E_Bin-MLL-120to200_TuneCP5_13p6TeV_powheg-pythia8", 21.65, flavours={"e": 1, "mu": 0, "tau": 0})})
    assert len(lines) == 1 and "2240.7" in lines[0] and "2228.3" in lines[0]


def test_single_w_rules():
    mg = _entry("WtoLNu-4Jets_Bin-HT-100to400-MLNu-0to120_TuneCP5_13p6TeV_madgraphMLM-pythia8", 1635.0, wmean=1.0, wfrac=ONE_LEP)
    assert gx.classify(mg, {"x": mg})["factor"] == 1.0
    pw = _entry("WtoLNu_SomePowhegSample_13p6TeV_powheg-pythia8", 20000.0, wmean=1.0, wfrac=ONE_LEP)
    d = gx.classify(pw, {"x": pw})
    assert d["factor"] is None and "decide by hand" in d["factor_reason"]
    assert gx.proposed_xsec(pw, d, "reference") is None


def test_process_and_reference_lookup():
    assert gx.process_of("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8") == ("ttbar", "both")
    assert gx.process_of("TTLNu-1Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8") == (None, "")
    assert gx.process_of("TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8") == ("tW", "antitop")
    assert gx.process_of("ST_tW_top_5f_NoFullyHadronicDecays_TuneCP5_13TeV-powheg-pythia8") == ("tW", "top")
    assert gx.process_of("ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8") == ("t-channel", "antitop")
    assert gx.process_of("ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8") == ("s-channel", "both")
    assert gx.reference_for("ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8", "13TeV") == (pytest.approx(10.32), "top + antitop: NLO, TOP twiki SingleTopRefXsec")
    assert gx.reference_for("TBbartoLplusNuBbar-s-channel-4FS_TuneCP5_13p6TeV_amcatnlo-pythia8", "13p6TeV") is None


def test_refresh_metadata_updates_only_config_fields(tmp_path):
    from wrcoffea.skim.plan import Plan, DatasetPlan, refresh_metadata
    das = "/X_TuneCP5_13p6TeV/RunIII2024Summer24NanoAODv15-v1/NANOAODSIM"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({das: {"das_name": das, "era": "RunIII2024Summer24", "dataset": "X", "physics_group": "tt_tW",
                                     "xsec": 2.0, "xsec_source": {"generator": 1.9}, "datatype": "mc"}}))
    ds = DatasetPlan(key="X", era="RunIII2024Summer24", run="Run3", year="2024", datatype="mc", physics_group="tt_tW",
                     category="backgrounds", das_paths=[das], metadata={"xsec": 1.0, "dataset": "X", "datatype": "mc"},
                     files=[{"lfn": "/store/a.root", "size": 1, "events": 1, "adler32": "0", "valid": True, "dataset": das}],
                     groups=[["/store/a.root"]], schema_id="lazy", dbs_summary={}, out_dir="x")
    plan = Plan(campaign="t", created="now", code_version="t", cuts={}, policy_version=2, settings={}, datasets={"X": ds}, schemas={})
    changed = refresh_metadata(plan, [cfg])
    assert changed == {"X": {"xsec": (1.0, 2.0), "xsec_source": (None, {"generator": 1.9}), "era": (None, "RunIII2024Summer24"),
                             "das_name": (None, das), "physics_group": (None, "tt_tW")}}
    assert plan.datasets["X"].metadata["xsec"] == 2.0 and plan.datasets["X"].files[0]["lfn"] == "/store/a.root"
    assert refresh_metadata(plan, [cfg]) == {}
    plan.save(tmp_path / "plan")
    assert Plan.load(tmp_path / "plan").datasets["X"].metadata["xsec_source"] == {"generator": 1.9}


def test_lo_dy_kfactor_from_measured_headers():
    e_nlo = _entry("DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", 6685.0, flavours={"e": 1, "mu": 0, "tau": 0})
    mu_nlo = _entry("DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", 6658.0, flavours={"e": 0, "mu": 1, "tau": 0})
    e_lo = _entry("DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8", 5469.0, flavours={"e": 1, "mu": 0, "tau": 0})
    mu_lo = _entry("DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8", 5442.0, flavours={"e": 0, "mu": 1, "tau": 0})
    ht = _entry("DYto2L-4Jets_Bin-HT-400to800-MLL-120_TuneCP5_13p6TeV_madgraphMLM-pythia8", 0.1733, flavours={"e": 1, "mu": 1, "tau": 1})
    low = _entry("DYto2E-4Jets_Bin-MLL-10to50_TuneCP5_13p6TeV_madgraphMLM-pythia8", 17190.0, flavours={"e": 1, "mu": 0, "tau": 0})
    r = {"a": e_nlo, "b": mu_nlo, "c": e_lo, "d": mu_lo, "e": ht, "f": low}
    k = (6685.0 + 6658.0) / (5469.0 + 5442.0)
    assert gx.dy_lo_kfactor(r, "13p6TeV") is not None and gx.dy_lo_kfactor(r, "13p6TeV")[0] == pytest.approx(k)
    assert gx.dy_lo_kfactor(r, "13TeV") is None
    d = gx.classify(e_lo, r)
    assert d["factor"] == pytest.approx(1 / 3) and d["kfactor"] == pytest.approx(k)
    assert gx.proposed_xsec(e_lo, d, "reference") == pytest.approx(5469.0 / 3 * k)
    assert gx.proposed_xsec(e_lo, d, "generator") == pytest.approx(5469.0 / 3)
    d = gx.classify(ht, r)
    assert d["factor"] == 1.0 and d["kfactor"] == pytest.approx(k)     # three-flavour LO: k only
    assert gx.classify(low, r)["kfactor"] == pytest.approx(k)          # low-mass companion: same flat k
    assert gx.classify(e_nlo, r)["kfactor"] == 1.0                      # NLO samples keep the generator value
    alone = gx.classify(e_lo, {"c": e_lo})
    assert alone.get("undecided_k") and alone["kfactor"] == 1.0


def test_unique_samples_skips_scan_and_data(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({
        "/WRtoNLtoLLJJ_MWR500to3500_TuneCP5-madgraph-pythia8/RunIIAutumn18NanoAODv7-rpscan-v1/NANOAODSIM":
            {"datatype": "mc", "era": "RunIISummer20UL18", "dataset": "scan", "physics_group": "Signal", "xsec": 1.0, "scan": True},
        "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_v17-v1/NANOAODSIM":
            {"datatype": "mc", "era": "RunIISummer20UL18", "dataset": "TTTo2L2Nu", "physics_group": "tt_tW", "xsec": 88.29},
        "/SingleMuon/Run2018A-UL2018_MiniAODv2_GT36-v1/NANOAOD":
            {"datatype": "data", "era": "RunIISummer20UL18", "dataset": "SingleMuon_Run2018A"},
    }))
    s = gx.unique_samples([cfg])
    assert list(s) == ["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8|13TeV|RunIISummer20UL18"]
    assert s["TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8|13TeV|RunIISummer20UL18"]["eras"] == ["RunIISummer20UL18"]


def test_samples_are_keyed_per_era_and_old_records_migrate(tmp_path):
    das23 = "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23NanoAODv12-130X_mcRun3_2023_realistic_v14-v2/NANOAODSIM"
    das23b = "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/Run3Summer23BPixNanoAODv12-130X_mcRun3_2023_realistic_postBPix_v2-v3/NANOAODSIM"
    c1, c2 = tmp_path / "a.json", tmp_path / "b.json"
    c1.write_text(json.dumps({das23: {"datatype": "mc", "era": "Run3Summer23", "dataset": "TT", "physics_group": "tt_tW", "xsec": 98.0}}))
    c2.write_text(json.dumps({das23b: {"datatype": "mc", "era": "Run3Summer23BPix", "dataset": "TT", "physics_group": "tt_tW", "xsec": 98.0}}))
    s = gx.unique_samples([c1, c2])
    assert sorted(s) == ["TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|Run3Summer23",
                         "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|Run3Summer23BPix"]
    assert gx.era_of_miniaod("/X/Run3Summer23BPixMiniAODv4-130X_v6-v2/MINIAODSIM") == "Run3Summer23BPix"
    assert gx.era_of_miniaod("/X/RunIISummer20UL16MiniAODAPVv2-106X_v11-v1/MINIAODSIM") == "RunIISummer20UL16APV"
    assert gx.era_of_miniaod("/X/RunIII2024Summer24MiniAODv6-150X_v2-v3/MINIAODSIM") == "RunIII2024Summer24"
    old = {"TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV": {
        "primary": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", "energy": "13p6TeV", "xsec_gen": 762.1,
        "miniaod": "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24MiniAODv6-150X_mcRun3_2024_realistic_v2-v3/MINIAODSIM",
        "eras": ["Run3Summer23", "RunIII2024Summer24"]}}
    new, changed = gx.migrate_results(old)
    assert list(new) == ["TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|RunIII2024Summer24"] and changed
    rec = new["TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|RunIII2024Summer24"]
    assert rec["era"] == "RunIII2024Summer24" and rec["eras"] == ["RunIII2024Summer24"]      # 2023 claim dropped
    assert gx.migrate_results(new)[1] == 0


def test_siblings_and_lo_kfactor_stay_within_an_era():
    def e(prim, gen, era, **kw):
        x = _entry(prim, gen, **kw); x["era"] = era; return x
    a24 = e("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", 762.1, "RunIII2024Summer24", wmean=2.0, wfrac=DILEP)
    b23 = e("TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8", 700.0, "Run3Summer23", wmean=2.0, wfrac=SEMI)
    r = {"a": a24, "b": b23}
    assert gx.sibling_headers(a24, r) == []           # a 2023 sibling does not vouch for a 2024 header
    nlo = e("DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", 6685.0, "RunIII2024Summer24", flavours={"e": 1, "mu": 0, "tau": 0})
    lo = e("DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8", 5469.0, "Run3Summer23", flavours={"e": 1, "mu": 0, "tau": 0})
    assert gx.dy_lo_kfactor({"n": nlo, "l": lo}, "13p6TeV", "Run3Summer23") is None
    assert gx.classify(lo, {"n": nlo, "l": lo}).get("undecided_k")


def test_parse_genxsec_output():
    txt = ("Before matching: total cross section = 1.319e+04 +- 2.1e+01 pb\n"
           "After matching: total cross section = 5.469e+03 +- 1.9e+01 pb\n"
           "Filter efficiency (event-level)= (46838) / (46838) = 1.000e+00 +- 0.000e+00\n"
           "After filter: final cross section = 5.469e+03 +- 1.936e+01 pb\n"
           "After filter: final fraction of events with negative weights = 1.2e-02\n")
    r = gx.parse_genxsec_output(txt)
    assert r["xsec_gen"] == 5469.0 and r["xsec_gen_unc"] == 19.36 and r["events_run"] == 46838
    assert r["xsec_before_matching"] == 13190.0 and r["filter_eff"] == 1.0 and r["neg_weight_fraction"] == 0.012
    assert "error" in gx.parse_genxsec_output("cmsRun crashed")


def test_condor_jobs_and_collect_round_trip(tmp_path, monkeypatch):
    work = tmp_path / "work"; work.mkdir()
    jobs = tmp_path / "jobs"
    prepared = {
        "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|Run3Summer23": {
            "primary": "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", "energy": "13p6TeV", "era": "Run3Summer23",
            "eras": ["Run3Summer23"], "das": "/TT/Run3Summer23NanoAODv12-x/NANOAODSIM", "xsec_config": 98.0,
            "physics_group": "tt_tW", "miniaod": "/TT/Run3Summer23MiniAODv4-x/MINIAODSIM",
            "lfns": ["/store/mc/a.root"], "release": "CMSSW_13_0_13", "image": "el8",
            "content": {"w_decays_mean": 2.0}},
        "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8|13p6TeV|Run3Summer23": {
            "primary": "DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", "energy": "13p6TeV",
            "era": "Run3Summer23", "eras": ["Run3Summer23"], "das": "/DY/x/NANOAODSIM", "xsec_config": 2228.3,
            "physics_group": "DYJets", "miniaod": "/DY/Run3Summer23MiniAODv4-x/MINIAODSIM",
            "lfns": ["/store/mc/b.root", "/store/mc/c.root"], "release": "CMSSW_13_0_13", "image": "el8"},
        "Broken|13p6TeV|Run3Summer23": {"primary": "Broken", "energy": "13p6TeV", "era": "Run3Summer23",
                                         "error": "no MiniAOD parent"},
    }
    n = gx.write_condor_jobs(prepared, jobs, per_job=1, proxy="/p/x509", cfg_text="# cfg\n")
    assert n == 2
    lines = (jobs / "jobs.txt").read_text().splitlines()
    assert lines[0].split()[1:3] == ["CMSSW_13_0_13", "el8_amd64_gcc11"] and lines[0].split()[3].endswith("el8:x86_64")
    listing = "".join((jobs / "lists" / l.split()[0]).read_text() for l in lines)
    assert "root://cmsxrootd.fnal.gov//store/mc/b.root,root://cmsxrootd.fnal.gov//store/mc/c.root" in listing
    assert "Broken" not in listing
    jdl = (jobs / "submit.jdl").read_text()
    assert "x509userproxy           = /p/x509" in jdl and "queue listfile,release,arch,image from jobs.txt" in jdl
    (jobs / "samples.json").write_text(json.dumps(prepared))
    # Condor brought back one good log and one failed one
    (jobs / "out").mkdir()
    tt = gx.safe_name("TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|Run3Summer23")
    (jobs / "out" / f"{tt}.log").write_text("After filter: final cross section = 7.621e+02 +- 9.5e-02 pb\n"
                                            "Filter efficiency (event-level)= (43210) / (43210) = 1.0e+00\ncmsRun rc=0\n")
    dy = gx.safe_name("DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8|13p6TeV|Run3Summer23")
    (jobs / "out" / f"{dy}.log").write_text("Fatal Exception: file open failure\ncmsRun rc=65\n")
    import argparse
    rc = gx.cmd_collect(argparse.Namespace(work=str(work), jobs_dir=str(jobs)))
    assert rc == 1
    res = json.loads((work / "genxsec_results.json").read_text())
    rec = res["TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8|13p6TeV|Run3Summer23"]
    assert rec["xsec_gen"] == 762.1 and rec["era"] == "Run3Summer23" and rec["rc"] == 0 and rec["files"] == ["/store/mc/a.root"]
    assert "lfns" not in rec and rec["content"] == {"w_decays_mean": 2.0}
    bad = res["DYto2E-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8|13p6TeV|Run3Summer23"]
    assert bad["xsec_gen"] if "xsec_gen" in bad else True
    assert bad.get("xsec_gen") is None and bad["rc"] == 65 and "error" in bad


def test_lo_kfactor_with_2023_style_names():
    def e(prim, gen, fl):
        x = _entry(prim, gen, flavours=fl); x["era"] = "Run3Summer23"; return x
    three = {"e": 1, "mu": 1, "tau": 1}
    nlo = e("DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", 6736.0, three)
    lo = e("DYto2L-4Jets_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8", 5435.0, three)
    ht = e("DYto2L-4Jets_MLL-50to120_HT-40to70_TuneCP5_13p6TeV_madgraphMLM-pythia8", 316.0, three)
    jet1 = e("DYto2L-4Jets_MLL-50_1J_TuneCP5_13p6TeV_madgraphMLM-pythia8", 900.0, three)        # not inclusive
    pt = e("DYto2L-2Jets_MLL-50_PTLL-100_TuneCP5_13p6TeV_amcatnloFXFX-pythia8", 100.0, three)    # not inclusive
    r = {"n": nlo, "l": lo, "h": ht, "j": jet1, "p": pt}
    k = gx.dy_lo_kfactor(r, "13p6TeV", "Run3Summer23")
    assert k is not None and k[0] == pytest.approx(6736.0 / 5435.0)
    d = gx.classify(ht, r)
    assert d["factor"] == 1.0 and d["kfactor"] == pytest.approx(6736.0 / 5435.0) and not d.get("undecided_k")
