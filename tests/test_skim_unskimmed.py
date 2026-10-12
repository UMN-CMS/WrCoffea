"""Unskimmed filesets from the configs and DBS: file URLs, metadata, optional datasets, private
file lists, the scan families left out, and the optional Runs-tree sums."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import uproot

from tests.nano_synth import write_nano_file
from wrcoffea.skim.catalog import FileRecord
from wrcoffea.skim.unskimmed import (build_family, config_files, dataset_entry, eras_with_configs, fileset_path,
                                     url_for, write_era_filesets)

ERA = "RunIII2024Summer24"
DATA_A = "/EGamma0/Run2024C-MINIv6NANOv15-v1/NANOAOD"
DATA_B = "/Muon0/Run2024C-MINIv6NANOv15-v1/NANOAOD"
MC = "/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/RunIII2024Summer24NanoAODv15-150X_v2-v2/NANOAODSIM"
MC_OPT = "/WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8/RunIII2024Summer24NanoAODv15-150X_v2-v2/NANOAODSIM"
MC_REQ = "/ZZ_TuneCP5_13p6TeV_pythia8/RunIII2024Summer24NanoAODv15-150X_v2-v2/NANOAODSIM"


class FakeDBS:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def files(self, dataset, *, valid_only=True):
        self.calls.append((dataset, valid_only))
        return [FileRecord(lfn=lfn, size=1, events=n, adler32="00", valid=True, dataset=dataset)
                for lfn, n in self.table.get(dataset, [])]


def _entry(das, dataset, group, datatype="mc", **extra):
    e = {"das_name": das, "run": "Run3", "year": "2024", "era": ERA, "dataset": dataset,
         "physics_group": group, "datatype": datatype}
    if datatype == "mc":
        e.update(xsec=1.5, order="NLO", total_uncertainty=0.1, xsec_source={"tool": "test"}, note="a note")
    e.update(extra)
    return e


@pytest.fixture
def roots(tmp_path):
    cfg = tmp_path / "configs" / "Run3" / "2024" / ERA
    cfg.mkdir(parents=True)
    (cfg / f"{ERA}_data.json").write_text(json.dumps({
        DATA_A: _entry(DATA_A, "EGamma0_Run2024C", "EGamma", "data"),
        DATA_B: _entry(DATA_B, "Muon0_Run2024C", "Muon", "data")}))
    (cfg / f"{ERA}_mc_dy_lo_ht.json").write_text(json.dumps({
        MC: _entry(MC, "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8", "tt_tW"),
        MC_OPT: _entry(MC_OPT, "WWW_4F", "Other", optional=True)}))
    (cfg / f"{ERA}_signal.json").write_text(json.dumps({
        "WRtoNEltoElElJJ_MWR2000_N100": _entry("WRtoNEltoElElJJ_MWR2000_N100", "WRtoNEltoElElJJ_MWR2000_N100", "Signal",
                                               files=["root://cmsxrootd.hep.wisc.edu//store/user/k/b.root",
                                                      "root://cmsxrootd.hep.wisc.edu//store/user/k/a.root"],
                                               file_events={"root://cmsxrootd.hep.wisc.edu//store/user/k/a.root": 400,
                                                            "root://cmsxrootd.hep.wisc.edu//store/user/k/b.root": 600},
                                               file_sizes={}, file_adler32={})}))
    (cfg / f"{ERA}_signal_scan.json").write_text(json.dumps({MC: _entry(MC, "scan", "Signal", scan=True)}))
    dbs = FakeDBS({DATA_A: [("/store/data/Run2024C/EGamma0/NANOAOD/x/2.root", 10), ("/store/data/Run2024C/EGamma0/NANOAOD/x/1.root", 20)],
                   DATA_B: [("/store/data/Run2024C/Muon0/NANOAOD/x/1.root", 5)],
                   MC: [("/store/mc/RunIII2024Summer24NanoAODv15/TT/x/1.root", 1000)]})
    return tmp_path / "configs", tmp_path / "filesets", dbs


def test_url_for():
    assert url_for("/store/mc/x/1.root", "root://cmsxrootd.fnal.gov/") == "root://cmsxrootd.fnal.gov//store/mc/x/1.root"
    assert url_for("/store/mc/x/1.root", "root://cms-xrd-global.cern.ch") == "root://cms-xrd-global.cern.ch//store/mc/x/1.root"
    assert url_for("root://host//store/user/x.root", "root://cmsxrootd.fnal.gov/") == "root://host//store/user/x.root"
    assert url_for("/tmp/local.root", "") == "/tmp/local.root"


def test_config_files_leaves_scan_out_unless_asked(roots):
    config_root, _, _ = roots
    assert list(config_files(config_root, ERA)) == ["data", "mc_dy_lo_ht", "signal"]
    assert list(config_files(config_root, ERA, ["signal_scan"])) == ["signal_scan"]
    with pytest.raises(FileNotFoundError, match="no config for family nope"):
        config_files(config_root, ERA, ["nope"])
    assert eras_with_configs(config_root) == [ERA]


def test_write_era_filesets(roots, caplog):
    config_root, out_root, dbs = roots
    paths = write_era_filesets(ERA, config_root=config_root, out_root=out_root, dbs=dbs, today="2026-10-11")
    assert paths == [fileset_path(out_root, ERA, f) for f in ("data", "mc_dy_lo_ht", "signal")]
    assert paths[0] == out_root / "Run3" / "2024" / ERA / "unskimmed" / f"{ERA}_data_fileset.json"
    data = json.loads(paths[0].read_text())
    assert list(data) == [DATA_A, DATA_B]
    assert list(data[DATA_A]["files"]) == ["root://cmsxrootd.fnal.gov//store/data/Run2024C/EGamma0/NANOAOD/x/1.root",
                                           "root://cmsxrootd.fnal.gov//store/data/Run2024C/EGamma0/NANOAOD/x/2.root"]
    assert set(data[DATA_A]["files"].values()) == {"Events"}
    md = data[DATA_A]["metadata"]
    assert md["sample"] == "EGamma0_Run2024C" and md["nfiles"] == 2 and md["nevts"] == 30
    assert md["physics_group"] == "EGamma" and md["datatype"] == "data" and md["das_name"] == DATA_A
    assert md["listing"] == {"date": "2026-10-11", "source": "DBS valid files", "redirector": "root://cmsxrootd.fnal.gov/"}
    assert all((d, True) in dbs.calls for d in (DATA_A, DATA_B, MC, MC_OPT))
    # MC: the optional dataset without files is skipped and reported; bookkeeping keys and note dropped
    mc = json.loads(paths[1].read_text())
    assert list(mc) == [MC]
    assert "WWW_4F" in caplog.text and "optional, skipped" in caplog.text
    md = mc[MC]["metadata"]
    assert md["xsec"] == 1.5 and md["order"] == "NLO" and "note" not in md and "genEventSumw" not in md
    assert "dataset" not in md             # reserved by coffea's preprocess
    assert md["sample"] == "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8" and md["nevts"] == 1000
    # private signal: the config's own file list, sorted, with its event counts
    sig = json.loads(paths[2].read_text())
    (key,) = sig
    assert list(sig[key]["files"]) == ["root://cmsxrootd.hep.wisc.edu//store/user/k/a.root",
                                       "root://cmsxrootd.hep.wisc.edu//store/user/k/b.root"]
    md = sig[key]["metadata"]
    assert md["nevts"] == 1000 and md["nfiles"] == 2 and md["listing"]["source"] == "files listed in the config"
    assert not any(k in md for k in ("files", "file_events", "file_sizes", "file_adler32"))


def test_family_restriction_and_dry_run(roots):
    config_root, out_root, dbs = roots
    paths = write_era_filesets(ERA, config_root=config_root, out_root=out_root, dbs=dbs, families=["data"])
    assert [p.name for p in paths] == [f"{ERA}_data_fileset.json"]
    dry = write_era_filesets(ERA, config_root=config_root, out_root=out_root / "dry", dbs=None, dry_run=True)
    assert len(dry) == 3 and not (out_root / "dry").exists()
    assert not dbs.calls[len(dbs.calls):]


def test_required_dataset_without_files_stops(roots):
    config_root, _, dbs = roots
    with pytest.raises(ValueError, match="not optional"):
        dataset_entry(MC_REQ, _entry(MC_REQ, "ZZ", "Other"), dbs=dbs, redirector="root://x/", today="d")
    with pytest.raises(ValueError, match="neither a DAS path"):
        dataset_entry("junk", {"dataset": "junk"}, dbs=dbs, redirector="root://x/", today="d")


def test_runs_sums_from_local_files(tmp_path):
    a, b = tmp_path / "a.root", tmp_path / "b.root"
    write_nano_file(a, n=600, seed=3)
    write_nano_file(b, n=400, seed=4, n_runs_entries=2)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({MC: _entry(MC, "TT", "tt_tW")}))
    dbs = FakeDBS({MC: [(str(a), 600), (str(b), 400)]})
    fileset, skipped = build_family(cfg, dbs=dbs, redirector="", runs=True, workers=2, today="d")
    assert not skipped and list(fileset[MC]["files"]) == sorted([str(a), str(b)])
    md = fileset[MC]["metadata"]
    expected = 0.0
    for p in (a, b):
        with uproot.open(p) as f:
            expected += float(f["Runs"]["genEventSumw"].array().to_numpy().sum())
    assert md["genEventSumw"] == pytest.approx(expected) and md["genEventCount"] == 1000
    assert len(md["LHEScaleSumw"]) == 9 and md["PSSumw"] == pytest.approx([1.0, 1.01, 0.99, 1.0], rel=1e-6)
    # data: no Runs sums even when asked
    dcfg = tmp_path / "data.json"
    dcfg.write_text(json.dumps({DATA_A: _entry(DATA_A, "EGamma0_Run2024C", "EGamma", "data")}))
    fileset, _ = build_family(dcfg, dbs=FakeDBS({DATA_A: [(str(a), 600)]}), redirector="", runs=True, today="d")
    assert "genEventSumw" not in fileset[DATA_A]["metadata"]


def test_cli_help():
    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "bin" / "skim.py"), "unskimmed-fileset", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--runs-sums" in out.stdout and "--all" in out.stdout
