"""The PSSumw estimate from original files: the means, the fileset update, entries left alone."""
import importlib.util
import json
from pathlib import Path

import awkward as ak
import numpy as np
import pytest
import uproot

from tests.nano_synth import write_nano_file
from wrcoffea.skim.catalog import FileRecord

_spec = importlib.util.spec_from_file_location(
    "estimate_ps_sums", str(Path(__file__).resolve().parent.parent / "scripts" / "estimate_ps_sums.py"))
eps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eps)

ERA = "RunIISummer20UL18"
DAS_A = "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_v16_L1v1-v1/NANOAODSIM"
DAS_B = "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_v16_L1v1-v1/NANOAODSIM"
DAS_C = "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_v16_L1v1-v1/NANOAODSIM"


class FakeDBS:
    def __init__(self, table):
        self.table = table

    def files(self, dataset, *, valid_only=True):
        return [FileRecord(lfn=p, size=1, events=100, adler32="00", valid=True, dataset=dataset)
                for p in self.table.get(dataset, [])]


def _direct(paths):
    gw = np.concatenate([ak.to_numpy(uproot.open(p)["Events"]["genWeight"].array()).astype(float) for p in paths])
    ps = np.concatenate([ak.to_numpy(uproot.open(p)["Events"]["PSWeight"].array()).astype(float) for p in paths])
    return (ps * gw[:, None]).sum(axis=0) / gw.sum()


@pytest.fixture
def setup(tmp_path):
    files = [tmp_path / f"a{i}.root" for i in range(3)]
    for i, p in enumerate(files):
        write_nano_file(p, n=700, seed=20 + i)
    fs_dir = tmp_path / "filesets" / "RunII" / "2018" / ERA / "skimmed"
    fs_dir.mkdir(parents=True)
    md = {"datatype": "mc", "physics_group": "tt_tW", "sample": "TT", "xsec": 1.0}
    fileset = {DAS_A: {"files": {"x.root": "Events"}, "metadata": {**md, "das_name": DAS_A}},
               DAS_B: {"files": {"y.root": "Events"}, "metadata": {**md, "das_name": DAS_B, "PSSumw": [1.0, 1.0, 1.0, 1.0]}},
               "/EGamma/Run2018A-UL2018-v1/NANOAOD": {"files": {"d.root": "Events"}, "metadata": {"datatype": "data", "sample": "EGamma_Run2018A"}}}
    (fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").write_text(json.dumps(fileset))
    (fs_dir / f"{ERA}_mc_dy_nlo_inc_fileset.json").write_text(json.dumps({DAS_A: fileset[DAS_A]}))
    (fs_dir / f"{ERA}_data_fileset.json").write_text(json.dumps({"/EGamma/x": fileset["/EGamma/Run2018A-UL2018-v1/NANOAOD"]}))
    (fs_dir / f"{ERA}_signal_scan_fileset.json").write_text(json.dumps({DAS_A: fileset[DAS_A]}))
    return tmp_path, files, fs_dir


def test_pick_files_spreads_over_the_list():
    assert eps.pick_files([f"/store/{i}.root" for i in range(10)], 2) == ["/store/0.root", "/store/5.root"]
    assert eps.pick_files(["/store/b.root", "/store/a.root"], 5) == ["/store/a.root", "/store/b.root"]


def test_estimate_updates_only_entries_without_pssumw(setup):
    tmp_path, files, fs_dir = setup
    dbs = FakeDBS({DAS_A: [str(f) for f in files]})
    cache = eps.update_era(ERA, dbs=dbs, fileset_dir=tmp_path / "filesets", files_per_dataset=2,
                           workers=2, dry_run=False)
    # two of the three files were read, spread over the sorted list
    expected = _direct([files[0], files[1]]) if eps.pick_files([str(f) for f in files], 2) == [str(files[0]), str(files[1])] else None
    res = cache[DAS_A]
    assert res["files"] == 2 and res["events"] == 1400 and len(res["PSSumw"]) == 4
    if expected is not None:
        np.testing.assert_allclose(res["PSSumw"], expected, rtol=1e-6)
    for name in (f"{ERA}_mc_dy_lo_ht_fileset.json", f"{ERA}_mc_dy_nlo_inc_fileset.json"):
        fs = json.loads((fs_dir / name).read_text())
        md = fs[DAS_A]["metadata"]
        assert md["PSSumw"] == pytest.approx(res["PSSumw"]) and md["PSSumw_source"]["events"] == 1400
    lo_ht = json.loads((fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").read_text())
    assert lo_ht[DAS_B]["metadata"]["PSSumw"] == [1.0, 1.0, 1.0, 1.0] and "PSSumw_source" not in lo_ht[DAS_B]["metadata"]
    assert "PSSumw" not in lo_ht["/EGamma/Run2018A-UL2018-v1/NANOAOD"]["metadata"]
    # the data and scan filesets are not touched
    assert "PSSumw" not in json.loads((fs_dir / f"{ERA}_signal_scan_fileset.json").read_text())[DAS_A]["metadata"]


def test_dry_run_reads_and_writes_nothing(setup):
    tmp_path, files, fs_dir = setup
    before = (fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").read_text()
    out = eps.update_era(ERA, dbs=None, fileset_dir=tmp_path / "filesets", files_per_dataset=2,
                         workers=1, dry_run=True)
    assert out == {} and (fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").read_text() == before


def test_sample_without_four_ps_weights_gets_a_status_only(setup, tmp_path):
    _, files, fs_dir = setup
    p = tmp_path / "one.root"
    write_nano_file(p, n=300, seed=9)
    # rewrite PSWeight with a single entry per event
    with uproot.open(p) as f:
        arrays = f["Events"].arrays()
    import awkward as ak
    data = {k: arrays[k] for k in arrays.fields if k != "PSWeight"}
    data["PSWeight"] = ak.unflatten(np.ones(300, dtype=np.float32), np.ones(300, dtype=np.int64))
    with uproot.recreate(p) as f:
        f["Events"] = data
    res = eps.estimate_dataset(DAS_C, dbs=FakeDBS({DAS_C: [str(p)]}), files_per_dataset=1)
    assert "PSSumw" not in res and "1 entries" in res["status"] and res["events"] == 300


def test_unreadable_file_is_a_recorded_failure_and_rerun_resumes_from_the_cache(setup, tmp_path):
    tmp, files, fs_dir = setup
    cache_path = tmp_path / "cache.json"
    # DAS_A: one readable file; a second dataset points at a file that does not exist
    DAS_X = "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_v16_L1v1-v1/NANOAODSIM"
    fs = json.loads((fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").read_text())
    fs[DAS_X] = {"files": {"z.root": "Events"}, "metadata": {"datatype": "mc", "sample": "WZ", "das_name": DAS_X}}
    (fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").write_text(json.dumps(fs))
    dbs = FakeDBS({DAS_A: [str(files[0])], DAS_X: [str(tmp_path / "nowhere.root")]})
    cache = eps.update_era(ERA, dbs=dbs, fileset_dir=tmp / "filesets", files_per_dataset=1, workers=2,
                           dry_run=False, cache_path=cache_path)
    assert "PSSumw" in cache[DAS_A] and cache[DAS_X]["status"].startswith("failed")
    out = json.loads((fs_dir / f"{ERA}_mc_dy_lo_ht_fileset.json").read_text())
    assert "PSSumw" in out[DAS_A]["metadata"] and "PSSumw" not in out[DAS_X]["metadata"]
    assert out[DAS_X]["metadata"]["PSSumw_source"]["status"].startswith("failed")
    assert json.loads(cache_path.read_text())[DAS_A] == cache[DAS_A]
    # rerun: the finished dataset is taken from the cache (no DBS call), the failed one is retried
    calls = []
    class CountingDBS(FakeDBS):
        def files(self, dataset, *, valid_only=True):
            calls.append(dataset); return super().files(dataset, valid_only=valid_only)
    dbs2 = CountingDBS({DAS_A: [str(files[0])], DAS_X: [str(files[1])]})
    cache2 = eps.update_era(ERA, dbs=dbs2, fileset_dir=tmp / "filesets", files_per_dataset=1, workers=1,
                            dry_run=False, cache=dict(cache), cache_path=cache_path)
    assert calls == [DAS_X] and "PSSumw" in cache2[DAS_X]
