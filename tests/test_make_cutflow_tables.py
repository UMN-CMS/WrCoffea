import importlib.util
from pathlib import Path


def _load_make_cutflow_tables_module():
    module_path = Path(__file__).resolve().parents[1] / "WR_Plotter" / "bin" / "make_cutflow_tables.py"
    spec = importlib.util.spec_from_file_location("make_cutflow_tables", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compare_data_era1_to_mc_era2_uses_data_and_dyjets(monkeypatch, tmp_path):
    module = _load_make_cutflow_tables_module()
    calls = []

    def fake_process_dy_pair(era1_root_path, era2_root_path, output_dir, era1, era2):
        calls.append((Path(era1_root_path), Path(era2_root_path), Path(output_dir), era1, era2))
        return True

    era1_dir = tmp_path / "era1"
    era2_dir = tmp_path / "era2"
    era1_dir.mkdir()
    era2_dir.mkdir()
    (era1_dir / "WRAnalyzer_Muon.root").write_text("data", encoding="utf-8")
    (era2_dir / "WRAnalyzer_DYJets.root").write_text("mc", encoding="utf-8")

    monkeypatch.setattr(module, "process_dy_pair", fake_process_dy_pair)

    result = module.compare_data_era1_to_mc_era2(
        era1_dir,
        era2_dir,
        tmp_path / "out",
        "Run3Summer22",
        "Run3Summer23",
    )

    assert result == 1
    assert len(calls) == 1
    assert calls[0][0].name == "WRAnalyzer_Muon.root"
    assert calls[0][1].name == "WRAnalyzer_DYJets.root"
    assert calls[0][2].name == "out"
