"""scripts/pdf_sets.py: the per-era table of each sample's LHAPDF set."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pdf_sets", REPO / "scripts" / "pdf_sets.py")
pdf_sets = importlib.util.module_from_spec(spec)
sys.modules["pdf_sets"] = pdf_sets
spec.loader.exec_module(pdf_sets)

ERA = "RunIII2024Summer24"


def _args(out, **extra):
    base = dict(era=ERA, fileset_dir="unused", config_dir="unused", das_command=None, tries=1,
                redirector="root://x/", workers=2, out=str(out), refresh=False)
    base.update(extra)
    return SimpleNamespace(**base)


def _patch(monkeypatch, samples):
    read = []
    monkeypatch.setattr(pdf_sets, "mc_samples", lambda *a: {s: {"LHEPdfSumw": [1.0] * n} for s, n in samples.items()})
    monkeypatch.setattr(pdf_sets, "private_files", lambda *a: {s: [f"root://x//{s}.root"] for s in samples})

    def read_lhaid(url):
        read.append(url)
        n = samples[url.rsplit("/", 1)[1][:-5]]
        first = 325300 if n == 103 else 325500
        return first, first + n - 1, n

    monkeypatch.setattr(pdf_sets, "read_lhaid", read_lhaid)
    return read


def test_rerun_reads_only_the_samples_the_table_lacks(tmp_path, monkeypatch):
    out = tmp_path / "table.json"
    read = _patch(monkeypatch, {"A": 103, "B": 101})
    assert pdf_sets.scan(_args(out)) == 0
    assert sorted(read) == ["root://x//A.root", "root://x//B.root"]
    table = json.loads(out.read_text())
    assert table["samples"]["A"]["lhaid"] == 325300 and table["samples"]["B"] == {
        "lhaid": 325500, "lhaid_last": 325600, "n": 101, "source": "root://x//B.root"}
    assert table["sets"] == {"325300": "NNPDF31_nnlo_as_0118_mc_hessian_pdfas",
                             "325500": "NNPDF31_nnlo_as_0118_nf_4_mc_hessian"}

    # a new sample C joins, B leaves the filesets: only C is read, A keeps its entry, B leaves the table
    read = _patch(monkeypatch, {"A": 103, "C": 103})
    assert pdf_sets.scan(_args(out)) == 0
    assert read == ["root://x//C.root"]
    assert sorted(json.loads(out.read_text())["samples"]) == ["A", "C"]

    # --refresh reads every sample again
    read = _patch(monkeypatch, {"A": 103, "C": 103})
    assert pdf_sets.scan(_args(out, refresh=True)) == 0
    assert sorted(read) == ["root://x//A.root", "root://x//C.root"]


def test_a_failure_writes_nothing(tmp_path, monkeypatch, capsys):
    out = tmp_path / "table.json"
    _patch(monkeypatch, {"A": 103})

    def broken(url):
        raise RuntimeError("no file")

    monkeypatch.setattr(pdf_sets, "read_lhaid", broken)
    assert pdf_sets.scan(_args(out)) == 1
    assert not out.exists() and "FAILED" in capsys.readouterr().err
