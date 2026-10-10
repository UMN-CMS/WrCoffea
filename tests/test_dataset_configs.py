"""Checks on the dataset configs under data/configs (the archived ones excepted)."""
import json
from pathlib import Path

CONFIGS = sorted(p for p in (Path(__file__).resolve().parent.parent / "data" / "configs").rglob("*.json")
                 if "archived" not in p.parts)


def test_configs_carry_no_genEventSumw():
    """The filesets take genEventSumw from the skims (skim.py fileset); a config copy only goes stale."""
    offenders = [f"{p.relative_to(p.parents[3])}: {key}" for p in CONFIGS
                 for key, entry in json.loads(p.read_text()).items()
                 if isinstance(entry, dict) and "genEventSumw" in entry]
    assert not offenders, offenders[:5]
