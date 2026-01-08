#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

# Pattern for keys and fields like "WRtoNLtoLLJJ_WR600_N100"
PATTERN = re.compile(r"^(WRtoNLtoLLJJ)_WR(\d+)_N(\d+)$")


def rename_label(label: str) -> str:
    """
    Convert 'WRtoNLtoLLJJ_WR600_N100' -> 'WRtoNLtoLLJJ_MWR600_MN100'
    If it doesn't match the pattern, return unchanged.
    """
    m = PATTERN.match(label)
    if not m:
        return label
    prefix, mwr, mn = m.groups()
    return f"{prefix}_MWR{mwr}_MN{mn}"


def main():
    parser = argparse.ArgumentParser(
        description="Rename WR signal labels in JSON from WR..._N... to MWR..._MN..."
    )
    parser.add_argument(
        "--in-json",
        required=True,
        help="Input JSON file (e.g. RunIISummer20UL18_signal.json)",
    )
    parser.add_argument(
        "--out-json",
        help="Output JSON file (default: overwrite input)",
    )
    args = parser.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_json) if args.out_json else in_path

    with in_path.open() as f:
        data = json.load(f)

    new_data = {}

    for old_key, entry in data.items():
        new_key = rename_label(old_key)

        # Copy entry and fix das_name / dataset if they match the pattern
        entry = dict(entry)  # shallow copy

        if "das_name" in entry:
            entry["das_name"] = rename_label(entry["das_name"])

        if "dataset" in entry:
            entry["dataset"] = rename_label(entry["dataset"])

        # If you also have "sample" fields later, uncomment this:
        # if "sample" in entry:
        #     entry["sample"] = rename_label(entry["sample"])

        new_data[new_key] = entry

    with out_path.open("w") as f:
        json.dump(new_data, f, indent=4, sort_keys=True)

    print(f"Renamed {len(data)} entries")
    print(f"Wrote updated JSON to {out_path}")


if __name__ == "__main__":
    main()
