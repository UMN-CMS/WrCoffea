from pathlib import Path
import json
import logging
# Mapping of eras to dataset paths
ERA_MAPPING = {
    "RunIISummer20UL16APV": {"run": "RunII", "year": "2016"},
    "RunIISummer20UL16": {"run": "RunII", "year": "2016"},
    "RunIISummer20UL17": {"run": "RunII", "year": "2017"},
    "RunIISummer20UL18": {"run": "RunII", "year": "2018"},
    "Run3Summer22": {"run": "Run3", "year": "2022"},
    "Run3Summer22EE": {"run": "Run3", "year": "2022"},
    "Run3Summer23": {"run": "Run3", "year": "2023"},
    "Run3Summer23BPix": {"run": "Run3", "year": "2023"},
    "RunIII2024Summer24": {"run": "Run3", "year": "2024"},
    # 2025 is a data-only era: PdmV recommends the RunIII2024Summer24 MC for it.
    "RunIII2025Summer25": {"run": "Run3", "year": "2025"},
}

# era -> "<run>/<year>/<era>", the directory layout of data/configs, data/filesets, the skims and the
# analyzer outputs.  The single definition; wrcoffea.das_utils and wrcoffea.skim.plan re-export it.
ERA_SUBDIRS = {era: f"{m['run']}/{m['year']}/{era}" for era, m in ERA_MAPPING.items()}


def era_subdir(era):
    """``<run>/<year>/<era>`` of a known era (ValueError otherwise)."""
    if era not in ERA_SUBDIRS:
        raise ValueError(f"Unsupported era: {era}. Valid eras: {sorted(ERA_SUBDIRS)}")
    return ERA_SUBDIRS[era]


def get_era_details(era):
    """
    Retrieves the 'run' and 'year' associated with a given era.
    """
    mapping = ERA_MAPPING.get(era)
    if mapping is None:
        raise ValueError(f"Unsupported era: {era}. Valid eras: {sorted(ERA_MAPPING)}")
    
    return mapping["run"], mapping["year"], era

def load_json(filepath):
    """
    Load JSON data from the specified file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logging.info(f"Successfully loaded JSON file: {filepath}")
            return data
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file {filepath}: {e}") from e

