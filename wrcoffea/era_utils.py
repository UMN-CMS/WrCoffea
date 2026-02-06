from pathlib import Path
import json
import logging
# Mapping of eras to dataset paths
ERA_MAPPING = {
    "RunIISummer20UL18": {"run": "RunII", "year": "2018"},
    "Run3Summer22": {"run": "Run3", "year": "2022"},
    "Run3Summer22EE": {"run": "Run3", "year": "2022"},
    "RunIII2024Summer24": {"run": "Run3", "year": "2024"},
}

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

