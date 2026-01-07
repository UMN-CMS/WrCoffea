from pathlib import Path
import json
import logging
import difflib

# Mapping of eras to dataset paths
ERA_MAPPING = {
    "RunIISpring16": {"run": "RunII", "year": "2016"},
    "RunIIAutumn18": {"run": "RunII", "year": "2018"},
    "RunIISummer20UL16": {"run": "RunII", "year": "2016"},
    "RunIISummer20UL17": {"run": "RunII", "year": "2017"},
    "RunIISummer20UL18": {"run": "RunII", "year": "2018"},
    "Run3Summer22": {"run": "Run3", "year": "2022"},
    "Run3Summer22EE": {"run": "Run3", "year": "2022"},
    "Run3Summer23": {"run": "Run3", "year": "2023"},
    "Run3Summer23BPix": {"run": "Run3", "year": "2023"},
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
#            logging.info(f"Successfully loaded JSON file: {filepath}")
            return data
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file {filepath}: {e}") from e

def save_json(output_file, data, data_all):
    """
    Save the processed JSON data to a file. If data and data_all are different,
    throw an error and output the differences. If they are the same, save only data.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data != data_all:
        data_str = json.dumps(data, indent=4, sort_keys=True)
        data_all_str = json.dumps(data_all, indent=4, sort_keys=True)

        diff = difflib.unified_diff(
            data_str.splitlines(), data_all_str.splitlines(),
            fromfile='data', tofile='data_all', lineterm=''
        )

        diff_output = '\n'.join(diff)

        diff_path = output_path.with_suffix(output_path.suffix + ".diff")
        with open(diff_path, "w", encoding="utf-8") as f:
            f.write(diff_output)

        logging.error(
            "Aborting save: 'data' and 'data_all' differ. Wrote unified diff to %s",
            diff_path,
        )

        raise ValueError(
            "Aborting save due to differences between 'data' and 'data_all'. "
            f"See diff: {diff_path}"
        )

    if output_path.exists():
        logging.warning(f"{output_file} already exists, overwriting.")

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)
    logging.info(f"JSON data successfully saved to {output_file}")
