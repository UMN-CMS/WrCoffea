#!/usr/bin/env python3

import argparse
import sys
import subprocess
import time
import logging
from pathlib import Path
import os

# ------------------------------------------------------------------------------
# Hardcoded base directory string
# ------------------------------------------------------------------------------
BASE_DIR = "/uscms/home/bjackson/nobackup/"

# ------------------------------------------------------------------------------
# Add the repo root to sys.path so "from python.era_utils import get_era_details" works
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, repo_root)

from python.preprocess_utils import get_era_details

# ------------------------------------------------------------------------------
# Configure logging
# ------------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# ------------------------------------------------------------------------------
# Function: run_command_with_retries
# ------------------------------------------------------------------------------
def run_command_with_retries(command, retries=3, delay=5, timeout=120):
    """
    Runs a subprocess command with retries if it times out.

    Args:
        command (list): The command to run as a list of arguments.
        retries (int): Number of retries allowed.
        delay (int): Delay (in seconds) between retries.
        timeout (int): Timeout (in seconds) for each command execution.

    Returns:
        subprocess.CompletedProcess: The result of the successful command execution.

    Raises:
        RuntimeError: If the command fails after all retries.
    """
    attempt = 0
    while attempt < retries:
        try:
            logging.info(f"Attempt {attempt + 1} of {retries}: Running command: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)

            if result.returncode == 0:
                return result  # Return the result if successful

            logging.error(f"Command failed with return code {result.returncode}: {result.stderr}")
            break  # Stop retrying if the command itself failed

        except subprocess.TimeoutExpired:
            logging.warning(f"Command timed out on attempt {attempt + 1}. Retrying in {delay} seconds...")
            attempt += 1
            time.sleep(delay)

    raise RuntimeError(f"Command failed after {retries} retries: {' '.join(command)}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates lists of MINIAOD files for a given dataset era using DAS queries."
    )
    parser.add_argument(
        "era",
        type=str,
        choices=[
            "RunIISpring16",
            "RunIIAutumn18", "RunIISummer20UL16", "RunIISummer20UL17", "RunIISummer20UL18",
            "Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix"
        ],
        help="Era (e.g., RunIISummer20UL16, Run3Summer22, etc.)"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to the input dataset list file (default: auto-determined from era)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to the output directory (default: auto-determined from era)"
    )
    args = parser.parse_args()

    # Get era details (assuming get_era_details returns (run, year, era))
    run, year, era = get_era_details(args.era)

    # Determine input/output based on arguments or defaults
    default_input_file = Path(f"{BASE_DIR}/WrCoffea/data/xsec/miniaod/{run}/{year}/{era}_datasets.txt")
    default_output_dir = Path(f"{BASE_DIR}/WrCoffea/data/xsec/miniaod/{run}/{year}/{era}")

    input_file = args.input_file if args.input_file else default_input_file
    output_dir = args.output_dir if args.output_dir else default_output_dir

    logging.info(f"Using era: {era} (Run: {run}, Year: {year})")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output directory: {output_dir}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset list
    try:
        with open(input_file, 'r') as f:
            dataset_paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        logging.error(f"Input dataset file not found: {input_file}")
        sys.exit(1)

    if not dataset_paths:
        logging.warning("No datasets found in the input file. Exiting.")
        sys.exit(0)

    us_redirector = "root://cmsxrootd.fnal.gov/"
    for dataset in dataset_paths:
        logging.info(f"Finding files for dataset: {dataset}")
        try:
            # Run the command with retries
            result = run_command_with_retries(
                ["dasgoclient", "-query", f"file dataset={dataset}"],
                retries=10,
                delay=5,
                timeout=10
            )

            files = result.stdout.strip().split("\n")
            files = [x for x in files if x]  # Remove empty lines

            if not files:
                logging.warning(f"No files returned by DAS for dataset: {dataset}. Skipping.")
                continue

            # Prepend the redirector
            prepended_files = [f"{us_redirector}{x}" for x in files]

            # Derive output file name
            try:
                dataset_name = dataset.split("/")[1]
            except IndexError:
                dataset_name = "unknown_dataset"

            output_txt_path = output_dir / f"{dataset_name}_MINIAOD_files.txt"
            # Write the file paths
        
            with open(output_txt_path, "w") as txt_file:
                for line in prepended_files:
                    txt_file.write(f"{line}\n")

            logging.info(f"Saved {len(prepended_files)} file paths to {output_txt_path}")

        except RuntimeError as e:
            logging.error(f"Error processing dataset {dataset}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing dataset {dataset}: {e}")
