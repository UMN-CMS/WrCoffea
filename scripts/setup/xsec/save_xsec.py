import sys
import os
import argparse
import logging
from pathlib import Path

# ------------------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Add the repo root to sys.path so "from wrcoffea.era_utils import ..." works
# ------------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, repo_root)

from wrcoffea.era_utils import get_era_details

def extract_xsec_summary(input_file, output_file):
    """Extracts the GenXsecAnalyzer section from the log and saves it to output_file."""
    
    if not os.path.isfile(input_file):
        logger.error(f"Input log file '{input_file}' does not exist.")
        sys.exit(1)

    logger.info(f"Reading input log file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
        lines = infile.readlines()

    start_index = None
    for i, line in enumerate(lines):
        if "GenXsecAnalyzer:" in line:
            start_index = max(0, i - 6)  # Ensure exactly 6 lines above are included
            break

    if start_index is None:
        logger.error("Error: 'GenXsecAnalyzer' section not found in the log file.")
        sys.exit(1)

    extracted_lines = lines[start_index:]

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(extracted_lines)

    logger.info(f"Extracted {len(extracted_lines)} lines and saved log to:\n   {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract the 'GenXsecAnalyzer' section from a log file and save it."
    )
    parser.add_argument(
        "era",
        type=str,
        help="MC Campaign (e.g., UL18, UL17, Run3_2022, etc.)"
    )
    parser.add_argument(
        "input_log",
        type=str,
        help="Path to the input log file."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose mode for additional logging."
    )

    args = parser.parse_args()

    # Enable debug logging if verbose mode is set
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Get era details (assuming get_era_details returns (run, year, era))
    run, year, era = get_era_details(args.era)

    logger.info(f"MC Campaign Details: Run={run}, Year={year}, Era={era}")

    # Define the output directory and ensure it exists
    output_dir = Path(f'/uscms/home/bjackson/nobackup/WrCoffea/data/xsec/{run}/{year}/{era}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract base filename from input_log
    base_name = os.path.basename(args.input_log)

    # Define output file path
    output_log = output_dir / base_name
    
    # Run extraction
    extract_xsec_summary(args.input_log, output_log)

if __name__ == "__main__":
    main()
