#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import json

jdl = """\
universe = vanilla
executable = ./{PROCESS}.sh
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
request_memory = 20000
output = ../../../../../../../../data/skims/{RUN}/{YEAR}/{CAMPAIGN}/{PROCESS}/{PROCESS}_out/{PROCESS}_$(ProcId).out
error = ../../../../../../../../data/skims/{RUN}/{YEAR}/{CAMPAIGN}/{PROCESS}/{PROCESS}_err/{PROCESS}_$(ProcId).err
log = ../../../../../../../../data/skims/{RUN}/{YEAR}/{CAMPAIGN}/{PROCESS}/{PROCESS}_log/{PROCESS}_$(ProcId).log
transfer_input_files = WrCoffea.tar.gz
transfer_output_files = {PROCESS}_skim$(ProcId).tar.gz
queue arguments from arguments.txt\
"""
#20000

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main(campaign, process, dataset):
    print(f"\nStarting job creation for dataset: {dataset}")
#    run = campaign[:4]
    if campaign == "Run3Summer22" or campaign == "Run3Summer22EE":
        run = "Run3"
        year = "2022"
    elif campaign == "Run3Summer23" or campaign == "Run3Summer23BPix":
        run = "Run3"
        year = "2023"
    elif campaign == "RunIISummer20UL18":
        run = "RunII"
        year = "2018"
    elif campaign == "RunIII2024Summer24":
        run = "Run3"
        year = "2024"

    jobdir = f"/uscms_data/d1/bjackson/WrCoffea/scripts/setup/skims/tmp/{run}/{year}/{campaign}"

    # Build argument list
#    print("Filelist:")
    arguments = []
    counter = 1

    fileset_path = Path(f"/uscms_data/d1/bjackson/WrCoffea/data/jsons/{run}/{year}/{campaign}/unskimmed/{campaign}_{process}_fileset.json")

    try:
        with fileset_path.open("r") as jf:
            fileset = json.load(jf)
    except Exception as e:
        raise RuntimeError(f"Failed to read fileset JSON at {fileset_path}: {e}")

    # Pick the entry matching this dataset (by metadata.sample or key containing /<dataset>/)
    entry = None
    for key, val in fileset.items():
        if val.get("metadata", {}).get("sample") == dataset or f"/{dataset}/" in key:
            entry = val
            break

    # Fallback: use the first (and warn) if nothing matched explicitly
    if entry is None:
        key0 = next(iter(fileset))
        entry = fileset[key0]
        print(f"[warn] No JSON entry matched dataset='{dataset}'. Using first entry: {key0}")

    # Extract file URLs (keys of the "files" dict) and build arguments
    file_urls = sorted(entry.get("files", {}).keys())
    if not file_urls:
        raise RuntimeError(f"No 'files' found in fileset for dataset '{dataset}' at {fileset_path}")

    for url in file_urls:
        arguments.append(f"{counter} {campaign} {process} {dataset} {url}\n")
        counter += 1

    print(f"Number of jobs: {len(arguments)}")

    # Create jobdir and subdirectories
    jobdir = Path(jobdir) / dataset
    print(f"Jobdir: {jobdir}")

    outdir = f"/uscms_data/d1/bjackson/WrCoffea/data/skims/{run}/{year}/{campaign}" 
    outdir = Path(outdir)/ dataset
    for subdir in ["", f"{dataset}_out", f"{dataset}_log", f"{dataset}_err"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    # Write jdl file
    jdl_path = jobdir / "job.jdl"
    with jdl_path.open("w") as out:
        out.write(jdl.format(RUN=run, YEAR=year, CAMPAIGN=campaign, PROCESS=dataset))

    # Write argument list
    arglist_path = jobdir / "arguments.txt"
    with arglist_path.open("w") as arglist:
        arglist.writelines(arguments)

    # Write job file
    job_script_path = jobdir / f"{dataset}.sh"
    try:
        jobfile = Path("job.sh").read_text()
        job_script_path.write_text(jobfile)
    except Exception as e:
        print(f"Error reading job.sh: {e}")

    # Read the original unzip_files.py content
    try:
        unzip_content = Path("unzip_files.sh").read_text()

        # Replace argument handling with a hardcoded dataset
        unzip_content = unzip_content.replace(
            '# Ensure an argument is provided\nif [ "$#" -ne 1 ]; then\n    echo "Usage: $0 <dataset_name>"\n    exit 1\nfi\n\n# Get the dataset name from the argument\nDATASET_NAME="$1"',
            f'DATASET_NAME="{dataset}"'
        )

        # Write the modified unzip_files.py to jobdir
        unzip_path = jobdir / "unzip_files.sh"
        unzip_path.write_text(unzip_content)
        unzip_path.chmod(0o755)  # Make it executable

    except Exception as e:
        print(f"Error reading or modifying unzip_files.py: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create job submission files for HTCondor.")
    parser.add_argument("campaign", help="The MC Campaign (Run2Summer20UL18, Run3 etc.")
    parser.add_argument("process", help="The process name (e.g., DYJets, TTbar, etc.)")
    parser.add_argument("dataset", help="The dataset name")

    args = parser.parse_args()
    main(args.campaign, args.process, args.dataset)
