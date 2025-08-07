#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path

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

    jobdir = f"/uscms_data/d1/bjackson/WrCoffea/scripts/setup/skims/tmp/{run}/{year}/{campaign}"
    # Define base directory
    base_path = Path(f"/uscms_data/d1/bjackson/WrCoffea/data/filepaths/{run}/{year}/{campaign}")

    # Ensure base directory exists
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist.")
        return

    # Build argument list
#    print("Filelist:")
    arguments = []
    counter = 1

    for filename in os.listdir(base_path):
#        print(f"filename: {filename}")
#        print("dataset", dataset)
        if filename == f"{dataset}.txt":
            file_path = base_path / filename  # Use Path object

            try:
                with file_path.open("r") as f:
                    lines = f.readlines()
                    for line in lines:
#                        print(line)
                        arguments.append(f"{counter} {campaign} {process} {dataset} {line.strip()}\n")
                        counter += 1
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    print(f"Number of jobs: {len(arguments)}")

    # Create jobdir and subdirectories
    jobdir = Path(jobdir) / dataset
    print(f"Jobdir: {jobdir}")

    outdir = f"/uscms_data/d1/bjackson/WrCoffea/data/skims/{run}/{year}/{campaign}" 
    outdir = Path(outdir)/ dataset
    for subdir in ["", f"{dataset}_out", f"{dataset}_log", f"{dataset}_err"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)
#        (jobdir / subdir).mkdir(parents=True, exist_ok=True)


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
####

    # Read the original hadd_files.py content
    try:
        haddfile_content = Path("hadd_files.sh").read_text()

        # Replace argument handling with a hardcoded dataset
        haddfile_content = haddfile_content.replace(
            '# Ensure an argument is provided\nif [ "$#" -ne 1 ]; then\n    echo "Usage: $0 <dataset_name>"\n    exit 1\nfi\n\n# Get the dataset name from the argument\nDATASET_NAME="$1"',
            f'DATASET_NAME="{dataset}"'
        )

        # Write the modified hadd_files.py to jobdir
        hadd_path = jobdir / "hadd_files.sh"
        hadd_path.write_text(haddfile_content)
        hadd_path.chmod(0o755)  # Make it executable

    except Exception as e:
        print(f"Error reading or modifying hadd_files.py: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create job submission files for HTCondor.")
    parser.add_argument("campaign", help="The MC Campaign (Run2Summer20UL18, Run3 etc.")
    parser.add_argument("process", help="The process name (e.g., DYJets, TTbar, etc.)")
    parser.add_argument("dataset", help="The dataset name")

    args = parser.parse_args()
    main(args.campaign, args.process, args.dataset)
