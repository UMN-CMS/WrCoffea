import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
import argparse
import time
import json
import logging
import csv
from pathlib import Path
from coffea.nanoevents import NanoAODSchema
from coffea.dataset_tools import apply_to_fileset, max_chunks, max_files
import sys
import os

from dask.distributed import Client, LocalCluster
from coffea.processor import Runner, DaskExecutor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import ProcessorABC

# Add the src/, data/, and python/ directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python`')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer import WrAnalysis
#from dask.distributed import Client
#from dask.diagnostics import ProgressBar
#import dask
import uproot
from python.save_hists import save_histograms
from python.preprocess_utils import get_era_details, load_json

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.error_missing_event_ids = False

def load_masses_from_csv(file_path):
    mass_choices = []
    try:
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  
            for row in csv_reader:
                if len(row) >= 2:
                    wr_mass = row[0].strip()
                    n_mass = row[1].strip()
                    mass_choice = f"WR{wr_mass}_N{n_mass}"
                    mass_choices.append(mass_choice)
    except FileNotFoundError:
        logging.error(f"Mass CSV file not found at: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        raise
    return mass_choices

def filter_by_process(fileset, desired_process, mass=None):
    if desired_process == "Signal":
        return {ds: data for ds, data in fileset.items() if mass in data['metadata']['sample']}
    else:
        return {ds: data for ds, data in fileset.items() if data['metadata']['physics_group'] == desired_process}

def validate_arguments(args, sig_points):
    if args.sample == "Signal" and not args.mass:
        logging.error("For 'Signal', you must provide a --mass argument (e.g. --mass WR2000_N1900).")
        raise ValueError("Missing mass argument for Signal sample.")
    if args.sample == "Signal" and args.mass not in sig_points:
        logging.error(f"The provided signal point {args.mass} is not valid. Choose from {sig_points}.")
        raise ValueError("Invalid mass argument for Signal sample.")
    if args.sample != "Signal" and args.mass:
        logging.error("The --mass option is only valid for 'Signal' samples.")
        raise ValueError("Mass argument provided for non-signal sample.")
    if args.reweight and args.sample != "DYJets":
        logging.error("Reweighting can only be applied to DY")
        raise ValueError("Invalid sample for reweighting.")

def run_analysis(args, filtered_fileset):

    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    client = Client(cluster)

    run = Runner(
        executor = DaskExecutor(client=client, compression=None),
        chunksize=50_000,
        maxchunks = 1,
        skipbadfiles=True,
        xrootdtimeout = 60,
        align_clusters = False,
        savemetrics=True,
        schema=NanoAODSchema,
    )

    print(f"\n***FILTERED FILESET***\n\n{filtered_fileset}")

    print(f"\n***PREPROCESSING***\n")

    preproc_debug = run.preprocess(filtered_fileset, treename="Events")
    preproc_list = list(preproc_debug)
    print(preproc_list)
    print()

    preproc_for_run = run.preprocess(fileset=filtered_fileset, treename="Events")
    to_compute, metrics = run(
        preproc_for_run, 
        processor_instance=WrAnalysis(mass_point=None),
    )
    print(f"\n***ANALYZER OUTPUT***\n{to_compute}")

    print(f"\n***METRICS***\n{metrics}\n")

#    to_compute = apply_to_fileset(
#        data_manipulation=WrAnalysis(mass_point=args.mass, sf_file=args.reweight),
#        fileset=max_files(max_chunks(filtered_fileset,1),1),
#        schemaclass=NanoAODSchema,
#        uproot_options={"handler": uproot.MultiThreadedXRootDSource, "timeout": 60}
    return to_compute

def save_hists(to_compute):
    logging.info("Computing histograms...")
#    with ProgressBar():
#        (histograms,) = dask.compute(to_compute)
    print()
    print(to_compute)
    save_histograms(to_compute, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing script for WR analysis.")
    parser.add_argument("era", type=str, choices=["RunIISummer20UL18", "Run3Summer22", "Run3Summer22EE"], help="Campaign to analyze.")
    parser.add_argument("sample", type=str, choices=["DYJets", "TTbar", "TW", "WJets", "SingleTop", "TTbarSemileptonic", "TTV", "Diboson", "Triboson", "EGamma", "Muon", "Signal"], help="MC sample to analyze (e.g., Signal, DYJets).")
    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument("--mass", type=str, default=None, help="Signal mass point to analyze.")
    optional.add_argument("--dir", type=str, default=None, help="Create a new output directory.")
    optional.add_argument("--name", type=str, default=None, help="Append the filenames of the output ROOT files.")
    optional.add_argument("--debug", action='store_true', help="Debug mode (don't compute histograms)")
    optional.add_argument("--reweight", type=str, default=None, help="Path to json file of DY reweights")
    args = parser.parse_args()

    signal_points = Path(f'data/{args.era}_mass_points.csv')
    MASS_CHOICES = load_masses_from_csv(signal_points)

    print()
    logging.info(f"Analyzing {args.era} - {args.sample} events")
    
    validate_arguments(args, MASS_CHOICES)
    run, year, era = get_era_details(args.era)

    if "EGamma" in args.sample or "Muon" in args.sample:
        filepath = Path("data/jsons") / run / year / era / "skimmed" / f"{era}_data_skimmed_fileset.json"
    elif "Signal" in args.sample:
        filepath = Path("data/jsons") / run / year / era / "skimmed" / f"{era}_signal_skimmed_fileset.json"
    else:
        filepath = Path("data/jsons") / run / year / era / "skimmed" / f"{era}_mc_lo_dy_skimmed_fileset.json"

    preprocessed_fileset = load_json(str(filepath))
    filtered_fileset = filter_by_process(preprocessed_fileset, args.sample, args.mass)

    t0 = time.monotonic()
    to_compute = run_analysis(args, filtered_fileset)

    print(to_compute)
    if not args.debug:
        save_hists(to_compute)
    exec_time = time.monotonic() - t0
    logging.info(f"Execution took {exec_time/60:.2f} minutes")
