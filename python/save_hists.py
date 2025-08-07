import uproot
import os
import logging
from pathlib import Path
import hist
from hist import Hist
from python.preprocess_utils import get_era_details

# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def save_histograms(histograms, args):
    """
    Takes in raw histograms, processes them and saves the output to ROOT files.
    """
    run, year, era = get_era_details(args.era)
    sample = args.sample
    hnwr_mass= args.mass

    # Define working directory and era mapping
    working_dir = Path("WR_Plotter")

    # Build working directory
   # Build working directory
    if getattr(args, 'dir', None):
        output_dir = working_dir / 'rootfiles' / run / year / era / args.dir
    else:
        output_dir = working_dir / 'rootfiles' / run / year / era
    
    if args.exc:
        output_dir = output_dir / 'exclusive'
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename based on sample

    if getattr(args, 'name', None):
        filename_prefix = f"WRAnalyzer_{args.name}"
    else:
        filename_prefix = f"WRAnalyzer"

    if sample == "Signal":
        output_file = output_dir / f"{filename_prefix}_signal_{hnwr_mass}.root"
    else:
        output_file= output_dir / f"{filename_prefix}_{sample}.root"

    # Process histograms
    scaled_hists = scale_hists(histograms)
    summed_hist = sum_hists(scaled_hists)
    split_histograms_dict = split_hists(summed_hist)

    with uproot.recreate(output_file) as root_file:
        for (region, hist_name), hist_obj in split_histograms_dict.items():
            path = f'/{region}/{hist_name}_{region}'
            root_file[path] = hist_obj

    logging.info(f"Histograms saved to {output_file}.")


def scale_hists(data):
    """
    Scale histograms by x_sec/sumw.
    """
    for dataset_key, dataset_info in data.items():
#        if 'x_sec' in dataset_info and 'sumw' in dataset_info and dataset_info['process'] != 'Signal':
        if 'x_sec' in dataset_info and 'sumw' in dataset_info:
            sf = dataset_info['x_sec']/dataset_info['sumw']
            for key, value in dataset_info.items():
                if isinstance(value, Hist):
                    value *= sf
        else:
            logging.warning(f"Dataset {dataset_key} missing 'x_sec' or 'sumw'. Skipping scaling.")
    return data

def sum_hists(my_hists):
    """
    Sum histograms across datasets (e.g. Merge all of the HT binned DY histograms into a single DYJets).
    """
    if not my_hists:
        raise ValueError("No histogram data provided.")

    original_histograms = list(my_hists.values())[0]
    sum_histograms = {
        key: Hist(*original_histograms[key].axes, 
            storage=original_histograms[key].storage_type())
        for key in original_histograms
        if isinstance(original_histograms[key], Hist)
    }

    for dataset_info in my_hists.values():
        for key, value in dataset_info.items():
            if isinstance(value, Hist):
                hist_name = key
                hist_data = value
                if hist_name in sum_histograms:
                    sum_histograms[hist_name] += hist_data
                else:
                    sum_histograms[hist_name] = hist_data.copy()

    return sum_histograms

def split_hists(summed_hists):
    """
    Take the hist object and split it into seperate histogram (for example, make seperate histograms for ee and mumu).
    """
    split_histograms = {}

    for hist_name, sum_hist in summed_hists.items():
        try:
            process_axis = sum_hist.axes['process']
            regions_axis = sum_hist.axes['region']
        except KeyError as e:
            logging.error(f"Missing expected axis in histogram '{hist_name}': {e}")
            continue

        unique_processes = [process_axis.value(i) for i in range(process_axis.size)]
        unique_regions = [regions_axis.value(i) for i in range(regions_axis.size)]

        for process in unique_processes:
            for region in unique_regions:
                sub_hist = sum_hist[{process_axis.name: process, regions_axis.name: region}]
                key = (region, hist_name)
                split_histograms[key] = sub_hist

    return split_histograms
