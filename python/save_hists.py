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

# --- Add these helpers ---

def _normalize_syst_name(syst: str) -> str:
    """
    Turn 'RenFactScaleUp' -> 'renfactscaleup' (root-key friendly, lowercase).
    """
    return "".join(ch.lower() for ch in syst if ch.isalnum())

def _folder_and_hist_names(region: str, syst: str, hist_stem: str):
    """
    For syst == 'Nominal':
        folder:  <region>/
        hist:    <hist_stem>_<region>
    For syst != 'Nominal':
        folder:  syst_<norm>_<region>/
        hist:    <hist_stem>_syst_<norm>_<region>
    """
    if syst == "Nominal":
        folder = f"{region}"
        hname  = f"{hist_stem}_{region}"
    else:
        norm = _normalize_syst_name(syst)
        folder = f"syst_{norm}_{region}"
        hname  = f"{hist_stem}_syst_{norm}_{region}"
    return folder, hname


def split_hists_with_syst(summed_hists, *, sum_over_process=True):
    out = {}
    for hist_name, h in summed_hists.items():
        try:
            region_ax = h.axes["region"]
            syst_ax   = h.axes["syst"]
        except KeyError as e:
            logging.error("Missing expected axis in histogram '%s': %s", hist_name, e)
            continue

        has_proc = any(ax.name == "process" for ax in h.axes)

        regions = [region_ax.value(i) for i in range(region_ax.size)]
        systs   = [syst_ax.value(i)   for i in range(syst_ax.size)]

        for reg in regions:
            for sy in systs:
                # Slice first
                hh = h[{region_ax.name: reg, syst_ax.name: sy}]

                # Then optionally remove the 'process' axis
                if sum_over_process and has_proc and any(ax.name == "process" for ax in hh.axes):
                    idxs_to_keep = [i for i, ax in enumerate(hh.axes) if ax.name != "process"]
                    if len(idxs_to_keep) < hh.ndim:
                        hh = hh.project(*idxs_to_keep)

                out[(reg, sy, hist_name)] = hh
    return out

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
#    scaled_hists = scale_hists(histograms)

    summed_hist = sum_hists(histograms)
# summed_hist = sum_hists(histograms)          # you already have this
    split_histograms_dict = split_hists_with_syst(summed_hist, sum_over_process=True)

    with uproot.recreate(output_file) as root_file:
        for (region, syst, hist_name), hist_obj in split_histograms_dict.items():
            # Use the *variable axis* name (e.g. 'phi_leadlep') as the stem:
            # We assume your last axis is the variable axis and its name is what you want on disk.
            # If you prefer to use 'hist_name' literally, replace 'var_stem' with 'hist_name'.
            try:
                # Grab the 1D/ND variable axis name(s); here we use the first non-category axis.
                var_axes = [ax for ax in hist_obj.axes if ax.__class__.__name__ != "StrCategory"]
                var_stem = var_axes[0].name if var_axes else hist_name
            except Exception:
                var_stem = hist_name

            folder, hname = _folder_and_hist_names(region, syst, var_stem)

            # Examples:
            # Nominal:            'wr_ee_resolved_dy_cr/phi_leadlep_wr_ee_resolved_dy_cr'
            # RenFactScaleUp:     'syst_renfactscaleup_wr_ee_resolved_dy_cr/phi_leadlep_syst_renfactscaleup_wr_ee_resolved_dy_cr'
            path = f"/{folder}/{hname}"
            root_file[path] = hist_obj

#    split_histograms_dict = split_hists(summed_hist)

#    with uproot.recreate(output_file) as root_file:
#        for (region, hist_name), hist_obj in split_histograms_dict.items():
#            path = f'/{region}/{hist_name}_{region}'
 #           root_file[path] = hist_obj

    logging.info(f"Histograms saved to {output_file}.")


def scale_hists(data):
    """
    Scale histograms by x_sec/sumw.
    """
    for dataset_key, dataset_info in data.items():
#        if 'x_sec' in dataset_info and 'sumw' in dataset_info and dataset_info['process'] != 'Signal':
        if 'x_sec' in dataset_info and 'sumw' in dataset_info:
            sf = dataset_info['x_sec']/dataset_info['nevts']
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
