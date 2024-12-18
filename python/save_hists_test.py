import uproot
import os
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import progress
import hist
from hist import Hist

def save_histograms(my_histograms, args):
    run = args.run
    sample = args.sample
    hnwr_mass= args.mass
    
    working_dir = "/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter"
    
    if args.run == "Run3Summer22":
        dataset = "Run3"
        year = "2022"
    elif args.run == "Run2Summer20UL18":
        dataset = "Run2UltraLegacy"
        year = "2018"
    elif args.run == "Run2Autumn18":
        dataset = "Run2Legacy"
        year = "2018"
    output_dir = working_dir+'/rootfiles/'+dataset+'/Regions/'+year

    Filename_prefix = "WRAnalyzer"
    Filename_suffix = ""
    Filename_skim = "_SkimTree_LRSMHighPt"

    os.makedirs(output_dir, exist_ok=True)

    if "EGamma" in sample or "SingleMuon" in sample:
        output_file = os.path.join(output_dir, f"{Filename_prefix}{Filename_skim}_data_{sample}_test.root")
    elif sample == "Signal":
        output_file = os.path.join(output_dir, f"{Filename_prefix}{Filename_skim}_signal_{hnwr_mass}.root")
    else:
        output_file= os.path.join(output_dir, f"{Filename_prefix}{Filename_skim}_{sample}.root")

    my_histograms = scale_hists(my_histograms)
    summed_hist = sum_hists(my_histograms)

    my_split_hists = split_hists(summed_hist)

    with uproot.recreate(output_file) as root_file:
        for key, hist in my_split_hists.items():
            region, hist_name = key
            path = f'/{region}/{hist_name}_{region}'
            root_file[path] = hist

    print(f"Histograms saved to {output_file}.")

def scale_hists(data):
    for dataset_key, dataset_info in data.items():
        if 'x_sec' in dataset_info and 'sumw' in dataset_info:
            sf = dataset_info['x_sec']/dataset_info['sumw']
            for key, value in dataset_info.items():
                if isinstance(value, Hist):
                    value *= sf
    return data

def sum_hists(my_hists):
    original_histograms = list(my_hists.values())[0]
    sum_histograms = {key: Hist(*original_histograms[key].axes, storage=original_histograms[key].storage_type()) for key in original_histograms if isinstance(original_histograms[key], Hist)}
    for dataset_info in my_hists.values():
        for key, value in dataset_info.items():
            if isinstance(value, Hist):
                hist_name = key
                hist_data = value
                if key in sum_histograms:
                    sum_histograms[hist_name] += hist_data
    return sum_histograms

def split_hists(summed_hists):
    split_histograms = {}

    for hist_name, sum_hist in summed_hists.items():
        process_axis = sum_hist.axes['process']
        regions_axis = sum_hist.axes['region']

        unique_processes = [process_axis.value(i) for i in range(process_axis.size)]
        unique_regions = [regions_axis.value(i) for i in range(regions_axis.size)]

        for process in unique_processes:
            for region in unique_regions:
                sub_hist = sum_hist[{process_axis.name: process, regions_axis.name: region}]
                key = (region, hist_name)
                split_histograms[key] = sub_hist

    return split_histograms
