import uproot
import os
import logging
from pathlib import Path
import hist
from hist import Hist
from python.preprocess_utils import get_era_details
from typing import Dict

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

def _is_simple_cutflow(kinds) -> bool:
    if isinstance(kinds, Hist):
        return True
    if isinstance(kinds, dict):
        keys = set(kinds.keys())
        return keys <= {"cumulative"} and isinstance(kinds.get("cumulative"), Hist)
    return False


def _normalize_syst_name(syst: str) -> str:
    """
    Turn 'RenFactScaleUp' -> 'renfactscaleup' (root-key friendly, lowercase).
    """
    return "".join(ch.lower() for ch in syst if ch.isalnum())

def _folder_and_hist_names(region: str, syst: str, hist_stem: str):
    if syst == "Nominal":
        folder = f"{region}"
        hname  = f"{hist_stem}_{region}"
    else:
        norm = _normalize_syst_name(syst)
        folder = f"syst_{norm}_{region}"
        hname  = f"{hist_stem}_syst_{norm}_{region}"
    return folder, hname

def _sum_cutflow_hists(my_hists):
    """
    Sum ALL cutflow histograms across datasets, recursively.

    Supports structures like:
      cutflow = {
        "no_cuts": Hist,
        "no_cuts_unweighted": Hist,
        "ee":   {"two_tight_electrons": Hist, "e_trigger": Hist, ..., "onecut": Hist, ...},
        "mumu": {...},
        "em":   {...},
      }
    """
    def _merge(dst, src):
        # src can be Hist or dict
        if isinstance(src, Hist):
            if dst is None:
                return src.copy()
            if isinstance(dst, Hist):
                dst += src
                return dst
            # dst is dict -> inconsistent types; treat as error or ignore
            raise TypeError("Cutflow key has mixed types (Hist vs dict) across datasets.")
        elif isinstance(src, dict):
            if dst is None or isinstance(dst, Hist):
                dst = {}  # (re)start a dict
            for k, v in src.items():
                dst[k] = _merge(dst.get(k), v)
            return dst
        else:
            # unknown type: ignore
            return dst

    out = {}
    for dataset_payload in my_hists.values():
        cfmap = dataset_payload.get("cutflow") or dataset_payload.get("__cutflow__")
        if not isinstance(cfmap, dict):
            continue
        for k, v in cfmap.items():
            out[k] = _merge(out.get(k), v)
    return out


def _save_cutflows(root_file, cutflow_summed: Dict[str, dict]):
    """
    Recursively write all cutflow histograms while avoiding duplicate writes.
    If a given path is encountered twice, the second write is skipped to prevent ROOT ;2 cycles.
    """
    seen = set()

    def _recurse(prefix, obj):
        if isinstance(obj, Hist):
            if prefix not in seen:
                root_file[prefix] = obj
                seen.add(prefix)
            return
        if isinstance(obj, dict):
            for name, child in obj.items():
                _recurse(f"{prefix}/{name}", child)

    _recurse("/cutflow", cutflow_summed)

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
                hh = h[{region_ax.name: reg, syst_ax.name: sy}]

                if sum_over_process and has_proc and any(ax.name == "process" for ax in hh.axes):
                    idxs_to_keep = [i for i, ax in enumerate(hh.axes) if ax.name != "process"]
                    if len(idxs_to_keep) < hh.ndim:
                        hh = hh.project(*idxs_to_keep)

                out[(reg, sy, hist_name)] = hh
    return out

def save_histograms(histograms, args):
    run, year, era = get_era_details(args.era)
    sample = args.sample
    hnwr_mass= args.mass

    working_dir = Path("WR_Plotter")

    if getattr(args, 'dir', None):
        output_dir = working_dir / 'rootfiles' / run / year / era / args.dir
    else:
        output_dir = working_dir / 'rootfiles' / run / year / era

    output_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, 'name', None):
        filename_prefix = f"WRAnalyzer_{args.name}"
    else:
        filename_prefix = f"WRAnalyzer"

    if sample == "Signal":
        output_file = output_dir / f"{filename_prefix}_signal_{hnwr_mass}.root"
    else:
        output_file= output_dir / f"{filename_prefix}_{sample}.root"

    summed_hist = sum_hists(histograms)
    split_histograms_dict = split_hists_with_syst(summed_hist, sum_over_process=True)

    cutflow_summed = _sum_cutflow_hists(histograms)

    with uproot.recreate(output_file) as root_file:
        for (region, syst, hist_name), hist_obj in split_histograms_dict.items():
            try:
                var_axes = [ax for ax in hist_obj.axes if ax.__class__.__name__ != "StrCategory"]
                var_stem = var_axes[0].name if var_axes else hist_name
            except Exception:
                var_stem = hist_name
            folder, hname = _folder_and_hist_names(region, syst, var_stem)
            root_file[f"/{folder}/{hname}"] = hist_obj

        _save_cutflows(root_file, cutflow_summed)

    logging.info(f"Histograms saved to {output_file}.")


def scale_hists(data):
    for dataset_key, dataset_info in data.items():
        if 'x_sec' in dataset_info and 'sumw' in dataset_info:
            sf = dataset_info['x_sec']/dataset_info['nevts']
            for key, value in dataset_info.items():
                if isinstance(value, Hist):
                    value *= sf
        else:
            logging.warning(f"Dataset {dataset_key} missing 'x_sec' or 'sumw'. Skipping scaling.")
    return data

def sum_hists(my_hists):
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
