import json
import sys
import time
import warnings
from dask import delayed
from dask.diagnostics import ProgressBar
import awkward as ak
import dask_awkward as dak
import gc
import argparse
import uproot
from dask.distributed import Client
import os 
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="coffea.*")
from dask import compute
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

# Suppress FutureWarnings for better output readability
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea")
warnings.filterwarnings("ignore", category=FutureWarning, module="htcondor")
warnings.simplefilter("ignore", category=FutureWarning)

from coffea.dataset_tools import preprocess, apply_to_fileset, max_files, max_chunks, slice_files

NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.error_missing_event_ids = False

from coffea.nanoevents import NanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False
NanoAODSchema.error_missing_event_ids = False

def is_rootcompat(a):
    """Check if the data is a flat or 1-d jagged array for compatibility with uproot."""
    t = dak.type(a)
    if isinstance(t, ak.types.NumpyType):
        return True
    if isinstance(t, ak.types.ListType) and isinstance(t.content, ak.types.NumpyType):
        return True
    return False

def uproot_writeable(events):
    """Restrict to columns that uproot can write compactly."""
    out_event = events[list(x for x in events.fields if not events[x].fields)]
    for bname in events.fields:
        if events[bname].fields:
            out_event[bname] = ak.zip(
                {
                    n: ak.without_parameters(events[bname][n])
                    for n in events[bname].fields
                    if is_rootcompat(events[bname][n])
                }
            )
    return out_event

def read_runs_tree_asdict(src_path):
    """Return the full Runs tree as a dict of numpy arrays, or None if missing."""
    try:
        with uproot.open(src_path) as fin:
            if "Runs" not in fin:
                return None
            # dict: {branch_name: numpy_array (len = n_runs_entries)}
            return fin["Runs"].arrays(library="np")
    except Exception as e:
        print(f"[warn] Failed to read Runs from {src_path}: {e}", file=sys.stderr)
        return None

def get_unskimmed_events(rootFile):
    filepath = f"{rootFile}"
    try:
        events = NanoEventsFactory.from_root(
                {filepath: "Events"},
                mode="dask",
                schemaclass=NanoAODSchema
        ).events()

        if not hasattr(events, "event"):
            print(f"File {filepath} does not contain 'event'. Skipping...", file=sys.stderr)
            return 0  # Return 0 so that it doesn't affect the sum

        total_events = dak.num(events, axis=0).compute()
        return total_events

    except Exception as e:
        print(f"Exception occurred while processing file {filepath}: {e}", file=sys.stderr)
        return 0  # Return 0 on error


def make_skimmed_events(events):
    """Apply event selection to create skimmed datasets.

    Require:
      - ≥2 leptons (leading > 52 GeV, subleading > 45 GeV)
        with per-lepton preselection: pT > 45 GeV and |η| < 2.5
      - (≥ 2 AK4 jets with pT > 30 GeV OR ≥ 1 FatJet with pT > 180 GeV)
    """

    # --- Lepton preselection (Loose ID / HEEP) ---
    e = events.Electron
    m = events.Muon

    loose_electrons = e[
        (abs(e.eta) < 2.5)
        & (e.pt > 45)
    ]

    loose_muons = m[
        (abs(m.eta) < 2.5)
        & (m.pt > 45)
    ]

    # --- Take top-2 leptons by pT ---
    leptons = ak.with_name(
        ak.concatenate((loose_electrons, loose_muons), axis=1),
        "PtEtaPhiMCandidate",
    )

    leptons = leptons[ak.argsort(leptons.pt, axis=1, ascending=False)]
    leptons_padded = ak.pad_none(leptons, 2, clip=True)
    leading_lepton = leptons_padded[:, 0]
    subleading_lepton = leptons_padded[:, 1]

    # Event-level lepton pT cuts (η already enforced in preselection)
    lead_pt_cut = 52
    sublead_pt_cut = 45
    lep2_mask = (
        (ak.fill_none(leading_lepton.pt, 0) > lead_pt_cut)
        & (ak.fill_none(subleading_lepton.pt, 0) > sublead_pt_cut)
    )

    # --- Hadronic: (≥2 AK4 pT > 30) OR (≥1 FatJet pT > 180) ---
    ak4_pt_cut = 32
    ak8_pt_cut = 180

    # Tight AK4 jets + |eta| cut
    ak4_tight = (
        (events.Jet.pt > ak4_pt_cut)
        & (abs(events.Jet.eta) < 2.5)
    )
    two_ak4_pt30 = ak.sum(ak4_tight, axis=1) >= 2

    # AK8 FatJets with ID + |eta| cut
    ak8 = (
        (events.FatJet.pt > ak8_pt_cut)
        & (abs(events.FatJet.eta) < 2.5)
    )
    one_fj_pt180 = ak.sum(ak8, axis=1) >= 1

    had_or_mask = two_ak4_pt30 | one_fj_pt180

    # --- Final event mask ---
    event_filters = lep2_mask & had_or_mask

    # --- Apply selection ---
    skimmed = events[event_filters]

    # --- Keep everything except a small exclude list ---
    kept = {
        "event", "run", "luminosityBlock", "Electron", "Muon", "GenDressedLepton", "Jet", "GenJet", "FatJet", "GenJetAK8", "GenPart", "genWeight", "GenPart", "HLT", "LHE", "LHEPart", "LHEScaleWeight", "MET", "Pileup", "bunchCrossing", "Generator", "PSWeight", "GenMET",
        "LHEPdfWeight", "LHEReweightingWeight", "LHEWeight", "Flag", "genTtbarId", "GenProton", "HLTriggerFinalPath", "HLTriggerFirstPath", "Rho", "PV", "SV", "SubGenJetAK8", "SubJet", "OtherPV"
    }
    available_fields = ak.fields(skimmed)
    keep = [f for f in available_fields if f in kept]
    dropped = sorted([f for f in available_fields if f not in kept])
    skimmed = skimmed[keep]

    # --- Log summary ---
    print("\n### **Skim Details**")
    print("-------------------------------------------")
    print("Lepton selection:")
    print(f"  • Preselection (per lepton): pT > 45 GeV and |η| < 2.5")
    print(f"  • Event-level: leading pT > {lead_pt_cut} GeV, subleading pT > {sublead_pt_cut} GeV")
    print()
    print("Jet selection:")
    print(f"  • AK4: pT > {ak4_pt_cut} GeV, |η| < 2.5")
    print(f"  • AK8: pT > {ak8_pt_cut} GeV, |η| < 2.5")
    print()
    print("Event-level hadronic requirement:")
    print(f"  • (≥ 2 AK4 passing above) OR (≥ 1 AK8 passing above)")
    print("-------------------------------------------")
    print(f"Kept branches: {', '.join(keep) if keep else 'None'}")
    print()
    print(f"Dropped branches: {', '.join(dropped) if dropped else 'None'}")
    print("-------------------------------------------\n")

    return skimmed

def filter_json_by_primary_ds_name(json_data, primary_ds_name):
    filtered_data = {
        key: value for key, value in json_data.items()
        if "metadata" in value and value["metadata"].get("sample") == primary_ds_name
    }
    return filtered_data

def process_file(sliced_dataset, dataset_key, dataset, file_index, era, run):
    """Process and skim individual files."""
    file_names = sliced_dataset[dataset_key]['files']
    print(f"Name: {list(file_names.keys())[0]}")
    print(f"-------------------------------------------")
    file_path = list(file_names.keys())[0]

    file_name = os.path.basename(file_path)
    root_file = os.path.splitext(file_name)[0]

    dataset_key = list(sliced_dataset.keys())[0]
    datatype = sliced_dataset[dataset_key]["metadata"]["datatype"]

    unskimmed_count = get_unskimmed_events(file_path)

    """Process and skim individual files."""
    skimmed_dict = apply_to_fileset(
        make_skimmed_events,
        sliced_dataset,
        schemaclass=NanoAODSchema,
        uproot_options={"handler": uproot.MultithreadedXRootDSource, "timeout": 3600}
    )

    if era == "Run3Summer22" or era == "Run3Summer22EE":
        year = "2022"
    elif era == "Run3Summer23" or era == "Run3Summer23BPix":
        year = "2023"
    elif era == "RunIII2024Summer24":
        year = "2024"
    elif era == "RunIISummer20UL18":
        year = "2018"
    elif era == "RunIISummer20UL17":
        year = "2017"
    elif era == "RunIISummer20UL16":
        year = "2016"

    for dataset_name, skimmed in skimmed_dict.items():
        total_events = dak.num(skimmed, axis=0).compute()

        print(f"\n### **Skim Performance**")
        print(f"-------------------------------------------")
        print(f"**Unskimmed events:** {unskimmed_count}")
        print(f"**Skimmed events:** {total_events}")
        efficiency = (total_events / unskimmed_count) * 100
        print(f"**Skim efficiency:** {efficiency:.2f}%")

        conv = uproot_writeable(skimmed)

        # Extract short unique filename stem from the original NanoAOD file
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dest_dir  = f"scripts/setup/skims/tmp/{run}/{year}/{era}/{dataset}"
        os.makedirs(dest_dir, exist_ok=True)
        out_path  = os.path.join(dest_dir, f"{base_name}_skim.root")

        fields = list(conv.fields)
        vals = [conv[k] for k in fields]
        computed_vals = compute(*vals)
        conv_mat = {k: ak.to_packed(v, highlevel=True) for k, v in zip(fields, computed_vals)}


        comp = uproot.ZSTD(7)  # or ZLIB/LZ4
        with uproot.recreate(out_path) as fout:
            fout["Events"] = conv_mat

            runs_payload = read_runs_tree_asdict(file_path)
            if runs_payload is not None:
                fout["Runs"] = runs_payload
            else:
                print(f"[warn] No Runs tree found in source; skipping Runs write for {out_path}", file=sys.stderr)
#
        print(f"[ok] Wrote single file: {out_path}")
        print(f"[ok] File size: {os.path.getsize(out_path)/1_048_576:.2f} MB")

        del conv
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset and run.')
    parser.add_argument("era", type=str, choices=["RunIISummer20UL18", "Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix", "RunIII2024Summer24"], help="Era (e.g. RunIISummer20UL18NanoAODv9)")
    parser.add_argument('process', type=str, choices=["DYJets", "tt_tW", "Nonprompt", "Other",  "EGamma", "Muon"],  help='Physics group to process (e.g. DYJets)')
    parser.add_argument('dataset', type=str, help='Dataset to process (e.g. TTTo2L2Nu')
    parser.add_argument('--start', type=int, default=1, help='File number at which to start')
    args = parser.parse_args()

    t0 = time.monotonic()

    if "18" in args.era:
        run = "RunII"
        year = "2018"
    elif "Run3Summer22" in args.era:
        run = "Run3"
        year = "2022"
    elif "Run3Summer23" in args.era:
        run = "Run3"
        year = "2023"
    elif "RunIII2024Summer24" in args.era:
        run = "Run3"
        year = "2024"

    json_file_path = f'data/jsons/{run}/{year}/{args.era}/unskimmed/{args.era}_{args.process}_fileset.json'
    with open(json_file_path, 'r') as file:
        fileset = json.load(file)

    full_dataset = filter_json_by_primary_ds_name(fileset, args.dataset)

    dataset_key = list(full_dataset.keys())[0]
    num_files = len(full_dataset[dataset_key]['files'])

    print()
    sliced_dataset = slice_files(full_dataset, slice(args.start - 1, args.start))

    print(f"\n### **Processing Information**")
    print(f"-------------------------------------------")
    print(f"**File:** {args.start} of {num_files}")

    task = process_file(sliced_dataset, dataset_key, args.dataset, args.start, args.era, run)

    exec_time = time.monotonic() - t0
    print(f"**Execution time:** {exec_time/60:.2f} minutes")
    print(f"-------------------------------------------\n")
