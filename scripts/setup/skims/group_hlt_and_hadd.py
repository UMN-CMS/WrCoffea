#!/usr/bin/env python3
import os
import argparse
import subprocess
import hashlib
from collections import defaultdict

import uproot

def get_hlt_signature_and_entries(filepath, tree_name="Events"):
    """
    Open a ROOT file, read HLT_* branches from the given tree,
    and return (signature, n_entries).

    signature = tuple of sorted HLT branch names (strings)
    """
    try:
        with uproot.open(filepath) as f:
            if tree_name not in f:
                raise KeyError(f"Tree '{tree_name}' not found in {filepath}")
            tree = f[tree_name]

            # Get all branches starting with HLT_
            hlt_branches = sorted(tree.keys(filter_name="HLT_*"))

            # Number of events (entries in the tree)
            n_entries = tree.num_entries

            signature = tuple(hlt_branches)
            return signature, int(n_entries)
    except Exception as e:
        raise RuntimeError(f"Failed to inspect {filepath}: {e}")


def short_hash(signature):
    """Produce a short hash string for a given HLT signature."""
    sig_str = "|".join(signature)
    return hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:8]


def chunk_files_by_events(file_info_list, max_events):
    """
    Given a list of (filepath, n_events), split into chunks such that
    the sum of events per chunk <= max_events.

    Returns a list of chunks, each chunk is a list of (filepath, n_events).
    """
    chunks = []
    current_chunk = []
    current_events = 0

    for fname, nev in file_info_list:
        # If adding this file would exceed the limit, start a new chunk
        if current_chunk and (current_events + nev > max_events):
            chunks.append(current_chunk)
            current_chunk = [(fname, nev)]
            current_events = nev
        else:
            current_chunk.append((fname, nev))
            current_events += nev

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def run_hadd(output_file, input_files, dry_run=False):
    """Run hadd to merge input_files into output_file."""
    if len(input_files) == 1:
        print(f"[INFO] Single file in chunk, still using hadd: {input_files[0]}")
    cmd = ["hadd", "-f", output_file] + input_files
    print(f"[HADD] {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Group ROOT files by HLT branches and hadd them, "
                    "ensuring each output has <= max_events events."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing input ROOT files"
    )
    parser.add_argument(
        "--tree",
        default="Events",
        help="Name of the TTree containing HLT branches (default: Events)"
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=1_000_000,
        help="Maximum number of events per output file (default: 1,000,000)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output ROOT files (default: input_dir/hadded)"
    )
    parser.add_argument(
        "--prefix",
        default="group",
        help="Output filename prefix (default: group)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run hadd, just print what would be done"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if args.output_dir is None:
        output_dir = os.path.join(input_dir, "hadded")
    else:
        output_dir = os.path.abspath(args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Collect ROOT files
    root_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".root") and os.path.isfile(os.path.join(input_dir, f))
    ]

    if not root_files:
        print(f"[ERROR] No ROOT files found in {input_dir}")
        return

    print(f"[INFO] Found {len(root_files)} ROOT files in {input_dir}")
    print(f"[INFO] Using tree: {args.tree}")
    print(f"[INFO] Max events per output: {args.max_events}")
    print(f"[INFO] Output directory: {output_dir}")

    # Map: HLT signature -> list of (filepath, n_entries)
    groups = defaultdict(list)

    # Inspect files
    for i, fpath in enumerate(sorted(root_files)):
        print(f"[SCAN] ({i+1}/{len(root_files)}) {os.path.basename(fpath)}")
        try:
            signature, n_entries = get_hlt_signature_and_entries(fpath, args.tree)
        except RuntimeError as e:
            print(f"[WARN] {e}")
            continue

        groups[signature].append((fpath, n_entries))

    print(f"[INFO] Found {len(groups)} distinct HLT branch sets")

    total_hadded_events = 0  # sum over all output chunks

    # For each group, chunk by events and hadd
    for idx, (signature, file_info_list) in enumerate(groups.items()):
        sig_hash = short_hash(signature)
        print("\n" + "=" * 80)
        print(f"[GROUP {idx}] HLT signature hash: {sig_hash}")
        print(f"[GROUP {idx}] Number of HLT_* branches: {len(signature)}")
        print(f"[GROUP {idx}] Number of files: {len(file_info_list)}")

        total_events_group = sum(nev for _, nev in file_info_list)
        print(f"[GROUP {idx}] Total events: {total_events_group}")

        # Sort files for reproducibility (e.g. by name)
        file_info_list = sorted(file_info_list, key=lambda x: x[0])

        chunks = chunk_files_by_events(file_info_list, args.max_events)
        print(f"[GROUP {idx}] Will produce {len(chunks)} output file(s)")

        for chunk_idx, chunk in enumerate(chunks):
            chunk_files = [f for f, _ in chunk]
            chunk_events = sum(nev for _, nev in chunk)

            out_name = f"{args.prefix}_hlt{sig_hash}_part{chunk_idx}.root"
            out_path = os.path.join(output_dir, out_name)

            print(f"[GROUP {idx}]  Chunk {chunk_idx}: "
                  f"{len(chunk_files)} file(s), ~{chunk_events} events")
            run_hadd(out_path, chunk_files, dry_run=args.dry_run)

            # Accumulate total events in all (hadded) outputs
            total_hadded_events += chunk_events

    # Final summary of total events in all output files
    print("\n" + "=" * 80)
    if args.dry_run:
        print(f"[INFO] (DRY RUN) Total events that would be in all hadded files: {total_hadded_events}")
    else:
        print(f"[INFO] Total events in all hadded files: {total_hadded_events}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
