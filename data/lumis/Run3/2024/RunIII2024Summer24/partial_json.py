#!/usr/bin/env python3
import json

# Input and output file names
INPUT_TXT  = "Cert_Collisions2024_378981_386951_Golden.txt"
OUTPUT_TXT = "Cert_Collisions2024_378981_386951_Partial_Golden.txt"

# Load the original JSON from text file
with open(INPUT_TXT) as f:
    golden = json.load(f)

# Sort runs numerically (keys are strings)
runs = sorted(golden.keys(), key=int)

n_runs = len(runs)
target_fraction = 0.095
target_n = max(1, round(target_fraction * n_runs))
# Uniformly spaced selection
step = max(1, n_runs // target_n)
selected_runs = runs[0::step]

print(f"Total runs: {n_runs}")
print(f"First run in file: {runs[0]}")
print(f"Starting with run {selected_runs[0]} and picking every {step}th run")
print(f"Selected {len(selected_runs)} runs (~{100*len(selected_runs)/n_runs:.1f}%)")

# Build smaller JSON
thinned = {run: golden[run] for run in selected_runs}

# Write to output text file
with open(OUTPUT_TXT, "w") as f:
    json.dump(thinned, f, indent=4, sort_keys=True)

print(f"Saved partial GOLDEN JSON: {OUTPUT_TXT}")
