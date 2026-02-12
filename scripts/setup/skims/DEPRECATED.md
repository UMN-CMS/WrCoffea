# Deprecated

The skimming scripts in this directory have been replaced by:

| Old | New |
|-----|-----|
| `skim_files.py` | `wrcoffea/skimmer.py` (library) + `bin/skim.py run` (CLI) |
| `submit_jobs.sh` + `create_job.py` | `bin/skim.py submit` |
| `job.sh` | `bin/skim_job.sh` |
| `hadd_dataset.sh` / `hadd_dataset2.sh` / `unzip_files.sh` | `wrcoffea/skim_merge.py` (library) + `bin/skim.py merge` (CLI) |
| *(no equivalent)* | `bin/skim.py check` (new: detect missing jobs) |
| *(no equivalent)* | `wrcoffea/das_utils.py` (DAS query utilities) |

All subcommands now accept a DAS dataset path instead of era/process/dataset.
See the README for updated usage instructions.
