# Skimming

## Skim the 2025 data at the LPC

On a cmslpc login node. Replace `<you>` by your LPC username and `<your Rucio account>` by your
Rucio account, usually your CERN username.

```bash
# Clone under nobackup
cd ~/nobackup && git clone --recursive --branch billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea

# A proxy the Condor jobs can use for the whole campaign (8 days)
voms-proxy-init --rfc --voms cms -valid 192:00 --out ~/nobackup/x509up
export X509_USER_PROXY=~/nobackup/x509up

# Where the plan and the local records go, where the jobs write the skims, your Rucio account
export SKIM_PLAN=~/nobackup/skims_v2/plans/v2_RunIII2025Summer25 SKIM_OUT=~/nobackup/skims_v2 SKIM_PROXY=~/nobackup/x509up
export REMOTE=root://cmseos.fnal.gov//store/user/<you>/WRAnalyzer/skims_v2
export RUCIO_ACCOUNT=<your Rucio account>

# Open a shell in the container; every skim.py command below runs inside it
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-almalinux9-noml:2026.9.0-py3.12
apptainer exec -B /cvmfs -B /uscms_data -B /uscms/home -B /uscms/homes $IMG bash

# Build the plan: the files of every 2025 dataset from DBS, cut into groups of 60 GB (a few minutes)
python3 bin/skim.py plan --campaign v2 --plan-dir $SKIM_PLAN --era RunIII2025Summer25

# Type exit to leave the container, then submit one Condor job per group; the jobs write the skims to $REMOTE
bin/lpc_condor.sh --jobs-dir ~/nobackup/skimjobs/v2_2025 --era RunIII2025Summer25 --remote $REMOTE --cpus 2 --decompression-threads 2 --submit

# Watch the queue
condor_q

# Back in the container (the apptainer command again): copy the finished jobs' records from $REMOTE to $SKIM_OUT
python3 bin/skim.py mirror --plan-dir $SKIM_PLAN --out $SKIM_OUT --remote $REMOTE

# Show which groups are done, failed or pending, and how many events they kept (run mirror first, every time)
python3 bin/skim.py -v status --plan-dir $SKIM_PLAN --out $SKIM_OUT

# Once the queue is empty: list the inputs the jobs could not read, and mark the ones with no replica anywhere as unavailable
python3 bin/skim.py bad-files --plan-dir $SKIM_PLAN --out $SKIM_OUT --classify --rucio-account $RUCIO_ACCOUNT

# Resubmit the groups that failed or had unreachable inputs: exit the container, same command as before with a new jobs dir
bin/lpc_condor.sh --jobs-dir ~/nobackup/skimjobs/v2_2025_retry --era RunIII2025Summer25 --remote $REMOTE --cpus 2 --decompression-threads 2 --submit

# When everything is done: check the skims against DBS and the golden JSON, and write the processed-lumisection masks
python3 bin/skim.py validate --plan-dir $SKIM_PLAN --out $SKIM_OUT --skip-reread --remote $REMOTE --remote-lumis --report $SKIM_PLAN/validation.json

# Write the analyzer fileset, data/filesets/Run3/2025/RunIII2025Summer25/skimmed/RunIII2025Summer25_data_fileset.json
python3 bin/skim.py fileset --plan-dir $SKIM_PLAN --out $SKIM_OUT --era RunIII2025Summer25 --remote $REMOTE
```

Notes:

- The datasets come from `data/configs/Run3/2025/RunIII2025Summer25/RunIII2025Summer25_data.json`; add
  new periods or a reprocessing there before planning.
- Running the 2024 data, the same volume, took about a day and a half on Condor. A job that runs longer
  than 8 hours is held and restarted on another node.
- `validate` exits 1 while some input is bad; add `--accept-bad-files` once unreadable or unavailable
  inputs are the only problem. `fileset` refuses a dataset that is not complete; `--allow-missing` writes
  the file without it.
- To try one group locally before submitting: `python3 bin/skim.py run --plan-dir $SKIM_PLAN --dataset
  Muon0_Run2025C --limit-groups 1 --workers 1 --out ~/nobackup/skim_test --scratch ~/nobackup/skim_scratch`
  (a test root, not `$SKIM_OUT`; delete it afterwards).
- The 2025 data skims take about 0.2 TB of the 2 TB EOS quota. Tested end to end on 2026-10-12 on one
  dataset with Billy's grid account.

Then (Billy): the luminosity from the masks (`scripts/processed_lumi.py`) into `config.yaml` and
`WR_Plotter/data/lumi.yaml`; the 2024-MC alias filesets with `python3 bin/skim.py alias-filesets --era
RunIII2025Summer25`; commit the fileset, the lumi files and the config; then
`python3 bin/run_analysis.py RunIII2025Summer25 data --dir <tag>` from any machine. Unskimmed filesets
need no skims: `python3 bin/skim.py unskimmed-fileset --era RunIII2025Summer25`.

## What a skim is

A NanoAOD file reduced to the events and branches the analysis can ever need, written by
`bin/skim.py` (`wrcoffea/skim/`). skims_v2 covers the four UL eras, 2022, 2022EE, 2023, 2023BPix and
2024, in `root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2/`; every skimmed
fileset points there. The 2025 data follow the recipe above.

## Selection

Looser than every analysis cut, so threshold changes and systematic shifts stay inside the skims.

| Cut | Value |
| --- | --- |
| Lepton pT min, both flavours | 45 GeV |
| Lepton eta max | 2.5 |
| Leading lepton pT min | 52 GeV |
| Subleading lepton pT min | 45 GeV |
| AK4 jet pT min, eta max | 32 GeV, 2.5 |
| AK8 jet pT min, eta max | 180 GeV, 2.5 |

- Kept: at least two leptons with the leading one above 52 GeV, and at least two AK4 jets or one AK8 jet.
- Muons are ranked on the larger of PF pT and TuneP pT. No trigger, MET filter or golden JSON at skim
  time; the analyzer applies them.
- Two independent implementations (`selection.py`, awkward; `recount.py`, NumPy) evaluate every chunk;
  a disagreement aborts the job. `python3 bin/skim.py cuts` prints the cuts.

## Branches

`wrcoffea/skim/branches.yaml`; collections are kept or dropped whole, so cross-reference indices stay valid.

- Kept: Electron, Muon, Jet, FatJet, SubJet, the Gen collections, LHE*, Generator, Pileup, PV, OtherPV,
  every MET, Rho, Flag, L1PreFiringWeight, HTXS, PVBS, event identifiers, genWeight, genTtbarId,
  bunchCrossing, LHEScaleWeight, LHEPdfWeight, LHEReweightingWeight, PSWeight, HLTriggerFinalPath/FirstPath.
- Dropped (35 to 45 % of every input): Tau, boostedTau, Photon, LowPtElectron, FsrPhoton, IsoTrack, SV,
  TrigObj, `L1_*`, `DST_*`, SoftActivityJet*, CorrT1METJet, PFCand, TrackGenJetAK4, Proton, PPS, GenVisTau,
  GenCands, btagWeight_*.
- Unknown branches (new NanoAOD versions) are kept and reported. Per-object fields are never filtered.
- Triggers: every `HLT_*` single-lepton, photon, jet, HT and MET path found in the sampled inputs, plus
  the required analysis paths; a required path missing from an input is written as `False` and noted in
  the sidecar, so run periods with different menus share one schema.

## Output

```
<out>/<run>/<year>/<era>/<category>/<dataset>/
    <dataset>_g0007p02.root      part 2 of group 7: Events, Runs[, LuminosityBlocks]
    <dataset>_g0007p02.json      sidecar: inputs, entry ranges, event-id digests, checksums, Runs sums
    _groups/g0007.json           group status: done/failed, bad files with attempt logs, timings
<out>/_logs/<dataset>_g0007.log
```

- Runs tree: one entry per part for MC (`genEventSumw`, `genEventSumw2`, `genEventCount` summed;
  `LHEScaleSumw`, `LHEPdfSumw`, `PSSumw` as the genEventSumw-weighted mean, NanoAOD stores ratios), one
  entry per run for data.
- A ratio branch the input lacks is derived from the events (`weight_sums.py`): UL NanoAODv9 has no
  `PSSumw`, so the worker sums `genWeight * PSWeight` before the selection; the sidecar records it
  under `derived_runs_branches`. Native branches are never recomputed.
- Data parts carry the merged `LuminosityBlocks` tree.
- Events of one input are contiguous in a part; the sidecar holds each input's entry range and a digest
  of its (run, lumi, event) triples. Parts and sidecars are written as `.partial` and renamed when complete.

## Workflow

- A job processes one group: consecutive files of one dataset up to 60 GB (data), 30 GB (MC) or one
  group per signal dataset. `skim.py group` skims it, `skim.py ship` copies parts, sidecars and the
  status to the remote, status last.
- MC eras: the recipe above with `--datatype mc` on `bin/lpc_condor.sh`, `mirror` and `status`;
  `--exclude _scan` leaves out the UL multi-model scans, which need `split-scan` before `fileset`.
  With a selection, `fileset` replaces only the selected entries of a family file.
- Laptop variant: the same `plan`, then `skim.py run --plan-dir ... --out <root> --scratch <scratch>
  --workers 3`, `validate` without `--skip-reread`, `skim.py upload --plan-dir ... --out <root>
  --remote <remote>`, `fileset`. `scripts/skim_closure.py` compares the analyzer on a skimmed dataset
  with its raw inputs (`--keep-inputs`).
- Defaults a new user overrides: `--remote` (Billy's Wisconsin area) on `condor`, `mirror`, `upload`
  and `fileset`; `--rucio-account` (his; `RUCIO_ACCOUNT` is read first and passed to the jobs);
  `--out` and `--scratch` of `run` (his laptop layout); `SKIM_PLAN` and `SKIM_OUT` of the wrapper.
- Config entries with `"optional": true` (the `MLL-10to50` DY companions, 1.5 B events in 2024) are
  skipped unless `plan --include-optional`.

## Plan

`plan` reads the era's configs (`archived/` excluded), lists every file of every DAS path through DBS
with size, events and adler32 (invalid files excluded and noted), samples one file per DAS path for
the branch layout, and cuts groups by input size. Data versions sharing a `dataset` name (`v1`, `v2`)
become one dataset. Non-DAS entries need a `files` list with `file_sizes`, `file_events`,
`file_adler32` (`scripts/private_signal_config.py` writes them), or `--skip-private`. `plan.json` and
`schemas.json` are the only inputs of every later step; nothing re-queries DAS.

## Fetching and checking inputs

- Each input is downloaded to scratch with `xrdcp --cksum adler32`: the FNAL redirector, the other
  redirectors, then every disk replica Rucio knows (HTTPS WebDAV with the proxy), US sites first. The
  next file is prefetched while the current one is skimmed.
- A transfer whose file has not grown for 180 s is killed and the next source tried (category `stall`).
- Per input: entries equal the DBS event count; both selections agree on every event; the closed part is
  re-read and must reproduce every input's digests, Runs sums and checksum. A failure deletes the part
  and marks the group failed.

## Bad inputs

| Category | Meaning | Retried |
| --- | --- | --- |
| `unreachable` | no server could serve the file | yes, by the next submission |
| `corrupt` | checksum matches DBS, a basket does not decompress or deserialize on any replica | no |
| `unavailable` | valid in DBS, no replica anywhere in Rucio (`bad-files --classify`) | no |

- A group finishes with its bad files listed; `skim.py group` exits 0 then, 1 only for a failed group.
- Skipped files cost statistics, not a bias: the filesets take `genEventSumw` from the parts written.
- `validate` exits 1 while a dataset has bad files; `--accept-bad-files` accepts them once they are the
  only problem.

## Stopping and resuming

- Everything is written atomically. A rerun with the same arguments skips finished groups, redoes
  `running` and `failed` ones, deletes `.partial` leftovers and reuses scratch inputs whose checksum
  matches. On a laptop keep the lid open: `caffeinate -i` does not prevent lid-close sleep.
- `SIGTERM` to `skim.py run` stops the pool: each worker kills its transfer, marks its group `failed`
  and removes its partial parts. Workers are `spawn` processes without `skim.py` on their command line,
  so `pkill -f "bin/skim.py run"` leaves them as orphans; every `running` status records the worker pid,
  and `run` refuses a group whose recorded pid is alive (`--force` overrides).

## Validation and filesets

- `validate`: the per-part re-read for the whole campaign (`--skip-reread` with `--remote`: every remote
  part is stat-ed against its sidecar size instead), every planned file in exactly one part, entries equal
  to the DBS count, summed `genEventSumw` and `genEventCount` reported. Data: the merged
  `LuminosityBlocks` must hold exactly the lumisections DBS lists; certified lumisections in no file are a
  warning (0.31 % for Muon0_Run2024I, a Tier-0 property). `--remote-lumis` writes
  `<dataset>_processed_lumis.json` next to the parts; the era luminosity is brilcalc on that mask
  (`scripts/processed_lumi.py`).
- UL18: `validate --remote-lumis` fails SingleMuon 2018A, 2018D and EGamma 2018A as expected: DBS's
  per-file lumisection lists are incomplete there. Nothing is lost.
- `scripts/skim_closure.py`: the analyzer on raw inputs and on their parts agrees bin by bin on every
  physics histogram and on the cumulative cutflows from the first two-lepton row on (72 of 72 at 1e-15
  on the test signal).
- Multi-model scans (`"scan": true`; the pre-legacy `WRtoNLtoLLJJ_MWR*` NanoAODv7 samples): skimmed as
  one dataset while the worker records per-model event counts and weight sums; `split-scan` writes one
  dataset per mass point with `--model-config` metadata; `fileset` writes the models, never the raw
  scan. The UL18 scans live in `RunIISummer20UL18_signal_scan.json` and its own fileset.
- Plans built before a cross-section `apply` keep the old numbers until `skim.py refresh-metadata
  --plan-dir ...`; `fileset` writes them into the filesets. The cross-section campaign itself is
  `scripts/genxsec_campaign.py` (`run`, `review`, `apply`, `annotate-order`).

## Edge eras (2022B, 2023B, 2024B)

Certified runs before the PdmV eras are skimmed and used like any other stream; the JME residuals
take the run number and cover them, and the `lumis` in `config.yaml` include them. To take one out,
move its entries into a `<era>_data_edge<period>.json` config and use the luminosity without it.

## Known input defects (skims_v2)

| era | dataset | file | defect | events | handling |
| --- | --- | --- | --- | --- | --- |
| RunIII2024Summer24 | `DYto2L-4Jets_Bin-HT-100to400-MLL-50to120` | `0709bde6-7950-40d1-a10f-26facdbb06f1.root` | corrupt: checksum matches DBS, LZMA errors in several branches | 161,368 | skipped |
| RunIII2024Summer24 | `DYto2Mu-2Jets_Bin-2J-MLL-50` | 51 files (`skim.py bad-files` lists them) | unavailable: valid in DBS, no replica in Rucio | 8,124,688 (2.8 %) | skipped |
| Run3Summer22 | `WtoLNu-4Jets_MLNu-0to120_HT-40to100` | `18f2ea1a-97f7-490a-94a1-675992a6752a.root` | corrupt: checksum matches DBS, LZMA error | 53,991 | skipped |
| Run3Summer23 | `DYto2L-4Jets_MLL-50to120_HT-70to100` | `6ed4f6d2-7a63-42b0-811d-d8e2d2d8eb99.root` | valid in DBS but zero events; empty LHE/PS sum vectors | 0 | kept, harmless |
| Run3Summer22 | `ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8` | `7c4f3eb2-3c7e-4c21-98ed-c1892bb3a057.root` | events without `LHEScaleWeight` / `LHEPdfWeight` (`PSWeight` present) | 12,342 of 1,970,234 (0.6 %) | skimmed, see below |
| Run3Summer22EE | `WRtoNEltoElElJJ_MWR6000_N1500` | `8f24526c-43f0-40e5-b062-4864ea6329bf.root` | same as ZZZ | 621 of 210,000 (0.3 %) | skimmed, see below |

- Full paths of the two files without LHE weights:
  `/store/mc/Run3Summer22NanoAODv12/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/50000/7c4f3eb2-3c7e-4c21-98ed-c1892bb3a057.root`
  and
  `/store/mc/Run3Summer22EENanoAODv12/WRtoNEltoElElJJ_MWR6000_N1500_TuneCP5_13p6TeV_madgraph-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/2810000/8f24526c-43f0-40e5-b062-4864ea6329bf.root`.
- Events without LHE weights are kept with empty weight vectors; the part's Runs LHE sums are the
  genEventSumw-weighted mean of the other inputs. The analyzer gives them the nominal weight for the
  `scale` variations and warns once per dataset; the yield is unchanged and the variation is
  underestimated by the affected share (0.61 % of skimmed ZZZ, 0.29 % of the WR point). Any other
  weight-count mismatch stops the analyzer.
- The list is complete for 2022 to 2024 (checked 2026-10-08 over all 4,124 MC part sidecars). No input was
  skipped in Run3Summer22EE, Run3Summer23 or Run3Summer23BPix.
- No LHE weights by design: the pure-Pythia 8 `WW`, `WZ` and `ZZ` samples of every Run 3 era (4 `PSWeight`
  entries only); the analyzer fills their `scale` variations with the nominal weight.

## Condor at the LPC

- `bin/lpc_condor.sh` saves the queue (`condor_q -af SkimGroup JobStatus Iwd`) on the host, runs
  `skim.py condor --inflight-file` in the container, and `condor_submit` on the host with `--submit`.
  Environment: `SKIM_PLAN`, `SKIM_OUT`, `SKIM_PROXY` (default `$X509_USER_PROXY`), `SKIM_IMAGE` (default
  the coffea almalinux9 image on cvmfs). Always pass `--remote`.
- `skim.py condor` lists the pending groups like `run` does, minus the groups still in the queue
  (ClassAd `SkimGroup` = `era/dataset:gid`), and writes the JDL, `run_group.sh`, a code tarball, one plan
  slice per dataset (carrying the known corrupt inputs, which the jobs skip) and `groups.txt`. Rerunning
  it is the retry. Use a new `--jobs-dir` while earlier jobs are queued: the command rewrites files they read.
- Resources: `--cpus` (`--cpus 4 --decompression-threads 4` is about 1.7 times faster than one core; the
  LZMA decompression releases the GIL), `--memory-mb` (default 4000), `--disk-gb` (default 20).
- Slow nodes: a job past `--max-hours` (default 8) is held and released elsewhere, at most 4 starts,
  avoiding its last three machines; `--avoid-hosts` excludes nodes outright.
- Logs: `logs/<dataset>_g<gid>.out|.err` and `logs/condor.log` in the jobs directory.
- Parts written by jobs exist only at the remote. `mirror` brings the JSON records to the local root;
  validate with `--skip-reread --remote <url> --remote-lumis` (data LuminosityBlocks are read through
  xrootd). Never list directories at Wisconsin: 30 s on a good day, a timeout on a busy one; a stat or a
  fetch by name takes a second.
- Developing on a laptop: rsync `bin/skim.py`, `wrcoffea/`, `data/configs` and `data/lumis` to the LPC
  clone, and the laptop root's JSON files (`--include='*/' --include='*.json' --exclude='*'`) to `SKIM_OUT`.

## Earlier skims

The skims made before October 2026 (`WRAnalyzer/skims/` at Wisconsin: 1209 branches, about 3.2 kB per
event, zlib:1, no LuminosityBlocks tree, genEventSumw from a separate DAS pass) are used by no fileset.
