# Skimming

What this page answers: what the skims contain, how a skim campaign is planned, run (on a
laptop or as LPC Condor jobs), validated and turned into analyzer filesets, and which inputs could
not be skimmed. The skims are NanoAOD files reduced to the events and branches the WR analysis can
ever need. The toolkit lives in `wrcoffea/skim/` with one driver, `bin/skim.py`: a plan-driven
campaign in which every input file is downloaded with checksum verification, skimmed from local
scratch, written into size-bounded parts with one fixed schema per dataset, and verified twice
before the local copy is deleted. Verified on 2026-10-12.

Status:

- The skims_v2 campaign covers the four UL eras, 2022, 2022EE, 2023, 2023BPix and 2024. Every
  skimmed fileset of those eras points at it, in Billy's area at Wisconsin
  (`root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2/`).
- The 2025 data are not skimmed yet; the workflow below is the recipe.

## Selection

Looser than every analysis cut (`wrcoffea/config.yaml`), so the skims survive
threshold changes and systematic shifts of object momenta:

| Cut | Value |
| --- | --- |
| Lepton pT min, both flavours | 45 GeV |
| Lepton eta max | 2.5 |
| Leading lepton pT min | 52 GeV |
| Subleading lepton pT min | 45 GeV |
| AK4 jet pT min, eta max | 32 GeV, 2.5 |
| AK8 jet pT min, eta max | 180 GeV, 2.5 |

An event is kept when it has at least two leptons passing the lepton cuts with
the leading one above 52 GeV, **and** at least two AK4 jets or at least one AK8
jet. Muons are ranked on the larger of the PF pT and the TuneP pT
(`Muon_pt * Muon_tunepRelPt`). No trigger, MET filter or golden-JSON
requirement is applied at skim time; the analyzer does that. The AN loose
leptons are 53 GeV, so the boosted region's loose lepton is covered.

The selection exists twice: the production implementation
(`wrcoffea/skim/selection.py`, awkward) and an independent recount
(`wrcoffea/skim/recount.py`, flat NumPy) that uses a different formulation.
Every chunk of every file is evaluated by both; a disagreement aborts the job.

```bash
python3 bin/skim.py cuts
```

## Branch policy

`wrcoffea/skim/branches.yaml` decides what is written. Collections are kept
or dropped whole, so every cross-reference index stays valid:

- **kept**: Electron, Muon, Jet, FatJet, SubJet, GenJet, GenJetAK8,
  SubGenJetAK8, GenPart, GenDressedLepton, GenIsolatedPhoton, GenVtx, GenMET, LHEPart, LHE,
  LHEWeight, Generator, Pileup, PV, OtherPV, every MET variant, Rho and the
  flat `fixedGridRho*` branches, Flag, L1PreFiringWeight, HTXS, PVBS, the
  event identifiers, genWeight, genTtbarId, bunchCrossing, LHEScaleWeight,
  LHEPdfWeight, LHEReweightingWeight, PSWeight, HLTriggerFinalPath/FirstPath.
- **dropped** (unused and measured at 35-45 % of every input): Tau, boostedTau,
  TauProd, Photon, LowPtElectron, FsrPhoton, IsoTrack, SV, TrigObj, all `L1_*`
  and `DST_*` bits, SoftActivityJet*, CorrT1METJet, PFCand, TrackGenJetAK4,
  Proton, PPSLocalTrack, GenProton, GenVisTau, GenCands, btagWeight_*.
- **unknown** branches (new in a NanoAOD version) are kept and reported.
- **triggers**: every `HLT_*` path matching the single-lepton, photon, jet,
  HT and MET patterns found in the sampled input files, plus a required list
  of the analysis paths (Run 2 and Run 3). A required path missing from an
  input is written as `False` and recorded in the sidecar, so files from
  different run periods (different HLT menus) share one schema and can be
  read together.

Per-object fields are never filtered, so b-tagging, regression and ID fields
all survive.

## Output

```
<out>/<run>/<year>/<era>/<category>/<dataset>/
    <dataset>_g0007p02.root      part 2 of group 7 (TTree: Events, Runs[, LuminosityBlocks])
    <dataset>_g0007p02.json      sidecar: inputs, entry ranges, event-id digests, checksums, Runs sums
    _groups/g0007.json           group status (done/failed, bad files with attempt logs, timings)
<out>/_logs/<dataset>_g0007.log
```

- The `Runs` tree is collapsed to one entry per part for MC (`genEventSumw`,
  `genEventSumw2`, `genEventCount` summed; `LHEScaleSumw`, `LHEPdfSumw`,
  `PSSumw` combined as the `genEventSumw`-weighted mean, because NanoAOD stores
  them as ratios) and to one entry per run for data.
- A ratio branch the input's `Runs` tree lacks is derived from the events
  (`wrcoffea/skim/weight_sums.py`, since 2026-10-11): UL NanoAODv9 has no
  `PSSumw`, so the worker sums `genWeight * PSWeight` over every event of the
  input, before the selection, and writes the ratio to `sum(genWeight)` as the
  input's `PSSumw` row(s); the collapse, the sidecar (`derived_runs_branches`
  of the input), the upload metadata and the fileset then treat it like a
  native branch, and the analyzer's `--systs ps` works for the era. Native
  branches are never recomputed; an empty input contributes empty vectors;
  scan samples (per-model sums) are left alone. The same applies to
  `LHEScaleSumw` / `LHEPdfSumw` should a production lack them.
- Data parts carry a merged `LuminosityBlocks` tree, so the processed
  luminosity can be recomputed and the golden-JSON coverage checked.
- Events of one input are contiguous inside a part; the sidecar records the
  entry range and a digest of the (run, luminosityBlock, event) triples for
  each input. Part names carry the group index so parallel workers never
  collide.
- Parts and sidecars are written as `.partial` files and renamed only when
  complete.

## Workflow: skimming the 2025 data at the LPC from a fresh clone

The commands below are what a new person runs on a cmslpc login node to skim every 2025 data
dataset. Replace `<you>` by your username and `<remote>` by the xrootd storage the jobs write to:
your FNAL EOS area, `root://cmseos.fnal.gov//store/user/<you>/WRAnalyzer/skims_v2` (the default EOS
quota is 2 TB; the 2025 data skims take about 0.2 TB), or a Wisconsin area if you have one
(`root://cmsxrootd.hep.wisc.edu//store/user/<you>/...`, where this analysis's skims live). The whole
workflow, from a fresh clone with the jobs writing to EOS through an analyzer run on the result, was
run on one 2025 dataset (Muon0_Run2025B) on 2026-10-12. Every `skim.py` command runs
inside the coffea container; only the Condor submission and `condor_q` run on the host.

```bash
# 0. Clone under nobackup (the plan, the job files and the mirrored statuses live there too). Clone the branch
#    directly: `git switch billy_dev` after a clone of main leaves WR_Plotter at main's submodule pointer
#    (if you did that, run `git submodule update --init --recursive` after the switch).
cd ~/nobackup && git clone --recursive --branch billy_dev git@github.com:UMN-CMS/WrCoffea.git && cd WrCoffea

# 1. A proxy on a shared path (asks for your grid key passphrase). The Condor jobs use it, so it must stay
#    valid for the whole campaign.
voms-proxy-init --rfc --voms cms -valid 192:00 --out ~/nobackup/x509up
export X509_USER_PROXY=~/nobackup/x509up
export SKIM_PLAN=~/nobackup/skims_v2/plans/v2_RunIII2025Summer25 SKIM_OUT=~/nobackup/skims_v2 SKIM_PROXY=~/nobackup/x509up
export REMOTE=root://cmseos.fnal.gov//store/user/<you>/WRAnalyzer/skims_v2   # exported: steps 7-9 read it in the container
export RUCIO_ACCOUNT=<your Rucio account>   # usually your CERN username; run, group, bad-files and the Condor jobs use it

# 2. The container holds coffea, uproot and the XRootD bindings; open a shell in it for steps 3, 4, 7, 8 and 9
IMG=/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-almalinux9-noml:2026.9.0-py3.12
apptainer exec -B /cvmfs -B /uscms_data -B /uscms/home -B /uscms/homes $IMG bash

# 3. The datasets: data/configs/Run3/2025/RunIII2025Summer25/RunIII2025Summer25_data.json lists the DAS paths
#    (EGamma0/1 and Muon0/1 per period). Compare with DAS before planning and add new periods or a reprocessing
#    by hand (docs/inputs.md); the golden JSON is data/lumis/Run3/2025/RunIII2025Summer25/Cert_...Golden.json.
#    The plan takes a few minutes (2026-10-12: 24 datasets, 20,183 files, 12.4 TB, 222 groups). If the login node
#    reboots under it, rerun it (a branch cache under ~/.cache/wrcoffea makes the rerun faster); use nohup for long plans.
python3 bin/skim.py plan --campaign v2 --plan-dir $SKIM_PLAN --era RunIII2025Summer25     # DBS through the proxy

# 4. (optional) one group locally, to see the tooling work before submitting everything. Write it to a test root,
#    not $SKIM_OUT: a group done in $SKIM_OUT is skipped by step 5, so its parts would never reach $REMOTE.
#    Delete ~/nobackup/skim_test afterwards.
python3 bin/skim.py run --plan-dir $SKIM_PLAN --dataset Muon0_Run2025C --limit-groups 1 --workers 1 \
    --out ~/nobackup/skim_test --scratch ~/nobackup/skim_scratch

# 5. Submit one Condor job per group (on the HOST, not in the container): writes parts straight to <remote>.
#    Use a new --jobs-dir for every batch while earlier jobs are still queued: the command rewrites groups.txt, the
#    plan slices and the code tarball that queued jobs read when they start.
bin/lpc_condor.sh --jobs-dir ~/nobackup/skimjobs/v2_2025 --era RunIII2025Summer25 --remote $REMOTE \
    --cpus 2 --decompression-threads 2 --submit

# 6. Watch the jobs
condor_q; ls ~/nobackup/skimjobs/v2_2025/logs/

# 7. Bring the statuses and sidecars back, look at them, classify inputs no server can serve.
#    Step 5 again is the retry, but only after this step: what is pending is read from the statuses
#    in $SKIM_OUT, which only `mirror` brings up to date (without it, finished groups are resubmitted).
python3 bin/skim.py mirror --plan-dir $SKIM_PLAN --out $SKIM_OUT --remote $REMOTE
python3 bin/skim.py -v status --plan-dir $SKIM_PLAN --out $SKIM_OUT
python3 bin/skim.py bad-files --plan-dir $SKIM_PLAN --out $SKIM_OUT --classify --rucio-account $RUCIO_ACCOUNT

# 8. Validate against DBS and the golden JSON; writes the processed-lumisection masks of every stream
python3 bin/skim.py validate --plan-dir $SKIM_PLAN --out $SKIM_OUT --skip-reread --remote $REMOTE --remote-lumis \
    --report $SKIM_PLAN/validation.json            # add --accept-bad-files once corrupt/unavailable inputs are the only problem

# 9. The analyzer fileset: data/filesets/Run3/2025/RunIII2025Summer25/skimmed/RunIII2025Summer25_data_fileset.json.
#    A family file with an unfinished selected dataset is NOT written: the command lists every missing
#    dataset and exits 1 (`--allow-missing` writes the file without them).
python3 bin/skim.py fileset --plan-dir $SKIM_PLAN --out $SKIM_OUT --era RunIII2025Summer25 --remote $REMOTE
```

Then:

- **Luminosity.** Make the per-lumisection brilcalc table of the 2025 golden JSON and run
  `scripts/processed_lumi.py` on the masks of step 8 (recipe in [luminosity.md](luminosity.md));
  write the number into `wrcoffea/config.yaml` (`lumis`) and into a `RunIII2025Summer25` entry of
  `WR_Plotter/data/lumi.yaml` (the plotter has no 2025 entry yet; see
  [adding_a_new_era.md](adding_a_new_era.md#7-wr_plotter-for-plotting)), and copy
  the masks and the result into `data/lumis/Run3/2025/RunIII2025Summer25/processed_skims_v2/`.
- **MC for 2025.** 2025 is data-only and uses the 2024 MC (`mc_alias` in `config.yaml`). The 2025 MC
  filesets are in the repository (`data/filesets/Run3/2025/RunIII2025Summer25/skimmed/`): the
  committed 2024 ones with the era and year rewritten, written by
  `python3 bin/skim.py alias-filesets --era RunIII2025Summer25`, which needs no campaign root. Rerun
  it whenever the 2024 filesets change.
- **Commit** the fileset, the lumi files and the config, push, and run the data from any machine:
  `python3 bin/run_analysis.py RunIII2025Summer25 data --dir <tag>`.
- **Unskimmed filesets** need no skims: `python3 bin/skim.py unskimmed-fileset --era RunIII2025Summer25`
  (a proxy is enough).

**MC eras.** The workflow above is data-only. An MC era uses the same steps with the selection
`--datatype mc` on `bin/lpc_condor.sh` (the wrapper passes selection arguments through to
`skim.py condor`), `mirror` and `status`. Add `--exclude _scan` to leave out the UL
multi-model scan samples (`<era>_signal_scan.json`); when they are skimmed, `skim.py split-scan`
must run before `fileset` ([Multi-model scan samples](#multi-model-scan-samples)). `validate`,
`upload` and `fileset` take the same selection. With a selection, `fileset` replaces only the
selected entries of an existing family file and keeps the others; without one it rewrites each
family file from the campaign.

Names to know: a job processes one *group* (a slice of a dataset, ~60 GB of data input or 30 GB
of MC); `skim.py group --dataset KEY --gid N` is what the job runs, followed by `skim.py ship`,
which copies the parts (adler32-verified), the sidecars and the group status to the remote tree.

**Laptop variant** (how the 2022 to 2024 and UL campaigns were partly run): the same `plan`, then
`skim.py run --plan-dir ... --out <campaign root> --scratch <scratch> --workers 3`
dataset by dataset, `validate` without `--skip-reread` (every part is re-read), `skim.py upload
--plan-dir ... --out <campaign root> --remote <remote>` to copy the parts to the remote, and
`fileset`. The closure test `scripts/skim_closure.py --plan-dir ... --dataset <key> --out ... --inputs-dir <kept inputs>
--sample <group>` compares the analyzer on a skimmed dataset with its raw inputs (`--keep-inputs`).

**Billy's defaults.** Several `skim.py` defaults are Billy's setup; a new user overrides them:

- `--remote` (`condor`, `mirror`, `upload`, `fileset`) defaults to his Wisconsin area
  (`/store/user/wijackso/WRAnalyzer/skims_v2`): always pass `--remote`.
- `--rucio-account` defaults to `wijackso`. `run`, `group`, `bad-files` and `condor` read
  `RUCIO_ACCOUNT` first, and `condor` sets it in the jobs' environment (the account their Rucio
  replica lookup, the fallback after the redirectors, authenticates as): export it (step 1).
- `--out` and `--scratch` default to his laptop layout (`skim.py run --help` prints them).
- `bin/lpc_condor.sh` defaults `SKIM_PROXY` to his proxy file and `SKIM_PLAN` / `SKIM_OUT` to his
  LPC layout: export all three (step 1 above).

Config entries flagged `"optional": true` (the low-mass `MLL-10to50` DY companions, about 1.5
billion events in 2024 alone and outside every analysis selection) are left out of plans unless
`plan --include-optional` is given.

### Stopping and resuming

Everything on disk is written atomically (parts, sidecars, group statuses),
so the laptop can be shut down at any moment. On restart, `skim.py run` with
the same arguments skips finished groups, redoes groups left in `running` or
`failed`, deletes any `.partial` leftovers, and reuses inputs still in scratch
whose checksum matches. Keep the lid open and the power connected:
`caffeinate -i` prevents idle sleep but not lid-close sleep, and a sleeping
laptop stalls the downloads.

A transfer whose output file has not grown for 180 s (`fetch.STALL_TIMEOUT`)
is killed and the file is fetched from the next source: a redirector can hand
out a server that accepts the connection and then sends nothing, and waiting
for the size-based deadline would cost up to half an hour per file. The
attempt is recorded with category `stall`.

An input that passes its checksum but cannot be read (a basket that does not
decompress or deserialize on every replica; the known cases are in
[Known input defects](#known-input-defects-skims_v2)) is corrupt at the source, so
retrying it is futile. The attempt that meets it fails, discards its partial
parts and records the file in the group status with category `corrupt`; the
next run of that group skips it from the start (also in Condor plan slices,
which carry the corrupt inputs seen in the statuses), records it again and
finishes with `bad_files`. A second permanent category is `unavailable`: files DBS still lists as
valid but that no server can serve because Rucio knows no replica at all.
Jobs cannot tell these from transient failures, so
`skim.py bad-files --classify` is run where Rucio is reachable (laptop or
login node) on the mirrored statuses: unreachable files whose every attempt
said "no servers" and that have no disk replica are rewritten as
`unavailable`; `bad-files` without `--classify` tallies every category per
dataset with the events lost. Groups whose bad files are all corrupt or
unavailable are not retried again (transient unreachable inputs still are),
`validate` lists the files, and `validate --accept-bad-files` accepts the
dataset once they are the only problem. The normalisation is unaffected: the
filesets take `genEventSumw` from the Runs trees of the parts actually
written, so a lost file costs statistics, not a bias. `skim.py group` exits 0
for a group that is done with bad inputs (they are in its status; `run` and
`condor` decide retries) and 1 only for a failed group, so Condor exit codes
count real failures.

The retry round at the LPC is therefore: `skim.py mirror` on the laptop,
`skim.py bad-files --classify`, rsync the JSON files to the LPC root, then
`bin/lpc_condor.sh ... --submit`, whose plan slices carry the permanently bad
inputs so the jobs skip them from the start.

To stop deliberately, send `SIGTERM` to the `run` process.

- It terminates its worker pool. Each worker kills the transfer in flight,
  records its group as `failed` and removes its partial parts and scratch
  files, so the next `run` redoes exactly those groups.
- Workers are separate `spawn` processes whose command line does not contain
  `skim.py`, so `pkill -f "bin/skim.py run"` stops only the parent and leaves
  the workers running as orphans.
- Every `running` group status records its worker's `pid`, so a stop script
  can terminate orphans too.
- An orphan cannot cause a double run: `run` refuses to start a group whose
  status says `running` while that pid is still a live Python process (use
  `--force` to override).

### Plan

`plan` reads the config JSONs of an era (`data/configs/<run>/<year>/<era>/*.json`,
`archived/` excluded), lists every file of every DAS path through the DBS REST
API with its size, event count and adler32 checksum (invalid files are
excluded and noted), samples one file per DAS path to learn the branch layout,
applies the branch policy, and splits the files into groups by input size
(defaults: 60 GB for data, 30 GB for MC, one group per signal point). Data
sub-eras that share a `dataset` name in the config (for example the `v1` and
`v2` versions of a prompt era) become one dataset. Config entries that are
not DAS paths need an explicit `files` list, otherwise `--skip-private`;
with `file_sizes`, `file_events` and `file_adler32` next to it (written by
`scripts/private_signal_config.py`) such private files get the same checksum
and event-count checks as DBS files, and their schema is derived from the
first input by the worker.
The plan (`plan.json` + `schemas.json`) is the single source of truth for
every later step; nothing re-queries DAS.

### Fetching inputs

Each input is downloaded to scratch with `xrdcp --cksum adler32:<DBS value>`
(a corrupt transfer fails the copy), trying the FNAL redirector, the other
redirectors, and then every disk replica Rucio knows about (as HTTPS WebDAV
with the grid proxy) in US-first site order. A file that fails everywhere is
recorded as a bad file with the full attempt log; the group continues, and
`validate` refuses the dataset unless `--accept-bad-files` is given. The next
file is prefetched while the current one is skimmed.

### Verification inside the worker

For every input: the entry count must equal the DBS event count; production
selection and recount must agree on every event; after the part is closed it
is re-read and the stored event identifiers of every input must reproduce
both digests, the Runs sums must equal the per-input sums, and the checksum
must match the sidecar. A failure deletes the part and marks the group failed.

### Analyzer closure

`scripts/skim_closure.py` runs the analyzer on the raw inputs of a dataset
and on the parts made from them and compares every histogram. Physics
histograms must agree bin by bin; cumulative cutflows must agree from the
first row that requires two leptons onward (the earlier rows count events the
skim removes); one-cut-at-a-time cutflows are diagnostics and are skipped.
On the test signal sample all 72 physics histograms agreed to 1e-15.

### Campaign validation

`validate` repeats the per-part re-read for the whole campaign and adds the
dataset-level checks: every planned file in exactly one part, entries read
equal to the DBS event count of the dataset, the summed `genEventSumw` and
`genEventCount` (reported), and for data a lumisection audit: the merged
`LuminosityBlocks` trees must contain exactly the lumisections DBS lists for
the dataset's files (anything lost or extra is a problem), and the certified
lumisections of the dataset's runs that exist in no file at all are reported
as a warning with their fraction (0.31 % for Muon0_Run2024I, mostly at run
ends; a Tier-0 property, not a skim loss). The processed lumisections are
written as `<dataset>_processed_lumis.json` in golden-JSON format next to the
parts; the integrated luminosity must be computed with brilcalc on that mask,
not on the golden JSON. The report is JSON.

### Multi-model scan samples

The pre-legacy WR signal scans (`WRtoNLtoLLJJ_MWR500to3500`, `3500to5000`,
`5000to6000`; NanoAODv7 for Autumn18, Fall17, Summer16) hold up to a few
hundred mass points per dataset, flagged per event by boolean
`GenModel_<model>` branches. Config entries with `"scan": true` are skimmed as
one dataset (every file is sampled for the schema, since the flags differ
between files) while the worker records, per input file and model, the
generated-event count and the sums of genWeight and genWeight² before the
selection. `skim.py split-scan` then reads the skimmed parts once per batch of
models and writes one ordinary dataset per mass point (Runs tree from the
recorded sums, sidecars, group status, plan entry), taking the per-model
metadata such as the cross section from `--model-config`. `skim.py fileset`
then writes the scan's config entry as its model datasets (plan entries whose
`metadata.scan_source` is the scan), keyed by the model name as the
private-grid filesets always were, so the raw scan never reaches the analyzer.
The UL18 scans have their own config, `RunIISummer20UL18_signal_scan.json`, so
their models go to `RunIISummer20UL18_signal_scan_fileset.json` and never into
the default signal fileset.

### Cross sections

The configs' cross sections are measured, not copied, by
`scripts/genxsec_campaign.py`. The normalisation rules that turn a generator
number into the `xsec` the analyzer uses are documented with the config keys
in [`docs/inputs.md`](inputs.md) ("How `xsec` is built"):

- the NNLO references for DY, W+jets, ttbar, tW, t-channel and the dibosons;
- the branching fractions;
- the `order` labels.

This section covers only the tool.

- **Measure.** `run` runs GenXSecAnalyzer on one MiniAOD file per unique MC
  sample (two for aMC@NLO) in a CMSSW container (the OrbStack VM on the
  laptop). It records what the events contain: LHE lepton flavours, W decay
  modes from GenPart. `prepare-condor` and `collect` do the same as LPC Condor
  jobs that stream the MiniAOD.
- **Review.** `review` decides from the events and the sibling samples what
  the generator header is; it never takes the header at face value:
  - **per flavour** (POWHEG `DYto2E`/`DYto2Mu` mass bins): factor 1;
  - **three-flavour** (MG5_aMC `DYto2E-*`/`DYto2Mu-*` with one flavour in the
    events): factor 1/3;
  - **inclusive** although every event is one decay channel (POWHEG `hvq`
    ttbar, `ST_wtch` tW): factor = the branching fraction, with
    BR(W to l nu) = 0.3258 including taus;
  - **includes decay** (MG5_aMC, POWHEG `ST_tch_4f`): factor 1.

  Undecidable cases are marked CHECK.
- **Apply.** `apply` writes `xsec`, `order`, `total_uncertainty` and the whole
  derivation (`xsec_source`) into the configs (dry run unless `--write`).
  `annotate-order` fills `order` into configs written before the key existed.
  - Results are keyed by primary dataset and centre-of-mass energy, as XSDB
    does. A sample measured for another campaign (same gridpack, same header)
    stands for the other eras too, but only for the eras it is recorded for
    (`--any-era` overrides).
  - `total_uncertainty` is the generator's statistical uncertainty, not a
    theory uncertainty; nothing in the analysis reads it.
- **Propagate.** Plans built before an `apply` keep the old numbers until
  `bin/skim.py refresh-metadata --plan-dir ...` copies the config fields in
  (files, groups and schemas are untouched); `skim.py fileset` then writes them
  into the filesets.

## Luminosity of the skimmed data

`validate --remote-lumis` writes `<dataset>_processed_lumis.json` next to the parts of every data
stream: the lumisections actually present in the skims. The luminosity of an era is the brilcalc
luminosity of the golden lumisections among them, computed by `scripts/processed_lumi.py`; the method,
the recipe, the numbers and the per-era notes are in [luminosity.md](luminosity.md).

## Edge eras (2022B, 2023B, 2024B)

The golden JSONs certify runs before the eras that PdmV recommends (2022B, 2023B, 2024B). Their
data are skimmed like any other stream and are part of the analysis data: the data residuals of the
current JME payloads take the run number as an input and cover these runs, and the era luminosities
in `wrcoffea/config.yaml` (`lumis`) include them. To take a B period out again, move its entries
into a separate `<era>_data_edge<period>.json` config (its fileset is then never loaded by the
analyzer) and use the luminosity without it. The JME payloads and where they come from are in
[corrections.md](corrections.md).

## Known input defects (skims_v2)

Inputs that could not be skimmed normally are recorded per group (`bad_files`
in `_groups/gNNNN.json`, tallied by `skim.py bad-files`); every input that
was read also has its record (events, generator sums, digests) in the part
sidecars. Skipped inputs are excluded consistently: the Runs-tree sums of a
dataset cover exactly the processed files, so the normalisation stays right.

| era | dataset | file | defect | events | handling |
| --- | --- | --- | --- | --- | --- |
| RunIII2024Summer24 | `DYto2L-4Jets_Bin-HT-100to400-MLL-50to120` | `0709bde6-7950-40d1-a10f-26facdbb06f1.root` | corrupt: checksum matches DBS, LZMA errors in several branches | 161,368 | skipped |
| RunIII2024Summer24 | `DYto2Mu-2Jets_Bin-2J-MLL-50` | 51 files (`skim.py bad-files` lists them) | unavailable: valid in DBS, no replica in Rucio | 8,124,688 (2.8 %) | skipped |
| Run3Summer22 | `WtoLNu-4Jets_MLNu-0to120_HT-40to100` | `18f2ea1a-97f7-490a-94a1-675992a6752a.root` | corrupt: checksum matches DBS, LZMA error | 53,991 | skipped |
| Run3Summer23 | `DYto2L-4Jets_MLL-50to120_HT-70to100` | `6ed4f6d2-7a63-42b0-811d-d8e2d2d8eb99.root` | valid in DBS but zero events; its Runs row has empty LHE/PS sum vectors | 0 | kept (zero-weight Runs row), harmless |
| Run3Summer22 | `ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8` | `7c4f3eb2-3c7e-4c21-98ed-c1892bb3a057.root` | events without `LHEScaleWeight` / `LHEPdfWeight` (empty LHE sums in Runs; `PSWeight` present) | 12,342 of 1,970,234 (0.6 %) | skimmed, see below |
| Run3Summer22EE | `WRtoNEltoElElJJ_MWR6000_N1500` | `8f24526c-43f0-40e5-b062-4864ea6329bf.root` | same as ZZZ | 621 of 210,000 (0.3 %) | skimmed, see below |

No input was skipped in Run3Summer23, Run3Summer23BPix or Run3Summer22EE. Full paths of the two files without LHE weights:
`/store/mc/Run3Summer22NanoAODv12/ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/50000/7c4f3eb2-3c7e-4c21-98ed-c1892bb3a057.root`
and
`/store/mc/Run3Summer22EENanoAODv12/WRtoNEltoElElJJ_MWR6000_N1500_TuneCP5_13p6TeV_madgraph-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/2810000/8f24526c-43f0-40e5-b062-4864ea6329bf.root`.

Files with events but no LHE weights: the skim keeps their events (with empty
`LHEScaleWeight` / `LHEPdfWeight`), and the part's Runs-tree LHE sums are the
genEventSumw-weighted mean over the other inputs (a warning is logged). The
analyzer gives those events the nominal weight for the `scale` variations
(decided 2026-10-08) and warns once per dataset; because the dataset-level
ratio comes from the files that have the weights, the total yield is
unchanged, and the variation is underestimated by about the affected share
(643 of 105,749 skimmed ZZZ events, 0.61 %; 572 of 195,178 for the WR point,
0.29 %). Any other mismatch in the number of weights stops the analyzer.

The list is complete for 2022-2024 (checked 2026-10-08): all batch jobs ran
code that failed on any group mixing files with and without LHE weights, and
only the groups above did; a scan of all 4,124 MC part sidecars found no
dataset whose parts disagree on the length of the LHE sums.

Samples without LHE weights by design: the pure-Pythia 8 dibosons
`WW_TuneCP5_13p6TeV_pythia8`, `WZ_TuneCP5_13p6TeV_pythia8` and
`ZZ_TuneCP5_13p6TeV_pythia8` have no `LHEScaleWeight` / `LHEPdfWeight` in any
Run 3 era (they do have the 4 `PSWeight` entries). The analyzer fills their
`scale` variations with the nominal weight and warns once per dataset.

UL18: `validate --remote-lumis` fails SingleMuon 2018A and 2018D and EGamma 2018A, as expected:
DBS's per-file lumisection lists are incomplete there, so the skims hold lumisections no DBS file
lists, while the planned files equal the current valid DBS files. Nothing is lost.

## Running on Condor at the LPC

One job per group, in the coffea container on the worker, writing straight to
Wisconsin, the skims' final home (`--remote` names any other xrootd root). Statuses and
sidecars store part names relative to the dataset directory (older absolute records still load), so the
remote tree has exactly the layout of a local campaign root.

Setup: a clone of the repository under `~/nobackup` on the LPC (the workflow above), a proxy on a
shared path, and the environment variables the wrapper reads: `SKIM_PLAN` (the plan directory),
`SKIM_OUT` (the campaign root that holds the mirrored statuses), `SKIM_PROXY` (the proxy file the
jobs use) and `SKIM_IMAGE` (the container; the default is the coffea almalinux9 image on cvmfs).
When developing on a laptop instead, rsync `bin/skim.py`, `wrcoffea/`, `data/configs` and
`data/lumis` to the LPC clone and the laptop root's JSON files
(`--include='*/' --include='*.json' --exclude='*'`) to `SKIM_OUT`, so the LPC knows which groups
the laptop already finished.

On the login node, `bin/lpc_condor.sh` does the two halves in the right
places: Condor only exists on the host and the skim tooling only runs inside
the container, so the wrapper saves the queue (`condor_q -af SkimGroup
JobStatus Iwd`) to a file, runs `skim.py condor --inflight-file` inside the
container, and runs `condor_submit` on the host when `--submit` is given:

```
bin/lpc_condor.sh --jobs-dir ~/nobackup/skimjobs/v2 --cpus 2 --decompression-threads 2 [selection] [--submit]
```

The wrapper's defaults for `SKIM_PLAN`, `SKIM_OUT` and `SKIM_PROXY` are Billy's (his plan, his
campaign root, his proxy file): a new user exports all three, as in step 1 of the workflow.
`SKIM_IMAGE` is optional. Pass `--remote` too; its default is his Wisconsin area.

`condor` lists the pending groups exactly like `run` does and leaves out the
groups `condor_q` still shows (ClassAd `SkimGroup` = `era/dataset:gid`). It
writes:

- the JDL;
- `run_group.sh`;
- a code tarball;
- one plan slice per dataset;
- `groups.txt`.

Rerunning it later submits only what is still pending, so it is also the retry
command.

The JDL:

- **Container:** `coffeateam/coffea-almalinux9-noml:2026.9.0-py3.12` from
  `/cvmfs/unpacked.cern.ch`.
- **Resources:** `--cpus` CPUs, `--memory-mb` (default 4000) and `--disk-gb`
  (default 20). A job is CPU-bound by the LZMA decompression of its input,
  which releases the GIL, so `--cpus 4 --decompression-threads 4` makes it
  about 1.7 times faster than one core.
- **Slow nodes:** a job running longer than `--max-hours` (default 8) is held
  and released to restart elsewhere, at most 4 starts. A restart avoids the
  last three machines the job ran on, and `--avoid-hosts` excludes nodes
  outright.
- **Logs:** `logs/<dataset>_g<gid>.out|.err` and `logs/condor.log` in the jobs
  directory. Each job runs `skim.py group` on its slice, then
`skim.py ship`, which copies parts (adler32-verified), sidecars and finally the
group status to `<remote>/<out_dir>/`. Afterwards `skim.py mirror` pulls the
JSON files of the remote tree into a local root (LPC or laptop), where
`status`, `validate` and `fileset` work as usual. Parts written by Condor jobs
exist only at the remote root, so validate them with `--skip-reread --remote
<campaign URL> --remote-lumis`: the sidecar-level checks run as for local
parts, every remote part is stat-ed and must have the size its sidecar
records (the shipper already verified the adler32 server-side), and for data
the LuminosityBlocks trees are read through xrootd for the DBS and golden-JSON
lumi checks and the processed-lumi mask. Do not list directories at
Wisconsin: a listing takes 30 s on a good day and times out on a busy one,
while a stat or a small copy by name takes about a second.

## Output format and earlier skims

The compression and part-size study behind the defaults is in
[studies/skim_output_format.md](studies/skim_output_format.md). The skims made before October 2026
(`WRAnalyzer/skims/` at Wisconsin: 1209 branches, about 3.2 kB per event, zlib:1, no
LuminosityBlocks tree, genEventSumw from a separate DAS pass) are no longer used by any fileset.
