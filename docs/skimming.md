# Skimming

The skims are NanoAOD files reduced to the events and branches the WR analysis
can ever need. The toolkit lives in `wrcoffea/skim/` with one driver,
`bin/skim.py`. It replaces the 2026-02 skimmer (per-file Condor jobs, tarballs,
`hadd`-style merging) with a plan-driven campaign in which every input file is
downloaded with checksum verification, skimmed from local scratch, written into
size-bounded parts with one fixed schema per dataset, and verified twice before
the local copy is deleted.

Status (2026-10-10):

- The skims_v2 campaign is done for 2022, 2023, 2024 and UL18, and every active
  skimmed fileset of those eras points at it
  (`root://cmsxrootd.hep.wisc.edu//store/user/wijackso/WRAnalyzer/skims_v2/`).
- UL17 is running on LPC Condor.
- UL16, UL16APV and the 2025 data have plans but no skims yet.

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

## Workflow

```bash
# 1. Plan: resolve configs through DBS, sample branch layouts, group files into jobs
#    (defaults: zstd:19 compression, 2 GB parts; see "Output format study")
python3 bin/skim.py plan --campaign v2 --plan-dir plans/v2 --era RunIII2024Summer24

# 2. Run locally, dataset by dataset (3 workers, each prefetching the next file)
python3 bin/skim.py run --plan-dir plans/v2 --dataset TTto2L2Nu --workers 3 \
    --out ~/work/skims_v2 --scratch ~/work/skim_scratch

# 3. Progress and problems
python3 bin/skim.py status --plan-dir plans/v2 --out ~/work/skims_v2 -v

# 4. Validate (re-reads every part, checks DBS counts, Runs sums, golden-JSON coverage)
python3 bin/skim.py validate --plan-dir plans/v2 --out ~/work/skims_v2 --report plans/v2/validation.json

# 5. Analyzer closure on a dataset whose raw inputs were kept (--keep-inputs)
python3 scripts/skim_closure.py --plan-dir plans/v2 --dataset TTto2L2Nu --out ~/work/skims_v2 \
    --inputs-dir ~/work/skim_scratch/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8 --sample tt_tW

# 6. Upload to Wisconsin (checksummed, resumable) and write the analyzer filesets
python3 bin/skim.py upload  --plan-dir plans/v2 --out ~/work/skims_v2
python3 bin/skim.py fileset --plan-dir plans/v2 --out ~/work/skims_v2 --era RunIII2024Summer24
# 7. Unskimmed filesets (the original NanoAOD files, for --unskimmed runs; needs only a proxy)
python3 bin/skim.py unskimmed-fileset --era RunIII2024Summer24   # or --all
```

Steps 2 and 6 can also run on the LPC; most groups of the 2022-2024 and UL18
campaigns did.

- `skim.py condor` submits one job per pending group.
- Each job runs `skim.py group` and then `skim.py ship`, which copies the
  parts to Wisconsin.
- `skim.py mirror` brings the sidecars and statuses back to a local root,
  where `status`, `validate` and `fileset` work as usual.

See [Running on Condor at the LPC](#running-on-condor-at-the-lpc).
`skim.py group --dataset KEY --gid N` processes a single group (the unit of a
Condor job); the plan directory is the only input it needs.

Config entries flagged `"optional": true` (the low-mass `MLL-10to50` DY
companions, about 1.5 billion events in 2024 alone and outside every analysis
selection) are left out of plans unless `plan --include-optional` is given.
A data-only era listed under `mc_alias` in `wrcoffea/config.yaml` (2025 reuses
the 2024 MC, as PdmV recommends) gets its MC filesets written from the alias
era's skims with the era and year rewritten.

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
out a server that accepts the connection and then sends nothing (seen on
2026-10-05 from campus, three transfers frozen for 20 minutes), and waiting
for the size-based deadline would cost up to half an hour per file. The
attempt is recorded with category `stall`.

An input that passes its checksum but cannot be read (a basket that does not
decompress or deserialize; the first one met was
`DYto2L-4Jets_Bin-HT-100to400-MLL-50to120` file `0709bde6-...`, LZMA data
errors in several branches on every replica) is corrupt at the source, so
retrying it is futile. The attempt that meets it fails, discards its partial
parts and records the file in the group status with category `corrupt`; the
next run of that group skips it from the start (also in Condor plan slices,
which carry the corrupt inputs seen in the statuses), records it again and
finishes with `bad_files`. A second permanent category is `unavailable`: files DBS still lists as
valid but that no server can serve because Rucio knows no replica at all
(44 files, 2.4 % of the events, of `DYto2Mu-2Jets_Bin-2J-MLL-50` on
2026-10-06). Jobs cannot tell these from transient failures, so
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

The golden JSONs certify runs before the eras that PdmV recommends (2022B,
2023B, 2024B); their data are skimmed like any other stream, but whether the
analysis can use them depends on the jet corrections.

History. Until 2026-10-07 the analyzer used older JME payloads: the 2023
residual (Summer23Prompt23_V2) started at run 367080 = 2023C, so 2023B data
failed (flow = error), and the 2022 residual (22Sep2023 RunCD_V3) had no run
input and was named and documented for 2022C/D only, so 2022B would have
received it silently. Both B periods were moved out of the analysis data into
edge-era configs (`*_data_edge2022B.json`, `*_data_edge2023B.json`) on
2026-10-07.

The same day the 2022 and 2023 payloads were updated to the latest official
ones (V4, table below). Their data residuals take the run number as an input
(flow = error outside the bins): 2022 `[355065, 359022)`, i.e. 2022B, C and D
together; 2022EE bins at 359022 / 360332 / 362350 / 362761 (E, F, G); 2023
`[366365, 367765, 369803)`, whose first bin includes 2023B; 2023BPix
`[369803, 372416)`. With both B periods covered officially, they were brought
back into the analysis data (Billy, 2026-10-07), like 2024B:

- 2022B is in `Run3Summer22_data.json` again; Run3Summer22 uses the
  2022B+C+D processed luminosity (ee 8.082 / mumu 8.076, so 8.08 fb⁻¹; the 2022
  skims were switched in on 2026-10-08).
- 2023B is in `Run3Summer23_data.json` again; Run3Summer23 uses the 2023B+C
  processed luminosity (ee 18.587 / mumu 18.588, so 18.59 fb⁻¹).
- 2024B is part of the 2024 analysis data and of its luminosity (109.70): the
  2024 residual (Summer24Prompt24_V5) starts at run 378971, which covers it.

To take a B period out again, move its entries into a separate
`<era>_data_edge<period>.json` config (its fileset is then never loaded by
the analyzer) and use the luminosity without it (2022C+D 7.98, 2023C 17.96).

Where the jet corrections come from: the JME POG publishes them as
correctionlib files on cvmfs, one directory per campaign with dated versions
and a `latest` link, under
`/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/<campaign>/` (for example
`Run3-22CDSep23-Summer22-NanoAODv12`, `Run3-22EFGSep23-Summer22EE-NanoAODv12`,
`Run3-23CSep23-Summer23-NanoAODv12`, `Run3-23DSep23-Summer23BPix-NanoAODv12`,
`Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15`); each has a
`changes.md`. The source repositories are
`gitlab.cern.ch/cms-analysis-corrections/JME/<campaign>`. The older mirror
`/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/JME/` holds
older versions (on 2026-10-07 its 2022 file was still `RunCD_V2`). The copies
in `data/jsonpog/JME/` are tagged in `wrcoffea/config.yaml` (`jerc_jsons`);
compare versions by the canonical JSON content, not the gzip checksum.

Payloads in the repo (updated 2026-10-07 from cvmfs `latest`; only
`jet_jerc.json.gz` and `fatJet_jerc.json.gz` changed, the jet veto maps and
jet ID files were identical):

| era | before 2026-10-07 | now (= latest official) |
| --- | --- | --- |
| Run3Summer22 | `Summer22_22Sep2023_RunCD_V3`, `JRV1` | `Summer22_22Sep2023_V4`, `JRV2` (2026-06-05) |
| Run3Summer22EE | `Summer22EE_22Sep2023_RunE/F/G_V3`, `JRV1` | `Summer22EE_22Sep2023_V4`, `JRV2` (2026-06-05) |
| Run3Summer23 | `Summer23Prompt23_V2`, `RunCv1234_JRV1` | `Summer23Prompt23_V4`, `RunCv1234_JRV3` (2026-07-15) |
| Run3Summer23BPix | `Summer23BPixPrompt23_V3`, `RunD_JRV1` | `Summer23BPixPrompt23_V4`, `RunD_JRV3` (2026-07-15) |
| RunIII2024Summer24 | `Summer24Prompt24_V5`, `JRV2` | unchanged |

The new JER payloads give the nominal scale factor and its uncertainty as
separate corrections (`ScaleFactor` without a `systematic` input, plus
`SFUncertainty`), the schema the analyzer already handled for 2024.

The update is a repackaging, not a recalibration: comparing old and new
files on a grid of eta, pT (and phi for BPix), the data L2L3 residuals, the
MC L2Relative, the JES Total uncertainty, the JER scale factors and the
resolutions agree to the printed precision for all four eras; only the JER
scale factor "up" value differs by about 0.001 in places (nominal plus
uncertainty instead of a stored up value). What changes is the explicit run
validity of the data residuals. Smoke tests after the switch (muon and
electron data, signal, ttbar with `--systs jes jer`, one chunk each, all
four eras) ran cleanly with filled, finite histograms.

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

UL18 (skim completed 2026-10-09: 742 groups, 318 datasets, 5.51 billion
events in, 280.6 million kept, 474 GB):

- Three official signal files could not be read, costing 1-3 % of the events
  of three mass points.
- The three pre-legacy scan samples validated with per-model event counts
  equal to DBS.
- `validate --remote-lumis` passes 315 of 318 datasets. The three failures,
  SingleMuon 2018A and 2018D and EGamma 2018A, are expected: the skims hold
  lumisections (612, 2799 and 38) that no DBS file lists. DBS's per-file
  lumisection lists are incomplete there (for example run 316240 of EGamma:
  DBS lists 1184 of its 1222 lumisections), while the planned files equal the
  current valid DBS files. Nothing is lost.

## Running on Condor at the LPC

One job per group, in the coffea container on the worker, writing straight to
Wisconsin: the EOS user area at FNAL is 86 % full (1.73 of 2 TB), and the
skims' final home is Wisconsin anyway. Statuses and sidecars store part names
relative to the dataset directory (older absolute records still load), so the
remote tree has exactly the layout of a local campaign root.

Setup (once per code change, from the laptop): rsync `bin/skim.py`,
`wrcoffea/` and `data/lumis` to `~/nobackup/WrCoffea_skim_v2/` on the LPC and
the plan to `~/nobackup/skims_v2/plans/`; rsync the laptop root's JSON files
(`--include='*/' --include='*.json' --exclude='*'`) to `~/nobackup/skims_v2/`
so the LPC knows which groups the laptop already finished. Jobs need a valid
proxy on the LPC (`voms-proxy-init`, or copy `~/.x509up_proxy` to
`/uscms/homes/b/bjackson/x509up_u24705`).

On the login node, `bin/lpc_condor.sh` does the two halves in the right
places: Condor only exists on the host and the skim tooling only runs inside
the container, so the wrapper saves the queue (`condor_q -af SkimGroup
JobStatus`) to a file, runs `skim.py condor --inflight-file` inside the
container, and runs `condor_submit` on the host when `--submit` is given:

```
bin/lpc_condor.sh --jobs-dir ~/nobackup/skimjobs/v2 --cpus 2 --decompression-threads 2 [selection] [--submit]
```

(`SKIM_PLAN`, `SKIM_OUT`, `SKIM_PROXY` and `SKIM_IMAGE` override the defaults.)

`condor` lists the pending groups exactly like `run` does and leaves out the
groups `condor_q` still shows (ClassAd `SkimGroup` = `era/dataset:gid`). It
writes:

- the JDL;
- `run_group.sh`;
- a 140 kB code tarball;
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
(`WRAnalyzer/skims/` at Wisconsin) are no longer used by any fileset; what they were is recorded in
[history.md](history.md).
