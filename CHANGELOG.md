# Changelog

## [Unreleased]

Reproducibility and consistency pass ahead of the reviewed state, tracked in
this pull request:

- Added `pipeline/11_score_crg.py`, computing CRG from a Best-of-N selection
  run with the paper's exact formula, including the 1,564-case
  series-de-duplication protocol as a flag and a scan-level bootstrap CI.
- Added `reproduce/` with the reported Table 1 and Figure 3 numbers
  (`results_table1.csv`, `results_scaling.csv`) and the procedures that
  produced them (`run_table1.sh`, `run_scaling.sh`).
- `scripts/verify_release.py` now also recomputes and prints the
  image-vs-LLM label agreement and the assert/deny support split for each
  labelling rule, cited in the paper's discussion but previously only
  asserted in the README.
- Clarified the corpus table and surrounding prose: the LLM label set covers
  all 5,000 studies, the image-evidence set covers 4,955 of them (not "the
  same 5,000" for both, which was ambiguous).
- README: documented all three `08_build_prm_dataset.py` invocations and
  both `09_train_prm.py` backbones, not only the 0.5B/tau=0.3 pair; added
  the reproducibility gap for the validation manifest and candidate pool to
  "What is (and is not) in this repository"; retitled to match the repository
  slug (`ct-report-prm-corpus`).
- Pinned `requirements.txt` to the exact versions used to build and train
  this release, with a Python 3.10 note; two packages (`requests`,
  `radgraph`) are pinned to current versions but were not independently
  verified against the training run's environment records -- flagged inline.
- Added `data/val_manifest.csv` (the real 3,039-volume validation manifest,
  with the 1,564-case series-dedup flag) and `data/candidates_pool/*.jsonl`
  (the real 32-sample-per-volume candidate pool used for Table 1 and
  Figure 3, 24 shards, 128 MB, 3,039 records) -- both were flagged as an
  unresolved reproducibility gap in this PR's first revision; the training
  environment's evaluation-output CSVs turned out to still hold the exact
  data, so this closes that gap rather than leaving it open for camera-ready.
  Each candidate-pool record had its local filesystem path (`npz`, which
  embedded the authors' cluster username) stripped before release;
  `scripts/verify_release.py` now checks that field is gone.

## [v1.0-anon] (planned, not yet tagged)

To be cut from this branch once merged into the reviewed state. See the pull
request description for why it is not tagged yet.
