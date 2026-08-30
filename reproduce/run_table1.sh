#!/usr/bin/env bash
# Reproduce Table 1 (Best-of-8 selection on the CT-RATE validation split).
#
# This script documents the exact sequence used to produce
# reproduce/results_table1.csv. The validation manifest and the 32-sample
# candidate pool are included in this repository (data/val_manifest.csv,
# data/candidates_pool/*.jsonl) -- see the top-level README, "What is (and
# is not) in this repository". The one input still not shipped here is the
# RadBERT-based 18-class labeller released with CT-CHAT, needed for step 3;
# everything from step 4 onwards is deterministic given its output, so the
# CRG/BLEU-1/METEOR/ROUGE-L/CIDEr numbers this produces should match
# reproduce/results_table1.csv exactly. The Judge column (LLM-as-judge) is
# not reproduced by this script -- see reproduce/run_scaling.sh's note.
set -euo pipefail

CANDIDATES=../data/candidates_pool           # 24 shards, 3,039 volumes, 32 candidates each
MANIFEST=../data/val_manifest.csv            # volume_id, in_dedup1564
PRM_0P5B=work/prm_0.5b/final                  # from pipeline/09_train_prm.py, tau=0.3
PRM_3B=work/prm_3b/final                      # from pipeline/09_train_prm.py, tau=0.4
OUT=out
mkdir -p "$OUT"

# 1. Concatenate the 24 shards into one candidates.jsonl (already 3,039
#    records of 32 candidates each -- no sampling to do, the pool is fixed).
cat "$CANDIDATES"/shard*.jsonl > "$OUT/candidates.jsonl"

# 2. Score and select with each reward model (Best-of-8: pipeline/10 takes
#    the first 8 of each record's 32 candidates to reproduce the paper's
#    protocol -- see pipeline/10_select_bestofn.py's docstring).
for MODEL_NAME in "PRM_0P5B:$PRM_0P5B" "PRM_3B:$PRM_3B"; do
  NAME=${MODEL_NAME%%:*}; MODEL=${MODEL_NAME##*:}
  python ../pipeline/10_select_bestofn.py --model "$MODEL" \
      --candidates "$OUT/candidates.jsonl" --out "$OUT/selected_${NAME}.jsonl"
done

# 3. Extract 18-class labels for the reference reports and each selected set
#    with the RadBERT-based classifier released with CT-CHAT (not part of
#    this repository -- see the README). Reference reports themselves are
#    CT-RATE's own (gated; not part of this repository either).
#    ... labeller.py --reports <ct-rate-validation-reports> --out "$OUT/gt_labels.csv"
#    ... labeller.py --reports <selected-report-text> --out "$OUT/pred_labels_<name>.csv"

# 4. Score CRG for each selector, on the paper's 1,564-case protocol.
for NAME in PRM_0P5B PRM_3B; do
  python ../pipeline/11_score_crg.py \
      --gt-labels "$OUT/gt_labels.csv" --pred-labels "$OUT/pred_labels_${NAME}.csv" \
      --selected "$OUT/selected_${NAME}.jsonl" --dedup-series
done

# Greedy, Random-of-8 and Oracle are read off the same candidate pool without
# a trained reward model (greedy = the generator's own top-1; Random-of-8 =
# one of the first 8 samples drawn uniformly at random per volume, averaged
# over 20 draws; Oracle = the highest-CRG of the first 8 samples per volume).
# BLEU-1/METEOR/ROUGE-L/CIDEr are computed against the reference report with
# the standard implementations cited in the paper's evaluation section.
