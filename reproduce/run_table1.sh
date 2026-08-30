#!/usr/bin/env bash
# Reproduce Table 1 (Best-of-8 selection on the CT-RATE validation split).
#
# This script documents the exact sequence used to produce
# reproduce/results_table1.csv from a trained reward model and a set of
# sampled candidates; it is not a one-command rerun, because two of its
# inputs are not shipped in this repository (see the top-level README,
# "What is (and is not) in this repository"):
#
#   - the CT-RATE validation volumes/reports (CT-RATE's own licence), and
#   - the 32-sample-per-volume candidate pool used for the paper.
#
# Fill in the placeholders below and run each step in order. Every step
# after candidate generation is deterministic given its inputs, so the
# CRG/BLEU-1/METEOR/ROUGE-L/CIDEr numbers this produces should match
# reproduce/results_table1.csv exactly; the Judge column (LLM-as-judge) is
# not reproduced by this script -- see reproduce/run_scaling.sh's note.
set -euo pipefail

CT_RATE_VAL_METADATA=<path-to-ct-rate>/validation_metadata.csv
CT_RATE_VAL_REPORTS=<path-to-ct-rate>/validation_reports.csv
CANDIDATES_PER_VOLUME=32          # the paper draws Best-of-8 from a 32-sample pool
PRM_0P5B=work/prm_0.5b/final       # from pipeline/09_train_prm.py, tau=0.3
PRM_3B=work/prm_3b/final           # from pipeline/09_train_prm.py, tau=0.4
OUT=reproduce/out
mkdir -p "$OUT"

# 1. Sample CANDIDATES_PER_VOLUME reports per validation volume (needs the
#    CT-CHAT checkpoint and a GPU; see pipeline/02_generate_reports.py).
#    Sampling is seeded but not bit-reproducible across hardware/library
#    versions -- see the top-level README's note on step 2.
python ../pipeline/02_generate_reports.py <shard> <n> \
    --ctchat ... --weights ... --base ... --encodings ... \
    --num-samples "$CANDIDATES_PER_VOLUME" \
    --out "$OUT/candidates.jsonl"

# 2. Score and select with each reward model (Best-of-8: pass only the first
#    8 of the 32 samples per volume to reproduce the paper's protocol).
for MODEL_NAME in "PRM_0P5B:$PRM_0P5B" "PRM_3B:$PRM_3B"; do
  NAME=${MODEL_NAME%%:*}; MODEL=${MODEL_NAME##*:}
  python ../pipeline/10_select_bestofn.py --model "$MODEL" \
      --candidates "$OUT/candidates.jsonl" --out "$OUT/selected_${NAME}.jsonl"
done

# 3. Extract 18-class labels for the reference reports and each selected set
#    with the RadBERT-based classifier released with CT-CHAT (not part of
#    this repository -- see the README).
#    ... labeller.py --reports "$CT_RATE_VAL_REPORTS" --out "$OUT/gt_labels.csv"
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
