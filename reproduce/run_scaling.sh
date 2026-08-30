#!/usr/bin/env bash
# Reproduce Figure 3 (test-time scaling, N in {1,2,4,8,16,32}).
#
# Same inputs and the same caveats as reproduce/run_table1.sh: this documents
# the procedure rather than running end to end, because the CT-RATE
# validation reports and the 32-sample candidate pool are not shipped here.
set -euo pipefail

OUT=reproduce/out
mkdir -p "$OUT"

# 1. Reuse the 32-sample-per-volume candidate pool from run_table1.sh step 1
#    (do not resample per N -- Figure 3 uses nested prefixes of one common
#    pool, so N=2 is the first 2 of the same 32 samples used for N=32, etc.,
#    which is why Random-of-N's mean does not change monotonically with N in
#    reproduce/results_scaling.csv: it is resampling *within* a fixed pool).
for N in 1 2 4 8 16 32; do
  python - "$N" <<'PY'
import json, sys
n = int(sys.argv[1])
with open("reproduce/out/candidates.jsonl", encoding="utf-8") as fin, \
     open(f"reproduce/out/candidates_N{n}.jsonl", "w", encoding="utf-8") as fout:
    for line in fin:
        row = json.loads(line)
        row["candidates"] = row["candidates"][:n]
        fout.write(json.dumps(row) + "\n")
PY
  for MODEL_NAME in "PRM_0P5B:work/prm_0.5b/final" "PRM_3B:work/prm_3b/final"; do
    NAME=${MODEL_NAME%%:*}; MODEL=${MODEL_NAME##*:}
    python ../pipeline/10_select_bestofn.py --model "$MODEL" \
        --candidates "$OUT/candidates_N${N}.jsonl" --out "$OUT/selected_${NAME}_N${N}.jsonl"
    # ... extract labels for the selected reports, then:
    python ../pipeline/11_score_crg.py \
        --gt-labels "$OUT/gt_labels.csv" --pred-labels "$OUT/pred_labels_${NAME}_N${N}.csv" \
        --selected "$OUT/selected_${NAME}_N${N}.jsonl" --dedup-series
  done
done

# Random-of-N at each N: draw N of the pool uniformly at random, 20 times,
# and report the mean and standard deviation of CRG across draws (the
# CRG_std column in reproduce/results_scaling.csv; Random-of-N is the only
# selector reported this way, everyone else gets a 95% scan-level bootstrap
# CI from pipeline/11_score_crg.py). Oracle at each N: the highest-CRG
# candidate of the first N, an upper bound rather than a trained selector.
#
# The LLM judge (Table 1's Judge column, reproduce/results_table1.csv) is
# reported only at N=8 in the paper and is not part of Figure 3; this repo
# does not ship the judging script, only its reported means (see the
# top-level README's evaluation-metrics note).
