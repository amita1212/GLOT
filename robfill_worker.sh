#!/usr/bin/env bash
# One worker of the RoBERTa hyperbolic-fill campaign.
#
#   usage: robfill_worker.sh <worker_index> <n_workers> "<tasks>"
#
# WHY THIS EXISTS: the released RoBERTa campaign searched only the CALIBRATION
# arms (baseline, paper_tau, density_fix, no_graph, A), because the question
# RoBERTa was brought in to answer was "is tau miscalibrated?". That leaves the
# whole hyperbolic story resting on a single encoder. These six arms close that
# gap so the B-harm and the C-gain can be tested on a second backbone.
#
# WHY FULL 40-TRIAL SEARCH AND NOT CONFIRM-ONLY: the existing RoBERTa baseline
# was tuned with 40 trials ON RoBERTa. Dropping in BERT-selected configs for the
# new arms would hand the baseline exactly the budget advantage that forced the
# Stage A retraction (see paper App. "A retracted result of our own"), only
# pointed the other way -- it would manufacture a negative. Equal budget or
# nothing.
#
# Flags below MUST stay identical to roberta_worker.sh so the new arms are
# comparable to the already-released baseline in the same table.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python

IDX="${1:?worker index}"
N="${2:?n workers}"
TASKS="${3:-cola stsb}"

export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL="${MODEL:-roberta-base}"
TAG="${TAG:-robfill}"
ALL_ARMS=(B C AB AC BC ABC)
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
TRIALS="${TRIALS:-40}"

MY_ARMS=()
for ((j = IDX; j < ${#ALL_ARMS[@]}; j += N)); do
    MY_ARMS+=("${ALL_ARMS[$j]}")
done

echo "[w$IDX] model=$MODEL arms: ${MY_ARMS[*]}"
if [ ${#MY_ARMS[@]} -eq 0 ]; then
    echo "[w$IDX] nothing to do"
    exit 0
fi

for task in $TASKS; do
    echo "[w$IDX] ===== $task ===== $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model "$MODEL" \
        --arms "${MY_ARMS[@]}" --trials "$TRIALS" --stage both \
        --wide --fine_baseline --confirm_seeds $SEEDS \
        --out "results/campaign_${TAG}_${task}_w${IDX}.csv" \
        >> "logs/${TAG}_w${IDX}_${task}.log" 2>&1
    echo "[w$IDX] done $task rc=$? $(date -Is)"
done

echo "[w$IDX] FINISHED $(date -Is)"
