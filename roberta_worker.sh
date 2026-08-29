#!/usr/bin/env bash
# One parallel worker of the RoBERTa calibration campaign.
#
#   usage: roberta_worker.sh <worker_index> <n_workers> "<tasks>"
#
# Mirrors wide_worker.sh exactly (per-worker CSV, OMP_NUM_THREADS=1, --wide)
# but over the CALIBRATION arms rather than the hyperbolic stages, because the
# question RoBERTa answers is "is tau miscalibrated, and does density-matching
# repair it?" -- not "does hyperbolic geometry help?".
#
# Arms:
#   baseline    GLOT as published (absolute tau, tuned)
#   paper_tau   GLOT pinned to the published tau grid {0.1,0.3,0.6}
#   density_fix quantile-thresholded graph, density-matched across backbones
#   no_graph    control that deletes the graph entirely
#   A           Poincare token graph (the one stage still in play)
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python

IDX="${1:?worker index}"
N="${2:?n workers}"
TASKS="${3:-stsb cola}"

export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL="${MODEL:-roberta-base}"
TAG="${TAG:-rob}"
ALL_ARMS=(baseline no_graph paper_tau density_fix A)
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
