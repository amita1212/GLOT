#!/usr/bin/env bash
# One parallel worker of the wide sweep.
#
#   usage: wide_worker.sh <worker_index> <n_workers> "<tasks>"
#
# Each worker owns a DISJOINT subset of arms and writes its OWN csv:
#     results/campaign_wide_<task>_w<idx>.csv
#
# WHY SEPARATE CSVs: ResultsCSV is a plain append-to-file with no locking. Two
# workers appending to one file would interleave rows and race the run_key dedup
# check, silently corrupting the campaign. Per-worker files make the split
# lock-free; merge_wide.py reassembles them for analysis.
#
# WHY OMP_NUM_THREADS=1: a single run already pegs exactly one core (measured at
# 101% CPU). With 4 workers, letting torch/BLAS each spawn 8 threads would
# oversubscribe 8 vCPUs by 4x and make every worker slower than the serial run.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python

IDX="${1:?worker index}"
N="${2:?n workers}"
TASKS="${3:-stsb cola}"

export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL="bert-base-uncased"
ALL_ARMS=(baseline no_graph A B C AB AC BC ABC)
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
TRIALS="${TRIALS:-40}"

# Stride assignment: worker i takes arms i, i+N, i+2N, ...
MY_ARMS=()
for ((j = IDX; j < ${#ALL_ARMS[@]}; j += N)); do
    MY_ARMS+=("${ALL_ARMS[$j]}")
done

echo "[w$IDX] arms: ${MY_ARMS[*]}"
if [ ${#MY_ARMS[@]} -eq 0 ]; then
    echo "[w$IDX] nothing to do"
    exit 0
fi

for task in $TASKS; do
    echo "[w$IDX] ===== $task ===== $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model "$MODEL" \
        --arms "${MY_ARMS[@]}" --trials "$TRIALS" --stage both \
        --wide --fine_baseline --confirm_seeds $SEEDS \
        --out "results/campaign_wide_${task}_w${IDX}.csv" \
        >> "logs/wide_w${IDX}_${task}.log" 2>&1
    echo "[w$IDX] done $task $(date -Is)"
done

echo "[w$IDX] FINISHED $(date -Is)"
