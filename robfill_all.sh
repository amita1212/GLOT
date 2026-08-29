#!/usr/bin/env bash
# RoBERTa hyperbolic-fill campaign: 6 arms x 2 tasks, 3 parallel workers.
#
# NO PREWARM STEP: data/roberta-base_{cola,sts}_{train,val}_batches already
# exist from the released RoBERTa campaign, so every run here starts warm --
# which is required, because a cold cache consumes the global RNG through the
# shuffled loader and shifts the same seed by several MCC. Verified present
# before launch; do not add a prewarm pass that would rebuild them mid-flight.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results

MODEL="${MODEL:-roberta-base}"
TAG="${TAG:-robfill}"
TASKS="${TASKS:-cola stsb}"
NW=3

export MODEL TAG

echo "[robfill] ===== START $(date -Is) model=$MODEL tasks=$TASKS ====="
for t in cola sts; do
    d="data/roberta-base_${t}_train_batches"
    [ -d "$d" ] || echo "[robfill] WARNING: cache missing: $d (runs will be COLD)"
done

for ((i = 0; i < NW; i++)); do
    bash robfill_worker.sh "$i" "$NW" "$TASKS" > "logs/${TAG}_par_w${i}.log" 2>&1 &
    echo "[robfill] worker $i pid $!"
done
wait
echo "[robfill] all workers finished $(date -Is)"

PY=~/glotenv/bin/python
for task in $TASKS; do
    "$PY" merge_glob.py "results/campaign_${TAG}_${task}_w*.csv" \
        "results/campaign_${TAG}_${task}.csv"
done

echo "[robfill] ===== ALL DONE $(date -Is) ====="
