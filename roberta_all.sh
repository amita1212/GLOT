#!/usr/bin/env bash
# Full RoBERTa calibration campaign: prewarm, then 4 parallel workers, then analyse.
#
# WHY PREWARM FIRST AND SERIALLY: the hidden-state cache is consumed by a
# shuffled DataLoader whose iteration draws from the global RNG. A cold cache
# therefore changes classifier init at a FIXED seed (measured 40.37 vs 45.54 MCC
# on CoLA). If the 4 workers started cold they would also race to build the same
# cache directory. One serial prewarm pass makes every later run warm and
# comparable.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results

MODEL="${MODEL:-roberta-base}"
TAG="${TAG:-rob}"
TASKS="${TASKS:-stsb cola}"
NW=4

export MODEL TAG

echo "[rob] ===== START $(date -Is)  model=$MODEL tasks=$TASKS ====="

echo "[rob] --- prewarm (serial, override_precompute=1) ---"
bash prewarm_model.sh "$MODEL" -1 $TASKS 2>&1 | tail -20
echo "[rob] prewarm done $(date -Is)"

echo "[rob] --- launching $NW workers ---"
for ((i = 0; i < NW; i++)); do
    bash roberta_worker.sh "$i" "$NW" "$TASKS" > "logs/${TAG}_par_w${i}.log" 2>&1 &
    echo "[rob] worker $i pid $!"
done
wait
echo "[rob] all workers finished $(date -Is)"

PY=~/glotenv/bin/python
for task in $TASKS; do
    "$PY" merge_glob.py "results/campaign_${TAG}_${task}_w*.csv" \
        "results/campaign_${TAG}_${task}.csv"
done

for task in $TASKS; do
    f="results/campaign_${TAG}_${task}.csv"
    [ -e "$f" ] || continue
    echo
    echo "[rob] ================= $task ================="
    "$PY" paired_analysis.py "$f" 2>&1 | head -24
    echo "--- vs no_graph ---"
    "$PY" vs_reference.py "$f" no_graph 2>&1 | head -24
    "$PY" analyze_campaign.py "$f" > "results/campaign_${TAG}_${task}.report.txt" 2>&1
done

echo "[rob] ===== ALL DONE $(date -Is) ====="
