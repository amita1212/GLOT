#!/usr/bin/env bash
# Full multi-task queue: transfer-test the layer choice, then run the 12-arm
# campaign on every task at whichever layer that task actually prefers.
#
# WHY THE PROBE COMES FIRST
# -------------------------
# Layer 8 was selected by searching 6 layers on CoLA's dev set (48.24 at L8 vs
# 43.62 at L12). That search is exactly the kind of thing that overfits a single
# dev set, and GLOT never spent that budget, so part of the gain is selection
# bias rather than a property of BERT. Running the expensive 12-arm campaign at
# L8 on STS-B/RTE/MRPC *without* checking would bake that bias into every other
# task. So: probe baseline-only at L8 and L12 first (~30 min/task), then commit
# the ~6 GPU-hours to whichever layer wins.
#
# Interpreting the probe:
#   L8 > L12 on most tasks  -> layer selection is real and publishable
#   L8 wins only on CoLA    -> it was CoLA dev overfitting; campaign at L12
#
# Everything is resumable: campaign.py dedups on run_key, so a spot preemption
# costs only the run that was in flight.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

TASKS="${1:-stsb rte mrpc}"
MODEL=bert-base-uncased

echo "[queue] waiting for any running campaign to exit..."
while pgrep -f 'campaign.py' >/dev/null 2>&1; do
    sleep 60
done
echo "[queue] GPU free at $(date -Is)"

# --- 0. finish the CoLA question: equal-budget baseline --------------------
# The L8 campaign gave hyperbolic arms 10 draws and the cosine baseline only 5,
# because the baseline's space IS 5 points. --fine_baseline widens it to 10 so
# best-of-10 is compared with best-of-10.
echo "[queue] === equal-budget baseline, cola L8 === $(date -Is)"
"$PY" campaign.py --target glue --task cola --model "$MODEL" \
    --arms baseline no_graph --trials 10 --stage both \
    --hidden_layer 8 --fine_baseline --confirm_seeds 1 2 3 \
    --out results/campaign_glue_colaL8.csv \
    >> logs/campaign_glue_colaL8.log 2>&1
"$PY" show_rows.py results/campaign_glue_colaL8.csv | tail -15

# --- 1. layer transfer probe on every new task -----------------------------
for task in $TASKS; do
    echo "[queue] === layer probe: $task === $(date -Is)"
    bash layer_probe_any.sh "$task" "8 12" "$MODEL" \
        2>&1 | tee "logs/layer_probe_${task}.log"
done

# --- 2. pick the layer per task, then run the full 12-arm campaign ---------
for task in $TASKS; do
    CSV="results/layer_probe_${MODEL}_${task}.csv"
    LAYER=$("$PY" pick_layer.py "$CSV" 2>/dev/null || echo 12)
    echo "[queue] === campaign: $task at layer $LAYER === $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model "$MODEL" \
        --trials 10 --stage both --hidden_layer "$LAYER" \
        --fine_baseline --confirm_seeds 1 2 3 \
        --out "results/campaign_glue_${task}L${LAYER}.csv" \
        >> "logs/campaign_glue_${task}L${LAYER}.log" 2>&1
    "$PY" analyze_campaign.py "results/campaign_glue_${task}L${LAYER}.csv" \
        > "results/campaign_glue_${task}L${LAYER}.report.txt" 2>&1
    echo "[queue] wrote results/campaign_glue_${task}L${LAYER}.report.txt"
done

# --- 3. ModernBERT transfer (lowest priority) ------------------------------
echo "[queue] === ModernBERT layer probe === $(date -Is)"
bash layer_probe_any.sh cola "4 8 12 16 20 22" answerdotai/ModernBERT-base \
    > logs/layer_probe_modernbert.log 2>&1
tail -20 logs/layer_probe_modernbert.log

echo "[queue] ALL DONE at $(date -Is)"
