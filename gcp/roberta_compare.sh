#!/usr/bin/env bash
# Second backbone for the paper comparison: RoBERTa, the other ENCODER in
# GLOT's Table 1. (The remaining four -- SmolLM2, TinyLlama, LLaMA-3B,
# Mistral-7B -- are decoders and do not fit on a single L4.)
#
# WHY THIS IS MORE THAN A REPLICATION
#   RoBERTa's token cosines sit at p10=0.701 / p50=0.838, versus BERT's
#   0.081 / 0.401. GLOT thresholds edges at an ABSOLUTE cosine tau, and its
#   published search space is {0.1, 0.3, 0.6} (Table 6, "searched consistently
#   across all backbone models"). Measured density at layer 12:
#       tau      0.1     0.3     0.6
#       BERT    0.850   0.638   0.149
#       RoBERTa 1.000   1.000   0.992
#   No setting in the published grid gives RoBERTa a sparse graph, so all of its
#   Table 1 numbers -- including the paper's best CoLA result, 56.08 -- come
#   from a near-complete token graph.
#
# ARMS
#   paper_tau     tau=0.6 exactly as published (RoBERTa: density 0.992)
#   density_fix   same cosine metric, threshold chosen by QUANTILE so sparsity
#                 matches BERT's. Isolates calibration from method.
#   A             our Stage A hyperbolic graph, also density-matched
#   no_graph      the control that must be beaten before anything is a "win"
#
# 15 confirmation seeds from the start: n=3 gave a minimum detectable effect of
# 2.90 on CoLA and produced an outright false positive on MRPC.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

MODEL=roberta-base
ARMS="paper_tau density_fix A no_graph"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
TASKS="${1:-cola stsb mrpc rte}"

echo "[roberta] waiting for the GPU..."
while pgrep -f 'structural_arms.sh|test_scale_fix.sh|fix_modernbert.sh|campaign.py' \
      >/dev/null 2>&1; do
    sleep 60
done
echo "[roberta] GPU free at $(date -Is)"

# Hidden-state caches must be built ONCE up front. A cold cache consumes
# torch.randperm via the shuffled DataLoader and shifts classifier init, worth
# ~5 MCC on CoLA -- whichever arm ran first would otherwise be handicapped.
for task in $TASKS; do
    echo "[roberta] pre-warming cache: $task"
    bash prewarm_model.sh "$MODEL" -1 "$task" > /dev/null 2>&1 \
        || echo "[roberta] PREWARM FAILED for $task"
done

for task in $TASKS; do
    echo "[roberta] ===== $task ===== $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model "$MODEL" \
        --arms $ARMS --trials 6 --stage both \
        --confirm_seeds $SEEDS \
        --out "results/campaign_roberta_${task}.csv" \
        >> "logs/campaign_roberta_${task}.log" 2>&1
    "$PY" paired_analysis.py "results/campaign_roberta_${task}.csv" 2>&1 | head -12
done

echo
echo "===================== PAPER COMPARISON ====================="
"$PY" compare_to_paper.py --results results/campaign_roberta_*.csv \
                                    results/campaign_glue_stsbL12.csv \
                                    results/campaign_glue_rteL12.csv \
                                    results/campaign_glue_mrpcL12.csv
echo "[roberta] DONE at $(date -Is)"
