#!/usr/bin/env bash
# Pre-warm the hidden-state caches for the FIVE Table-1 cells, on the right
# backbones. gcp/prewarm_caches.sh cannot be used: it hardcodes
# bert-base-uncased, so on this machine it would warm nothing we need.
#
# WHY THIS IS NOT OPTIONAL
#   precompute_hidden_states() early-returns on a warm cache, so it advances the
#   global torch RNG only when the cache is COLD. The classifier head is built
#   AFTER that call, so a cold run and a warm run give different init and
#   different batch order for the SAME seed -- measured 40.37 vs 45.54 MCC on
#   CoLA, about six times the seed sd. Whichever arm ran first would be scored
#   under different conditions from the other eight. Warming everything first
#   removes that confound.
#
# The cache key is "{model with / -> _}_{dataset}_{split}{_Lk}_batches", so it
# is model-scoped and the three backbones cannot collide. We use the ROOT
# main.py -- the same entry point campaign.py drives -- rather than
# glot_original/main.py, so the path and contents are produced by identical
# code. batch_size 32 and max_length 128 match campaign.py exactly; a different
# batch size would lay the cache out differently.
set -uo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

# model                                  task
PAIRS="
roberta-base                             mrpc
roberta-base                             rte
TinyLlama/TinyLlama-1.1B-Chat-v1.0       cola
TinyLlama/TinyLlama-1.1B-Chat-v1.0       mrpc
TinyLlama/TinyLlama-1.1B-Chat-v1.0       rte
"

mkdir -p logs data
echo "$PAIRS" | while read -r MODEL TASK; do
    [ -z "${MODEL:-}" ] && continue
    TAG=$(echo "$MODEL" | tr '/' '_')
    echo "=================================================================="
    echo "PRE-WARM  model=$MODEL  task=$TASK   $(date -Is)"
    echo "=================================================================="
    "$PY" main.py \
        --model_name_or_path="$MODEL" \
        --task="$TASK" \
        --max_length=128 \
        --adaptive_length=0 \
        --epochs=1 \
        --batch_size=32 \
        --eval_batch_size=64 \
        --lr=2e-4 \
        --weight_decay=0.0 \
        --seed=42 \
        --verbose=0 \
        --pooling_method=glot \
        --gnn_type=gat \
        --scorer_hidden=128 \
        --gat_hidden_dim=128 \
        --num_layers=2 \
        --jk_mode=cat \
        --graph_adj=threshold \
        --tau=0.6 \
        --proj_dim=256 \
        --precompute_hidden_states=1 \
        --override_precompute=0 \
        --finetune_backbone=0 \
        > "logs/prewarm_${TAG}_${TASK}.log" 2>&1
    if [ $? -eq 0 ]; then echo "  OK"; else echo "  FAILED -- see logs/prewarm_${TAG}_${TASK}.log"; fi
    ls -d data/*"${TAG}"*"${TASK}"*batches 2>/dev/null || true
done

echo
echo "===== cache dirs present ====="
du -sh data/*_batches 2>/dev/null | sort -k2
