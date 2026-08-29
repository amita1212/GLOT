#!/usr/bin/env bash
# Pre-warm every hidden-state cache BEFORE running any experiment.
#
# WHY: upstream precompute_hidden_states() early-returns when the cache exists,
# so it only advances the global torch RNG on a COLD cache. Because the
# classifier head is constructed *after* the precompute call, a cold run and a
# warm run produce different weight init and different batch order for the SAME
# seed (measured: 40.37 vs 45.54 MCC on CoLA tau=0.6).
#
# Any experiment sweep therefore has a systematic confound: whichever arm runs
# FIRST builds the cache and is scored under different conditions from all the
# others. Pre-warming makes every subsequent run identical in this respect.
#
# Usage: bash prewarm_caches.sh cola stsb rte [more tasks...]
set -euo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
    TASKS=(cola stsb rte)
fi

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

for task in "${TASKS[@]}"; do
    if [ "$task" = "imdb" ]; then MAXLEN=512; else MAXLEN=128; fi
    echo "=================================================================="
    echo "PRE-WARMING cache for task=$task (override_precompute=1)"
    echo "=================================================================="
    "$PY" glot_original/main.py \
        --model_name_or_path=bert-base-uncased \
        --decoder_cls_last_token=0 \
        --task="$task" \
        --max_length=$MAXLEN \
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
        --override_precompute=1 \
        --finetune_backbone=0 \
        > "prewarm_${task}.log" 2>&1
    echo "  done -> cache dirs now:"
    ls -d data/*"${task}"*batches 2>/dev/null || ls -d data/*sts*batches 2>/dev/null || true
done

echo
echo "=================================================================="
echo "ALL CACHES WARM. Every experiment from here on must use"
echo "--override_precompute=0 so all runs are mutually comparable."
echo "=================================================================="
ls -d data/*batches
