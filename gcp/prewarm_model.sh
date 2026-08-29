#!/usr/bin/env bash
# Pre-warm the hidden-state cache for a given backbone on the given tasks.
#
# WHY THIS MUST RUN BEFORE ANY EXPERIMENT
# ---------------------------------------
# In upstream `train_single_classification` / `train_pair_classification` the
# order is: DataLoader(shuffle=True) -> precompute_hidden_states() -> classifier
# construction. `precompute_hidden_states` EARLY-RETURNS on a cache hit, so a
# COLD run iterates the shuffled loader and consumes `torch.randperm`, while a
# WARM run does not. Same seed, different classifier init and batch order.
# Measured on CoLA tau=0.6 seed 42: cold 40.36 +/- 0.31 vs warm 45.30 +/- 0.42,
# a 4.9-point systematic gap at ~13x the pooled seed std. Whichever arm happens
# to run FIRST is silently handicapped -- that is what invalidated the original
# "+3.4 MCC" result. So: build every cache once here with override_precompute=1,
# then every experiment runs with override_precompute=0.
#
# Usage: bash prewarm_model.sh <hf-model-name> <hidden_layer> [tasks...]
#        hidden_layer = -1 for the last layer (GLOT's default), or 1..N.
set -u
# Resolve the repo root from this script's OWN location. This is called on
# other people's machines where /home/t-amitalfasi does not exist, and a failed
# `cd` would silently leave us in the wrong directory writing caches nowhere
# useful.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HOME/glotenv/bin/python}"
MODEL="${1:-bert-base-uncased}"
LAYER="${2:--1}"
shift 2 || true
TASKS=${@:-"cola stsb rte"}

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
mkdir -p logs

for task in $TASKS; do
  if [ "$task" = "imdb" ]; then MAXLEN=512; else MAXLEN=128; fi
  echo "=== pre-warming $MODEL / $task / layer $LAYER ==="
  "$PY" hyperglot/main.py \
      --model_name_or_path="$MODEL" \
      --hidden_layer="$LAYER" \
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
      > "logs/prewarm_${MODEL//\//_}_L${LAYER}_${task}.log" 2>&1
  echo "  rc=$? ; caches now:"
  ls -d data/*batches 2>/dev/null | tail -5
done
echo "DONE. All later runs must use --override_precompute=0."
