#!/usr/bin/env bash
# Launch E2 (Stage A geometry search) in the background.
# Must run from ~/glot so the RELATIVE ./data cache path resolves and every run
# stays WARM (cold caches consume torch.randperm and shift results ~5 MCC).
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
STAGE="${1:-1}"
TASKS="${2:-cola}"
nohup ~/glotenv/bin/python geometry_sweep.py \
  --stage "$STAGE" \
  --tasks $TASKS \
  --seeds 42 \
  --out /home/t-amitalfasi/glot/results/e2_geometry.csv \
  > /home/t-amitalfasi/glot/logs/e2_geometry.log 2>&1 &
echo "launched pid $! (stage=$STAGE tasks=$TASKS)"
