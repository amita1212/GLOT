#!/usr/bin/env bash
# Launch E1 (curvature sweep) in the background on the VM.
# Always launched from ~/glot so the RELATIVE ./data cache path resolves and
# every run stays WARM (see the cold-cache RNG confound).
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
nohup ~/glotenv/bin/python curvature_sweep.py \
  --tasks cola \
  --seeds 42 \
  --out /home/t-amitalfasi/glot/results/e1_curvature.csv \
  > /home/t-amitalfasi/glot/logs/e1_curvature.log 2>&1 &
echo "launched pid $!"
