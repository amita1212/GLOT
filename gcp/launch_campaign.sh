#!/usr/bin/env bash
# Launch a HyperGLOT campaign in the background.
#   bash launch_campaign.sh <target> <task-or-ratio> <trials> [extra args...]
# Always run from ~/glot so the RELATIVE ./data cache resolves and every GLUE
# run stays on a WARM cache.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

TARGET="${1:-stress}"
TAG="${2:-default}"
shift 2 || true

nohup ~/glotenv/bin/python campaign.py --target "$TARGET" "$@" \
  --out /home/t-amitalfasi/glot/results/campaign_${TARGET}_${TAG}.csv \
  > /home/t-amitalfasi/glot/logs/campaign_${TARGET}_${TAG}.log 2>&1 &
echo "launched pid $! -> logs/campaign_${TARGET}_${TAG}.log"
