#!/bin/bash
# ACgatfix probe: does stabilising the Stage-C entry lift (learnable scale +
# interior clip so raw BERT norms don't saturate expmap0 at the ball boundary)
# let the attention-weighted hyperbolic GNN finally match/beat Stage A?
# Compare the ACgatfix_* rows here against:
#   - ACgat_knn CoLA MCC 0.489 (unfixed Stage C, from the improved probe)
#   - A_* CoLA MCC 0.496       (hyperbolic-graph-only, the target to beat)
# CoLA is the fast probe task. Separate CSV; resumable.
cd ~/GLOT
source ~/env.sh
export WANDB_MODE=disabled
export WANDB_DIR=~/wandbtmp
export TMPDIR=~/wandbtmp
mkdir -p ~/wandbtmp
python3 run_all_experiments.py \
  --models bert-base-uncased \
  --configs ACgatfix_threshold ACgatfix_knn \
  --tasks cola \
  --seeds 42 \
  --results_csv ~/GLOT/results/hyperglot_acgatfix_probe.csv \
  >> ~/acgatfix_probe.log 2>&1
