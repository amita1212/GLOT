#!/bin/bash
# ACgat multi-task sweep: does the attention-weighted hyperbolic GNN (Stage A+C
# with attention) ever beat hyperbolic-graph-only (Stage A) beyond CoLA?
# Runs ACgat (threshold+knn) on the tasks NOT already covered by the improved
# probe (which does cola/mrpc/stsb/rte). Compare vs the A_* rows already in
# hyperglot_all_results.csv. Separate CSV; resumable.
cd ~/GLOT
source ~/env.sh
export WANDB_MODE=disabled
export WANDB_DIR=~/wandbtmp
export TMPDIR=~/wandbtmp
mkdir -p ~/wandbtmp
python3 run_all_experiments.py \
  --models bert-base-uncased \
  --configs ACgat_threshold ACgat_knn \
  --tasks sst2 qqp mnli qnli wnli \
  --seeds 42 \
  --results_csv ~/GLOT/results/hyperglot_acgat_sweep.csv \
  >> ~/sweep_acgat.log 2>&1
