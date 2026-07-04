#!/bin/bash
# ACgat multi-task sweep: does the attention-weighted hyperbolic GNN (Stage A+C
# with attention) ever beat hyperbolic-graph-only (Stage A) beyond CoLA?
# Runs ACgat (threshold+knn) on all GLUE + IMDB. Compare vs the A_* rows already
# in hyperglot_all_results.csv. Separate CSV; resumable.
cd ~/GLOT
source ~/env.sh
python3 run_all_experiments.py \
  --models bert-base-uncased \
  --configs ACgat_threshold ACgat_knn \
  --tasks cola sst2 stsb mrpc qqp mnli qnli rte wnli imdb \
  --seeds 42 \
  --results_csv ~/GLOT/results/hyperglot_acgat_sweep.csv \
  >> ~/sweep_acgat.log 2>&1
