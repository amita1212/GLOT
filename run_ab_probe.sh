#!/bin/bash
# Isolated AB-arm probe: Stage A (hyperbolic graph) + Stage B (hyperbolic
# Einstein readout), WITHOUT Stage C. Runs on a few fast, diverse tasks and
# writes to a SEPARATE results CSV so it never interferes with the main BERT
# sweep's resumability or its CSV.
cd ~/GLOT
source ~/env.sh
python3 run_all_experiments.py \
  --models bert-base-uncased \
  --configs AB_threshold AB_knn \
  --tasks cola mrpc stsb rte \
  --seeds 42 \
  --results_csv ~/GLOT/results/hyperglot_ab_results.csv \
  >> ~/sweep_ab.log 2>&1
