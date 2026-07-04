#!/bin/bash
# Improved hyperbolic arms probe:
#   ABfix = Stage A + fixed Stage-B readout (learnable scale + feature clipping
#           + learnable curvature) -- the diagnosed fix for why plain B hurt.
#   ACgat = Stage A + attention-weighted hyperbolic GNN (hyperbolic GAT) -- fixes
#           the confound where plain hyperbolic-GCN dropped GLOT's attention.
# Fast, diverse tasks; SEPARATE CSV so it never touches the main sweep.
cd ~/GLOT
source ~/env.sh
python3 run_all_experiments.py \
  --models bert-base-uncased \
  --configs ABfix_threshold ABfix_knn ACgat_threshold ACgat_knn \
  --tasks cola mrpc stsb rte \
  --seeds 42 \
  --results_csv ~/GLOT/results/hyperglot_improved_results.csv \
  >> ~/sweep_improved.log 2>&1
