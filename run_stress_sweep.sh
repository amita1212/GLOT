#!/bin/bash
# Negation "needle-in-a-haystack" stress diagnostic.
# Sweeps distractor ratio [0.2, 0.5, 0.8, 0.9] for each arm and records accuracy.
# This is the qualitative test of WHY hyperbolic graph construction (Stage A)
# helps: can the pooler still isolate the negation "needle" as the haystack of
# distractor tokens grows?
#
# Arms (as requested): the pure Stage-A arm, the fixed Stage-C arm (ACgatfix =
# attention-weighted hyperbolic GNN + interior input scale/clip), and the fixed
# Stage-B arm (ABfix2 = gyro-midpoint readout with interior clip + learnable
# scale/curvature). Baseline (Euclidean cosine graph + Euclidean readout) is the
# control the hyperbolic arms are measured against. Both threshold & knn graphs.
# Separate CSV; resumable (keyed on model/arm/distractor/seed).
cd ~/GLOT
source ~/env.sh
export WANDB_MODE=disabled
export WANDB_DIR=~/wandbtmp
export TMPDIR=~/wandbtmp
mkdir -p ~/wandbtmp
python3 run_all_experiments.py \
  --stress_only \
  --models bert-base-uncased \
  --configs baseline A_threshold A_knn ACgatfix_threshold ACgatfix_knn ABfix2_threshold ABfix2_knn \
  --seeds 42 \
  --stress_csv ~/GLOT/results/hyperglot_stress_results.csv \
  >> ~/stress_sweep.log 2>&1
