#!/bin/bash
# Corrected Stage-B test: ABfix2 = Stage A + fixed readout with a PROPERLY
# interior clip (0.7 -> ball radius ~0.60 at c=1) instead of the too-loose
# clip=2.0 (radius 0.964, still saturated). This is the real test of whether
# the boundary-saturation fix rehabilitates Stage B. Separate CSV.
cd ~/GLOT
source ~/env.sh
python3 run_all_experiments.py \
  --models bert-base-uncased \
  --configs ABfix2_threshold ABfix2_knn \
  --tasks cola mrpc stsb rte \
  --seeds 42 \
  --results_csv ~/GLOT/results/hyperglot_abfix2_results.csv \
  >> ~/sweep_abfix2.log 2>&1
