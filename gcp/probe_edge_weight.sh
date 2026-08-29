#!/usr/bin/env bash
# Decisive test: hold EVERYTHING fixed (seed, density, features) and vary only
# `edge_weight_mode`. If the three scores are identical, the GAT is ignoring
# edge_attr and every "edge weighting" result in this project is meaningless.
#
# Standing rule: bit-identical scores across a knob's settings == the knob is
# disconnected. That signature already caught the empty-graph bug and the
# curvature=1.0 saturation bug.
set -u
cd /home/t-amitalfasi/glot
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
PY=~/glotenv/bin/python

for MODE in binary depth hyp_z; do
  echo "=== edge_weight_mode=$MODE ==="
  "$PY" hyperglot/diagnostic_stress_test.py \
      --model_name_or_path=bert-base-uncased \
      --seed=42 --verbose=1 \
      --num_train_samples=2000 --num_eval_samples=1000 --max_length=128 \
      --distractor_ratio=0.95 --relational_distance=60 --signal_position=random \
      --epochs=3 --batch_size=32 --eval_batch_size=64 --lr=1e-4 \
      --pooling_method=glot --jk_mode=cat --gat_hidden_dim=128 \
      --num_layers=2 --scorer_hidden=128 \
      --graph_metric=cosine --tau_quantile=0.025 \
      --feature_mode=center_unit --graph_curvature=1.0 \
      --edge_weight_mode="$MODE" --edge_temp=0.5 \
      --results_csv=/tmp/ewm_probe.csv 2>&1 \
    | grep -E "Eval Accuracy|\[graph\]|Error|Traceback"
done
