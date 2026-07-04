#!/bin/bash
# Small ACgat hyperparameter test on CoLA (fast, clean MCC signal).
# Base = ACgat_knn (poincare graph + attention hyperbolic GNN, c=1, 2 layers,
# gat_hidden=256). Each run varies ONE knob so we can see what moves MCC.
# Writes to a separate CSV with distinct --arm labels.
cd ~/GLOT
source ~/env.sh

CSV=~/GLOT/results/hyperglot_acgat_hp.csv

run() {
  arm="$1"; shift
  python3 main.py \
    --model_name_or_path=bert-base-uncased --decoder_cls_last_token=0 \
    --task=cola --max_length=128 --seed=42 --verbose=1 \
    --pooling_method=glot --graph_adj=knn --graph_metric=poincare \
    --hyperbolic_gnn=1 --hyperbolic_readout=0 --hyp_gnn_type=gat \
    --readout_clip=0.0 --readout_scale=0 \
    --arm="$arm" --results_csv="$CSV" --run_tag=hyperglot \
    --epochs=3 --batch_size=32 --eval_batch_size=64 --lr=2e-4 --weight_decay=0.0 \
    --gnn_type=gat --scorer_hidden=128 --gat_hidden_dim=256 --jk_mode=cat \
    --tau=0.8 --rho=1.0 --knn_k=8 --proj_dim=256 \
    --precompute_hidden_states=1 --finetune_backbone=0 --adaptive_length=0 \
    "$@"
}

# 0) reference: exact ACgat_knn defaults (c=1, 2 layers)
run ACgathp_base       --num_layers=2 --curvature=1.0 --learnable_curvature=0
# 1) learnable curvature
run ACgathp_lc         --num_layers=2 --curvature=1.0 --learnable_curvature=1
# 2) smaller curvature (less boundary saturation on the raw-feature lift)
run ACgathp_c05        --num_layers=2 --curvature=0.5 --learnable_curvature=0
# 3) larger curvature
run ACgathp_c20        --num_layers=2 --curvature=2.0 --learnable_curvature=0
# 4) deeper (3 layers)
run ACgathp_L3         --num_layers=3 --curvature=1.0 --learnable_curvature=0
# 5) smaller curvature + learnable (combine the two most likely wins)
run ACgathp_c05_lc     --num_layers=2 --curvature=0.5 --learnable_curvature=1

echo "[acgat_hp] done $(date)"
