#!/usr/bin/env bash
# Fix ModernBERT: separate the DENSITY bug from the SCALE bug.
#
# MEASURED DIAGNOSIS
#   density: GLOT's absolute tau=0.6 is calibrated on BERT (token cosines
#     0.25-0.40, density 0.02-0.15). ModernBERT's cosines sit at 0.75-0.79, so
#     the SAME tau gives density 0.9957 at L4/L8 -- a near-complete graph with
#     no structural information. This is the dominant bug and explains why even
#     ModernBERT's well-conditioned L4 scored only 0.13.
#   scale:   token-norm mean/median is 1.01 at L4 (clean) but 7.66 at L12 and
#     9.87 at L22, with max/median ~100. Those deep layers additionally need
#     median-based rescaling; L4 does not.
#
# The two factors are tested SEPARATELY so their contributions are attributable
# rather than confounded -- which is the mistake that produced the earlier
# "ModernBERT confirms the negative result" reading.
#
# BERT control runs the same density-matched setting: tau_quantile=0.05 is close
# to BERT's natural density at tau=0.6 (0.023-0.046), so BERT should be roughly
# unchanged. If BERT moves a lot, the knob is doing something unintended.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

run () {  # model layer  extra-args...
    local model="$1" layer="$2"; shift 2
    echo "--- $model L$layer $* ---"
    "$PY" hyperglot/main.py \
        --model_name_or_path="$model" --hidden_layer="$layer" \
        --decoder_cls_last_token=0 --task=cola --max_length=128 \
        --adaptive_length=0 --epochs=2 --batch_size=32 --eval_batch_size=64 \
        --lr=2e-4 --weight_decay=0.0 --seed=42 --verbose=0 \
        --pooling_method=glot --gnn_type=gat --scorer_hidden=128 \
        --gat_hidden_dim=128 --num_layers=2 --jk_mode=cat \
        --graph_adj=threshold --proj_dim=256 \
        --precompute_hidden_states=1 --override_precompute=0 \
        --finetune_backbone=0 "$@" 2>&1 | grep -E 'RESULT_JSON|\[graph\]' | tail -2
}

echo "########## CONTROL: BERT ##########"
run bert-base-uncased 12 --tau=0.6
run bert-base-uncased 12 --tau_quantile=0.05

echo
echo "########## ModernBERT L4 (norms CLEAN, density BROKEN 0.9957) ##########"
run answerdotai/ModernBERT-base 4 --tau=0.6
run answerdotai/ModernBERT-base 4 --tau_quantile=0.05
run answerdotai/ModernBERT-base 4 --tau_quantile=0.10

echo
echo "########## ModernBERT L12 (norms SKEWED 7.66, density 0.838) ##########"
run answerdotai/ModernBERT-base 12 --tau=0.6
run answerdotai/ModernBERT-base 12 --tau_quantile=0.05
run answerdotai/ModernBERT-base 12 --tau_quantile=0.05 --input_scale_norm=median
run answerdotai/ModernBERT-base 12 --tau_quantile=0.05 --input_scale_norm=rms

echo "DONE at $(date -Is)"
