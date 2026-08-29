#!/usr/bin/env bash
# Test whether the ModernBERT collapse is a feature-SCALE bug, not a model bug.
#
# EVIDENCE FOR THE HYPOTHESIS
#   ModernBERT scored 17-20 MCC on CoLA at EVERY probed layer (4/8/12/16/20/22)
#   while BERT scores 43-48. A model that is uniformly bad at every depth is a
#   broken pipeline, not a weak encoder. Measured mean||x||: BERT ~15,
#   ModernBERT 874 at L12 and 2614 at L16 (massive activations / attention
#   sinks). GLOT feeds raw hidden states into the GAT at a fixed lr of 2e-4.
#
# THE TEST
#   Re-run ModernBERT with --input_scale_norm=rms, which divides by the batch
#   mean token norm so mean||x||=1 for any backbone. Pure scale change; relative
#   norms (and therefore the Stage A depth signal) are preserved.
#
#   Prediction if the hypothesis is right: scores jump from ~18 into the 30s-40s.
#   Prediction if wrong: scores stay ~18, and ModernBERT genuinely does not suit
#   a frozen-backbone graph pooler -- also a publishable observation.
#
#   CONTROL: BERT is re-run with rms too. Since BERT's norms are already O(15)
#   and low-variance (cv 0.07), rms should be roughly neutral there. If BERT
#   moves a lot, the knob is doing something other than fixing scale and the
#   whole interpretation is suspect.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

run () {  # model layer scale_norm
    local model="$1" layer="$2" sn="$3"
    local slug="${model//\//_}"
    echo "--- $model L$layer input_scale_norm=$sn ---"
    "$PY" hyperglot/main.py \
        --model_name_or_path="$model" --hidden_layer="$layer" \
        --decoder_cls_last_token=0 --task=cola --max_length=128 \
        --adaptive_length=0 --epochs=2 --batch_size=32 --eval_batch_size=64 \
        --lr=2e-4 --weight_decay=0.0 --seed=42 --verbose=0 \
        --pooling_method=glot --gnn_type=gat --scorer_hidden=128 \
        --gat_hidden_dim=128 --num_layers=2 --jk_mode=cat \
        --graph_adj=threshold --tau=0.6 --proj_dim=256 \
        --input_scale_norm="$sn" \
        --precompute_hidden_states=1 --override_precompute=0 \
        --finetune_backbone=0 2>&1 | grep -E 'mcc|MCC|best|Best' | tail -3
}

echo "########## CONTROL: BERT (norms already ~15, rms should be ~neutral) #####"
run bert-base-uncased 12 none
run bert-base-uncased 12 rms

echo
echo "########## TEST: ModernBERT (norms 874 at L12) ##########################"
for L in 4 12 22; do
    run answerdotai/ModernBERT-base "$L" none
    run answerdotai/ModernBERT-base "$L" rms
done
echo "DONE at $(date -Is)"
