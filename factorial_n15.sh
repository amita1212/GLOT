#!/usr/bin/env bash
# The factorial ran at n=5, where the exact sign test floors at p=0.0625 and so
# CANNOT satisfy the paper's own "both tests must agree" criterion. The median
# effect is directionally positive on both ModernBERT layers but untestable at
# that n. Extend to 15 seeds, matching every other campaign in the paper.
#
# RoBERTa is excluded: its effects are ~0.1 MCC with tight variance, already
# clearly null, and it would cost a third of the budget for nothing.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p logs

BKS=(bert_final mbert_L12 mbert_final)
for i in 0 1 2; do
    nohup "$PY" -u factorial_scale.py --task cola \
        --backbones "${BKS[$i]}" \
        --seeds 6 7 8 9 10 11 12 13 14 15 \
        --out "results/factorial_scale_n15_w${i}.csv" \
        > "logs/factorial_n15_w${i}.log" 2>&1 &
    echo "worker $i (${BKS[$i]}) pid=$!"
done
wait
echo "ALL FACTORIAL-N15 DONE $(date -Is)"
