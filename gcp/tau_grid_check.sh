#!/usr/bin/env bash
# Edge density produced by EVERY value in the GLOT paper's tau search space
# (Table 6: tau in {0.1, 0.3, 0.6}, "searched consistently across all backbone
# models and datasets") for the paper's two encoder backbones.
#
# WHY THIS MATTERS
#   GLOT thresholds edges at an ABSOLUTE cosine tau. RoBERTa's token cosines are
#   far higher than BERT's (p50 0.838 vs 0.401; p10 0.701). If RoBERTa's 10th
#   percentile already exceeds the LARGEST tau in the grid, then no setting in
#   the published search space can produce a sparse graph for RoBERTa -- every
#   one of its reported numbers would come from a near-complete token graph, in
#   which case the "relational structure" is not doing what the paper describes.
#
# Uses a script file because `for t in ...; $t` gets mangled when passed through
# `gcloud compute ssh --command` on Windows (an inline attempt reported density
# 0.0000 for BERT at tau=0.6, which is known to be 0.149).
set -u
cd /home/t-amitalfasi/glot
for t in 0.1 0.3 0.6; do
    echo "########## tau = $t ##########"
    CUDA_VISIBLE_DEVICES= ~/glotenv/bin/python cosine_stats.py \
        --models bert-base-uncased roberta-base --layers 12 --tau "$t" 2>&1 \
        | grep -E '^===|^ +12'
done
