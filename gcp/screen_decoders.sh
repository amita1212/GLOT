#!/usr/bin/env bash
# Geometry screen for the paper's DECODER backbones, before any training.
#
# WHY THIS RUNS FIRST
#   Twice now an absolute cosine threshold has silently produced a degenerate
#   graph: ModernBERT (density 0.9957 at tau=0.6) and RoBERTa (0.992, and >=0.99
#   at EVERY tau in the paper's {0.1,0.3,0.6} grid). Decoder-only models are
#   reported to be even more anisotropic than encoders, so the same failure is
#   likely -- and launching a full 12-arm sweep on a complete graph would burn
#   many GPU-hours measuring nothing.
#
#   Also screens token NORMS, because massive activations decide whether the
#   scale knob must use the mean or the median (ModernBERT L12 mean/median 7.66).
#
# Paper backbones, sizes from its Figure axis "110M 360M 1.1B 3B 7B":
#   SmolLM2  = HuggingFaceTB/SmolLM2-360M   (360M)
#   TinyLlama= TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)
# Both are frozen and only ever forward-passed, so an L4 is ample.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
MODELS="HuggingFaceTB/SmolLM2-360M TinyLlama/TinyLlama-1.1B-Chat-v1.0"

echo "############ TOKEN NORMS (decides rms vs median scaling) ############"
CUDA_VISIBLE_DEVICES= "$PY" norm_stats.py --models $MODELS --layers -1 2>&1 \
    | grep -vE 'Warning|warn|dtype|^$'

echo
echo "############ COSINE / DENSITY AT THE PAPER'S tau GRID ############"
for t in 0.1 0.3 0.6; do
    echo "---------- tau = $t ----------"
    CUDA_VISIBLE_DEVICES= "$PY" cosine_stats.py --models $MODELS --layers -1 \
        --tau "$t" 2>&1 | grep -E '^===|^ +[0-9]+ '
done
