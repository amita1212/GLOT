#!/usr/bin/env bash
# Re-run the HyperGLOT A/B/C ablation under the CORRECTED protocol.
#
# What was wrong before (July 2026, results/hyperglot_all_results.csv):
#   For every single task the `baseline` arm was the FIRST row, so it built the
#   hidden-state cache (COLD) while every hyperbolic arm ran afterwards (WARM).
#   Cold runs score ~4.9 points LOWER on CoLA (40.36 vs 45.30, pooled std ~0.37),
#   so the baseline was systematically handicapped in all nine tasks. The
#   confound aligned perfectly with the treatment variable.
#
# What is fixed here:
#   1. Every cache is pre-warmed BEFORE any arm runs  -> all arms warm.
#   2. Hyperparameters follow the paper (2 epochs, hidden 128), not the README.
#   3. tau is tuned PER TASK (paper Table 8) so the cosine baseline is not
#      detuned relative to the hyperbolic arms' rho / knn_k.
#   4. Multiple seeds, so we can report mean +/- std instead of a single run.
#
# IMPORTANT: launched from ~/glot so that main.py's "./data/" cache path
# resolves to the same pre-warmed directory.
set -euo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

TASKS="${TASKS:-cola stsb rte}"
SEEDS="${SEEDS:-42 1 2}"
CONFIGS="${CONFIGS:-baseline A_threshold A_knn AB_threshold AC_threshold ACgat_knn}"

echo "########## STEP 1: pre-warm all caches ##########"
bash prewarm_caches.sh $TASKS

echo
echo "########## STEP 2: ablation, all arms warm ##########"
echo "tasks   = $TASKS"
echo "seeds   = $SEEDS"
echo "configs = $CONFIGS"
echo

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

"$PY" hyperglot/run_all_experiments.py \
    --models bert-base-uncased \
    --tasks $TASKS \
    --configs $CONFIGS \
    --seeds $SEEDS \
    2>&1 | tee logs_ablation_warm.txt | grep -E "^\[|DONE|FAIL|Traceback|Error"

echo
echo "########## DONE ##########"
echo "results -> hyperglot/results/hyperglot_all_results.csv"
