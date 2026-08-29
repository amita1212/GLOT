#!/usr/bin/env bash
# SST-2 reduced design: the experiment the paper's Limitations calls "the single
# most valuable experiment this paper is missing".
#
#   usage: worker_sst2.sh
#
# THE CONFOUND IT ATTACKS
#   Every task in the paper is small: CoLA 8,551 / STS-B 5,749 / MRPC 3,668 /
#   RTE 2,490 training examples. Stage B's failure mode is variance
#   amplification, and variance shrinks as data grows. So the paper's FIRMEST
#   result -- "the hyperbolic readout is reliably harmful" -- may be a
#   small-data artifact rather than a property of the geometry. RTE, the
#   smallest task, is the one place B-containing arms trend positive, which is
#   exactly what that hypothesis predicts.
#   SST-2 has 67,349 examples, 7.9x CoLA. It is the cheapest task that can
#   distinguish the two explanations.
#
# WHY THE REDUCED DESIGN IS LEGITIMATE HERE
#   The full nine-arm design on SST-2 is ~170 GPU-hours. This runs four arms,
#   confirmation seeds only, replaying the CoLA-selected configurations rather
#   than tuning on SST-2. That is a WEAKER design and must be reported as one:
#   it can show whether B's harm persists at 8x the data, but it cannot rank
#   arms, because no arm was tuned on this task. Do not put these numbers in
#   the main results table.
#   Crucially the reduction is applied IDENTICALLY to the baseline and to the
#   treatment arms, so no arm gets a budget advantage -- which is the property
#   that actually matters.
#
# EXPECT ~20 GPU-HOURS: 4 arms x 15 seeds x ~1,200 s.
set -u
# Resolve the repo root from this script's own location.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/glotenv/bin/python}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs results

MODEL="bert-base-uncased"
SRC="results/campaign_wide_cola.csv"

if [ ! -f "$SRC" ]; then
    echo "MISSING $SRC -- copy it from hyperglot-l4 first (it holds the"
    echo "confirmed config of every arm; this job replays those configs)."
    exit 1
fi

echo "=== pre-warming BERT/SST-2 cache ==="
echo "SST-2 is 7.9x CoLA; check disk before and after."
df -h /home | tail -1
bash gcp/prewarm_model.sh "$MODEL" -1 sst2 > logs/sst2_prewarm.log 2>&1
if [ $? -ne 0 ]; then
    echo "PREWARM FAILED -- see logs/sst2_prewarm.log"
    tail -20 logs/sst2_prewarm.log
    exit 1
fi
df -h /home | tail -1

echo
echo "=== SST-2, arms baseline B C BC, seeds 1..15 ==="
# baseline and B answer the question; C and BC are nearly free at this point
# and tell us whether the one positive result survives more data too.
"$PY" seed_extend.py --src "$SRC" --arms baseline B C BC \
    --task sst2 --seeds 1 15 --model "$MODEL" \
    --out results/sst2_reduced.csv >> logs/sst2_reduced.log 2>&1
echo "rc=$? $(date -Is)"

echo "FINISHED -- ship back results/sst2_reduced.csv"
