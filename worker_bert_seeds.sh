#!/usr/bin/env bash
# BERT/CoLA power job: resolve the two questions the paper leaves open at n=15.
#
#   usage: worker_bert_seeds.sh
#
# This machine answers, in order:
#   1. THE STAGE C DECOMPOSITION at n=65. Right now the +1.42 MCC gain splits
#      into geometry (+0.40), configuration (-0.01) and interaction (+1.03),
#      the total is significant and no part of it is. ~65 seeds is what 80%
#      power on the interaction costs. If this resolves, the paper's weakest
#      sentence ("real but unattributed") becomes a result.
#   2. STAGE A ON COLA at n=50. Currently +0.70 MCC, 11/4 seeds, p=0.118 --
#      the strongest of seven settings and significant in none of them. The
#      paper already names a larger CoLA seed budget as the cheapest way to
#      settle it.
#
# Both are BERT/CoLA at ~152 s a run, which is why this is ~15 GPU-hours and
# not days. It is the best evidence-per-hour available to this project.
#
# WHY EVERY SEED IS RE-RUN, INCLUDING 1..15
#   Seeds 1..15 of the existing campaign ran on a different VM at a different
#   time. Pooling them with new seeds would put a machine difference inside a
#   paired delta. Each analysis below is self-contained on THIS machine.
#   Cost of that discipline: ~2.5 extra GPU-hours. Cost of skipping it: an
#   uninterpretable result.
#
# COMMIT TO THE ANSWER. Both jobs are powered a priori. Do not re-run either
# with a different seed count after seeing which way it went.
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
    echo "MISSING $SRC -- copy it from hyperglot-l4 before starting."
    echo "It is the record of which config each arm confirmed at; without it"
    echo "this machine would have to re-tune, which answers a different"
    echo "question."
    exit 1
fi

echo "=== pre-warming BERT/CoLA cache (load-bearing, not an optimisation) ==="
bash gcp/prewarm_model.sh "$MODEL" -1 cola > logs/bs_prewarm_cola.log 2>&1
df -h /home | tail -1

echo
echo "=== 1/2  Stage C factorial, 4 cells x 65 seeds  (~11 GPU-h) ==="
"$PY" factorial_geom_full.py 65 >> logs/factorial65.log 2>&1
echo "  rc=$? $(date -Is)"

echo
echo "=== 2/2  Stage A on CoLA, baseline + A, 50 seeds  (~4 GPU-h) ==="
"$PY" seed_extend.py --src "$SRC" --arms baseline A \
    --seeds 1 50 --model "$MODEL" \
    --out results/seedext_A_cola.csv >> logs/seedext_A_cola.log 2>&1
echo "  rc=$? $(date -Is)"

echo
echo "FINISHED $(date -Is)"
echo "ship back: results/factorial_geom_full_cola_n65.csv"
echo "           results/seedext_A_cola.csv"
