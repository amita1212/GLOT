#!/usr/bin/env bash
# Attack the three FLAT tasks (CoLA, RTE, MRPC) with STRUCTURAL arms rather than
# more geometry.
#
# THE HYPOTHESIS
#   Stage A wins on STS-B (+0.223 Spearman, 15/15 seeds) and is flat on CoLA,
#   RTE and MRPC. STS-B is the one task whose objective IS geometric -- Spearman
#   on cosine similarity between pooled vectors -- so reshaping the embedding
#   shows up directly. The others are classification, where a linear head can
#   absorb a differently-shaped embedding.
#
#   So for the flat tasks, change what the graph CARRIES, not its geometry.
#   Every existing arm leaves the token graph PERMUTATION-INVARIANT: edges come
#   only from feature similarity, so permuting tokens permutes the graph and the
#   pooled embedding is unchanged. CoLA scores linguistic ACCEPTABILITY, which is
#   very largely word order -- a structure that cannot tell "the cat sat" from
#   "sat the cat" cannot represent what the task measures.
#
# THE ARMS
#   POS       similarity edges + sequence window w (order-aware topology)
#   POS_ONLY  ONLY the sequence window -- the ablation that attributes any POS
#             gain to word order rather than to simply having more edges
#   A_POS     Stage A hyperbolic graph + sequence window
#   KNN       fixed DEGREE instead of a global threshold; never swept, and
#             robust to the per-sentence anisotropy that mis-calibrates tau
#
# PROTOCOL (lessons already paid for)
#   - 15 confirmation seeds from the start. n=3 gave a minimum detectable effect
#     of 2.90 on CoLA and produced one outright false positive on MRPC.
#   - --fine_baseline so the cosine arm gets a genuinely equal 10-trial budget.
#   - paired_analysis.py at the end: arms share seeds, so per-seed differencing
#     cancels the dominant variance component (5x tighter SE on STS-B).
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

ARMS="baseline POS POS_ONLY A_POS KNN"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
TASKS="${1:-cola rte mrpc}"

echo "[struct] waiting for the GPU..."
while pgrep -f 'test_scale_fix.sh|fix_modernbert.sh|campaign.py' >/dev/null 2>&1; do
    sleep 60
done
echo "[struct] GPU free at $(date -Is)"

for task in $TASKS; do
    echo "[struct] ===== $task L12, arms: $ARMS ===== $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model bert-base-uncased \
        --arms $ARMS --trials 10 --stage both \
        --hidden_layer 12 --fine_baseline --confirm_seeds $SEEDS \
        --out "results/campaign_struct_${task}.csv" \
        >> "logs/campaign_struct_${task}.log" 2>&1
    echo "[struct] --- paired result: $task ---"
    "$PY" paired_analysis.py "results/campaign_struct_${task}.csv" 2>&1 | head -14
done

echo
echo "===================== ALL STRUCTURAL RESULTS ====================="
"$PY" paired_analysis.py results/campaign_struct_*.csv
echo "[struct] DONE at $(date -Is)"
