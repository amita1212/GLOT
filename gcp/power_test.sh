#!/usr/bin/env bash
# Properly powered test of arm A vs the cosine baseline.
#
# WHY
# ---
# The 3-seed confirmation could not resolve the effect. Minimum detectable
# effect at n=3 was: STS-B 0.64, MRPC 0.68, CoLA 2.90, RTE 6.12. A's paired
# deltas are ~+0.1 to +0.2, so only STS-B and MRPC are even in range -- CoLA and
# RTE are too noisy to be informative at any seed count we can afford, so they
# are deliberately excluded rather than run and over-interpreted.
#
# 15 seeds shrinks the paired SE by sqrt(15/3) = 2.24x, taking the minimum
# detectable effect to roughly 0.29 (STS-B) and 0.31 (MRPC) -- enough to resolve
# a 0.2 point effect, and enough to EXCLUDE one if it is not there.
#
# Only two arms run, so this is 2 comparisons rather than 44: no multiple-
# comparison correction is needed and the test is pre-registered by construction.
# run_key dedup reuses seeds 1-3 from the existing campaign, so the marginal
# cost is 2 arms x 12 new seeds x 2 tasks = 48 runs (~1.6 h).
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

echo "[power] waiting for GPU..."
while pgrep -f 'campaign.py' >/dev/null 2>&1; do sleep 60; done

for task in stsb mrpc; do
    echo "[power] === $task L12, arms baseline+A, 15 seeds === $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model bert-base-uncased \
        --arms baseline A --trials 10 --stage both \
        --hidden_layer 12 --fine_baseline --confirm_seeds $SEEDS \
        --out "results/campaign_glue_${task}L12.csv" \
        >> "logs/campaign_glue_${task}L12.log" 2>&1
    echo "[power] $task done at $(date -Is)"
done

echo
echo "===================== PAIRED RESULT (15 seeds) ====================="
"$PY" paired_analysis.py results/campaign_glue_stsbL12.csv \
                        results/campaign_glue_mrpcL12.csv
echo "[power] ALL DONE at $(date -Is)"
