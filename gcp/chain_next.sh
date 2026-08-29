#!/usr/bin/env bash
# Post-campaign queue, in priority order. Waits for the GPU, then:
#
#   1. EQUAL-BUDGET BASELINE. The layer-8 campaign gave every hyperbolic arm 10
#      draws but the cosine baseline only 5, because the baseline's search space
#      IS 5 points. Best-of-10 vs best-of-5 favours the bigger space even when
#      both are identical in truth (~0.64 MCC at the measured sd of 1.7), which
#      is most of A's observed +1.58. --fine_baseline gives the baseline a
#      10-point density grid so the comparison becomes best-of-10 vs best-of-10.
#      run_key dedup means the 5 already-computed points are reused, so this
#      costs ~5 runs, not 10.
#
#   2. MODERNBERT LAYER PROBE. Transfer test for the layer-selection finding on
#      a second backbone. Not run because ModernBERT is more tree-like -- the
#      angular-delta screen showed that was a massive-activation artefact.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

echo "[queue] waiting for campaign.py to exit..."
while pgrep -f 'campaign.py' >/dev/null 2>&1; do
    sleep 60
done
echo "[queue] GPU free at $(date -Is)"

# Full confirmation table for the layer-8 campaign, saved before anything else
# can crash.
"$PY" analyze_campaign.py results/campaign_glue_colaL8.csv \
    > results/campaign_glue_colaL8.report.txt 2>&1
echo "[queue] wrote results/campaign_glue_colaL8.report.txt"

# --- 1. equal-budget baseline ---------------------------------------------
echo "[queue] equal-budget baseline at $(date -Is)"
"$PY" campaign.py --target glue --task cola \
    --arms baseline no_graph --trials 10 --stage both \
    --hidden_layer 8 --fine_baseline --confirm_seeds 1 2 3 \
    --out results/campaign_glue_colaL8.csv \
    >> logs/campaign_glue_colaL8.log 2>&1
echo "[queue] equal-budget baseline done at $(date -Is)"

"$PY" show_rows.py results/campaign_glue_colaL8.csv | tail -20

# --- 2. ModernBERT layer probe --------------------------------------------
echo "[queue] ModernBERT layer probe at $(date -Is)"
bash layer_probe_any.sh cola "4 8 12 16 20 22" answerdotai/ModernBERT-base \
    > logs/layer_probe_modernbert.log 2>&1
tail -20 logs/layer_probe_modernbert.log

echo "[queue] DONE at $(date -Is)"
