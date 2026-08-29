#!/usr/bin/env bash
# Wait for the running campaign to finish, then run the ModernBERT pipeline.
#
# WHY CHAINED: the L4 has one GPU. The layer-8 CoLA campaign owns it for ~3h.
# Rather than idle the GPU between jobs, block on the campaign PID and start
# the ModernBERT layer probe the instant it exits.
#
# ModernBERT is run as a TRANSFER TEST of the layer-selection finding on a
# second, stronger backbone. It is NOT run because it is more tree-like: the
# angular delta screen showed its apparent tree-likeness is a massive-activation
# artefact (max||x||/median||x|| up to 156x at L16).
set -u
cd /home/t-amitalfasi/glot

MODEL=answerdotai/ModernBERT-base
LAYERS="${1:-4 8 12 16 20 22}"
TASK="${2:-cola}"

echo "[chain] waiting for campaign.py to exit..."
while pgrep -f 'campaign.py' >/dev/null 2>&1; do
    sleep 60
done
echo "[chain] campaign finished at $(date -Is)"

# Confirmation table for the layer-8 BERT campaign, written to its own file so
# it survives even if the next stage crashes.
~/glotenv/bin/python analyze_campaign.py results/campaign_glue_colaL8.csv \
    > results/campaign_glue_colaL8.report.txt 2>&1
echo "[chain] wrote results/campaign_glue_colaL8.report.txt"

# --- ModernBERT: which layer should the token graph be read from? ----------
# layer_probe_any.sh pre-warms each layer's cache itself (and skips a layer
# cleanly if the backbone fails to load), so no separate prewarm loop is needed.
echo "[chain] layer probe at $(date -Is)"
bash layer_probe_any.sh "$TASK" "$LAYERS" "$MODEL" \
    2>&1 | tee logs/layer_probe_modernbert.log

echo "[chain] DONE at $(date -Is)"
