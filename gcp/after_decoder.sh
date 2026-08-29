#!/usr/bin/env bash
# Resume the two paused campaigns once the decoder sweep finishes.
#
# Both were stopped mid-flight to give the decoder priority. Every campaign
# dedups on run_key, so they pick up exactly where they left off:
#   campaign_struct_cola.csv  had 84 rows preserved
# Only the single in-flight run of each was lost.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs

echo "[chain] waiting for the decoder sweep to finish... $(date -Is)"
while pgrep -f 'decoder_sweep.sh' >/dev/null 2>&1; do
    sleep 300
done
echo "[chain] decoder done at $(date -Is)"

echo "[chain] resuming structural arms $(date -Is)"
bash structural_arms.sh cola rte mrpc >> logs/structural.log 2>&1
echo "[chain] structural done at $(date -Is)"

echo "[chain] resuming roberta comparison $(date -Is)"
bash roberta_compare.sh cola stsb mrpc rte >> logs/roberta.log 2>&1
echo "[chain] roberta done at $(date -Is)"

echo "[chain] ALL CAMPAIGNS COMPLETE $(date -Is)"
