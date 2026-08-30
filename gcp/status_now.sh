#!/usr/bin/env bash
# Read-only status. A FILE, not an inline --command string: PowerShell
# re-expands $(...) on the client side and has already mangled three of these.
cd /home/t-amitalfasi/glot || exit 1
echo "=== time ==="; date -Is
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo
echo "=== processes ==="
ps -eo pid,etime,args | grep -E 'campaign\.py|queue_rest|factorial_geom|run_all_exp' | grep -v grep
echo
echo "=== queue log (last 12) ==="
tail -12 logs/queue_rest.log 2>/dev/null
echo
echo "=== robfill stsb progress (want 111 each) ==="
for f in results/campaign_robfill_stsb_w*.csv; do
    [ -f "$f" ] && printf '%-45s %s\n' "$f" "$(( $(wc -l < "$f") - 1 ))"
done
echo
echo "=== queue outputs ==="
for f in results/factorial_scale_roberta.csv results/factorial_geom_cola.csv \
         results/stageA_n50_cola.csv results/mteb_trained.csv \
         results/campaign_decoder_stsb_BC.csv; do
    if [ -f "$f" ]; then printf '%-45s %s rows\n' "$f" "$(( $(wc -l < "$f") - 1 ))"
    else printf '%-45s -\n' "$f"; fi
done
