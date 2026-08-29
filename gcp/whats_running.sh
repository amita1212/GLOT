#!/usr/bin/env bash
# Status snapshot: what is running, what finished, how far along.
cd "$(dirname "$0")" || exit 1

echo "=== RUNNING PROCESSES ==="
pgrep -af 'structural_arms|roberta_compare|decoder_sweep|launch_decoder|campaign.py|fix_modernbert|test_scale_fix' \
  | grep -v whats_running | cut -c1-115
echo

echo "=== UPTIME / GPU ==="
uptime | sed 's/^ *//'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo

echo "=== DISK ==="
df -h / | tail -1
echo "data cache: $(du -sh data 2>/dev/null | cut -f1)"
echo

echo "=== RESULT CSVs (newest first) ==="
ls -lt results/*.csv 2>/dev/null | head -14 | awk '{printf "%-10s %s %s %s  %s\n", $5, $6, $7, $8, $9}'
echo

echo "=== ROW COUNTS PER CAMPAIGN CSV ==="
for f in results/campaign_*.csv; do
  [ -e "$f" ] || continue
  n=$(( $(wc -l < "$f") - 1 ))
  printf "%-46s %5d rows\n" "$(basename "$f")" "$n"
done
echo

echo "=== LOG TAILS ==="
for f in logs/structural.log logs/roberta.log logs/decoder.log logs/fix_modernbert.log; do
  echo "--- $f ---"
  if [ -e "$f" ]; then
    tail -6 "$f" | cut -c1-115
  else
    echo "(not created yet)"
  fi
done
