#!/usr/bin/env bash
# Read-only status probe. Touches nothing that is running.
cd ~/glot || exit 1
echo "=== HOST ==="; hostname; date -u '+%Y-%m-%d %H:%M:%S UTC'; uptime | sed 's/^/  /'
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo "  -- processes on the GPU --"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo
echo "=== CAMPAIGN PROCESSES (queue/worker/python main) ==="
ps -eo pid,ppid,etime,pcpu,cmd --sort=start_time \
  | grep -E 'queue_|worker|campaign\.py|main\.py|factorial|run_all_exp' \
  | grep -v grep | cut -c1-150 | sed 's/^/  /'
echo
echo "=== WHICH JOBS DOES THE QUEUE DEFINE, AND WHERE IS IT? ==="
grep -m1 -n '^JOBS=' queue_table1_gap.sh 2>/dev/null | cut -c1-200 | sed 's/^/  /'
echo
echo "=== LOG ACTIVITY (growth over 15s tells us if it is alive) ==="
for f in $(ls -t logs/*.log nohup.out 2>/dev/null | head -3); do
  a=$(stat -c %s "$f"); sleep 0; echo "  $f  size=$a  modified=$(stat -c %y "$f" | cut -d. -f1)"
done
newest=$(ls -t logs/*.log nohup.out 2>/dev/null | head -1)
if [ -n "$newest" ]; then
  b1=$(stat -c %s "$newest"); sleep 15; b2=$(stat -c %s "$newest")
  echo "  growth of $newest over 15s: $((b2-b1)) bytes"
  echo "  --- last 8 lines ---"; tail -8 "$newest" | cut -c1-150 | sed 's/^/    /'
fi
echo
echo "=== CONFIRM-ROW PROGRESS PER RESULTS CSV (today) ==="
for f in results/*.csv; do
  [ -f "$f" ] || continue
  n=$(awk -F, 'NR>1 && /confirm/' "$f" 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || continue
  printf '  %-42s confirm_rows=%-6s last=%s\n' "$(basename "$f")" "$n" "$(stat -c %y "$f" | cut -d. -f1)"
done
