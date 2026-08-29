#!/usr/bin/env bash
# Status of the PARALLEL wide sweep.
cd "$(dirname "$0")" || exit 1
PY=/home/t-amitalfasi/glotenv/bin/python

echo "=== WORKERS ==="
pgrep -af 'wide_worker|campaign\.py' | grep -v par_status | cut -c1-100 || echo "  (none)"
echo

echo "=== MACHINE (was: load 1.07/8 cores, GPU 0-8%) ==="
uptime | sed 's/^ *//'
free -g | awk '/^Mem:/{print "  RAM: "$3" GB used, "$7" GB available of "$2" GB"}'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader \
  | awk '{print "  GPU: "$0}'
echo

echo "=== PER-WORKER ROWS ==="
tot=0
for f in results/campaign_wide_*_w*.csv; do
    [ -e "$f" ] || continue
    n=$(( $(wc -l < "$f") - 1 ))
    tot=$(( tot + n ))
    printf '  %-40s %4d\n' "$(basename "$f")" "$n"
done
echo "  ------------------------------------------------"
printf '  %-40s %4d / 990\n' "TOTAL" "$tot"
echo

echo "=== THROUGHPUT ==="
$PY - <<'PYEOF'
import csv, glob, statistics, time, os
times, newest, oldest = [], 0, float("inf")
for f in glob.glob("results/campaign_wide_*_w*.csv"):
    for r in csv.DictReader(open(f)):
        if r.get("elapsed_sec"):
            try:
                times.append(float(r["elapsed_sec"]))
            except ValueError:
                pass
    m = os.path.getmtime(f)
    newest, oldest = max(newest, m), min(oldest, m)
if times:
    med = statistics.median(times)
    print(f"  per-run median: {med:.0f} s (n={len(times)})")
    print(f"  serial ETA was 59 h; with 4 workers expect ~{59/4:.0f} h")
else:
    print("  (no timed runs yet -- still prewarming or first runs in flight)")
PYEOF
echo

echo "=== ORCHESTRATOR ==="
tail -12 logs/wide_parallel.log 2>/dev/null || echo "  (no log)"
