#!/usr/bin/env bash
# Compact status: what has landed since 1 September?
set -u
cd /home/t-amitalfasi/glot 2>/dev/null || cd ~/glot || exit 1
echo "=== $(hostname) $(date -Is) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null

echo "--- running ---"
ps -eo etime,args | grep -E 'glotenv/bin/python|queue' | grep -v grep | cut -c1-115 || echo "  idle"

echo "--- results changed in the last 36h ---"
find results -name '*.csv' -mmin -2160 -printf '%TY-%Tm-%Td %TH:%TM  %-46p  ' -exec sh -c 'echo "$(($(wc -l < "$1") - 1)) rows"' _ {} \; 2>/dev/null | sort

echo "--- newest queue logs ---"
for f in $(ls -t logs/*.log 2>/dev/null | head -3); do
    echo "[$f]"; tail -3 "$f"
done
