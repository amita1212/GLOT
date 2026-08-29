#!/usr/bin/env bash
# Is this workload GPU-bound, CPU-bound, or IO-bound?
#
# Decides whether a SECOND GPU VM is worth buying, or whether the existing one
# is simply under-used and should run more workers.
cd "$(dirname "$0")" || exit 1

echo "=== MACHINE ==="
echo "vCPUs      : $(nproc)"
echo "RAM        : $(free -g | awk '/^Mem:/{print $2" GB total, "$7" GB available"}')"
echo

echo "=== SAMPLING GPU FOR 30s (util %, mem MiB) ==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv,noheader -l 5 -c 6
echo

echo "=== CPU / LOAD ==="
uptime | sed 's/^ *//'
echo "(load >> vCPUs means CPU-saturated; load ~1 with 8 vCPUs means 1 core busy)"
echo

echo "=== WHERE IS THE TIME GOING? (top processes) ==="
ps -eo pcpu,pmem,etime,comm --sort=-pcpu | head -6
echo

echo "=== DISK THROUGHPUT (cache reads) ==="
if command -v iostat >/dev/null 2>&1; then
    iostat -x 5 2 | tail -12
else
    echo "  iostat unavailable; cat /proc/diskstats delta over 5s:"
    a=$(awk '/ sda | nvme0n1 |root/{print $6+$10; exit}' /proc/diskstats 2>/dev/null || echo 0)
    sleep 5
    b=$(awk '/ sda | nvme0n1 |root/{print $6+$10; exit}' /proc/diskstats 2>/dev/null || echo 0)
    echo "  sectors r+w in 5s: $(( b - a ))  (~$(( (b-a)/2048 )) MB)"
fi
echo

echo "=== VERDICT INPUTS ==="
echo "If GPU util stays < 20% and load ~= 1, the run is single-threaded and"
echo "serialised, NOT GPU-limited. In that case a second GPU VM buys nothing"
echo "that extra concurrent WORKERS on this VM would not buy for free."
