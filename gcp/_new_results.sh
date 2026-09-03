#!/usr/bin/env bash
# Read-only: what has landed since yesterday.
cd ~/glot || exit 1
echo "=== host $(hostname)  $(date -u '+%F %T') UTC ==="
echo
echo "--- live processes ---"
ps -eo pid,etime,args --no-headers | grep -E 'python|\.sh' | grep -v grep \
  | grep -vE '_status|_new_results|unattended|shutdown_risk' \
  | awk '{printf "  %-8s %-12s %s\n",$1,$2,substr($0,index($0,$3),90)}'
echo
echo "--- results CSVs touched in the last 36h ---"
find results -name '*.csv' -mmin -2160 -printf '  %TY-%Tm-%Td %TH:%TM  %8s  %p\n' 2>/dev/null | sort -k1,2
echo
echo "--- confirmation row counts (per campaign file) ---"
for f in results/*.csv; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  [ "$n" -le 1 ] && continue
  printf '  %-46s %5d lines\n' "$(basename "$f")" "$n"
done
echo
echo "--- MTEB markers ---"
for m in bert-base-uncased FacebookAI_roberta-base TinyLlama_TinyLlama-1.1B-Chat-v1.0; do
  n=$(find .mteb3_done -name "${m}__*" 2>/dev/null | wc -l)
  printf '  %-42s %3d/90\n' "$m" "$n"
done
echo
echo "--- chain log tail ---"
for L in logs/chain_gpu1.log logs/queue_table1_gap.log logs/queue_mteb_table3.log; do
  [ -f "$L" ] || continue
  echo "  == $L (modified $(date -r "$L" '+%F %H:%M')) =="
  tail -n 8 "$L" | sed 's/^/     /'
done
