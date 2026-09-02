#!/usr/bin/env bash
# Is the RoBERTa fill campaign still progressing, or has it stalled?
cd /home/t-amitalfasi/glot/results || exit 1

echo "--- robfill shards: rows / last modified ---"
shopt -s nullglob
for f in campaign_robfill_*.csv; do
    printf '%-42s %5s rows   %s\n' "$f" "$(wc -l < "$f")" "$(date -r "$f" '+%m-%d %H:%M')"
done
[ -z "$(echo campaign_robfill_*.csv)" ] && echo "(no robfill shards found)"

echo
echo "--- now ---"
date '+%m-%d %H:%M %Z'

echo
echo "--- worker python processes ---"
ps -eo pid,etime,args | grep 'hyperglot/main.py' | grep -v grep | \
    awk '{printf "pid=%s elapsed=%s\n", $1, $2}'

echo
echo "--- driver processes ---"
ps -eo pid,etime,args | grep -E 'run_all_experiments|campaign\.py|worker_' | grep -v grep | \
    cut -c1-120

echo
echo "--- newest lines in run logs ---"
cd /home/t-amitalfasi/glot/logs 2>/dev/null || exit 0
for f in $(ls -t *.log 2>/dev/null | head -3); do
    echo "== $f (modified $(date -r "$f" '+%m-%d %H:%M')) =="
    tail -3 "$f"
done
