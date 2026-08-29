#!/usr/bin/env bash
# Replace the old ModernBERT-only chain with the re-prioritised queue.
# Kills the waiting chain by PID (never `pkill -f`, which has matched and killed
# the ssh command itself in this project) and launches chain_next.sh detached.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results

for pid in $(pgrep -f 'chain_modernbert.sh'); do
    echo "killing stale chain pid $pid"
    kill "$pid" 2>/dev/null || true
done

nohup bash chain_next.sh > logs/chain_next.log 2>&1 &
echo "launched chain_next pid $!"
