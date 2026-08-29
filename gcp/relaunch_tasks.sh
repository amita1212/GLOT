#!/usr/bin/env bash
# Swap the running queue for the full multi-task queue.
# Kills by PID (never `pkill -f`, which has matched and killed the ssh command
# itself in this project).
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results

for pid in $(pgrep -f 'chain_next.sh|chain_modernbert.sh'); do
    echo "killing stale chain pid $pid"
    kill "$pid" 2>/dev/null || true
done

nohup bash chain_tasks.sh "stsb rte mrpc" > logs/chain_tasks.log 2>&1 &
echo "launched chain_tasks pid $!"
