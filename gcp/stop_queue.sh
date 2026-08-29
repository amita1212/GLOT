#!/usr/bin/env bash
# Stop the queue, verify the redeploy is safe, syntax-check, relaunch.
#
# Exists as a FILE because inline gcloud --command strings are re-expanded by
# PowerShell on the client: $(...) runs locally, so `kill $(cat pidfile)` became
# `kill` with an empty argument and then tried to kill PID 2. Anything with
# $(...), pipes or redirection goes in a script and gets scp'd.
set -u
cd /home/t-amitalfasi/glot || exit 1
PIDFILE=logs/queue_rest.pid

echo "--- campaigns running: $(pgrep -cf 'campaign.py') (queue is idle if >0)"

if [ -f "$PIDFILE" ]; then
    pid=$(cat "$PIDFILE")
    # only kill it if it really is our queue, never a recycled pid
    if kill -0 "$pid" 2>/dev/null && ps -p "$pid" -o args= | grep -q queue_rest; then
        kill "$pid" && echo "--- stopped queue pid $pid"
        sleep 2
    else
        echo "--- pidfile $pid is stale or not the queue; not killing"
    fi
else
    echo "--- no pidfile"
fi

# belt and braces: any surviving queue shell, matched on the real process only
for p in $(pgrep -f 'bash .*queue_rest.sh' || true); do
    echo "--- killing leftover queue pid $p"
    kill "$p" 2>/dev/null
done
sleep 1
echo "--- queue procs remaining: $(pgrep -cf 'bash .*queue_rest.sh')"
echo "--- campaigns still running: $(pgrep -cf 'campaign.py')"
