#!/usr/bin/env bash
# Detached launcher for queue_rest.sh.
#
# nohup/& get mangled when passed through `gcloud compute ssh --command`, so
# the backgrounding must happen inside a script on the VM.
#
# The double-start guard uses a PIDFILE, not pgrep: `pgrep -f queue_rest.sh`
# also matches the ssh command line that invokes this script, so it reports
# "already running" every single time and silently refuses to launch.
set -u
cd /home/t-amitalfasi/glot || exit 1
mkdir -p logs
PIDFILE=logs/queue_rest.pid

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "ALREADY RUNNING pid $(cat "$PIDFILE")"
    exit 0
fi

sed -i 's/\r$//' queue_rest.sh
setsid nohup bash queue_rest.sh > logs/queue_rest.log 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 3
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "LAUNCHED pid $(cat "$PIDFILE")"
else
    echo "FAILED to stay alive; see logs/queue_rest.log"
fi
echo "--- log head ---"
head -5 logs/queue_rest.log 2>/dev/null || echo "(no output yet)"
