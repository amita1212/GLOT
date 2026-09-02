#!/usr/bin/env bash
# Swap the running queue for the corrected one.
#
# We cannot edit queue_rest.sh in place: bash reads a script incrementally by
# byte offset, and items 4-5 have not been read yet, so an edit would make the
# running interpreter resume mid-token. Stopping the queue shell and starting a
# corrected one is the only safe route.
#
# Stage A (campaign.py) is a CHILD of the queue but is killed by nothing here:
# a SIGTERM to the parent shell is not propagated to it, and its stdout is a
# file, so it cannot take a SIGPIPE either. It keeps running, orphaned, and the
# new queue's wait loop blocks on it exactly as the old one would have.
set -u
cd /home/t-amitalfasi/glot || exit 1
OLD_PID=${1:?usage: swap_queue.sh <old-queue-pid>}

echo "=== before ==="
ps -p "$OLD_PID" -o pid=,etime=,args= 2>/dev/null | cut -c1-100 || {
    echo "  queue pid $OLD_PID not running; nothing to stop"; }
CAMP=$(pgrep -f 'campaign[.]py' | head -1)
echo "  campaign.py (Stage A) pid: ${CAMP:-<none>}"

# Refuse to kill anything that is not the queue shell.
if ps -p "$OLD_PID" -o args= 2>/dev/null | grep -q 'queue_rest\.sh'; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 3
    ps -p "$OLD_PID" >/dev/null 2>&1 && { kill -9 "$OLD_PID" 2>/dev/null || true; sleep 2; }
    echo "  queue shell stopped"
else
    echo "  pid $OLD_PID is not queue_rest.sh -- REFUSING to kill"
    exit 1
fi

echo
echo "=== Stage A must have SURVIVED the parent's death ==="
if [ -n "${CAMP:-}" ] && ps -p "$CAMP" >/dev/null 2>&1; then
    ps -p "$CAMP" -o pid=,etime=,args= | cut -c1-110
    echo "  OK: still training"
else
    echo "  WARNING: campaign.py is gone. The new queue will restart Stage A"
    echo "  from its guard (it resumes by run_key, so no work is duplicated)."
fi

echo
echo "=== launching corrected queue ==="
# setsid so it survives this SSH session. The launch command must not contain
# the string the queue greps for, or its wait loop would match this very line.
setsid nohup bash queue_rest2.sh > logs/queue2.log 2>&1 < /dev/null &
echo $! > /tmp/queue2.pid
sleep 5
NEW_PID=$(cat /tmp/queue2.pid)
if ps -p "$NEW_PID" >/dev/null 2>&1; then
    echo "  running as pid $NEW_PID"
else
    echo "  FAILED to start -- see logs/queue2.log"
fi

echo
echo "=== queue2 log ==="
tail -5 logs/queue2.log

echo
echo "=== final state ==="
ps -eo pid,etime,args | grep -E 'queue_rest2|campaign[.]py' | grep -v grep | cut -c1-110
