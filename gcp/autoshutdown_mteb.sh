#!/usr/bin/env bash
# =============================================================================
# Power the VM off once the MTEB Table-3 queue has drained.
#
# Runs as a SEPARATE process and never touches the running queue script.
# Editing a bash script while bash is executing it can corrupt the run --
# bash re-reads the file by byte offset -- so the queue is left alone and this
# watcher only observes its lock file.
#
# The queue writes its pid to .mteb3.lock and removes it on exit (EXIT trap).
# We treat "lock gone" OR "lock names a pid that no longer exists" as done,
# which also covers the queue being killed or dying.
#
# Stop, not delete: the disk, the results and the environment all survive, and
# the instance can be restarted in a minute.
# =============================================================================
set -uo pipefail

LOCK=/home/t-amitalfasi/glot/.mteb3.lock
LOG=/home/t-amitalfasi/glot/logs/autoshutdown.log
POLL=300      # seconds between checks
GRACE=120     # seconds to wait after the queue ends, so the last CSV flush lands

echo "$(date '+%F %T') watcher started, pid $$" >> "$LOG"

while true; do
    if [ ! -e "$LOCK" ]; then
        echo "$(date '+%F %T') lock file gone -- queue finished" >> "$LOG"
        break
    fi
    pid=$(cat "$LOCK" 2>/dev/null || echo 0)
    if [ ! -d "/proc/$pid" ]; then
        echo "$(date '+%F %T') lock names pid $pid which is not running -- queue gone" >> "$LOG"
        break
    fi
    sleep "$POLL"
done

echo "$(date '+%F %T') grace period ${GRACE}s" >> "$LOG"
sleep "$GRACE"

echo "$(date '+%F %T') final state:" >> "$LOG"
ls -1 /home/t-amitalfasi/glot/.mteb3_done/ 2>/dev/null | wc -l >> "$LOG"
wc -l /home/t-amitalfasi/glot/results/mteb_table3.csv >> "$LOG" 2>&1

echo "$(date '+%F %T') powering off" >> "$LOG"
sudo shutdown -h now
