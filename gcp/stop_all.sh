#!/usr/bin/env bash
# Stop all running campaigns cleanly before the VM is stopped.
#
# NOTE: never `pkill -f <pattern>` from an inline ssh --command in this project;
# the pattern has matched and killed the ssh command itself. Resolve PIDs first,
# exclude our own process tree, then kill by PID.
set -u
cd /home/t-amitalfasi/glot

SELF=$$
PIDS=$(pgrep -f 'structural_arms.sh|roberta_compare.sh|decoder_sweep.sh|campaign.py|prewarm_model.sh' 2>/dev/null \
       | grep -vw "$SELF" | grep -vw "$PPID")

if [ -z "$PIDS" ]; then
    echo "[stop] nothing running"
else
    echo "[stop] terminating:"
    for p in $PIDS; do
        printf '   %s  %s\n' "$p" "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-90)"
    done
    kill $PIDS 2>/dev/null
    sleep 10
    STILL=$(for p in $PIDS; do kill -0 "$p" 2>/dev/null && echo "$p"; done)
    [ -n "$STILL" ] && { echo "[stop] force-killing: $STILL"; kill -9 $STILL 2>/dev/null; }
fi

sleep 2
echo "[stop] remaining:"
pgrep -af 'structural_arms.sh|roberta_compare.sh|decoder_sweep.sh|campaign.py' | grep -vw "$SELF" || echo "   (none)"

echo "[stop] rows preserved (campaigns resume from run_key):"
for f in results/campaign_*.csv; do
    [ -e "$f" ] || continue
    printf '   %-46s %5d rows\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
done
