#!/usr/bin/env bash
# Restart the GPU-1 chain so it picks up a new stage.
#
# Safe right now because the chain is still in its wait loop -- it has not
# pulled, verified parity or started a campaign, so killing it discards
# nothing. Editing the file underneath the running shell would NOT be safe:
# bash re-reads a script by byte offset, so an in-place edit of a running
# script can make it execute garbage. Replace, then relaunch.
set -uo pipefail
ROOT=/home/t-amitalfasi/glot
LOCK=$ROOT/.chain_gpu1.lock
cd "$ROOT" || exit 1

echo "=== stop the waiting chain ==="
if [ -f "$LOCK" ]; then
    old=$(cat "$LOCK")
    if [ -d "/proc/$old" ]; then
        # refuse if it has moved past waiting -- never kill a live campaign
        if grep -q 'START' logs/chain_gpu1.log 2>/dev/null; then
            echo "  chain has already STARTED a campaign -- refusing to restart."
            tail -3 logs/chain_gpu1.log
            exit 1
        fi
        kill -TERM "$old" 2>/dev/null
        sleep 2
        [ -d "/proc/$old" ] && kill -KILL "$old" 2>/dev/null
        echo "  stopped pid $old (was waiting only)"
    else
        echo "  lock is stale"
    fi
    rm -f "$LOCK"
else
    echo "  no lock present"
fi

echo
echo "=== install the updated chain ==="
cp /tmp/chain_gpu1.sh "$ROOT/chain_gpu1.sh"
sed -i 's/\r//' "$ROOT/chain_gpu1.sh"
echo "  stages now in the chain:"
grep -n '^say "START' "$ROOT/chain_gpu1.sh" | sed 's/^/    /'

echo
echo "=== relaunch ==="
nohup bash "$ROOT/chain_gpu1.sh" > "$ROOT/logs/chain_boot.log" 2>&1 &
CHILD=$!
echo "$CHILD" > "$LOCK"
sleep 5
if [ -d "/proc/$CHILD" ]; then
    echo "  relaunched, pid $CHILD"
else
    echo "  FAILED; boot log:"; cat "$ROOT/logs/chain_boot.log"; exit 1
fi
tail -2 "$ROOT/logs/chain_gpu1.log" | sed 's/^/    /'
