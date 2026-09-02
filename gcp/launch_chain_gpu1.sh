#!/usr/bin/env bash
# Launch the GPU-1 chain exactly once.
#
# NOT using pgrep: any -f pattern naming the chain is also matched by the ssh
# command that mentions it, which is how the previous attempt reported "already
# running" and launched nothing -- the same self-match the Table-1 queue
# documents. A lock file holding a pid, tested against /proc, cannot be fooled
# by an observer.
set -uo pipefail
ROOT=/home/t-amitalfasi/glot
LOCK=$ROOT/.chain_gpu1.lock
cd "$ROOT" || exit 1
mkdir -p logs

if [ -f "$LOCK" ]; then
    old=$(cat "$LOCK" 2>/dev/null || echo "?")
    if [ -n "$old" ] && [ -d "/proc/$old" ]; then
        echo "chain already running as pid $old"
        exit 0
    fi
    echo "stale lock from pid $old -- removing"
    rm -f "$LOCK"
fi

nohup bash "$ROOT/chain_gpu1.sh" > "$ROOT/logs/chain_boot.log" 2>&1 &
CHILD=$!
echo "$CHILD" > "$LOCK"
sleep 5

if [ -d "/proc/$CHILD" ]; then
    echo "launched, pid $CHILD"
else
    echo "FAILED to stay alive; boot log:"
    cat "$ROOT/logs/chain_boot.log"
    exit 1
fi

echo "--- chain log ---"
tail -8 "$ROOT/logs/chain_gpu1.log" 2>/dev/null | sed 's/^/  /'
echo "--- what it is waiting for ---"
tail -2 "$ROOT/logs/queue_table1_gap.log" | sed 's/^/  /'
