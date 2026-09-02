#!/usr/bin/env bash
# Free the stalled queue on hyperglot-bgu.
#
# PID 3039 is a stranded shell from the setup step. Its own wait loop
# (`while pgrep -f setup_vm.sh`) matches its own command line, so it never
# exits. Its command line also contains the venv interpreter path, so
# queue_table1_gap.sh's wait loop matches it and blocks too.
#
# Verified before killing: zero processes have the venv python as their
# executable, and the GPU is at 0% / 0 MiB. Nothing real is being killed.
#
# NOTE: queue_table1_gap.sh is currently RUNNING (pid 7662). We do not edit it
# -- bash reads scripts incrementally and editing in place corrupts execution.

set -u
TARGET=3039

echo "=== target before kill ==="
if ! ps -p "$TARGET" -o pid=,etime=,args= 2>/dev/null | cut -c1-160; then
    echo "  PID $TARGET already gone"
    exit 0
fi

# Safety: refuse if it is a real trainer.
exe=$(readlink "/proc/$TARGET/exe" 2>/dev/null || true)
case "$exe" in
    *glotenv*|*python*)
        echo "REFUSING: PID $TARGET is a real interpreter ($exe)"
        exit 1;;
esac
echo "  exe=${exe:-<shell, no interpreter>}  -> safe to kill"

kill "$TARGET" 2>/dev/null || true
sleep 3
if ps -p "$TARGET" >/dev/null 2>&1; then
    kill -9 "$TARGET" 2>/dev/null || true
    sleep 2
fi

echo "=== after kill ==="
ps -p "$TARGET" >/dev/null 2>&1 && echo "  STILL ALIVE" || echo "  PID $TARGET gone"
echo "  wait-pattern matches now: $(pgrep -cf 'glotenv/bin/pyth[o]n')  (0 = queue will proceed)"

echo "=== queue process still alive? ==="
ps -eo pid,etime,args | grep -E 'queue_table1_gap|run_chain' | grep -v grep | cut -c1-120

echo "=== env verify (the check PID 3039 never delivered) ==="
~/glotenv/bin/python gcp/verify_env.py 2>&1 | tail -15
