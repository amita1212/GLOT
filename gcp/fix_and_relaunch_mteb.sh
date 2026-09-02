#!/usr/bin/env bash
# Stop the TinyLlama MTEB block and relaunch on RoBERTa with the corrected
# model key.
#
# WHY THIS IS SAFE TO RESTART: the queue keeps a per-unit marker at
# $MARK/<model>__<arm>__seed<n> and skips any unit whose marker exists. The 90
# BERT units are marked, so they will not re-run. The RoBERTa units returned
# rc=2 ("No valid models selected") and were therefore NEVER marked, so they
# will run. Three TinyLlama baseline units are marked and will be skipped.
#
# THE BUG: MODELS carried 'roberta-base'; run_all_experiments.py keys its
# registry on 'FacebookAI/roberta-base'. Every RoBERTa unit was skipped in
# under a second with a warning, and the queue moved on.
set -uo pipefail
ROOT=/home/t-amitalfasi/glot
Q=$ROOT/queue_mteb_table3.sh
LOCK=$ROOT/.mteb3.lock

cd "$ROOT" || exit 1

echo "=== 1. stop the running queue (by PID from its own lock file) ==="
if [ -f "$LOCK" ]; then
    QPID=$(cat "$LOCK")
    echo "  lock holds pid $QPID"
    # children first, then the queue itself, so it cannot start another unit
    pkill -TERM -P "$QPID" 2>/dev/null && echo "  signalled children of $QPID"
    sleep 3
    kill -TERM "$QPID" 2>/dev/null && echo "  signalled queue $QPID"
    sleep 3
    # anything still holding the GPU from this queue
    pkill -TERM -f 'run_all_experiments.py --with_mteb' 2>/dev/null
    sleep 2
    if [ -d "/proc/$QPID" ]; then
        echo "  still alive, sending KILL"
        kill -KILL "$QPID" 2>/dev/null
    fi
    rm -f "$LOCK"
else
    echo "  no lock file; assuming not running"
fi
echo

echo "=== 2. confirm the GPU is released ==="
sleep 5
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /'
echo "  (no rows above = GPU free)"
echo

echo "=== 3. fix the model key ==="
cp "$Q" "$Q.bak.$(date +%s)"
sed -i 's#^MODELS=(bert-base-uncased roberta-base #MODELS=(bert-base-uncased FacebookAI/roberta-base #' "$Q"
grep -n '^MODELS=' "$Q" | sed 's/^/  /'
if grep -q 'FacebookAI/roberta-base' "$Q"; then
    echo "  OK: corrected key present"
else
    echo "  FAILED to patch MODELS -- aborting"; exit 1
fi
echo

echo "=== 4. preflight: what will actually run ==="
MARKDIR=$ROOT/.mteb3_done
for m in bert-base-uncased FacebookAI_roberta-base TinyLlama_TinyLlama-1.1B-Chat-v1.0; do
    n=$(find "$MARKDIR" -name "${m}__*" 2>/dev/null | wc -l)
    echo "  $m: $n/90 units already marked done"
done
echo "  MS MARCO triplets: $( [ -f "$ROOT/data/msmarco-triplets.jsonl" ] && echo PRESENT || echo MISSING )"
echo "  corrected driver : $( [ -f "$ROOT/hyperglot_new/run_all_experiments.py" ] && echo PRESENT || echo MISSING )"
echo

echo "=== 5. relaunch ==="
nohup bash "$Q" >> "$ROOT/logs/mteb_table3.log" 2>&1 &
sleep 20
echo "  new pid: $(cat "$LOCK" 2>/dev/null || echo '?')"
echo "  --- log tail ---"
tail -15 "$ROOT/logs/mteb_table3.log" | sed 's/^/    /'
