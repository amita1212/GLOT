#!/usr/bin/env bash
# Launch the MS MARCO triplets download detached from the SSH session.
# Kept as a file because backgrounding inside `gcloud ssh --command` reliably
# mis-parses: the `&` terminates the `cd &&` chain and later commands run from
# $HOME instead of the repo.
set -euo pipefail

ROOT=/home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python
OUT="$ROOT/data/msmarco-triplets.jsonl"
LOG="$ROOT/logs/msmarco.log"

cd "$ROOT"
mkdir -p logs data

if pgrep -f "fetch_msmarco.py" > /dev/null; then
    echo "already running (pid $(pgrep -f fetch_msmarco.py | tr '\n' ' '))"
    exit 0
fi

setsid nohup "$PY" hyperglot/fetch_msmarco.py --out "$OUT" \
    > "$LOG" 2>&1 < /dev/null &

sleep 25
echo "--- pid ---"
pgrep -f fetch_msmarco.py || echo "(process already finished)"
echo "--- $LOG ---"
cat "$LOG" 2>/dev/null || echo "(no log yet)"
echo "--- output so far ---"
ls -la "$OUT" "$OUT.partial" 2>/dev/null || echo "(no output file yet)"
