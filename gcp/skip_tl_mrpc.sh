#!/usr/bin/env bash
# Make the queue skip tl/mrpc via its OWN documented cache guard.
#
# queue_table1_gap.sh checks for data/<model>_<task>_<split>_batches/metadata.json
# and, if absent, logs "ABORT ... cache missing" and `continue 2` -- skipping
# that job and moving on. Renaming the marker therefore skips MRPC without
# editing the running script (bash re-reads a script by byte offset, so editing
# one mid-run can corrupt execution), without killing the queue, and without
# writing a single fabricated result row.
#
# Fully reversible: rerun with `restore` to put the markers back.
set -u
cd ~/glot || exit 1
ACTION="${1:-skip}"

for s in train val; do
  D="data/TinyLlama_TinyLlama-1.1B-Chat-v1.0_mrpc_${s}_batches"
  if [ "$ACTION" = "restore" ]; then
    if [ -f "$D/metadata.json.skipped" ]; then
      mv "$D/metadata.json.skipped" "$D/metadata.json"
      echo "  restored $D/metadata.json"
    else
      echo "  nothing to restore in $D"
    fi
  else
    if [ -f "$D/metadata.json" ]; then
      mv "$D/metadata.json" "$D/metadata.json.skipped"
      echo "  parked   $D/metadata.json -> metadata.json.skipped"
    else
      echo "  already parked/absent: $D"
    fi
  fi
done

echo
echo "=== verification: what the queue will see for tl/mrpc ==="
for s in train val; do
  D="data/TinyLlama_TinyLlama-1.1B-Chat-v1.0_mrpc_${s}_batches"
  if [ -f "$D/metadata.json" ]; then echo "  PRESENT -> would RUN"; else echo "  MISSING -> will SKIP"; fi
done
echo
echo "=== the cached tensors themselves are untouched ==="
du -sh data/TinyLlama_TinyLlama-1.1B-Chat-v1.0_mrpc_train_batches 2>/dev/null
ls data/TinyLlama_TinyLlama-1.1B-Chat-v1.0_mrpc_train_batches | head -3 | sed 's/^/  /'
