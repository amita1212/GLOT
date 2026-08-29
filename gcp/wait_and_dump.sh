#!/usr/bin/env bash
# Block until the named sweep finishes, then dump its results CSV.
# Kept in a file because `$(...)`, `>` and `!` all break through plink when
# passed inline via `gcloud compute ssh --command`.
set -u
cd /home/t-amitalfasi/glot
PAT="${1:-geometry_sweep.py}"
CSV="${2:-results/e2_geometry.csv}"
MAX="${3:-80}"
for i in $(seq 1 "$MAX"); do
  if ! pgrep -f "$PAT" > /dev/null; then
    echo "FINISHED after $((i-1)) polls"
    break
  fi
  sleep 45
done
pgrep -f "$PAT" > /dev/null && echo "STILL RUNNING (timed out waiting)"
bash dump_csv.sh "$CSV"
