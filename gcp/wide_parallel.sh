#!/usr/bin/env bash
# Wide sweep, run as N parallel workers on ONE VM.
#
# WHY PARALLEL AND NOT A SECOND VM (measured, not assumed):
#     GPU utilisation   0-8 %
#     GPU memory        812 MiB of 23034      (3.5 %)
#     CPU load          1.07 on 8 vCPUs      (one core busy, seven idle)
#     disk              ~0 MB/s
#     one python proc   101 % CPU            (single-threaded)
# The campaign is a serial single-core loop; the L4 is almost entirely idle. A
# second GPU VM would double the bill for ~2x. N workers on this VM give ~Nx for
# nothing. RAM (~5 GB/process of 31 GB) and cores are the binding limits, not the
# GPU, so 4 workers is the safe operating point.
#
# ORDERING MATTERS: prewarm runs FIRST and alone. If N workers start on a cold
# cache they each build the same hidden-state tensors simultaneously -- duplicated
# work plus a write race on the cache directory.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
mkdir -p logs results

NW="${NW:-4}"
TASKS="${1:-stsb cola}"
MODEL="bert-base-uncased"

echo "[par] ===== wide sweep, $NW workers ===== $(date -Is)"
echo "[par] tasks: $TASKS"

echo "[par] --- prewarm caches (serial, before any worker) ---"
for task in $TASKS; do
    echo "[par] prewarm $MODEL / $task"
    bash prewarm_model.sh "$MODEL" -1 "$task" > /dev/null 2>&1
done
echo "[par] prewarm done $(date -Is)"

echo "[par] --- launching $NW workers ---"
PIDS=()
for ((i = 0; i < NW; i++)); do
    nohup bash wide_worker.sh "$i" "$NW" "$TASKS" > "logs/wide_w${i}.log" 2>&1 &
    PIDS+=($!)
    echo "[par] worker $i pid ${PIDS[-1]}"
    sleep 2
done

echo "[par] waiting for workers: ${PIDS[*]}"
for p in "${PIDS[@]}"; do
    wait "$p" || echo "[par] worker pid $p exited non-zero"
done
echo "[par] all workers finished $(date -Is)"

echo "[par] --- merging per-worker csvs ---"
"$PY" merge_wide.py $TASKS

for task in $TASKS; do
    f="results/campaign_wide_${task}.csv"
    [ -e "$f" ] || continue
    echo
    echo "[par] ================= $task ================="
    "$PY" paired_analysis.py "$f" 2>&1 | head -24
    "$PY" analyze_campaign.py "$f" > "results/campaign_wide_${task}.report.txt" 2>&1
done

echo "[par] ===== ALL DONE $(date -Is) ====="
