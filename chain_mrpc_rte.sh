#!/usr/bin/env bash
# Wide campaign on MRPC and RTE, chained to start when the RoBERTa job finishes.
#
# WHY CHAINED RATHER THAN PARALLEL: the L4 is already saturated by RoBERTa's 4
# workers (load 3.98 on 8 vCPU, GPU 89%). Adding 4 more would roughly halve the
# throughput of both jobs and buy nothing.
#
# WHY THIS RUN EXISTS: the wide campaign covers CoLA and STS-B only. MRPC and
# RTE are still at n=3, where the minimum detectable effect (0.68 and 6.12) is
# larger than any gap we observe -- so we currently cannot say whether any arm
# beats the baseline on those two tasks. Until this lands, "no arm beats the
# baseline on any task" is unsupported.
#
# WHY POLL THE LOG RATHER THAN pgrep: `pgrep -f roberta_all` also matches the
# polling process and any ssh command line containing the pattern, which has
# silently broken chained launches in this project before. The DONE marker is
# written by roberta_all.sh itself and is unambiguous.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results

ROB_LOG=logs/roberta_all.log
TASKS="mrpc rte"
NW=4

echo "[chain] waiting for RoBERTa to finish $(date -Is)"
while ! grep -q "ALL DONE" "$ROB_LOG" 2>/dev/null; do
    sleep 300
done
echo "[chain] RoBERTa finished, starting MRPC/RTE $(date -Is)"

echo "[chain] --- prewarm bert-base-uncased on $TASKS (serial) ---"
bash prewarm_model.sh bert-base-uncased -1 $TASKS 2>&1 | tail -10
echo "[chain] prewarm done $(date -Is)"

for ((i = 0; i < NW; i++)); do
    bash wide_worker.sh "$i" "$NW" "$TASKS" > "logs/wide2_par_w${i}.log" 2>&1 &
    echo "[chain] worker $i pid $!"
done
wait
echo "[chain] all workers finished $(date -Is)"

PY=~/glotenv/bin/python
for task in $TASKS; do
    "$PY" merge_glob.py "results/campaign_wide_${task}_w*.csv" \
        "results/campaign_wide_${task}.csv"
done

for task in $TASKS; do
    f="results/campaign_wide_${task}.csv"
    [ -e "$f" ] || continue
    echo
    echo "[chain] ================= $task ================="
    "$PY" paired_analysis.py "$f" 2>&1 | head -24
    echo "--- vs no_graph ---"
    "$PY" vs_reference.py "$f" no_graph 2>&1 | head -24
    "$PY" analyze_campaign.py "$f" > "results/campaign_wide_${task}.report.txt" 2>&1
done

echo "[chain] ===== MRPC/RTE ALL DONE $(date -Is) ====="
