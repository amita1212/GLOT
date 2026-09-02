#!/usr/bin/env bash
# End-to-end smoke test of the corrected MTEB pipeline.
#
# Deliberately runs from a SEPARATE clone (hyperglot_new/) so the in-flight
# RoBERTa campaign keeps using the exact main.py it started with -- swapping
# code under a running campaign would make its trials non-comparable.
#
# CWD stays ~/glot so the relative "./data/" cache and the MS MARCO file are
# shared rather than rebuilt.
set -uo pipefail

ROOT=/home/t-amitalfasi/glot
NEW=$ROOT/hyperglot_new
PY=/home/t-amitalfasi/glotenv/bin/python
BRANCH=hyperglot-stageA
REPO=https://github.com/amita1212/GLOT.git

if [ ! -d "$NEW/.git" ]; then
    echo "=== cloning $BRANCH into $NEW ==="
    rm -rf "$NEW"
    git clone --depth 1 --branch "$BRANCH" "$REPO" "$NEW" || exit 1
else
    echo "=== updating existing clone ==="
    git -C "$NEW" fetch --depth 1 origin "$BRANCH" && git -C "$NEW" reset --hard FETCH_HEAD
fi

echo "=== clone HEAD ==="
git -C "$NEW" log --oneline -1

echo "=== sanity: does the clone have the fixes? ==="
grep -c 'save_ckpt_path' "$NEW/main.py" "$NEW/run_all_experiments.py"

cd "$ROOT" || exit 1
mkdir -p logs

echo "=== launching smoke test ==="
# wandb.init is unconditional in main.py; disabled so it cannot block on login.
WANDB_MODE=disabled setsid nohup "$PY" "$NEW/run_all_experiments.py" \
    --with_mteb \
    --models bert-base-uncased \
    --configs baseline \
    --seeds 1 \
    --mteb_tasks STS13 \
    --mteb_train_file "$ROOT/data/msmarco-triplets.jsonl" \
    --mteb_ckpt_dir "$ROOT/checkpoints_smoke" \
    --results_csv "$ROOT/results/smoke_mteb.csv" \
    --stop_on_error \
    --tasks \
    > "$ROOT/logs/smoke_mteb.log" 2>&1 < /dev/null &

sleep 30
echo "=== pid ==="
pgrep -f 'run_all_experiments.py' || echo "(driver not running -- check log)"
echo "=== log so far ==="
tail -20 "$ROOT/logs/smoke_mteb.log" 2>/dev/null || echo "(no log yet)"
