#!/usr/bin/env bash
# =============================================================================
# MTEB Table-3 campaign.
#
#   6 arms  x  15 seeds  x  3 backbones  =  270 units
#   each unit = ONE contrastive MS MARCO training pass, then all 7 Table-3
#   tasks evaluated from that single checkpoint.
#
# The arms are baseline, A, B, C, AC, ABC.  BASELINE IS NOT OPTIONAL: every
# result in this paper is a paired delta against the backbone's own baseline,
# so an arms-only run would produce absolute numbers that cannot be tested.
#
# Task list is the ORIGINAL PAPER'S TABLE 3, not the appendix Table 12 set the
# old results used.  All seven names were resolved against the installed
# mteb 2.1.7 registry before this script was written -- none needed a .v2
# rename.
#
# RESUMABLE.  Completion is recorded by a marker file per (model, arm, seed),
# touched only on exit 0.  Re-running skips finished units.  Kill it whenever
# you like; nothing is lost but the unit in flight.
#
# ORDER IS CHEAPEST-FIRST (bert -> roberta -> tinyllama) so that stopping early
# still leaves you with COMPLETE backbone blocks rather than three ragged ones.
#
# NOTE ON PROCESS CHECKS: this uses a lock FILE, never `pgrep -f`.  A pgrep
# pattern naming a process also matches any command line that merely mentions
# it -- including a monitoring ssh -- which has already cost this project ~90
# minutes of idle GPU.
# =============================================================================
set -uo pipefail

ROOT=/home/t-amitalfasi/glot
NEW=$ROOT/hyperglot_new
PY=/home/t-amitalfasi/glotenv/bin/python
CSV=$ROOT/results/mteb_table3.csv
CKPT=$ROOT/checkpoints_mteb3
MARK=$ROOT/.mteb3_done
LOCK=$ROOT/.mteb3.lock
LOG=$ROOT/logs/mteb_table3.log

TASKS=(EmotionClassification SciFact RedditClustering AskUbuntuDupQuestions
       STS12 TwitterSemEval2015 SummEval)
ARMS=(baseline A_threshold B_threshold C_threshold AC_threshold ABC_threshold)
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
MODELS=(bert-base-uncased roberta-base TinyLlama/TinyLlama-1.1B-Chat-v1.0)

mkdir -p "$MARK" "$CKPT" "$ROOT/results" "$ROOT/logs"

# ---- single-instance lock ---------------------------------------------------
if [ -e "$LOCK" ]; then
    old=$(cat "$LOCK" 2>/dev/null || echo "?")
    if [ -d "/proc/$old" ]; then
        echo "ALREADY RUNNING as pid $old -- refusing to start a second copy."
        exit 1
    fi
    echo "stale lock from pid $old -- removing"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# ---- preflight --------------------------------------------------------------
if [ ! -f "$ROOT/data/msmarco-triplets.jsonl" ]; then
    echo "FATAL: MS MARCO triplets missing -- the contrastive stage cannot run."
    exit 1
fi
if [ ! -f "$NEW/run_all_experiments.py" ]; then
    echo "FATAL: corrected driver missing at $NEW/run_all_experiments.py"
    echo "       (this is EXACTLY the bug that killed the 30 August launch:"
    echo "        it invoked \$ROOT/run_all_experiments.py, which does not exist)"
    exit 1
fi

total=$(( ${#MODELS[@]} * ${#ARMS[@]} * ${#SEEDS[@]} ))
done_n=$(find "$MARK" -type f 2>/dev/null | wc -l)
echo "=== MTEB Table-3 campaign ==="
echo "units: $total total, $done_n already done"
echo "tasks: ${TASKS[*]}"
echo "arms : ${ARMS[*]}"
echo "csv  : $CSV"
echo

first=1
for model in "${MODELS[@]}"; do
    msafe=${model//\//_}
    for arm in "${ARMS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            marker="$MARK/${msafe}__${arm}__seed${seed}"
            if [ -f "$marker" ]; then
                continue
            fi

            echo "--- $(date '+%F %T')  $model  $arm  seed $seed ---"
            cd "$ROOT" || exit 1

            WANDB_MODE=disabled "$PY" "$NEW/run_all_experiments.py" \
                --with_mteb \
                --models "$model" \
                --configs "$arm" \
                --seeds "$seed" \
                --mteb_tasks "${TASKS[@]}" \
                --mteb_train_file "$ROOT/data/msmarco-triplets.jsonl" \
                --mteb_ckpt_dir "$CKPT" \
                --results_csv "$CSV" \
                --tasks
            rc=$?

            if [ $rc -eq 0 ]; then
                touch "$marker"
                echo "    ok"
            else
                echo "    FAILED rc=$rc"
                if [ $first -eq 1 ]; then
                    echo
                    echo "FATAL: the FIRST unit failed. Aborting rather than"
                    echo "burning days of GPU on a broken configuration."
                    exit 1
                fi
            fi
            first=0
        done
    done
    echo "=== finished backbone: $model ==="
done

echo "=== ALL DONE $(date '+%F %T') ==="
