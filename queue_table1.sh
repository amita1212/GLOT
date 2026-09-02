#!/usr/bin/env bash
# Complete Table 1 on six GLUE tasks x three backbones x six arms.
#
# TASKS   cola stsb mrpc sst2 qnli mnli   (7 of the original's 10 columns;
#         MNLI yields both -m and -mm from one run). qqp/rte/wnli deliberately
#         dropped -- see _table1six.py for the reasoning.
# ARMS    baseline A B C AC BC
#
# DESIGN NOTES
#   * One task at a time. Its hidden-state cache is built, used, then DELETED
#     before the next task starts, so peak disk is one task's cache (81 GB at
#     the worst, TinyLlama/MNLI) rather than the 585 GB the full matrix needs.
#   * Idempotent. campaign.py dedups on run_key, and each task is skipped
#     outright once its CSV already holds the expected row count, so the script
#     can be killed and restarted (spot preemption) without losing work.
#   * TRIALS is per backbone. The existing TinyLlama STS-B campaign used 10
#     trials, so setting TL_TRIALS=40 would make that block unequal-budget --
#     the exact defect the paper criticises. If you set 40, you must also
#     re-run TinyLlama STS-B at 40; the script refuses otherwise.
#   * Never runs two GPU jobs at once: waits for any existing campaign to end.
#
# USAGE   bash queue_table1.sh [--dry-run]
set -u

cd "$(dirname "$0")" || exit 1
PY=/home/t-amitalfasi/glotenv/bin/python
LOG=logs/queue_table1.log
mkdir -p logs results

# ---- configuration ---------------------------------------------------------
TASKS=(cola stsb mrpc sst2 qnli mnli)
ARMS="baseline A B C AC BC"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
ENC_TRIALS=${ENC_TRIALS:-40}      # bert, roberta
TL_TRIALS=${TL_TRIALS:-10}        # tinyllama; 10 matches its existing campaign
KEEP_CACHE=${KEEP_CACHE:-0}       # 1 = do not delete caches between tasks

BACKBONES=("bert-base-uncased" "roberta-base" "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say() { echo "[table1 $(date -Is)] $*" | tee -a "$LOG"; }

# ---- guard: TinyLlama budget must match its existing STS-B campaign ---------
if [ "$TL_TRIALS" -ne 10 ] && [ ! -f results/campaign_tl_stsb_t${TL_TRIALS}.csv ]; then
    say "REFUSING: TL_TRIALS=$TL_TRIALS but TinyLlama STS-B was run at 10."
    say "  Re-run it at $TL_TRIALS first, or leave TL_TRIALS=10."
    exit 1
fi

trials_for() {
    case "$1" in
        TinyLlama*) echo "$TL_TRIALS" ;;
        *)          echo "$ENC_TRIALS" ;;
    esac
}

short() {   # model id -> short tag used in filenames
    case "$1" in
        bert-base-uncased) echo bert ;;
        roberta-base)      echo rob ;;
        TinyLlama*)        echo tl ;;
    esac
}

cache_glob() {   # model id, task -> the cache dirs that task owns
    local m t
    m=$(echo "$1" | tr '/' '_')
    t="$2"; [ "$t" = "stsb" ] && t=sts     # main.py names STS-B's cache 'sts'
    echo "data/${m}_${t}_train_batches data/${m}_${t}_val_batches"
}

# ---- wait for any other GPU job to finish ---------------------------------
while pgrep -f 'campaign\.py|factorial_geom\.py|run_all_experiments\.py' \
        | grep -qv "^$$\$"; do
    say "waiting for the running campaign to finish..."
    sleep 300
done
say "GPU free. ENC_TRIALS=$ENC_TRIALS TL_TRIALS=$TL_TRIALS KEEP_CACHE=$KEEP_CACHE"

# ---- main loop: cheapest backbone/task first -------------------------------
for MODEL in "${BACKBONES[@]}"; do
  TR=$(trials_for "$MODEL")
  TAG=$(short "$MODEL")
  WANT=$(( 6 * (TR + 15) ))            # 6 arms x (tune + confirm)
  for TASK in "${TASKS[@]}"; do
    OUT="results/campaign_t1_${TAG}_${TASK}.csv"
    HAVE=0
    [ -f "$OUT" ] && HAVE=$(( $(wc -l < "$OUT") - 1 ))
    if [ "$HAVE" -ge "$WANT" ]; then
        say "SKIP $TAG/$TASK ($HAVE/$WANT rows)"
        continue
    fi
    say "START $TAG/$TASK trials=$TR ($HAVE/$WANT rows) -> $OUT"
    if [ "$DRY" = 1 ]; then
        say "  dry-run: $PY campaign.py --target glue --task $TASK --model $MODEL"
        say "           --arms $ARMS --trials $TR --wide --stage both"
        continue
    fi
    "$PY" -u campaign.py --target glue --task "$TASK" --model "$MODEL" \
        --arms $ARMS --trials "$TR" --wide --stage both \
        --confirm_seeds $SEEDS --out "$OUT" \
        > "logs/t1_${TAG}_${TASK}.log" 2>&1 \
        && say "DONE $TAG/$TASK" || { say "FAILED $TAG/$TASK"; continue; }

    # reclaim this task's cache before the next one, unless it is shared with a
    # campaign that already exists (cola/stsb/mrpc on the encoders).
    if [ "$KEEP_CACHE" = 0 ]; then
        case "$TAG/$TASK" in
            bert/cola|bert/stsb|bert/mrpc|rob/cola|rob/stsb|tl/cola|tl/stsb)
                say "  keeping $TAG/$TASK cache (shared with an earlier campaign)" ;;
            *)
                SZ=$(du -sc $(cache_glob "$MODEL" "$TASK") 2>/dev/null | tail -1 | cut -f1)
                rm -rf $(cache_glob "$MODEL" "$TASK")
                say "  freed $(( ${SZ:-0} / 1048576 )) GB ; $(df -h /home | tail -1 | awk '{print $4}') now free" ;;
        esac
    fi
  done
done

say "===== TABLE 1 QUEUE FINISHED ====="
say "Dropped by design: qqp (dev set 40,430 pairs, most expensive in GLUE),"
say "  rte (MDE 2.06 > every effect in the paper), wnli (635 train, degenerate)."
