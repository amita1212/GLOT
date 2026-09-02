#!/usr/bin/env bash
# Complete Table 1: give RoBERTa and TinyLlama the four tasks BERT already has.
#
# THE GAP (everything else in Table 1 is already done)
#   RoBERTa    MRPC, RTE            40 trials/arm   BERT and RoBERTa's existing
#                                                    tasks both used 40
#   TinyLlama  CoLA, MRPC, RTE      10 trials/arm   TinyLlama's existing STS-B
#                                                    used 10; raising this to 40
#                                                    would make its own block
#                                                    unequal-budget and force a
#                                                    re-run of STS-B
#
# ARMS: the nine BERT has -- baseline A B C AB AC BC ABC no_graph.
#
# NOT HERE, on purpose:
#   * TinyLlama STS-B arms B and C alone are running on hyperglot-l4 as queue
#     item 5. Re-running them here would duplicate ~11 h and, worse, produce a
#     SECOND draw of the same cell; see app:determinism on why two draws of one
#     configuration are not interchangeable.
#   * The RoBERTa calibration arms (published_tau, quantile_tau) belong to
#     tab:roberta, not Table 1.
#
# Resumable: campaign.py dedups on run_key and each task is skipped once its
# CSV holds the expected row count, so a preemption costs only the in-flight run.
#
# usage:  bash queue_table1_gap.sh [--dry-run]
set -u

cd "$(dirname "$0")" || exit 1
PY="$HOME/glotenv/bin/python"
LOG=logs/queue_table1_gap.log
mkdir -p logs results data

ARMS="baseline A B C AB AC BC ABC no_graph"
NARMS=9
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
ROB="roberta-base"
TL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

say() { echo "[t1gap $(date -Is)] $*" | tee -a "$LOG"; }

# cheapest first, so a preemption loses the least and the encoder results --
# which are the ones that extend an existing 40-trial block -- land first.
#         model   tag   task  trials
JOBS="
$ROB rob  rte   40
$ROB rob  mrpc  40
$TL  tl   rte   10
$TL  tl   cola  10
$TL  tl   mrpc  10
"

if [ "$DRY" = 1 ]; then
    say "DRY RUN -- nothing is launched"
    echo "$JOBS" | while read -r MODEL TAG TASK TR; do
        [ -z "${MODEL:-}" ] && continue
        say "would run $TAG/$TASK trials=$TR runs=$(( NARMS * (TR + 15) ))"
    done
    exit 0
fi

# NOTE: there is deliberately NO "wait for the GPU to be free" loop here.
#
# Two attempts at one both failed the same way, and the failure is instructive.
# Any pgrep/ps pattern that names a process is also matched by ANY OTHER command
# line that merely MENTIONS it -- including a monitoring ssh command that greps
# for exactly that string to report status. The first version matched
# 'campaign\.py' and waited on an ssh wrapper; the second matched
# 'glotenv/bin/python' and waited on my own status check, spinning for 90
# minutes with an idle GPU while the log cheerfully said "waiting for another
# campaign to finish".
#
# This machine is dedicated to this queue, so the guard bought nothing. What
# actually protects correctness is the per-cell cache check below, which tests
# for a FILE rather than a process and so cannot be fooled by an observer.
say "starting. arms=[$ARMS]"

echo "$JOBS" | while read -r MODEL TAG TASK TR; do
    [ -z "${MODEL:-}" ] && continue
    OUT="results/campaign_t1_${TAG}_${TASK}.csv"
    WANT=$(( NARMS * (TR + 15) ))
    HAVE=0
    [ -f "$OUT" ] && HAVE=$(( $(wc -l < "$OUT") - 1 ))
    if [ "$HAVE" -ge "$WANT" ]; then
        say "SKIP $TAG/$TASK ($HAVE/$WANT rows)"
        continue
    fi

    # REFUSE to run without a warm cache. A cold cache advances the global torch
    # RNG (precompute_hidden_states early-returns when warm), so the FIRST arm
    # would get a different init and batch order from the other eight -- worth
    # ~5 MCC on CoLA, far larger than anything we are measuring. Silently
    # running cold is the single worst failure mode for this queue, so it is a
    # hard stop rather than a warning.
    MTAG=$(echo "$MODEL" | tr '/' '_')
    for SPLIT in train val; do
        D="data/${MTAG}_${TASK}_${SPLIT}_batches"
        if [ ! -f "$D/metadata.json" ]; then
            say "ABORT $TAG/$TASK -- cache missing: $D"
            say "      run 'bash prewarm_t1.sh' first; NOT running this cell cold"
            continue 2
        fi
    done

    say "START $TAG/$TASK model=$MODEL trials=$TR ($HAVE/$WANT rows)"
    "$PY" -u campaign.py --target glue --task "$TASK" --model "$MODEL" \
        --arms $ARMS --trials "$TR" --wide --stage both \
        --confirm_seeds $SEEDS --out "$OUT" \
        > "logs/t1_${TAG}_${TASK}.log" 2>&1 \
        && say "DONE $TAG/$TASK" || say "FAILED $TAG/$TASK (see logs/t1_${TAG}_${TASK}.log)"
done

say "===== TABLE 1 GAP FINISHED ====="
say "Merge into the paper with:  python _revalidate.py"
