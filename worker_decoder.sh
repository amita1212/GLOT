#!/usr/bin/env bash
# One TASK of the decoder (TinyLlama) sweep, run start-to-finish on ONE machine.
#
#   usage: worker_decoder.sh <task> ["<arms>"]
#   e.g.   worker_decoder.sh cola
#          worker_decoder.sh stsb "B C"
#
# WHY ONE TASK PER MACHINE, NOT ONE ARM PER MACHINE
#   Every number we report is a PAIRED difference against the baseline on
#   shared seeds. If the baseline runs on machine 1 and the arm on machine 2,
#   any hardware/driver difference lands inside the delta. Cache-warming order
#   alone is worth ~5 MCC on CoLA -- six times the seed sd -- so this is not a
#   theoretical concern. Keeping a whole task (baseline + all its arms) on one
#   machine makes every comparison within-machine by construction.
#
# WHY --trials 10 AND NOT 40  <-- THE IMPORTANT ONE
#   The BERT campaigns used --trials 40 --wide. The DECODER campaign did not:
#   decoder_sweep.sh ran `--trials 10 --fine_baseline` with no --wide, because
#   a TinyLlama run costs ~4x a BERT run. The already-released decoder baseline
#   (STS-B 79.95) was therefore selected from 10 narrow trials.
#   Giving new arms 40 wide trials against that baseline would hand them
#   exactly the budget advantage that forced our Stage A retraction, only
#   pointed the other way: it would MANUFACTURE a positive. Equal budget or
#   nothing. Do not "improve" these flags.
#
# WHAT IS MISSING AND WHY
#   decoder STS-B has baseline, A, AB, AC, BC, ABC confirmed -- but NOT B and
#   NOT C standalone; decoder_sweep.sh dropped them for cost. So the paper
#   currently says "B is harmful on a decoder" with no B-alone decoder run, and
#   Stage C -- its only positive result -- has never been run on a second
#   backbone at all. Those two arms are the cheapest real gap in the paper.
#   decoder CoLA got 31 tuning rows and zero confirmation rows before it was
#   cut off, so it starts from scratch.
set -u
# Resolve the repo root from this script's own location -- this runs on other
# people's machines, where a hardcoded /home/<user> path does not exist.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/glotenv/bin/python}"
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs results

TASK="${1:?task, e.g. cola or stsb}"
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SLUG="TinyLlama_TinyLlama-1.1B-Chat-v1.0"
# no_graph is deliberately absent: it is not reported in the paper.
ARMS="${2:-baseline A B C AB AC BC ABC}"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
TRIALS=10
TAG="dec2"

echo "[$TAG/$TASK] arms: $ARMS"
echo "[$TAG/$TASK] start $(date -Is)"

# PREWARM FIRST, ALWAYS. On a cache miss the shuffled loader consumes the
# global RNG before the classifier is built, so the same seed gives a different
# initialisation cold vs warm. Whichever arm ran first would be handicapped.
echo "[$TAG/$TASK] pre-warming cache"
bash gcp/prewarm_model.sh "$MODEL" -1 "$TASK" > "logs/${TAG}_prewarm_${TASK}.log" 2>&1
if ! ls -d data/${SLUG}_*batches >/dev/null 2>&1; then
    echo "[$TAG/$TASK] PREWARM FAILED -- refusing to run"
    tail -20 "logs/${TAG}_prewarm_${TASK}.log"
    exit 1
fi
df -h /home | tail -1

"$PY" campaign.py --target glue --task "$TASK" --model "$MODEL" \
    --arms $ARMS --trials "$TRIALS" --stage both \
    --fine_baseline --confirm_seeds $SEEDS \
    --out "results/campaign_${TAG}_${TASK}.csv" \
    >> "logs/${TAG}_${TASK}.log" 2>&1
echo "[$TAG/$TASK] campaign rc=$? $(date -Is)"

"$PY" paired_analysis.py "results/campaign_${TAG}_${TASK}.csv" 2>&1 | head -25
echo "[$TAG/$TASK] FINISHED $(date -Is)"
