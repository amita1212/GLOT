#!/usr/bin/env bash
# Master queue: runs after the robfill (RoBERTa) campaign drains.
#
# Design notes:
#  * ONE GPU, so items run sequentially. Order is value-per-hour.
#  * Every item is guarded by an output-exists check, so re-launching this
#    script after a crash or reboot cannot double-run a finished item.
#  * Caches are NOT rebuilt anywhere here. Every item below reuses an existing
#    warm cache. A cold cache consumes the global RNG through the shuffled
#    loader and shifts the same seed by several MCC (paper S3), so a prewarm
#    pass mid-queue would silently break comparability with what already ran.
#  * Each item logs to its own file so a failure is attributable.
set -u
cd /home/t-amitalfasi/glot || exit 1
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p logs results

say() { echo "[queue $(date -Is)] $*"; }

# ---- wait for the GPU ------------------------------------------------------
say "waiting for robfill / any campaign.py to finish..."
while pgrep -f 'campaign.py' >/dev/null 2>&1; do sleep 120; done
say "GPU free."

# ---- 1. RoBERTa scale-fix cells (~<1 h) ------------------------------------
# Fills the two \PEND cells in the RoBERTa block of tab:fix (q=0.05 x
# {rms, median}). n=5 matches the rest of that table.
OUT=results/factorial_scale_roberta.csv
if [ -s "$OUT" ]; then
    say "SKIP scale-fix roberta (exists)"
else
    say "START scale-fix roberta"
    "$PY" -u factorial_scale.py --task cola \
        --backbones roberta_final \
        --seeds 1 2 3 4 5 \
        --out "$OUT" > logs/q1_scale_roberta.log 2>&1 \
        && say "DONE scale-fix roberta" || say "FAILED scale-fix roberta"
fi

# ---- 2. MTEB with a pooler actually trained on MS MARCO (~5 h) --------------
# Replaces the untrained-pooler block of tab:docmteb. Pipeline validated by the
# smoke test: 2-stage train-then-eval, checkpoint written and reloaded.
OUT=results/mteb_trained.csv
if [ -s "$OUT" ]; then
    say "SKIP mteb-trained (exists)"
else
    say "START mteb-trained"
    "$PY" -u run_all_experiments.py --with_mteb \
        --models bert-base-uncased \
        --configs baseline A C AC \
        --seeds 1 2 3 4 5 \
        --mteb_tasks Banking77Classification STS12 STS13 SciFact ArguAna \
                     TwentyNewsgroupsClustering SprintDuplicateQuestions \
        --mteb_ckpt_dir checkpoints_mteb \
        --results_csv "$OUT" > logs/q2_mteb_trained.log 2>&1 \
        && say "DONE mteb-trained" || say "FAILED mteb-trained"
fi

# ---- 3. Decoder STS-B, arms B and C ALONE (~11 h) --------------------------
# Fills the \PEND rows of app:decoder. The paper currently claims a decoder
# replication of Stage B without ever having run B by itself on a decoder.
# MODEL is read from the existing decoder script so this cannot drift from the
# backbone the rest of that campaign used.
# Backbone pinned to the literal value verified in decoder_sweep.sh line 53, so
# these two arms cannot drift from the six already in app:decoder.
OUT=results/campaign_decoder_stsb_BC.csv
DEC_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
if ! grep -q "$DEC_MODEL" decoder_sweep.sh; then
    say "FAILED decoder-stsb-BC: $DEC_MODEL no longer matches decoder_sweep.sh"
elif [ -s "$OUT" ]; then
    say "SKIP decoder-stsb-BC (exists)"
else
    say "START decoder-stsb-BC model=$DEC_MODEL"
    "$PY" -u campaign.py --target glue --task stsb --model "$DEC_MODEL" \
        --arms B C --trials 10 --stage both \
        --confirm_seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
        --out "$OUT" > logs/q3_decoder_stsb_BC.log 2>&1 \
        && say "DONE decoder-stsb-BC" || say "FAILED decoder-stsb-BC"
fi

say "===== QUEUE FINISHED ====="
say "Still NOT queued (need code first, see chat): Stage C factorial at n=65"
say "  (factorial_geom.py hardcodes its seeds), Stage A CoLA n=50, SST-2, IMDB."
