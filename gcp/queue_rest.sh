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

# Items 2 and 3 resume into a PARTIALLY complete CSV, so `[ -s $OUT ]` is the
# wrong guard for them -- it would skip work that is only a third done. Both
# underlying scripts skip run_keys they already have, so re-running is always
# safe; these counters just avoid a pointless pass once an item is finished.
nrows() {
    [ -f "$1" ] || { echo 0; return; }
    "$PY" -c "import csv,sys;print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))" "$1" 2>/dev/null || echo 0
}
nconfirm() {
    [ -f "$1" ] || { echo 0; return; }
    "$PY" -c "import csv,sys
f,arms=sys.argv[1],set(sys.argv[2:])
print(sum(1 for r in csv.DictReader(open(f)) if r.get('stage')=='confirm' and r.get('arm') in arms))" "$@" 2>/dev/null || echo 0
}

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

# ---- 2. Stage C factorial at n=65 (~8 h) -----------------------------------
# THE HIGHEST-VALUE ITEM, so it runs before the additive ones. At n=15 the
# +1.42 MCC decomposes into geometry (+0.40), configuration (-0.01) and
# interaction (+1.03) with NONE of the three significant; ~65 seeds resolves
# the interaction at 80% power. If it lands on configuration rather than
# curvature, the paper's one positive result stops being about geometry.
#
# All FOUR cells run here, not just the two off-diagonal ones. The n=15 version
# borrowed baseline and C from results/campaign_wide_cola.csv, which only has
# seeds 1-15 and is a different campaign context. Seeds 1-15 of the diagonal
# cells therefore double as a reproducibility check: base_at_base should return
# 45.368 and C_at_C 46.784, and drift there means the cache state or the code
# moved under us. Configs were verified against the campaign CSV before
# queueing (_verify_factorial_cfg.py: exact match, both means reproduced).
OUT=results/factorial_geom_cola.csv
WANT=260                                   # 4 cells x 65 seeds
HAVE=$(nrows "$OUT")
if [ "$HAVE" -ge "$WANT" ]; then
    say "SKIP factorial-n65 ($HAVE/$WANT rows)"
else
    say "START factorial-n65 ($HAVE/$WANT rows)"
    "$PY" -u factorial_geom.py \
        --arms base_at_base C_at_base base_at_C C_at_C \
        --seeds $(seq 1 65) \
        --out "$OUT" > logs/q2_factorial_n65.log 2>&1 \
        && say "DONE factorial-n65" || say "FAILED factorial-n65"
fi

# ---- 3. Stage A on CoLA at n=50 (~2.5 h) -----------------------------------
# Settles the one stage the paper openly leaves unresolved: A is nominally
# +0.70 on CoLA at p=0.118, and ahead in 5 of 7 settings with a sign test of
# p=0.45 across them. n=50 turns "unresolved" into a positive or a null. It is
# powered in advance, so whichever way it goes is the answer -- re-running at a
# different n after seeing it would be the practice the paper criticises.
#
# campaign.py --stage confirm re-reads the TUNE rows from --out, picks the same
# best config per arm and replays it at --confirm_seeds, skipping run_keys it
# already has. So it needs the campaign file -- but writing into
# results/campaign_wide_cola.csv would leave baseline and A at n=50 while the
# other seven arms sit at n=15, and any analysis reading that file would
# silently mix sample sizes (the exact defect already found in the merged
# RoBERTa files). Work on a COPY; the released log stays untouched.
SRC=results/campaign_wide_cola.csv
OUT=results/stageA_n50_cola.csv
WANT=100                                   # 2 arms x 50 confirm seeds
if [ ! -f "$OUT" ]; then
    cp "$SRC" "$OUT" && say "seeded $OUT from $SRC ($(nrows "$OUT") rows)"
fi
HAVE=$(nconfirm "$OUT" baseline A)
if [ "$HAVE" -ge "$WANT" ]; then
    say "SKIP stageA-n50 ($HAVE/$WANT confirm rows for baseline+A)"
else
    say "START stageA-n50 ($HAVE/$WANT confirm rows for baseline+A)"
    "$PY" -u campaign.py --target glue --task cola --model bert-base-uncased \
        --wide --arms baseline A --stage confirm \
        --confirm_seeds $(seq 1 50) \
        --out "$OUT" > logs/q3_stageA_n50.log 2>&1 \
        && say "DONE stageA-n50" || say "FAILED stageA-n50"
fi

# ---- 4. MTEB with a pooler actually trained on MS MARCO (~5 h) --------------
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
        --results_csv "$OUT" > logs/q4_mteb_trained.log 2>&1 \
        && say "DONE mteb-trained" || say "FAILED mteb-trained"
fi

# ---- 5. Decoder STS-B, arms B and C ALONE (~11 h) --------------------------
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
        --out "$OUT" > logs/q5_decoder_stsb_BC.log 2>&1 \
        && say "DONE decoder-stsb-BC" || say "FAILED decoder-stsb-BC"
fi

say "===== QUEUE FINISHED ====="
say "Still NOT queued: SST-2 reduced design and IMDB. Neither has a script,"
say "  both need a cache built first (IMDB's is 79 GB), and SST-2 additionally"
say "  needs CoLA-selected configs replayed on a task nothing was tuned on."
