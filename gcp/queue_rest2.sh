#!/usr/bin/env bash
# Master queue, REVISION 2 -- takes over from queue_rest.sh.
#
# WHY A NEW FILE. queue_rest.sh is running. bash reads a script incrementally
# by byte offset, so editing it in place shifts the offsets under the running
# interpreter and makes it resume mid-token. The only safe way to change an
# item that has not executed yet is to stop that queue and start a corrected
# one. Every item below is guarded, so the corrected queue re-derives what is
# already finished from the output files and resumes rather than repeating.
#
# WHAT CHANGED vs queue_rest.sh: item 4 (MTEB) only. Items 1, 2, 3 and 5 are
# byte-identical, and 1-3 are already satisfied by their guards.
#
# Item 4 had FOUR defects, all confirmed by dry-running the driver with its
# exact arguments (_dry_item4.sh) rather than by reading it:
#
#  (1) WRONG PATH. It ran `run_all_experiments.py` from ~/glot, where no such
#      file exists -- the driver lives in hyperglot_new/, a separate clone of
#      the hyperglot-stageA branch, kept apart so running campaigns keep the
#      main.py they started with. It would have died on file-not-found.
#
#  (2) RUNAWAY SCOPE. It omitted `--tasks`, and the driver resolves
#      tasks = args.tasks if args.tasks is not None else (GLUE_TASKS + DOC_TASKS)
#      so it would ALSO have trained the whole GLUE grid plus IMDB: cola sst2
#      stsb mrpc qqp mnli qnli rte wnli imdb -- including QQP and MNLI, which
#      this paper estimates at 38 and 41 DAYS. The smoke test passes a bare
#      `--tasks` (empty list) to select MTEB only; so do we.
#
#  (3) ONE ARM, LABELLED AS FOUR. It passed `--configs baseline A C AC`. Those
#      arm names do not exist -- build_configs() names them baseline,
#      A_threshold, C_threshold, AC_threshold. This clone filters unknown names
#      SILENTLY, so the dry run shows exactly one arm surviving: baseline.
#
#  (4) NO TRAINING DATA. It omitted `--mteb_train_file`, whose default is
#      os.path.join(HERE, "data", ...) with HERE = the DRIVER's directory. That
#      resolves to hyperglot_new/data/msmarco-triplets.jsonl, which does not
#      exist; the corpus is at ~/glot/data/. The MS MARCO stage would have
#      written no checkpoint, and the driver would then refuse to evaluate.
#
#  (5) DESIGN. The task list was the original paper's APPENDIX Table 12 set,
#      but the comparison we want is its main Table 3. Only STS12 and SciFact
#      are common to both. All twelve names below were resolved against the
#      installed mteb 2.1.7 registry (_verify_mteb_alias.py): every one exists
#      exactly and none is silently aliased to a .v2 revision.
#
#      We run the UNION of the two sets (12 unique tasks) rather than replacing
#      one with the other. Evaluation is the cheap half of this item -- the MS
#      MARCO stage dominates -- and the union lets the trained-pooler numbers
#      replace BOTH the existing tab:docmteb rows and build the Table 3
#      comparison from one job instead of two. Stage 2 skips per
#      (model, task, cfg, seed) key, so the extra tasks resume independently.
#
# Table 3 metrics are NOT commensurable -- accuracy, ndcg@10, v_measure,
# map@1000, cosine_spearman and max_ap across six task types -- so whatever
# table is built from this must not carry a row average.
#
#  * ONE GPU, so items run sequentially. Order is value-per-hour.
#  * Caches are NOT rebuilt anywhere here. A cold cache consumes the global RNG
#    through the shuffled loader and shifts the same seed by several MCC.
set -u
cd /home/t-amitalfasi/glot || exit 1
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p logs results

say() { echo "[queue2 $(date -Is)] $*"; }

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
# Count only the mteb rows. The embedding (MS MARCO) rows are not rewritten
# once a checkpoint exists, so a total-row target would never be reached on a
# resume and the item would re-enter every time.
nmteb() {
    [ -f "$1" ] || { echo 0; return; }
    "$PY" -c "import csv,sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r.get('task')=='mteb'))" "$1" 2>/dev/null || echo 0
}

# ---- wait for the GPU ------------------------------------------------------
# Stage A (item 3) is still running as a campaign.py process, orphaned from the
# queue we replaced. Wait for it exactly as the original queue would have.
say "waiting for any campaign.py to finish..."
while pgrep -f 'campaign[.]py' >/dev/null 2>&1; do sleep 120; done
say "GPU free."

# ---- 1. RoBERTa scale-fix cells (~<1 h) ------------------------------------
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

# ---- 4. MTEB with a pooler actually trained on MS MARCO (~9 h) --------------
# Replaces the untrained-pooler block of tab:docmteb AND supplies the original
# paper's Table 3 task list. See the header for the five defects this corrects.
#
# Guard counts mteb rows, not `[ -s ]`: this item resumes into a partially
# complete CSV (Stage 2 skips per model/task/cfg/seed key, Stage 1 reuses any
# checkpoint on disk), and `[ -s ]` would skip work that is 1/12 done.
#   4 configs x 5 seeds x 12 tasks = 240 mteb rows (+20 embedding rows).
ROOT=/home/t-amitalfasi/glot
NEW=$ROOT/hyperglot_new
OUT=$ROOT/results/mteb_trained.csv
WANT=240
MTEB_TASKS="EmotionClassification SciFact RedditClustering AskUbuntuDupQuestions STS12 TwitterSemEval2015 SummEval Banking77Classification STS13 ArguAna TwentyNewsgroupsClustering SprintDuplicateQuestions"

HAVE=$(nmteb "$OUT")
if [ "$HAVE" -ge "$WANT" ]; then
    say "SKIP mteb-trained ($HAVE/$WANT mteb rows)"
else
    # Refresh the clone so it carries the fail-fast abort on unknown config
    # names. Best effort: if the fetch fails we still run, because the names
    # below are spelled correctly and the pre-flight check is what actually
    # protects us.
    if git -C "$NEW" fetch --depth 1 origin hyperglot-stageA >/dev/null 2>&1 &&
       git -C "$NEW" reset --hard FETCH_HEAD >/dev/null 2>&1; then
        say "clone refreshed to $(git -C "$NEW" log --oneline -1)"
    else
        say "clone refresh skipped (offline?); continuing"
    fi

    # PRE-FLIGHT. Dry-run the exact command and assert it expands to what we
    # intend. This is the guard that would have caught every defect above: it
    # fails if any arm was silently dropped, or if a GLUE/IMDB task leaked into
    # the plan. Costs a second, and cannot start a 40-day job by accident.
    DRY=/tmp/q4_preflight.log
    CUDA_VISIBLE_DEVICES= "$PY" -u "$NEW/run_all_experiments.py" --with_mteb --dry_run \
        --models bert-base-uncased \
        --configs baseline A_threshold C_threshold AC_threshold \
        --seeds 1 2 3 4 5 \
        --mteb_tasks $MTEB_TASKS \
        --mteb_train_file "$ROOT/data/msmarco-triplets.jsonl" \
        --mteb_ckpt_dir "$ROOT/checkpoints_mteb" \
        --results_csv "$OUT" \
        --tasks > "$DRY" 2>&1
    N_ARMS=$(grep -o -- '--arm=[A-Za-z0-9_]*' "$DRY" | sort -u | wc -l)
    N_LEAK=$(grep -o -- '--task=[a-z0-9]*' "$DRY" | sort -u \
             | grep -c -v -e '--task=mteb' -e '--task=embedding')
    N_CMD=$(grep -c 'main.py' "$DRY")
    if [ "$N_ARMS" -ne 4 ] || [ "$N_LEAK" -ne 0 ]; then
        say "FAILED mteb-trained PRE-FLIGHT: arms=$N_ARMS (want 4), non-mteb tasks=$N_LEAK (want 0)"
        say "  see $DRY -- refusing to launch"
    else
        say "pre-flight OK: 4 arms, no GLUE/IMDB leakage, $N_CMD commands"
        say "START mteb-trained ($HAVE/$WANT mteb rows)"
        "$PY" -u "$NEW/run_all_experiments.py" --with_mteb \
            --models bert-base-uncased \
            --configs baseline A_threshold C_threshold AC_threshold \
            --seeds 1 2 3 4 5 \
            --mteb_tasks $MTEB_TASKS \
            --mteb_train_file "$ROOT/data/msmarco-triplets.jsonl" \
            --mteb_ckpt_dir "$ROOT/checkpoints_mteb" \
            --results_csv "$OUT" \
            --tasks > logs/q4_mteb_trained.log 2>&1 \
            && say "DONE mteb-trained" || say "FAILED mteb-trained"
    fi
fi

# ---- 5. Decoder STS-B, arms B and C ALONE (~11 h) --------------------------
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
