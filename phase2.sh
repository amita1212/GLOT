#!/usr/bin/env bash
# Phase 2 of the campaign: everything still missing for the paper.
#
# Runs a SMOKE pass first (one config per stream) because a failure discovered
# eight hours in is eight hours lost. Smoke runs are not recorded -- they exist
# to prove the flags are accepted and to warm the feature cache, since cold-vs-
# warm cache is a known confound in this codebase.
#
# Three streams then run concurrently. The MRPC/RTE chain already occupies 4
# workers on 8 vCPUs; 3 more takes the machine to ~7, which is the most it can
# absorb without the existing runs slowing down.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
mkdir -p logs results

log () { echo "[phase2 $(date -Is)] $*"; }

log "===== SMOKE ====="

"$PY" factorial_scale.py  --smoke > logs/smoke_factorial.log 2>&1; RC_F=$?
log "factorial smoke rc=$RC_F"; tail -6 logs/smoke_factorial.log

"$PY" stress_poolers.py   --smoke --out results/_smoke_poolers.csv \
                                  > logs/smoke_poolers.log 2>&1; RC_P=$?
log "poolers smoke rc=$RC_P"; tail -12 logs/smoke_poolers.log

"$PY" backbone_recipe.py  --smoke > logs/smoke_bkrec.log 2>&1; RC_B=$?
log "backbone/recipe smoke rc=$RC_B"; tail -8 logs/smoke_bkrec.log

rm -f results/_smoke_poolers.csv

log "===== LAUNCH ====="

if [ "$RC_F" -eq 0 ]; then
    nohup "$PY" -u factorial_scale.py --task cola --seeds 1 2 3 4 5 \
        > logs/factorial.log 2>&1 &
    log "factorial pid=$!"
else
    log "factorial SKIPPED (smoke failed) -- see logs/smoke_factorial.log"
fi

if [ "$RC_P" -eq 0 ]; then
    nohup "$PY" -u stress_poolers.py > logs/poolers.log 2>&1 &
    log "poolers pid=$!"
else
    log "poolers SKIPPED (smoke failed) -- see logs/smoke_poolers.log"
fi

if [ "$RC_B" -eq 0 ]; then
    nohup "$PY" -u backbone_recipe.py --seeds 1 2 3 4 5 \
        > logs/bkrec.log 2>&1 &
    log "backbone/recipe pid=$!"
else
    log "backbone/recipe SKIPPED (smoke failed) -- see logs/smoke_bkrec.log"
fi

wait
log "===== ALL PHASE2 DONE ====="
