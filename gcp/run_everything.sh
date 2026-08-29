#!/usr/bin/env bash
# HyperGLOT master pipeline -- runs every experiment we need, unattended.
#
# Ordered by scientific value, so a spot preemption loses the least. Every
# stage is resumable (each script skips rows already present in its CSV), so
# re-running this script simply continues where it stopped.
#
# Stage 0  pre-warm caches            ~10 min   (MANDATORY -- see cache confound)
# Stage 1  fair A/B/C ablation        ~5 h      answers "do our methods work?"
# Stage 2  diagnostic stress test     ~40 min   the paper's headline claim
# Stage 3  paper Table 6 sweep, CoLA  ~3 h      reproduction verdict + tuned baseline
# Stage 4  paper Table 6 sweep, RTE   ~2 h
# Stage 5  paper Table 6 sweep, STS-B ~5 h
#
# Usage:  cd ~/glot && nohup bash run_everything.sh > logs_pipeline.txt 2>&1 &
set -uo pipefail            # NOT -e: one failing stage must not kill the rest
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

banner() {
    echo
    echo "################################################################"
    echo "# $* "
    echo "# $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "################################################################"
}

banner "STAGE 0: pre-warm hidden-state caches"
bash prewarm_caches.sh cola stsb rte

banner "STAGE 1: fair A/B/C ablation (tune each arm, then 3 seeds)"
"$PY" ablation_fair.py \
    --tasks cola stsb rte \
    --seeds 42 1 2 \
    --out results/ablation_fair.csv \
    2>&1 | tee -a logs_ablation_fair.txt

banner "STAGE 2: diagnostic stress test (paper's headline claim)"
for arm in "cosine threshold 0 0 baseline" \
           "poincare threshold 0 0 A_thresh" \
           "poincare knn 0 0 A_knn"; do
    set -- $arm
    gm=$1; ga=$2; hg=$3; hr=$4; name=$5
    for ratio in 0.2 0.5 0.8 0.9; do
        echo "--- stress $name ratio=$ratio ---"
        "$PY" hyperglot/diagnostic_stress_test.py \
            --model_name_or_path=bert-base-uncased \
            --decoder_cls_last_token=0 \
            --distractor_ratio=$ratio \
            --epochs=3 --batch_size=32 --eval_batch_size=32 \
            --gat_hidden_dim=64 --scorer_hidden=256 --num_layers=4 \
            --tau=0.6 --lr=1e-4 --seed=0 \
            --pooling_method=glot \
            --graph_metric=$gm --graph_adj=$ga \
            --hyperbolic_gnn=$hg --hyperbolic_readout=$hr \
            --arm=$name \
            --results_csv=results/stress_warm.csv \
            --run_tag=stress_fixed \
            2>&1 | tail -5
    done
done

banner "STAGE 3: paper Table 6 grid sweep -- CoLA"
"$PY" sweep_paper_grid.py --task cola --n_trials 100 --noise_std 0.81 \
    --resume --out results/sweep_cola.csv 2>&1 | tee -a logs_sweep_cola.txt

banner "STAGE 4: paper Table 6 grid sweep -- RTE"
"$PY" sweep_paper_grid.py --task rte --n_trials 100 --noise_std 0.81 \
    --resume --out results/sweep_rte.csv 2>&1 | tee -a logs_sweep_rte.txt

banner "STAGE 5: paper Table 6 grid sweep -- STS-B"
"$PY" sweep_paper_grid.py --task stsb --n_trials 100 --noise_std 0.81 \
    --resume --out results/sweep_stsb.csv 2>&1 | tee -a logs_sweep_stsb.txt

banner "PIPELINE COMPLETE"
ls -la results/
