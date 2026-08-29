#!/usr/bin/env bash
# FULL arm sweep + hyperparameter search on a DECODER backbone.
#
# BACKBONE CHOICE (measured, not arbitrary)
#   TinyLlama-1.1B-Chat-v1.0, one of the paper's six Table 1 backbones.
#   Screened before launch (screen_decoders.sh):
#       backbone            cos p50   density@tau=0.6   norm mean/med
#       TinyLlama-1.1B       0.349        0.121             0.99
#       SmolLM2-360M         0.736        0.731             1.01
#       BERT (reference)     0.401        0.149             1.02
#   TinyLlama sits in BERT's density regime, so tau=0.6 is properly calibrated
#   and no input-scale fix is needed. SmolLM2 would have repeated the
#   near-complete-graph failure already seen on RoBERTa (0.992) and
#   ModernBERT (0.9957), and any sweep there would measure nothing.
#
#   Backbone is FROZEN and hidden states are precomputed, so 1.1B is only ever
#   forward-passed: an L4 is ample. Cache is ~9 GB per split at hidden=2048
#   (176 GB free at launch).
#
# ARMS: every combination requested, plus the two controls that make them
# interpretable.
#   baseline   cosine graph = GLOT as published ("dense")
#   no_graph   tau=0.999, self-loops only -- the graph removed entirely. A past
#              "Stage A wins" headline turned out to be "removing the graph
#              wins", so no arm counts until it beats this too.
#   A AB AC BC ABC     the requested hyperbolic combinations
#
#   Standalone B and C are DROPPED. On TinyLlama a single run costs 935 s
#   (vs 201 s for the same task on BERT) because hidden dim goes 768 -> 2048,
#   so the GAT layers and cache I/O both scale with it. At 9 arms x 2 tasks the
#   sweep is 4.8 days on a preemptible VM; at 7 arms it is 3.7 days with the
#   full 10-trial / 15-seed protocol intact. Trading arms is cheap, trading
#   seeds is not -- 15 seeds is what made the STS-B result trustworthy and what
#   exposed the MRPC false positive.
#
# PROTOCOL (each item is a lesson already paid for in this project)
#   --trials 10        equal budget per arm (Bergstra & Bengio)
#   --fine_baseline    the cosine arm has ONE knob and would otherwise exhaust
#                      its grid at 5 while hyperbolic arms drew 10
#   --confirm_seeds 1..15   n=3 gave a minimum detectable effect of 2.90 on CoLA
#                      and produced an outright false positive on MRPC
#   prewarm first      a cold cache consumes torch.randperm via the shuffled
#                      loader and shifts classifier init (~5 MCC on CoLA);
#                      whichever arm ran first would be silently handicapped
#   paired_analysis    arms share seeds, so per-seed differencing cancels the
#                      dominant variance term (5x tighter SE on STS-B)
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SLUG="TinyLlama_TinyLlama-1.1B-Chat-v1.0"
ARMS="baseline no_graph A AB AC BC ABC"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
# STS-B first: it is the ONLY task where Stage A shows a real effect on BERT
# (+0.223 Spearman, 15/15 seeds, p=0.00006), so it is the one where "does this
# transfer to a decoder?" is a meaningful question rather than a fishing trip.
TASKS="${1:-stsb cola}"

# DEC_NOWAIT=1 skips the queue wait. Set when the decoder is deliberately run
# FIRST and the other campaigns have already been stopped.
if [ "${DEC_NOWAIT:-0}" = "1" ]; then
    echo "[dec] DEC_NOWAIT=1 -- not waiting, decoder has priority"
else
    echo "[dec] waiting for the GPU..."
    while pgrep -f 'structural_arms.sh|roberta_compare.sh' >/dev/null 2>&1; do
        sleep 60
    done
    echo "[dec] GPU free at $(date -Is)"
fi

for task in $TASKS; do
    echo "[dec] pre-warming cache: $MODEL / $task"
    bash prewarm_model.sh "$MODEL" -1 "$task" > /dev/null 2>&1
    if ! ls -d data/${SLUG}_*batches >/dev/null 2>&1; then
        echo "[dec] PREWARM FAILED for $task -- see logs/prewarm_${SLUG}_L-1_${task}.log"
        tail -12 "logs/prewarm_${SLUG}_L-1_${task}.log" 2>/dev/null
        continue
    fi
    df -h /home | tail -1

    echo "[dec] ===== $task : $ARMS ===== $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model "$MODEL" \
        --arms $ARMS --trials 10 --stage both \
        --fine_baseline --confirm_seeds $SEEDS \
        --out "results/campaign_decoder_${task}.csv" \
        >> "logs/campaign_decoder_${task}.log" 2>&1

    echo "[dec] --- paired result: $task ---"
    "$PY" paired_analysis.py "results/campaign_decoder_${task}.csv" 2>&1 | head -18
    "$PY" analyze_campaign.py "results/campaign_decoder_${task}.csv" \
        > "results/campaign_decoder_${task}.report.txt" 2>&1
done

echo "[dec] DONE at $(date -Is)"
