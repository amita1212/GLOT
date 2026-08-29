#!/usr/bin/env bash
# Noise-floor probe: run the SAME config repeatedly (seed 42 five times) to
# measure pure run-to-run variance, then five DIFFERENT seeds to measure
# seed variance. If the noise floor is large, every delta we (and the paper)
# report on a single seed is uninterpretable.
set -euo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

# 1) same seed, repeated -> pure nondeterminism (cuDNN / GPU atomics)
nohup "$PY" repro_paper.py \
    --tasks cola \
    --taus 0.6 \
    --seeds 42 42 42 42 42 \
    --out results/noise_same_seed.csv \
    > logs_noise_same_seed.txt 2>&1

# 2) different seeds -> seed variance
nohup "$PY" repro_paper.py \
    --tasks cola \
    --taus 0.6 \
    --seeds 1 2 3 4 5 \
    --out results/noise_diff_seed.csv \
    > logs_noise_diff_seed.txt 2>&1

echo "=== SAME SEED (42 x5) ==="
tail -20 logs_noise_same_seed.txt
echo
echo "=== DIFFERENT SEEDS (1..5) ==="
tail -20 logs_noise_diff_seed.txt
