#!/usr/bin/env bash
# Decisive test of the cold/warm hidden-state-cache confound.
#
# Upstream main.py, train_single_classification():
#     train_loader = DataLoader(train_ds, shuffle=True, collate_fn=...)   # loader #1
#     train_ds = precompute_hidden_states(backbone, train_loader, ...)    # iterates #1
#     ...
#     classifier = SingleClassifier(...)                                  # random init
#
# precompute_hidden_states() early-returns when the cache exists:
#     if not override and os.path.exists(meta_file):
#         return BatchedHiddenStateDataset(batch_dir)                     # loader NEVER iterated
#
# COLD cache -> loader #1 IS iterated -> torch.randperm() consumes global RNG
# WARM cache -> loader #1 is NOT iterated -> RNG untouched
#
# => the classifier's random init and the training batch order differ between
#    cold and warm runs *with the same seed*.
#
# Prediction: override_precompute=1 (always cold) reproduces ~40.4, while
# override_precompute=0 with a warm cache gives ~45.5.
set -euo pipefail
cd /home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python

echo "############ COLD cache (override_precompute=1), seed 42 x3 ############"
"$PY" repro_paper.py --tasks cola --taus 0.6 --seeds 42 42 42 \
    --override_precompute 1 \
    --out results/cache_cold.csv 2>&1 | grep -E "DONE|RUN-TO-RUN|^cola|std"

echo
echo "############ WARM cache (override_precompute=0), seed 42 x3 ############"
"$PY" repro_paper.py --tasks cola --taus 0.6 --seeds 42 42 42 \
    --override_precompute 0 \
    --out results/cache_warm.csv 2>&1 | grep -E "DONE|RUN-TO-RUN|^cola|std"
