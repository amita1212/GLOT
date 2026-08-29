#!/usr/bin/env bash
# Layer probe: baseline GLOT accuracy at every candidate layer.
#
# WHY
# ---
# delta-hyperbolicity says the EARLY layers of BERT are the tree-like ones
# (L1 .079 vs L12 .196), which is where hyperbolic geometry has a mechanism to
# help. But tree-likeness is not the only thing that matters: a layer also has
# to carry task-relevant information. Launching a full 12-arm campaign at
# layer 2 produced a BASELINE of 5.14 / 4.73 / 3.02 MCC on CoLA, versus 45.54
# at layer 12 -- layer 2 is geometrically beautiful and semantically empty, so
# every arm comparison there would be noise around a broken model.
#
# This probe measures the OTHER half of the trade-off cheaply: run only the
# cosine baseline at each layer and find where the task is still solvable. The
# layer to run the real campaign on is the most tree-like layer that retains
# competitive accuracy -- not the most tree-like layer outright.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

TASK="${1:-cola}"
LAYERS="${2:-4 6 8 10 12}"

for L in $LAYERS; do
  echo "=================== layer $L ==================="
  # Build the cache for this layer if it does not exist yet (cache key includes
  # the layer, so this can never collide with another layer's features).
  if [ "$L" = "-1" ]; then TAG=""; else TAG="_L${L}"; fi
  if [ ! -f "data/bert-base-uncased_${TASK}_train${TAG}_batches/metadata.json" ] \
     && [ ! -f "data/bert-base-uncased_sts_train${TAG}_batches/metadata.json" ]; then
    echo "  (pre-warming cache for layer $L)"
    bash prewarm_model.sh bert-base-uncased "$L" "$TASK" > /dev/null 2>&1
  fi
  "$PY" campaign.py --target glue --task "$TASK" \
      --arms baseline --trials 3 --stage tune \
      --hidden_layer "$L" \
      --out "results/layer_probe_${TASK}.csv" 2>&1 \
    | grep -E '^\s+->|FAIL'
done

echo
echo "=== layer probe summary ==="
"$PY" - "results/layer_probe_${TASK}.csv" <<'PY'
import csv, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
best = defaultdict(float)
for r in rows:
    best[r["setting"]] = max(best[r["setting"]], float(r["score"]))
# delta_rel measured by screen_backbones.py on bert-base-uncased
DELTA = {1: .079, 2: .087, 3: .090, 4: .110, 5: .127, 6: .149,
         7: .146, 8: .151, 9: .167, 10: .193, 11: .199, 12: .196}
print(f"{'setting':<16} {'best score':>10} {'delta_rel':>10}")
for k in sorted(best, key=lambda s: -best[s]):
    layer = int(k.split('_L')[1]) if '_L' in k else 12
    print(f"{k:<16} {best[k]:>10.2f} {DELTA.get(layer, float('nan')):>10.3f}")
print()
print("Pick the layer with the LOWEST delta_rel among those whose score is")
print("competitive with layer 12. Tree-likeness is useless without task signal.")
PY
