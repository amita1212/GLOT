#!/usr/bin/env bash
# Layer probe for ANY backbone: baseline GLOT accuracy at every candidate layer.
#
# WHY THIS EXISTS SEPARATELY FROM layer_probe.sh
# ----------------------------------------------
# layer_probe.sh hardcodes bert-base-uncased. This version takes the model as an
# argument so the layer-selection finding can be TRANSFER-TESTED on a second
# backbone. Transfer is the whole point: layer 8 was chosen by searching 6
# layers on CoLA dev, so some of its +4.6 MCC over layer 12 is selection bias.
# If "read the token graph from a mid layer" is a real property of transformer
# encoders it must reproduce on a different model; if it only holds for BERT on
# CoLA it was dev-set overfitting and the claim dies.
#
# NOTE ON MODEL CHOICE: ModernBERT is run here as a transfer test, NOT because
# it is more tree-like. The angular-delta screen showed its apparent
# tree-likeness is a massive-activation artefact (max||x||/median||x|| = 156 at
# L16 drives Euclidean delta to 0.0021 while the angular delta stays at 0.2127).
#
# Usage: bash layer_probe_any.sh [task] ["layers"] [model]
#   e.g. bash layer_probe_any.sh cola "4 8 12 16 20 22" answerdotai/ModernBERT-base
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

TASK="${1:-cola}"
LAYERS="${2:-4 6 8 10 12}"
MODEL="${3:-bert-base-uncased}"
SLUG="${MODEL//\//_}"

for L in $LAYERS; do
  echo "=================== $MODEL layer $L ==================="
  # Build the cache for this layer if it does not exist yet. The cache key
  # includes BOTH model and layer, so probes can never collide.
  if [ "$L" = "-1" ]; then TAG=""; else TAG="_L${L}"; fi
  if [ ! -f "data/${SLUG}_${TASK}_train${TAG}_batches/metadata.json" ] \
     && [ ! -f "data/${SLUG}_sts_train${TAG}_batches/metadata.json" ]; then
    echo "  (pre-warming cache for layer $L)"
    bash prewarm_model.sh "$MODEL" "$L" "$TASK" > /dev/null 2>&1
    if [ ! -f "data/${SLUG}_${TASK}_train${TAG}_batches/metadata.json" ] \
       && [ ! -f "data/${SLUG}_sts_train${TAG}_batches/metadata.json" ]; then
      echo "  PREWARM FAILED for layer $L -- last lines of its log:"
      tail -8 "logs/prewarm_${SLUG}_L${L}_${TASK}.log" 2>/dev/null
      continue
    fi
  fi
  "$PY" campaign.py --target glue --task "$TASK" --model "$MODEL" \
      --arms baseline --trials 3 --stage tune \
      --hidden_layer "$L" \
      --out "results/layer_probe_${SLUG}_${TASK}.csv" 2>&1 \
    | grep -E '^\s+->|FAIL'
done

echo
echo "=== layer probe summary: $MODEL / $TASK ==="
"$PY" - "results/layer_probe_${SLUG}_${TASK}.csv" "$MODEL" <<'PY'
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
model = sys.argv[2]
best = defaultdict(float)
for r in rows:
    best[r["setting"]] = max(best[r["setting"]], float(r["score"]))

# Measured by screen_backbones.py. d_ang is the ANGULAR delta-hyperbolicity,
# which is immune to the massive-activation outliers that make the Euclidean
# delta meaningless for modern encoders. ALWAYS rank on d_ang, never d_eucl.
D_ANG = {
    "bert-base-uncased": {1: .1045, 4: .1257, 8: .1586, 12: .2114},
    "answerdotai/ModernBERT-base": {1: .1030, 4: .1389, 8: .1950, 12: .1980,
                                    16: .2127, 20: .1717, 22: .1646},
}.get(model, {})

print(f"{'setting':<26} {'best score':>10} {'d_ang':>8}")
for k in sorted(best, key=lambda s: -best[s]):
    layer = int(k.split("_L")[1]) if "_L" in k else -1
    d = D_ANG.get(layer)
    print(f"{k:<26} {best[k]:>10.2f} {(f'{d:.4f}' if d else 'n/a'):>8}")
print()
print("Pick the layer with the LOWEST d_ang among those whose score is")
print("competitive with the best. Tree-likeness is useless without task signal.")
PY
