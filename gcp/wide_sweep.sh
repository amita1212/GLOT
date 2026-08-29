#!/usr/bin/env bash
# WIDE sweep: graph knobs AND the paper's Table 6 optimizer/architecture grid.
#
# WHY THIS RUN EXISTS
#   Every earlier campaign varied only graph-construction knobs and inherited
#   lr / weight_decay / depth / width / jk / proj_dim from PAPER_DEFAULTS -- i.e.
#   from values the original authors tuned for a EUCLIDEAN pooler. Our sharpest
#   result (AB -3.56, ABC -4.51 on TinyLlama, 0/15 seeds positive) is exactly the
#   signature of a structured layer trained at the wrong learning rate, so it is
#   currently unsupportable as "hyperbolic readout/GNN are harmful".
#
#   Two outcomes, both worth having:
#     - arms recover  -> the negative result is retracted and replaced by a
#                        stronger claim: hyperbolic poolers need structure-aware
#                        LR, and comparisons that omit it are unfair.
#     - arms stay bad -> the negative survives its strongest objection.
#
# FAIRNESS
#   --wide applies the SAME optimizer grid to every arm, baseline included, and
#   --fine_baseline keeps the cosine density grid at 10 points. Equal budget per
#   arm is the whole point; giving the hyperbolic arms a bigger space would
#   manufacture the bias this campaign exists to detect.
#
# BACKBONE
#   BERT first, deliberately. A run costs ~200 s here versus ~935 s on
#   TinyLlama, so the entire question can be answered on BERT for the price of
#   a fraction of one decoder task. If the optimizer axis matters, it will show
#   up here and the decoder rerun becomes justified.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
mkdir -p logs results

MODEL="bert-base-uncased"
# baseline + no_graph are the two controls; A/B/C and combinations are the arms
# whose verdict the optimizer axis could actually change.
ARMS="baseline no_graph A B C AB AC BC ABC"
SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
TRIALS="${TRIALS:-40}"
TASKS="${1:-stsb cola}"

echo "[wide] ===== start $(date -Is) ====="
echo "[wide] model=$MODEL trials=$TRIALS tasks=$TASKS"
echo "[wide] arms=$ARMS"

for task in $TASKS; do
    echo "[wide] pre-warming cache: $MODEL / $task"
    bash prewarm_model.sh "$MODEL" -1 "$task" > /dev/null 2>&1

    echo "[wide] ===== $task ===== $(date -Is)"
    "$PY" campaign.py --target glue --task "$task" --model "$MODEL" \
        --arms $ARMS --trials "$TRIALS" --stage both \
        --wide --fine_baseline --confirm_seeds $SEEDS \
        --out "results/campaign_wide_${task}.csv" \
        >> "logs/campaign_wide_${task}.log" 2>&1

    echo "[wide] --- paired result: $task ---"
    "$PY" paired_analysis.py "results/campaign_wide_${task}.csv" 2>&1 | head -24
    "$PY" analyze_campaign.py "results/campaign_wide_${task}.csv" \
        > "results/campaign_wide_${task}.report.txt" 2>&1

    # The headline question: does the BEST lr differ between the Euclidean
    # baseline and the hyperbolic arms? If it does, every previous comparison in
    # this project (and arguably in the original paper) was unfair.
    echo "[wide] --- winning lr per arm: $task ---"
    "$PY" - <<'PYEOF' 2>/dev/null
import csv, glob, collections
for f in sorted(glob.glob("results/campaign_wide_*.csv")):
    best = {}
    for r in csv.DictReader(open(f)):
        if r.get("stage") != "tune":
            continue
        try:
            v = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
        a = r.get("arm")
        if a not in best or v > best[a][0]:
            best[a] = (v, r.get("detail", ""))
    print(f"  {f}")
    for a, (v, d) in sorted(best.items()):
        lr = next((p for p in d.split(";") if p.startswith("lr=")), "lr=?")
        jk = next((p for p in d.split(";") if p.startswith("jk_mode=")), "")
        nl = next((p for p in d.split(";") if p.startswith("num_layers=")), "")
        print(f"    {a:<10} best={v:6.2f}  {lr:<12} {jk:<12} {nl}")
PYEOF
done

echo "[wide] ===== ALL DONE $(date -Is) ====="
