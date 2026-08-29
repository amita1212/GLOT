#!/usr/bin/env bash
# Print every confirmation row for the given arms, so identical aggregate
# statistics can be traced to either a genuine coincidence or a dead knob.
# Bit-identical scores across a knob's settings mean the knob is not reaching
# the model -- that signature is how the empty-graph and c=1.0 bugs were found.
set -u
cd /home/t-amitalfasi/glot
CSV="${1:-results/campaign_stress_hard.csv}"
shift || true
ARMS="${@:-W_hyp W_depth baseline}"
~/glotenv/bin/python - "$CSV" $ARMS <<'PY'
import csv, sys
p, arms = sys.argv[1], set(sys.argv[2:])
rows = [r for r in csv.DictReader(open(p))
        if r.get("arm") in arms and r.get("stage") == "confirm"]
for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
    print(f"{r['arm']:<10} seed={r['seed']:<3} score={r['score']:<8} "
          f"dens={r.get('mean_density',''):<9} ewm={r.get('edge_weight_mode','-'):<8} "
          f"fm={r.get('feature_mode','-'):<15} gc={r.get('graph_curvature','-'):<6} "
          f"tq={r.get('tau_quantile','-')}")
PY
