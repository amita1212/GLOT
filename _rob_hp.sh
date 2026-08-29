#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
~/glotenv/bin/python - <<'EOF'
import csv, collections
for task in ("stsb", "cola"):
    f = f"results/campaign_rob_{task}.csv"
    rows = list(csv.DictReader(open(f)))
    tune = [r for r in rows if r["stage"] == "tune"]
    conf = [r for r in rows if r["stage"] == "confirm"]
    print("=" * 78)
    print(f"  RoBERTa {task}:  {len(tune)} tune rows, {len(conf)} confirm rows")
    print(f"  tune seeds  = {sorted({r['seed'] for r in tune})}")
    print(f"  confirm seeds = {len(sorted({r['seed'] for r in conf}))} distinct")
    print(f"  settings    = {sorted({r['setting'] for r in rows})}")
    print(f"  models      = {sorted({r['model'] for r in rows})}")
    print("=" * 78)
    by = collections.defaultdict(list)
    for r in tune:
        by[r["arm"]].append(r)
    print(f"  {'arm':12s} {'draws':>6} {'distinct cfg':>13} {'best tune':>10}")
    best = {}
    for a in sorted(by):
        rs = by[a]
        d = {r["detail"] for r in rs}
        b = max(rs, key=lambda r: float(r["score"]))
        best[a] = b
        print(f"  {a:12s} {len(rs):6d} {len(d):13d} {float(b['score']):10.2f}")
    print("\n  selected config per arm (used for all 15 confirm seeds):")
    for a in sorted(best):
        print(f"   {a}:")
        for kv in sorted(best[a]["detail"].split(";")):
            if kv:
                print(f"      {kv}")
    print()
EOF
