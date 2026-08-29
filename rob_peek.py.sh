#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
~/glotenv/bin/python - <<'PY'
import csv, glob, statistics as st
from collections import defaultdict

rows = []
for p in glob.glob("results/campaign_rob_*_w*.csv"):
    rows += list(csv.DictReader(open(p)))

# --- density per arm: this is the load-bearing number for the calibration claim
dens = defaultdict(list)
for r in rows:
    v = r.get("mean_density")
    if v not in (None, "", "nan"):
        dens[r.get("arm")].append(float(v))

print("\n  RoBERTa realised edge density, by arm")
print(f"  {'arm':<13}{'n':>4}{'min':>8}{'median':>9}{'max':>8}")
print("  " + "-" * 42)
for arm in sorted(dens, key=lambda a: -st.median(dens[a])):
    v = sorted(dens[arm])
    print(f"  {arm:<13}{len(v):>4}{v[0]:>8.3f}{st.median(v):>9.3f}{v[-1]:>8.3f}")

# --- anything with density > 1 is not a density; flag it
bad = [r for r in rows
       if r.get("mean_density") not in (None, "", "nan")
       and float(r["mean_density"]) > 1.0]
print(f"\n  rows with density > 1.0: {len(bad)}")
for r in bad[:5]:
    print(f"    arm={r.get('arm')} d={r.get('mean_density')} "
          f"tau={r.get('tau')} tq={r.get('tau_quantile')} "
          f"metric={r.get('graph_metric')} score={r.get('score')}")

# --- confirm-stage scores so far
conf = defaultdict(list)
for r in rows:
    if r.get("stage") != "confirm":
        continue
    try:
        conf[r.get("arm")].append(float(r["score"]))
    except (KeyError, TypeError, ValueError):
        pass
print("\n  confirmation stage so far (STS-B Spearman, PARTIAL -- not all 15 seeds)")
for arm in sorted(conf, key=lambda a: -st.mean(conf[a])):
    v = conf[arm]
    s = st.stdev(v) if len(v) > 1 else 0.0
    print(f"    {arm:<13} n={len(v):>3}  mean={st.mean(v):7.3f}  std={s:5.3f}")

# --- best tuning config per arm (indicative only)
best = {}
for r in rows:
    if r.get("stage") != "tune":
        continue
    try:
        s = float(r["score"])
    except (KeyError, TypeError, ValueError):
        continue
    a = r.get("arm")
    if a not in best or s > best[a][0]:
        best[a] = (s, r.get("mean_density"))
print("\n  best tuning score per arm (SELECTION-BIASED, not a result)")
for a, (s, d) in sorted(best.items(), key=lambda kv: -kv[1][0]):
    print(f"    {a:<13} {s:7.3f}   density={d}")
print()
PY
