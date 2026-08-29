"""Are Stage B tuning runs collapsing to a degenerate classifier?

An MCC of exactly 0.00 means the model predicted a single class. Seeing one in
the B arm is suggestive given the paper's "B amplifies variance" story, but it
could equally be a low-learning-rate artifact that hits every arm. This asks
the CSVs, per arm and per learning rate.

NOTE: these are TUNING rows. Nothing here is a result and none of it goes in
the paper. It is a diagnostic about optimisation behaviour only.
"""
import csv
import glob
import os
from collections import defaultdict

os.chdir(os.path.expanduser("~/glot"))

rows = []
for p in glob.glob("results/campaign_robfill_*.csv"):
    with open(p, newline="") as f:
        rows += list(csv.DictReader(f))

if not rows:
    raise SystemExit("no robfill rows yet")

print(f"{len(rows)} tuning rows so far\n")

by_arm = defaultdict(lambda: [0, 0])          # arm -> [scored, degenerate]
by_lr = defaultdict(lambda: [0, 0])           # lr  -> [scored, degenerate]
by_arm_lr = defaultdict(lambda: [0, 0])

for r in rows:
    s = r.get("score")
    if s in (None, ""):
        continue
    try:
        v = float(s)
    except ValueError:
        continue
    arm = r.get("arm", "?")
    lr = r.get("lr", "?")
    degen = abs(v) < 1e-9
    for d in (by_arm[arm], by_lr[lr], by_arm_lr[(arm, lr)]):
        d[0] += 1
        d[1] += int(degen)

print("per arm:")
for arm, (n, z) in sorted(by_arm.items()):
    print(f"   {arm:5s} scored={n:3d}  degenerate={z:3d}  ({100*z/n:.0f}%)")

print("\nper learning rate:")
for lr, (n, z) in sorted(by_lr.items(), key=lambda kv: float(kv[0]) if kv[0] not in ("?", "") else 0):
    print(f"   lr={lr:>8s} scored={n:3d}  degenerate={z:3d}  ({100*z/n:.0f}%)")

print("\narm x lr (only cells with a degenerate run):")
for (arm, lr), (n, z) in sorted(by_arm_lr.items()):
    if z:
        print(f"   {arm:5s} lr={lr:>8s}  {z}/{n}")

print("\nbest tuning score so far per arm (NOT a result -- tuning maxima are")
print("inflated and have already reversed a ranking in this project):")
best = defaultdict(float)
for r in rows:
    try:
        v = float(r.get("score") or 0)
    except ValueError:
        continue
    a = r.get("arm", "?")
    best[a] = max(best[a], v)
for a, v in sorted(best.items()):
    print(f"   {a:5s} {v:.2f}")
