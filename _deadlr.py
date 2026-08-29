"""Is lr=2e-5 a dead region of the published grid, or is that RoBERTa-only?

If a third of every search lands on a learning rate that cannot train, that is
worth one line in the reproducibility appendix -- the nominal 40-trial budget
is really ~27. It is equal across arms, so it does not threaten any comparison;
it just means the search is smaller than it looks.

Counts only TUNING rows. A run is called dead if its score is <= 0 for a
correlation/MCC metric (no better than chance) or exactly 0.
"""
import csv
import glob
import os
import re
from collections import defaultdict

os.chdir(os.path.expanduser("~/glot"))

# NB: the shard filter must be anchored. "_w" alone also matches
# campaign_WIDE_cola.csv, which silently dropped every BERT campaign the first
# time this was run.
SHARD = re.compile(r"_w\d+\.csv$")

CAMPAIGNS = sorted(glob.glob("results/campaign_wide_*.csv")
                   + glob.glob("results/campaign_rob_*.csv")
                   + glob.glob("results/campaign_decoder_*.csv"))

grand = defaultdict(lambda: [0, 0])

for p in CAMPAIGNS:
    if SHARD.search(os.path.basename(p)):        # skip per-worker shards
        continue
    with open(p, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("stage") == "tune"]
    if not rows:
        continue
    by_lr = defaultdict(lambda: [0, 0, 0.0])   # lr -> [n, dead, sum_score]
    for r in rows:
        try:
            v = float(r.get("score") or "nan")
        except ValueError:
            continue
        if v != v:
            continue
        lr = r.get("lr", "?")
        d = by_lr[lr]
        d[0] += 1
        d[1] += int(v <= 0.0)
        d[2] += v
        g = grand[lr]
        g[0] += 1
        g[1] += int(v <= 0.0)

    task = rows[0].get("task", "?")
    model = (rows[0].get("model", "?") or "?").split("/")[-1]
    print(f"\n{os.path.basename(p)}   {model} / {task}   ({len(rows)} tuning rows)")
    for lr, (n, dead, tot) in sorted(
            by_lr.items(), key=lambda kv: float(kv[0]) if kv[0] not in ("?", "") else 0):
        print(f"    lr={lr:>8s}  n={n:4d}  dead={dead:4d} ({100*dead/n:5.1f}%)"
              f"  mean_score={tot/n:6.2f}")

print("\n" + "=" * 60)
print("ALL CAMPAIGNS POOLED")
print("=" * 60)
tot_n = tot_d = 0
for lr, (n, dead) in sorted(grand.items(),
                            key=lambda kv: float(kv[0]) if kv[0] not in ("?", "") else 0):
    tot_n += n
    tot_d += dead
    print(f"  lr={lr:>8s}  n={n:5d}  dead={dead:5d}  ({100*dead/n:5.1f}%)")
print(f"\n  overall {tot_d}/{tot_n} = {100*tot_d/tot_n:.1f}% of tuning trials are dead")
