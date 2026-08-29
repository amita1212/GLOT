"""What is actually in the decoder campaigns, and how fast is robfill going.

Written as a file and scp'd rather than passed inline: inline python in a
gcloud --command gets mangled by PowerShell quoting (twice, this project).
"""
import csv
import glob
import os
from collections import Counter

os.chdir(os.path.expanduser("~/glot"))


def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


print("=" * 66)
print("DECODER CAMPAIGNS")
print("=" * 66)
for p in sorted(glob.glob("results/campaign_decoder_*.csv")):
    rows = load(p)
    print(f"\n{p}  ({len(rows)} rows)")
    if not rows:
        continue
    keys = rows[0].keys()
    stage_k = "stage" if "stage" in keys else None
    arm_k = "arm" if "arm" in keys else None
    if arm_k and stage_k:
        c = Counter((r[arm_k], r[stage_k]) for r in rows)
        for (arm, st), n in sorted(c.items()):
            print(f"   {arm:10s} {st:8s} {n:4d}")
    elif arm_k:
        for arm, n in sorted(Counter(r[arm_k] for r in rows).items()):
            print(f"   {arm:10s} {n:4d}")
    print("   columns:", ", ".join(list(keys)[:12]))

print()
print("=" * 66)
print("ROBFILL PROGRESS")
print("=" * 66)
tot = 0
for p in sorted(glob.glob("results/campaign_robfill_*.csv")):
    rows = load(p)
    tot += len(rows)
    secs = [float(r["seconds"]) for r in rows if r.get("seconds")]
    mt = os.path.getmtime(p)
    print(f"{os.path.basename(p):34s} {len(rows):4d} rows  "
          f"mean {sum(secs)/len(secs) if secs else 0:6.1f}s  "
          f"last write {mt:.0f}")
print(f"\ntotal robfill rows: {tot} / 660 target")
for p in sorted(glob.glob("logs/robfill_w*.log")):
    st = os.stat(p)
    print(f"{p}  size={st.st_size}")
