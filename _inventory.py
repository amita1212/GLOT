"""Complete inventory: every campaign we ran. Task, model, arms, seeds, stage."""
import csv, glob, os
from collections import defaultdict

rows_by_file = {}
for path in sorted(glob.glob("results/*.csv")):
    b = os.path.basename(path)
    if any(b.endswith(f"_w{i}.csv") for i in range(4)):
        continue
    try:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    except Exception as e:
        print(f"  !! {b}: {e}")
        continue
    if not rows:
        continue
    rows_by_file[b] = rows

print(f"{'file':38s} {'task':7s} {'model':22s} {'arms':>4s} {'confirm':>7s} "
      f"{'seeds':>5s} {'total':>6s}")
print("-" * 100)
for b, rows in rows_by_file.items():
    tasks = sorted({r.get("task", "?") for r in rows})
    models = sorted({(r.get("model") or "?").split("/")[-1] for r in rows})
    arms = sorted({r.get("arm", "?") for r in rows})
    conf = [r for r in rows if r.get("stage") == "confirm"]
    seeds = sorted({r.get("seed") for r in conf if r.get("seed")})
    print(f"{b:38s} {','.join(tasks)[:7]:7s} {','.join(models)[:22]:22s} "
          f"{len(arms):4d} {len(conf):7d} {len(seeds):5d} {len(rows):6d}")

print()
print("=" * 100)
print("ARMS PER FILE")
print("=" * 100)
for b, rows in rows_by_file.items():
    arms = sorted({r.get("arm", "?") for r in rows})
    print(f"  {b}")
    print(f"      {', '.join(arms)}")

print()
print("=" * 100)
print("TASK x MODEL COVERAGE  (confirm-stage runs only)")
print("=" * 100)
cov = defaultdict(int)
for b, rows in rows_by_file.items():
    for r in rows:
        if r.get("stage") != "confirm":
            continue
        t = r.get("task", "?")
        m = (r.get("model") or "?").split("/")[-1]
        cov[(m, t)] += 1
models = sorted({m for m, _ in cov})
tasks = sorted({t for _, t in cov})
print(f"{'model':24s} " + " ".join(f"{t:>9s}" for t in tasks))
for m in models:
    print(f"{m:24s} " + " ".join(
        f"{cov[(m,t)]:9d}" if cov[(m, t)] else f"{'-':>9s}" for t in tasks))
