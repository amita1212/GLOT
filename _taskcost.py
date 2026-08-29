"""What would the remaining GLUE tasks actually cost? Use measured runtimes."""
import csv, glob, os
from collections import defaultdict

# GLUE training-set sizes
SIZE = {"cola": 8551, "sst2": 67349, "mrpc": 3668, "stsb": 5749,
        "qqp": 363846, "mnli": 392702, "qnli": 104743, "rte": 2490,
        "wnli": 635}

# measured seconds/run from our own campaign logs
secs = defaultdict(list)
for path in glob.glob("results/campaign_*.csv"):
    b = os.path.basename(path)
    if any(b.endswith(f"_w{i}.csv") for i in range(4)):
        continue
    task = None
    for t in SIZE:
        if b.endswith(f"_{t}.csv"):
            task = t
    if not task:
        continue
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        try:
            s = float(r.get("secs") or r.get("seconds") or r.get("elapsed") or 0)
            if s > 0:
                secs[task].append(s)
        except ValueError:
            pass

print("MEASURED cost per run")
print("-" * 62)
ref = None
for t in ("rte", "mrpc", "stsb", "cola"):
    if secs[t]:
        m = sum(secs[t]) / len(secs[t])
        print(f"  {t:6s} n_train={SIZE[t]:7,d}  {m:7.1f} s/run  (n={len(secs[t])})")
        if t == "cola":
            ref = m
if not ref:
    ref = 157.0
    print("  (no cola timing found; assuming 157 s/run)")

print()
print(f"PROJECTED, scaling linearly from CoLA ({ref:.0f} s/run at 8,551 ex.)")
print("-" * 78)
print(f"{'task':7s} {'n_train':>9s} {'x CoLA':>7s} {'s/run':>8s} "
      f"{'FULL campaign':>15s} {'reduced':>12s}")
print(f"{'':7s} {'':>9s} {'':>7s} {'':>8s} {'(9 arms, 495 runs)':>15s} "
      f"{'(4 arms, 60)':>12s}")
for t in ("wnli", "rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"):
    mult = SIZE[t] / SIZE["cola"]
    s = ref * mult
    full = 495 * s / 3600
    red = 60 * s / 3600
    mark = "  <- done" if t in ("cola", "stsb", "mrpc") else ""
    fs = f"{full:8.1f} h" if full < 480 else f"{full/24:7.1f} d"
    rs = f"{red:6.1f} h" if red < 96 else f"{red/24:5.1f} d"
    print(f"{t:7s} {SIZE[t]:9,d} {mult:6.1f}x {s:7.0f}s {fs:>15s} {rs:>12s}{mark}")

print()
print("Cache size (fp16, 128 tok, 768 dim) -- disk is the other limit")
for t in ("sst2", "qnli", "qqp", "mnli"):
    gb = SIZE[t] * 128 * 768 * 2 / 1e9
    print(f"  {t:6s} ~{gb:6.1f} GB")
