"""Measured per-run cost per campaign, and what it would take to fill Table 1."""
import csv
import glob
import os
from collections import defaultdict

d = defaultdict(list)
for p in glob.glob("results/campaign_*.csv"):
    b = os.path.basename(p)
    if any(b.endswith(f"_w{i}.csv") for i in range(4)):
        continue
    for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
        for key in ("elapsed_sec", "secs", "seconds"):
            try:
                s = float(r.get(key) or 0)
            except (ValueError, TypeError):
                s = 0
            if s > 0:
                d[b].append(s)
                break

print("MEASURED per-run seconds")
print("-" * 62)
rate = {}
for k in sorted(d):
    v = d[k]
    m = sum(v) / len(v)
    rate[k] = m
    print(f"  {k:34s} n={len(v):4d}  mean={m:7.1f}s")

print()
print("=" * 74)
print("COST TO FILL THE GAPS IN TABLE 1")
print("=" * 74)

rc = rate.get("campaign_rob_cola.csv", 300.0)
rs = rate.get("campaign_rob_stsb.csv", 220.0)
dec = rate.get("campaign_decoder_stsb.csv", 600.0)

MISSING_ARMS = 6          # B, C, AB, AC, BC, ABC
TUNE, SEEDS = 40, 15
per_arm = TUNE + SEEDS

plans = [
    ("RoBERTa CoLA  + 6 hyperbolic arms", MISSING_ARMS * per_arm, rc),
    ("RoBERTa STS-B + 6 hyperbolic arms", MISSING_ARMS * per_arm, rs),
    ("  ^ same, CONFIRM ONLY (reuse BERT cfg)", MISSING_ARMS * SEEDS, rc),
    ("  ^ same, CONFIRM ONLY (reuse BERT cfg)", MISSING_ARMS * SEEDS, rs),
    ("RoBERTa MRPC  full 9-arm campaign", 9 * per_arm, rc),
    ("RoBERTa RTE   full 9-arm campaign", 9 * per_arm, rc),
    ("TinyLlama CoLA full 9-arm campaign", 9 * per_arm, dec),
]
for name, runs, sec in plans:
    h = runs * sec / 3600
    print(f"  {name:42s} {runs:5d} runs  {h:7.1f} h")

print()
print("RECOMMENDED: confirm-only fill of the 6 missing arms on RoBERTa,")
print("both tasks, reusing each arm's BERT-selected configuration.")
tot = MISSING_ARMS * SEEDS * (rc + rs) / 3600
print(f"  total = {MISSING_ARMS * SEEDS * 2} runs, {tot:.1f} GPU-hours")
print("  -> tests whether the B-harm and C-help replicate on a 2nd encoder,")
print("     which is currently the single biggest hole in the paper.")
