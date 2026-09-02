"""Sanity-check the very large Stage B effects on MTEB before believing them.

Three questions:
  1. Are B's scores degenerate (constant, zero, or pinned at a chance value)?
  2. Does B amplify seed variance, as the paper's GLUE mechanism predicts?
  3. Is the magnitude plausible relative to an arm that does NOT touch the
     readout (A), which is the control for "is the MTEB pipeline just noisy?"
"""
import json
import pandas as pd

TASKS = ["EmotionClassification", "SciFact", "RedditClustering",
         "AskUbuntuDupQuestions", "STS12", "TwitterSemEval2015", "SummEval"]

df = pd.read_csv("results_vm/mteb_table3.csv")
df = df[df["model"].astype(str).str.contains("bert-base", na=False)]

rows = []
for _, r in df.iterrows():
    try:
        per = json.loads(r["mteb_per_task"])
    except Exception:
        continue
    for t, v in per.items():
        rows.append({"arm": r["arm"], "seed": int(r["seed"]),
                     "task": t, "score": float(v)})
long = pd.DataFrame(rows).drop_duplicates(subset=["arm", "seed", "task"])

arms = ["baseline", "A_threshold", "B_threshold"]
long = long[long["arm"].isin(arms)]

print("=== 1. Is B degenerate?  min / max / n_distinct over 15 seeds ===")
print(f"{'task':<24}{'arm':<14}{'min':>8}{'max':>8}{'range':>8}{'distinct':>10}")
for t in TASKS:
    for a in arms:
        s = long[(long.task == t) & (long.arm == a)]["score"]
        if len(s) == 0:
            continue
        print(f"{t:<24}{a:<14}{s.min():8.2f}{s.max():8.2f}"
              f"{s.max() - s.min():8.2f}{s.nunique():10d}")
    print()

print("=== 2. Seed sd per arm (paper's mechanism: B amplifies variance) ===")
print(f"{'task':<24}{'baseline':>10}{'A':>10}{'B':>10}{'B/base':>9}")
ratios = []
for t in TASKS:
    sds = {}
    for a in arms:
        s = long[(long.task == t) & (long.arm == a)]["score"]
        sds[a] = s.std(ddof=1) if len(s) > 1 else float("nan")
    r = sds["B_threshold"] / sds["baseline"]
    ratios.append(r)
    print(f"{t:<24}{sds['baseline']:10.3f}{sds['A_threshold']:10.3f}"
          f"{sds['B_threshold']:10.3f}{r:9.2f}")
print(f"{'median ratio':<24}{'':>30}{pd.Series(ratios).median():9.2f}")

print()
print("=== 3. Magnitude: largest |delta| per arm (A does not touch readout) ===")
base = long[long.arm == "baseline"].set_index(["task", "seed"])["score"]
for a in ["A_threshold", "B_threshold"]:
    arm = long[long.arm == a].set_index(["task", "seed"])["score"]
    d = (arm - base).dropna()
    per_task = d.groupby(level=0).mean()
    print(f"{a:<14} largest |mean delta| = {per_task.abs().max():6.2f} "
          f"on {per_task.abs().idxmax()}")
