"""Paired analysis of the MTEB Table-3 campaign, arm vs baseline, per task.

Same rules as every other test in this paper: paired t-test AND exact two-sided
sign test on the shared seed set, both must agree. Only arms with a complete
15-seed confirmation set are tested.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

CSV = "results_vm/mteb_table3.csv"
TASKS = ["EmotionClassification", "SciFact", "RedditClustering",
         "AskUbuntuDupQuestions", "STS12", "TwitterSemEval2015", "SummEval"]

d = pd.read_csv(CSV)
m = d[(d.task == "mteb")].dropna(subset=["mteb_per_task"])

rec = []
for _, r in m.iterrows():
    for k, v in json.loads(r.mteb_per_task).items():
        rec.append((str(r.arm), int(r.seed), k, float(v) * 100))
t = pd.DataFrame(rec, columns=["arm", "seed", "task", "score"])

# de-duplicate defensively (same run key must not be counted twice)
t = t.drop_duplicates(subset=["arm", "seed", "task"], keep="first")

wide = t.pivot_table(index=["arm", "seed"], columns="task", values="score")
counts = wide.groupby("arm").size()
print("seeds per arm:", dict(counts))

complete = [a for a in counts.index if counts[a] >= 15 and a != "baseline"]
if "baseline" not in counts or counts["baseline"] < 15:
    print("baseline incomplete -- nothing to test")
    sys.exit(0)
print("arms with a full 15-seed set:", complete or "(none yet)")

base = wide.loc["baseline"]
print(f"\nBASELINE, BERT, mean +/- sd over {counts['baseline']} seeds")
for task in TASKS:
    if task in base:
        print(f"  {task:<24}{base[task].mean():6.2f} +/- {base[task].std(ddof=1):.2f}")

for arm in complete:
    a = wide.loc[arm]
    seeds = sorted(set(a.index) & set(base.index))
    print(f"\n=== {arm}  vs baseline, n={len(seeds)} paired seeds ===")
    print(f"{'task':<24}{'mean':>7}{'delta':>8}{'t':>8}{'t_p':>9}"
          f"{'sign_p':>9}{'p/n':>8}")
    for task in TASKS:
        if task not in a:
            continue
        x = a.loc[seeds, task].values
        y = base.loc[seeds, task].values
        dif = x - y
        tt = stats.ttest_rel(x, y)
        pos = int((dif > 0).sum())
        neg = int((dif < 0).sum())
        nz = pos + neg
        sign_p = stats.binomtest(pos, nz, 0.5).pvalue if nz else 1.0
        flag = ""
        if tt.pvalue < 0.05 and sign_p < 0.05:
            flag = " <-- both tests"
        print(f"{task:<24}{x.mean():7.2f}{dif.mean():+8.2f}{tt.statistic:8.2f}"
              f"{tt.pvalue:9.4f}{sign_p:9.4f}{pos:5d}/{neg:<3d}{flag}")

    # minimum detectable effect at n=15 from the observed paired sd
    print("  MDE at n=15 (two-sided 0.05, 80% power):", end=" ")
    mdes = []
    for task in TASKS:
        if task in a:
            sd = (a.loc[seeds, task].values - base.loc[seeds, task].values).std(ddof=1)
            mdes.append(f"{task[:12]}={2.98 * sd / np.sqrt(15):.2f}")
    print(", ".join(mdes))
