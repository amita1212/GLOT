"""Can a single-seed draw explain our MTEB gap to the published Table 3?
Compares our 15-seed baseline distribution against the published single draw."""
import csv
from collections import defaultdict

PUB = {
    "bert-base-uncased": {"EmotionClassification": 37.15, "SciFact": 24.85,
                          "RedditClustering": 36.30, "AskUbuntuDupQuestions": 50.20,
                          "STS12": 48.62, "TwitterSemEval2015": 56.23,
                          "SummEval": 30.68},
    "roberta-base": {"EmotionClassification": 29.09, "SciFact": 26.05,
                     "RedditClustering": 21.84, "AskUbuntuDupQuestions": 46.87,
                     "STS12": 36.88, "TwitterSemEval2015": 55.98,
                     "SummEval": 30.83},
}

v = defaultdict(list)
seeds = defaultdict(set)
for r in csv.DictReader(open("results/mteb_table3.csv", encoding="utf-8")):
    if r["task"] == "mteb" and r["arm"] == "baseline" and r["mteb_score"]:
        m = r["model"].split("/")[-1]
        v[(m, r["mteb_task"])].append(float(r["mteb_score"]) * 100)
        seeds[m].add(int(r["seed"]))

for m in ["bert-base-uncased", "roberta-base"]:
    print(f"\n=== {m}   our baseline seeds: {sorted(seeds[m])} ===")
    print(f"{'task':<24s}{'pub':>7s}{'our min':>9s}{'our mean':>10s}"
          f"{'our max':>9s}   verdict")
    for t, p in PUB[m].items():
        x = v[(m, t)]
        if not x:
            continue
        lo, hi, mu = min(x), max(x), sum(x) / len(x)
        if lo <= p <= hi:
            verd = "REPRODUCED (published inside our seed range)"
        elif p > hi:
            verd = f"NOT reproduced: published is {p-hi:+.2f} above our best seed"
        else:
            verd = f"we exceed it by {lo-p:+.2f} at our worst seed"
        print(f"{t:<24s}{p:7.2f}{lo:9.2f}{mu:10.2f}{hi:9.2f}   {verd}")
