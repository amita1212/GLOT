"""Paired MTEB deltas vs baseline, per backbone. Only seeds where both arms
scored the same task are used, so a partially-complete block is honest."""
import csv, math
from collections import defaultdict

PATH = "results/mteb_table3.csv"
TASKS = ["EmotionClassification", "SciFact", "RedditClustering",
         "AskUbuntuDupQuestions", "STS12", "TwitterSemEval2015", "SummEval"]


def sign_p(pos, n):
    """Exact two-sided sign test."""
    if n == 0:
        return 1.0
    k = min(pos, n - pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


cell = {}
for r in csv.DictReader(open(PATH, encoding="utf-8")):
    if r["task"] != "mteb" or not r["mteb_task"] or not r["mteb_score"]:
        continue
    cell[(r["model"], r["mteb_task"], r["arm"], int(r["seed"]))] = \
        float(r["mteb_score"]) * 100

for model in sorted({k[0] for k in cell}):
    arms = sorted({k[2] for k in cell if k[0] == model})
    if "baseline" not in arms or len(arms) < 2:
        print(f"\n{model}: only {arms} -- skipping")
        continue
    seeds = defaultdict(set)
    for (m, t, a, s) in cell:
        if m == model:
            seeds[a].add(s)
    print(f"\n{'='*72}\n{model}")
    print("  seeds per arm: " + ", ".join(f"{a}={len(seeds[a])}" for a in arms))
    for a in [x for x in arms if x != "baseline"]:
        print(f"\n  --- {a} vs baseline ---")
        print(f"    {'task':<24s}{'base':>7s}{'delta':>9s}{'n':>4s}"
              f"{'p/n':>8s}{'t':>8s}{'sign p':>9s}")
        for t in TASKS:
            common = sorted(seeds[a] & seeds["baseline"])
            pr = [(cell[(model, t, "baseline", s)], cell[(model, t, a, s)])
                  for s in common
                  if (model, t, "baseline", s) in cell and (model, t, a, s) in cell]
            if len(pr) < 2:
                print(f"    {t:<24s}{'--':>7s}{'--':>9s}{len(pr):>4d}")
                continue
            base = sum(x for x, _ in pr) / len(pr)
            d = [y - x for x, y in pr]
            n = len(d)
            mean = sum(d) / n
            sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1))
            tt = mean / (sd / math.sqrt(n)) if sd > 0 else float("inf")
            pos = sum(1 for v in d if v > 0)
            print(f"    {t:<24s}{base:7.2f}{mean:+9.2f}{n:>4d}"
                  f"{f'{pos}/{n-pos}':>8s}{tt:8.2f}{sign_p(pos, n):9.4f}")
