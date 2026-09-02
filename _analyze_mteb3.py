"""MTEB Table 3, in flight. What is complete, and what does it already say?

Design: 6 arms (baseline, A, B, C, AC, ABC) x 15 seeds x 7 Table-3 tasks,
BERT first, then two more backbones. Pooler contrastively trained on MS MARCO
before each evaluation, which is the stage the old tab:docmteb never ran.

Only arms COMPLETE at 15 seeds are tested. Every arm shares the seed set, so
deltas are paired and within one campaign.

The seven metrics are not commensurable (accuracy, ndcg@10, v_measure,
map@1000, cosine_spearman, max_ap), so there is deliberately no row average.
"""
import csv
import collections
from scipy import stats

ARMS = ["baseline", "A_threshold", "B_threshold", "C_threshold",
        "AC_threshold", "ABC_threshold"]
SHORT = {"baseline": "baseline", "A_threshold": "A", "B_threshold": "B",
         "C_threshold": "C", "AC_threshold": "AC", "ABC_threshold": "ABC"}

rows = [r for r in csv.DictReader(open("results/mteb_table3.csv", encoding="utf-8"))
        if r.get("task") == "mteb"]

# arm -> task -> seed -> score   (de-duplicated)
d = collections.defaultdict(lambda: collections.defaultdict(dict))
models = set()
for r in rows:
    models.add(r["model"])
    try:
        d[r["arm"]][r["mteb_task"]][int(r["seed"])] = float(r["mteb_score"]) * 100
    except (ValueError, KeyError):
        pass

tasks = sorted({r["mteb_task"] for r in rows})
print(f"backbones present: {sorted(models)}")
print(f"mteb rows {len(rows)}; tasks {len(tasks)}")
print("\n=== completeness (seeds per arm x task) ===")
print(f"{'arm':<10}" + "".join(f"{t[:11]:>13}" for t in tasks))
for a in ARMS:
    if a in d:
        print(f"{SHORT[a]:<10}" + "".join(f"{len(d[a].get(t, {})):>13}" for t in tasks))

done = [a for a in ARMS if a in d and all(len(d[a].get(t, {})) >= 15 for t in tasks)]
print(f"\narms complete at 15 seeds: {[SHORT[a] for a in done]}")

print("\n=== absolute means (x100), complete arms only ===")
print(f"{'task':<26}" + "".join(f"{SHORT[a]:>11}" for a in done))
for t in tasks:
    line = f"{t:<26}"
    for a in done:
        v = list(d[a][t].values())
        line += f"{sum(v)/len(v):>11.2f}"
    print(line)

if "baseline" in done:
    print("\n=== paired deltas vs baseline (both-tests rule) ===")
    n_arms = len(done) - 1
    alpha = 0.05 / n_arms if n_arms else 0.05
    print(f"    Bonferroni over {n_arms} arms -> alpha={alpha:.4f}\n")
    print(f"{'task':<26}" + "".join(f"{SHORT[a]:>18}" for a in done if a != "baseline"))
    for t in tasks:
        line = f"{t:<26}"
        for a in done:
            if a == "baseline":
                continue
            common = sorted(set(d[a][t]) & set(d["baseline"][t]))
            x = [d[a][t][s] for s in common]
            y = [d["baseline"][t][s] for s in common]
            dif = [p - q for p, q in zip(x, y)]
            m = sum(dif) / len(dif)
            _, pt = stats.ttest_rel(x, y)
            pos = sum(1 for z in dif if z > 0)
            neg = sum(1 for z in dif if z < 0)
            ps = min(1.0, 2 * stats.binom.cdf(min(pos, neg), pos + neg, 0.5)) if pos + neg else 1.0
            mark = "**" if (pt < alpha and ps < alpha) else ("*" if (pt < .05 and ps < .05) else "")
            line += f"{m:>+11.2f}{mark:<2}{pos:>2}/{neg:<2}"
        print(line)
    print("\n  ** significant on both tests after Bonferroni; * both tests uncorrected")

print("\n=== seed sd per task (baseline) -- can this benchmark resolve anything? ===")
for t in tasks:
    v = list(d["baseline"][t].values())
    if len(v) > 1:
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        mde = 2.05 * sd / len(v) ** 0.5 * (2 ** 0.5)
        print(f"  {t:<26} mean {m:7.2f}  sd {sd:5.2f}  approx MDE {mde:5.2f}")
