"""Confirmation means and paired deltas vs baseline for a campaign CSV.
Usage: python tl_cola_analyze.py [path.csv]
Only rows with stage=confirm are used, de-duplicated by run_key."""
import csv, math, sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/campaign_t1_tl_cola.csv"
ORDER = ["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC", "no_graph"]


def sign_p(pos, n):
    if n == 0:
        return 1.0
    k = min(pos, n - pos)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
conf = [r for r in rows if r["stage"] == "confirm" and r["score"]]
print(f"{len(rows)} rows, {len(conf)} confirmation rows; "
      f"metric={sorted({r['metric'] for r in conf})}")

# de-duplicate by run_key, keeping the last write
score = {}
seen = {}
for r in conf:
    seen[r["run_key"]] = r
for r in seen.values():
    score[(r["arm"], int(r["seed"]))] = float(r["score"])

present = {a for a, _ in score}
ARMS = [a for a in ORDER if a in present] + sorted(present - set(ORDER))

print(f"\n{'arm':<10s}{'mean':>8s}{'sd':>7s}{'n':>4s}"
      f"{'delta':>9s}{'p/n':>8s}{'t':>8s}{'sign p':>9s}")
base = {s: v for (a, s), v in score.items() if a == "baseline"}
for a in ARMS:
    vals = {s: v for (x, s), v in score.items() if x == a}
    if not vals:
        print(f"{a:<10s}{'-- missing --':>28s}")
        continue
    xs = list(vals.values())
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1)) if len(xs) > 1 else 0
    line = f"{a:<10s}{m:8.2f}{sd:7.2f}{len(xs):4d}"
    if a != "baseline":
        common = sorted(set(vals) & set(base))
        d = [vals[s] - base[s] for s in common]
        n = len(d)
        if n > 1:
            md = sum(d) / n
            sdd = math.sqrt(sum((v - md) ** 2 for v in d) / (n - 1))
            tt = md / (sdd / math.sqrt(n)) if sdd else float("inf")
            pos = sum(1 for v in d if v > 0)
            line += (f"{md:+9.2f}{f'{pos}/{n-pos}':>8s}{tt:8.2f}"
                     f"{sign_p(pos, n):9.4f}")
    print(line)

print("\nBonferroni: alpha = 0.05 / (number of arms tested against this "
      "baseline within the campaign)")
