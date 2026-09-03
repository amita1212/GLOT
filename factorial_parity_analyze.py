"""Corrective Stage C factorial: decomposition on whatever seeds are complete.
Cells: base_at_base (Euclidean MP, base config), C_at_base (hyperbolic MP,
base config), base_at_C (Euclidean MP, C config), C_at_C (both)."""
import csv, math

PATH = "results/factorial_geom_cola_parity.csv"


def sign_p(pos, n):
    if n == 0:
        return 1.0
    k = min(pos, n - pos)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def stats(d, label):
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    t = m / (sd / math.sqrt(n)) if sd else float("inf")
    pos = sum(1 for v in d if v > 0)
    print(f"  {label:<38s}{m:+8.3f}{sd:7.2f}{t:8.2f}"
          f"{sign_p(pos, n):9.4f}{f'{pos}/{n-pos}':>9s}")


cell = {}
for r in csv.DictReader(open(PATH, encoding="utf-8")):
    if r["stage"] == "confirm" and r["score"]:
        cell[(r["arm"], int(r["seed"]))] = float(r["score"])

arms = ["base_at_base", "C_at_base", "base_at_C", "C_at_C"]
complete = sorted(s for s in {k[1] for k in cell}
                  if all((a, s) in cell for a in arms))
print(f"complete seeds: {len(complete)} of 65 "
      f"({len(cell)} cells written of 260)\n")

print(f"  {'cell':<38s}{'mean':>8s}{'sd':>7s}")
for a in arms:
    xs = [cell[(a, s)] for s in complete]
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))
    print(f"  {a:<38s}{m:8.2f}{sd:7.2f}")

print(f"\n  {'component':<38s}{'delta':>8s}{'sd':>7s}{'t':>8s}"
      f"{'sign p':>9s}{'p/n':>9s}")
stats([cell[("C_at_base", s)] - cell[("base_at_base", s)] for s in complete],
      "geometry alone (at base config)")
stats([cell[("base_at_C", s)] - cell[("base_at_base", s)] for s in complete],
      "configuration alone (Euclidean MP)")
stats([(cell[("C_at_C", s)] - cell[("base_at_C", s)])
       - (cell[("C_at_base", s)] - cell[("base_at_base", s)]) for s in complete],
      "interaction")
stats([cell[("C_at_C", s)] - cell[("base_at_base", s)] for s in complete],
      "TOTAL (C config hyp - base config eucl)")

print("\n  for reference, the pre-fix n=65 factorial reported:")
print("    geometry +0.534 (t=1.65, ns) | config +0.880 | inter +0.271 "
      "| total +1.685")
