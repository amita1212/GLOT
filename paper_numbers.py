"""Every number short.tex needs from the campaigns that finished 3-4 Sep.
Prints absolutes, paired deltas, both tests, MDE, and Bonferroni verdicts."""
import csv, math
from collections import defaultdict

TCRIT15 = 2.145   # t_{0.975, df=14}


def sign_p(pos, n):
    if n == 0:
        return 1.0
    k = min(pos, n - pos)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def tp(t, df):
    """two-sided p for Student t via regularised incomplete beta"""
    t = abs(t)
    x = df / (df + t * t)
    a, b = df / 2, 0.5

    def betacf(a, b, x):
        qab, qap, qam = a + b, a + 1, a - 1
        c, d = 1.0, 1 - qab * x / qap
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        h = d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1 / max(abs(1 + aa * d), 1e-300) * (1 if 1 + aa * d > 0 else -1)
            c = 1 + aa / c
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1 / max(abs(1 + aa * d), 1e-300) * (1 if 1 + aa * d > 0 else -1)
            c = 1 + aa / c
            de = d * c
            h *= de
            if abs(de - 1) < 3e-16:
                break
        return h
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    return (math.exp(lb) * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2)
            else 1 - math.exp(lb) * betacf(b, a, 1 - x) / b)


def campaign(path, label, alpha):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("stage") == "confirm" and r.get("score"):
            rows[r["run_key"]] = r
    sc = {}
    for r in rows.values():
        sc[(r["arm"], int(r["seed"]))] = float(r["score"])
    arms = sorted({a for a, _ in sc})
    order = ["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC", "no_graph"]
    arms = [a for a in order if a in arms] + [a for a in arms if a not in order]
    base = {s: v for (a, s), v in sc.items() if a == "baseline"}
    print(f"\n{'='*76}\n{label}   (alpha = {alpha})")
    print(f"{'arm':<10s}{'mean':>8s}{'sd':>7s}{'delta':>9s}{'p/n':>8s}"
          f"{'t':>8s}{'t-p':>9s}{'sign p':>9s}  verdict")
    ses = []
    for a in arms:
        vals = {s: v for (x, s), v in sc.items() if x == a}
        xs = list(vals.values())
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))
        line = f"{a:<10s}{m:8.2f}{sd:7.2f}"
        if a == "baseline":
            print(line)
            continue
        common = sorted(set(vals) & set(base))
        d = [vals[s] - base[s] for s in common]
        n = len(d)
        md = sum(d) / n
        sdd = math.sqrt(sum((v - md) ** 2 for v in d) / (n - 1))
        ses.append(sdd / math.sqrt(n))
        t = md / (sdd / math.sqrt(n))
        pt, ps = tp(t, n - 1), sign_p(sum(1 for v in d if v > 0), n)
        both = pt < 0.05 and ps < 0.05
        verd = ("BONFERRONI" if both and pt < alpha and ps < alpha
                else "both tests" if both else "ns")
        print(f"{line}{md:+9.2f}"
              f"{f'{sum(1 for v in d if v>0)}/{n-sum(1 for v in d if v>0)}':>8s}"
              f"{t:8.2f}{pt:9.4f}{ps:9.4f}  {verd}")
    ses.sort()
    med = ses[len(ses) // 2] if len(ses) % 2 else (ses[len(ses)//2 - 1] + ses[len(ses)//2]) / 2
    print(f"  minimum detectable effect (median paired SE x {TCRIT15}) = {TCRIT15*med:.3f}")


campaign("results/campaign_t1_tl_cola.csv", "TinyLlama CoLA (9 arms)", 0.05 / 8)
campaign("results/campaign_decoder_stsb_matched.csv",
         "Matched decoder STS-B (baseline/B/C)", 0.05 / 8)
campaign("results/sst2_reduced.csv", "SST-2 reduced (baseline/B/C/BC)", 0.05 / 3)
