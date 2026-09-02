"""Recompute every number short.tex claims, straight from the campaign CSVs.

Nothing here is copied from the paper. Each block prints what the data says;
comparison against the draft is done by eye afterwards.
"""
import glob
import os
from math import comb, sqrt

import pandas as pd

RES = "results_vm"

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 20: 2.086, 30: 2.042}


def t_crit(n):
    df = max(1, n - 1)
    return T_CRIT_95.get(df, T_CRIT_95[min(T_CRIT_95, key=lambda k: abs(k - df))])


def sign_p(d):
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def confirm(path):
    d = pd.read_csv(path)
    d = d[d["stage"] == "confirm"].copy()
    # de-duplicate on run_key: shard files and merged file can overlap
    d = d.drop_duplicates(subset="run_key", keep="last")
    return d


def arm_table(path, label):
    d = confirm(path)
    if d.empty:
        print(f"\n### {label}: NO confirm rows")
        return
    piv = d.pivot_table(index="seed", columns="arm", values="score")
    print(f"\n### {label}   file={os.path.basename(path)}  metric={d['metric'].iloc[0]}")
    print(f"    seeds present: {sorted(piv.index.tolist())}  (n={len(piv)})")
    if "baseline" not in piv.columns:
        print("    !! no baseline arm")
        print(piv.describe().T[["count", "mean", "std"]].to_string())
        return
    base = piv["baseline"]
    rows = []
    for arm in piv.columns:
        s = piv[arm].dropna()
        common = s.index.intersection(base.dropna().index)
        diffs = (piv.loc[common, arm] - base.loc[common]).dropna()
        n = len(diffs)
        if arm == "baseline":
            rows.append((arm, len(s), s.mean(), s.std(ddof=1), 0.0, None, None, "", ""))
            continue
        if n < 2:
            rows.append((arm, len(s), s.mean(), s.std(ddof=1), float("nan"),
                         None, None, "", "n<2"))
            continue
        m = diffs.mean()
        se = diffs.std(ddof=1) / sqrt(n)
        t = m / se if se > 0 else float("nan")
        p = sign_p(diffs.tolist())
        pos = int((diffs > 0).sum())
        neg = int((diffs < 0).sum())
        sig_t = abs(t) > t_crit(n)
        sig_s = p < 0.05
        rows.append((arm, len(s), s.mean(), s.std(ddof=1), m, t, p,
                     f"{pos}/{neg}", "BOTH" if (sig_t and sig_s) else
                     ("t-only" if sig_t else ("sign-only" if sig_s else "n.s."))))
    print(f"    {'arm':<14}{'n':>3}{'mean':>9}{'sd':>7}{'delta':>9}{'t':>8}"
          f"{'signP':>10}{'p/n':>8}  flag")
    for a, n, mu, sd, dm, t, p, pn, fl in sorted(rows, key=lambda r: -r[4]):
        ts = f"{t:8.2f}" if t is not None and t == t else " " * 8
        ps = f"{p:10.5f}" if p is not None else " " * 10
        print(f"    {a:<14}{n:>3}{mu:>9.3f}{sd:>7.3f}{dm:>+9.3f}{ts}{ps}{pn:>8}  {fl}")
    ses = [abs(r[4] / r[5]) for r in rows if r[5] not in (None, 0) and r[5] == r[5]]
    if ses:
        med = sorted(ses)[len(ses) // 2]
        nmax = max(r[1] for r in rows)
        print(f"    -> median paired SE {med:.3f}; MDE at n={nmax} = "
              f"{t_crit(nmax) * med:.3f}")
    nb = len([r for r in rows if r[0] != "baseline"])
    print(f"    -> arms tested vs baseline: {nb}  => Bonferroni alpha = {0.05 / nb:.4f}")


def tuning_winner(path, label):
    """Which configuration won the tuning stage for each arm."""
    d = pd.read_csv(path)
    d = d[d["stage"] == "tune"]
    if d.empty:
        return
    print(f"\n### selected configs {label}")
    for arm, g in d.groupby("arm"):
        best = g.loc[g["score"].idxmax()]
        print(f"    {arm:<14} best={best['score']:6.2f} of {len(g):3d} trials | "
              f"{best['detail']}")


def dead_lr(paths):
    print("\n### trials that fail to train, by learning rate")
    allrows = []
    for p in paths:
        d = pd.read_csv(p)
        d = d[d["stage"] == "tune"]
        if "lr" not in d.columns:
            continue
        d = d.assign(src=os.path.basename(p))
        allrows.append(d[["src", "task", "model", "lr", "score"]])
    if not allrows:
        return
    d = pd.concat(allrows)
    g = d.groupby("lr").apply(
        lambda x: pd.Series({"n": len(x), "zero_or_neg": int((x["score"] <= 0).sum())}),
        include_groups=False)
    g["pct"] = 100 * g["zero_or_neg"] / g["n"]
    print(g.to_string())
    print("\n    per (model, task, lr):")
    g2 = d.groupby(["model", "task", "lr"]).apply(
        lambda x: pd.Series({"n": len(x), "zero": int((x["score"] <= 0).sum())}),
        include_groups=False)
    g2["pct"] = (100 * g2["zero"] / g2["n"]).round(1)
    print(g2[g2["n"] > 0].to_string())


def factorial():
    p = f"{RES}/factorial_geom_cola.csv"
    d = confirm(p)
    print("\n### Stage C 2x2 factorial (CoLA MCC)")
    piv = d.pivot_table(index="seed", columns="arm", values="score")
    print(f"    seeds n={len(piv)}  arms={list(piv.columns)}")
    print(piv.agg(["count", "mean", "std"]).T.to_string())
    dens = d.groupby("arm")["mean_density"].mean()
    print("\n    realised mean edge density per cell:")
    print(dens.to_string())
    # need the two pre-existing cells: baseline and C from campaign_wide_cola
    w = confirm(f"{RES}/campaign_wide_cola.csv")
    wp = w.pivot_table(index="seed", columns="arm", values="score")
    cells = {}
    for name, col in [("base_euc", ("wide", "baseline")), ("base_hyp", ("fact", None)),
                      ("C_euc", ("fact", None)), ("C_hyp", ("wide", "C"))]:
        pass
    print("\n    available factorial arm names:", list(piv.columns))
    print("    wide arm names:", list(wp.columns))
    return piv, wp


def density_check():
    print("\n### realised density of the tuned winners (confirm rows)")
    for t in ["cola", "stsb", "mrpc", "rte"]:
        p = f"{RES}/campaign_wide_{t}.csv"
        if not os.path.exists(p):
            continue
        d = confirm(p)
        g = d.groupby("arm").agg(density=("mean_density", "mean"),
                                 q=("tau_quantile", "mean"),
                                 rho=("rho_quantile", "mean") if "rho_quantile" in d else ("mean_density", "mean"))
        print(f"  -- {t}")
        print(g.round(4).to_string())


if __name__ == "__main__":
    print("=" * 78)
    print("BERT wide campaigns")
    print("=" * 78)
    for t in ["cola", "stsb", "mrpc", "rte"]:
        arm_table(f"{RES}/campaign_wide_{t}.csv", f"BERT {t}")

    print("\n" + "=" * 78)
    print("RoBERTa")
    print("=" * 78)
    for t in ["cola", "stsb"]:
        # merge the calibration campaign and the hyperbolic fill-in
        a = confirm(f"{RES}/campaign_rob_{t}.csv")
        b = confirm(f"{RES}/campaign_robfill_{t}.csv")
        m = pd.concat([a, b]).drop_duplicates(subset="run_key", keep="last")
        m.to_csv(f"{RES}/_merged_rob_{t}.csv", index=False)
        arm_table(f"{RES}/_merged_rob_{t}.csv", f"RoBERTa {t} (merged)")

    print("\n" + "=" * 78)
    print("Decoder")
    print("=" * 78)
    for t in ["stsb", "cola"]:
        p = f"{RES}/campaign_decoder_{t}.csv"
        if os.path.exists(p):
            arm_table(p, f"TinyLlama {t}")

    print("\n" + "=" * 78)
    factorial()
    density_check()
    dead_lr(sorted(glob.glob(f"{RES}/campaign_wide_*.csv")) +
            sorted(glob.glob(f"{RES}/campaign_rob*.csv")))

    print("\n" + "=" * 78)
    print("selected configurations")
    print("=" * 78)
    for t in ["cola", "stsb", "mrpc", "rte"]:
        tuning_winner(f"{RES}/campaign_wide_{t}.csv", f"BERT {t}")
