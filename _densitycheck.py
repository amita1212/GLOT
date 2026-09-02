"""Is the C-vs-baseline contrast density-matched at CONFIRM time?

Both arms draw sparsity from the same DENSITY_Q grid, but tuning picks a q per
arm independently. If baseline landed on a different q than C, the +1.42 MCC is
confounded with edge density and the paper's 'no density-matched Euclidean
control' caveat stands. If they match, the caveat is too strong.

Reads the confirm-stage rows only. Density is the logged (defective) statistic
-- it over-counts self-loops -- but the offset is identical across arms, so
arm-to-arm comparison is valid.
"""
import csv
import glob
import os
import statistics as st

ARMS = ["baseline", "no_graph", "A", "C", "AC"]
KEYS = ["tau_quantile", "rho_quantile", "graph_metric", "hyperbolic_gnn",
        "hyperbolic_readout", "num_layers", "gat_hidden_dim", "lr",
        "curvature", "graph_curvature", "feature_mode"]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


for path in sorted(glob.glob("results/campaign_wide_*.csv")):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    conf = [r for r in rows if r.get("stage") == "confirm"]
    if not conf:
        continue
    print("=" * 78)
    print(os.path.basename(path), "  confirm rows:", len(conf))
    print("=" * 78)
    for arm in ARMS:
        sub = [r for r in conf if r.get("arm") == arm]
        if not sub:
            continue
        dens = [num(r.get("mean_density")) for r in sub]
        dens = [d for d in dens if d is not None]
        seeds = sorted({r.get("seed") for r in sub})
        # the confirmed config should be identical across seeds; verify that
        cfgs = {r.get("detail") for r in sub}
        print(f"\n  arm={arm}  n={len(sub)}  distinct_configs={len(cfgs)}")
        if dens:
            print(f"    logged density: mean={st.mean(dens):.5f} "
                  f"min={min(dens):.5f} max={max(dens):.5f}")
        sel = {}
        for k in KEYS:
            vals = {r.get(k) for r in sub if r.get(k) not in ("", None)}
            if vals:
                sel[k] = "|".join(sorted(vals))
        print("    selected:", ", ".join(f"{k}={v}" for k, v in sel.items()))
        if len(cfgs) != 1:
            print("    !! more than one config in confirm stage:")
            for c in sorted(cfgs):
                print("      -", c[:150])
    # the actual question
    b = [r for r in conf if r.get("arm") == "baseline"]
    c = [r for r in conf if r.get("arm") == "C"]
    if b and c:
        bd = [num(r["mean_density"]) for r in b if num(r.get("mean_density"))]
        cd = [num(r["mean_density"]) for r in c if num(r.get("mean_density"))]
        bq = {r.get("tau_quantile") for r in b}
        cq = {r.get("tau_quantile") for r in c}
        print("\n  --> baseline q =", bq, " C q =", cq,
              " MATCH" if bq == cq else " *** MISMATCH ***")
        if bd and cd:
            print(f"  --> baseline density {st.mean(bd):.5f} vs "
                  f"C density {st.mean(cd):.5f}  "
                  f"(diff {st.mean(cd) - st.mean(bd):+.5f})")
    print()
