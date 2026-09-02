"""Did RoBERTa RTE actually tune, and did the search move?"""
import pandas as pd

d = pd.read_csv("results_vm/campaign_t1_rob_rte.csv")
print("rows by stage:", dict(d.groupby("stage").size()))
print("tune rows per arm:", dict(d[d.stage == "tune"].groupby("arm").size()))
print()

# what the search actually explored, and what it picked
knobs = [c for c in ["lr", "num_layers", "gat_hidden_dim", "proj_dim",
                     "tau_quantile", "rho_quantile", "jk_mode", "weight_decay",
                     "scorer_hidden", "curvature"] if c in d.columns]

t = d[d.stage == "tune"]
print("distinct values SEARCHED across all tuning trials:")
for k in knobs:
    v = sorted(t[k].dropna().unique().tolist())
    print(f"  {k:16s} {len(v):2d} values: {v[:8]}")

print()
print("SELECTED configuration per arm (from the confirm rows):")
cf = d[d.stage == "confirm"]
sel = cf.groupby("arm")[knobs].first()
print(sel.to_string())

print()
print("does the winner differ across arms?")
for k in knobs:
    n = sel[k].nunique(dropna=True)
    print(f"  {k:16s} {n} distinct winner(s) across the 9 arms")

print()
print("tuning max vs confirmation mean (the inflation the paper warns about):")
for arm in ["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC", "no_graph"]:
    tm = t[t.arm == arm].score.max()
    cm = cf[cf.arm == arm].score.mean()
    print(f"  {arm:10s} tuning max {tm:6.2f}   confirm mean {cm:6.2f}   inflation {tm - cm:+6.2f}")
