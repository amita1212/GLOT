"""Stress test + docmteb + density + delta numbers."""
import json
import os

import pandas as pd

pd.set_option("display.width", 250)

print("=" * 78)
print("STRESS TEST -- hyperbolic arms (results/hyperglot_stress_results.csv)")
print("=" * 78)
d = pd.read_csv("results/hyperglot_stress_results.csv")
print("cols:", [c for c in d.columns if c in
                ("arm", "distractor_ratio", "seed", "acc", "acc_final", "rho_quantile")])
key = "acc" if "acc" in d.columns else "acc_final"
print(d.pivot_table(index="arm", columns="distractor_ratio", values=key,
                    aggfunc=["mean", "std", "count"]).round(4).to_string())

print("\n" + "=" * 78)
print("STRESS TEST -- bag-of-tokens poolers (results/stress_poolers.csv)")
print("=" * 78)
p = pd.read_csv("results/stress_poolers.csv")
print(p.pivot_table(index="arm", columns="distractor_ratio", values=key,
                    aggfunc=["mean", "std", "count"]).round(4).to_string())

print("\n" + "=" * 78)
print("campaign_stress_hard.csv (the 12-arm stress campaign)")
print("=" * 78)
s = pd.read_csv("results_vm/campaign_stress_hard.csv")
print(s.groupby(["arm", "stage"])["score"].agg(["count", "mean", "std"]).round(3).to_string())

print("\n" + "=" * 78)
print("DOC / MTEB")
print("=" * 78)
m = pd.read_csv("results/hyperglot_docmteb_results.csv")
imdb = m[m.task == "imdb"][["arm", "acc", "elapsed_sec", "epochs", "max_length", "seed"]]
print("IMDB:\n", imdb.to_string())
mt = m[m.task == "mteb"][["arm", "mteb_task", "best_val_avg", "elapsed_sec"]]
piv = mt.pivot_table(index="mteb_task", columns="arm", values="best_val_avg")
piv = piv * 100
piv.loc["MEAN"] = piv.mean()
print("\nMTEB (x100):\n", piv.round(3).to_string())
print("\nMTEB wall-clock seconds per task:")
print(mt.pivot_table(index="mteb_task", columns="arm", values="elapsed_sec").to_string())

print("\n" + "=" * 78)
print("BACKBONE RECIPE / factorial_scale (Table fix)")
print("=" * 78)
b = pd.read_csv("results_vm/factorial_scale.csv")
print("cols:", list(b.columns))
gcols = [c for c in ["model", "layer", "arm", "rescale", "input_scale_norm",
                     "tau", "tau_quantile", "setting"] if c in b.columns]
print(b.groupby(gcols)["score"].agg(["count", "mean", "std"]).round(3).to_string())
