#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
~/glotenv/bin/python - <<'EOF'
import csv, collections, statistics as st, itertools, math

def mean(v): return sum(v)/len(v)

print("="*90)
print("A. DENSITY x SCALE FACTORIAL  (CoLA MCC, 5 seeds)")
print("   THE question: is the ModernBERT recovery from density-matching or median scaling?")
print("="*90)
rows = list(csv.DictReader(open("results/factorial_scale.csv")))
d = collections.defaultdict(list)
for r in rows:
    d[(r["backbone"], r["density"], r["scale"])].append(float(r["score"]))
bks = ["bert_final", "mbert_L12", "mbert_final", "roberta_final"]
print(f"\n  {'backbone':15s} {'density':7s} | " + " ".join(f"{s:>16s}" for s in ("none","rms","median")))
for b in bks:
    for dens in ("abs06", "q05"):
        cells = []
        for sc in ("none", "rms", "median"):
            v = d.get((b, dens, sc), [])
            cells.append(f"{mean(v):7.2f}+-{st.pstdev(v):4.2f}({len(v)})" if v else "         --     ")
        print(f"  {b:15s} {dens:7s} | " + " ".join(f"{c:>16s}" for c in cells))
    print()

print("  MAIN EFFECTS (averaged over the other factor, per backbone):")
print(f"  {'backbone':15s} {'q05 - abs06':>12s} {'median - none':>14s} {'rms - none':>12s}")
for b in bks:
    def m(dens=None, sc=None):
        v = [x for (bb, dd, ss), vs in d.items() if bb == b
             and (dens is None or dd == dens) and (sc is None or ss == sc) for x in vs]
        return mean(v) if v else float("nan")
    print(f"  {b:15s} {m(dens='q05')-m(dens='abs06'):12.2f} "
          f"{m(sc='median')-m(sc='none'):14.2f} {m(sc='rms')-m(sc='none'):12.2f}")

print()
print("="*90)
print("B. COMPETING POOLERS ON THE STRESS TEST  (accuracy, 5 seeds)")
print("="*90)
rows = list(csv.DictReader(open("results/stress_poolers.csv")))
key = None
for cand in ("score", "acc", "eval_acc", "accuracy"):
    if rows and cand in rows[0]:
        key = cand; break
d = collections.defaultdict(list)
for r in rows:
    try:
        v = float(r[key])
    except (TypeError, ValueError, KeyError):
        continue
    d[(r["arm"], float(r["distractor_ratio"]))].append(v*100 if v <= 1.0 else v)
ratios = sorted({k[1] for k in d})
arms = sorted({k[0] for k in d}, key=lambda a: -mean(d.get((a, ratios[-1]), [0])))
print(f"\n  {'arm':10s} " + " ".join(f"{int(r*100):>13d}%" for r in ratios))
for a in arms:
    cells = []
    for r in ratios:
        v = d.get((a, r), [])
        cells.append(f"{mean(v):5.1f}+-{st.pstdev(v):4.1f}" if v else "     --   ")
    print(f"  {a:10s} " + " ".join(f"{c:>14s}" for c in cells))
print("\n  GLOT paper Table 7 (BERT, single seed) for reference:")
for nm, vs in [("GLOT",[97.2,97.0,97.8,98.8]), ("AdaPool",[91.4,78.8,65.6,61.6]),
               ("[CLS]",[70.8,58.2,57.2,67.6]), ("Mean",[68.0,58.6,64.2,53.4]),
               ("Max",[57.4,50.8,51.6,50.4])]:
    print(f"  {nm:10s} " + " ".join(f"{v:>14.1f}" for v in vs))

print()
print("="*90)
print("C. GNN BACKBONE  +  README-vs-PAPER RECIPE")
print("="*90)
rows = list(csv.DictReader(open("results/backbone_recipe.csv")))
bb = collections.defaultdict(list)
rc = collections.defaultdict(list)
for r in rows:
    (bb if r["block"] == "backbone" else rc)[(r["task"], r["variant"])].append(float(r["score"]))
print("\n  --- GNN backbone (their Table 11 equivalent) ---")
for task in ("cola", "stsb"):
    got = [(v, bb[(task, v)]) for v in ("gat", "gcn", "gin") if bb.get((task, v))]
    if not got: continue
    base = mean(dict(got)["gat"]) if bb.get((task,"gat")) else None
    print(f"  {task}:")
    for v, vals in got:
        dlt = f"{mean(vals)-base:+6.2f}" if base is not None else "   -- "
        print(f"    {v:5s} {mean(vals):7.2f} +- {st.pstdev(vals):4.2f}  (n={len(vals)})  vs gat {dlt}")
print("\n  --- recipe decomposition on STS-B (published 83.86) ---")
for (task, var), vals in sorted(rc.items()):
    tag = "  <= seed 42, their seed" if len(vals) == 1 else ""
    print(f"    {var:22s} {mean(vals):7.2f} +- {st.pstdev(vals):4.2f} (n={len(vals)}){tag}")
EOF
