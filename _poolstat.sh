#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
~/glotenv/bin/python - <<'EOF'
import csv, collections, statistics as st, itertools, math

rows = list(csv.DictReader(open("results/stress_poolers.csv")))
key = next(k for k in ("score","acc","eval_acc","accuracy") if k in rows[0])
d = collections.defaultdict(dict)          # (arm, ratio) -> {seed: score}
for r in rows:
    try: v = float(r[key])
    except (TypeError, ValueError): continue
    d[(r["arm"], float(r["distractor_ratio"]))][r["seed"]] = v*100 if v <= 1.0 else v

ratios = sorted({k[1] for k in d})
arms = sorted({k[0] for k in d})

def sign_p(pos, n):
    if n == 0: return 1.0
    tot = sum(math.comb(n, k) for k in range(0, n+1))
    ext = sum(math.comb(n, k) for k in range(0, n+1) if k >= max(pos, n-pos))
    return min(1.0, 2*ext/tot)

print("="*78)
print("PAIRED, WITHIN OUR OWN DATA: every pooler vs full GLOT, same seeds")
print("  (this comparison does NOT depend on matching the published numbers)")
print("="*78)
for ratio in ratios:
    print(f"\n--- {int(ratio*100)}% distractors ---")
    ref = d[("glot", ratio)]
    print(f"  {'arm':10s} {'mean':>7s} {'delta':>8s} {'pos/neg':>9s} {'signP':>8s}")
    for a in sorted(arms, key=lambda a: -st.mean(list(d[(a,ratio)].values() or [0]))):
        cur = d[(a, ratio)]
        common = sorted(set(cur) & set(ref))
        if not common: continue
        diffs = [cur[s]-ref[s] for s in common]
        pos = sum(1 for x in diffs if x > 0); neg = sum(1 for x in diffs if x < 0)
        mark = "" if a == "glot" else f"{st.mean(diffs):+8.2f} {pos:4d}/{neg:<4d} {sign_p(pos,pos+neg):8.3f}"
        print(f"  {a:10s} {st.mean(list(cur.values())):7.1f} " + (mark or "   (ref)"))

print()
print("="*78)
print("DIVERGENCE FROM THE PUBLISHED TABLE, per pooler")
print("  static poolers vs TRAINED poolers -- do they diverge differently?")
print("="*78)
pub = {"cls":[70.8,58.2,57.2,67.6], "mean":[68.0,58.6,64.2,53.4],
       "max":[57.4,50.8,51.6,50.4], "adapool":[91.4,78.8,65.6,61.6],
       "glot":[97.2,97.0,97.8,98.8]}
kind = {"cls":"static","mean":"static","max":"static",
        "adapool":"TRAINED","glot":"TRAINED","glot_K0":"TRAINED"}
print(f"  {'arm':10s} {'kind':8s} " + " ".join(f"{int(r*100):>12d}%" for r in ratios))
for a in ("cls","mean","max","adapool","glot"):
    cells = []
    for i, r in enumerate(ratios):
        ours = st.mean(list(d[(a,r)].values()))
        cells.append(f"{ours-pub[a][i]:+12.1f}")
    print(f"  {a:10s} {kind[a]:8s} " + " ".join(cells))
print("\n  (positive = we score HIGHER than the published value)")
EOF
