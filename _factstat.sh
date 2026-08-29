#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
~/glotenv/bin/python - <<'EOF'
import csv, collections, statistics as st, math

def sign_p(pos, n):
    if n == 0: return 1.0
    tot = 2**n
    ext = sum(math.comb(n,k) for k in range(n+1) if k >= max(pos, n-pos))
    return min(1.0, 2*ext/tot)

def tstat(diffs):
    n = len(diffs)
    if n < 2: return float("nan"), float("nan")
    m = st.mean(diffs); s = st.stdev(diffs)
    se = s/math.sqrt(n)
    return m, (m/se if se else float("nan"))

rows = list(csv.DictReader(open("results/factorial_scale.csv")))
d = collections.defaultdict(dict)
for r in rows:
    d[(r["backbone"], r["density"], r["scale"])][r["seed"]] = float(r["score"])

print("="*86)
print("FACTORIAL, PAIRED BY SEED  (this is what I failed to test before)")
print("="*86)
for b in ("bert_final","mbert_L12","mbert_final","roberta_final"):
    print(f"\n--- {b} ---")
    print(f"  {'contrast':34s} {'mean d':>8s} {'t':>7s} {'pos/neg':>9s} {'signP':>7s}")
    tests = []
    for dens in ("abs06","q05"):
        for sc in ("rms","median"):
            tests.append((f"{dens}: {sc} vs none", (b,dens,sc), (b,dens,"none")))
    for sc in ("none","rms","median"):
        tests.append((f"{sc}: q05 vs abs06", (b,"q05",sc), (b,"abs06",sc)))
    for name, a, ref in tests:
        A, R = d.get(a,{}), d.get(ref,{})
        common = sorted(set(A) & set(R))
        if len(common) < 2:
            print(f"  {name:34s}      incomplete ({len(common)} paired seeds)")
            continue
        diffs = [A[s]-R[s] for s in common]
        m, t = tstat(diffs)
        pos = sum(1 for x in diffs if x > 0); neg = sum(1 for x in diffs if x < 0)
        flag = "  <== both tests agree" if (abs(t) > 2.78 and sign_p(pos,pos+neg) <= 0.0625) else ""
        print(f"  {name:34s} {m:8.2f} {t:7.2f} {pos:4d}/{neg:<4d} {sign_p(pos,pos+neg):7.3f}{flag}")
print("\n  n=5 => the exact sign test CANNOT go below p=0.0625. Two-sided t at")
print("  df=4 needs |t|>2.78 for p<0.05. Both must hold before anything is claimed.")

print()
print("="*86)
print("GNN BACKBONE, PAIRED BY SEED")
print("="*86)
rows = list(csv.DictReader(open("results/backbone_recipe.csv")))
bb = collections.defaultdict(dict)
for r in rows:
    if r["block"] == "backbone":
        bb[(r["task"], r["variant"])][r["seed"]] = float(r["score"])
for task in ("cola","stsb"):
    ref = bb.get((task,"gat"), {})
    for v in ("gcn","gin"):
        cur = bb.get((task,v), {})
        common = sorted(set(cur)&set(ref))
        if len(common) < 2: continue
        diffs = [cur[s]-ref[s] for s in common]
        m, t = tstat(diffs)
        pos = sum(1 for x in diffs if x > 0); neg = sum(1 for x in diffs if x < 0)
        print(f"  {task:5s} {v} vs gat: {m:+6.2f}  t={t:5.2f}  {pos}/{neg}  signP={sign_p(pos,pos+neg):.3f}")
EOF
