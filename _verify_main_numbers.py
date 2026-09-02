"""Re-derive every main.tex-only number from the source CSVs.

Nothing from main.tex is copied into short.tex on the strength of the old
draft's typesetting: main.tex was frozen on 30 August and several of its other
numbers are known to be stale. Each claim below is recomputed here, and
anything that does not reproduce is reported as such rather than ported.

  1. tab:structural  -- POS / POS_ONLY / A_POS / KNN on CoLA, tuning stage
  2. tab:seed42      -- best at seed 42 over GLOT's own grid + selected config
  3. tab:tau-noise   -- our CoLA spread at fixed tau; our RTE spread at seed 42
"""
import csv
import glob
import collections

STRUCT = ["baseline", "POS", "POS_ONLY", "A_POS", "KNN"]


def rows(path):
    try:
        return list(csv.DictReader(open(path, encoding="utf-8")))
    except Exception:
        return []


print("=" * 74)
print("1. STRUCTURAL ARMS (main.tex claims: baseline 10/45.32/43.42,")
print("   POS 10/47.08/44.50, POS_ONLY 4/46.80/44.38, A_POS 10/44.87/43.48,")
print("   KNN 4/45.32/44.84)")
print("=" * 74)
src = None
for f in glob.glob("results/*.csv"):
    if any(r.get("arm") in ("POS", "A_POS", "POS_ONLY", "KNN") for r in rows(f)):
        src = f
        break
if not src:
    print("  NO source file found -- cannot verify, do not port")
else:
    print(f"  source: {src}")
    d = collections.defaultdict(list)
    for r in rows(src):
        a = r.get("arm")
        if a in STRUCT:
            try:
                d[a].append((r.get("stage", "?"), float(r["score"])))
            except (ValueError, KeyError):
                pass
    for a in STRUCT:
        if a not in d:
            print(f"  {a:<10} ABSENT")
            continue
        st = collections.Counter(s for s, _ in d[a])
        tune = [v for s, v in d[a] if s == "tune"]
        allv = [v for _, v in d[a]]
        use = tune or allv
        print(f"  {a:<10} n={len(d[a]):<3} stages={dict(st)}  "
              f"best={max(use):6.2f}  mean={sum(use)/len(use):6.2f}  (n_used={len(use)})")

print()
print("=" * 74)
print("2. SEED-42 GRID SWEEPS (main.tex: CoLA 50.52, RTE 59.57, STS-B 82.99;")
print("   78 configurations per task)")
print("=" * 74)
KEYS = ["tau", "num_layers", "gat_hidden_dim", "lr", "jk_mode", "scorer_hidden",
        "proj_dim", "graph_adj", "seed"]
for task, f in [("CoLA", "results/sweep_cola.csv"),
                ("RTE", "results/sweep_rte.csv"),
                ("STS-B", "results/sweep_stsb.csv")]:
    rs = rows(f)
    if not rs:
        print(f"  {task:<6} {f}: MISSING")
        continue
    scored = []
    for r in rs:
        for k in ("score", "mcc", "acc", "spearman", "best_val_avg"):
            if r.get(k):
                try:
                    scored.append((float(r[k]), r))
                    break
                except ValueError:
                    pass
    if not scored:
        print(f"  {task:<6} no numeric score column; columns={list(rs[0])[:12]}")
        continue
    vals = [v for v, _ in scored]
    best, brow = max(scored, key=lambda t: t[0])
    seeds = {r.get("seed") for _, r in scored}
    print(f"  {task:<6} rows={len(rs):<4} scored={len(scored):<4} seeds={sorted(seeds)}")
    print(f"         best={best:.2f}   spread(max-min)={max(vals)-min(vals):.2f}")
    cfg = {k: brow.get(k) for k in KEYS if brow.get(k) not in (None, "")}
    print(f"         best config: {cfg}")

print()
print("=" * 74)
print("3. CoLA spread across 15 seeds at FIXED tau (main.tex: 5.46)")
print("=" * 74)
for f in ("results/campaign_wide_cola.csv", "results/stageA_n50_cola.csv"):
    rs = rows(f)
    if not rs:
        continue
    base = {}
    for r in rs:
        if r.get("stage") == "confirm" and r.get("arm") == "baseline":
            try:
                s = int(r["seed"])
                if s <= 15:
                    base[s] = float(r["score"])
            except (ValueError, KeyError):
                pass
    if base:
        v = list(base.values())
        print(f"  {f}: n={len(v)} baseline confirm seeds<=15  "
              f"min={min(v):.2f} max={max(v):.2f} spread={max(v)-min(v):.2f}")
