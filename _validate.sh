#!/usr/bin/env bash
# Validate the control arms actually do what the paper says they do.
cd /home/t-amitalfasi/glot
~/glotenv/bin/python - <<'EOF'
import csv, glob, statistics as st, collections

print("=" * 84)
print("CHECK 1: does no_graph actually have ~no edges, and paper_tau ~complete?")
print("  (paper claims no_graph 'raises tau until essentially no edge survives')")
print("=" * 84)
for pat, label in [("results/campaign_wide_*.csv", "BERT wide"),
                   ("results/campaign_rob_*.csv", "RoBERTa"),
                   ("results/campaign_decoder_*.csv", "TinyLlama")]:
    d = collections.defaultdict(list)
    for f in glob.glob(pat):
        if any(k in f for k in ("_w0", "_w1", "_w2", "_w3", "_w9")):
            continue
        for r in csv.DictReader(open(f)):
            if r.get("stage") != "confirm":
                continue
            try:
                d[(r["arm"], r["task"])].append(float(r["mean_density"]))
            except (ValueError, KeyError, TypeError):
                pass
    if not d:
        continue
    print(f"\n--- {label} ---")
    print(f"  {'arm':13s} {'task':6s} {'n':>3} {'mean dens':>10} {'min':>8} {'max':>8}")
    for (arm, task), v in sorted(d.items()):
        print(f"  {arm:13s} {task:6s} {len(v):3d} {st.mean(v):10.4f} "
              f"{min(v):8.4f} {max(v):8.4f}")

print()
print("=" * 84)
print("CHECK 2: are density values still >1 anywhere? (the pre-patch metric bug")
print("  counted self-loops in the numerator but not the denominator)")
print("=" * 84)
bad = 0
tot = 0
for f in glob.glob("results/*.csv"):
    for r in csv.DictReader(open(f)):
        try:
            v = float(r["mean_density"])
        except (ValueError, KeyError, TypeError):
            continue
        tot += 1
        if v > 1.0:
            bad += 1
print(f"  {bad} of {tot} recorded densities exceed 1.0")
print("  -> any nonzero count means those rows predate the metric fix and their")
print("     density column is inflated by ~1/(n-1); scores are unaffected.")

print()
print("=" * 84)
print("CHECK 3: seed sets identical across arms? (paired tests assume this)")
print("=" * 84)
for f in ["results/campaign_wide_cola.csv", "results/campaign_wide_stsb.csv",
          "results/campaign_rob_cola.csv", "results/campaign_rob_stsb.csv"]:
    try:
        rows = [r for r in csv.DictReader(open(f)) if r.get("stage") == "confirm"]
    except FileNotFoundError:
        continue
    per = collections.defaultdict(set)
    for r in rows:
        per[r["arm"]].add(r["seed"])
    sizes = {a: len(s) for a, s in per.items()}
    same = len({frozenset(s) for s in per.values()}) == 1
    print(f"  {f:38s} arms={len(per)} sizes={sorted(set(sizes.values()))} "
          f"identical_seed_sets={same}")

print()
print("=" * 84)
print("CHECK 4: duplicate run_keys? (would double-count a seed in a paired test)")
print("=" * 84)
for f in glob.glob("results/campaign_*.csv"):
    keys = [r.get("run_key") for r in csv.DictReader(open(f)) if r.get("run_key")]
    if len(keys) != len(set(keys)):
        dup = [k for k, c in collections.Counter(keys).items() if c > 1]
        print(f"  DUPLICATES in {f}: {len(dup)} keys, e.g. {dup[:3]}")
print("  (no output above = clean)")
EOF
