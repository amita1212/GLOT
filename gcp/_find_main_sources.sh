#!/usr/bin/env bash
# Locate the source data for the three main.tex-only items, so nothing is
# ported into short.tex on the strength of an old draft's typesetting.
#   1. structural arms  (POS, POS_ONLY, A_POS, KNN)  -> tab:structural
#   2. seed-42 grid sweep selected configs           -> tab:seed42
#   3. RTE spread across configs at seed 42          -> tab:tau-noise
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== files containing structural arm names ==="
grep -l -E 'POS_ONLY|A_POS|(^|,)POS(,|$)' results/*.csv 2>/dev/null || echo "  none in results/"
grep -rl -E 'POS_ONLY|A_POS' --include='*.csv' . 2>/dev/null | head -10

echo
echo "=== distinct arm names across every results CSV ==="
~/glotenv/bin/python - <<'PY'
import csv, glob, collections
arms = collections.Counter()
for f in glob.glob("results/*.csv"):
    try:
        for r in csv.DictReader(open(f, encoding="utf-8")):
            a = r.get("arm") or r.get("config") or ""
            if a:
                arms[a] += 1
    except Exception:
        pass
for a, n in sorted(arms.items()):
    print(f"  {a:<24} {n}")
PY

echo
echo "=== seed-42 grid sweep file? ==="
ls -la results/ | grep -i -E 'seed42|grid|sweep|paper_grid' || echo "  none obvious"
grep -l 'seed42\|seed_42' results/*.csv 2>/dev/null | head
