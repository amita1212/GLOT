#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
for f in results/campaign_stress_hard.csv results/stress_warm.csv results/stress_followup.csv results/hyperglot_stress_results.csv; do
  echo "=== $f ==="
  [ -e "$f" ] || { echo MISSING; continue; }
  wc -l < "$f"
  head -1 "$f"
  ~/glotenv/bin/python - "$f" <<'PY'
import csv,sys
from collections import Counter
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows: raise SystemExit("empty")
for k in ("arm","model","distractor_ratio","seed","stage"):
    if k in rows[0]:
        c=Counter(str(r.get(k)) for r in rows)
        print(f"  {k}: {dict(sorted(c.items())[:12])}")
PY
done
