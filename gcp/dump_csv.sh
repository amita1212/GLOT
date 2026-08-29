#!/usr/bin/env bash
# Dump a results CSV as "run_key score density elapsed" for quick inspection.
set -u
cd /home/t-amitalfasi/glot
F="${1:-results/e1_curvature.csv}"
~/glotenv/bin/python - "$F" <<'PY'
import csv, sys, os
p = sys.argv[1]
if not os.path.exists(p):
    print("missing:", p); raise SystemExit
rows = list(csv.DictReader(open(p)))
print(f"{len(rows)} rows in {p}")
for r in rows:
    print(f"  {r.get('run_key',''):<44} {r.get('score',''):>8} "
          f"dens={r.get('mean_density','')!s:<9} {r.get('elapsed_sec','')}s")
PY
