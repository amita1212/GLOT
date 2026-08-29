#!/usr/bin/env bash
# Diagnose the confirmation-stage config reconstruction.
# campaign.py rebuilds each arm's winning config by parsing the `detail` string
# out of the CSV. If that column is empty or misaligned, EVERY arm silently
# re-runs with default flags -- which shows up as bit-identical scores across
# arms that are supposed to differ.
set -u
cd /home/t-amitalfasi/glot
CSV="${1:-results/campaign_stress_hard.csv}"
~/glotenv/bin/python - "$CSV" <<'PY'
import csv, sys
p = sys.argv[1]
rows = list(csv.DictReader(open(p)))
print("HEADER:")
print("  " + ", ".join(rows[0].keys()))
print()
for stage in ("tune", "confirm"):
    for arm in ("W_hyp", "W_depth"):
        sel = [r for r in rows if r["arm"] == arm and r["stage"] == stage]
        if not sel:
            continue
        r = sel[0]
        print(f"{stage:<8} {arm:<9} score={r['score']:<7} "
              f"edge_weight_mode={r.get('edge_weight_mode','<MISSING>')!r}")
        print(f"         detail={r.get('detail','<MISSING>')!r}")
    print()
PY
