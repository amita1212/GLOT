#!/usr/bin/env bash
# Status of the wide sweep: is it running, and what lr is each arm picking?
cd "$(dirname "$0")" || exit 1
PY=/home/t-amitalfasi/glotenv/bin/python

echo "=== PROCESSES ==="
pgrep -af 'wide_sweep|launch_wide|campaign.py' | grep -v wide_status | cut -c1-105
echo

echo "=== SMOKE (all arms, 1 trial, --wide) ==="
if [ -e results/_smoke_wide.csv ]; then
  $PY - <<'PYEOF'
import csv
rows = list(csv.DictReader(open("results/_smoke_wide.csv")))
print(f"  {len(rows)} / 9 arms")
for r in rows:
    d = r.get("detail", "")
    g = {p.split("=")[0]: p.split("=", 1)[1] for p in d.split(";") if "=" in p}
    print(f"  {r.get('arm',''):<10} {r.get('score',''):>7}  "
          f"lr={g.get('lr','?'):<8} jk={g.get('jk_mode','?'):<5} "
          f"K={g.get('num_layers','?'):<3} h={g.get('gat_hidden_dim','?')}")
PYEOF
else
  echo "  (no rows yet)"
fi
echo

echo "=== WIDE CAMPAIGN PROGRESS ==="
for f in results/campaign_wide_*.csv; do
  [ -e "$f" ] || { echo "  (not started)"; break; }
  printf '  %-40s %4d rows\n' "$(basename "$f")" "$(( $(wc -l < "$f") - 1 ))"
done
echo

echo "=== BEST lr PER ARM (tuning stage; the headline question) ==="
$PY - <<'PYEOF'
import csv, glob, os
for f in sorted(glob.glob("results/campaign_wide_*.csv")):
    best = {}
    for r in csv.DictReader(open(f)):
        if r.get("stage") != "tune":
            continue
        try:
            v = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
        a = r.get("arm")
        if a not in best or v > best[a][0]:
            best[a] = (v, r.get("detail", ""))
    if not best:
        continue
    print(f"  {os.path.basename(f)}")
    for a, (v, d) in sorted(best.items()):
        g = {p.split("=")[0]: p.split("=", 1)[1] for p in d.split(";") if "=" in p}
        print(f"    {a:<10} best={v:6.2f}  lr={g.get('lr','?'):<8} "
              f"wd={g.get('weight_decay','?'):<8} jk={g.get('jk_mode','?'):<5} "
              f"K={g.get('num_layers','?'):<3} h={g.get('gat_hidden_dim','?')}")
PYEOF
echo

echo "=== LOG ==="
tail -8 logs/wide.log 2>/dev/null || echo "  (no log yet)"
