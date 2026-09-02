#!/usr/bin/env bash
# Progress of the parallel session's Table 3 MTEB job, and confirmation that
# the two newly-finished queue items are complete before we analyse them.
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== queue_mteb_table3.sh plan ==="
grep -E 'configs|seeds|for |SEEDS|CONFIGS|ARMS' queue_mteb_table3.sh | head -20

echo
echo "=== mteb_table3.csv progress ==="
~/glotenv/bin/python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open('results/mteb_table3.csv')))
m = [r for r in rows if r.get('task') == 'mteb']
print(f"  total rows {len(rows)}, mteb rows {len(m)}")
by_arm = collections.Counter(r['arm'] for r in m)
print("  mteb rows per arm:", dict(by_arm))
seeds = collections.defaultdict(set)
tasks = collections.defaultdict(set)
for r in m:
    seeds[r['arm']].add(int(r['seed']))
    tasks[r['arm']].add(r['mteb_task'])
for a in sorted(seeds):
    print(f"    {a:<14} seeds={sorted(seeds[a])}  tasks={len(tasks[a])}")
print("  distinct tasks seen:", sorted({r['mteb_task'] for r in m}))
PY

echo
echo "=== decoder B/C campaign completeness ==="
~/glotenv/bin/python - <<'PY'
import csv, collections
rows = list(csv.DictReader(open('results/campaign_decoder_stsb_BC.csv')))
c = [r for r in rows if r.get('stage') == 'confirm']
print(f"  rows {len(rows)}, confirm {len(c)}")
d = collections.defaultdict(set)
for r in c:
    d[r['arm']].add(int(r['seed']))
for a in sorted(d):
    print(f"    {a:<9} n={len(d[a])} seeds {min(d[a])}..{max(d[a])}")
PY
