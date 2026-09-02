#!/usr/bin/env bash
# Recover the exact, already-validated MTEB invocation from the smoke test.
# The queued item 4 pointed at ./run_all_experiments.py, which does not exist
# at the repo root -- the driver lives in hyperglot/ and hyperglot_new/. The
# smoke test is the only invocation known to have worked end to end, so it is
# the authority for path, entry point and argument spelling.
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== launch_smoke_mteb.sh ==="
cat launch_smoke_mteb.sh

echo
echo "=== results/smoke_mteb.csv (header + first rows) ==="
head -3 results/smoke_mteb.csv
echo "rows: $(wc -l < results/smoke_mteb.csv)"

echo
echo "=== tail of logs/smoke_mteb.log ==="
tail -15 logs/smoke_mteb.log

echo
echo "=== do the two driver copies differ? ==="
if diff -q hyperglot/run_all_experiments.py hyperglot_new/run_all_experiments.py >/dev/null 2>&1; then
    echo "  identical"
else
    echo "  DIFFER:"
    diff hyperglot/run_all_experiments.py hyperglot_new/run_all_experiments.py | head -20
fi

echo
echo "=== which copy has the pooling_method fix and the unknown-config abort? ==="
for f in hyperglot/run_all_experiments.py hyperglot_new/run_all_experiments.py; do
    printf '%-42s pooling=%s abort=%s A_threshold=%s\n' "$f" \
        "$(grep -c 'pooling_method=' "$f")" \
        "$(grep -c 'unknown config name' "$f")" \
        "$(grep -c 'A_threshold' "$f")"
done
