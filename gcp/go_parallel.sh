#!/usr/bin/env bash
# Stop the serial wide sweep and relaunch it as N parallel workers.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
mkdir -p logs results

echo "=== 1. stop the serial sweep ==="
SELF=$$
OLD=$(pgrep -f 'wide_sweep\.sh|campaign\.py' 2>/dev/null | grep -vw "$SELF" | grep -vw "$PPID")
if [ -n "$OLD" ]; then
    for p in $OLD; do
        printf '  kill %s  %s\n' "$p" "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-80)"
    done
    kill $OLD 2>/dev/null
    sleep 8
    STILL=$(for p in $OLD; do kill -0 "$p" 2>/dev/null && echo "$p"; done)
    [ -n "$STILL" ] && kill -9 $STILL 2>/dev/null
else
    echo "  (nothing running)"
fi

echo
echo "=== 2. preserve any rows the serial run produced ==="
# The serial run wrote campaign_wide_<task>.csv, which is also the MERGE TARGET
# of the parallel run. Move it aside as worker 9's file so merge_wide.py folds it
# back in via run_key dedup instead of overwriting it.
for f in results/campaign_wide_stsb.csv results/campaign_wide_cola.csv; do
    [ -e "$f" ] || continue
    n=$(( $(wc -l < "$f") - 1 ))
    dest="${f%.csv}_w9.csv"
    mv "$f" "$dest"
    echo "  $(basename "$f") -> $(basename "$dest")  ($n rows kept)"
done

echo
echo "=== 3. compile check ==="
"$PY" -m py_compile campaign.py merge_wide.py || exit 1
echo "COMPILE_OK"

echo
echo "=== 4. arm split across workers ==="
"$PY" - <<'PYEOF'
ALL = ["baseline", "no_graph", "A", "B", "C", "AB", "AC", "BC", "ABC"]
N = 4
for i in range(N):
    mine = ALL[i::N]
    print(f"  w{i}: {' '.join(mine):<28} {len(mine)} arms, {len(mine)*55} runs")
tot = sum(len(ALL[i::N]) for i in range(N))
assert tot == len(ALL), "arm split lost an arm"
assert len({a for i in range(N) for a in ALL[i::N]}) == len(ALL), "arm split overlaps"
print(f"  total {tot} arms, disjoint and complete")
PYEOF

echo
echo "=== 5. launch ==="
NW=4 nohup bash wide_parallel.sh "stsb cola" > logs/wide_parallel.log 2>&1 &
echo "launched parallel sweep pid $!"
sleep 10
tail -12 logs/wide_parallel.log
