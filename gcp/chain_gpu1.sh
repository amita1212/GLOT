#!/usr/bin/env bash
# GPU 1 chain: finish Table-1 -> corrective Stage C factorial -> matched decoder.
#
# ORDERING IS LOAD-BEARING, FOR TWO REASONS.
#
# 1. The parity fix must NOT arrive while the Table-1 CoLA campaign is running.
#    That campaign started on the pre-fix code; pulling mid-campaign would split
#    it across two code versions, which is the cross-version form of the splice
#    Appendix "A seed does not pin a run" forbids. So we wait for the queue to
#    finish BEFORE pulling.
#
# 2. We detect "finished" by grepping the queue's OWN log for its terminal
#    banner, not by looking for a process. queue_table1_gap.sh documents why:
#    any pgrep/ps pattern that names a process also matches a monitoring command
#    that merely mentions it, and two earlier attempts at a process guard
#    deadlocked against the observer. A file test cannot be fooled that way.
set -uo pipefail
ROOT=/home/t-amitalfasi/glot
PY=/home/t-amitalfasi/glotenv/bin/python
QLOG=$ROOT/logs/queue_table1_gap.log
LOG=$ROOT/logs/chain_gpu1.log
cd "$ROOT" || exit 1

say() { echo "[chain $(date -Is)] $*" | tee -a "$LOG"; }

say "chain started; waiting for the Table-1 queue to finish"

# ---- 1. wait for the Table-1 queue -----------------------------------------
while ! grep -q 'TABLE 1 GAP FINISHED' "$QLOG" 2>/dev/null; do
    sleep 300
done
say "Table-1 queue reported FINISHED"
say "  tl/mrpc should read ABORT (cache marker parked deliberately):"
grep -E 'ABORT|SKIP|DONE|FAILED' "$QLOG" | tail -6 | sed 's/^/    /' | tee -a "$LOG"

# ---- 2. take the parity fix -------------------------------------------------
# The BGU checkout carries queue_table1_gap.sh and prewarm_t1.sh as UNTRACKED
# files, and both are tracked (and therefore incoming) upstream. git refuses to
# fast-forward over an untracked file, so a bare `git pull --ff-only` fails with
# "untracked working tree files would be overwritten". We park them instead of
# deleting them: queue_table1_gap.sh is the exact script that produced the
# Table-1 results, and the paper releases the scripts behind its tables. By the
# time we get here the queue has finished, so moving it is safe -- doing this
# any earlier would pull the script out from under a running job.
say "preparing the working tree for a fast-forward"
git -C "$ROOT" fetch --all --quiet
UP=$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)
say "upstream is $UP"
STAMP=$(date +%Y%m%d-%H%M%S)
git -C "$ROOT" diff --name-only "HEAD..$UP" 2>/dev/null | while read -r f; do
    [ -z "$f" ] && continue
    if [ -e "$ROOT/$f" ] && ! git -C "$ROOT" ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        mv "$ROOT/$f" "$ROOT/$f.local-$STAMP"
        say "  parked untracked $f -> $f.local-$STAMP"
    fi
done

say "pulling the control-parity fix"
if ! git -C "$ROOT" pull --ff-only 2>&1 | tail -3 | sed 's/^/    /' | tee -a "$LOG"; then
    say "PULL FAILED -- chain stops rather than running the old code."
    exit 1
fi
say "now at $(git -C "$ROOT" log --oneline -1)"

# the parked copies should be identical to what just arrived; say so either way
for f in queue_table1_gap.sh prewarm_t1.sh; do
    P="$ROOT/$f.local-$STAMP"
    if [ -f "$P" ] && [ -f "$ROOT/$f" ]; then
        if diff -q "$P" "$ROOT/$f" >/dev/null 2>&1; then
            say "  $f: parked copy is identical to the pulled version"
        else
            say "  $f: parked copy DIFFERS from the pulled version (kept at $P)"
        fi
    fi
done

# ---- 3. refuse to continue unless parity actually holds ---------------------
# The whole point of the corrective factorial is that the two arms differ in
# curvature alone. If the fix did not arrive, running it would burn 10 GPU-hours
# reproducing the defect we are trying to remove.
say "verifying control parity before spending any GPU time"
PARITY=$("$PY" "$ROOT/test_edge_parity.py" 2>&1 | tee -a "$LOG")
if echo "$PARITY" | grep -q 'MATCHED'; then
    say "parity OK (self-loops matched)"
else
    say "PARITY CHECK FAILED -- refusing to run the factorial. Chain stops here."
    exit 1
fi
if ! grep -q 'gat_edge_attr' "$ROOT/main.py"; then
    say "main.py has no gat_edge_attr flag -- fix did not arrive. Stopping."
    exit 1
fi
say "edge-attribute parity flag present"

# ---- 4. corrective Stage C factorial ---------------------------------------
say "START corrective Stage C factorial: 4 cells x 65 seeds, seed-major"
"$PY" -u "$ROOT/factorial_geom.py" --seeds $(seq 1 65) \
    > "$ROOT/logs/factorial_parity.log" 2>&1 \
    && say "DONE factorial" || say "FAILED factorial (see logs/factorial_parity.log)"

# ---- 5. matched decoder STS-B ----------------------------------------------
say "START matched decoder STS-B: baseline/B/C x 15 seeds, one file"
"$PY" -u "$ROOT/decoder_stsb_matched.py" --seeds $(seq 1 15) \
    > "$ROOT/logs/decoder_matched.log" 2>&1 \
    && say "DONE decoder matched" || say "FAILED decoder matched (see logs/decoder_matched.log)"

say "===== GPU1 CHAIN FINISHED ====="
say "results/factorial_geom_cola_parity.csv and results/campaign_decoder_stsb_matched.csv"
