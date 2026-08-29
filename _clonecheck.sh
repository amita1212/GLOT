#!/usr/bin/env bash
# Validate that a CLEAN CLONE of the pushed branch is actually runnable.
#
# WHY: the worker scripts were written against hyperglot-l4, where the repo has
# a layout and absolute paths that do not exist on anyone else's machine. This
# clones from GitHub into /tmp and checks the things that would otherwise fail
# on the collaborator's box an hour into setup.
set -u
DEST=/tmp/clonetest
rm -rf "$DEST"
git clone -q --branch hyperglot-stageA https://github.com/amita1212/GLOT.git "$DEST" || {
    echo "CLONE FAILED (private repo? no access?)"; exit 1; }
cd "$DEST"
echo "commit: $(git rev-parse --short HEAD)"

echo
echo "=== files the workers call ==="
fail=0
for f in gcp/prewarm_model.sh paired_analysis.py exp_runner.py main.py \
         campaign.py worker_decoder.sh worker_bert_seeds.sh worker_sst2.sh \
         bootstrap_machine.sh seed_extend.py factorial_geom_full.py \
         gcp/setup_vm.sh; do
    if [ -f "$f" ]; then echo "  OK      $f"; else echo "  MISSING $f"; fail=1; fi
done

echo
echo "=== bash syntax ==="
for s in worker_decoder.sh worker_bert_seeds.sh worker_sst2.sh \
         bootstrap_machine.sh gcp/prewarm_model.sh; do
    if bash -n "$s" 2>/dev/null; then echo "  SYNTAX_OK $s"; else echo "  SYNTAX_FAIL $s"; fail=1; fi
done

echo
echo "=== python compiles ==="
if ~/glotenv/bin/python -m py_compile campaign.py exp_runner.py main.py \
        seed_extend.py factorial_geom_full.py 2>&1; then
    echo "  PY_OK"
else
    echo "  PY_FAIL"; fail=1
fi

echo
echo "=== no absolute /home/<user> paths left in the workers ==="
if grep -n "/home/t-amitalfasi" worker_*.sh bootstrap_machine.sh gcp/prewarm_model.sh; then
    echo "  ^^ STILL HARDCODED"; fail=1
else
    echo "  none found"
fi

echo
echo "=== does exp_runner find main.py in this layout? ==="
~/glotenv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import exp_runner
print("  MAIN =", exp_runner.MAIN)
print("  exists:", os.path.exists(exp_runner.MAIN))
PY

echo
echo "=== arms campaign.py knows about ==="
~/glotenv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
import campaign
arms = getattr(campaign, "ARMS", None)
if arms is None:
    print("  no ARMS symbol")
else:
    print("  ", sorted(arms))
    need = ["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC"]
    missing = [a for a in need if a not in arms]
    print("  worker_decoder.sh needs:", need)
    print("  MISSING:", missing if missing else "none")
PY

echo
if [ "$fail" = "0" ]; then echo "CLEAN CLONE LOOKS RUNNABLE"; else echo "PROBLEMS ABOVE"; fi
