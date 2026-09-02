#!/usr/bin/env bash
# What do the flags that queue item 4 OMITTED default to?
#
# The validated smoke test passes --tasks (empty) and --mteb_train_file.
# Item 4 passes neither. If --tasks defaults to the full GLUE list, item 4
# would launch the entire GLUE ablation grid as well as MTEB; if
# --mteb_train_file defaults to a path that does not exist, the MS MARCO stage
# writes no checkpoint and the driver refuses to evaluate.
set -u
cd /home/t-amitalfasi/glot || exit 1
D=hyperglot_new/run_all_experiments.py

echo "=== argparse defaults ==="
grep -n -- '--tasks\|--mteb_train_file\|--configs\|--models' "$D" | head

echo
echo "=== GLUE_TASKS / DEFAULT list values ==="
grep -n 'GLUE_TASKS =\|GLUE_SINGLE =\|GLUE_PAIR =\|GLUE_STS =' "$D"

echo
echo "=== does the MS MARCO triplets file exist? ==="
ls -la data/msmarco-triplets.jsonl 2>/dev/null || echo "  MISSING"

echo
echo "=== does the clone have the unknown-config abort yet? ==="
grep -c 'unknown config name' "$D" || true
echo "  (0 = silently filters unknown arms; my fix is not in this clone yet)"

echo
echo "=== clone HEAD vs branch ==="
git -C hyperglot_new log --oneline -1 2>/dev/null || echo "  not a git clone"
