#!/usr/bin/env bash
# Verify then launch the wide sweep.
#
# Checks before spending GPU time, each corresponding to a failure already paid
# for in this project:
#   1. compile
#   2. every WIDE key is a REAL argparse flag in main.py -- campaign.py forwards
#      dict keys verbatim as --key=value, so one typo silently kills an arm
#   3. jk_mode=lstm is NOT in the grid (it accepts then crashes at runtime)
#   4. a one-trial smoke over all nine arms with --wide actually produces scores
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
mkdir -p logs results

echo "=== 1. compile ==="
"$PY" -m py_compile campaign.py hyperglot/main.py || exit 1
echo "COMPILE_OK"

echo
echo "=== 2. do all WIDE keys exist as CLI flags, and is lstm excluded? ==="
"$PY" - <<'PYEOF' || exit 1
import re, sys
sys.path.insert(0, ".")
from campaign import WIDE

src = open("hyperglot/main.py", encoding="utf-8", errors="ignore").read()
flags = set(re.findall(r'add_argument\(\s*"--([A-Za-z0-9_]+)"', src))

missing = [k for k in WIDE if k not in flags]
print("  WIDE keys :", ", ".join(sorted(WIDE)))
print("  missing   :", missing or "none")
if missing:
    print("  FAIL: campaign.py would emit unrecognized --flags")
    sys.exit(1)

# jk_mode declares choices=["cat","lstm","max"]; lstm crashes at runtime.
m = re.search(r'add_argument\(\s*"--jk_mode".*?choices=\[([^\]]*)\]', src, re.S)
declared = re.findall(r'"([^"]+)"', m.group(1)) if m else []
print("  jk_mode declared in code :", declared)
print("  jk_mode in WIDE grid     :", WIDE["jk_mode"])
if "lstm" in WIDE["jk_mode"]:
    print("  FAIL: lstm is in the grid and crashes at runtime")
    sys.exit(1)
bad = [v for v in WIDE["jk_mode"] if v not in declared]
if bad:
    print(f"  FAIL: {bad} not accepted by argparse choices")
    sys.exit(1)
print("  OK")
PYEOF

echo
echo "=== 3. smoke: all arms, 1 trial, --wide ==="
"$PY" campaign.py --target glue --task stsb --model bert-base-uncased \
    --arms baseline no_graph A B C AB AC BC ABC \
    --trials 1 --stage tune --wide --fine_baseline \
    --out results/_smoke_wide.csv 2>&1 \
  | grep -E '\->|FAIL|unrecognized|Traceback|Error' | head -30

echo
echo "=== 4. launch ==="
nohup bash wide_sweep.sh "stsb cola" > logs/wide.log 2>&1 &
echo "launched wide sweep pid $!"
sleep 5
tail -5 logs/wide.log
