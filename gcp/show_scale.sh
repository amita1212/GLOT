#!/usr/bin/env bash
# Compact summary of a scale-fix log: one line per (model, layer, scale_norm).
# Lives in a file because sed capture groups get mangled through
# `gcloud compute ssh --command` on Windows.
set -u
cd /home/t-amitalfasi/glot
LOG="${1:-logs/scale_fix2.log}"
echo "=== $LOG ==="
~/glotenv/bin/python - "$LOG" <<'PY'
import json, re, sys
label = None
for line in open(sys.argv[1], errors="ignore"):
    line = line.rstrip("\n")
    if line.startswith("---"):
        label = line.strip("- ").strip()
    elif line.startswith("RESULT_JSON"):
        try:
            d = json.loads(line[len("RESULT_JSON"):].strip())
        except json.JSONDecodeError:
            continue
        m = d.get("metrics", {})
        print(f"  {label:<62} mcc={m.get('mcc', float('nan')):.4f} "
              f"acc={m.get('acc', float('nan')):.4f}")
    elif line.startswith("####") or line.startswith("DONE"):
        print(line)
PY
echo "--- still running? ---"
pgrep -af 'test_scale_fix|hyperglot/main.py' | cut -c1-70 || echo "  (idle)"
