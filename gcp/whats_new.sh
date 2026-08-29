#!/usr/bin/env bash
# Show (a) the ModernBERT density/scale fix results, (b) decoder smoke, (c) structural progress.
cd "$(dirname "$0")" || exit 1
PY=/home/t-amitalfasi/glotenv/bin/python

echo "############ 1. ModernBERT fix: density vs scale ############"
$PY - <<'PYEOF'
import re, json
KEYS = ("mcc", "spearman", "pearson", "acc", "f1")

def score(d):
    m = d.get("metrics", {})
    for k in KEYS:
        if k in m:
            return k, m[k]
    return "?", None

rows, cur = [], {}
for line in open("logs/fix_modernbert.log", errors="ignore"):
    m = re.match(r"--- (\S+) L(\d+) (.*?) ---", line.strip())
    if m:
        cur = {"model": m.group(1), "layer": int(m.group(2)), "flags": m.group(3).strip()}
    if line.startswith("RESULT_JSON"):
        try:
            d = json.loads(line[len("RESULT_JSON"):].strip())
        except Exception:
            continue
        k, v = score(d)
        rows.append((d.get("model", cur.get("model", "?")), cur.get("layer", "?"),
                     cur.get("flags", ""), k, v))

print(f"  {'model':<24} {'L':>3} {'flags':<48} {'metric':>9} {'x100':>8}")
for mdl, L, f, k, v in rows:
    vs = f"{v * 100:.2f}" if isinstance(v, (int, float)) else "-"
    print(f"  {mdl.split('/')[-1]:<24} {L:>3} {f[:48]:<48} {k:>9} {vs:>8}")
PYEOF

echo
echo "############ 2. Decoder (TinyLlama) 9-arm smoke ############"
if [ -e results/_smoke_decoder.csv ]; then
  $PY - <<'PYEOF'
import csv
rows = list(csv.DictReader(open("results/_smoke_decoder.csv")))
if not rows:
    print("  (empty)")
else:
    want = [c for c in rows[0] if any(t in c.lower() for t in
            ("arm", "spearman", "pearson", "mcc", "acc", "density", "seed", "tau"))][:8]
    print("  " + " ".join(f"{c[:14]:>14}" for c in want))
    for r in rows:
        print("  " + " ".join(f"{str(r.get(c, ''))[:14]:>14}" for c in want))
PYEOF
else
  echo "  (smoke still running -- no rows yet)"
fi

echo
echo "############ 3. Structural arms progress (cola, MCC) ############"
if [ -e results/campaign_struct_cola.csv ]; then
  $PY - <<'PYEOF'
import csv, collections, statistics
by = collections.defaultdict(list)
for r in csv.DictReader(open("results/campaign_struct_cola.csv")):
    for k in ("metric_value", "mcc", "score"):
        if r.get(k):
            try:
                by[(r.get("arm"), r.get("stage"))].append(float(r[k]))
            except ValueError:
                pass
            break
print(f"  {'arm':<12} {'stage':<8} {'n':>3} {'best':>8} {'mean':>8}")
for (a, s), v in sorted(by.items()):
    print(f"  {a:<12} {s:<8} {len(v):>3} {max(v):>8.2f} {statistics.mean(v):>8.2f}")
PYEOF
fi
