#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
echo "=== MRPC / RTE wide chain ==="
tail -4 logs/chain_mrpc_rte.log 2>/dev/null
for t in mrpc rte; do
  ls results/campaign_wide_${t}_w*.csv > /dev/null 2>&1 || { echo "  $t: no files"; continue; }
  $PY merge_glob.py "results/campaign_wide_${t}_w*.csv" "results/campaign_wide_${t}.csv" 2>/dev/null \
    || { head -1 $(ls results/campaign_wide_${t}_w*.csv | head -1) > results/campaign_wide_${t}.csv
         for f in results/campaign_wide_${t}_w*.csv; do tail -n +2 "$f"; done >> results/campaign_wide_${t}.csv; }
  n=$(($(wc -l < results/campaign_wide_${t}.csv)-1))
  c=$(awk -F, 'NR>1' results/campaign_wide_${t}.csv | grep -c confirm)
  echo "  $t: $n rows, $c confirm"
  [ "$c" -gt 20 ] && $PY paired_analysis.py "results/campaign_wide_${t}.csv" 2>&1 | head -16
done
echo
echo "=== factorial completion ==="
$PY - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open("results/factorial_scale.csv")))
d = collections.Counter((r["backbone"], r["density"], r["scale"]) for r in rows)
missing = [k for b in ("bert_final","mbert_L12","mbert_final","roberta_final")
           for dn in ("abs06","q05") for sc in ("none","rms","median")
           if d.get((b,dn,sc),0) < 5 for k in [(b,dn,sc,d.get((b,dn,sc),0))]]
print(f"  {len(rows)}/120 rows")
for m in missing:
    print("   incomplete:", m)
if not missing:
    print("   COMPLETE")
EOF
