cd /home/t-amitalfasi/glot
echo "===== docmteb results file ====="
head -1 results/hyperglot_docmteb_results.csv
echo "--- rows ---"
wc -l < results/hyperglot_docmteb_results.csv
echo "--- tasks/models/arms present ---"
python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("results/hyperglot_docmteb_results.csv",
                                newline="", encoding="utf-8")))
print("n rows:", len(rows))
for k in ("task", "model", "arm", "score", "stage", "seed"):
    vals = sorted({(r.get(k) or "")[:40] for r in rows})
    print(f"  {k:8s}: {vals[:12]}")
print()
print("sample rows:")
for r in rows[:8]:
    print("  ", {k: r.get(k) for k in ("task", "model", "arm", "seed", "score")})
PY

echo
echo "===== how does main.py handle mteb / imdb? ====="
grep -n "mteb\|imdb\|msmarco\|contrastive\|MultipleNegatives" main.py | head -40
