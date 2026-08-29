cd /home/t-amitalfasi/glot
echo "=== workers alive ==="
pgrep -c -f robfill_worker && echo "" || echo "none"
echo
echo "=== rows written so far (tune + confirm) ==="
for f in results/campaign_robfill_*_w*.csv; do
  [ -e "$f" ] || continue
  n=$(($(wc -l < "$f") - 1))
  echo "  $(basename "$f")  $n rows"
done
echo
echo "=== per-worker latest line ==="
for i in 0 1 2; do
  for t in cola stsb; do
    f="logs/robfill_w${i}_${t}.log"
    [ -e "$f" ] && echo "  w$i/$t: $(tail -1 "$f" | cut -c1-110)"
  done
done
echo
echo "=== elapsed since launch ==="
head -1 logs/robfill_all.log
date -Is
echo
echo "=== throughput estimate ==="
python3 - <<'PY'
import csv, glob, os, time
done = 0
secs = []
for f in glob.glob("results/campaign_robfill_*_w*.csv"):
    for r in csv.DictReader(open(f, newline="", encoding="utf-8")):
        done += 1
        try:
            s = float(r.get("elapsed_sec") or 0)
            if s > 0:
                secs.append(s)
        except ValueError:
            pass
TOTAL = 6 * (40 + 15) * 2      # 6 arms x (40 tune + 15 confirm) x 2 tasks
print(f"  completed {done} / {TOTAL} runs ({100*done/TOTAL:.1f}%)")
if secs:
    mean = sum(secs) / len(secs)
    print(f"  mean {mean:.0f}s/run over {len(secs)} timed runs")
    remain = (TOTAL - done) * mean / 3 / 3600   # 3 workers in parallel
    print(f"  ~{remain:.1f} h wall-clock remaining at 3 workers")
PY
