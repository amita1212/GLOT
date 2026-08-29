#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo "=========== BODY LENGTH (must be <= 4 pages before references) ==========="
pdftotext -layout paper/short.pdf /tmp/short.txt 2>/dev/null || \
  { sudo apt-get install -y -qq poppler-utils > /dev/null 2>&1; pdftotext -layout paper/short.pdf /tmp/short.txt; }
awk 'BEGIN{p=1} /\f/{p++} /REFERENCES|References/{print "  references start on page " p; exit}' /tmp/short.txt
awk 'BEGIN{p=1} /\f/{p++} /Experimental protocol/{print "  appendix A starts on page " p; exit}' /tmp/short.txt
echo "  total pages: $(pdfinfo paper/short.pdf | awk '/Pages/{print $2}')"

echo
echo "=========== REMAINING PLACEHOLDERS IN THE PAPER ==========="
grep -n 'PEND\|TODO\|placeholder' paper/short.tex | grep -v newcommand | sed 's/^/  /'

echo
echo "=========== PHASE 2 ==========="
cat logs/phase2.log
for f in factorial poolers bkrec; do
  echo "--- smoke_$f ---"
  grep -E 'SMOKE-OK|FAIL|DONE|Traceback|rror' "logs/smoke_$f.log" 2>/dev/null | tail -6
done
echo "--- live streams ---"
ps -eo pid,etimes,cmd | grep -E 'factorial_scale|stress_poolers|backbone_recipe|campaign.py' | grep -v grep | awk '{print "  " $1, $2"s", $3, $4, $5, $6}'
echo "--- result rows ---"
for f in results/factorial_scale.csv results/stress_poolers.csv results/backbone_recipe.csv; do
  [ -f "$f" ] && echo "  $f  $(($(wc -l < $f)-1)) rows"
done
echo "--- machine ---"
uptime
