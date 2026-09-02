#!/usr/bin/env bash
# Where did the old queue and Stage A go?
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== time ==="; date -Is
echo "=== gpu ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== queue / trainer processes ==="
ps -eo pid,etime,args | grep -E 'queue_rest|glotenv/bin/python' | grep -v grep | cut -c1-130 \
    || echo "  none"
echo "=== pid 131091 ==="
ps -p 131091 -o pid=,args= 2>/dev/null | cut -c1-100 || echo "  gone"

echo
echo "=== last 16 lines of the old queue's log ==="
for f in logs/queue.log logs/queue_rest.log nohup.out logs/queue2.log; do
    [ -f "$f" ] && { echo "--- $f ---"; tail -16 "$f"; }
done

echo
echo "=== outputs ==="
for f in results/stageA_n50_cola.csv results/mteb_trained.csv \
         results/campaign_decoder_stsb_BC.csv; do
    if [ -f "$f" ]; then
        printf '%-45s %s rows\n' "$f" "$(($(wc -l < "$f") - 1))"
    else
        printf '%-45s absent\n' "$f"
    fi
done

echo
echo "=== stage A confirm rows (want 100 = 2 arms x 50) ==="
[ -f results/stageA_n50_cola.csv ] && ~/glotenv/bin/python -c "
import csv
r=list(csv.DictReader(open('results/stageA_n50_cola.csv')))
c=[x for x in r if x.get('stage')=='confirm' and x.get('arm') in ('baseline','A')]
print(' confirm rows for baseline+A:', len(c))
for a in ('baseline','A'):
    s={int(x['seed']) for x in c if x['arm']==a}
    print(f'   {a:<9} n={len(s)}  seeds {min(s) if s else 0}..{max(s) if s else 0}')
"

echo
echo "=== item 4 / 5 logs, if they ran ==="
for f in logs/q4_mteb_trained.log logs/q5_decoder_stsb_BC.log; do
    [ -f "$f" ] && { echo "--- $f ---"; tail -8 "$f"; } || echo "--- $f: absent ---"
done
