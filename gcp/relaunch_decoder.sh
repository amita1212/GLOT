#!/usr/bin/env bash
# Relaunch the decoder sweep with PRIORITY (nothing else running).
#
# The full pre-flight (decoder load, is_decoder assert, last-real-token vs
# index -1 norm, all-arms smoke) already passed -- see logs/decoder_verify.log.
# Re-running the 9-arm smoke would cost ~2.5 h, and the new arm list is a strict
# subset of what was verified, so it is skipped here.
set -u
cd /home/t-amitalfasi/glot
PY=~/glotenv/bin/python
mkdir -p logs results

echo "=== compile check ==="
"$PY" -m py_compile campaign.py hyperglot/main.py hyperglot/hyperbolic_graph.py || exit 1
echo "COMPILE_OK"

echo "=== nothing else may be on the GPU ==="
if pgrep -f 'structural_arms.sh|roberta_compare.sh|campaign.py' >/dev/null 2>&1; then
    echo "REFUSING: another campaign is still running:"
    pgrep -af 'structural_arms.sh|roberta_compare.sh|campaign.py' | cut -c1-100
    exit 1
fi
echo "GPU is free"

echo "=== arms that will run ==="
grep -m1 '^ARMS=' decoder_sweep.sh

DEC_NOWAIT=1 nohup bash decoder_sweep.sh "stsb cola" > logs/decoder.log 2>&1 &
echo "launched decoder sweep pid $!"
sleep 5
tail -5 logs/decoder.log 2>/dev/null
