#!/usr/bin/env bash
# Put the patched main.py where prewarm/campaign expect it, then run the
# corrected ModernBERT scale test detached.
#
# The previous attempt normalised to mean||x|| = 1, which OVER-corrected: BERT
# fell 0.4554 -> 0.4082 and every ModernBERT layer collapsed to MCC 0.0. The
# target is now BERT's natural scale (15.0), which is what GLOT's lr and
# architecture were actually tuned around.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
if [ -f main.py ]; then
    mv -f main.py hyperglot/main.py
fi
sed -i 's/\r$//' hyperglot/main.py test_scale_fix.sh
~/glotenv/bin/python -m py_compile hyperglot/main.py || { echo "COMPILE FAILED"; exit 1; }
echo "COMPILE_OK"
nohup bash test_scale_fix.sh > logs/scale_fix2.log 2>&1 &
echo "launched scale test pid $!"
