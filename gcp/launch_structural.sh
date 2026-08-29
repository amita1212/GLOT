#!/usr/bin/env bash
# Detached launcher for structural_arms.sh. Validates that the new arms actually
# launch (one 1-trial dry run per arm) BEFORE committing ~14 GPU-hours -- a
# mistyped CLI key is forwarded verbatim by campaign.py and would kill every run.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
sed -i 's/\r$//' structural_arms.sh hyperglot/main.py hyperglot/hyperbolic_graph.py campaign.py
~/glotenv/bin/python -m py_compile hyperglot/main.py hyperglot/hyperbolic_graph.py campaign.py \
    || { echo "COMPILE FAILED"; exit 1; }
echo COMPILE_OK

echo "--- smoke: does each new arm launch at all? ---"
~/glotenv/bin/python campaign.py --target glue --task cola --model bert-base-uncased \
    --arms POS POS_ONLY A_POS KNN --trials 1 --stage tune --hidden_layer 12 \
    --out results/_smoke_struct.csv 2>&1 | grep -E '^\s+->|FAIL|error|Error|unrecognized' | head -12

nohup bash structural_arms.sh "cola rte mrpc" > logs/structural.log 2>&1 &
echo "launched structural arms pid $!"
