#!/usr/bin/env bash
# Detached launcher for roberta_compare.sh. Smoke-tests that RoBERTa loads
# through GLOT's own loader and that the two new arms launch, BEFORE queueing
# hours of work behind the structural campaign.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
sed -i 's/\r$//' roberta_compare.sh campaign.py
~/glotenv/bin/python -m py_compile campaign.py || { echo "COMPILE FAILED"; exit 1; }
echo COMPILE_OK

echo "--- smoke: does RoBERTa load through load_backbone? ---"
cd hyperglot
CUDA_VISIBLE_DEVICES= ~/glotenv/bin/python smoke_backbone.py \
    --models roberta-base --layers 12 2>&1 | grep -E 'tokenizer|n_layers|L12|FAILED|ALL OK'
cd ..

nohup bash roberta_compare.sh "cola stsb mrpc rte" > logs/roberta.log 2>&1 &
echo "launched roberta comparison pid $!"
