#!/bin/bash
# Single sequential HyperGLOT sweep pipeline (replaces the fragile watcher setup).
# Runs BERT -> RoBERTa -> decoder LLMs one after another on the single L4 GPU.
# The orchestrator is resumable (skips (model,task,arm,seed) rows already in the
# CSV), so all completed work is preserved and Stage B (ABC) arms are skipped
# because they were removed from run_all_experiments.py CONFIGS.
set -u
cd ~/GLOT
source ~/env.sh

echo "[pipeline] $(date) starting BERT (resume, no ABC)"
python3 run_all_experiments.py --models bert-base-uncased --seeds 42 >> ~/sweep_bert.log 2>&1

echo "[pipeline] $(date) starting RoBERTa"
python3 run_all_experiments.py --models FacebookAI/roberta-base --seeds 42 >> ~/sweep_roberta.log 2>&1

echo "[pipeline] $(date) starting decoder LLMs"
python3 run_all_experiments.py \
  --models "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "HuggingFaceTB/SmolLM2-1.7B" "meta-llama/Llama-3.2-3B" "mistralai/Mistral-7B-v0.1" \
  --seeds 42 >> ~/sweep_decoders.log 2>&1

echo "[pipeline] $(date) ALL DONE"
