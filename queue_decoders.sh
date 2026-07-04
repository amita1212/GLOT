#!/bin/bash
# Wait for the BERT and RoBERTa sweeps to finish, then run all decoder LLMs.
# Resumable: skips (model,task,arm,seed) rows already in the CSV.
cd ~/GLOT
source ~/env.sh

echo "[decoders] waiting for encoder sweeps (bert + roberta) to finish..."
while pgrep -f "run_all_experiments.py --models bert-base-uncased" >/dev/null \
   || pgrep -f "run_all_experiments.py --models FacebookAI/roberta-base" >/dev/null; do
  sleep 60
done
echo "[decoders] encoder sweeps done. Starting decoder LLMs..."

# Smallest -> largest so quick wins land first. Gated models (Llama, Mistral)
# require the token to have accepted their license on huggingface.co; if a
# download 403s, that model's runs fail and the orchestrator moves on.
python3 run_all_experiments.py \
  --models "TinyLlama/TinyLlama-1.1B-Chat-v1.0" "HuggingFaceTB/SmolLM2-1.7B" "meta-llama/Llama-3.2-3B" "mistralai/Mistral-7B-v0.1" \
  --seeds 42 >> ~/sweep_decoders.log 2>&1
echo "[decoders] all decoder sweeps finished."
