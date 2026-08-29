#!/usr/bin/env bash
# Provision the GLOT reproduction environment on the GCP L4 VM.
# Pins follow the upstream GLOT requirements.txt exactly (torch 2.8.0 + cu129),
# so the reproduction runs against the library stack the paper used.
set -euo pipefail

echo "=== apt deps ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev build-essential git >/dev/null

echo "=== venv ==="
python3 -m venv "$HOME/glotenv"
# shellcheck disable=SC1091
source "$HOME/glotenv/bin/activate"
pip install -q --upgrade pip setuptools wheel

echo "=== torch 2.8.0 + cu129 ==="
pip install -q torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu129

echo "=== PyG binaries (wheels built for torch-2.8.0+cu129) ==="
pip install -q torch-scatter torch-sparse pyg-lib \
    -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
pip install -q torch-geometric==2.6.1

echo "=== NLP / research stack ==="
pip install -q \
    transformers==4.57.0 \
    datasets==3.6.0 \
    sentence-transformers==5.1.1 \
    mteb==2.1.7 \
    peft==0.17.1 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    scikit-learn==1.7.2 \
    wandb==0.22.2 \
    accelerate==1.10.1 \
    tqdm==4.67.1

echo "=== verify ==="
python - <<'PY'
import torch, torch_scatter, torch_geometric, transformers, datasets, mteb, peft, wandb, sklearn
print("torch           ", torch.__version__)
print("cuda available  ", torch.cuda.is_available())
print("gpu             ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("torch_geometric ", torch_geometric.__version__)
print("transformers    ", transformers.__version__)
print("datasets        ", datasets.__version__)
print("mteb            ", mteb.__version__)
print("ALL IMPORTS OK")
PY
echo "=== setup complete ==="
