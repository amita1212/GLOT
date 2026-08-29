#!/usr/bin/env bash
# Bring a BARE GPU VM to the point where a worker script can run.
#
#   usage:  bash bootstrap_machine.sh <git-remote-url> [branch]
#   e.g.    bash bootstrap_machine.sh https://github.com/amita1212/GLOT.git hyperglot-stageA
#
# Run this ONCE per machine. It is idempotent enough to re-run safely.
#
# THE ONE THING THAT WILL BITE YOU
#   Results are only comparable to the rest of the project if the library stack
#   matches. gcp/setup_vm.sh pins torch 2.8.0+cu129 and every downstream
#   package exactly, on purpose. Do not substitute "whatever pip gives me" --
#   a different cuDNN can move a score in the 2nd decimal, and the effects we
#   are measuring are 0.2-1.4 points.
#   The check at the end prints the versions. Compare them against the machine
#   that produced the baseline you will be differencing against.
set -u

REMOTE="${1:?git remote url}"
BRANCH="${2:-hyperglot-stageA}"
cd "$HOME"

if [ ! -d glot ]; then
    echo "=== clone ==="
    git clone "$REMOTE" glot
fi
cd glot
git fetch --all -q
git checkout "$BRANCH"
git pull -q
echo "commit: $(git rev-parse --short HEAD)"

if [ ! -d "$HOME/glotenv" ]; then
    echo "=== environment (10-15 min) ==="
    bash gcp/setup_vm.sh
else
    echo "=== environment already present, skipping setup_vm.sh ==="
fi

mkdir -p logs results data

# CRLF kills bash and python alike, and this repo gets edited on Windows.
sed -i 's/\r$//' *.sh *.py 2>/dev/null || true

echo
echo "=== compile check ==="
~/glotenv/bin/python -m py_compile campaign.py seed_extend.py \
    factorial_geom_full.py hyperglot/main.py \
    || { echo "COMPILE FAILED -- stop here"; exit 1; }
echo "COMPILE_OK"

echo
echo "=== environment fingerprint (compare across machines!) ==="
~/glotenv/bin/python - <<'PY'
import torch, transformers, torch_geometric, numpy, sklearn
print("torch          ", torch.__version__)
print("cuda           ", torch.version.cuda)
print("cudnn          ", torch.backends.cudnn.version())
print("gpu            ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("transformers   ", transformers.__version__)
print("torch_geometric", torch_geometric.__version__)
print("numpy          ", numpy.__version__)
print("sklearn        ", sklearn.__version__)
PY
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h "$HOME" | tail -1

echo
echo "READY. Now run exactly one of:"
echo "  nohup bash worker_decoder.sh cola     > logs/run.log 2>&1 &"
echo "  nohup bash worker_decoder.sh stsb \"B C\" > logs/run.log 2>&1 &"
echo "  nohup bash worker_bert_seeds.sh       > logs/run.log 2>&1 &"
echo "  nohup bash worker_sst2.sh             > logs/run.log 2>&1 &"
echo
echo "Then watch:  tail -f logs/run.log"
