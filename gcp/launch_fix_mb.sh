#!/usr/bin/env bash
# Detached launcher for fix_modernbert.sh (nohup/& get mangled through
# `gcloud compute ssh --command` on Windows). Waits for any GPU job first.
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
cat > _chain_mb.sh <<'EOF'
set -u
cd /home/t-amitalfasi/glot
while pgrep -f 'test_scale_fix.sh|campaign.py' >/dev/null 2>&1; do sleep 30; done
bash fix_modernbert.sh
EOF
sed -i 's/\r$//' _chain_mb.sh fix_modernbert.sh hyperglot/main.py
~/glotenv/bin/python -m py_compile hyperglot/main.py || { echo "COMPILE FAILED"; exit 1; }
echo COMPILE_OK
nohup bash _chain_mb.sh > logs/fix_modernbert.log 2>&1 &
echo "launched fix_modernbert pid $!"
