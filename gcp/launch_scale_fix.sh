#!/usr/bin/env bash
# Wait for the power test, then run the ModernBERT scale-fix diagnosis.
# Detached launcher (nohup/& get mangled through `gcloud compute ssh --command`).
set -u
cd /home/t-amitalfasi/glot
mkdir -p logs results
cat > _chain_scale.sh <<'EOF'
set -u
cd /home/t-amitalfasi/glot
echo "[scale] waiting for power test to finish..."
while pgrep -f 'power_test.sh|campaign.py' >/dev/null 2>&1; do sleep 60; done
echo "[scale] GPU free at $(date -Is)"
bash test_scale_fix.sh
EOF
sed -i 's/\r$//' _chain_scale.sh
nohup bash _chain_scale.sh > logs/scale_fix.log 2>&1 &
echo "launched scale-fix chain pid $!"
