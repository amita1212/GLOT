#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo '=== prewarm_model.sh ==='
cat prewarm_model.sh 2>/dev/null || echo MISSING
echo '=== ARMS keys in campaign.py ==='
grep -n 'ARMS' campaign.py | head -30
echo '=== arm names ==='
sed -n '/^ARMS = {/,/^}/p' campaign.py | grep -o '^\s*"[a-zA-Z_0-9]*"' | tr -d ' "'
echo '=== argparse flags ==='
grep -o -- '--[a-z_]*' campaign.py | sort -u
