#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo "=== now $(date -Is) ==="
echo "--- roberta ---"
tail -3 logs/roberta_all.log
echo "--- chain mrpc/rte ---"
tail -3 logs/chain_mrpc_rte.log 2>/dev/null
echo "--- machine ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
uptime
echo "--- result files (newest first) ---"
ls -lt results/*.csv | head -14
