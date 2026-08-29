#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo '=== prewarm_model.sh head ==='
head -30 prewarm_model.sh
echo '=== existing caches ==='
ls -d data/*batches 2>/dev/null
echo '=== disk ==='
df -h / | tail -1
echo '=== running ==='
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
uptime
