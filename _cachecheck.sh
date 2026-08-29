cd /home/t-amitalfasi/glot
echo "=== hidden-state cache dirs ==="
find . -maxdepth 3 -type d -name '*cache*' 2>/dev/null | head
ls -d cache* .cache* 2>/dev/null
echo
echo "=== anything mentioning roberta in a cache path ==="
find . -path ./glot_original -prune -o -iname '*roberta*' -print 2>/dev/null | grep -iv '\.log' | head -20
echo
echo "=== disk ==="
df -h / | tail -1
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo
echo "=== anything still running? ==="
pgrep -fa 'campaign.py|factorial_geom|main.py' | head
