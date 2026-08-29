#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
echo '=== density computation sites in hyperglot/main.py ==='
grep -n -i 'density' hyperglot/main.py | head -40
echo
echo '=== context around each ==='
for L in $(grep -n -i 'density' hyperglot/main.py | cut -d: -f1 | head -8); do
  echo "----- line $L -----"
  sed -n "$((L-12)),$((L+6))p" hyperglot/main.py
done
