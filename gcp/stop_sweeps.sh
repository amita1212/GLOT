#!/usr/bin/env bash
# Stop any running sweep. NEVER inline `pkill -f <pattern>` through
# `ssh --command`: the pattern matches the remote `bash -c` wrapper and kills
# the SSH session itself. Keeping it in a file avoids that.
set -u
pkill -f curvature_sweep.py || true
pkill -f geometry_sweep.py || true
sleep 2
pgrep -fa "sweep.py" || echo "no sweep running"
