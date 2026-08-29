#!/usr/bin/env bash
# Stop a running campaign. Kept in a file because `pkill -f <pattern>` sent
# inline through `ssh --command` matches the remote `bash -c` wrapper and kills
# the SSH session itself.
set -u
pkill -f campaign.py || true
sleep 2
pgrep -fa campaign.py || echo "no campaign running"
