#!/usr/bin/env bash
# Detached launcher for after_decoder.sh (resume paused campaigns).
#
# WHY A FILE: `nohup ... &` gets mangled when passed through
# `gcloud compute ssh --command` on Windows, so backgrounding must live in a
# script on the VM.
cd /home/t-amitalfasi/glot
mkdir -p logs

# The "already running" guard must not match THIS script, nor the ssh `bash -c`
# line that invoked it -- both contain the string "after_decoder". Matching on
# the exact `bash after_decoder.sh` argv and excluding our own process tree is
# what makes the check honest; a plain `pgrep -f after_decoder.sh` self-matches
# and silently refuses to launch.
EXISTING=$(pgrep -f 'bash after_decoder\.sh' 2>/dev/null \
           | grep -vw "$$" | grep -vw "$PPID")
if [ -n "$EXISTING" ]; then
    echo "chain already running: $(echo "$EXISTING" | tr '\n' ' ')"
    exit 0
fi

nohup bash after_decoder.sh > logs/chain.log 2>&1 &
echo "chain pid $!"
