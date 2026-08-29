#!/usr/bin/env bash
# Give the WIDE sweep the GPU to itself.
#
# Killing decoder_sweep.sh let after_decoder.sh (the resume chain) fire, which
# relaunched structural_arms.sh. That now competes with the wide sweep for the
# same GPU. This stops the chain and every NON-wide campaign, and leaves the
# wide sweep untouched.
#
# Selection is by --out path, not by pgrep pattern: campaign.py children are not
# killed by killing their parent shell, and a plain `pkill -f campaign.py` would
# take the wide run down too.
set -u
cd /home/t-amitalfasi/glot

SELF=$$
echo "=== chain / wrapper scripts ==="
CH=$(pgrep -f 'after_decoder\.sh|structural_arms\.sh|roberta_compare\.sh|decoder_sweep\.sh' 2>/dev/null \
     | grep -vw "$SELF" | grep -vw "$PPID")
if [ -n "$CH" ]; then
    for p in $CH; do
        printf '  kill %s  %s\n' "$p" "$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-80)"
    done
    kill $CH 2>/dev/null
else
    echo "  (none)"
fi

echo
echo "=== campaign.py processes NOT belonging to the wide sweep ==="
for p in $(pgrep -f 'campaign\.py' 2>/dev/null); do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    case "$cmd" in
        *wide*) printf '  KEEP %s  (wide)\n' "$p" ;;
        *)      printf '  kill %s  %s\n' "$p" "$(echo "$cmd" | cut -c1-80)"
                kill "$p" 2>/dev/null ;;
    esac
done

sleep 8
echo
echo "=== still running ==="
pgrep -af 'campaign\.py|wide_sweep|structural_arms|roberta_compare|after_decoder' \
  | grep -vw "$SELF" | cut -c1-100 || echo "  (none)"
