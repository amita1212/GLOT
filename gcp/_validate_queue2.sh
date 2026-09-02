#!/usr/bin/env bash
# Validate queue_rest2.sh before it replaces the running queue:
#   1. syntax
#   2. items 1,2,3,5 still byte-identical to the live queue
#   3. the pre-flight guard's own arithmetic returns PASS on the corrected
#      command and FAIL on the original one (so the guard actually guards)
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== 1. syntax ==="
bash -n queue_rest2.sh && echo "  OK"

echo
echo "=== 2. items 1,2,3,5 unchanged ==="
for tag in "1\. RoBERTa" "2\. Stage C" "3\. Stage A" "5\. Decoder"; do
    a=$(awk "/---- $tag/,/^# ---- /" queue_rest.sh  | md5sum | cut -d' ' -f1)
    b=$(awk "/---- $tag/,/^# ---- /" queue_rest2.sh | md5sum | cut -d' ' -f1)
    [ "$a" = "$b" ] && echo "  $tag IDENTICAL" || echo "  $tag ***DIFFERS***"
done

echo
echo "=== 3. does the guard pass the CORRECTED plan and fail the ORIGINAL? ==="
check() {
    local f=$1 label=$2
    local arms leak
    arms=$(grep -o -- '--arm=[A-Za-z0-9_]*' "$f" | sort -u | wc -l)
    leak=$(grep -o -- '--task=[a-z0-9]*' "$f" | sort -u \
           | grep -c -v -e '--task=mteb' -e '--task=embedding')
    if [ "$arms" -ne 4 ] || [ "$leak" -ne 0 ]; then
        echo "  $label -> REFUSE (arms=$arms want 4, leaked tasks=$leak want 0)"
    else
        echo "  $label -> LAUNCH (arms=$arms, no leakage, $(grep -c 'main.py' "$f") commands)"
    fi
}
[ -f /tmp/_dry_item4.log  ] && check /tmp/_dry_item4.log  "original args" || echo "  (no original dry log)"
[ -f /tmp/_dry_item4b.log ] && check /tmp/_dry_item4b.log "corrected args" || echo "  (no corrected dry log)"
