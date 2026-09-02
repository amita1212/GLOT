#!/usr/bin/env bash
# Diagnose the stalled wait loop on hyperglot-bgu.
# Question: is any REAL training process alive, or is the queue blocked by a
# stranded shell whose command line merely CONTAINS the interpreter path?
# A real trainer has the venv python as its executable (/proc/PID/exe).
# A shell that merely mentions the path does not.

echo "=== processes whose EXECUTABLE is the venv python (real trainers) ==="
found=0
for p in /proc/[0-9]*; do
    e=$(readlink "$p/exe" 2>/dev/null) || continue
    case "$e" in
        *glotenv*) echo "  $p -> $e"; found=$((found+1));;
    esac
done
echo "  real trainers: $found"

echo "=== processes matching the wait-loop pattern (cmdline substring) ==="
pgrep -af 'glotenv/bin/pyth[o]n' | cut -c1-200
echo "  matches: $(pgrep -cf 'glotenv/bin/pyth[o]n')"

echo "=== is setup_vm.sh actually still running? ==="
if pgrep -af 'setup_vm[.]sh' ; then
    echo "  YES - leave it alone"
else
    echo "  NO - so PID 3039's own wait loop is self-matching and will never exit"
fi

echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
