#!/usr/bin/env bash
# URGENT: something is running the MTEB driver that this session did not start.
# If it omits --tasks it will also train the full GLUE grid + IMDB, including
# QQP and MNLI (38 and 41 days). Print its FULL argument vector and decide.
set -u
cd /home/t-amitalfasi/glot || exit 1

echo "=== full command line of every run_all_experiments.py ==="
for p in $(pgrep -f 'run_all_experiment[s]\.py'); do
    echo "--- pid $p (elapsed $(ps -p "$p" -o etime= | tr -d ' ')) ---"
    tr '\0' ' ' < "/proc/$p/cmdline"; echo
    echo "  parent: $(ps -o ppid= -p "$p" | tr -d ' ') -> $(ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" 2>/dev/null | cut -c1-80)"
done

echo
echo "=== DANGER CHECK: does it carry --tasks (scope limiter)? ==="
for p in $(pgrep -f 'run_all_experiment[s]\.py'); do
    if tr '\0' ' ' < "/proc/$p/cmdline" | grep -q -- '--tasks'; then
        echo "  pid $p: --tasks PRESENT (scope limited to mteb/embedding)"
    else
        echo "  pid $p: *** --tasks MISSING -> would also run GLUE + IMDB ***"
    fi
done

echo
echo "=== what tasks are the running main.py children actually on? ==="
for p in $(pgrep -f 'hyperglot_new/main[.]py'); do
    tr '\0' ' ' < "/proc/$p/cmdline" | grep -o -- '--task=[a-z0-9]*\|--arm=[A-Za-z0-9_]*\|--mteb_task=[A-Za-z0-9]*' | tr '\n' ' '
    echo "  (pid $p)"
done

echo
echo "=== which results file is it writing? ==="
for p in $(pgrep -f 'run_all_experiment[s]\.py'); do
    tr '\0' '\n' < "/proc/$p/cmdline" | grep -A0 'results_csv'
done

echo
echo "=== mteb_trained.csv state ==="
ls -la results/mteb_trained.csv 2>/dev/null || echo "  absent"
ls -la results/*mteb*.csv 2>/dev/null
