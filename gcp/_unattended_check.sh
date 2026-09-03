#!/usr/bin/env bash
# Is anything set to shut this machine down or kill a job while unattended?
cd ~/glot 2>/dev/null || cd ~
echo "=== host ==="; hostname; date -u '+%F %T UTC'
echo
echo "=== autoshutdown / watchdog processes ==="
ps -eo pid,etime,cmd | grep -iE 'autoshut|shutdown|poweroff|watchdog' | grep -v grep | cut -c1-110 | sed 's/^/  /'
echo "  (nothing above = none running)"
echo
echo "=== scheduled jobs ==="
echo "  crontab:"; crontab -l 2>/dev/null | grep -v '^#' | sed 's/^/    /' || echo "    none"
echo "  at queue:"; atq 2>/dev/null | sed 's/^/    /' || echo "    none"
echo "  systemd timers:"; systemctl list-timers --no-pager --no-legend 2>/dev/null | head -5 | sed 's/^/    /'
echo
echo "=== live jobs ==="
ps -eo pid,etime,cmd | grep -E 'chain_gpu1|queue_mteb|queue_table1|run_all_exp|campaign\.py|factorial_geom|decoder_stsb|worker_sst2' | grep -v grep | cut -c1-110 | sed 's/^/  /'
echo
echo "=== disk headroom (SST-2 needs ~26 GB, MTEB checkpoints grow) ==="
df -h / | tail -1 | sed 's/^/  /'
