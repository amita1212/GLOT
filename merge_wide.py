#!/usr/bin/env python3
"""Merge per-worker wide-sweep CSVs into one file per task.

The parallel sweep writes results/campaign_wide_<task>_w<i>.csv, one per worker,
because ResultsCSV has no file locking. Analysis wants a single table per task.

Dedup is on run_key: workers own disjoint arms so collisions should not happen,
but a resumed/relaunched worker can legitimately re-emit a row, and silently
double-counting one would corrupt every mean and standard deviation downstream.
Columns are unioned rather than taken from the first row -- an earlier bug in
ResultsCSV keyed the schema off rows[0] and dropped every later column.
"""
import csv
import glob
import os
import re
import sys

OUT_DIR = "results"


def merge_task(task: str) -> str | None:
    parts = sorted(glob.glob(os.path.join(OUT_DIR, f"campaign_wide_{task}_w*.csv")))
    if not parts:
        return None

    rows, seen, cols = [], set(), []
    for p in parts:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                for c in r:
                    if c not in cols:
                        cols.append(c)
                key = r.get("run_key")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                rows.append(r)

    dest = os.path.join(OUT_DIR, f"campaign_wide_{task}.csv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    per_worker = ", ".join(f"{os.path.basename(p)}={sum(1 for _ in csv.DictReader(open(p)))}"
                           for p in parts)
    print(f"  {task}: {len(rows)} rows from {len(parts)} workers  [{per_worker}]")
    return dest


def main() -> None:
    tasks = sys.argv[1:]
    if not tasks:
        found = set()
        for p in glob.glob(os.path.join(OUT_DIR, "campaign_wide_*_w*.csv")):
            m = re.match(r"campaign_wide_(.+)_w\d+\.csv", os.path.basename(p))
            if m:
                found.add(m.group(1))
        tasks = sorted(found)

    if not tasks:
        print("no per-worker csvs found")
        return

    print("merged:")
    for t in tasks:
        merge_task(t)


if __name__ == "__main__":
    main()
