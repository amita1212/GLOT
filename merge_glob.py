#!/usr/bin/env python3
"""Merge per-worker campaign CSVs matching a glob into one file.

merge_wide.py is hardcoded to the campaign_wide_<task>_w<i> naming. This is the
same logic with the pattern passed in, so it works for any campaign tag.

    usage: merge_glob.py "results/campaign_rob_stsb_w*.csv" results/campaign_rob_stsb.csv

Unions the column set across files rather than taking the schema from row 0 --
different arms emit different config keys, and schema-from-row-0 silently drops
every column the first row happens not to have.
"""
import csv
import glob
import sys


def main():
    pattern, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match {pattern}")
        return

    rows, cols, seen = [], [], set()
    for path in files:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                for k in r:
                    if k not in seen:
                        seen.add(k)
                        cols.append(k)
                key = r.get("run_key")
                rows.append((key, r))

    deduped, keys = [], set()
    for key, r in rows:
        if key and key in keys:
            continue
        if key:
            keys.add(key)
        deduped.append(r)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in deduped:
            w.writerow(r)

    print(f"{out}: {len(deduped)} rows from {len(files)} files "
          f"({len(rows) - len(deduped)} dupes dropped)")


if __name__ == "__main__":
    main()
