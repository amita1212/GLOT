#!/usr/bin/env python
"""Drop ablation rows that were produced with the broken absolute-`rho` grid.

Rows with knob == "rho" trained on EMPTY graphs (measured: 0 edges on 16/16
sentences for every rho in {0.5..3.0}), so they are not measurements of Stage A
at all. They must not survive into the arm-selection step, which takes a max
over all rows for an arm.

Rows that stay valid and are worth keeping (they cost ~2 min each):
  * knob == "tau"    -> baseline and C_thresh use the cosine graph, unaffected.
  * knob == "knn_k"  -> kNN uses distance RANKINGS, unaffected by the saturation
                        bug, but keep only values inside the new grid so every
                        arm still gets an equal-size tuning budget.
"""

from __future__ import annotations

import csv
import os
import sys

VALID = {
    "tau": {"0.0", "0.2", "0.4", "0.6", "0.8"},
    "knn_k": {"1", "2", "4", "8", "16"},
}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/ablation_fair.csv"
    if not os.path.exists(path):
        print(f"{path}: nothing to clean")
        return

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    kept, dropped = [], []
    for r in rows:
        knob, val = r["knob"], r["knob_value"]
        if knob in VALID and val in VALID[knob]:
            kept.append(r)
        else:
            dropped.append(r)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)

    print(f"kept    {len(kept)} rows")
    print(f"dropped {len(dropped)} rows (broken absolute-rho / out-of-grid)")
    by_arm = {}
    for r in kept:
        by_arm[r["arm"]] = by_arm.get(r["arm"], 0) + 1
    for arm, n in sorted(by_arm.items()):
        print(f"  kept {arm:12s} {n}")


if __name__ == "__main__":
    main()
