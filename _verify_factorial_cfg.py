"""Check factorial_geom.py's transcribed configs against the campaign CSV.

factorial_geom.py hardcodes the baseline and Stage C configurations, copied by
hand out of results/campaign_wide_cola.csv. Every number in the 2x2 depends on
that transcription being exact: if BASE_CFG is not what the baseline actually
confirmed, the factorial decomposes an effect that was never measured.

This reads the confirmed rows back and diffs them against the hardcoded dicts,
and also reports the confirmation mean per arm so the diagonal cells can be
checked against the paper's 45.368 / 46.784.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from factorial_geom import BASE_CFG, C_CFG, HYP  # noqa: E402

CSV_PATH = os.path.join(HERE, "results", "campaign_wide_cola.csv")
# arm name in the campaign -> hardcoded dict in factorial_geom.py
EXPECT = {"baseline": BASE_CFG, "C": {**C_CFG, **HYP}}


def main():
    if not os.path.exists(CSV_PATH):
        print(f"MISSING {CSV_PATH}")
        return 1
    rows = [r for r in csv.DictReader(open(CSV_PATH))
            if r.get("stage") == "confirm"]
    print(f"confirm rows: {len(rows)}")

    bad = 0
    for arm, expect in EXPECT.items():
        got = [r for r in rows if r.get("arm") == arm]
        if not got:
            print(f"\n{arm}: NO CONFIRM ROWS")
            bad += 1
            continue
        seeds = sorted({int(r["seed"]) for r in got})
        scores = [float(r["score"]) for r in got]
        mean = sum(scores) / len(scores)
        print(f"\n{arm}: n={len(got)} seeds={seeds[0]}..{seeds[-1]} "
              f"mean={mean:.3f}")

        # every confirm row of an arm must share one config; check that first
        for k, want in sorted(expect.items()):
            vals = {r.get(k) for r in got}
            if len(vals) != 1:
                print(f"  !! {k}: arm is not on one config: {sorted(vals)}")
                bad += 1
                continue
            have = vals.pop()
            # compare numerically where possible so 0.1 == 0.10 == 1e-1
            same = str(have) == str(want)
            if not same:
                try:
                    same = abs(float(have) - float(want)) < 1e-12
                except (TypeError, ValueError):
                    same = False
            if not same:
                print(f"  !! {k}: csv={have!r}  hardcoded={want!r}")
                bad += 1

        # anything the csv sets that the hardcoded dict omits is also a risk
        ignored = {"run_key", "target", "model", "setting", "task", "arm",
                   "stage", "trial", "seed", "metric", "detail", "score",
                   "elapsed_sec", "mean_density"}
        for k in sorted(set(got[0]) - ignored - set(expect)):
            vals = {r.get(k) for r in got}
            if len(vals) == 1:
                v = vals.pop()
                if v not in ("", None):
                    print(f"  .. csv also sets {k}={v!r} (not in hardcoded cfg)")

    print("\nOK" if not bad else f"\n{bad} MISMATCH(ES) -- do not run the "
          f"factorial until these are resolved")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
