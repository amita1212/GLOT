"""Dump the exact confirmed baseline / C configs on CoLA, with per-seed scores."""
import csv

rows = list(csv.DictReader(open("results/campaign_wide_cola.csv",
                                newline="", encoding="utf-8")))
for arm in ("baseline", "C"):
    s = [r for r in rows if r["arm"] == arm and r["stage"] == "confirm"]
    if not s:
        print(arm, "-- none")
        continue
    s.sort(key=lambda r: int(r["seed"]))
    print("=" * 70)
    print(f"arm={arm}  n={len(s)}")
    print("  target  :", s[0]["target"])
    print("  model   :", s[0]["model"])
    print("  setting :", s[0]["setting"])
    print("  task    :", s[0]["task"])
    print("  metric  :", s[0]["metric"])
    print("  run_key :", s[0]["run_key"])
    for d in sorted({r["detail"] for r in s}):
        print("  detail  :", d)
    print("  seeds   :", [r["seed"] for r in s])
    print("  scores  :", [r["score"] for r in s])
    print("  mean    : %.4f" % (sum(float(r["score"]) for r in s) / len(s)))
print("=" * 70)
