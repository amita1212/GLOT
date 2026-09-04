"""Verify numbers written into paper/short.tex against the campaign CSVs.
Fails loudly on any mismatch. Covers the cells added in the final revision."""
import csv, math, re, sys

TEX = open("paper/short.tex", encoding="utf-8").read()
bad, ok = [], 0


def check(label, want, tol=0.006):
    """Assert `want` appears in the tex, allowing the value to be \\textbf{}."""
    global ok
    bolded = re.sub(r"^([-+0-9.]+)", r"\\textbf{\1}", want)
    if want in TEX or bolded in TEX:
        ok += 1
    else:
        bad.append(f"{label}: expected {want!r} (or bolded) not found")


def campaign(path):
    rows = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("stage") == "confirm" and r.get("score"):
            rows[r["run_key"]] = r
    sc = {}
    for r in rows.values():
        sc[(r["arm"], int(r["seed"]))] = float(r["score"])
    return sc


def stat(sc, arm):
    xs = [v for (a, _), v in sc.items() if a == arm]
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((v - m) ** 2 for v in xs) / (len(xs) - 1))
    return m, sd, len(xs)


def delta(sc, arm):
    base = {s: v for (a, s), v in sc.items() if a == "baseline"}
    vals = {s: v for (a, s), v in sc.items() if a == arm}
    common = sorted(set(base) & set(vals))
    d = [vals[s] - base[s] for s in common]
    n = len(d)
    m = sum(d) / n
    pos = sum(1 for v in d if v > 0)
    return m, pos, n - pos


# ---- TinyLlama CoLA -------------------------------------------------------
tl = campaign("results/campaign_t1_tl_cola.csv")
for arm, tag in [("baseline", None), ("A", "A"), ("B", "B"), ("C", "C"),
                 ("AB", "AB"), ("AC", "AC"), ("BC", "BC"), ("ABC", "ABC"),
                 ("no_graph", "no graph")]:
    m, sd, n = stat(tl, arm)
    assert n == 15, f"TinyLlama CoLA {arm} has {n} seeds"
    check(f"tl_cola {arm} mean", f"{m:.2f}\\,$\\pm${sd:.2f}")
    if arm != "baseline":
        d, p, ng = delta(tl, arm)
        check(f"tl_cola {arm} delta", f"{d:+.2f}$")
        check(f"tl_cola {arm} split", f"({p}/{ng})")

# ---- matched decoder STS-B -----------------------------------------------
dec = campaign("results/campaign_decoder_stsb_matched.csv")
for arm in ["baseline", "B", "C"]:
    m, sd, n = stat(dec, arm)
    assert n == 15
    check(f"decoder {arm}", f"{m:.2f}")
d, p, ng = delta(dec, "B")
check("decoder B delta", f"{d:+.2f}")

# ---- SST-2 ----------------------------------------------------------------
sst = campaign("results/sst2_reduced.csv")
for arm in ["baseline", "B", "C", "BC"]:
    m, sd, n = stat(sst, arm)
    assert n == 15, f"SST-2 {arm} has {n} seeds"
    check(f"sst2 {arm}", f"{m:.2f}\\,$\\pm${sd:.2f}")

# ---- corrective factorial -------------------------------------------------
fac = {}
for r in csv.DictReader(open("results/factorial_geom_cola_parity.csv",
                             encoding="utf-8")):
    if r["stage"] == "confirm" and r["score"]:
        fac[(r["arm"], int(r["seed"]))] = float(r["score"])
seeds = sorted({s for _, s in fac})
assert len(seeds) == 65, f"factorial has {len(seeds)} seeds"
for arm in ["base_at_base", "C_at_base", "base_at_C", "C_at_C"]:
    m, sd, n = stat(fac, arm)
    check(f"factorial {arm}", f"{m:.2f}\\,$\\pm${sd:.2f}")

# ---- MTEB RoBERTa ---------------------------------------------------------
cell = {}
for r in csv.DictReader(open("results/mteb_table3.csv", encoding="utf-8")):
    if r["task"] == "mteb" and r["mteb_task"] and r["mteb_score"]:
        cell[(r["model"].split("/")[-1], r["mteb_task"], r["arm"],
              int(r["seed"]))] = float(r["mteb_score"]) * 100
TASKS = ["EmotionClassification", "SciFact", "RedditClustering",
         "AskUbuntuDupQuestions", "STS12", "TwitterSemEval2015", "SummEval"]
for t in TASKS:
    b = [cell[("roberta-base", t, "baseline", s)] for s in range(1, 16)]
    check(f"mteb rob baseline {t}", f"{sum(b)/len(b):.2f}")
    for arm in ["A_threshold", "B_threshold", "C_threshold", "AC_threshold",
                "ABC_threshold"]:
        d = [cell[("roberta-base", t, arm, s)] - cell[("roberta-base", t,
             "baseline", s)] for s in range(1, 16)]
        check(f"mteb rob {arm} {t}", f"{sum(d)/len(d):+.2f}")

print(f"{ok} literals verified against the CSVs")
if bad:
    print(f"\n{len(bad)} MISMATCHES:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("all checked numbers present in short.tex")
