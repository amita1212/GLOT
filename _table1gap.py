"""Cost of giving RoBERTa and TinyLlama everything BERT has in Table 1.

TABLE 1 SCOPE: four tasks (CoLA, STS-B, MRPC, RTE) x nine arms
  baseline, A, B, C, AB, AC, BC, ABC, no_graph

WHAT EXISTS TODAY (read off tab:absolute / tab:verdict)
  BERT       4 tasks x 9 arms                     COMPLETE
  RoBERTa    CoLA + STS-B x 9 arms (+2 calib)     missing MRPC, RTE
  TinyLlama  STS-B x 7 arms; B and C alone are    missing CoLA, MRPC, RTE
             running in the queue right now

TRIALS. The budget has to be equal *within* a backbone, because that is what
the paired tests need. RoBERTa's existing tasks used 40 trials/arm, so its new
tasks must too. TinyLlama's existing STS-B used 10, so its new tasks use 10 --
raising TinyLlama to 40 would make its own block unequal-budget and would also
force a re-run of STS-B.

RUNTIME. Measured seconds/run from results_vm campaign CSVs:
  BERT       cola 138.4  stsb 223.5  mrpc 165.5  rte 124.0
  RoBERTa    cola 155.8  stsb 225.7  (mrpc/rte never run -> scale)
  TinyLlama  cola 548.4  stsb 808.0  (mrpc/rte never run -> scale)
Backbone ratio vs BERT, from the tasks both have run:
  RoBERTa    155.8/138.4 = 1.126 ; 225.7/223.5 = 1.010  -> mean 1.068
  TinyLlama  548.4/138.4 = 3.962 ; 808.0/223.5 = 3.615  -> mean 3.788

CACHE. MB per cached example, measured from `du -sh data/*`:
  BERT/RoBERTa  single 0.374  pair 0.748
  TinyLlama     single 0.982  pair 2.087
"""
BERT_S = {"cola": 138.4, "stsb": 223.5, "mrpc": 165.5, "rte": 124.0}
MEASURED = {                       # per-backbone measured s/run where we have it
    "roberta":   {"cola": 155.8, "stsb": 225.7},
    "tinyllama": {"cola": 548.4, "stsb": 808.0},
}
RATIO = {"roberta": 1.068, "tinyllama": 3.788}
TRIALS = {"roberta": 40, "tinyllama": 10}
MISSING = {"roberta": ["mrpc", "rte"], "tinyllama": ["cola", "mrpc", "rte"]}
SEEDS = 15
# n_train, n_dev, is_pair
SIZE = {"cola": (8551, 1043, False), "stsb": (5749, 1500, True),
        "mrpc": (3668, 408, True), "rte": (2490, 277, True)}
MB = {"roberta": (0.374, 0.748), "tinyllama": (0.982, 2.087)}


def secs(bb, t):
    if t in MEASURED[bb]:
        return MEASURED[bb][t], "measured"
    return BERT_S[t] * RATIO[bb], "scaled"


def cache_gb(bb, t):
    n_tr, n_dev, pair = SIZE[t]
    return (n_tr + n_dev) * MB[bb][1 if pair else 0] / 1024


for arms, label in ((9, "NINE arms - exactly what BERT has"),
                    (6, "SIX arms - baseline, A, B, C, AC, BC (your earlier pick)")):
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    runs_per_task = {bb: arms * (TRIALS[bb] + SEEDS) for bb in TRIALS}
    print(f"{'backbone':10s} {'task':5s} {'trials':>6s} {'runs':>5s} {'s/run':>7s} "
          f"{'':8s} {'hours':>7s} {'cache GB':>8s}")
    grand = 0.0
    for bb in ("roberta", "tinyllama"):
        sub = 0.0
        for t in MISSING[bb]:
            s, how = secs(bb, t)
            h = runs_per_task[bb] * s / 3600
            sub += h
            print(f"{bb:10s} {t:5s} {TRIALS[bb]:6d} {runs_per_task[bb]:5d} "
                  f"{s:7.0f} {how:>8s} {h:7.1f} {cache_gb(bb,t):8.1f}")
        print(f"{bb:10s} {'SUBTOTAL':11s} {'':5s} {'':7s} {'':8s} {sub:7.1f} h "
              f"= {sub/24:.1f} d")
        grand += sub
        print()
    print(f"  TOTAL {grand:6.1f} h = {grand/24:.1f} days on one L4")
    print(f"  peak disk: {max(cache_gb(bb,t) for bb in MISSING for t in MISSING[bb]):.1f} GB "
          f"(181 GB free) -- not a constraint")
    print()

print("=" * 78)
print("ALREADY IN FLIGHT (do not re-run): TinyLlama STS-B arms B and C alone,")
print("  queue item 5 on hyperglot-l4, ~11 h. That completes TinyLlama STS-B")
print("  to 9 arms, so it is excluded above.")
print()
print("NOT INCLUDED, because they are separate tables rather than Table 1:")
print("  * the RoBERTa calibration arms (published_tau / quantile_tau) on")
print("    MRPC+RTE -- would extend tab:roberta, +2 arms x 2 tasks")
print("  * the Stage C 2x2 factorial (tab:factorial) on RoBERTa / TinyLlama")
print("  * the noise diagnostic (tab:stress) on RoBERTa / TinyLlama")
print("  * the density x scale factorial (tab:fix)")
print("  Ask if you want any of these costed too.")
