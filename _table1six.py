"""Cost of a SIX-TASK completion of Table 1, six arms, full per-task search.

TASK SELECTION -- why these six, and why not the other three.

  KEEP
    CoLA   the one positive result (Stage C +1.42) lives here. Non-negotiable.
    STS-B  where Stage B's harm is firmest and replicates across backbones.
    SST-2  67,349 train examples, 7.9x CoLA. The ONLY task that can kill the
           small-data confound on Stage B (variance amplification shrinks with
           data). Highest scientific value of anything not yet run.
    MRPC   second pair task, cheap, and already done on BERT so it costs
           nothing there.
    QNLI   mid-size pair task capped at 20k, a genuinely different task type
           (QA entailment) and ~1/2 the cost of QQP.
    MNLI   the only 3-CLASS task. Tests whether the stages transfer beyond
           binary/regression at all, and yields TWO reported columns
           (MNLI-m and MNLI-mm) from ONE run. Cheaper than QQP.

  DROP
    QQP    most expensive task in GLUE for us (dev set is 40,430 pairs, twice
           the capped train set). Adds a third binary pair task -- the axis
           MRPC and QNLI already cover.
    RTE    measured MDE 2.06 accuracy points, larger than every effect in the
           paper. It provably cannot resolve anything. Already have BERT.
    WNLI   635 train / 71 dev. Degenerate; contributes noise only.

  Result: 6 tasks give 7 of the original's 10 reported columns (MNLI counts
  twice), and every claim the paper actually makes is testable on them.
"""
EPOCHS, CAPPED, SUB = 2, {"qqp", "qnli", "mnli"}, 20000
C0, C1 = 62.6, 5.307e-3
GLUE = {  # n_train, n_dev, is_pair
    "mrpc": (3668, 408, True), "stsb": (5749, 1500, True),
    "cola": (8551, 1043, False), "sst2": (67349, 872, False),
    "qnli": (104743, 5463, True), "mnli": (392702, 9815 + 9832, True),
}
# already run at 40 trials x 9 arms (a superset of the 6 arms requested)
DONE = {"bert": {"cola", "stsb", "mrpc"}, "roberta": {"cola", "stsb"},
        "tinyllama": set()}          # tinyllama stsb exists but only 10 trials
XRUN = {"bert": 1.00, "roberta": 1.05, "tinyllama": 3.80}
MB = {"bert": (0.374, 0.748), "roberta": (0.374, 0.748),
      "tinyllama": (0.982, 2.087)}
ARMS = 6
SEEDS = 15


def secs(t, bb):
    n_tr = min(GLUE[t][0], SUB) if t in CAPPED else GLUE[t][0]
    _, dev, pair = GLUE[t]
    return (C0 + C1 * EPOCHS * (n_tr + dev) * (2 if pair else 1)) * XRUN[bb]


def gb(t, bb):
    n_tr = min(GLUE[t][0], SUB) if t in CAPPED else GLUE[t][0]
    _, dev, pair = GLUE[t]
    return (n_tr + dev) * MB[bb][1 if pair else 0] / 1024


def report(trials_for, label):
    print("=" * 86)
    print(label)
    print("=" * 86)
    print(f"{'backbone':10s} {'task':5s} {'trials':>6s} {'runs':>5s} {'s/run':>6s}"
          f" {'hours':>7s} {'cache GB':>8s}")
    grand = 0.0
    peak = 0.0
    for bb in ("bert", "roberta", "tinyllama"):
        sub = 0.0
        for t in GLUE:
            if t in DONE[bb]:
                print(f"{bb:10s} {t:5s} {'--':>6s} {'--':>5s} {'--':>6s} "
                      f"{'done':>7s} {'--':>8s}")
                continue
            tr = trials_for(bb)
            runs = ARMS * (tr + SEEDS)
            h = runs * secs(t, bb) / 3600
            sub += h
            peak = max(peak, gb(t, bb))
            print(f"{bb:10s} {t:5s} {tr:6d} {runs:5d} {secs(t,bb):6.0f} "
                  f"{h:7.1f} {gb(t,bb):8.1f}")
        print(f"{bb:10s} {'SUBTOTAL':13s} {'':5s} {'':6s} {sub:7.1f} h "
              f"= {sub/24:.1f} d")
        grand += sub
        print()
    print(f"  GRAND TOTAL {grand:7.0f} h = {grand/24:.1f} days on one L4")
    print(f"  peak disk (cache deleted between tasks): {peak:.0f} GB")
    print(f"  free after reclaiming the 63 GB of layer-probe caches: 181 GB")
    print()
    return grand


a = report(lambda bb: 40, "OPTION 1 -- 40 tuning trials everywhere "
                          "(equal budget across all three backbones)")
b = report(lambda bb: 10 if bb == "tinyllama" else 40,
           "OPTION 2 -- 40 trials on the encoders, 10 on TinyLlama\n"
           "            (matches the existing TinyLlama STS-B campaign, which "
           "used 10 trials,\n             so the TinyLlama block stays "
           "internally equal-budget)")
print("=" * 86)
print(f"Option 1 costs {a - b:.0f} h ({(a-b)/24:.1f} d) more than Option 2.")
print()
print("NOTE on Option 1: the existing TinyLlama STS-B campaign used 10 trials.")
print("  Running the new TinyLlama tasks at 40 would make that block unequal")
print("  budget, which is the exact defect this paper criticises. Option 1")
print("  therefore also requires RE-RUNNING TinyLlama STS-B at 40 trials:")
extra = ARMS * (40 + SEEDS) * secs("stsb", "tinyllama") / 3600
print(f"    + {extra:.0f} h  -> Option 1 true total {a + extra:.0f} h "
      f"= {(a + extra)/24:.1f} days")
