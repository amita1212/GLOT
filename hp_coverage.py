#!/usr/bin/env python3
"""Audit hyperparameter coverage: what is searched, what is pinned, what is unreachable.

Answers three separate questions that are easy to conflate:
  1. How big is the space each arm is drawn from, and what FRACTION do 40 random
     trials actually cover?
  2. Which main.py CLI flags does the campaign never vary at all?
  3. What is the probability that 40 random draws land in the top-k% of the space
     (Bergstra & Bengio's argument for why exhaustive search is unnecessary)?
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from campaign import ARMS, WIDE, _merge  # noqa: E402

MAIN = os.path.join(HERE, "hyperglot", "main.py")
if not os.path.exists(MAIN):
    MAIN = os.path.join(HERE, "main.py")

TRIALS = 40

# Pinned by the campaign at PAPER_DEFAULTS -- deliberately, to match the paper.
KNOWN_FIXED = {
    "epochs", "batch_size", "eval_batch_size", "max_length", "seed",
    "pooling_method", "task", "model_name_or_path", "hidden_layer",
    "precompute_hidden_states", "override_precompute", "finetune_backbone",
    "verbose", "adaptive_length", "decoder_cls_last_token", "label_scale",
}


def main() -> None:
    src = open(MAIN, encoding="utf-8", errors="ignore").read()
    flags = re.findall(r'add_argument\(\s*"--([A-Za-z0-9_]+)"', src)
    flags = sorted(set(flags))

    searched = set(WIDE)
    for space in ARMS.values():
        searched |= set(space)

    print("=" * 74)
    print("1. SPACE SIZE PER ARM, AND COVERAGE AT 40 RANDOM TRIALS")
    print("=" * 74)
    wide_n = 1
    for v in WIDE.values():
        wide_n *= len(v)
    print(f"  optimizer/architecture grid (WIDE) = {wide_n} points")
    print()
    print(f"  {'arm':<10} {'graph pts':>10} {'total space':>13} {'40 trials cover':>17}")
    for name, space in sorted(ARMS.items()):
        full = _merge(space, WIDE)
        graph_only = 1
        for k, v in space.items():
            graph_only *= len(v)
        total = 1
        for v in full.values():
            total *= len(v)
        print(f"  {name:<10} {graph_only:>10} {total:>13} {TRIALS / total:>16.2%}")

    print()
    print("=" * 74)
    print("2. CLI FLAGS THE CAMPAIGN NEVER VARIES")
    print("=" * 74)
    never = [f for f in flags if f not in searched and f not in KNOWN_FIXED]
    print(f"  searched by some arm : {len(searched & set(flags))}")
    print(f"  pinned on purpose    : {len(KNOWN_FIXED & set(flags))}")
    print(f"  never varied         : {len(never)}")
    print()
    for f in never:
        print(f"    --{f}")

    print()
    print("=" * 74)
    print("3. WHY EXHAUSTIVE SEARCH IS NOT THE GOAL")
    print("=" * 74)
    print("  P(at least one of N random draws lands in the top k% of the space):")
    print(f"  {'k%':>6} {'N=10':>9} {'N=40':>9} {'N=100':>9}")
    for k in (1, 2, 5, 10, 25):
        p = k / 100.0
        row = [1 - (1 - p) ** n for n in (10, 40, 100)]
        print(f"  {k:>5}% {row[0]:>8.0%} {row[1]:>8.0%} {row[2]:>8.0%}")
    print()
    print("  So 40 trials gives ~87% odds of reaching the top 5% of each arm's own")
    print("  space, and every arm gets the SAME 40. That is what makes the between-arm")
    print("  comparison fair. Exhaustive search would cost weeks and would not change")
    print("  the comparison -- it would only raise every arm by a similar amount.")


if __name__ == "__main__":
    main()
