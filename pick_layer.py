"""Pick which transformer layer a task should run its campaign at.

Reads a layer-probe CSV (baseline arm only, several layers) and prints the layer
number with the best score. Prints nothing but the integer so it can be captured
in a shell variable.

WHY NOT JUST ALWAYS USE LAYER 8: layer 8 won on CoLA, but it was chosen by
searching 6 layers on CoLA's dev set. Whether that transfers is precisely the
open question, so each task picks its own layer from its own probe. Falls back
to 12 (GLOT's default, `last_hidden_state`) if the CSV is missing or unreadable,
so a probe failure degrades to upstream behaviour rather than to nonsense.

A layer only displaces 12 if it beats it by more than the task's noise floor --
otherwise we would be re-running the same dev-set overfitting one level down.
"""
import csv
import sys
from collections import defaultdict

# Same figures used elsewhere in the project: same-config, different-seed spread.
NOISE = {"cola": 0.81, "stsb": 0.53, "rte": 1.40, "mrpc": 1.20}
DEFAULT_LAYER = 12


def main():
    path = sys.argv[1]
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        print(DEFAULT_LAYER)
        return

    best = defaultdict(lambda: float("-inf"))
    task = ""
    for r in rows:
        try:
            score = float(r["score"])
        except (KeyError, TypeError, ValueError):
            continue
        setting = r.get("setting", "")
        task = r.get("task", task)
        layer = int(setting.split("_L")[1]) if "_L" in setting else DEFAULT_LAYER
        best[layer] = max(best[layer], score)

    if not best:
        print(DEFAULT_LAYER)
        return

    tol = NOISE.get(task, 1.0)
    ref = best.get(DEFAULT_LAYER, float("-inf"))
    winner, top = max(best.items(), key=lambda kv: kv[1])
    # Only move off the default if the margin clears the noise floor.
    print(winner if (ref == float("-inf") or top - ref > tol) else DEFAULT_LAYER)


if __name__ == "__main__":
    main()
