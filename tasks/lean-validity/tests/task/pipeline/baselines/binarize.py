"""Turn a baseline's continuous scores into yes/no answers (organizer side).

The threshold is chosen on the validation split only (the value maximizing validation
macro F1) and then applied unchanged to held-out rows.

    python binarize.py VAL_ROWS.jsonl VAL_SCORES.json HELD_ROWS.jsonl HELD_SCORES.json
    (score files: a JSON list, or {"predictions": [...]} as written by llm_baselines.py)
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # task dir (validate.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))  # repo root

from validate import macro_f1, score  # noqa: E402


def choose_threshold(scores, labels):
    """Answer yes iff score >= threshold; pick the validation-optimal threshold."""
    best = (-1.0, 0.5)
    for threshold in sorted(set(scores)) + [float("inf")]:
        value = macro_f1(labels, [s >= threshold for s in scores])
        if value > best[0]:
            best = (value, threshold)
    return best[1], best[0]


def evaluate(val_rows, val_scores, held_rows, held_scores):
    threshold, val_f1 = choose_threshold(val_scores, [r["valid"] for r in val_rows])
    held = score(held_rows, [{"id": r["id"], "valid": s >= threshold} for r, s in zip(held_rows, held_scores)])
    return {"threshold": threshold, "validation_macro_f1": val_f1, "heldout_macro_f1": held["score"],
            "heldout_accuracy": held["details"]["diagnostics"]["accuracy"],
            "heldout_predicted_valid_fraction": held["details"]["diagnostics"]["predicted_valid_fraction"]}


def _load(path):
    value = json.load(open(path))
    return value["predictions"] if isinstance(value, dict) else value


if __name__ == "__main__":
    rows = lambda p: [json.loads(l) for l in open(p)]
    print(json.dumps(evaluate(rows(sys.argv[1]), _load(sys.argv[2]), rows(sys.argv[3]), _load(sys.argv[4]))))
