#!/usr/bin/env python3
"""Evaluate your Lean proof-validity server on the public validation attempts.

    python /task/validate.py --server-url http://127.0.0.1:8000

Sends POST /check {"lean": "<header + theorem + proof>"} for every attempt, exactly as
the trusted final client does, and expects {"valid": true} or {"valid": false} (a JSON
boolean; anything else fails the run). The headline metric is macro F1 over the two
classes "compiles" and "does not compile", pooled over all attempts: one confusion
matrix, F1 of each class, averaged. Always answering yes (or always no) scores 1/3 on
these balanced splits; random answers score about 0.5. Accuracy, per-class precision and
recall, recall of non-compiling attempts by error category, and per-theorem accuracy are
diagnostics only.

Throughput is reported, not enforced: the final run must answer all 3,540 held-out
attempts within 600 seconds (about 5.9 attempts per second at up to 64 concurrent
requests), so the report shows the observed items/s against that rate
(required_items_per_second, throughput_target_met). The rate does not depend on set
size; a validation batch finishing in under 526 / 5.9 = 89 s meets it. A small batch is
only indicative of full-batch speed. Compliance checks (no-Lean file scan, architecture
review) run only in the trusted verifier.
"""
import re
from collections import defaultdict
from statistics import mean

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import run_http_evaluation

TASK = "lean-validity"
DEFAULT_DATA = "/task/public/validation.jsonl"
HELDOUT_ATTEMPTS = 3540
BATCH_BUDGET_SECONDS = 600
REQUIRED_ITEMS_PER_SECOND = HELDOUT_ATTEMPTS / BATCH_BUDGET_SECONDS  # diagnostic target, enforced only by the verifier
ERROR_RULES = [
    ("heartbeats", r"heartbeats|maximum recursion"),
    ("unknown_identifier", r"unknown (identifier|constant)|unknown namespace"),
    ("unsolved_goals", r"unsolved goals"),
    ("linarith_failed", r"linarith failed"),
    ("type_mismatch", r"type mismatch|application type mismatch|failed to synthesize"),
    ("syntax", r"unexpected token|expected|unknown tactic"),
    ("simp_failed", r"simp made no progress|simp failed|norm_num failed"),
    ("omega_failed", r"omega could not"),
    ("rewrite_failed", r"did not find instance of the pattern|motive is not type correct"),
    ("no_goals", r"no goals"),
]


def error_category(errors):
    """Coarse category of the first Lean error message (diagnostic only)."""
    if not errors:
        return "none"
    for name, pattern in ERROR_RULES:
        if re.search(pattern, errors[0]):
            return name
    return "other"


def build_request(sample, data_root):
    return "/check", {"lean": sample["lean"]}


def parse_response(sample, response):
    if not isinstance(response, dict) or "valid" not in response:
        raise VerificationFailure("Response must be a JSON object with a boolean valid")
    if type(response["valid"]) is not bool:
        raise VerificationFailure("valid must be a JSON boolean (true or false)")
    return {"id": sample["id"], "valid": response["valid"]}


def failed_prediction(sample, failure):
    """A request that failed (see CONTRACT.md) counts as the wrong verdict for its attempt; macro F1 is not a
    per-item mean, so the shared evaluator cannot simply leave the attempt out."""
    return {"id": sample["id"], "valid": not sample["valid"]}


def _f1(tp, fp, fn):
    """F1 of one class; 0 when the class is never predicted or never present."""
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def confusion(labels, answers):
    """Counts with "compiles" as the positive class."""
    tp = sum(1 for y, a in zip(labels, answers) if y and a)
    fp = sum(1 for y, a in zip(labels, answers) if not y and a)
    fn = sum(1 for y, a in zip(labels, answers) if y and not a)
    tn = len(labels) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def macro_f1(labels, answers):
    c = confusion(labels, answers)
    return (_f1(c["tp"], c["fp"], c["fn"]) + _f1(c["tn"], c["fn"], c["fp"])) / 2


def score(samples, predictions, config=None):
    if not samples or len(samples) != len(predictions):
        raise ValueError("Expected exactly one prediction per attempt")
    labels, answers = [], []
    per_theorem = defaultdict(list)
    categories = defaultdict(list)
    for row, prediction in zip(samples, predictions, strict=True):
        if not isinstance(prediction, dict) or prediction.get("id") != row["id"]:
            raise ValueError("Prediction ID/order mismatch")
        answer = prediction.get("valid")
        if type(answer) is not bool:
            raise ValueError("valid must be a boolean")
        valid = row["valid"]
        if type(valid) is not bool:
            raise ValueError("Reference label must be boolean")
        labels.append(valid)
        answers.append(answer)
        per_theorem[row["theorem"]].append(answer == valid)
        if not valid:
            category = row.get("error_category") or error_category(row.get("errors") or [])
            categories[category].append(not answer)
    c = confusion(labels, answers)

    def ratio(a, b):
        return a / b if b else None

    details = {
        "metric": "macro_f1",
        "samples": len(samples),
        "theorems": len(per_theorem),
        "confusion": c,
        "per_theorem_accuracy": {t: mean(v) for t, v in sorted(per_theorem.items())},
        "diagnostics": {
            "accuracy": mean(float(y == a) for y, a in zip(labels, answers)),
            "compiles": {"precision": ratio(c["tp"], c["tp"] + c["fp"]), "recall": ratio(c["tp"], c["tp"] + c["fn"]),
                         "f1": _f1(c["tp"], c["fp"], c["fn"])},
            "does_not_compile": {"precision": ratio(c["tn"], c["tn"] + c["fn"]), "recall": ratio(c["tn"], c["tn"] + c["fp"]),
                                 "f1": _f1(c["tn"], c["fn"], c["fp"])},
            "invalid_recall_by_error_category": {k: {"n": len(v), "recall": mean(v)} for k, v in sorted(categories.items())},
            "valid_fraction": mean(float(y) for y in labels),
            "predicted_valid_fraction": mean(float(a) for a in answers),
        },
    }
    return {"score": macro_f1(labels, answers), "details": details}


def run(transport, data_path, config):
    """Send every attempt, compute the metric and report throughput against the final rate (never enforced here)."""
    config = dict(config or {}, validation_mode=True)
    result = run_http_evaluation(transport, data_path, config, build_request=build_request,
                                 parse_response=parse_response, score=score, failed_prediction=failed_prediction,
                                 minimum_per_second=REQUIRED_ITEMS_PER_SECOND, default_concurrency=64)
    for timing in result["details"].get("timing", []):
        timing.pop("deadline_seconds", None)
    return result


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
