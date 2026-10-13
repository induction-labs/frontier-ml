#!/usr/bin/env python3
"""Evaluate your AI-text detector on the public validation passages.

    python /task/validate.py --server-url http://127.0.0.1:8000

Sends POST /detect {"text": "..."} for every passage in the supplied validation file,
using the same request as final scoring. Expects {"p_ai": <number in [0, 1]>}, higher
meaning more likely AI-generated. Final scoring uses a fixed 5,000-passage subset.

Metric: the false-negative rate (FNR) at 1% false-positive rate. The threshold is set from
YOUR scores on the human-written passages so that at most 1% of them score strictly above it;
a cell's FNR is the fraction of its AI-generated passages that do NOT score strictly above
it. Cells are language x generator family x condition and receive equal weight. The
reported score is 1 - macro FNR (higher is better). Diagnostics include FNR at 5% FPR,
AUROC, and results by language, family, domain and condition. The 1% constraint is global;
individual languages can have higher false-positive rates. Observed throughput is a
diagnostic here; the final evaluation applies every requirement in /task/CONTRACT.md.
"""
import math

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import run_http_evaluation

TASK = "ai-text-detection"
DEFAULT_DATA = "/task/public/validation.jsonl"
REQUIRED_MESSAGES_PER_SECOND = 10  # diagnostic comparison only here; the verifier enforces it
FPR = 0.01
DIAGNOSTIC_FPR = 0.05


def build_request(sample, data_root):
    return "/detect", {"text": sample["text"]}


def parse_response(sample, response):
    if not isinstance(response, dict):
        raise VerificationFailure("Response must be a JSON object with p_ai")
    p = response.get("p_ai")
    # The range test alone rejects NaN and infinities; math.isfinite would overflow on a huge JSON integer.
    if type(p) not in (int, float) or not 0 <= p <= 1:
        raise VerificationFailure("p_ai must be a finite number in [0, 1]")
    return {"id": sample["id"], "p_ai": float(p)}


def failed_prediction(sample, failure):
    """A request that failed (see CONTRACT.md) scores as the worst answer for its passage: a human passage
    flagged with certainty, an AI passage missed with certainty. The FNR/FPR metric is not a per-item mean."""
    return {"id": sample["id"], "p_ai": 1.0 if sample["label"] == 0 else 0.0}


def threshold(human_scores, fpr=FPR):
    """Smallest t such that at most floor(fpr * n) human scores are strictly above t."""
    ordered = sorted(human_scores)
    return ordered[len(ordered) - 1 - math.floor(fpr * len(ordered))]


def auroc(positive, negative):
    if not positive or not negative:
        return None
    ranked = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(label for _, label in ranked[i:j])
        i = j
    return (rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative))


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def score(samples, predictions, config=None):
    if not samples or len(samples) != len(predictions):
        raise ValueError("Expected exactly one prediction per row")
    human, ai = [], []
    for row, prediction in zip(samples, predictions, strict=True):
        if prediction.get("id") != row["id"]:
            raise ValueError("Prediction ID/order mismatch")
        for field in ("language", "family", "condition", "domain"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Missing multilingual evaluation field: {field}")
        if type(row["label"]) is not int:
            raise ValueError("Labels must be 0 (human) or 1 (AI)")
        if row["label"] == 0:
            human.append((prediction["p_ai"], row))
        elif row["label"] == 1:
            ai.append((prediction["p_ai"], row))
        else:
            raise ValueError("Labels must be 0 (human) or 1 (AI)")
    if not human or not ai:
        raise ValueError("Scoring needs human and AI rows")
    human_scores = [p for p, _ in human]
    t1, t5 = threshold(human_scores, FPR), threshold(human_scores, DIAGNOSTIC_FPR)

    def slice_stats(rows, humans=None):
        values = [p for p, _ in rows]
        out = {"n": len(values), "fnr_at_1pct_fpr": sum(v <= t1 for v in values) / len(values) if values else None,
               "fnr_at_5pct_fpr": sum(v <= t5 for v in values) / len(values) if values else None,
               "auroc": auroc(values, human_scores)}
        if humans is not None:
            h = [p for p, _ in humans]
            out.update(humans=len(h), false_positive_rate_at_threshold=sum(v > t1 for v in h) / len(h) if h else None,
                       auroc_within=auroc(values, h))
        return out

    families = {}
    for p, row in ai:
        families.setdefault(row["family"], []).append((p, row))
    per_family = {name: slice_stats(rows) for name, rows in sorted(families.items())}
    per_domain = {}
    for domain in sorted({r.get("domain") for _, r in human + ai if r.get("domain")}):
        per_domain[domain] = slice_stats([x for x in ai if x[1].get("domain") == domain],
                                         [x for x in human if x[1].get("domain") == domain])
    per_language = {language: slice_stats([x for x in ai if x[1]["language"] == language],
                                         [x for x in human if x[1]["language"] == language])
                    for language in sorted({r["language"] for _, r in human + ai})}
    per_condition = {condition: slice_stats([x for x in ai if x[1]["condition"] == condition])
                     for condition in sorted({r["condition"] for _, r in ai})}
    cells = {}
    for p, row in ai:
        key = (row["language"], row["family"], row["condition"])
        cells.setdefault(key, []).append(p)
    per_cell = [{"language": key[0], "family": key[1], "condition": key[2], "n": len(values),
                 "fnr_at_1pct_fpr": sum(v <= t1 for v in values) / len(values),
                 "fnr_at_5pct_fpr": sum(v <= t5 for v in values) / len(values)}
                for key, values in sorted(cells.items())]
    macro_fnr = _mean(f["fnr_at_1pct_fpr"] for f in per_cell)
    details = {"metric": "macro_fnr_at_1pct_fpr", "macro_fnr_at_1pct_fpr": macro_fnr, "reward": 1 - macro_fnr,
               "reward_definition": "1 - mean language/family/condition FNR at global 1% FPR", "humans": len(human), "ai": len(ai), "threshold_1pct": t1, "threshold_5pct": t5,
               "false_positive_rate": sum(h > t1 for h in human_scores) / len(human_scores),
               "macro_fnr_at_5pct_fpr": _mean(f["fnr_at_5pct_fpr"] for f in per_cell),
               "macro_auroc": _mean(f["auroc"] for f in per_family.values()),
               "families": per_family, "domains": per_domain, "languages": per_language,
               "conditions": per_condition, "cells": per_cell}
    return {"score": 1 - macro_fnr, "details": details}


def run(transport, data_path, config):
    """The metric over every passage, plus observed throughput as a diagnostic (never enforced here)."""
    config = dict(config or {}, validation_mode=True)
    result = run_http_evaluation(transport, data_path, config, build_request=build_request,
                                 parse_response=parse_response, score=score, failed_prediction=failed_prediction)
    timing = result["details"].pop("timing")[0]
    result["details"]["throughput"] = {
        "observed_messages_per_second": timing["items_per_second"], "elapsed_seconds": timing["elapsed_seconds"],
        "concurrency": timing["concurrency"], "required_messages_per_second": REQUIRED_MESSAGES_PER_SECOND,
        "meets_requirement": timing["items_per_second"] >= REQUIRED_MESSAGES_PER_SECOND, "enforced": False}
    return result


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
