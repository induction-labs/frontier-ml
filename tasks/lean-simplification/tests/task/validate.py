#!/usr/bin/env python3
"""Evaluate a proof simplifier.

python /task/validate.py --server-url http://127.0.0.1:8000 --checker-url http://127.0.0.1:8124

The checker must be separate from the candidate server. Without --checker-url,
collect predictions only; no correctness score is reported. Score is mean valid
relative token reduction. Unchanged, invalid and empty proofs earn zero.
"""
import math
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import (FAILED_REQUEST_PERCENT, FailedRequests, RequestFailure, attempt_request,
                                             load_samples, timed_batch)
from frontier_ml.validation.progress import progress
from simplification_client import CheckerClient
from simplification_metric import MAX_PROOF_BYTES, proof_length, proof_policy, reduction

TASK = "lean-simplification"
DEFAULT_DATA = "/task/public/validation.jsonl"


def build_request(sample, data_root):
    return "/simplify", {"statement": sample["statement"], "proof": sample["proof"]}


def parse_response(sample, response):
    if not isinstance(response, dict) or set(response) != {"proof"} or not isinstance(response["proof"], str):
        raise VerificationFailure('Expected exactly {"proof": <string>}')
    try:
        size = len(response["proof"].encode("utf-8"))
    except UnicodeError as exc:
        raise VerificationFailure("Proof must be valid UTF-8") from exc
    if size > MAX_PROOF_BYTES:
        raise VerificationFailure("Proof exceeds 65,536 UTF-8 bytes")
    return {"id": sample["id"], "proof": response["proof"]}


def failed_prediction(sample, failure):
    """A request that failed after the shared retries, timed out or answered in an undocumented shape
    (see CONTRACT.md) scores as the unchanged proof: zero reduction, and no checker time is spent on it."""
    return {"id": sample["id"], "proof": sample["proof"],
            "request_failed": {"reason": failure.reason, "error": failure.error}}


def score(samples, predictions, config=None):
    config = config or {}
    checker = config.get("proof_checker")
    if not callable(checker):
        raise VerificationFailure("A trusted proof checker is required for a score")
    if not samples or len(samples) != len(predictions):
        raise VerificationFailure("Expected one prediction per input")
    for row, prediction in zip(samples, predictions, strict=True):
        if prediction.get("id") != row["id"]:
            raise VerificationFailure("Prediction IDs/order differ from inputs")
        parse_response(row, {"proof": prediction.get("proof")})

    def measure(pair):
        row, pred = pair
        original = proof_length(row["proof"])
        if original <= 0:
            raise VerificationFailure("Input proof has no tokens")
        problem = proof_policy(pred["proof"])
        if pred.get("request_failed"):
            length, valid, status = original, False, "request_failed"
        elif problem:
            length, valid, status = 0, False, "policy"
        else:
            length = proof_length(pred["proof"])
            if pred["proof"] == row["proof"]:
                valid, status = True, "unchanged"
            else:
                verdict = checker(row, pred["proof"])
                valid, status = verdict["valid"], verdict["status"]
                if type(valid) is not bool:
                    raise VerificationFailure("Nonboolean checker verdict")
        return {"id": row["id"], "original_tokens": original, "candidate_tokens": length,
                "valid": valid, "status": status, "reduction": reduction(original, length, valid)}

    records = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for record in pool.map(measure, zip(samples, predictions, strict=True)):
            records.append(record)
            progress(config, "proof_checking", completed=len(records), total=len(samples))
    n = len(records)
    mean = sum(r["reduction"] for r in records) / n
    se = math.sqrt(sum((r["reduction"] - mean) ** 2 for r in records) / (n * (n - 1))) if n > 1 else None
    return {"score": mean, "details": {"metric": "mean_valid_token_reduction", "samples": n,
            "valid_fraction": sum(r["valid"] for r in records) / n,
            "shorter_fraction": sum(r["reduction"] > 0 for r in records) / n,
            "standard_error": se, "records": records}}


def collect_predictions(transport, data_path, config, on_window=None, limit_percent=FAILED_REQUEST_PERCENT):
    """Eight simultaneous requests share each 60-second generation window.

    Native Harbor batches time HTTP inside the inference sandbox, so provider
    upload/exec latency does not consume the candidate's generation allowance.
    For the self-check only: `on_window(seconds)` hears each window's duration, and
    `limit_percent` replaces the failed-request limit (its sample is far smaller).
    """
    samples = load_samples(data_path)
    if config.get("expected_samples") is not None and len(samples) != config["expected_samples"]:
        raise VerificationFailure("Unexpected evaluation sample count")
    concurrency = config.get("concurrency", 8)
    per_proof = config.get("proof_timeout_seconds", 60)
    if type(concurrency) is not int or not 1 <= concurrency <= 8:
        raise VerificationFailure("Proof generation concurrency must be between 1 and 8")
    if type(per_proof) not in (int, float) or not math.isfinite(per_proof) or per_proof <= 0:
        raise VerificationFailure("Proof generation timeout must be positive and finite")
    budget = min(config.get("inference_timeout_seconds", 2700), config.get("batch_budget_seconds", 2700))
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise VerificationFailure("Proof generation batch budget must be positive and finite")
    # The shared per-item policy (frontier_ml.validation.evaluation): a transport failure is retried twice on the
    # identical payload inside the window, a request that still fails or answers in an undocumented shape scores
    # as the unchanged proof, and more than 5% of the run's requests failing fails the run closed.
    failures = FailedRequests(len(samples), limit_percent=limit_percent)

    def settle(row, response):
        if not isinstance(response, RequestFailure):
            try:
                return parse_response(row, response)
            except (VerificationFailure, ValueError, TypeError, KeyError, IndexError) as error:
                response = RequestFailure("malformed", f"{type(error).__name__}: {str(error)[:200]}")
        failures.record(row["id"], response)
        return failed_prediction(row, response)

    predictions, elapsed, retried = [], 0.0, 0
    for offset in range(0, len(samples), concurrency):
        chunk = samples[offset:offset + concurrency]
        requests = [build_request(row, Path(data_path).parent) for row in chunk]
        remaining = min(per_proof, budget - elapsed)
        if remaining <= 0:
            raise VerificationFailure("Proof generation batch deadline exceeded")
        progress(config, "requests", completed=offset, total=len(samples))
        batch = getattr(transport, "request_batch", None)
        if callable(batch):
            responses, timing = batch(requests, concurrency=concurrency, timeout_seconds=remaining, raise_on_failure=False)
            retried += timing.get("retried_requests", 0)
        else:
            responses, timing = timed_batch(requests,
                lambda item, timeout: attempt_request(transport, "POST", item[0], item[1], timeout=timeout, failures=failures),
                concurrency=concurrency, timeout_seconds=remaining)
        started = time.monotonic()
        results = [settle(row, response) for row, response in zip(chunk, responses, strict=True)]
        duration = timing["elapsed_seconds"] + time.monotonic() - started
        if not 0 <= duration <= remaining:
            raise VerificationFailure(f"Proof generation exceeded the {remaining:g}s request window")
        predictions.extend(results)
        elapsed += duration
        if on_window is not None:
            on_window(duration)
        progress(config, "requests", completed=len(predictions), total=len(samples))
    timing = {"elapsed_seconds": elapsed, "items": len(samples), "concurrency": concurrency,
              "items_per_second": len(samples) / max(elapsed, 1e-12), "deadline_seconds": budget,
              "request_window_seconds": per_proof, "batches": math.ceil(len(samples) / concurrency),
              "retried_requests": retried + failures.retried,
              "throughput_gate_enforced": not config.get("validation_mode", False),
              "failed_requests": failures.summary()}
    return samples, predictions, timing


def run(transport, data_path, config):
    config = dict(config)
    if not config.get("proof_checker") and config.get("checker_url"):
        config["proof_checker"] = CheckerClient(config["checker_url"], config.get("checker_fingerprint"))
    if not config.get("proof_checker") and config.get("validation_mode"):
        config["predictions_only"] = True
    samples, predictions, timing = collect_predictions(transport, data_path, config)
    if config.get("validation_mode") and config.get("predictions_only"):
        result = {"score": None, "details": {"scoring": "not requested"}}
    else:
        result = score(samples, predictions, config)
    result["details"]["failed_requests"] = timing.pop("failed_requests")
    result["details"]["timing"] = [timing]
    if config.get("validation_mode"):
        result["predictions"] = predictions
    return result


def main(argv=None):
    def arguments(parser):
        parser.set_defaults(concurrency=8)
        parser.add_argument("--checker-url", help="trusted isolated checker origin")
        parser.add_argument("--checker-fingerprint", help="expected immutable checker fingerprint")
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=arguments,
                       extra_config=lambda a: {"checker_url": a.checker_url, "checker_fingerprint": a.checker_fingerprint})


if __name__ == "__main__":
    raise SystemExit(main())
