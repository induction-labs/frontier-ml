#!/usr/bin/env python3
"""Validate a classifier with macro-F1 * (1 - top-label ECE), averaged across datasets.

python /task/validate.py --server-url http://127.0.0.1:8000

Up to 32 concurrent requests classify one text each against all supplied labels.
Public throughput is diagnostic: official timing runs locally on the inference H100.
Final evaluation also requires training and data-provenance review.
"""
import time

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.evaluation import load_samples
from zsc_metric import failed_prediction, score
from zsc_protocol import (CONCURRENCY, WARMUP_REQUESTS, TransportFailure, build_request, evaluate_batch,
                          failure_summary, latency_details, parse_response, prepare_samples, timed_request)

TASK = "zero-shot-classification"
DEFAULT_DATA = "/task/public/validation.jsonl"


def run(transport, data_path, config):
    if hasattr(transport, "max_response_bytes"):
        transport.max_response_bytes = None
    if config.get("concurrency", CONCURRENCY) != CONCURRENCY:
        raise ValueError("This task measures throughput at concurrency 32")
    samples = prepare_samples(load_samples(data_path))
    expected = config.get("expected_samples")
    if expected is not None and len(samples) != expected:
        raise ValueError(f"Expected {expected} samples; found {len(samples)}")
    budget = config.get("inference_timeout_seconds", 3600)
    deadline = time.monotonic() + budget
    warmup_failures = 0
    for payload in WARMUP_REQUESTS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Validation deadline exceeded")
        try:
            timed_request(transport, payload, timeout=remaining)
        except TransportFailure as exc:
            if exc.reason == "timeout":
                raise TimeoutError("Validation deadline exceeded") from exc
            warmup_failures += 1  # unscored synthetic request: noted, never fatal
        except ValueError:
            warmup_failures += 1
    # A document whose request failed after the retries (see CONTRACT.md) scores as a confidently wrong answer;
    # more than one in twenty failing raises UnreliableServer and fails the validation.
    records, timing = evaluate_batch(transport, [build_request(row)[1] for row in samples],
                                     timeout_seconds=deadline - time.monotonic())
    predictions = [failed_prediction(row, record["request_failed"]) if record.get("request_failed")
                   else parse_response(row, record["response"]) for row, record in zip(samples, records, strict=True)]
    result = score(samples, predictions)
    failed = [(row["id"], record["request_failed"]) for row, record in zip(samples, records, strict=True)
              if record.get("request_failed")]
    result["details"]["failed_requests"] = failure_summary(failed, len(samples),
                                                           retried=timing.pop("failed_requests")["retried"])
    result["details"]["warmup_failures"] = warmup_failures
    result["details"]["throughput_diagnostic"] = dict(timing, official=False)
    result["details"]["request_latency"] = latency_details([record["elapsed_seconds"] for record in records])
    if config.get("validation_mode"):
        result["predictions"] = predictions
    return result


def main(argv=None):
    def arguments(parser):
        parser.set_defaults(concurrency=CONCURRENCY)
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__, extra_arguments=arguments)


if __name__ == "__main__":
    raise SystemExit(main())
