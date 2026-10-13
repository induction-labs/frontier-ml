"""Shared request preparation and timing rules; no benchmark provenance or labels on the wire."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
import math
import random
from threading import Lock
import time

from zsc_metric import probabilities, validate_samples

MINIMUM_DOCS_PER_SECOND = 50
CONCURRENCY = 32
WARMUP_REQUESTS = [
    {"text": "The parcel arrived on Tuesday.", "labels": ["A delivery update.", "A cooking instruction."]},
    {"text": "Bake the bread for twenty minutes.", "labels": ["A delivery update.", "A cooking instruction."]},
] * 8

# The shared per-item request policy of frontier_ml.validation.evaluation, restated here because the trusted
# runner executes inside the inference sandbox with `python -I` and only these stdlib modules (a test pins the
# values equal). One failed or malformed response scores zero for its document, never for the run.
REQUEST_ATTEMPTS = 3            # the request plus two retries
RETRY_BACKOFF_SECONDS = (1, 3)  # before the second and the third attempt
FAILED_REQUEST_PERCENT = 5
RECORDED_ERRORS = 5             # error strings kept in the failed_requests summary
SLEEP = time.sleep              # the backoff clock; tests replace it


class TransportFailure(Exception):
    """A request without a usable response body: HTTP status, connection error, timeout or non-JSON. Retried on
    the identical payload (a timeout is not). A body that parses but violates the interface raises ValueError
    from zsc_metric.probabilities instead: 'malformed', never retried."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class UnreliableServer(RuntimeError):
    """More than FAILED_REQUEST_PERCENT of the scored requests failed: a broken server, not a hiccup."""


def failure_summary(failures, requests, *, retried=0):
    """The failed_requests record: `failures` is a list of (item id, failure dict with reason and error)."""
    reasons = Counter(failure["reason"] for _, failure in failures)
    return {"total": len(failures), "requests": requests, "limit_percent": FAILED_REQUEST_PERCENT,
            "attempts": REQUEST_ATTEMPTS, "backoff_seconds": list(RETRY_BACKOFF_SECONDS), "retried": retried,
            "reasons": dict(sorted(reasons.items())),
            "errors": [f"{item}: {failure['error']}" for item, failure in failures[:RECORDED_ERRORS]],
            "ids": [str(item) for item, _ in failures]}


def over_limit_message(failures, requests):
    errors = [f"{item}: {failure['error']}" for item, failure in failures[:RECORDED_ERRORS]]
    return (f"{len(failures)} of {requests} requests failed, timed out or returned malformed responses, more than "
            f"{FAILED_REQUEST_PERCENT}%; the server is not serving reliably. First errors: {errors}")


class FailedRequests:
    """The per-document failures of one batch and the fail-closed limit (thread-safe)."""

    def __init__(self, requests):
        if type(requests) is not int or requests < 1:
            raise ValueError("A batch sends a positive number of requests")
        self.requests, self.lock, self.failures, self.retried = requests, Lock(), [], 0

    def record(self, item_id, failure):
        with self.lock:
            self.failures.append((item_id, failure))
            if len(self.failures) * 100 > FAILED_REQUEST_PERCENT * self.requests:
                raise UnreliableServer(over_limit_message(self.failures, self.requests))

    def count_retry(self):
        with self.lock:
            self.retried += 1

    def summary(self):
        with self.lock:
            return failure_summary(list(self.failures), self.requests, retried=self.retried)


def prepare_samples(samples, rng=None):
    validate_samples(samples)
    rng = rng or random.Random(20260925)
    prepared = []
    for row in samples:
        order = list(range(len(row["labels"])))
        rng.shuffle(order)
        prepared.append(dict(row, labels=[row["labels"][j] for j in order], target=order.index(row["target"])))
    rng.shuffle(prepared)
    return prepared


def build_request(sample, data_root=None):
    return "/classify", {"text": sample["text"], "labels": list(sample["labels"])}


def parse_response(sample, response):
    return {"id": sample["id"], "probabilities": probabilities(response, len(sample["labels"]))}


def timed_request(transport, payload, *, timeout, clock=time.perf_counter):
    """One request: (validated response, seconds). TransportFailure for no usable body, ValueError for a body that
    violates the interface."""
    if timeout <= 0:
        raise TransportFailure("timeout", "batch deadline reached before the request")
    body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    started = clock()
    try:
        status, raw = transport.raw_request("POST", "/classify", body=body,
                                           headers={"Content-Type": "application/json"}, timeout=timeout)
    except TimeoutError as exc:  # socket timeouts included: the request outlived the remaining batch budget
        raise TransportFailure("timeout", f"{type(exc).__name__}: {str(exc)[:200]}") from exc
    except (OSError, http.client.HTTPException, RuntimeError) as exc:
        reason = "response_too_large" if "size limit" in str(exc) else "connection"
        raise TransportFailure(reason, f"{type(exc).__name__}: {str(exc)[:200]}") from exc
    elapsed = clock() - started
    if status != 200:
        raise TransportFailure("http_status", f"Expected HTTP 200, got {status}")
    try:
        response = json.loads(raw)
    except ValueError as exc:
        raise TransportFailure("invalid_json", "Response body is not valid JSON") from exc
    return {"probabilities": probabilities(response, len(payload["labels"]))}, elapsed


def attempt_request(transport, payload, *, deadline, clock=time.perf_counter, sleep=None):
    """timed_request under the shared retry policy, as a record: {"response", "elapsed_seconds", "attempts"} plus
    "request_failed": {"reason", "error"} (with response None) when no attempt produced a valid answer.

    Transport failures are retried on the identical payload after RETRY_BACKOFF_SECONDS while the backoff still
    fits before `deadline`; a timeout and a malformed body are not retried. elapsed_seconds spans every attempt
    and backoff, so a failed document costs the timed batch exactly what the server made it cost.
    """
    started = clock()
    errors, attempts = [], 0

    def record(response=None, failure=None):
        result = {"response": response, "elapsed_seconds": clock() - started, "attempts": attempts}
        if failure is not None:
            result["request_failed"] = failure
        return result

    for attempt in range(REQUEST_ATTEMPTS):
        if attempt:
            backoff = RETRY_BACKOFF_SECONDS[attempt - 1]
            if deadline - clock() - backoff <= 0:
                break
            (sleep or SLEEP)(backoff)
        attempts = attempt + 1
        try:
            response, _ = timed_request(transport, payload, timeout=deadline - clock(), clock=clock)
        except TransportFailure as exc:
            text = f"{exc.reason}: {str(exc)[:200]}"
            if exc.reason == "timeout":
                return record(failure={"reason": "timeout", "error": f"attempt {attempts} timed out: {text}"})
            errors.append((exc.reason, text))
            continue
        except ValueError as exc:
            return record(failure={"reason": "malformed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        return record(response)
    return record(failure={"reason": errors[-1][0],
                           "error": f"failed {len(errors)} attempts: " + " | ".join(text for _, text in errors)})


def latency_details(latencies):
    if not latencies or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in latencies):
        raise ValueError("Missing or invalid trusted timings")
    ordered = sorted(latencies)
    return {"requests": len(ordered),
            "mean_seconds": math.fsum(ordered) / len(ordered), "max_seconds": ordered[-1],
            "p95_seconds": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def throughput_details(count, elapsed):
    if type(count) is not int or count <= 0 or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Missing or invalid trusted batch timing")
    return {"documents": count, "concurrency": CONCURRENCY,
            "minimum_docs_per_second": MINIMUM_DOCS_PER_SECOND,
            "elapsed_seconds": elapsed, "docs_per_second": count / elapsed,
            "deadline_seconds": count / MINIMUM_DOCS_PER_SECOND,
            "passed": elapsed <= count / MINIMUM_DOCS_PER_SECOND}


def evaluate_batch(transport, requests, *, timeout_seconds, sleep=None):
    """One continuous timed batch. Queueing, serialization, HTTP, validation, retries and backoff all count.

    A document whose request failed after the retries carries "request_failed" (response None) in its record;
    the timing carries the failed_requests summary. More than FAILED_REQUEST_PERCENT failed documents raise
    UnreliableServer; the batch deadline is enforced regardless of failures.
    """
    if not requests or type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Expected requests and a positive finite batch timeout")
    started = time.perf_counter()
    deadline = started + timeout_seconds
    failures = FailedRequests(len(requests))

    def call(index, payload):
        if deadline - time.perf_counter() <= 0:
            raise TimeoutError("Whole-batch deadline exceeded")
        record = attempt_request(transport, payload, deadline=deadline, sleep=sleep)
        if time.perf_counter() > deadline:
            raise TimeoutError("Whole-batch deadline exceeded")
        failure = record.get("request_failed")
        if failure is not None:
            failures.record(index, failure)
        elif record["attempts"] > 1:
            failures.count_retry()
        return {"index": index, **record}

    pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
    try:
        futures = {pool.submit(call, index, payload): index for index, payload in enumerate(requests)}
        records = [None] * len(requests)
        for future in as_completed(futures, timeout=max(0, deadline - time.perf_counter())):
            records[futures[future]] = future.result()
        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            raise TimeoutError("Whole-batch deadline exceeded")
        return records, dict(throughput_details(len(records), elapsed), failed_requests=failures.summary())
    except TimeoutError as exc:
        raise TimeoutError("Whole-batch deadline exceeded") from exc
    finally:
        # The isolated official runner exits without joining stalled HTTP workers.
        pool.shutdown(wait=False, cancel_futures=True)
