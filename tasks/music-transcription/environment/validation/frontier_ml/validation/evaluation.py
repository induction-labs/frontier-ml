"""HTTP request batching, response validation, metrics and timing."""

import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import mimetypes
from pathlib import Path
import re
import threading
import time

from .errors import EvaluationError, VerificationFailure
from .progress import progress


def load_samples(path):
    path = Path(path)
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise VerificationFailure("Evaluation dataset is empty")
    ids = [str(sample["id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise VerificationFailure("Evaluation sample IDs must be unique")
    return samples


def image_url(reference, data_root):
    root = Path(data_root).resolve()
    path = (root / reference).resolve()
    if not path.is_relative_to(root):
        raise VerificationFailure("Image reference escapes evaluation data directory")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in ("image/png", "image/jpeg", "image/webp"):
        raise VerificationFailure(f"Unsupported image format: {path.name}")
    return "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode()


def timed_batch(items, function, *, concurrency, minimum_per_second=None, timeout_seconds=3600, on_complete=None):
    """Time the entire batch, including scheduling, HTTP and response validation.

    Every request receives only its remaining batch budget. On expiry no further
    tasks are submitted and queued work is cancelled, without waiting for hung
    workers. The caller immediately tears down the candidate on any failure.
    """
    if not items or not isinstance(concurrency, int) or not 1 <= concurrency <= 64:
        raise VerificationFailure("Batch must be nonempty; concurrency must be between 1 and 64")
    budget = timeout_seconds
    if minimum_per_second is not None:
        if minimum_per_second <= 0:
            raise ValueError("Throughput must be positive")
        budget = min(budget, len(items) / minimum_per_second)
    if budget <= 0 or not math.isfinite(budget):
        raise ValueError("Batch timeout must be positive and finite")
    started = time.monotonic()
    deadline = started + budget

    def run(item):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batch deadline exceeded before request")
        result = function(item, remaining)
        if time.monotonic() > deadline:
            raise TimeoutError("Batch deadline exceeded during request")
        return result

    executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="verifier-http")
    futures = {}
    try:
        futures = {executor.submit(run, item): index for index, item in enumerate(items)}
        results = [None] * len(items)
        for completed, future in enumerate(as_completed(futures, timeout=max(0, deadline - time.monotonic())), 1):
            results[futures[future]] = future.result()
            if on_complete is not None:
                on_complete(completed)
        elapsed = time.monotonic() - started
        if elapsed > budget:
            raise TimeoutError("Batch deadline exceeded")
        return results, {"elapsed_seconds": elapsed, "items": len(items),
                         "items_per_second": len(items) / max(elapsed, 1e-12),
                         "deadline_seconds": budget, "concurrency": concurrency}
    except TimeoutError as exc:
        raise VerificationFailure(f"Batch exceeded {budget:g}s deadline; score is zero") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def chat_answer(response):
    """Text of the first OpenAI-style chat completion choice."""
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VerificationFailure("Malformed OpenAI chat completion") from exc
    if not isinstance(content, str):
        raise VerificationFailure("Completion content must be text")
    return content


# Request resilience, mirroring the language suite (frontier_ml/scoring/language.py). One dropped or malformed
# response must not zero a whole run: a transport failure (HTTP error status, connection or bridge error,
# non-JSON body) is retried on the identical payload; a request that still fails, that times out (never
# retried: it would cost the timeout again), or whose response the task's parser rejects scores zero for its
# item and is recorded in details.failed_requests. More than FAILED_REQUEST_PERCENT of the requests failing is
# a broken server, not a hiccup: the evaluation fails closed. Retries and their backoff run inside the timed
# batch, so they count against the throughput deadline: a retry is the server's problem.
REQUEST_ATTEMPTS = 3            # the request plus two retries
RETRY_BACKOFF_SECONDS = (1, 3)  # before the second and the third attempt
FAILED_REQUEST_PERCENT = 5
RECORDED_ERRORS = 5             # error strings kept in details.failed_requests
SLEEP = time.sleep              # the backoff clock; tests replace it
_TIMEOUT_TEXT = re.compile(r"time(?:d\s*)?out|deadline", re.IGNORECASE)
_PARSE_ERRORS = (VerificationFailure, ValueError, TypeError, KeyError, IndexError)


class RequestFailure:
    """A request that produced no usable response: a transport failure after every permitted attempt, a
    timeout, or a response the task's parse_response rejected. Never a JSON value, so callers can tell it apart."""
    __slots__ = ("reason", "error", "attempts")

    def __init__(self, reason, error, attempts=1):
        self.reason, self.error, self.attempts = reason, str(error)[:400], attempts

    def __repr__(self):
        return f"RequestFailure({self.reason!r}, {self.error!r}, attempts={self.attempts})"


def failure_reason(error):
    """Classify a transport exception: http_status, invalid_json, response_too_large or connection are retried;
    a timeout (socket, bridge or batch deadline) is not."""
    text = str(error)
    if "returned HTTP" in text:
        return "http_status"
    if isinstance(error, TimeoutError) or _TIMEOUT_TEXT.search(text):
        return "timeout"
    if isinstance(error, json.JSONDecodeError) or "valid JSON" in text or "JSON object" in text:
        return "invalid_json"
    if "size limit" in text:
        return "response_too_large"
    return "connection"


class FailedRequests:
    """The per-item failures of one evaluation (details.failed_requests) and the fail-closed limit."""

    def __init__(self, requests, *, limit_percent=FAILED_REQUEST_PERCENT):
        if type(requests) is not int or requests < 1:
            raise ValueError("An evaluation sends a positive number of requests")
        self.requests, self.limit_percent = requests, limit_percent
        self.lock = threading.Lock()
        self.failures = []  # (item id, RequestFailure)
        self.retried = 0    # requests that succeeded only after a retry

    def record(self, item_id, failure):
        with self.lock:
            self.failures.append((str(item_id), failure))
            failed = len(self.failures)
            if failed * 100 > self.limit_percent * self.requests:
                errors = [f"{item}: {failure.error}" for item, failure in self.failures[:RECORDED_ERRORS]]
                raise VerificationFailure(
                    f"{failed} of {self.requests} requests failed, timed out or returned malformed responses, more than "
                    f"{self.limit_percent}%; the server is not serving reliably. First errors: {errors}")

    def summary(self):
        with self.lock:
            reasons = Counter(failure.reason for _, failure in self.failures)
            return {"total": len(self.failures), "requests": self.requests, "limit_percent": self.limit_percent,
                    "attempts": REQUEST_ATTEMPTS, "backoff_seconds": list(RETRY_BACKOFF_SECONDS), "retried": self.retried,
                    "reasons": dict(sorted(reasons.items())),
                    "errors": [f"{item}: {failure.error}" for item, failure in self.failures[:RECORDED_ERRORS]],
                    "ids": [item for item, _ in self.failures]}


def attempt_request(transport, method, path, payload=None, *, timeout, failures=None, sleep=None, **kwargs):
    """transport.request under the shared retry policy: the response, or a RequestFailure instead of an exception.

    Transport failures are retried on the identical payload after RETRY_BACKOFF_SECONDS, only while the backoff
    still fits in `timeout` (the remaining batch budget); a timeout is not retried. The verifier's own
    infrastructure errors (EvaluationError) propagate: the submission was not consulted.
    """
    deadline = time.monotonic() + timeout
    errors = []
    for attempt in range(REQUEST_ATTEMPTS):
        if attempt:
            backoff = RETRY_BACKOFF_SECONDS[attempt - 1]
            if deadline - time.monotonic() - backoff <= 0:
                break
            (sleep or SLEEP)(backoff)
        try:
            response = transport.request(method, path, payload, timeout=deadline - time.monotonic(), **kwargs)
        except EvaluationError:
            raise
        except Exception as error:  # the request layer only; the task's parser runs outside this function
            text = f"{type(error).__name__}: {str(error)[:200]}"
            reason = failure_reason(error)
            if reason == "timeout":
                return RequestFailure(reason, f"attempt {attempt + 1} timed out: {text}", attempt + 1)
            errors.append((reason, text))
            continue
        if attempt and failures is not None:
            with failures.lock:
                failures.retried += 1
        return response
    return RequestFailure(errors[-1][0], f"failed {len(errors)} attempts: " + " | ".join(text for _, text in errors),
                          len(errors))


def run_http_evaluation(transport, data_path, config, *, build_request, parse_response, score,
                        rounds=1, minimum_per_second=None, default_concurrency=32,
                        failed_prediction=None, per_item_mean=False):
    """Generic timed request/response evaluation; task verifiers supply the three hooks.

    build_request(sample, data_root) -> (path, json_payload)
    parse_response(sample, response) -> prediction (validated, with the sample id)
    score(samples, predictions, config) -> {"score": float, "details": {...}}

    What a failed request (see RequestFailure) scores is the task's declaration, never this function's guess;
    exactly one of the two must be given:
    failed_prediction(sample, failure) -> the prediction that scores as this task's zero for that item. Every
        row then reaches `score` with its group membership and the metric's own denominators, which is the only
        correct treatment for a macro average over groups, a rate, a corpus statistic or a multi-round
        evaluation: leaving an item out would reweight the others (a failure could then score better than a
        wrong answer, review F2 of 2026-09-30).
    per_item_mean=True declares that `score` is an equal-weight mean over the items it receives. Failed items
        are then left out of `score` and the score is scaled by scored/total items, exactly a zero for each.

    All rounds form one sustained batch: `rounds` copies of the split, round-major, at the task concurrency,
    with the throughput budget rounds * samples / minimum_per_second. Predictions carry `round`
    (index // samples) when rounds > 1. Failed requests are retried, tolerated up to FAILED_REQUEST_PERCENT
    and reported in details.failed_requests; retries count against the batch time.
    """
    if (failed_prediction is None) == (per_item_mean is not True):
        raise TypeError("run_http_evaluation needs exactly one of failed_prediction or per_item_mean=True: the task "
                        "declares what a failed request scores")
    config = dict(config or {})
    samples = load_samples(data_path)
    expected = config.get("expected_samples")
    if expected is not None and len(samples) != expected:
        raise VerificationFailure(f"Expected {expected} evaluation samples; found {len(samples)}")
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if rounds > 1 and failed_prediction is None:
        raise TypeError("A multi-round evaluation must declare failed_prediction: predictions stay grouped by round")
    root = Path(data_path).parent
    requests = [(sample, *build_request(sample, root)) for sample in samples]
    items = requests * rounds
    total, count = len(items), len(samples)
    failures = FailedRequests(total)

    def item_id(index):
        sample_id = items[index][0].get("id")
        return f"{sample_id}@{index // count}" if rounds > 1 else sample_id

    def settle(index, response):
        """The parsed prediction, or the recorded RequestFailure (which may fail the run past the limit)."""
        sample = items[index][0]
        if isinstance(response, RequestFailure):
            failure = response
        else:
            try:
                return parse_response(sample, response)
            except EvaluationError:
                raise
            except _PARSE_ERRORS as error:
                failure = RequestFailure("malformed", f"{type(error).__name__}: {str(error)[:200]}")
        failures.record(item_id(index), failure)
        return failure

    diagnostic = config.get("validation_mode") is True
    progress(config, "requests", completed=0, total=total)
    options = {"concurrency": config.get("concurrency", default_concurrency),
               "minimum_per_second": None if diagnostic else minimum_per_second,
               "timeout_seconds": config.get("inference_timeout_seconds", 3600),
               "on_complete": (lambda n: progress(config, "requests", completed=n, total=total))
               if config.get("progress_path") else None}
    if callable(batch := getattr(transport, "request_batch", None)):
        responses, timing = batch([(path, payload) for _, path, payload in items], raise_on_failure=False, **options)
        started = time.monotonic()
        results = [settle(index, response) for index, response in enumerate(responses)]
        timing["elapsed_seconds"] += time.monotonic() - started
        if timing["elapsed_seconds"] > timing["deadline_seconds"]:
            raise VerificationFailure(f"Batch exceeded {timing['deadline_seconds']:g}s deadline; score is zero")
        timing["items_per_second"] = total / max(timing["elapsed_seconds"], 1e-12)
    else:
        def request(indexed, timeout):
            index, (_, path, payload) = indexed
            return settle(index, attempt_request(transport, "POST", path, payload, timeout=timeout, failures=failures))
        results, timing = timed_batch(list(enumerate(items)), request, **options)
    if minimum_per_second is not None:
        timing["required_items_per_second"] = minimum_per_second
        timing["throughput_target_met"] = timing["items_per_second"] >= minimum_per_second
        timing["throughput_gate_enforced"] = not diagnostic
    if rounds > 1:
        timing["rounds"] = rounds
    failed = failures.summary()
    scored_samples, predictions, echoed = [], [], []
    for index, result in enumerate(results):
        sample = items[index][0]
        if isinstance(result, RequestFailure):
            echo = {"id": sample.get("id"), "request_failed": {"reason": result.reason, "error": result.error}}
            if failed_prediction is not None:
                result = failed_prediction(sample, result)
                if not isinstance(result, dict):
                    raise TypeError("failed_prediction must return a prediction dictionary")
                echo = {**result, **echo}
        else:
            echo = result
        if isinstance(result, dict):
            if rounds > 1:
                result["round"] = index // count
            scored_samples.append(sample)
            predictions.append(result)
        if rounds > 1:
            echo["round"] = index // count
        echoed.append(echo)
    if diagnostic and config.get("predictions_only"):
        result = {"score": None, "details": {"scoring": "not requested"}}
    else:
        progress(config, "scoring")
        if not predictions:
            raise VerificationFailure("Every request failed; nothing to score")
        result = score(scored_samples, predictions, config)
        if len(predictions) < total:
            # per_item_mean: failed items were left out of the declared equal-weight mean; each scores zero.
            details = result.setdefault("details", {})
            details["scored_items"], details["failed_items"] = len(predictions), total - len(predictions)
            if isinstance(result.get("score"), (int, float)) and not isinstance(result["score"], bool):
                result["score"] = result["score"] * len(predictions) / total
    result.setdefault("details", {})["timing"] = [timing]
    result["details"]["failed_requests"] = failed
    if diagnostic:
        result["predictions"] = echoed
    return result
