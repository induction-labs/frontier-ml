"""Greedy speculative-decoding walk over a frozen reference continuation, and the acceptance-length metric.
"""
from concurrent.futures import ThreadPoolExecutor
import time

from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import (FAILED_REQUEST_PERCENT, RECORDED_ERRORS, RequestFailure,
                                               attempt_request)

DRAFT_TOKENS = 16           # proposals per request; the acceptance length tau is at most DRAFT_TOKENS + 1
TAU_FLOOR = 4.6             # score 0: untouched 4-bit Qwen3-1.7B (task.yaml score_normalization)
TAU_CEILING = 7.0           # score 1: owner-set target (task.yaml score_normalization)
MAX_TOKEN_ID = 2 ** 31 - 1  # ids above the target vocabulary are allowed but can never be accepted
PATH = "/draft"
# Per-request ceiling at the round barrier, stated in the task.yaml prompt and evaluation text. A request that has
# not answered by then is a timeout for its own item (an empty draft, never retried) and the round moves on, so a
# hung request costs 60 s of the serving budget instead of all of it. Wide margin for an honest drafter: a 16-token
# draft from a model under the 1 GB cap takes well under a second on an H100, the seven graded servers of
# 2026-09-28/29 averaged 3.9-7.3 s for the slowest request of a round at 32 in flight (serving_seconds / rounds),
# and even a server that answers its 32 in-flight requests one at a time with 4-bit HF generate finishes the last
# in about 20-25 s. At the 256-round worst case the 10,800 s budget (42 s per round) binds before this does.
REQUEST_TIMEOUT_SECONDS = 60


def validate_sample(sample):
    """A frozen evaluation row: {"id", "prompt_tokens": [int], "reference": [int], ...}."""
    for key in ("prompt_tokens", "reference"):
        tokens = sample.get(key)
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or not 0 <= t <= MAX_TOKEN_ID
                                                              for t in tokens):
            raise VerificationFailure(f"Sample {sample.get('id')!r} has an invalid {key}")


def build_request(sample, position):
    return {"tokens": sample["prompt_tokens"] + sample["reference"][:position]}


def parse_draft(response):
    """The proposed token ids; raises on anything but {"draft": [0..16 ints]}."""
    if not isinstance(response, dict) or "draft" not in response:
        raise VerificationFailure('Response must be a JSON object with a "draft" list')
    draft = response["draft"]
    if not isinstance(draft, list) or len(draft) > DRAFT_TOKENS:
        raise VerificationFailure(f'"draft" must be a list of at most {DRAFT_TOKENS} token ids')
    if any(type(t) is not int or not 0 <= t <= MAX_TOKEN_ID for t in draft):
        raise VerificationFailure('"draft" entries must be non-negative integer token ids')
    return draft


def accepted_length(draft, reference, position):
    """Length of the longest prefix of `draft` that equals reference[position:]."""
    accepted = 0
    for proposed, actual in zip(draft, reference[position:position + DRAFT_TOKENS]):
        if proposed != actual:
            break
        accepted += 1
    return accepted


class FailureLedger:
    """Per-request failures: each scores as an empty draft; more than FAILED_REQUEST_PERCENT fails the run.

    The number of requests depends on the drafts, so the limit is applied to the final count. During the walk the
    run stops as soon as failure is certain: past FAILED_REQUEST_PERCENT of the largest possible request count
    (one request per reference token).
    """

    def __init__(self, most_requests):
        self.most_requests = most_requests
        self.sent, self.failures, self.retried = 0, [], 0

    def record(self, item_id, failure):
        self.failures.append((item_id, failure))

    def check(self, *, final):
        if len(self.failures) * 100 > FAILED_REQUEST_PERCENT * (max(self.sent, 1) if final else self.most_requests):
            errors = [f"{item}: {failure.error}" for item, failure in self.failures[:RECORDED_ERRORS]]
            raise VerificationFailure(
                f"{len(self.failures)} of {self.sent} requests failed, timed out or returned malformed responses, more "
                f"than {FAILED_REQUEST_PERCENT}%; the server is not serving reliably. First errors: {errors}")

    def summary(self):
        reasons = {}
        for _, failure in self.failures:
            reasons[failure.reason] = reasons.get(failure.reason, 0) + 1
        return {"total": len(self.failures), "requests": self.sent, "limit_percent": FAILED_REQUEST_PERCENT,
                "retried": self.retried, "reasons": dict(sorted(reasons.items())),
                "errors": [f"{item}: {failure.error}" for item, failure in self.failures[:RECORDED_ERRORS]],
                "ids": [item for item, _ in self.failures[:100]]}


def send_round(transport, payloads, *, concurrency, timeout_seconds):
    """One request per active prompt. Returns (responses or RequestFailure, serving seconds, retried count).

    Each request may take the smaller of REQUEST_TIMEOUT_SECONDS and the remaining serving budget; one that has not
    answered by then comes back as a RequestFailure("timeout") for its item and is never retried. The trusted
    grading transport batches the round inside the inference sandbox and times it with its own clock; a plain HTTP
    transport (local development) sends it from a thread pool timed here.
    """
    if callable(batch := getattr(transport, "request_batch", None)):
        responses, timing = batch([(PATH, payload) for payload in payloads], concurrency=concurrency,
                                  timeout_seconds=timeout_seconds, raise_on_failure=False,
                                  request_timeout_seconds=REQUEST_TIMEOUT_SECONDS)
        return responses, timing["elapsed_seconds"], timing.get("retried_requests", 0)
    started = time.monotonic()
    deadline = started + timeout_seconds

    def one(payload):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return RequestFailure("timeout", "Serving budget exhausted before the request")
        return attempt_request(transport, "POST", PATH, payload, timeout=min(remaining, REQUEST_TIMEOUT_SECONDS))

    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="draft-http") as pool:
        responses = list(pool.map(one, payloads))
    return responses, time.monotonic() - started, 0


def walk(transport, samples, *, concurrency, serving_budget_seconds, enforce_budget=True, on_round=None):
    """Run every prompt's greedy walk. Returns per-sample traces plus timing and failures.

    All active prompts advance together: each round sends one request per unfinished prompt, so the number of
    rounds is the largest number of verification steps any prompt needs.
    """
    for sample in samples:
        validate_sample(sample)
    traces = {sample["id"]: {"position": 0, "steps": 0, "accepted": []} for sample in samples}
    ledger = FailureLedger(sum(len(sample["reference"]) for sample in samples))
    serving_seconds, rounds = 0.0, 0
    active = list(samples)
    while active:
        remaining = serving_budget_seconds - serving_seconds
        if remaining <= 0:
            if enforce_budget:
                raise VerificationFailure(f"Serving exceeded the {serving_budget_seconds:g}s budget; score is zero")
            remaining = 1e9
        payloads = [build_request(sample, traces[sample["id"]]["position"]) for sample in active]
        responses, elapsed, retried = send_round(transport, payloads, concurrency=concurrency,
                                                 timeout_seconds=remaining)
        serving_seconds += elapsed
        rounds += 1
        ledger.sent += len(payloads)
        ledger.retried += retried
        if enforce_budget and serving_seconds > serving_budget_seconds:
            raise VerificationFailure(f"Serving exceeded the {serving_budget_seconds:g}s budget; score is zero")
        still_active = []
        for sample, response in zip(active, responses, strict=True):
            trace = traces[sample["id"]]
            draft = []
            if isinstance(response, RequestFailure):
                ledger.record(f"{sample['id']}@{trace['position']}", response)
            else:
                try:
                    draft = parse_draft(response)
                except VerificationFailure as error:
                    ledger.record(f"{sample['id']}@{trace['position']}", RequestFailure("malformed", error))
            accepted = accepted_length(draft, sample["reference"], trace["position"])
            trace["accepted"].append(accepted)
            trace["steps"] += 1
            trace["position"] += accepted + 1
            if trace["position"] < len(sample["reference"]):
                still_active.append(sample)
        ledger.check(final=False)
        active = still_active
        if on_round is not None:
            on_round(rounds, len(samples) - len(active), len(samples))
    ledger.check(final=True)
    timing = {"serving_seconds": serving_seconds, "serving_budget_seconds": serving_budget_seconds,
              "budget_enforced": enforce_budget, "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
              "rounds": rounds, "requests": ledger.sent,
              "requests_per_second": ledger.sent / max(serving_seconds, 1e-12), "concurrency": concurrency,
              "timing_scope": ("trusted HTTP client inside the inference sandbox; provider transfer/exec latency excluded"
                               if callable(getattr(transport, "request_batch", None)) else "local client wall clock")}
    return traces, timing, ledger.summary()


def score(samples, traces):
    """tau = generated tokens / verification steps over the whole set; score = (tau - TAU_FLOOR) / (TAU_CEILING - TAU_FLOOR), clamped to [0, 1]."""
    if not samples:
        raise VerificationFailure("Nothing to score")
    tokens = steps = 0
    per_category, records, histogram = {}, [], [0] * (DRAFT_TOKENS + 1)
    for sample in samples:
        trace = traces[sample["id"]]
        length = len(sample["reference"])
        if trace["position"] < length or trace["steps"] != len(trace["accepted"]) or trace["steps"] < 1:
            raise VerificationFailure(f"Incomplete walk for {sample['id']!r}")
        tokens += length
        steps += trace["steps"]
        for accepted in trace["accepted"]:
            histogram[accepted] += 1
        category = per_category.setdefault(sample.get("category", "all"), [0, 0])
        category[0] += length
        category[1] += trace["steps"]
        records.append({"id": sample["id"], "category": sample.get("category"), "tokens": length,
                        "steps": trace["steps"], "tau": length / trace["steps"]})
    tau = tokens / steps
    value = min(1.0, max(0.0, (tau - TAU_FLOOR) / (TAU_CEILING - TAU_FLOOR)))
    return {"score": value,
            "details": {"metric": "mean_acceptance_length", "tau": tau, "tokens": tokens, "steps": steps,
                        "samples": len(samples), "draft_tokens": DRAFT_TOKENS,
                        "tau_by_category": {name: t / s for name, (t, s) in sorted(per_category.items())},
                        "tau_macro_over_prompts": sum(r["tau"] for r in records) / len(records),
                        "accepted_histogram": histogram, "records": records}}
