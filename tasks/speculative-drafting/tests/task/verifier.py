"""Trusted verifier: size gate, frozen data, the snapshot-bound LLM review (review.py POLICY: training-data provenance
and no continuation lookup), then the public replay under the trusted serving budget. selfcheck() is the agent's
`frontier selfcheck`: the size gate, one public request and windows of the public conversations walked the same way,
projected to the held-out serving time."""
import hashlib
import math
import os
from pathlib import Path
import re

from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import (FAILED_REQUEST_PERCENT, REQUEST_ATTEMPTS, RETRY_BACKOFF_SECONDS,
                                               RequestFailure, load_samples)
from frontier_ml.validation.progress import progress
from specdraft_walk import (DRAFT_TOKENS, REQUEST_TIMEOUT_SECONDS, build_request, parse_draft, send_round,
                            validate_sample, walk)

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
SLUG = "speculative-drafting"
CONCURRENCY = 32


def default_config():
    # None forces an explicit, calibrated operator value in task.yaml before any trial can grade.
    return {"max_submission_bytes": None, "serving_budget_seconds": None, "evaluation_sha256": None}


def _positive(config, name, kind=(int,)):
    value = config.get(name)
    if type(value) not in kind or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise VerificationFailure(f"Set a positive {name} before grading")
    return value


def submission_bytes(root):
    """Bytes of every regular file in the frozen snapshot; anything else is refused."""
    total, files = 0, 0
    root = Path(root)
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise VerificationFailure(f"Submission contains a non-regular file: {path.relative_to(root)}")
            total += path.stat().st_size
            files += 1
    return {"bytes": total, "files": files}


def size_gate(root, config):
    limit = _positive(config, "max_submission_bytes")
    measured = submission_bytes(root)
    measured["limit_bytes"] = limit
    measured["passed"] = measured["bytes"] <= limit
    if not measured["passed"]:
        raise VerificationFailure(f"Submission is {measured['bytes']:,} bytes, over the {limit:,}-byte limit; "
                                  "score is zero")
    return measured


def frozen_samples(data_path, config):
    expected_digest = config.get("evaluation_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise VerificationFailure("Set evaluation_sha256 (the frozen held-out file digest) before grading")
    with open(data_path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_digest:
        raise VerificationFailure("Evaluation data differs from the frozen held-out file")
    samples = load_samples(data_path)
    if config.get("expected_samples") != len(samples):
        raise VerificationFailure("expected_samples must match the frozen evaluation data")
    for sample in samples:
        validate_sample(sample)
    return samples


def preflight(config, data_path):
    """Everything that can fail before the H100 is provisioned, including the size limit on the snapshot."""
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    _positive(config, "serving_budget_seconds", (int, float))
    if config.get("concurrency", CONCURRENCY) != CONCURRENCY:
        raise VerificationFailure(f"The task fixes request concurrency at {CONCURRENCY}")
    frozen_samples(data_path, config)
    config["submission_size"] = size_gate(config["submission_path"], config)


def verify(server, data_path, config):
    size = size_gate(server["snapshot"], config)
    frozen_samples(data_path, config)
    review = verify_review(server["snapshot"], config["architecture_review_path"], SLUG, REVIEW_CHECKS)
    trusted = dict(config, validation_mode=False, concurrency=CONCURRENCY)
    result = public.run(server["transport"], data_path, trusted)
    result["details"]["gates"] = {"architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")},
                                  "submission_size": size,
                                  "serving_budget": {"seconds": result["details"]["timing"][0]["serving_seconds"],
                                                     "budget_seconds": config["serving_budget_seconds"]}}
    return result


# Self-check (frontier_ml/selfcheck.py): public inputs only, never the held-out file or the review receipt.
SELFCHECK_PUBLIC = ("validation.jsonl",)
SMOKE = "POST /draft"
SERVING_GATE = "held-out serving time (projected)"
FAILURE_GATE = "request failures"
# A self-check is a smoke test plus the speed gate, not a held-out-sized run: the serving gate walks a sample worth
# this much of the held-out budget and projects the full serving time from it.
SAMPLE_SECONDS = 600
# One smoke try: every attempt a round may make at the 60 s ceiling, the backoffs between them and 10 s of slack. The
# trusted client's deadline must exceed the ceiling, or it reports a slow answer as a missed batch deadline.
SMOKE_SECONDS = REQUEST_ATTEMPTS * REQUEST_TIMEOUT_SECONDS + sum(RETRY_BACKOFF_SECONDS) + 10
# The grading transport parses a body with json.loads, which raises RecursionError (not ValueError) on deep nesting;
# grade() then fails the run, so the self-check records a failure instead of its own error.
TOO_DEEP = "a response nested too deeply for the grader's JSON parser ends the evaluation with score zero"
# Measured by walking the public lengths against five copies of them (the held-out set's shape) at tau 1 to 17
# (tests/test_speculative_drafting_selfcheck.py).
ACCURACY = ("within about 10% of the held-out serving time: a few percent high for a server that answers requests one "
            "at a time, up to 10% low for one that batches the requests in flight")


def _selfcheck_inputs(directory, config):
    """The public conversations and the trusted budget; a problem here is the self-check's, never the submission's."""
    try:
        samples = load_samples(Path(directory) / "validation.jsonl")
        for sample in samples:
            validate_sample(sample)
        budget = _positive(config, "serving_budget_seconds", (int, float))
    except (OSError, ValueError, KeyError, TypeError, VerificationFailure) as error:
        raise EvaluationError(f"Self-check inputs are unusable: {type(error).__name__}: {error}") from error
    return samples, budget


def _sample(samples, budget):
    """A window of `length` continuation tokens from each of the first CONCURRENCY public conversations: the longest
    window whose tokens are worth at most SAMPLE_SECONDS of the held-out budget, and never under DRAFT_TOKENS + 2, so
    every conversation sends at least two requests. CONCURRENCY of them make every sampled round one full wave at the
    grader's concurrency, as six of a held-out round's seven are; all 40 would add a wave of 8 to every round and read
    about 1.5x high for a server that batches. The windows are spread from the first conversation's opening to the
    last one's end (the tokens before a window join its prompt), so their contexts and positions average as the
    held-out requests' do. None is repeated, since a cache across requests (allowed) would answer a repeat for free.
    Returns (samples, length)."""
    samples = samples[:CONCURRENCY]
    target = public.HELDOUT_TOKENS * SAMPLE_SECONDS / budget
    longest = max(len(sample["reference"]) for sample in samples)
    length = DRAFT_TOKENS + 2
    while length < longest and sum(min(len(sample["reference"]), length + 1) for sample in samples) <= target:
        length += 1
    walked = []
    for index, sample in enumerate(samples):
        reference = sample["reference"]
        start = max(len(reference) - length, 0) * index // max(len(samples) - 1, 1)
        walked.append(dict(sample, prompt_tokens=sample["prompt_tokens"] + reference[:start],
                           reference=reference[start:start + length]))
    return walked, length


def _smoke(check, server, sample):
    """POST /draft for the first public conversation, sent as a walk round sends it (retries for transport failures,
    the 60 s ceiling) and checked with parse_draft; returns the answering try's serving seconds. Grading scores a
    failed request as an empty draft and fails only past FAILED_REQUEST_PERCENT, so a failure ends the check only
    when a second try fails too."""
    from frontier_ml.selfcheck import excerpt
    payload, problems = build_request(sample, 0), []
    for _ in range(2):
        try:
            (response,), elapsed, retried = send_round(server["transport"], [payload], concurrency=1,
                                                       timeout_seconds=SMOKE_SECONDS)
        except EvaluationError:
            raise
        except RecursionError as error:
            check.fail(SMOKE, f"{TOO_DEEP} ({type(error).__name__}: {error})", kind="smoke")
        except VerificationFailure as error:  # the trusted client's deadline for the whole try
            problems.append(str(error))
            continue
        if isinstance(response, RequestFailure):
            problems.append(f"{response.reason}: {response.error}")
            continue
        try:
            parse_draft(response)
        except VerificationFailure as error:
            problems.append(f"the response was rejected: {error}; got {excerpt(response)}")
            continue
        detail = excerpt(response) + (f" on attempt {retried + 1}" if retried else "")
        if problems:
            detail = (f"the first try failed ({problems[0]}); the second answered {detail}. Grading scores such a "
                      f"request as an empty draft and fails past {FAILED_REQUEST_PERCENT}% of requests")
        check.record(SMOKE, True, kind="smoke", measured=round(elapsed, 3), unit="s", detail=detail)
        return elapsed
    check.fail(SMOKE, "two tries failed: " + " | ".join(problems), kind="smoke")


def selfcheck(check, public_dir, config):
    """Size gate, one public POST /draft through parse_draft, then windows of the public conversations (_sample)
    walked as verify() walks the held-out ones (rounds, 32 in flight, the 60 s request ceiling, the failure limit)
    under the serving budget scaled to their continuation tokens. The gate projects the held-out serving time as
    validate.py does, per continuation token, plus the smoke request once, since grading's first request is cold too.
    On the public data's shape the projection is within about 10% of the held-out time: a few percent high for a
    server that answers requests one at a time (each window ends on a clipped step), up to 10% low for one that
    batches the requests in flight (held-out rounds of 200 end on a partial wave)."""
    samples, budget = _selfcheck_inputs(public_dir, config)
    walked, length = _sample(samples, budget)
    tokens = sum(len(sample["reference"]) for sample in walked)
    share = budget * tokens / public.HELDOUT_TOKENS
    check.static("submission size", size_gate, check.snapshot, config)
    with check.launch() as server:
        warmup = _smoke(check, server, samples[0])
        check.stage("gates")
        rounds = 0

        def advanced(count, done, total):
            nonlocal rounds
            rounds = count
            progress(config, "gates", completed=done, total=total, message=f"round {count}")

        conversations = (f"the {len(walked)} public conversations" if len(walked) == len(samples) else
                         f"the first {len(walked)} of the {len(samples)} public conversations")
        if length < max(len(sample["reference"]) for sample in samples[:len(walked)]):
            conversations = f"{length}-token windows, spread from openings to ends, of {conversations}"
        replay = (f"Walking {conversations} as grading walks the held-out set (rounds of up to {CONCURRENCY} "
                  f"concurrent requests, at most {REQUEST_TIMEOUT_SECONDS} s per request)")
        check.note(f"The serving gate was projected from a sample: {conversations} ({tokens:,} of the held-out set's "
                   f"{public.HELDOUT_TOKENS:,} continuation tokens), walked under the budget scaled to them "
                   f"({share:.0f} s).")
        try:
            _, timing, failures = walk(server["transport"], walked, concurrency=CONCURRENCY,
                                       serving_budget_seconds=share, on_round=advanced)
        except EvaluationError:
            raise
        except RecursionError as error:
            check.fail(FAILURE_GATE, f"{replay}: {TOO_DEEP} ({type(error).__name__}: {error})",
                       required=FAILED_REQUEST_PERCENT, unit="%")
        except VerificationFailure as error:  # the walk's two verdicts by their fixed openings; others as in grading
            text = str(error)
            if re.match(r"\d+ of \d+ requests failed", text):
                check.fail(FAILURE_GATE, f"{replay}: {text}", required=FAILED_REQUEST_PERCENT, unit="%")
            if text.startswith(("Serving exceeded", "Batch exceeded")):  # the walk's budget, request_batch's deadline
                check.fail(SERVING_GATE, f"{replay} took more than {share:.1f} s of serving after {rounds} completed "
                           f"rounds: the {budget:,g} s held-out budget scaled to their {tokens:,} of "
                           f"{public.HELDOUT_TOKENS:,} continuation tokens, so validate.py's held-out projection is "
                           f"over the budget. That projection is {ACCURACY}; grading enforces the budget on the "
                           "held-out run itself", required=budget, unit="s")
            raise
        projected = warmup + public.heldout_projection(timing, tokens)["projected_heldout_serving_seconds"]
        requests = timing["requests"]
        check.gate(SERVING_GATE, round(projected, 1), budget, higher_is_better=False, unit="s",
                   detail=f"projected from {requests} of about {requests * public.HELDOUT_TOKENS / tokens:,.0f} "
                          f"held-out requests at concurrency {CONCURRENCY}: {timing['serving_seconds']:.1f} s of "
                          f"serving over {timing['rounds']} rounds for {tokens:,} public continuation tokens, scaled "
                          f"to the held-out set's {public.HELDOUT_TOKENS:,} as validate.py does, plus the smoke "
                          f"request's {warmup:.1f} s once; {ACCURACY}")
        detail = (f"{failures['total']} of {failures['requests']} requests failed, timed out or were malformed; each "
                  "scores as an empty draft")
        if failures["total"]:
            detail += f" ({failures['reasons']}; first: {failures['errors']})"
        check.gate(FAILURE_GATE, round(100 * failures["total"] / max(failures["requests"], 1), 2),
                   FAILED_REQUEST_PERCENT, higher_is_better=False, unit="%", detail=detail)
