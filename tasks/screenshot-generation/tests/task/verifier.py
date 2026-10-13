"""Trusted verifier: request, parsing and metric over the held-out pages, with every gate enforced here:

* the snapshot-bound LLM review passed (review.py POLICY: training-data provenance, no reference-image lookup,
  actual inference path); the receipt is required before any GPU is provisioned;
* requests follow the shared failure policy the contract appends (frontier_ml.validation.evaluation): a
  transport failure is retried twice on the same payload inside the request's ceiling; a request that still
  fails, times out, or returns anything but one valid 1280x720 8-bit RGB PNG scores zero for its image alone,
  and the run fails closed when more than FAILED_REQUEST_PERCENT of the requests end that way;
* the 200 requests fit `inference_timeout_seconds` of server time in total and each fits
  `request_timeout_seconds` (120 s, retries included); both are disclosed in the contract and the public
  validate.py defaults to the same per-request ceiling. Server time is the per-request clock of the latency gate
  below, so the grading transport's own transfer/exec RPCs never count (SCHEMA.md); the verifier host's clock
  only guards the sandbox lifetime, and exhausting it is an EvaluationError, never a submission failure;
* p95 per-image latency with sequential requests <= 30 s, timed by the trusted HTTP client inside the
  serving sandbox when the transport offers one (`request_batch`; Harbor/Modal), so provider transfer/exec
  RPCs never count; local transports use this process's clock around loopback HTTP;
* text and layout by the pinned OCR on the verifier's GPU, faithfulness and realism by the operator's
  gpt-6-luna judge (FRONTIER_JUDGE_API_KEY, passed to the verifier as OPENAI_API_KEY); a judge that stays
  unavailable after bounded retries, or a malformed reply, is an EvaluationError (status evaluation_error,
  no score imputed), never a submission failure. Grading-transport faults propagate the same way.

selfcheck() is the agent's `frontier selfcheck`: generate() on public pages under their share of the budget, the
latency and failure gates, and every answer through the response check and the scorer's decode; no OCR, judge or review.
"""
from collections import defaultdict
import hashlib
import math
import time

from pathlib import Path

from frontier_ml.runtime.judge import judge_from_config
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import FailedRequests
import screenshot_gates as gates
from frontier_ml.taskspec import sibling

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
SLUG = "screenshot-generation"

DEFAULT_BUDGET_SECONDS = 7200  # task.yaml inference_timeout_seconds: server time over the whole evaluation subset
# Every Harbor/Modal exchange costs the grader upload, exec, response download and rm RPCs around the timed request
# (~2 s, tail ~4-7 s; docs/tasks/screenshot-generation.md). The sandbox lifetime (default_lifetime: budget + startup
# + 1,800 s) absorbs them, so a run whose host wall clock outlives the budget by more than this per-request allowance
# is stopped as an infrastructure fault before the sandbox can expire under a compliant server.
TRANSPORT_ALLOWANCE_SECONDS_PER_REQUEST = 5.0


def select_samples(samples, count):
    """Fixed subset, proportional by screen type; preserve rare types when space permits."""
    if type(count) is not int or not 1 <= count <= len(samples):
        raise VerificationFailure(f"evaluation_samples must be an integer from 1 to {len(samples)}")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise VerificationFailure("Evaluation sample IDs must be unique")
    groups = defaultdict(list)
    for sample in samples:
        groups[sample.get("screen_type", "other")].append(sample)
    minimum = int(count >= len(groups))
    quotas = {kind: max(minimum, count * len(rows) // len(samples)) for kind, rows in groups.items()}
    # Largest deficits/remainders, with stable ties and at least one of each type when possible.
    while sum(quotas.values()) != count:
        adding = sum(quotas.values()) < count
        eligible = [kind for kind in groups
                    if (quotas[kind] < len(groups[kind]) if adding else quotas[kind] > minimum)]
        kind = min(eligible, key=lambda kind: (
            (quotas[kind] * len(samples) - count * len(groups[kind])) * (1 if adding else -1), kind))
        quotas[kind] += 1 if adding else -1

    def rank(sample):
        digest = hashlib.sha256(("screenshot-evaluation-v1:" + sample["id"]).encode()).digest()
        return digest, sample["id"]

    selected = [sample for kind, rows in groups.items() for sample in sorted(rows, key=rank)[:quotas[kind]]]
    return sorted(selected, key=rank)


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    # The judge credential must exist before a GPU is provisioned.
    config["judge"] = judge_from_config(config)
    gates.preflight_scorer(config, data_path)


def generation_budget(config):
    return float(config.get("inference_timeout_seconds", DEFAULT_BUDGET_SECONDS))


def generate(transport, samples, config, *, latencies=None, failures=None):
    """Sequential requests through the public template. Returns (pngs, latencies, failed_requests summary).

    A request that fails after the shared retries, times out, or returns an invalid image yields None for its
    image, which score_images scores 0 for that sample alone; its seconds still count for the latency gate and
    the budget. More than FAILED_REQUEST_PERCENT of the requests failing that way fails the run (FailedRequests).
    The generation budget (`inference_timeout_seconds`) is charged with the seconds public.request_image measured
    for each request, retries included: the trusted client inside the serving sandbox when the transport offers
    it, so the grading transport's own transfer/exec RPCs never count (SCHEMA.md), else this process's clock
    around loopback HTTP. A request may use the smaller of `request_timeout_seconds` and what is left of the
    budget. This process's clock only guards the sandbox lifetime: a run that outlives the budget by more than the
    transport allowance stops as an EvaluationError (the grader's fault, no score imputed), never as a submission
    failure. `latencies` (a list) and `failures` (a FailedRequests for these samples), when given, are filled in
    place, so the self-check keeps what a run that stopped part-way measured."""
    from frontier_ml.validation.progress import progress
    budget = generation_budget(config)
    per_request = float(config.get("request_timeout_seconds", public.REQUEST_TIMEOUT_SECONDS))
    wall_deadline = time.monotonic() + budget + TRANSPORT_ALLOWANCE_SECONDS_PER_REQUEST * len(samples)
    failures = FailedRequests(len(samples)) if failures is None else failures
    latencies = [] if latencies is None else latencies
    pngs, spent = [], 0.0
    progress(config, "image_generation", completed=0, total=len(samples))
    for sample in samples:
        if time.monotonic() > wall_deadline:
            raise EvaluationError(f"Grading transport overhead exhausted the run's wall-clock allowance after {len(pngs)} of "
                                  f"{len(samples)} requests; the server itself used {spent:g}s of its {budget:g}s budget. "
                                  "No score imputed")
        remaining = budget - spent
        if remaining <= 0:
            raise VerificationFailure(f"Generation exceeded the {budget:g}s budget of server time; score is zero")
        png, failure, elapsed = public.request_image(transport, sample, min(per_request, remaining), failures=failures)
        latencies.append(elapsed)
        spent += elapsed
        if spent > budget:
            raise VerificationFailure(f"Generation exceeded the {budget:g}s budget of server time; score is zero")
        if failure is not None:
            failures.record(sample["id"], failure)  # this image scores zero; past the limit the run fails closed
        pngs.append(png)
        progress(config, "image_generation", completed=len(pngs), total=len(samples))
    return pngs, latencies, failures.summary()


def evaluate(transport, data_path, config):
    config = dict(config)
    config.pop("limit", None)
    samples = public.load_checked_samples(data_path, config)
    expected = config.get("expected_samples")
    if expected is not None and len(samples) != expected:
        raise VerificationFailure(f"Expected {expected} evaluation samples; found {len(samples)}")
    available_samples = len(samples)
    if "evaluation_samples" in config:
        samples = select_samples(samples, config["evaluation_samples"])
    pngs, latencies, failed_requests = generate(transport, samples, config)
    scorer = public.scorer_config(config)
    scorer["components"] = list(public.sg.COMPONENTS)
    try:
        scored = public.sg.score_images(samples, pngs, scorer)  # a None image scores 0 for its sample
    except public.sg.ScorerUnavailable as error:  # the scorer's own fault (pinned models, judge): never the submission's
        raise EvaluationError(f"Scorer unavailable; no score imputed: {error}") from error
    if not scored["complete_metric"]:
        raise VerificationFailure("The final metric needs all of T, L, F and R")
    results = {"latency": {**gates.latency_gate(latencies), "timing_scope": public.timing_scope(transport),
                           "request_timeout_seconds": float(config.get("request_timeout_seconds", public.REQUEST_TIMEOUT_SECONDS)),
                           "generation_seconds": sum(latencies), "generation_budget_seconds": generation_budget(config)}}
    failed = [name for name, gate in results.items() if not gate["passed"]]
    return {"score": 0.0 if failed else scored["score"],
            "details": {"metric": public.METRIC, "samples": len(samples), "available_samples": available_samples,
                        "means": scored["means"], "judge_model": scored["judge_model"],
                        "ungated_score": scored["score"], "gates": results,
                        "failed_requests": failed_requests, "invalid_images": scored["invalid_samples"],
                        "failed_gates": failed, "records": scored["records"]}}


def verify(server, data_path, config):
    review = verify_review(server["snapshot"], config["architecture_review_path"], SLUG, REVIEW_CHECKS)
    result = evaluate(server["transport"], data_path, config)
    result["details"]["gates"]["architecture"] = {"reviewer": review.get("reviewer"), "checks": review.get("checks")}
    return result


# Self-check (frontier_ml/selfcheck.py): the public request rows only; their reference screenshots are never sent.
SELFCHECK_PUBLIC = ("validation.jsonl",)
# The shortest sequential series whose p95 and 5% failure limit tolerate the same share of slow or failed requests as
# grading's 200 do: one in 20 (p95 of 200 allows ten). It gets their 720 s share of the budget, against which the
# budget gate is projected; task.yaml selfcheck_timeout_seconds covers that share with the transport allowance.
SELFCHECK_REQUESTS = 20
FAILURE_GATE = "failed requests"
BUDGET_GATE = "generation budget (projected)"
LATENCY_GATE = "p95 latency"
IMAGES = "valid images"


def _selfcheck_samples(public_dir, config):
    """SELFCHECK_REQUESTS public pages: grading's stratified selection when the public split has enough, else the
    split cycled under unique IDs. The public split is the operator's data, so a problem with it is never the
    submission's."""
    try:
        pages = public.load_checked_samples(Path(public_dir) / "validation.jsonl", {})
        count = min(SELFCHECK_REQUESTS, config.get("evaluation_samples") or SELFCHECK_REQUESTS)
        if len(pages) >= count:
            return select_samples(pages, count)
        return [{**pages[index % len(pages)], "id": f"{pages[index % len(pages)]['id']}~{index}"}
                for index in range(count)]
    except (OSError, ValueError, KeyError, TypeError, VerificationFailure) as error:
        raise EvaluationError(f"The public pages could not be prepared: {type(error).__name__}: {error}") from error


def _slow(requests):
    """How many requests may exceed the limit before p95 does (sg.percentile's rank)."""
    return requests - math.ceil(95 / 100 * requests)


def _selfcheck_failures(failed, sent):
    if not failed["total"]:
        return f"none of the {sent} requests failed, timed out or answered anything but one 1280x720 8-bit RGB PNG"
    reasons = ", ".join(f"{reason}: {count}" for reason, count in failed["reasons"].items())
    return (f"{failed['total']} of {sent} requests failed after the retries, timed out or answered anything but one "
            f"1280x720 8-bit RGB PNG ({reasons}); grading scores each such image zero, and the whole run zero past "
            f"{failed['limit_percent']}% of its requests. First errors: {failed['errors']}")


def _selfcheck_partial_latency(check, latencies, total, series, clock):
    """p95 of a series that stopped part-way. The requests it never sent can only raise the latency grading's
    percentile rank picks, so the same rank counted from the slowest finished request is a lower bound."""
    allowed = _slow(total)
    if len(latencies) <= allowed:
        return
    required = public.sg.LATENCY_P95_SECONDS
    bound = sorted(latencies)[-allowed - 1]
    check.record(LATENCY_GATE, bound <= required, measured=bound, required=required, unit="s", higher_is_better=False,
                 detail=f"at least this: {len(latencies)} of the {series} finished, "
                        f"{sum(seconds > required for seconds in latencies)} of them above {required:g} s, and p95 of "
                        f"{total} allows {allowed}; clock: {clock}")


def _selfcheck_series(check, server, samples, config):
    """generate() as grading runs it, on the public pages under their share of the server-time budget, then grading's
    run gates (failed requests, budget, p95) and every image's validity. A run that stops part-way records the gate
    that stopped it, where grading stops too, and the p95 its finished requests already imply."""
    total = len(samples)
    graded = config.get("evaluation_samples") or config.get("expected_samples") or total
    budget = generation_budget(config)
    share = budget * total / graded
    check.note(f"The speed gates were measured on a sample: {total} sequential public requests against their "
               f"{share:,g} s share of grading's {budget:,g} s for {graded}, the budget projected to all {graded} and "
               f"p95 and the failure limit taken over the {total}.")
    latencies, failures, pngs, stopped = [], FailedRequests(total), [], None
    try:
        pngs, _, _ = generate(server["transport"], samples, dict(config, inference_timeout_seconds=share),
                              latencies=latencies, failures=failures)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - failure limit, budget, or an answer the grader's check breaks on
        kind = type(error)
        stopped = f"{'' if kind.__module__ == 'builtins' else kind.__module__ + '.'}{kind.__name__}: {error}"
    spent, done, failed = sum(latencies), len(latencies), failures.summary()
    series = f"{total} sequential public requests, the first as soon as /health answered"
    clock = public.timing_scope(server["transport"])
    if stopped is not None and failed["total"] * 100 > failed["limit_percent"] * total:
        check.record(FAILURE_GATE, False, measured=round(100 * failed["total"] / total, 1),
                     required=failed["limit_percent"], unit="%", higher_is_better=False,
                     detail=f"stopped after {done} of the {series}: {_selfcheck_failures(failed, done)}")
        _selfcheck_partial_latency(check, latencies, total, series, clock)
        return
    if stopped is not None and spent >= share:
        # The last request was cut off at what was left of the share, so the per-request rate is a lower bound.
        rate = spent / max(done, 1)
        check.record(BUDGET_GATE, False, measured=round(rate * graded, 1), required=budget, unit="s",
                     higher_is_better=False,
                     detail=f"projected from {done} of {graded} requests at concurrency 1; stopped after {done} of the "
                            f"{series}: they took {spent:.1f} s of server time (slowest "
                            f"{max(latencies, default=0.0):.1f} s), all of their {share:,g} s share of grading's "
                            f"{budget:,g} s for {graded}; at {rate:.1f} s or more a request grading's {graded} need at "
                            f"least the projection; {_selfcheck_failures(failed, done)}")
        _selfcheck_partial_latency(check, latencies, total, series, clock)
        return
    if stopped is not None:  # e.g. a PNG header too short to unpack: grading's run ends on it as well
        check.fail(IMAGES, f"the answer to request {done + 1} of {total} broke the grader's response check "
                           f"({stopped}); grading fails the whole run on such an answer", kind="smoke")
    latency = gates.latency_gate(latencies)
    projected = spent * graded / total
    # Verdicts on the unrounded values, as grading compares them; only the reported measurement is rounded.
    check.record(FAILURE_GATE, failed["total"] * 100 <= failed["limit_percent"] * total,
                 measured=round(100 * failed["total"] / total, 1), required=failed["limit_percent"], unit="%",
                 higher_is_better=False, detail=_selfcheck_failures(failed, total))
    check.record(BUDGET_GATE, projected <= budget, measured=round(projected, 1), required=budget, unit="s",
                 higher_is_better=False,
                 detail=f"projected from {total} of {graded} requests at concurrency 1: they took {spent:.1f} s of "
                        f"server time; at that rate grading's {graded} take {projected:,.0f} s of the {budget:,g} s it "
                        f"allows")
    check.record(LATENCY_GATE, latency["passed"], measured=latency["p95_seconds"],
                 required=latency["required_p95_seconds"], unit="s", higher_is_better=False,
                 detail=f"{series}: mean {latency['mean_seconds']:.1f} s, max {latency['max_seconds']:.1f} s; "
                        f"p95 of {total} allows {_slow(total)} above {latency['required_p95_seconds']:g} s as "
                        f"grading's p95 of {graded} allows {_slow(graded)}; clock: {clock}")
    corrupt = []
    for sample, png in zip(samples, pngs):
        if png is not None:
            try:
                public.sg.load_png(png)
            except Exception as error:  # noqa: BLE001 - the scorer's decode: a corrupt PNG body scores zero
                corrupt.append(f"{sample['id']}: {type(error).__name__}: {error}")
    invalid = failed["total"] + len(corrupt)
    problems = [f"{FAILURE_GATE}: {failed['total']}"] if failed["total"] else []
    if corrupt:
        problems.append(f"PNG bodies the scorer cannot decode: {len(corrupt)} ({corrupt[:3]})")
    check.record(IMAGES, not invalid, kind="smoke", measured=total - invalid, required=total, unit="valid images",
                 detail=(f"{invalid} of {total} images would score zero: " + "; ".join(problems)) if invalid
                 else f"all {total} answers were valid 1280x720 RGB PNGs the scorer decodes")


def selfcheck(check, public_dir, config):
    """Grading's request loop on SELFCHECK_REQUESTS public pages, each answer through the grader's response check (with
    its retries) and the scorer's PNG decode. The first request goes as soon as /health answers, which the contract
    allows; grading hashes the snapshot for the review receipt first, so a server still warming up gets that pause
    there and none here (a cold first request counts). OCR and the judge never run, so nothing is scored. The hook adds
    no static check: selfcheck.run already checks the review's free prerequisites, there is no prescreen, and preflight
    checks only the review receipt, the judge credential and the verifier's own OCR install."""
    samples = _selfcheck_samples(public_dir, config)
    with check.launch() as server:
        check.stage("gates")
        _selfcheck_series(check, server, samples, config)
