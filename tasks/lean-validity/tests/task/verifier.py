"""Trusted verifier: every lean-validity gate, reusing validate.py's request, parsing and metric.

Gate order (all fail closed):
1. Content inspection for compiled Lean artifacts and runtime symbols. Filenames,
   example counts and uninspectable archives are not automatic rule violations.
2. The LLM review receipt (review.py POLICY) bound to the same snapshot digest.
3. The held-out batch in shuffled order, completed within BATCH_BUDGET_SECONDS end to end
   (whole-batch deadline; queued requests are cancelled on expiry and the score is zero).

selfcheck: gate 1, then one public attempt and a held-out-sized batch of distinct copies of the public attempts
through the same gated path and deadline, without the review or a score.
"""
import json
from pathlib import Path
import random
import re
import tempfile
import time

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import FAILED_REQUEST_PERCENT, load_samples, run_http_evaluation

public = sibling(__file__, "validate.py", require=("build_request", "parse_response", "score", "failed_prediction"))
import lean_scan as scanner
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
TASK = "lean-validity"
BATCH_BUDGET_SECONDS = 600
SELFCHECK_PUBLIC = ("validation.jsonl",)
SELFCHECK_SAMPLING = "lean-validity-selfcheck-v1"


def default_config():
    return {"batch_budget_seconds": BATCH_BUDGET_SECONDS}


def prescreen(config, data_path):
    snapshot = config.get("submission_path")
    if not snapshot or not Path(snapshot).is_dir():
        raise VerificationFailure("Trusted submission_path is required for the no-Lean scan")
    result = scanner.scan_submission(snapshot)
    if not result["passed"]:
        raise VerificationFailure("No-Lean scan failed: " + "; ".join(result["violations"][:10]))


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    prescreen(config, data_path)


def run_gated(transport, data_path, config, score=None):
    """The public request/parse/metric under the whole-batch deadline (never diagnostic mode). The self-check sends
    its batch through here too, with a `score` that scores nothing."""
    config = dict(config, validation_mode=False, predictions_only=False)
    count = len(load_samples(data_path))
    return run_http_evaluation(transport, data_path, config, build_request=public.build_request,
                               parse_response=public.parse_response, score=score or public.score,
                               failed_prediction=public.failed_prediction,
                               minimum_per_second=count / BATCH_BUDGET_SECONDS, default_concurrency=64)


def verify(server, data_path, config):
    scan = scanner.scan_submission(server["snapshot"])
    if not scan["passed"]:
        return {"score": 0.0, "details": {"failure": "no_lean_scan", "scan": scan}}
    review = verify_review(server["snapshot"], config["architecture_review_path"], TASK, REVIEW_CHECKS)
    if config.get("batch_budget_seconds", BATCH_BUDGET_SECONDS) != BATCH_BUDGET_SECONDS:
        raise VerificationFailure("The 600 s batch budget is fixed by the task")
    result = run_gated(server["transport"], data_path, config)
    timing = result["details"]["timing"][0]
    if not timing.get("throughput_gate_enforced") or timing["elapsed_seconds"] > BATCH_BUDGET_SECONDS:
        return {"score": 0.0, "details": {"failure": "throughput", "timing": timing}}
    result["details"]["gates"] = {
        "no_lean_scan": {k: scan[k] for k in ("passed", "files_scanned", "archives_scanned", "lean_source_files", "uninspected")},
        "architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")},
        "batch_seconds": {"elapsed": timing["elapsed_seconds"], "budget": BATCH_BUDGET_SECONDS},
    }
    return result


def resampled(rows, count):
    """`count` public attempts for the batch gate: shuffled passes over every attempt, so file lengths keep the public
    mix, each copy made distinct (`distinct`); IDs stay unique, as the evaluator requires."""
    if type(count) is not int or count < 1:
        raise EvaluationError("expected_samples must be a positive integer")
    rng, picked = random.Random(SELFCHECK_SAMPLING), []
    while len(picked) < count:
        deck = list(rows)
        rng.shuffle(deck)
        picked.extend(deck)
    return [dict(row, id=f"{row['id']}#{index}", lean=distinct(row["lean"], index))
            for index, row in enumerate(picked[:count])]


def distinct(lean, index):
    """`lean` with a numbered block comment (whitespace to Lean) right after its first `:= by`. The held-out files are
    all distinct, so verbatim repeats would let a prefix or response cache skip most of the work grading pays for;
    attempts on one theorem still share the header and statement up to there, as held-out attempts do."""
    at = lean.find(":= by")
    if at < 0:
        return f"/- {index} -/\n{lean}"
    at += len(":= by")
    return f"{lean[:at]} /- {index} -/{lean[at:]}"


def ignore_scores(samples, predictions, config):
    return {"score": None, "details": {}}


def selfcheck(check, public_dir, config):
    """Self-check (frontier_ml/selfcheck.py), after its review prerequisites: the no-Lean scan, one instance launched
    as grading launches it, the first public attempt through the grader's parser, then as many distinct copies of
    public attempts as the held-out split through run_gated at the task concurrency (3540 in 600 s: the full batch).
    Grading sends its batch to a server that has just answered /health, so the first request's time, retries
    included, counts against the same 600 s: the batch gets what it left, and the gate adds both. Nothing is scored
    and the review does not run."""
    try:
        rows = load_samples(Path(public_dir) / "validation.jsonl")
    except (OSError, ValueError, KeyError, VerificationFailure) as error:
        raise EvaluationError(f"The public attempts are unavailable: {error}") from error
    count = config.get("expected_samples") or public.HELDOUT_ATTEMPTS
    batch = resampled(rows, count)
    check.static("no-Lean scan", prescreen, config, None)  # needs neither the receipt nor the held-out data
    with check.launch() as server:
        path, payload = public.build_request(rows[0], Path(public_dir))
        started = time.monotonic()
        # As long as grading's cold batch would wait on it, retried as the batch retries it; what it takes comes out
        # of the batch's 600 s.
        check.request(server, path, payload, parse=public.parse_response, sample=rows[0],
                      timeout=BATCH_BUDGET_SECONDS)
        first = time.monotonic() - started
        check.stage("gates")
        remaining = BATCH_BUDGET_SECONDS - first
        if remaining <= 0:
            check.fail("batch deadline", f"the first request alone took {first:.1f} s of the {BATCH_BUDGET_SECONDS} s "
                       "grading allows for the whole held-out batch", measured=round(first, 1),
                       required=BATCH_BUDGET_SECONDS, unit="s", higher_is_better=False)
        concurrency = config.get("concurrency", 64)
        sent = f"{count} distinct copies of the {len(rows)} public attempts at concurrency {concurrency}"
        # The batch transport times the batch inside the sandbox and turns any overrun into "Batch exceeded", so a
        # failure-limit error from it always follows a batch that finished in time.
        timed_inside = callable(getattr(server["transport"], "request_batch", None))
        with tempfile.TemporaryDirectory(prefix="lean-validity-selfcheck-") as directory:
            data = Path(directory) / "batch.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in batch))
            options = dict(config, expected_samples=count, inference_timeout_seconds=remaining)
            batch_started = time.monotonic()
            try:
                result = run_gated(server["transport"], data, options, score=ignore_scores)
            except EvaluationError:
                raise
            except RecursionError:  # the batch transport's JSON decoder on a deeply nested answer
                check.fail("failed requests", "a batch response is nested too deeply to decode as JSON; grading "
                           "fails the run on it", kind="smoke")
            except VerificationFailure as error:
                limit = re.match(r"\d+ of \d+ requests failed", str(error))
                # Sent one by one (no request_batch), requests still open at the deadline time out and can trip the
                # failure limit first: that is the deadline. The evaluator stops counting at the limit, so the share
                # it quotes is a lower bound, not a measurement.
                if limit and (timed_inside or time.monotonic() - batch_started < remaining):
                    check.fail("failed requests", f"grading stops counting at its {FAILED_REQUEST_PERCENT}% limit and "
                               f"fails the run; at least {error}", required=FAILED_REQUEST_PERCENT, unit="%")
                expired = str(error).startswith("Batch exceeded")
                check.fail("batch deadline", f"{sent} did not finish in the {remaining:.1f} s left of grading's "
                           f"{BATCH_BUDGET_SECONDS} s after {first:.1f} s for the first request, so under "
                           f"{count / BATCH_BUDGET_SECONDS:.2f} attempts/s" + ("" if expired else f" ({error})"),
                           required=BATCH_BUDGET_SECONDS, unit="s")
        timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
        elapsed = first + timing["elapsed_seconds"]
        check.gate("batch deadline", round(elapsed, 2), BATCH_BUDGET_SECONDS, higher_is_better=False, unit="s",
                   detail=f"{sent}: {timing['elapsed_seconds']:.1f} s plus {first:.1f} s for the first request, "
                          f"{count / elapsed:.1f} attempts/s; grading allows {BATCH_BUDGET_SECONDS} s for its {count} "
                          "held-out attempts")
        check.gate("failed requests", round(100 * failed["total"] / count, 2), FAILED_REQUEST_PERCENT,
                   higher_is_better=False, unit="%",
                   detail=f"{failed['total']} of {count} failed, timed out or were rejected after retries; each scores "
                          "as the wrong verdict" + (f": {failed['errors']}" if failed["errors"] else ""))
