"""Trusted deadline + kernel-checked proof shortening, behind the snapshot-bound LLM review (review.py POLICY: proof
provenance and no proof lookup). Never execute submission code here."""
import itertools
import json
import math
from pathlib import Path
import re
import tempfile
import time

from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import FAILED_REQUEST_PERCENT, load_samples, timed_batch
from frontier_ml.validation.progress import progress
from simplification_local import LocalChecker
from simplification_metric import MAX_STATEMENT_BYTES, proof_length, proof_policy

public = sibling(__file__, "validate.py", require=("collect_predictions", "score"))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
SLUG = "lean-simplification"
SELFCHECK_PUBLIC = ("validation.jsonl",)
SELFCHECK_WINDOWS = 7  # all 55 public proofs (one twice), against grading's 45 held-out windows: at most 7 minutes
# Grading fails the run past 5% of its 354 requests, a rate 56 requests estimate only roughly. Past 10% (6 or more
# of 56) is clear evidence of more than 5%: a server failing 5% gets there about one run in 17, one failing 3% under
# one in 100.
SELFCHECK_FAILED_REQUEST_PERCENT = 10


def default_config():
    return {"batch_budget_seconds": 2700, "proof_timeout_seconds": 60, "preflight_timeout_seconds": 1800}


def validate_config(config):
    budget = config.get("batch_budget_seconds")
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise VerificationFailure("Set a positive finite batch_budget_seconds before provisioning")
    for name in ("proof_timeout_seconds", "preflight_timeout_seconds"):
        value = config.get(name, default_config()[name])
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise VerificationFailure(f"Set a positive finite {name}")
    if config.get("concurrency", 8) != 8:
        raise VerificationFailure("The task fixes request concurrency at 8")
    return budget


def validate_samples(samples, config):
    expected = config.get("expected_samples")
    if type(expected) is not int or expected < 1 or len(samples) != expected:
        raise VerificationFailure("Expected a frozen nonempty evaluation set")
    for row in samples:
        if not isinstance(row.get("statement"), str) or len(row["statement"].encode()) > MAX_STATEMENT_BYTES:
            raise VerificationFailure("Invalid theorem signature")
        if proof_policy(row.get("proof")) or proof_length(row["proof"]) < 1:
            raise VerificationFailure("Invalid input proof")


def require_review(config):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")


def reviewed(server, config):
    """The review receipt, bound to the served snapshot; a missing, rejected or mismatched receipt fails here."""
    review = verify_review(server["snapshot"], config["architecture_review_path"], SLUG, REVIEW_CHECKS)
    return {"reviewer": review.get("reviewer"), "checks": review.get("checks")}


def preflight(config, data_path):
    validate_config(config)
    require_review(config)
    samples = load_samples(data_path)
    validate_samples(samples, config)
    checker = LocalChecker()
    checker.selftest()
    progress(config, "source_proof_checking", completed=0, total=len(samples))
    verdicts, timing = timed_batch(samples, lambda row, timeout: checker(row, row["proof"]), concurrency=8,
                                  timeout_seconds=config.get("preflight_timeout_seconds", 1800),
                                  on_complete=lambda n: progress(config, "source_proof_checking", completed=n, total=len(samples)))
    if not all(v["valid"] for v in verdicts):
        failed = [(row["id"], verdict.get("status")) for row, verdict in zip(samples, verdicts, strict=True)
                  if not verdict["valid"]]
        raise VerificationFailure(f"Input proofs failed the pinned checker: {failed[:5]}")
    config["_lean_checker"] = checker
    config["_lean_preflight_seconds"] = timing["elapsed_seconds"]


def score_predictions(samples, predictions, timing, config, review=None):
    checker = config.get("_lean_checker")
    if checker is None:
        raise VerificationFailure("Local Lean checker preflight was not completed")
    trusted = dict(config, validation_mode=False, predictions_only=False, concurrency=8, proof_checker=checker)
    progress(config, "proof_checking", completed=0, total=len(samples))
    started = time.monotonic()
    result = public.score(samples, predictions, trusted)
    result["details"]["proof_checking_seconds"] = time.monotonic() - started
    result["details"]["preflight_seconds"] = config.get("_lean_preflight_seconds")
    result["details"]["failed_requests"] = timing.pop("failed_requests", None)
    result["details"]["timing"] = [timing]
    result["details"]["checker"] = checker.health
    result["details"]["batch_budget_seconds"] = config["batch_budget_seconds"]
    result["details"]["gates"] = {"architecture": review}
    progress(config, "proof_checking", completed=len(samples), total=len(samples))
    return result


def verify_isolated(launch, data_path, config):
    validate_config(config)
    if config.get("_lean_checker") is None:
        raise VerificationFailure("Local Lean checker preflight was not completed")
    trusted = dict(config, validation_mode=False, predictions_only=False, concurrency=8)
    with launch() as server:
        review = reviewed(server, config)
        samples, predictions, timing = public.collect_predictions(server["transport"], data_path, trusted)
    # The H100 is already stopped. Only proof text enters the local CPU checker.
    return score_predictions(samples, predictions, timing, config, review)


def verify(server, data_path, config):
    """Direct integration-test entry point; grading uses verify_isolated above."""
    validate_config(config)
    if config.get("_lean_checker") is None:
        raise VerificationFailure("Local Lean checker preflight was not completed")
    review = reviewed(server, config)
    samples, predictions, timing = public.collect_predictions(server["transport"], data_path,
        dict(config, validation_mode=False, predictions_only=False, concurrency=8))
    return score_predictions(samples, predictions, timing, config, review)


def selfcheck(check, public_dir, config):
    """Self-check (frontier_ml/selfcheck.py): one instance launched as grading launches it, then SELFCHECK_WINDOWS
    windows of 8 concurrent public proofs through grading's own loop (collect_predictions: its window, retries and
    response parser). As in grading, the first window is the first traffic after /health, so a server that warms up
    on its first requests pays for it there. No proof is checked by Lean or scored. Preflight has no static
    constraint on the submission (only the trusted config, the review receipt, the held-out inputs and the Lean
    checker), so beyond the review prerequisites selfcheck.run checks first there is none to run.

    Grading's 354 proofs take 45 windows, up to 2700 s; the sample's 7 windows take at most 420 s, inside the default
    work budget. Grading holds each window to its own 60 s, so the gate is the slowest sampled window against that,
    measured rather than projected."""
    try:
        validate_config(config)
        rows = load_samples(Path(public_dir) / "validation.jsonl")
    except VerificationFailure as error:  # the trusted config or the public examples, never the submission
        raise EvaluationError(f"Self-check setup failed: {error}") from error
    window = min(config.get("proof_timeout_seconds", 60), config.get("inference_timeout_seconds", 2700),
                 config["batch_budget_seconds"])
    count = SELFCHECK_WINDOWS * 8
    # Public rows in file order, cycled when there are fewer; the suffix keeps ids unique, as load_samples requires.
    batch = [dict(row, id=f"{row['id']}@{index}") for index, row in zip(range(count), itertools.cycle(rows))]
    trusted = dict(config, validation_mode=False, predictions_only=False, concurrency=8, expected_samples=count)
    graded = config.get("expected_samples")
    grading = f"grading fails the run past {FAILED_REQUEST_PERCENT}% of its held-out requests" + (
        f", {graded * FAILED_REQUEST_PERCENT // 100 + 1} or more of {graded}" if type(graded) is int else "")
    with tempfile.TemporaryDirectory(prefix="lean-selfcheck-") as directory:
        windows_path = Path(directory) / "windows.jsonl"
        windows_path.write_text("".join(json.dumps(row) + "\n" for row in batch))
        with check.launch() as server:
            check.stage("gates")
            check.note("No proof was checked by Lean: answers were checked only by grading's parser. validate.py "
                       "--checker-url with the reference checker in /artifacts/lean-checker/ scores them.")
            check.note(f"The batch gate was measured on a sample: up to {SELFCHECK_WINDOWS} windows of 8 public "
                       "proofs" + (f" against grading's {-(-graded // 8)} of held-out proofs" if type(graded) is int
                                   else "") + f", each window held to the same {window:g} s.")
            windows = []
            try:
                _, _, timing = public.collect_predictions(server["transport"], windows_path, trusted,
                                                          on_window=windows.append,
                                                          limit_percent=SELFCHECK_FAILED_REQUEST_PERCENT)
            except EvaluationError:
                raise
            except RecursionError:  # the batch transport's JSON decoder on a deeply nested answer
                check.fail("batch responses", f"a response in window {len(windows) + 1} is nested too deeply to "
                           "decode as JSON", kind="smoke")
            except VerificationFailure as error:
                where, text = f"window {len(windows) + 1} of {SELFCHECK_WINDOWS}", str(error)
                if text == "Invalid trusted HTTP batch result":  # the bridge's integrity check, not the server
                    raise EvaluationError(f"{where}: {text}") from error
                limit = re.match(r"(\d+) of \d+ requests failed", text)
                if limit:
                    check.fail("failed requests", f"{where}; the self-check fails past "
                               f"{SELFCHECK_FAILED_REQUEST_PERCENT}% of its {count} public requests, {grading}: {text}",
                               measured=round(100 * int(limit[1]) / count, 2), required=FAILED_REQUEST_PERCENT,
                               unit="%", higher_is_better=False)
                if "deadline" in text or "request window" in text:
                    cold = "" if windows else " It was the first traffic after /health, as grading's first window is."
                    check.fail("slowest batch of 8", f"{where}: {text}.{cold}", required=window, unit="s")
                check.fail("batch responses", f"{where}: {text}", kind="smoke")
            slowest = max(range(len(windows)), key=windows.__getitem__)
            check.gate("slowest batch of 8", windows[slowest], window, higher_is_better=False, unit="s",
                       detail=f"{len(windows)} windows of 8 concurrent public proofs, the first straight after "
                              f"startup as in grading: first {windows[0]:.1f} s, slowest window {slowest + 1}, mean "
                              f"{sum(windows) / len(windows):.1f} s, {timing['retried_requests']} retried; grading "
                              "holds every batch of the held-out split to the same window")
            failures = timing["failed_requests"]
            failed = failures["total"]
            # Within SELFCHECK_FAILED_REQUEST_PERCENT, which collect_predictions enforced: passed, with grading's
            # limit as the requirement, so a count above it shows as a negative margin.
            check.record("failed requests", True, measured=round(100 * failed / count, 2),
                         required=FAILED_REQUEST_PERCENT, unit="%", higher_is_better=False,
                         detail=f"{failed} of {count} failed, timed out or were rejected after retries; each scores as "
                                f"the unchanged proof, and {grading}. {count} public requests estimate that rate "
                                f"only roughly, so this check fails only past {SELFCHECK_FAILED_REQUEST_PERCENT}%"
                                + (f" and any failure is a risk: {failures['errors']}" if failed else ""))
