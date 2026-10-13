"""Trusted verifier: architecture review bound to the submission digest, the operator's gpt-6-luna answer judge
(which decides every question), then the public template and metric.
selfcheck: one public question, then a sample of public questions at the evaluation's concurrency within the sample's
share of its deadline (150 of the 900 in 600 of the 3600 s); no review, no judge and no score."""
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

from frontier_ml.runtime.judge import judge_from_config
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import RequestFailure, attempt_request

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
# Self-check (frontier_ml/selfcheck.py): the public questions and their images (about 50 MB), nothing else.
# The pinned public folder keeps its images under media/; a re-prepared copy uses images/.
SELFCHECK_PUBLIC = ("validation.jsonl", "media/*", "images/*")
SMOKE_TIMEOUT_SECONDS = 300  # the self-check's own limit for one question; kernels compiled on first use fit in it
# The timed sample's share of the deadline: the self-check projects the hour from a sample, it does not rerun it.
SAMPLE_SECONDS = 600


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    # The answer judge is the only scorer; its credential must exist before a GPU is provisioned.
    config.setdefault("judge_model", public.JUDGE_MODEL)
    config["judge"] = judge_from_config(config)


def verify(server, data_path, config):
    if not callable(config.get("judge")):
        raise VerificationFailure("The trusted answer judge is configured in preflight")
    review = verify_review(server["snapshot"], config["architecture_review_path"], "maple-vlm", REVIEW_CHECKS)
    result = public.run(server["transport"], data_path, config)
    result["details"]["gates"] = {"architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")}}
    return result


def selfcheck(check, public_dir, config):
    """The public question with the most images through the graded template, retry policy and parser, then a sample
    of the other public questions (_sample_size, _timed_questions) through public.run in its diagnostic mode, at the
    graded concurrency, stopped where grading's deadline would pass (_deadline). The judge decides every answer in
    grading, so only the replies' shape is checked and nothing is scored. preflight checks only the review receipt and
    the judge credential, so the review prerequisites selfcheck.run checks first are the only static checks."""
    public_dir = Path(public_dir)
    rows = _public_questions(public_dir)
    count, deadline = config.get("expected_samples"), config.get("inference_timeout_seconds", 3600)
    concurrency = config.get("concurrency")
    if (type(count) is not int or count < 1 or type(deadline) not in (int, float) or deadline <= 0
            or type(concurrency) is not int or concurrency < 1):
        raise EvaluationError("expected_samples, inference_timeout_seconds and concurrency must be positive")
    sample = max(rows, key=lambda row: len(_image_refs(row)))  # the contract allows up to seven images
    others = [row for row in rows if row is not sample] or rows
    batch = _timed_questions(others, _sample_size(count, deadline, concurrency))
    with tempfile.TemporaryDirectory(prefix="maple-selfcheck-") as directory:
        data = _batch_file(batch, public_dir, Path(directory))
        with check.launch() as server:
            first = _smoke(check, server, sample, public_dir)
            check.stage("gates")
            _deadline(check, server, data, len(batch), count, deadline, first, config,
                      repeats=max(0, len(batch) - len(others)))


def _sample_size(count, deadline, concurrency):
    """The timed requests: all `count` when grading's deadline fits in SAMPLE_SECONDS, otherwise the requests that fill
    SAMPLE_SECONDS at the graded rate (150 of 900 for 3600 s); at least 8 and two waves at the graded concurrency."""
    return min(count, max(2 * concurrency, 8, math.floor(count * SAMPLE_SECONDS / deadline)))


def _timed_questions(others, timed):
    """`timed` questions: each of `others` once, in order, then repeats of them under new IDs (the evaluator requires
    unique ones) while the public split is smaller than the sample. Distinct questions come first because the held-out
    ones are all distinct, and a prefix, image-encoder or response cache answers a repeat almost for free."""
    batch = []
    for index in range(timed):
        row, repeat = others[index % len(others)], index // len(others)
        batch.append({**row, "id": f"{row['id']}#{repeat + 1}"} if repeat else row)
    if len({str(row["id"]) for row in batch}) != timed:
        raise EvaluationError("The resampled public questions repeat an ID")
    return batch


def _image_refs(row):
    images = row.get("images") or {}
    return [images[key] for key in sorted(images)] if isinstance(images, dict) else list(images)


def _public_questions(public_dir):
    """The public rows, each checked to build a graded request: a broken public copy is the self-check's fault."""
    try:
        rows = [json.loads(line) for line in (public_dir / "validation.jsonl").read_text().splitlines() if line.strip()]
        for row in rows:
            public.build_request(row, public_dir)
        ids = [str(row["id"]) for row in rows]
    except (OSError, ValueError, KeyError, TypeError, SyntaxError, VerificationFailure) as error:
        raise EvaluationError(f"The public questions are unusable: {type(error).__name__}: {error}") from error
    if not rows:
        raise EvaluationError("The public split has no questions")
    if len(set(ids)) != len(ids):  # the evaluator requires unique IDs
        raise EvaluationError("The public split repeats a question ID")
    return rows


def _batch_file(rows, public_dir, directory):
    """The timed questions beside copies of their images: references resolve inside the JSONL's own folder."""
    try:
        for reference in {reference for row in rows for reference in _image_refs(row)}:
            (directory / reference).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(public_dir / reference, directory / reference)
    except OSError as error:
        raise EvaluationError(f"The public images could not be staged: {error}") from error
    path = directory / "questions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _smoke(check, server, sample, public_dir):
    """One question under grading's retries and parser; returns its time. A request that still fails or a reply the
    parser rejects ends the self-check; an empty reply, one wrong answer in grading, is recorded and the deadline is
    still measured."""
    from frontier_ml.selfcheck import excerpt
    path, payload = public.build_request(sample, public_dir)
    name = f"POST {path}"
    started = time.monotonic()
    response = attempt_request(server["transport"], "POST", path, payload, timeout=SMOKE_TIMEOUT_SECONDS)
    first = time.monotonic() - started
    if isinstance(response, RequestFailure):
        limit = (f"; {SMOKE_TIMEOUT_SECONDS} s is the self-check's own limit for one question, while grading charges a "
                 f"slow first answer against the batch deadline" if response.reason == "timeout" else "")
        check.fail(name, f"{response.reason}: {response.error}{limit}", kind="smoke")
    try:
        prediction = public.parse_response(sample, response)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - the grader would reject this answer
        check.fail(name, f"the response was rejected: {type(error).__name__}: {error}; got {excerpt(response)}",
                   kind="smoke")
    empty = not prediction["answer"].strip()
    check.record(name, not empty, kind="smoke", measured=round(first, 3), unit="s",
                 detail=("the reply is empty (no text content), which is scored wrong without a judge call; got "
                         if empty else "") + excerpt(prediction))
    return first


def _deadline(check, server, data, timed, count, deadline, first, config, *, repeats=0):
    """public.run as grading calls it, in its diagnostic mode (no judge), on `timed` public questions. Grading sends its
    `count` questions to a cold server right after /health, so the projection is the first question's time once plus
    the batch's time scaled to `count`; the batch runs until that projection reaches the deadline."""
    scale = count / timed
    share = (deadline - first) / scale
    concurrency = config["concurrency"]
    shape = (f"projected from {timed} of {count} requests at concurrency {concurrency}" if timed < count
             else f"all {count} requests at concurrency {concurrency}")
    graded = f"grading allows {deadline:g} s for the {count} held-out questions"
    if share <= 0:
        check.fail("evaluation deadline", f"the first question alone took {first:.1f} s; {graded}",
                   measured=round(first, 1), required=deadline, unit="s", higher_is_better=False)
    notes = [f"The evaluation deadline was projected from a sample of {timed} of the {count} graded requests at "
             f"concurrency {concurrency}"] if timed < count else []
    if repeats:
        notes.append(f"{repeats} of the {timed} repeat public questions under new IDs (the public split is smaller), "
                     f"which a prompt cache may answer faster than held-out ones")
    if notes:
        check.note("; ".join(notes) + ".")
    options = {key: value for key, value in config.items() if key != "judge"}
    options.update(validation_mode=True, expected_samples=timed, inference_timeout_seconds=share)
    started = time.monotonic()
    try:
        result = public.run(server["transport"], data, options)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - grading counts anything but its own infrastructure against the server
        elapsed = time.monotonic() - started
        # Requests cut off at the batch's remaining time can trip the failure limit before the deadline itself.
        if str(error).startswith("Batch exceeded") or elapsed >= share:
            check.fail("evaluation deadline", f"{shape}: the {timed} public questions had not finished after "
                       f"{elapsed:.1f} s, their {share:.1f} s share of the deadline after a {first:.1f} s first "
                       f"question, so the measurement is a lower bound; {graded}",
                       measured=round(first + elapsed * scale, 1), required=deadline, unit="s",
                       higher_is_better=False)
        check.fail("batch responses", f"{type(error).__name__}: {error}", kind="smoke")
    elapsed, failed = result["details"]["timing"][0]["elapsed_seconds"], result["details"]["failed_requests"]
    projected = first + elapsed * scale
    check.gate("evaluation deadline", round(projected, 1), deadline, higher_is_better=False, unit="s",
               detail=f"{shape}: the {timed} public questions took {elapsed:.1f} s after a {first:.1f} s first "
                      f"question; at that rate the {count} held-out questions take {projected:,.0f} s (the first "
                      f"once, the rest scaled by {count}/{timed}); {graded}")
    empty = sum(1 for prediction in result["predictions"]
                if not prediction.get("request_failed") and not prediction["answer"].strip())
    problems = []
    if failed["total"]:
        problems.append(f"{failed['total']} of {timed} requests failed, timed out or returned malformed completions "
                        f"({', '.join(f'{reason}: {n}' for reason, n in failed['reasons'].items())}); grading scores "
                        f"each as a wrong answer, and the whole run zero past {failed['limit_percent']}%. "
                        f"First errors: {failed['errors']}")
    if empty:
        problems.append(f"{empty} of {timed} replies were empty, which are wrong without a judge call")
    check.record("batch responses", not problems, kind="smoke", measured=timed - failed["total"] - empty,
                 required=timed, unit="replies",
                 detail="; ".join(problems) or f"all {timed} replies parsed and non-empty")
