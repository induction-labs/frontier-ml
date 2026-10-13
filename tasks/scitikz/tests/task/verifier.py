"""Trusted review, fixed full-benchmark request protocol and isolated paper metrics.

selfcheck: one public figure through the graded request, retries and parser, then the 611-figure deadline projected
from a sample of the public figures at the official concurrency, unscored."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

from frontier_ml.taskspec import sibling
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import (FAILED_REQUEST_PERCENT, FailedRequests, RequestFailure,
                                                attempt_request, load_samples)
from scitikz_render import MAX_ANSWER, extract_tex

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
# The self-check reads the public rows and the 50 figures they name (about 4 MB).
SELFCHECK_PUBLIC = ("validation.jsonl", "images/*.png")
# A self-check is a smoke test and the speed gate, not a test-split run: it times as many figures as fit this share of
# the deadline in whole waves (48 of 611, at most 566 s) and projects the whole batch.
SAMPLE_SECONDS = 600
# The self-check's own limit for the first figure, retries included; with the sample's deadline share, the health probe
# and the relay's overhead this stays within the default work budget (selfcheck.WORK_SECONDS).
SMOKE_TIMEOUT_SECONDS = 240


def official_config(config):
    return dict(config, validation_mode=False, predictions_only=False, prediction_file=None,
                expected_samples=611, concurrency=16, inference_timeout_seconds=7200,
                require_isolation=True, scorer_python="/opt/scitikz/venv/bin/python",
                scorer_models_dir="/opt/scitikz/models", scorer_device="cuda",
                tex_root="/opt/scitikz/texroot", tex_launcher="/opt/scitikz/sandbox")


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture review is required before provisioning")
    public.worker({}, official_config(config), check=True)


def verify(server, data_path, config):
    review = verify_review(server["snapshot"], config["architecture_review_path"], "scitikz", REVIEW_CHECKS)
    result = public.run(server["transport"], data_path, official_config(config))
    result["details"]["gates"] = {"architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")}}
    return result


def _sample_size(count, deadline, concurrency):
    """How many figures the self-check times: whole waves at the grading concurrency, as many as the graded rate
    answers in SAMPLE_SECONDS (all of them when the whole deadline fits), never fewer than two waves or eight figures.
    A partial last wave takes as long as a full one: 50 figures run as 4 waves, which scaled by 611/50 is 49, while
    grading runs its 611 as 39 (ceil(611/16), and the relay's chunks of 64 are whole waves too), so the projection
    would be 25% slow."""
    fits = int(count * SAMPLE_SECONDS // deadline)
    if fits >= count:
        return count
    return min(count, concurrency * max(2, -(-8 // concurrency), fits // concurrency))


def _closed_fence(answer):
    """extract_tex, deciding an opened ```latex fence that is never closed in linear time first: the lazy pattern
    backtracks quadratically over whitespace after such an opening (about 13 s for 64,000 characters). The result
    is the same, because a later ```latex would itself contain the closing ```."""
    if isinstance(answer, str) and len(answer) <= MAX_ANSWER:
        opened = re.search(r"```latex", answer, re.I)
        if not opened or answer.find("```", opened.end()) < 0:
            return None, "missing_closed_latex_fence"
    return extract_tex(answer)


def _selfcheck_batch(public_dir, directory, count):
    """The public rows, and `count` of them (cycled when there are fewer) under unique IDs as a dataset for
    validate.run, which reads figures only from below its JSONL file: the figures are copied next to it. Each public
    figure first goes through the grader's request builder, so a bad public file is an error here rather than the
    submission's failure later."""
    try:
        source = Path(public_dir) / "validation.jsonl"
        rows = load_samples(source)
        for row in rows:
            public.build_request(row, source.parent)
            target = directory / row["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.parent / row["image"], target)
        cycled = (rows[i % len(rows)] for i in range(count))
        batch = [{"id": f"{row['id']}#{i}", "image": row["image"]} for i, row in enumerate(cycled)]
        data = directory / "batch.jsonl"
        data.write_text("".join(json.dumps(row) + "\n" for row in batch))
    except (KeyError, TypeError, OSError, ValueError, VerificationFailure) as error:
        raise EvaluationError(f"The public figures could not be prepared: {type(error).__name__}: {error}") from error
    return rows, data


def _smoke(check, server, sample, public_dir, gate, deadline):
    """The first public figure under grading's retries and parser, capped by the deadline as a cold grading batch caps
    its first request, and by SMOKE_TIMEOUT_SECONDS; returns its time and the answer (None without one). A request
    that still fails or an answer the parser rejects is one failed figure in grading, scored as the empty answer: it
    is recorded and the deadline is still measured. A first figure that uses up the whole deadline ends the check at
    the gate; one that outlasts only the self-check's own limit ends it here."""
    from frontier_ml.selfcheck import excerpt
    path, payload = public.build_request(sample, public_dir)
    name, limit = f"POST {path}", min(SMOKE_TIMEOUT_SECONDS, deadline)
    retries = FailedRequests(1)
    started = time.monotonic()
    response = attempt_request(server["transport"], "POST", path, payload, timeout=limit, failures=retries)
    first = time.monotonic() - started
    prediction, problem = None, None
    timed_out = isinstance(response, RequestFailure) and response.reason == "timeout"
    if isinstance(response, RequestFailure):
        problem = f"{response.reason}: {response.error}"
    else:
        try:
            prediction = public.parse_response(sample, response)
        except EvaluationError:
            raise
        except Exception as error:  # noqa: BLE001 - the grader would reject this answer
            problem = f"the response was rejected: {type(error).__name__}: {error}; got {excerpt(response)}"
    if first >= deadline or (timed_out and limit >= deadline):
        check.fail(gate, f"the first figure alone used up the deadline: {first:.1f} s"
                   + (f" ({problem})" if problem else ""), measured=round(first, 1), required=deadline, unit="s",
                   higher_is_better=False)
    if timed_out:
        check.fail(name, f"{problem}; {limit:g} s is the self-check's own limit for the first figure, while grading "
                         f"charges a slow first figure against the {deadline:g} s deadline",
                   kind="smoke", measured=round(first, 1), unit="s")
    detail = (f"{problem}; grading scores such a figure as the empty answer and fails the run above "
              f"{FAILED_REQUEST_PERCENT}% failed figures" if problem else
              ("answered only on a retry; " if retries.retried else "") + excerpt(prediction))
    check.record(name, problem is None, kind="smoke", measured=round(first, 3), unit="s", detail=detail)
    return first, None if prediction is None else prediction["answer"]


def _fence(check, answer):
    """The scorer's first-fence rule on the smoke answer, recorded without ending the check so that the deadline is
    still measured. Nothing is compiled here."""
    source, failure = _closed_fence(answer)
    if failure == "response_size":
        detail = f"the answer has {len(answer):,} characters, over the {MAX_ANSWER:,} grading accepts"
    elif failure:
        detail = "the answer has no closed ```latex fence (```tex or an unclosed block do not count)"
    else:
        detail = f"the first closed ```latex fence holds {len(source):,} characters of LaTeX"
    check.record("closed latex fence", failure is None, kind="smoke",
                 detail=detail + ("" if failure is None else "; grading scores such a figure as not compiling"))


def selfcheck(check, public_dir, config):
    """The first public figure through the graded request, retries and parser, then the generation deadline on a sample
    of the batch (_sample_size; the public figures, cycled if the sample is larger) through validate.run at the
    official concurrency, request timeouts and retries, against its proportional share of the deadline, unscored. The
    review prerequisites selfcheck.run checks first are the only static checks: preflight checks the review receipt
    and the trusted scorer, not the submission.

    Grading sends its batch to a cold server right after /health, so the first figure's time counts against the
    deadline once: the sample gets (deadline - first) * sample / count, and the projection is the first figure plus
    the sample's time scaled to the whole batch. A server that loads its model on the first request pays for it here
    as it would there, up to the self-check's own SMOKE_TIMEOUT_SECONDS.
    """
    official = official_config(config)
    count, deadline, concurrency = (official[key] for key in ("expected_samples", "inference_timeout_seconds",
                                                               "concurrency"))
    timed = _sample_size(count, deadline, concurrency)
    scale, sampled = count / timed, timed < count
    gate = f"{count} figures in {deadline:g} s"
    head = f"projected from {timed} of {count} requests at concurrency {concurrency}"
    with tempfile.TemporaryDirectory(prefix="scitikz-selfcheck-") as directory:
        rows, data = _selfcheck_batch(public_dir, Path(directory), timed)
        shape = f"{timed} public figures" + (f" (the {len(rows)} examples cycled)" if timed > len(rows) else "")
        with check.launch() as server:
            first, answer = _smoke(check, server, rows[0], Path(public_dir), gate, deadline)
            if answer is not None:
                _fence(check, answer)
            check.stage("gates")
            share = (deadline - first) / scale
            if sampled:
                check.note(f"The {count}-figure deadline was projected from a sample: {timed} public figures at "
                           f"concurrency {concurrency} against their {share:,.0f} s share of it.")
            started = time.monotonic()
            try:
                result = public.run(server["transport"], data, dict(
                    official, validation_mode=True, predictions_only=True, expected_samples=timed,
                    inference_timeout_seconds=share))
            except EvaluationError:
                raise
            except Exception as error:  # noqa: BLE001 - late, failed or undecodable answers fail grading's run too
                elapsed = time.monotonic() - started
                if str(error).startswith("Batch exceeded") or elapsed >= share:
                    window = f"their {share:.1f} s share of the deadline" if sampled else f"the remaining {share:.1f} s"
                    check.fail(gate, (f"{head}, a lower bound: " if sampled else "")
                               + f"not all {timed} figures were answered within {window} after a {first:.1f} s first "
                                 f"figure ({error})",
                               measured=round(first + elapsed * scale, 1), required=deadline, unit="s",
                               higher_is_better=False)
                check.fail("batch responses", f"{type(error).__name__}: {error}", kind="smoke")
            timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
            elapsed = timing["elapsed_seconds"]
            check.gate(gate, first + elapsed * scale, deadline, higher_is_better=False, unit="s",
                       detail=(f"{head}: " if sampled else f"all {count} requests at concurrency {concurrency}: ")
                              + f"{shape} took {elapsed:.1f} s"
                              + (f" ({elapsed * scale:,.1f} s scaled to {count})" if sampled else "")
                              + f", plus {first:.1f} s for the first figure; held-out figures differ and may need "
                                "longer answers")
            fenced = sum(_closed_fence(prediction.get("answer"))[1] is None for prediction in result["predictions"])
            detail = (f"{failed['total']} of {timed} requests failed, timed out or answered in an undocumented shape "
                      f"{failed['reasons']}; grading scores each as the empty answer and fails the run above "
                      f"{failed['limit_percent']}%. First errors: {failed['errors']}"
                      if failed["total"] else f"all {timed} answers parsed")
            if failed["retried"]:
                detail += f"; {failed['retried']} answered only on a retry"
            check.record("batch responses", not failed["total"], kind="smoke",
                         detail=f"{detail}; {fenced} have a closed ```latex fence within {MAX_ANSWER:,} characters")
            check.note(f"Nothing was compiled: answers were checked for a closed ```latex fence within {MAX_ANSWER:,} "
                       "characters, not for whether pdfLaTeX renders them.")
