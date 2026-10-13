"""Self-check hook for svg-generation (frontier_ml/selfcheck.py): grading's static scan, one public drawing through
the graded request policy and parser, then the graded batch's speed gate on resampled public instructions, every
answer checked against the SVG schema. Grading's batch (254 requests within 10,800 s) is far longer than a self-check,
so the gate is projected from a sample at grading's concurrency held to its share of the deadline.

It lives beside verifier.py rather than in it because the judge calibration binds verifier.py's bytes
(calibration_binding): editing that file would make official scoring unavailable until a paid recalibration. task.yaml
names this module as verifier.selfcheck_module. Trusted and verifier-only: validate.py does not import it, so it never
reaches the agent image.
"""
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time

from frontier_ml.runtime.inference import serve_submission
from frontier_ml.selfcheck import excerpt
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import (_PARSE_ERRORS, FAILED_REQUEST_PERCENT, RECORDED_ERRORS, FailedRequests,
                                               RequestFailure, attempt_request, chat_answer, run_http_evaluation)
from svg_audit import inspect_submission
from svg_weights import inspect_weights
from svg_render import MAX_SVG_BYTES, InvalidSVG, render_svg, renderer_binary, sanitize_svg

public = sibling(__file__, "validate.py", require=("build_request", "parse_response", "failed_prediction"))
TRUSTED_SERVER = Path(__file__).resolve().parent / "trusted_server"
SELFCHECK_PUBLIC = ("validation.jsonl",)  # the 30 public examples, about 33 KB; the hook reads nothing else
CONCURRENCY = 8  # verifier.CONCURRENCY: verify() sends the held-out batch at exactly this many in flight
# A batch with a longer deadline is timed on a sample: a self-check is a smoke test and the speed gates, not a full
# inference run.
SAMPLE_SECONDS = 600
SMOKE = "POST /v1/chat/completions"
SAMPLING_VERSION = "svg-generation-selfcheck-v1"
# The graded parser's code-fence pattern (svg_render.FENCE, bound by the calibration) backtracks over every split of a
# fence's opening and over every run of blanks: quadratic in either, cubic in both. The regex holds the GIL, so
# neither the batch deadline nor the check's budget can interrupt it. fence_steps estimates its work in linear time;
# FENCE_STEPS is about 1.5 s of it (about 1.4e-8 s per step on the shapes measured).
FENCE_OPENING = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?")
BLANKS = re.compile(r"[ \t]{2,}")
FENCE_STEPS = 10 ** 8


class SlowFence(VerificationFailure):
    """An answer the graded parser's code-fence pattern would take seconds to hours on."""


def overdue(seconds):
    """A missed deadline's measurement, rounded up so it never reads as the deadline itself."""
    return math.ceil(seconds * 10) / 10


def sample_plan(count, deadline):
    """(requests sent, their deadline) for the timed batch: grading's whole batch when its deadline is at most
    SAMPLE_SECONDS, else two waves at grading's concurrency (or what SAMPLE_SECONDS holds at grading's rate, if more)
    with the same share of the deadline as of the requests."""
    if deadline <= SAMPLE_SECONDS:
        return count, deadline
    sent = min(count, max(2 * CONCURRENCY, 8, math.floor(count * SAMPLE_SECONDS / deadline)))
    return sent, deadline * sent / count


def fence_steps(answer):
    """About how many steps svg_render.FENCE takes on `answer`: a scan of the rest from every place its lazy body
    may start (only the first when a closing fence ends the answer), a run of n blanks adding about n²/16 to each."""
    source = answer.strip()
    if not source.startswith("```"):
        return 0
    starts = 1 if len(source) >= 6 and source.endswith("```") else FENCE_OPENING.match(source).end() - 2
    return starts * (len(source) + sum((blank.end() - blank.start()) ** 2 for blank in BLANKS.finditer(source)) // 16)


def parse(sample, response):
    """The graded parser, unless its code-fence pattern would stall on the answer (fence_steps)."""
    try:
        steps = fence_steps(chat_answer(response))
    except VerificationFailure:  # no text answer: the graded parser rejects it with its own message
        steps = 0
    if steps > FENCE_STEPS:
        raise SlowFence(f"grading's SVG extractor would backtrack for seconds or more on this answer ({steps:.1e} "
                        "steps, which no deadline interrupts): a ``` code fence with a long tag or blank run, or "
                        "never closed")
    return public.parse_response(sample, response)


def lenient(sample, response):
    """parse() for a sample, whose 5% failure limit allows no failure at all: a rejected answer becomes an empty
    drawing marked `rejected`, timed in the batch as grading times it and counted after the gate (unscored), so it
    cannot stop the batch before the speed gate is measured. Transport failures and timeouts still count."""
    try:
        return parse(sample, response)
    except EvaluationError:
        raise
    except _PARSE_ERRORS as error:
        return {"id": sample["id"], "svg": "", "rejected": f"{type(error).__name__}: {str(error)[:200]}"}


def weight_audit(root):
    """prescreen's weight audit (svg_weights.inspect_weights), failing with prescreen's message."""
    audit = inspect_weights(root)
    if not audit["passed"]:
        raise VerificationFailure("Weight audit failed: " + "; ".join(audit["problems"]))
    return audit


def trusted_launcher(snapshot, config):
    """The verifier's launch: the trusted gpt-oss-20b FP8 server on model/, never a submission serve.sh."""
    return serve_submission(snapshot, dict(config, trusted_serve_dir=str(TRUSTED_SERVER)))


def training_evidence(root):
    """preflight's training evidence scan (svg_audit.inspect_submission), failing with preflight's message."""
    try:
        scan = inspect_submission(root)
    except RecursionError:  # a manifest too deep for json, which fails preflight too; named, not a bare traceback
        scan = {"passed": False, "problems": ["training_manifest.json is nested too deeply"]}
    if not scan["passed"]:
        raise VerificationFailure("Training evidence scan failed: " + "; ".join(scan["problems"]))
    return scan


def public_rows(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise EvaluationError(f"The public examples could not be read: {error}") from error
    if not rows or any(not isinstance(row, dict) or "id" not in row or not isinstance(row.get("instruction"), str)
                       for row in rows):
        raise EvaluationError("The public examples are empty or lack id and instruction")
    return rows


def resampled(rows, count):
    """`count` public rows for the timed batch: shuffled passes over every example, IDs kept unique (the evaluator
    rejects duplicates)."""
    rng, picked = random.Random(SAMPLING_VERSION), []
    while len(picked) < count:
        deck = list(rows)
        rng.shuffle(deck)
        picked.extend(deck)
    return [dict(row, id=f"{row['id']}#{index}") for index, row in enumerate(picked[:count])]


def unscored(samples, predictions, config):
    """The batch is timed and parsed as graded; each answer then goes through the scorer's schema check
    (sanitize_svg, which render_svg runs first). Nothing is rasterized or judged; rendering is checked on the smoke
    answer."""
    invalid, rejected = Counter(), []
    for prediction in predictions:
        if prediction.get("request_failed"):  # failed_prediction: already counted in failed_requests
            continue
        if "rejected" in prediction:  # lenient(): the graded parser's rejection, which the hook counts
            rejected.append(f"{prediction['id']}: {prediction['rejected']}")
            continue
        try:
            sanitize_svg(prediction["svg"])
        except InvalidSVG as error:
            invalid[str(error)] += 1
        except Exception as error:  # noqa: BLE001 - the scorer choking on this answer would fail the graded run
            invalid[f"the scorer could not read it ({type(error).__name__})"] += 1
    return {"score": None, "details": {"invalid_svg": dict(invalid.most_common()), "rejected": rejected}}


def renderer(config):
    """The scorer's rsvg-convert, resolved before the GPU is held. A missing one is the verifier image's fault."""
    try:
        return renderer_binary(config.get("rsvg_binary"))
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        raise EvaluationError(f"The SVG renderer is unavailable: {type(error).__name__}: {error}") from error


def smoke(check, server, sample, public_dir, timeout):
    """The first public instruction under grading's request policy: transport failures retried (evaluation's
    attempt_request), the answer checked by the graded parser. Grading scores a request that still fails 0 and goes
    on, so a failure is recorded and the batch still runs. Returns the parsed answer, or None."""
    path, payload = public.build_request(sample, public_dir)
    retried = FailedRequests(1)  # counts an answer that needed a retry
    graded = (f". Grading scores such a drawing 0 and fails the run when more than {FAILED_REQUEST_PERCENT}% of the "
              "requests fail")
    started = time.monotonic()
    response = attempt_request(server["transport"], "POST", path, payload, timeout=timeout, failures=retried)
    elapsed = round(time.monotonic() - started, 3)
    if isinstance(response, RequestFailure):
        problem = f"no usable answer ({response.reason}): {response.error}"
    else:
        try:
            answer = parse(sample, response)
        except EvaluationError:
            raise
        except _PARSE_ERRORS as error:
            problem = f"the response was rejected: {type(error).__name__}: {error}; got {excerpt(response)}"
            graded = "" if isinstance(error, SlowFence) else graded  # grading would stall on it, not score it 0
        else:
            check.record(SMOKE, True, kind="smoke", measured=elapsed, unit="s",
                         detail=excerpt(answer) + ("; answered only after a retry" if retried.retried else ""))
            return answer
    check.record(SMOKE, False, kind="smoke", measured=elapsed, unit="s", detail=problem + graded)
    return None


def rendering(check, svg, binary, version):
    """The scorer's schema validation and 512x512 rendering of the smoke answer. Grading scores an invalid drawing 0
    and keeps going, so this is recorded without ending the check."""
    try:
        _, details = render_svg(svg, binary=binary)
    except InvalidSVG as error:
        check.record("valid SVG", False, kind="smoke", detail=f"{error}; grading scores such a drawing 0 (the "
                     "allowed elements and attributes are in /task/public/SCHEMA.md)")
        return
    except Exception as error:  # noqa: BLE001 - the scorer choking on this answer would fail the graded run
        check.record("valid SVG", False, kind="smoke",
                     detail=f"the scorer could not render this answer: {type(error).__name__}: {error}")
        return
    check.record("valid SVG", True, kind="smoke",
                 detail=f"{details['svg_bytes']:,} of {MAX_SVG_BYTES:,} bytes, {details['elements']} elements, "
                        f"rendered by {version}"
                        + ("" if details["nonuniform"] else "; the rendering is one flat color"))


def selfcheck(check, public_dir, config):
    """The training evidence scan, one public instruction through the graded request policy and parser (its SVG
    validated and rendered, never judged), then the throughput gate: sample_plan's share of the held-out batch,
    resampled from the public instructions, sent through the graded path at 8 in flight, its answers checked against
    the SVG schema, and the batch time projected to grading's size. The first request counts once against the
    deadline, since grading's batch starts cold right after /health. A sample counts rejected answers after the gate
    (lenient), since its share of the 5% failure limit is none. Nothing is scored."""
    public_dir = Path(public_dir)
    rows = public_rows(public_dir / "validation.jsonl")
    count, deadline = config.get("expected_samples"), config.get("inference_timeout_seconds", 3600)
    if type(count) is not int or count < 1 or type(deadline) not in (int, float) or not deadline > 0:
        raise EvaluationError("The grading config needs expected_samples and a positive inference_timeout_seconds")
    sent, share = sample_plan(count, deadline)
    scale = count / sent
    sampled = f"projected from {sent} of {count} requests at concurrency {CONCURRENCY}" if sent < count else None
    binary, version = renderer(config)
    check.static("training evidence", training_evidence, config["submission_path"])
    check.static("weight audit", weight_audit, config["submission_path"])
    batch_name = f"{count} requests, {CONCURRENCY} in flight"
    with check.launch(trusted_launcher) as server:
        started = time.monotonic()
        # Grading gives each request what is left of the batch deadline, so the first one may use all of it; here all
        # of the sample's share, which bounds the check.
        answer = smoke(check, server, rows[0], public_dir, share)
        first = time.monotonic() - started
        if answer is not None:
            rendering(check, answer["svg"], binary, version)
        check.stage("gates")
        if first >= deadline:
            check.fail(batch_name, f"the first request alone took {first:.1f} s of the {deadline:g} s batch deadline",
                       measured=overdue(first), required=deadline, unit="s", higher_is_better=False)
        # Noted with the gate it explains, not when the batch fails before measuring one.
        note = (f"The {count}-request speed gate was projected from a sample: {sent} public requests at {CONCURRENCY} "
                f"in flight within their share of the {deadline:,g} s deadline, scaled to {count}."
                if sampled else None)
        # What the first request left of the deadline, in the sample's share: the batch passes it exactly when the
        # projection (the first request once, the rest scaled) is within the deadline.
        budget = (deadline - first) / scale
        cut = f" (cut off at its {share:,.0f} s limit, so at least this)" if answer is None and first >= share else ""
        with tempfile.TemporaryDirectory(prefix="svg-selfcheck-") as directory:
            data = Path(directory) / "batch.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in resampled(rows, sent)))
            settings = dict(config, validation_mode=False, predictions_only=False, concurrency=CONCURRENCY,
                            expected_samples=sent, inference_timeout_seconds=budget)
            batch_started = time.monotonic()
            try:
                result = run_http_evaluation(server["transport"], data, settings, build_request=public.build_request,
                                             parse_response=lenient if sampled else parse, score=unscored,
                                             failed_prediction=public.failed_prediction)
            except EvaluationError:
                raise
            except VerificationFailure as error:  # the deadline, or more than 5% of the requests failed
                spent = time.monotonic() - batch_started
                # Requests cut off by the deadline time out and count as failures, which may trip the 5% limit first.
                if str(error).startswith("Batch exceeded") or spent >= budget:
                    within = (f"{sampled}: the {sent} answers did not all arrive within {budget:.1f} s, their share "
                              f"of what the first request ({first:.1f} s) left of the {deadline:g} s deadline"
                              if sampled else f"not every answer arrived within {deadline:g} s ({first:.1f} s of it "
                              "on the first request)")
                    # Cut off at the deadline, the batch says only that the server is at least this slow.
                    measured = overdue(first + spent * scale)
                    if note:
                        check.note(note)
                    check.fail(batch_name, f"{within}, so the batch was cut off and {measured:,.1f} s is only a lower "
                               "bound (the server may be much slower); grading scores a batch this slow zero",
                               measured=measured, required=deadline, unit="s", higher_is_better=False)
                limit = (f". Grading's {FAILED_REQUEST_PERCENT}% limit allows {count * FAILED_REQUEST_PERCENT // 100} "
                         f"failures in its {count} requests and {sent * FAILED_REQUEST_PERCENT // 100} in this sample "
                         f"of {sent}" if sampled else "")
                check.fail("batch responses", f"{error}{limit}", kind="smoke")
            except RecursionError as error:  # an answer nested too deeply to decode on the relay's batch path
                check.fail("batch responses", f"an answer could not be decoded: {error}", kind="smoke")
        check.stage("gates")  # the evaluator reported "scoring" before calling unscored()
        timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
        rejected, elapsed = result["details"]["rejected"], timing["elapsed_seconds"]
        drawn = f"{sent} instructions drawn from the {len(rows)} public examples"
        took = (f"{sampled}: {drawn} took {elapsed:.1f} s, about {elapsed * scale:,.0f} s for {count}" if sampled
                else f"{drawn}: {elapsed:.1f} s")
        if note:
            check.note(note)
        check.gate(batch_name, first + elapsed * scale, deadline, higher_is_better=False, unit="s",
                   detail=f"{took}, plus {first:.1f} s for the first request{cut}. The public examples are simple "
                          "icons only; detailed and complex held-out drawings usually need longer SVGs, so keep a wide "
                          "margin")
        failures = failed["total"] + len(rejected)
        if failures:
            reasons = dict(sorted((Counter(failed["reasons"]) + Counter(malformed=len(rejected))).items()))
            rate = f", about {failures * scale:.0f} of {count} at this rate" if sampled else ""
            check.record("batch responses", False, kind="smoke",
                         detail=f"{failures} of {sent} requests failed, timed out or were rejected ({reasons}){rate}; "
                                f"grading scores each 0 and fails the run above "
                                f"{count * FAILED_REQUEST_PERCENT // 100} of its {count}. First errors: "
                                f"{(failed['errors'] + rejected)[:RECORDED_ERRORS]}")
        else:
            retried = f"; {failed['retried']} only after a retry" if failed["retried"] else ""
            check.record("batch responses", True, kind="smoke", detail=f"all {sent} answers parsed{retried}")
        invalid, answered = result["details"]["invalid_svg"], sent - failures
        if invalid:
            reasons = "; ".join(f"{reason} ({number})" for reason, number in invalid.items())
            check.record("valid SVG (batch)", False, kind="smoke",
                         detail=f"{sum(invalid.values())} of {answered} answers are not valid SVG and would score 0: "
                                f"{reasons}. The allowed elements and attributes are in /task/public/SCHEMA.md")
        elif answered:  # nothing to check when every answer failed or was rejected
            check.record("valid SVG (batch)", True, kind="smoke",
                         detail=f"all {answered} answers pass the scorer's SVG schema check (not rendered)")
