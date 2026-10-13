"""Trusted data/model provenance review, then action F1 over the held-out clips."""
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import time

from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import failure_reason

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
# The self-check reads the public example row and its MP4 (about 1 MB), not SCHEMA.md.
SELFCHECK_PUBLIC = ("validation.jsonl", "*.mp4")
# A self-check is a smoke test and the speed gates, not a held-out-sized run: a batch whose deadline exceeds
# SELFCHECK_BATCH_SECONDS is sampled, at least SELFCHECK_MIN_CLIPS clips (max(2 x concurrency 1, 8)).
SELFCHECK_BATCH_SECONDS = 600
SELFCHECK_MIN_CLIPS = 8


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")


def verify(server, data_path, config):
    preflight(config, data_path)
    review = verify_review(server["snapshot"], config["architecture_review_path"], "idm-training", REVIEW_CHECKS)
    result = public.run(server["transport"], data_path,
                        {**config, "validation_mode": False, "predictions_only": False})
    result["details"]["gates"] = {"architecture": {"reviewer": review["reviewer"], "checks": review["checks"]}}
    return result


class TimedTransport:
    """The grading transport, timing every clip from its first attempt to the end of its last (retries and their
    backoff included, as the per-clip ceiling bounds them) and counting the clips that hit that ceiling: a timeout once
    the batch deadline has passed is the deadline's, not the ceiling's. Retries resend the identical body, so a new
    body is a new clip."""

    def __init__(self, transport):
        self.transport, self.seconds, self.timeouts, self.deadline = transport, [], 0, math.inf
        self.body = self.clip_started = None

    def request(self, *args, **kwargs):
        started = time.monotonic()
        if kwargs.get("body") is not self.body:
            self.body, self.clip_started = kwargs.get("body"), started
            self.seconds.append(0.0)
        try:
            return self.transport.request(*args, **kwargs)
        except Exception as error:
            if (not isinstance(error, EvaluationError) and failure_reason(error) == "timeout"
                    and time.monotonic() < self.deadline):
                self.timeouts += 1
            raise
        finally:
            self.seconds[-1] = time.monotonic() - self.clip_started


def nonfinite(value, path="result"):
    """Where the first NaN or infinite number in `value` is: grading writes result.json with allow_nan=False."""
    if isinstance(value, float) and not math.isfinite(value):
        return path
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in items:
        found = nonfinite(item, f"{path}.{key}" if isinstance(key, str) else f"{path}[{key}]")
        if found:
            return found
    return None


def selfcheck_sample(count, budget):
    """How many clips the self-check sends one at a time, and their deadline: grading's `count` clips and `budget` when
    it fits in SELFCHECK_BATCH_SECONDS, else count x SELFCHECK_BATCH_SECONDS / budget of them (at least
    SELFCHECK_MIN_CLIPS) in their proportional share of the budget."""
    if budget <= SELFCHECK_BATCH_SECONDS:
        return count, budget
    sent = min(count, max(SELFCHECK_MIN_CLIPS, math.floor(count * SELFCHECK_BATCH_SECONDS / budget)))
    return sent, budget * sent / count


def selfcheck_batch(rows, public_dir, directory, count):
    """`count` public rows, cycled with unique ids, as a dataset validate.run reads: it only reads videos from below
    the JSONL file's directory, so the clips are copied next to it."""
    for row in rows:
        target = directory / row["video"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(public_dir / row["video"], target)
    batch = [{**rows[index % len(rows)], "id": f"{rows[index % len(rows)]['id']}#{index}"} for index in range(count)]
    path = directory / "batch.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in batch))
    return path


def smoke_response(sample, response):
    """The grader's parser on one answer, plus the canonical-schema warning validate.py reports per clip."""
    actions, dropped = public.parse_actions(response)
    try:
        public.validate_actions(response, sample["n_frames"])
        warning = None
    except (ValueError, TypeError) as error:
        warning = str(error)
    return {"actions": len(actions), "dropped_actions": dropped, "warning": warning, "first": actions[:3]}


def selfcheck(check, public_dir, config):
    """One public clip through grading's request path and parser, then grading's batch loop on copies of it.

    The batch is validate.run itself: one clip at a time, the per-clip ceiling, retries, the batch deadline and the
    failed-clip limit; its F1 over copies of the public example is not reported. A deadline longer than
    SELFCHECK_BATCH_SECONDS is sampled (selfcheck_sample): the sample runs in its share of the deadline under the
    same percentage failed-clip limit, and its time is projected to the whole batch. preflight checks only the review
    receipt, whose free prerequisites selfcheck.run checks before this hook, so there is nothing static to run.
    """
    from frontier_ml.selfcheck import CheckFailed

    public_dir = Path(public_dir)
    rows = [json.loads(line) for line in (public_dir / "validation.jsonl").read_text().splitlines() if line.strip()]
    per_request = config.get("request_timeout_seconds", public.REQUEST_TIMEOUT_SECONDS)
    budget = config.get("inference_timeout_seconds", public.BATCH_TIMEOUT_SECONDS)
    count = int(config.get("expected_samples") or len(rows))
    sent, share = selfcheck_sample(count, budget)
    scale = count / sent
    sampled = f"projected from {sent} of {count} requests at concurrency 1" if sent < count else None
    allowed = count * public.MAX_FAILED_CLIP_PERCENT // 100  # score() fails the run above this many failed clips
    batch_name = f"{count} clips one at a time"
    with tempfile.TemporaryDirectory(prefix="idm-selfcheck-") as directory:
        batch = selfcheck_batch(rows, public_dir, Path(directory), sent)
        with check.launch() as server:
            timed = TimedTransport(server["transport"])
            body, headers = public.build_request(rows[0], public_dir, 0)
            # Through `timed`, so the slowest-request gate and the cold-clip charge below include this clip.
            try:
                check.request({"transport": timed}, "/actions", body=body, headers=headers, timeout=per_request,
                              parse=smoke_response, sample=rows[0])
            except CheckFailed:
                if timed.timeouts:  # the report's detail is the transport's; the ceiling is grading's
                    check.note(f"The public clip did not answer within grading's {per_request:g} s per-clip ceiling "
                               f"(cut after {timed.seconds[0]:.1f} s), which grading's first held-out clip has too; "
                               "a first request includes any lazy loading.")
                raise
            check.stage("gates")
            if sampled:
                check.note(f"Grading's {count}-clip batch was sampled: {sent} copies of the public clip in their "
                           f"{share:.0f} s share of the {budget:g} s deadline; its time and failed clips were "
                           "projected from them.")
            started = time.monotonic()
            timed.deadline = started + share
            failure = None
            try:
                result = public.run(timed, batch, {**config, "validation_mode": False, "predictions_only": False,
                                                   "expected_samples": sent, "inference_timeout_seconds": share})
            except (EvaluationError, OSError):  # the verifier's faults; OSError: reading the staged copies
                raise
            except Exception as error:  # grading fails the run on anything else verify raises
                failure = error
            elapsed = time.monotonic() - started
            slowest = max(timed.seconds)
            # A clip cut at the ceiling took the whole ceiling, whatever its last attempt's share of it was.
            check.record("slowest request", slowest <= per_request and not timed.timeouts,
                         measured=max(slowest, per_request) if timed.timeouts else slowest, required=per_request,
                         unit="s", higher_is_better=False,
                         detail=f"{len(timed.seconds)} clips one at a time, each from its first attempt to its last "
                                f"(retries included), median {statistics.median(timed.seconds):.2f} s, the first "
                                f"(cold) {timed.seconds[0]:.2f} s"
                                + (f"; {timed.timeouts} timed out at the ceiling, and grading scores such a clip "
                                   "as empty" if timed.timeouts else ""))
            if failure is not None and not isinstance(failure, VerificationFailure):
                # e.g. a huge MouseMove overflowing, or a list as an action's type; grading's run fails the same way
                check.fail("answers scoreable", f"grading's scorer raised {type(failure).__name__}: {failure} on "
                           "these answers, which fails the whole run", kind="smoke")
            if failure is not None and elapsed >= share:  # clips cut by the deadline count as failed clips
                # The elapsed time alone always comes to about the deadline; the clips' median shows how far over a
                # slow server is. The clip cut in flight counts only to the deadline, so both are lower bounds.
                median = statistics.median(timed.seconds[1:] or [0.0])
                cut = (f"the sample's {share:.0f} s share of the {budget:g} s deadline passed before all {sent} "
                       f"sampled clips were sent ({len(timed.seconds) - 1} sent, median {median:.2f} s; {sampled}, "
                       "a lower bound)" if sampled
                       else f"{budget:g} s passed before all {count} clips were sent ({len(timed.seconds) - 1} clips "
                            f"sent, median {median:.2f} s; projected to all {count}, a lower bound)")
                check.fail(batch_name, f"{cut}; grading scores unsent clips as empty and fails the run above "
                           f"{allowed} failed clips", measured=max(elapsed * scale, median * count), required=budget,
                           unit="s", higher_is_better=False)
            if failure is not None:
                # The sample's limit is grading's percentage, so its failure projects past grading's count.
                check.fail("failed clips", str(failure) + (f" ({sampled}: at that rate grading's {count} clips would "
                                                           f"have more than the {allowed} failed clips it allows)"
                                                           if sampled else ""), required=allowed, unit="clips")
            details = result["details"]
            # Grading's first held-out clip is the cold one; here the smoke clip took that cost, so it is added back
            # once, unscaled.
            cold = max(0.0, timed.seconds[0] - statistics.median(timed.seconds[1:] or [0.0]))
            warm = details["elapsed_seconds"]
            check.gate(batch_name, warm * scale + cold, budget, higher_is_better=False, unit="s",
                       detail=f"grading's request loop (validate.py run()) on {sent} copies of the public clip: "
                              f"{warm:.1f} s on the server the smoke request warmed up"
                              + (f", {warm * scale:.1f} s for {count} ({sampled})" if sampled else "")
                              + f", plus {cold:.1f} s the cold first clip took over the median; held-out clips "
                              "differ")
            failed = details["failed_clips"]
            check.gate("failed clips", len(failed) * scale if sampled else len(failed), allowed,
                       higher_is_better=False, unit="clips",
                       detail=(f"{len(failed)} of {sent} clips" + (f" ({sampled})" if sampled else "")
                               + f"; first: {failed[0]['error']}") if failed
                       else f"none of {sent} ({details['failed_requests']['retried']} needed a retry)")
            try:
                json.dumps(result, allow_nan=False)
            except ValueError as error:  # grading cannot write such a result: the run ends without a score
                check.fail("answers scoreable", f"grading could not record the result ({error}, at "
                           f"{nonfinite(result)}): a NaN or infinite action value on a labelled frame, such as "
                           "MouseMove details \"nan,nan\", makes the scorer's statistics non-finite", kind="smoke")
