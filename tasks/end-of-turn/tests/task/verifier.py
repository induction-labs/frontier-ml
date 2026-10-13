"""Trusted causal-model review and fixed-FPR EOT scoring.

The server returns a causal per-frame probability stream per speaker; the scorer
sweeps the threshold over the pooled held-out windows and the scalar is recall
at a false-positive rate of at most 0.05 (recall at 0.10 and 0.15, the chosen
threshold, the whole curve and latency percentiles are reported alongside).
There is no server-side operating point to qualify and no FP cliff. The one
timing gate is the published 3x real-time rule, charged in server time: the
trusted HTTP client inside the serving sandbox times each of the 30 sequential
requests (validate.ServerClock), so the grader's ~100 MB transfers never count.
task.yaml `timeout` (10,800 s, tests/config.json inference_timeout_seconds) is only
the wall-clock allowance for the pass; exhausting it while the server is within
its budget is an EvaluationError, never a zero. There are no per-frame latency
gates or fixed decision intervals.
"""
import json
from pathlib import Path
from frontier_ml.runtime.inference import DEFAULT_STARTUP_SECONDS
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import RequestFailure, attempt_request
import turn_taking

public = sibling(__file__, "validate.py")
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])

# Self-check (selfcheck below): two of the 8 public conversations near the held-out mean length (682 s), about
# 170 MB of the 820 MB of public audio. The first is the smoke sample and, as in grading, the cold first request.
SELFCHECK_CONVERSATIONS = ("122", "124")  # 684 s and 617 s of audio
SELFCHECK_PUBLIC = ("validation.jsonl",) + tuple(f"audio/{cid}/speaker_{speaker}.flac"
                                                 for cid in SELFCHECK_CONVERSATIONS for speaker in (1, 2))
HELDOUT_CONVERSATIONS = 30  # task.yaml expected_samples
HELDOUT_AUDIO_SECONDS = 20455.2  # the held-out pass (5.7 h); a third is the contract's 6,818 s
# A self-check request waits half its audio's duration (paces down to 2x real time are measured, slower ones fail
# on the timeout). The cold first one also gets the contract's 300 s readiness allowance for warm-up a server defers
# to its first request, which grading charges once against the whole pass. That is at most ~950 s of waiting plus
# two ~120 MB transfers, inside the self-check's work budget (task.yaml selfcheck_timeout_seconds).
WAIT_REAL_TIME_FACTOR = 2.0
SPEED_GATE = "real-time factor"


def default_config():
    # The budget is task.yaml `timeout`; the build applies these defaults over it, so it is not restated here.
    return {"concurrency": public.CONCURRENCY}


def preflight(config, data_path):
    # Incomplete labelled data fails before review or GPU provisioning.
    try:
        rows = turn_taking.load_rows(data_path)
    except ValueError as error:
        raise VerificationFailure(str(error)) from error
    expected = config.get("expected_samples")
    if expected is not None and len(rows) != int(expected):
        raise VerificationFailure(f"Expected {expected} conversations; found {len(rows)}")
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")


def verify(server, data_path, config):
    preflight(config, data_path)
    review = verify_review(server["snapshot"], config["architecture_review_path"], "end-of-turn", REVIEW_CHECKS)
    result = public.run(server["transport"], data_path,
                        {**default_config(), **config, "validation_mode": False, "predictions_only": False})
    result["details"]["gates"] = {"architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")}}
    return result


def selfcheck(check, public_dir, config):
    """Grading's request path and 3x rule on SELFCHECK_CONVERSATIONS (frontier_ml/selfcheck.py); never a score.

    Nothing static runs beyond the review prerequisites selfcheck.run checks first: preflight checks only the
    held-out rows and the review receipt, and serve_submission's serve.sh and file checks run at launch. The
    conversations go one at a time, as grading sends them, under the shared retry policy on validate.ServerClock (the
    clock grading charges the 3x rule on) and through the grader's parser. Grading counts its cold first request, so
    the first conversation counts here too: the held-out pass costs that request plus the rest of the held-out audio
    at the second conversation's pace, which the gate projects instead of sending 5.7 hours of audio. Failed attempts
    are charged once, as grading charges them; only the answered attempt sets the pace.
    """
    root = Path(public_dir)
    first, second = _selfcheck_rows(root)
    warmup = config.get("startup_timeout_seconds", DEFAULT_STARTUP_SECONDS)
    sampled = f"{len(SELFCHECK_CONVERSATIONS)} of {HELDOUT_CONVERSATIONS} requests at concurrency {public.CONCURRENCY}"
    with check.launch() as server:
        check.note(f"The 3x real-time gate is projected from {sampled}: public conversations {first['id']} (the cold "
                   f"first request) and {second['id']} (its pace). Each gets half its audio's duration to answer, the "
                   f"first {warmup:g} s more for warm-up; grading bounds only the whole pass.")
        clock = public.ServerClock(server["transport"])
        # Retried on the clock exactly as validate.run retries (always); the wait bounds all of its attempts.
        check.request({**server, "transport": clock}, *turn_taking.build_request(first, root),
                      name=_request_name(first), parse=_streams, sample=first, retry=True,
                      timeout=first["duration_s"] / WAIT_REAL_TIME_FACTOR + warmup)
        cold = clock.seconds
        check.stage("gates")
        warm, failed = _warm_seconds(check, clock, second, root)
        rest = HELDOUT_AUDIO_SECONDS - first["duration_s"]
        projected = max(cold + failed + warm / second["duration_s"] * rest, 1e-6)
        retried = f", plus {failed:.1f} s of its failed attempts, charged once" if failed else ""
        allowed = HELDOUT_AUDIO_SECONDS / public.REQUIRED_REAL_TIME_FACTOR
        detail = (f"Held-out pass projected from {sampled} at {projected:.0f} s of server time for "
                  f"{HELDOUT_AUDIO_SECONDS:.0f} s of audio, {allowed:.0f} s allowed: the cold first request "
                  f"(conversation {first['id']}, {first['duration_s']:.0f} s of audio in {cold:.1f} s) plus the other "
                  f"{rest:.0f} s at conversation {second['id']}'s pace ({second['duration_s']:.0f} s in {warm:.1f} s, "
                  f"{_pace(second['duration_s'], warm)}){retried}. The two conversations alone: "
                  f"{_pace(first['duration_s'] + second['duration_s'], clock.seconds)}. Clock: {clock.scope}.")
        check.gate(SPEED_GATE, HELDOUT_AUDIO_SECONDS / projected, public.REQUIRED_REAL_TIME_FACTOR,
                   higher_is_better=True, unit="x real time", detail=detail)


def _selfcheck_rows(root):
    """The SELFCHECK_CONVERSATIONS rows of the fetched public files; missing ones are the self-check's fault."""
    try:
        rows = {row["id"]: row for row in turn_taking.load_rows(root / "validation.jsonl", require_gold=False)}
        chosen = [rows[cid] for cid in SELFCHECK_CONVERSATIONS]
        missing = [reference for row in chosen for reference in row["audio"].values()
                   if not turn_taking.audio_path(reference, root).is_file()]
    except (OSError, ValueError, KeyError) as error:
        raise EvaluationError(f"The self-check's public conversations are unavailable: {type(error).__name__}: "
                              f"{error}") from error
    if missing:
        raise EvaluationError(f"The self-check's public audio is missing: {', '.join(missing)}")
    return chosen


def _request_name(row):
    return f"POST /predict (conversation {row['id']})"


def _streams(row, response):
    """The grader's parse_response, reduced to what the report shows: each speaker's hop, frame count and peak."""
    prediction = turn_taking.parse_response(row, response)
    return {"conversation": row["id"], **{speaker: {"hop_seconds": prediction[speaker]["hop_seconds"],
                                                    "frames": len(prediction[speaker]["eot_probability"]),
                                                    "peak": max(prediction[speaker]["eot_probability"], default=0)}
                                          for speaker in ("speaker_1", "speaker_2")}}


def _pace(audio_seconds, server_seconds):
    return f"{audio_seconds / max(server_seconds, 1e-9):.1f}x real time"


class _Charges:
    """The 3x clock with each attempt's charge kept apart: the projection scales only the answered attempt."""
    def __init__(self, clock):
        self.clock, self.seconds = clock, []

    def request(self, method, path, payload=None, *, timeout):
        before = self.clock.seconds
        try:
            return self.clock.request(method, path, payload, timeout=timeout)
        finally:
            self.seconds.append(self.clock.seconds - before)


def _warm_seconds(check, clock, row, root):
    """Server seconds of the warm conversation's answered attempt and of its failed ones. No answer in time fails
    the speed gate, a bad answer the request."""
    name, wait = _request_name(row), row["duration_s"] / WAIT_REAL_TIME_FACTOR
    charges = _Charges(clock)
    response = attempt_request(charges, "POST", *turn_taking.build_request(row, root), timeout=wait)
    if isinstance(response, RequestFailure) and response.reason == "timeout":
        check.fail(SPEED_GATE, f"conversation {row['id']} ({row['duration_s']:.0f} s of audio) got no answer within "
                   f"{wait:.0f} s, slower than {WAIT_REAL_TIME_FACTOR:g}x real time; the held-out pass needs "
                   f"{public.REQUIRED_REAL_TIME_FACTOR:g}x ({response.error})",
                   required=public.REQUIRED_REAL_TIME_FACTOR, unit="x real time")
    if isinstance(response, RequestFailure):
        check.fail(name, response.error, kind="smoke")
    try:
        streams = _streams(row, response)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - the grader would score this conversation as silence
        from frontier_ml.selfcheck import excerpt
        check.fail(name, f"the response was rejected: {type(error).__name__}: {error}; got {excerpt(response)}",
                   kind="smoke")
    check.record(name, True, kind="smoke", detail=json.dumps(streams))
    return charges.seconds[-1], sum(charges.seconds[:-1])
