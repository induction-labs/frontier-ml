#!/usr/bin/env python3
"""Evaluate causal end-of-turn probability streams from two aligned audio channels.

    python /task/validate.py --server-url http://127.0.0.1:8000

Sweeps a threshold over the returned per-frame probabilities and reports recall
at a false-positive rate of at most 0.05 (the score), the operating points at
0.05/0.10/0.15, the whole (threshold, recall, FPR) curve, signed latency
percentiles at the chosen threshold and the achieved real-time factor (audio
seconds served per second of server time). Conversations are sent one at a time,
and the whole pass must run at least 3x faster than real time: the server time of
the pass may not exceed total audio seconds / 3 (at least 60 s). On the trusted
path that clock is the HTTP client inside the serving sandbox, so request transfer
never counts; the held-out pass applies the same rule to 5.7 hours of audio in 30
conversations. --timeout-seconds (or the trusted config.json) sets only the
wall-clock allowance for the pass, which defaults to the same number locally.
"""
from pathlib import Path
import time
from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import FailedRequests, RequestFailure, attempt_request, timed_batch
from turn_taking import build_request, load_rows, parse_response, score, silent_stream, submission

TASK = "end-of-turn"
DEFAULT_DATA = "/task/public/validation.jsonl"
CONCURRENCY = 1  # one conversation in flight; a request carries the whole conversation
REQUIRED_REAL_TIME_FACTOR = 3.0  # audio seconds per second of server time; the published requirement
MINIMUM_BUDGET_SECONDS = 60.0  # floor for tiny local test sets; irrelevant at conversation lengths


def audio_seconds(rows):
    return sum(row["duration_s"] for row in rows)


def default_budget_seconds(rows):
    """The 3x real-time rule over the loaded audio: the server time the whole pass may use. Also the local default
    for the wall-clock allowance, which has no transfer to cover on a loopback transport."""
    return max(MINIMUM_BUDGET_SECONDS, audio_seconds(rows) / REQUIRED_REAL_TIME_FACTOR)


class ServingBudgetExceeded(VerificationFailure):
    """The server used more than audio / 3 seconds: the published 3x rule, a submission failure."""


class ServerClock:
    """attempt_request's transport under the clock that charges the 3x rule.

    On Harbor/Modal the transport offers `request_timed`: the trusted HTTP client inside the serving sandbox times
    the exchange, so the grader's own transfer and exec RPCs never count (about 100 MB of FLAC per request). A local
    transport is timed by this process's clock around the loopback request. A failed attempt is charged the smaller
    of its wall time and its timeout, so a hung server pays its whole remaining budget."""
    def __init__(self, transport):
        self.transport = transport
        self.trusted = callable(getattr(transport, "request_timed", None))
        self.seconds = 0.0
        self.scope = ("trusted HTTP client inside the serving sandbox; provider transfer/exec latency excluded"
                      if self.trusted else "this process's clock around each loopback request")

    def request(self, method, path, payload=None, *, timeout):
        started = time.monotonic()
        try:
            if self.trusted:
                response, elapsed = self.transport.request_timed(method, path, payload, timeout=timeout)
            else:
                response = self.transport.request(method, path, payload, timeout=timeout)
                elapsed = time.monotonic() - started
        except EvaluationError:
            raise
        except Exception:
            self.seconds += min(time.monotonic() - started, timeout)
            raise
        self.seconds += elapsed
        return response


def failed_prediction(row, failure):
    """A conversation whose request failed after the shared retries, timed out or answered in an undocumented
    shape (see CONTRACT.md) counts as all-zero probabilities: every gold end of turn in it is missed at any
    threshold, no pause is flagged. That is its natural zero under recall at a fixed false-positive rate."""
    return {"id": row["id"], "speaker_1": silent_stream(row["duration_s"]), "speaker_2": silent_stream(row["duration_s"]),
            "request_failed": {"reason": failure.reason, "error": failure.error}}


def run(transport, data_path, config):
    config = dict(config or {})
    prediction_only = config.get("predictions_only", False)
    rows = load_rows(data_path, require_gold=not prediction_only)
    limit = config.get("max_conversations")
    if limit:
        rows = rows[:int(limit)]
    expected = config.get("expected_samples")
    if expected is not None and len(rows) != expected:
        raise ValueError(f"Expected {expected} conversations; found {len(rows)}")
    root = Path(data_path).parent
    # The shared per-item policy: a transport failure is retried twice on the identical payload inside the pass
    # budget, a conversation that still fails or answers badly scores as all-zero probabilities, and more than 5%
    # of the conversations failing (two of the 30 held out) fails the run closed.
    failures = FailedRequests(len(rows))
    clock = ServerClock(transport)
    serving_budget = default_budget_seconds(rows)
    served = audio_seconds(rows)

    def over_budget():
        return ServingBudgetExceeded(
            f"Serving used {clock.seconds:.0f} s of server time for {served:.0f} s of audio "
            f"({served / max(clock.seconds, 1e-12):.2f}x real time); at least {REQUIRED_REAL_TIME_FACTOR:g}x is "
            "required, so the score is zero")

    def predict(row, timeout):
        remaining = serving_budget - clock.seconds
        if remaining <= 0:
            raise over_budget()
        path, payload = build_request(row, root)
        response = attempt_request(clock, "POST", path, payload, timeout=min(timeout, remaining), failures=failures)
        if isinstance(response, RequestFailure) and response.reason == "timeout" and clock.seconds >= serving_budget:
            raise over_budget()
        if not isinstance(response, RequestFailure):
            try:
                return parse_response(row, response)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                response = RequestFailure("malformed", f"{type(error).__name__}: {str(error)[:200]}")
        failures.record(row["id"], response)
        return failed_prediction(row, response)

    # The wall-clock allowance for the pass: --timeout-seconds, or the trusted config.json built from task.yaml
    # `timeout` (10,800 s: the 3x budget plus request transfer). Locally it defaults to the serving budget itself.
    wall_budget = config.get("inference_timeout_seconds") or serving_budget
    # Build the large lossless request lazily, one conversation at a time.
    try:
        predictions, timing = timed_batch(rows, predict, concurrency=CONCURRENCY, timeout_seconds=wall_budget)
    except VerificationFailure as error:
        # timed_batch's own deadline (raised from a TimeoutError) while the server stayed inside its 3x budget on the
        # trusted clock: the grader's transfer ate the allowance. The submission was within the rule: no score imputed.
        if isinstance(error.__cause__, TimeoutError) and clock.trusted and clock.seconds <= serving_budget:
            raise EvaluationError(f"The grading transport used up the {wall_budget:g} s wall-clock allowance while the "
                                  f"server itself used {clock.seconds:.0f} s of its {serving_budget:.0f} s budget "
                                  f"({clock.scope}); no score imputed") from error
        raise
    if clock.seconds > serving_budget:
        raise over_budget()
    timing.update({"audio_seconds": served, "server_seconds": clock.seconds, "serving_budget_seconds": serving_budget,
                   "wall_seconds": timing["elapsed_seconds"], "timing_scope": clock.scope,
                   "real_time_factor": served / max(clock.seconds, 1e-12),
                   "required_real_time_factor": REQUIRED_REAL_TIME_FACTOR, "retried_requests": failures.retried})
    result = {"score": None, "details": {"scoring": "predictions only"}} if prediction_only else score(rows, predictions)
    result["details"]["timing"] = timing
    result["details"]["failed_requests"] = failures.summary()
    if prediction_only:  # the streams are large (tens of thousands of frames per speaker); exported only on request
        result["submission"] = submission(predictions)
    return result


def extra_arguments(parser):
    parser.set_defaults(timeout_seconds=None)  # wall-clock allowance; default: total audio seconds / 3 (see above)
    parser.add_argument("--conversations", type=int, default=0, help="limit to the first N conversations")
    parser.add_argument("--predictions-only", action="store_true",
                        help="export the probability streams without labels or scoring (large; pair with --output)")


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=extra_arguments,
                       extra_config=lambda args: {"max_conversations": args.conversations, "predictions_only": args.predictions_only})


if __name__ == "__main__":
    raise SystemExit(main())
