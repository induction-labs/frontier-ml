#!/usr/bin/env python3
"""Score your screenshot generator on the public validation samples with the task metric.

    python /task/validate.py --prepare-models /tmp/scorer-models          # download the pinned OCR once
    python /task/validate.py --server-url http://127.0.0.1:8000 --scorer-models /tmp/scorer-models \\
        --judge-base-url https://YOUR_BROKER/v1 --judge-model gpt-6-luna  # key in $MLE_JUDGE_API_KEY
    python /task/validate.py --server-url http://127.0.0.1:8000 --predictions-only --save-dir out/

Sends each sample to POST /v1/images/generations one request at a time, with exactly the request
the final client sends ({"description", "texts", "size"}), and scores the returned images. A request
that fails (non-200, connection error, invalid JSON) is retried twice on the same payload, as in the
final evaluation, with the retries and their backoff inside the request's 120 s ceiling
(--request-timeout-seconds). One that still fails, does not arrive within the 120 s, or is not one
1280x720 8-bit RGB PNG scores 0 for its sample; every other sample is scored normally. Per sample,
quality = geomean(T, L, F) and score = (quality + R) / 2, the mean of quality and
realism: T is text accuracy (pinned PP-OCRv5 on each requested box crop), L layout F1
(pinned full-page OCR versus the requested boxes), F faithfulness to the description and R realism (the
gpt-6-luna judge on fixed 0-9 rubrics; each rating is the single digit in the judge's reply over 9); the
result is the mean over samples. Text pasted perfectly onto an unrealistic background therefore scores
little more than its realism rating. The requested texts and boxes are the pinned OCR's own output on
the real screenshots. OCR runs on a CUDA GPU and needs: pip install rapidocr==3.9.2 onnxruntime-gpu==1.23.2.
F and R need your budgeted OpenAI-compatible judge endpoint (--judge-base-url, --judge-model, key in
$MLE_JUDGE_API_KEY); the request sent is exactly the final verifier's (screenshot_generation.JUDGE_PARAMS:
max_completion_tokens and seed only; no logprobs or reasoning_effort, which the judge proxy rejects).
Without a judge, T and L are reported and no score is given (none is imputed). Use --components T L to
score only text and layout (the score is then their geomean). --predictions-only needs only the standard library.

The score is the metric only. As diagnostics that never change the score, the report also shows
p95 per-image latency against the final 30 s requirement and the failed requests against the final
limit: the final evaluation enforces the p95 limit, the 7,200 s total, and fails the run when more than
one request in twenty fails or returns an invalid image; see /task/CONTRACT.md. Latency is timed by
the HTTP client next to the server (here this process; in the final evaluation the trusted client inside
the serving sandbox), so grading-transport overhead never counts.
"""
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import screenshot_generation as sg
from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import (FAILED_REQUEST_PERCENT, REQUEST_ATTEMPTS, RETRY_BACKOFF_SECONDS,
                                               RequestFailure, attempt_request, failure_reason, load_samples)

TASK = "screenshot-generation"
DEFAULT_DATA = "/task/public/validation.jsonl"
REQUEST_TIMEOUT_SECONDS = 120  # per-request ceiling; task.yaml request_timeout_seconds must match (disclosed in the contract)
METRIC = sg.METRIC  # mean_accuracy_realism: mean over samples of (geomean(T, L, F) + R) / 2
SANDBOX_CLOCK = "trusted HTTP client inside the serving sandbox; grading transport (provider transfer/exec) excluded"
LOCAL_CLOCK = "this client's clock around each local HTTP request"


def build_request(sample, data_root=None):
    return sg.build_request(sample, data_root)


def timing_scope(transport):
    """Where per-request seconds are measured for this transport (recorded next to every latency figure)."""
    return SANDBOX_CLOCK if callable(getattr(transport, "request_batch", None)) else LOCAL_CLOCK


def request_image(transport, sample, timeout, failures=None):
    """One request under the shared retry policy. Returns (png_bytes or None, RequestFailure or None, seconds).

    A transport failure (non-200, connection error, invalid JSON) is retried twice on the identical payload; a
    timeout is not. The retries and their backoff run inside `timeout`, the request's ceiling, and are part of
    the seconds returned: a retry is the server's problem. A request that still fails, times out, or answers
    something other than one 1280x720 8-bit RGB PNG comes back as a RequestFailure for its sample alone.
    Seconds come from the trusted HTTP client inside the serving sandbox when the transport offers it
    (`request_batch`, used here with a single job so requests stay sequential); that clock excludes the
    grading transport's own transfer/exec latency. Otherwise they are this process's clock around a local
    HTTP request. `failures` (anything with a `lock` and a `retried` counter) counts the requests that succeeded
    only after a retry.
    A grading-infrastructure fault (EvaluationError) propagates unchanged; it is never a sample's failure.
    """
    path, payload = build_request(sample)
    batch = getattr(transport, "request_batch", None)
    started = time.monotonic()
    try:
        if callable(batch):
            responses, timing = batch([(path, payload)], concurrency=1, timeout_seconds=timeout, raise_on_failure=False)
            response, elapsed = responses[0], float(timing["elapsed_seconds"])
            if failures is not None and timing.get("retried_requests") and not isinstance(response, RequestFailure):
                with failures.lock:
                    failures.retried += 1
        else:
            response = attempt_request(transport, "POST", path, payload, timeout=timeout, failures=failures)
            elapsed = time.monotonic() - started
    except EvaluationError:
        raise
    except Exception as error:  # a transport fault the retry policy did not absorb, e.g. the sandbox client's deadline
        return None, RequestFailure(failure_reason(error), f"{type(error).__name__}: {error}"), time.monotonic() - started
    if isinstance(response, RequestFailure):
        return None, response, elapsed
    try:
        return sg.decode_response(response), None, elapsed
    except ValueError as error:
        return None, RequestFailure("malformed", str(error)), elapsed


def load_checked_samples(data_path, config):
    samples = load_samples(data_path)
    if config.get("limit"):
        samples = samples[:int(config["limit"])]
    for sample in samples:
        try:
            sg.validate_sample(sample)
        except ValueError as error:
            raise VerificationFailure(f"Malformed evaluation sample {sample.get('id')}: {error}") from error
    return samples


def scorer_config(config):
    return {"progress_path": config.get("progress_path"),
            "scorer_models_dir": config.get("scorer_models_dir"), "components": config.get("components"),
            "ocr_threads": config.get("ocr_threads", 8), "judge": config.get("judge"),
            "judge_model": config.get("judge_model") or sg.JUDGE_MODEL,
            "judge_concurrency": config.get("judge_concurrency", sg.JUDGE_CONCURRENCY)}


def run(transport, data_path, config):
    config = dict(config or {})
    samples = load_checked_samples(data_path, config)
    timeout = float(config.get("request_timeout_seconds", REQUEST_TIMEOUT_SECONDS))
    pngs, errors, seconds = [], {}, []
    retries = SimpleNamespace(lock=threading.Lock(), retried=0)  # a counter only: the public tool has no run limit
    for sample in samples:
        png, failure, elapsed = request_image(transport, sample, timeout, failures=retries)
        pngs.append(png)
        seconds.append(elapsed)
        if failure is not None:
            errors[sample["id"]] = {"reason": failure.reason, "error": failure.error, "attempts": failure.attempts}
    if config.get("save_dir"):
        out = Path(config["save_dir"])
        out.mkdir(parents=True, exist_ok=True)
        for sample, png in zip(samples, pngs):
            if png is not None:
                (out / f"{sample['id']}.png").write_bytes(png)
    diagnostics = {"latency": {**sg.latency_summary(seconds), "diagnostic_only": True,
                               "timing_scope": timing_scope(transport),
                               "request_timeout_seconds": timeout},
                   # Each failed sample scored 0 above; the final evaluation also fails the run past this limit.
                   "request_failures": {"failed": len(errors), "requests": len(samples), "retried": retries.retried,
                                        "attempts": REQUEST_ATTEMPTS, "backoff_seconds": list(RETRY_BACKOFF_SECONDS),
                                        "final_limit_percent": FAILED_REQUEST_PERCENT, "diagnostic_only": True}}
    if config.get("predictions_only"):
        return {"score": None, "details": {"metric": "not scored (--predictions-only)", "samples": len(samples),
                                           "invalid_responses": errors, "diagnostics": diagnostics,
                                           "saved_to": config.get("save_dir")}}
    scorer, unavailable = scorer_config(config), None
    if not scorer["components"] and not callable(scorer["judge"]):
        scorer["components"] = ["T", "L"]
        unavailable = "no judge configured (--judge-base-url and --judge-model, key in $MLE_JUDGE_API_KEY)"
    try:
        scored = sg.score_images(samples, pngs, scorer)
    except sg.ScorerUnavailable as error:  # the scorer's own fault (pinned models, judge): an evaluation error
        raise EvaluationError(f"Scorer unavailable; no score imputed: {error}") from error
    details = {"metric": METRIC, "samples": len(samples), "means": scored["means"], "components": scored["components"],
               "invalid_responses": errors, "diagnostics": diagnostics, "records": scored["records"]}
    if unavailable:
        details["metric"] = "not scored: faithfulness and realism unavailable, no score imputed (T and L means reported)"
        details["judge"] = {"available": False, "reason": unavailable}
        return {"score": None, "details": details}
    if "judge_model" in scored:
        details["judge_model"] = scored["judge_model"]
    return {"score": scored["score"], "details": details}


def _arguments(parser):
    parser.add_argument("--scorer-models", help="directory prepared with --prepare-models")
    parser.add_argument("--prepare-models", metavar="DIR", help="download and verify the pinned OCR models, then exit")
    parser.add_argument("--components", nargs="+", choices=list(sg.COMPONENTS), help="subset of metric parts (default all)")
    parser.add_argument("--predictions-only", action="store_true", help="only request and check images")
    parser.add_argument("--save-dir", help="write generated PNGs here")
    parser.add_argument("--limit", type=int, help="only the first N samples")
    parser.add_argument("--request-timeout-seconds", type=float, default=REQUEST_TIMEOUT_SECONDS,
                        help=f"per-request ceiling; the final evaluation enforces {REQUEST_TIMEOUT_SECONDS} s "
                             "(a slower response counts as invalid)")


def _config(args):
    return {"scorer_models_dir": args.scorer_models, "components": args.components,
            "predictions_only": args.predictions_only, "save_dir": args.save_dir, "limit": args.limit,
            "request_timeout_seconds": min(args.request_timeout_seconds, args.timeout_seconds)}


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--prepare-models" in argv:
        target = argv[argv.index("--prepare-models") + 1]
        print(sg.prepare_models(target))
        return 0
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__, judge=True,
                       extra_arguments=_arguments, extra_config=_config)


if __name__ == "__main__":
    raise SystemExit(main())
