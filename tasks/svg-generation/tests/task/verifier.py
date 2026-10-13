"""Trusted SVG verifier: frozen data, weight audit, training review, the trusted gpt-oss-20b FP8 server, byte gate,
safe rendering and the graded gpt-6-sol rubric."""
import hashlib
import http.client
import json
import math
from pathlib import Path
import time
import urllib.error

from frontier_ml.runtime import judge as judge_runtime
from frontier_ml.runtime.inference import serve_submission
from frontier_ml.runtime.judge import judge_from_config
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from svg_audit import inspect_submission
from svg_render import OFFICIAL_RENDERER_VERSION, renderer_binary
from svg_weights import inspect_weights

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
LOCK = Path(__file__).resolve().parent / "pipeline/prepared_lock.json"
READINESS = LOCK.with_name("judge_readiness.json")
REPORT = LOCK.with_name("judge_calibration_v5_results.json")
# The task serves the submitted weights itself; see trusted_server/serve.sh and Dockerfile.inference.
TRUSTED_SERVER = Path(__file__).resolve().parent / "trusted_server"
# Preregistered readiness thresholds for the graded rubric (pipeline/calibrate_v5.py checks the same numbers).
ADMISSION = {"samples_per_group": 24, "reference_minimum": 0.6, "constant_maximum": 0.05, "injection_maximum": 0.0,
             "repeat_macro_difference_maximum": 0.05, "repeat_item_correlation_minimum": 0.8}
CONCURRENCY = 8
# Transient transport failures are retried a bounded number of times with the identical payload;
# a malformed or non-transient answer still fails the run with no imputed score.
JUDGE_ATTEMPTS = 4
TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}
SLEEP = time.sleep


def _retry_delay(attempt, error=None):
    delay = 2 ** attempt
    header = getattr(getattr(error, "headers", None), "get", lambda *_: None)("Retry-After")
    if header and str(header).strip().isdigit():
        delay = max(delay, int(header))
    return min(delay, 30)


def calibrated_judge(judge):
    """Match the calibrated JSON payload; transport whitespace is immaterial."""
    if not callable(judge):
        raise VerificationFailure("A trusted SVG judge callable is required")

    def wrapped(**payload):
        if payload.get("service_tier", "default") != "default" or payload.get("store", False) is not False:
            raise VerificationFailure("SVG judging requires Standard service and store=false")
        request = dict(payload, service_tier="default", store=False)
        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            try:
                return judge(**request)
            except urllib.error.HTTPError as error:
                if error.code not in TRANSIENT_HTTP or attempt == JUDGE_ATTEMPTS:
                    raise
                SLEEP(_retry_delay(attempt, error))
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                if attempt == JUDGE_ATTEMPTS:
                    raise
                SLEEP(_retry_delay(attempt))

    return wrapped


def frozen_data(data_path):
    lock = json.loads(LOCK.read_text())
    path = Path(data_path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != lock["sha256"] or path.stat().st_size != lock["bytes"]:
        raise VerificationFailure("Frozen evaluation payload digest mismatch")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != lock["samples"] or len({r["id"] for r in rows}) != len(rows):
        raise VerificationFailure("Frozen evaluation row count/ID mismatch")
    return len(rows)


def calibration_binding(data_path):
    seed = Path(__file__).resolve().parent
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {"protocol": public.PROTOCOL, "judge_model": public.JUDGE_MODEL,
            "rubric_sha256": public.RUBRIC_SHA256, "scorer_sha256": digest(seed / "validate.py"),
            "verifier_sha256": digest(__file__), "judge_client_sha256": digest(judge_runtime.__file__),
            "renderer_code_sha256": digest(seed / "svg_render.py"),
            "renderer_version": OFFICIAL_RENDERER_VERSION, "evaluation_sha256": digest(data_path)}


def check_report(report):
    """The preregistered graded-judge admission checks, recomputed from the stored calibration report."""
    groups = report["groups"]
    for kind in ("reference_self", "constant_circle", "text_injection", "reference_repeat"):
        group = groups[kind]
        if group["samples"] != ADMISSION["samples_per_group"] or len(group["scores"]) != group["samples"]:
            raise ValueError(f"{kind} needs {ADMISSION['samples_per_group']} judged samples")
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in group["scores"].values()):
            raise ValueError(f"{kind} scores must be finite and within [0, 1]")
    mean = lambda kind: sum(groups[kind]["scores"].values()) / len(groups[kind]["scores"])
    if min(mean("reference_self"), mean("reference_repeat")) < ADMISSION["reference_minimum"]:
        raise ValueError("reference self-score is below the minimum")
    if mean("constant_circle") > ADMISSION["constant_maximum"] or mean("text_injection") > ADMISSION["injection_maximum"]:
        raise ValueError("constant or injection controls failed")
    if abs(mean("reference_self") - mean("reference_repeat")) > ADMISSION["repeat_macro_difference_maximum"]:
        raise ValueError("reference repeat differs too much")
    first, second = groups["reference_self"]["scores"], groups["reference_repeat"]["scores"]
    if set(first) != set(second):
        raise ValueError("reference repeat covers different samples")
    xs, ys = [first[k] for k in sorted(first)], [second[k] for k in sorted(first)]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx, sy = sum((x - mx) ** 2 for x in xs) ** 0.5, sum((y - my) ** 2 for y in ys) ** 0.5
    correlation = 1.0 if sx == 0 or sy == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
    if correlation < ADMISSION["repeat_item_correlation_minimum"]:
        raise ValueError("reference repeat item scores disagree")


def require_calibration(data_path):
    """Task readiness, never a candidate-dependent constraint or override."""
    try:
        readiness = json.loads(READINESS.read_text())
        if readiness.get("binding") != calibration_binding(data_path):
            raise ValueError("calibration binding differs from judge, rubric, renderer or data")
        if readiness.get("status") != "passed" or readiness.get("admission") != ADMISSION:
            raise ValueError("judge calibration is " + str(readiness.get("status", "pending")))
        raw = REPORT.read_bytes()
        if hashlib.sha256(raw).hexdigest() != readiness["report_sha256"]:
            raise ValueError("calibration evidence digest differs")
        report = json.loads(raw)
        if report["model"] != public.JUDGE_MODEL or report["binding"] != readiness["binding"]:
            raise ValueError("calibration evidence belongs to another judge or scorer")
        check_report(report)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise VerificationFailure("SVG official scoring is unavailable: " + str(error)) from error
    return readiness


def prescreen(config, data_path):
    """Free static checks before the paid architecture review: the training evidence and the weight audit."""
    scan = inspect_submission(config["submission_path"])
    if not scan["passed"]:
        raise VerificationFailure("Training evidence scan failed: " + "; ".join(scan["problems"]))
    weights = inspect_weights(config["submission_path"])
    if not weights["passed"]:
        raise VerificationFailure("Weight audit failed: " + "; ".join(weights["problems"]))
    config["weight_audit"] = weights


def preflight(config, data_path):
    require_calibration(data_path)
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("A trusted architecture review receipt is required before provisioning")
    if config.get("concurrency", CONCURRENCY) != CONCURRENCY or config.get("judge_model", public.JUDGE_MODEL) != public.JUDGE_MODEL:
        raise VerificationFailure("SVG protocol constants cannot be overridden")
    if config.get("judge_endpoint", "https://api.openai.com/v1/chat/completions") != "https://api.openai.com/v1/chat/completions":
        raise VerificationFailure("SVG judging requires the calibrated native endpoint")
    renderer_binary(config.get("rsvg_binary"), official=True)
    count = frozen_data(data_path)
    if config.get("expected_samples", count) != count:
        raise VerificationFailure("Evaluation sample count differs from frozen payload")
    if not config.get("submission_path"):
        raise VerificationFailure("Frozen submission_path is required")
    if not isinstance(config.get("weight_audit"), dict) or config["weight_audit"].get("passed") is not True:
        prescreen(config, data_path)
    config["judge"] = calibrated_judge(judge_from_config(config))
    config["judge_model"] = public.JUDGE_MODEL


def launch(snapshot, config, data_path):
    """The trusted gpt-oss-20b server (FP8, SGLang) on the submitted weights, in place of a submission serve.sh."""
    return serve_submission(snapshot, dict(config, trusted_serve_dir=str(TRUSTED_SERVER)))


def verify(server, data_path, config):
    require_calibration(data_path)
    count = frozen_data(data_path)
    renderer_binary(config.get("rsvg_binary"), official=True)
    review = verify_review(server["snapshot"], config["architecture_review_path"], "svg-generation", REVIEW_CHECKS)
    scan = inspect_submission(server["snapshot"])
    if not scan["passed"]:
        raise VerificationFailure("Training evidence scan failed")
    weights = config.get("weight_audit")
    if not isinstance(weights, dict) or weights.get("passed") is not True:
        raise VerificationFailure("The weight audit must pass before evaluation")
    trusted = dict(config, validation_mode=False, predictions_only=False, concurrency=CONCURRENCY, expected_samples=count)
    if trusted.get("judge_model", public.JUDGE_MODEL) != public.JUDGE_MODEL:
        raise VerificationFailure("Incorrect SVG judge deployment")
    trusted["judge"] = calibrated_judge(trusted.get("judge"))
    result = public.run(server["transport"], data_path, trusted)
    result["details"]["gates"] = {"svg_byte_budget": 32768, "training_evidence": scan, "weight_audit": weights,
                                   "architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")}}
    return result
