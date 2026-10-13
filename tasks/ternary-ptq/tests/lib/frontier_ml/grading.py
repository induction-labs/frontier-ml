"""Harbor tests/grade.py calls this library; review and grading form one verifier phase.

Optional hooks a task may define (all trusted, all in the task seed):

- ``review.py``: ``POLICY`` (required for ``review: true``), plus optional
  ``weight_inspector(path)``, ``binary_inspector(path)`` and
  ``submission_inspector(snapshot)`` (trusted audit evidence supplied before review). Without a task binary
  inspector, vendored wheels are audited by ``review_binaries.default_binary_inspector``.
- ``verifier.py``:
  - ``prescreen(config, data)``: free static checks of ``config["submission_path"]``
    that run before the paid architecture review.
  - ``preflight(config, data)``: static checks of ``config["submission_path"]`` that
    run before any GPU is provisioned.
  - ``launch(snapshot, config, data)``: a context manager yielding a server, replacing
    the default ``serve_submission`` (e.g. a training replay that then serves).
  - ``verify(server, data, config)``: the default single-instance evaluation, or
  - ``verify_isolated(launch, data, config)``: multi-phase evaluation; each
    ``with launch() as server`` provisions a fresh, independent instance.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import tempfile

from .review import ANTHROPIC_REVIEW_MODEL, AnthropicReviewer, OpenAIReviewer, create_review
from .review_agent import SDKReviewer, review_backend
from . import review_trajectory
from .review_binaries import default_binary_inspector
from .runtime.inference import serve_submission
from .runtime.server import VerificationFailure, snapshot_submission
from .runtime import submission_volume
from .scoring.normalization import normalize
from .taskspec import load_module, load_task, review_policy, served_by_verifier
from .validation.errors import EvaluationError
from .validation.progress import progress

INSPECTORS = ("weight_inspector", "binary_inspector", "submission_inspector")


def review_inspectors(task, config) -> dict:
    """Task-defined trusted inspectors from review.py, with the shared wheel audit as default."""
    policy_module = load_module(task.slug + ".review", task.path / "review.py")
    inspectors = {name: getattr(policy_module, name) for name in INSPECTORS
                  if callable(getattr(policy_module, name, None))}
    inspectors.setdefault("binary_inspector", default_binary_inspector(config.get("architecture_review", {})))
    return inspectors


def review_judge(options):
    """(model, judge) for the trusted reviewer: an OpenAI-compatible provider named by FML_REVIEW_BASE_URL/
    FML_REVIEW_MODEL/FML_REVIEW_API_KEY, else Claude through the Anthropic SDK with ANTHROPIC_API_KEY."""
    model = options.get("model") or os.environ.get("FML_REVIEW_MODEL")
    base_url = options.get("base_url") or os.environ.get("FML_REVIEW_BASE_URL")
    timeout = options.get("timeout_seconds", 300)
    if base_url:
        if not model:
            raise VerificationFailure("FML_REVIEW_BASE_URL needs FML_REVIEW_MODEL")
        return model, OpenAIReviewer(base_url, os.environ.get("FML_REVIEW_API_KEY", ""), timeout=timeout)
    if os.environ.get("ANTHROPIC_API_KEY"):
        return model or ANTHROPIC_REVIEW_MODEL, AnthropicReviewer(os.environ["ANTHROPIC_API_KEY"], timeout=timeout)
    raise VerificationFailure("Architecture review needs ANTHROPIC_API_KEY, or FML_REVIEW_BASE_URL and FML_REVIEW_MODEL")


def review_skip_enabled() -> bool:
    return os.environ.get("FML_REVIEW_SKIP", "").strip().lower() in ("1", "true", "yes")


def write_skipped_review(task, digest, receipt_path, config):
    """FML_REVIEW_SKIP=1: a receipt that says the review was not run, bound to this task and submission digest.

    Task verifiers accept it through scoring.architecture.verify_review only while FML_REVIEW_SKIP is set, so the
    score it enables is provisional until a real review of the same digest passes."""
    import hashlib
    policy = review_policy(task.slug, task.path)
    receipt = {"schema_version": 1, "task": task.slug, "submission_digest": digest, "decision": "skipped",
               "reviewer": "skipped (FML_REVIEW_SKIP)", "reviewer_model": None, "policy": policy,
               "policy_digest": hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
               "checks": {key: None for key in policy["checks"]}, "evidence": [],
               "limitations": "The architecture review was skipped by the operator; this receipt proves nothing."}
    Path(receipt_path).write_text(json.dumps(receipt, indent=2) + "\n")
    config["architecture_review_path"] = str(receipt_path)


class ReviewIncomplete(VerificationFailure):
    """The evaluator could not finish review; no model violation was established."""


def review_submission(task, snapshot, config, receipt_path):
    options = config.get("architecture_review", {})
    inspectors = review_inspectors(task, config)
    backend = review_backend(options)
    judge = None
    if backend == "api":
        model, judge = review_judge(options)
        reviewer = {"judge": judge}
    else:
        model = options.get("model") or os.environ.get("FML_REVIEW_MODEL")
        if not model and backend == "codex":
            raise ReviewIncomplete("Codex review requires an explicit FML_REVIEW_MODEL or architecture_review.model")
        model = model or ANTHROPIC_REVIEW_MODEL
        reviewer = {"agent": SDKReviewer(backend, options)}
    try:
        receipt = create_review(snapshot, receipt_path, task=task.slug, task_dir=task.path,
                                model=model, trajectory=config.get("review_trajectory"), **reviewer, **inspectors)
    finally:
        if judge is not None:
            judge.client.close()
    if receipt["decision"] == "incomplete":
        raise ReviewIncomplete("Architecture review could not complete: " + receipt.get("error", "see architecture-review.json"))
    if receipt["decision"] != "pass":
        raise VerificationFailure("Architecture review rejected the submission; see architecture-review.json")
    config["architecture_review_path"] = str(receipt_path)


def checked_score(result):
    score = result.get("score") if isinstance(result, dict) else None
    if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
        raise VerificationFailure("Scorer returned an invalid score")
    return result


def evaluate(module, snapshot, data, config):
    """Run the task's verifier against one or more fresh offline instances of the snapshot."""
    custom = getattr(module, "launch", None)
    instances = []

    @contextmanager
    def launch():
        progress(config, "provisioning")
        session = custom(snapshot, config, data) if callable(custom) else serve_submission(snapshot, config)
        with session as server:
            progress(config, "evaluation")
            instances.append({"startup_seconds": server.get("startup_elapsed"), "backend": server.get("backend"),
                              "sandbox_id": server.get("sandbox_id"), "gpu": server.get("gpu")})
            yield server

    isolated = getattr(module, "verify_isolated", None)
    if callable(isolated):
        result = checked_score(isolated(launch, data, config))
        if not instances:
            raise VerificationFailure("Isolated verifier never launched the submission")
    else:
        with launch() as server:
            result = checked_score(module.verify(server, data, config))
    startups = [i["startup_seconds"] for i in instances if isinstance(i["startup_seconds"], (int, float))]
    result.update(startup_seconds=max(startups) if startups else None, backend=instances[0]["backend"],
                  instances=instances)
    return result


def grade(task_slug, *, tests_dir=Path("/tests"), submission=Path("/workspace/output"),
          results_dir=Path("/logs/verifier"), trajectory=review_trajectory.VERIFIER_PATH):
    """Write Harbor's reward and detailed result, including zero on verifier failure.

    All inputs except the submission belong to the separate verifier image. The
    same immutable snapshot is reviewed, inspected, and sent to the offline GPU.
    """
    tests_dir, results_dir = Path(tests_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_dir.chmod(0o700)
    (results_dir / "reward.txt").write_text("0\n")
    config = {"progress_path": str(results_dir / "progress.json")}
    progress(config, "loading")
    try:
        task = load_task(task_slug, tests_dir / "task")
        module = task.module
        config = json.loads((tests_dir / "config.json").read_text())
        # Diagnostic switches and supplied receipts cannot bypass final checks.
        config.update(validation_mode=False, predictions_only=False, task_slug=task_slug,
                      results_dir=str(results_dir), progress_path=str(results_dir / "progress.json"))
        for key in ("architecture_review_path", "submission_volume", "submission_digest", "review_trajectory"):
            config.pop(key, None)
        # The participant's session, as staged by the plugin, is review evidence; absence never blocks grading.
        config["review_trajectory"] = review_trajectory.verifier_record(
            trajectory, os.environ.get(review_trajectory.ENV))
        data = Path(task.spec["data"]["evaluation_file"])
        if data.is_relative_to("/tests"):
            data = tests_dir / data.relative_to("/tests")
        if not data.is_file():
            raise VerificationFailure("Evaluation data file is unavailable")
        with tempfile.TemporaryDirectory(prefix="frontier-grade-") as directory:
            snapshot = Path(directory) / "submission"
            try:
                print("Preparing immutable submission snapshot", flush=True)
                with submission_volume.staged(submission, config, Path(directory) / "volume") as source:
                    digest = snapshot_submission(source, snapshot, require_serve_sh=not served_by_verifier(task))
                print(f"Submission snapshot ready: {digest}", flush=True)
                # Review, static preflight gates and the verifier all see the same frozen snapshot.
                config["submission_path"] = str(snapshot)
                config["submission_digest"] = digest
                prescreen = getattr(module, "prescreen", None)
                progress(config, "prescreen")
                if callable(prescreen):
                    prescreen(config, str(data))
                review_skipped = bool(task.spec.get("review")) and review_skip_enabled()
                if task.spec.get("review") and not review_skipped:
                    progress(config, "architecture_review")
                    review_submission(task, snapshot, config, results_dir / "architecture-review.json")
                elif review_skipped:
                    write_skipped_review(task, digest, results_dir / "architecture-review.json", config)
                preflight = getattr(module, "preflight", None)
                progress(config, "preflight")
                if callable(preflight):
                    preflight(config, str(data))
                print("Starting held-out evaluation", flush=True)
                # The verifier's raw metric becomes the reward between task.yaml's anchors (score_normalization).
                result = normalize(evaluate(module, snapshot, str(data), config), task.spec.get("score_normalization"))
                result.update(task=task_slug, status="completed", submission_digest=digest)
                if review_skipped:
                    # Operator switch (FML_REVIEW_SKIP=1): the architecture review was not run, so this score is
                    # provisional until a review of the same submission digest passes.
                    result["review"] = {"status": "skipped", "reason": "FML_REVIEW_SKIP", "submission_digest": digest}
            finally:
                # Read-only snapshots must be removable by non-root test runners too.
                if snapshot.exists():
                    for path, _, _ in os.walk(snapshot):
                        os.chmod(path, 0o700)
        code = 0
    except Exception as exc:
        result = {"task": task_slug, "score": 0.0,
                  "status": "evaluation_error" if isinstance(exc, (ReviewIncomplete, EvaluationError)) else "failed",
                  "error": f"{type(exc).__name__}: {exc}"}
        code = 1
    (results_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (results_dir / "reward.txt").write_text(str(result["score"]) + "\n")
    progress(config, result["status"], message=result.get("error"))
    print(json.dumps(result, allow_nan=False))
    return code
