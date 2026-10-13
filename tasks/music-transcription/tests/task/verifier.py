"""Trusted verifier: a static 1 GB submission-size gate
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import load_samples
from amt_notes import parse_notes

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
SELFCHECK_PUBLIC = ("validation.jsonl", "audio/*.flac")  # the 10 example clips, about 14 MB
# The self-check's timed pass may run to this multiple of the grading deadline, so a server somewhat too slow still
# gets a measured factor; 2 x 162.6 s plus one request fits the default self-check budget (selfcheck.WORK_SECONDS).
SELFCHECK_PATIENCE = 2
SELFCHECK_REQUEST_SECONDS = 60  # the single clip after the pass, retries included; 0.6 s at 50x real time


def default_config():
    # None forces an explicit operator value in task.yaml before any trial can grade.
    return {"concurrency": public.CONCURRENCY, "evaluation_sha256": None, "max_submission_bytes": None}


def submission_bytes(root):
    """Bytes of every regular file in the frozen snapshot; anything else is refused."""
    total, files = 0, 0
    root = Path(root)
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        for name in subdirectories:
            if (Path(directory) / name).is_symlink():
                raise VerificationFailure(f"Submission contains a non-regular file: {(Path(directory) / name).relative_to(root)}")
        for name in names:
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise VerificationFailure(f"Submission contains a non-regular file: {path.relative_to(root)}")
            total += path.stat().st_size
            files += 1
    return {"bytes": total, "files": files}


def size_gate(root, config):
    limit = config.get("max_submission_bytes")
    if type(limit) is not int or limit <= 0:
        raise VerificationFailure("Set a positive integer max_submission_bytes before grading")
    measured = {**submission_bytes(root), "limit_bytes": limit}
    measured["passed"] = measured["bytes"] <= limit
    if not measured["passed"]:
        raise VerificationFailure(f"Submission is {measured['bytes']:,} bytes, over the {limit:,}-byte limit; "
                                  "score is zero")
    return measured


def frozen_samples(data_path, config):
    expected = config.get("evaluation_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise VerificationFailure("Set evaluation_sha256 (the frozen held-out file digest) before grading")
    with open(data_path, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise VerificationFailure("Evaluation data differs from the frozen held-out file")
    samples = load_samples(data_path)
    if config.get("expected_samples") != len(samples):
        raise VerificationFailure("expected_samples must match the frozen evaluation data")
    root = Path(data_path).parent
    for sample in samples:
        try:
            parse_notes({"notes": sample["notes"]})
        except (KeyError, ValueError) as error:
            raise VerificationFailure(f"{sample.get('id')}: invalid reference notes: {error}") from error
        if not (root / sample["audio"]).is_file() or float(sample["duration_s"]) <= 0:
            raise VerificationFailure(f"{sample['id']}: missing audio or duration")
    return samples


def prescreen(config, data_path):
    """Free static checks before the paid review: an oversize submission scores zero without one."""
    size_gate(config["submission_path"], config)


def preflight(config, data_path):
    """Everything that can fail before the H100 is provisioned."""
    if config.get("concurrency") != public.CONCURRENCY:
        raise VerificationFailure(f"The task fixes request concurrency at {public.CONCURRENCY}")
    frozen_samples(data_path, config)
    limit = config.get("max_submission_bytes")
    if type(limit) is not int or limit <= 0:
        raise VerificationFailure("Set a positive integer max_submission_bytes before grading")
    if config.get("submission_path"):  # prescreen already checked it; verify() rechecks the frozen snapshot
        config["submission_size"] = size_gate(config["submission_path"], config)
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")


def verify(server, data_path, config):
    size = size_gate(server["snapshot"], config)
    preflight({k: v for k, v in config.items() if k != "submission_path"}, data_path)
    review = verify_review(server["snapshot"], config["architecture_review_path"], "music-transcription", REVIEW_CHECKS)
    trusted = dict(config, validation_mode=False, predictions_only=False, concurrency=public.CONCURRENCY)
    result = public.run(server["transport"], data_path, trusted)
    timing = result["details"]["timing"][0]
    result["details"]["gates"] = {
        "submission_size": size,
        "architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")},
        "real_time_factor": {"measured": timing.get("real_time_factor"),
                             "required": public.MINIMUM_REAL_TIME_FACTOR,
                             "deadline_seconds": timing.get("deadline_seconds")}}
    return result


def selfcheck_batch(public_dir, count, directory):
    """The graded pass's shape on public clips: `count` rows (the held-out split's size) cycling through the examples
    under unique IDs, in a folder of their own with copies of the audio (run() keeps audio inside its data folder).
    Returns the batch file, its rows and the examples. Unusable public data is never the submission's fault."""
    public_dir, directory = Path(public_dir), Path(directory)
    try:
        examples = load_samples(public_dir / "validation.jsonl")
        for example in examples:
            public.build_request(example, public_dir)  # the grader's own path checks, before any GPU time
            if not float(example["duration_s"]) > 0:
                raise ValueError(f"{example['id']}: duration_s must be positive")
            target = directory / example["audio"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(public_dir / example["audio"], target)
    except (OSError, KeyError, TypeError, ValueError, VerificationFailure) as error:
        raise EvaluationError(f"The public examples are unusable: {type(error).__name__}: {error}") from error
    rows = []
    for index in range(count):
        example = examples[index % len(examples)]
        rows.append({"id": f"{example['id']}#{index}", "audio": example["audio"], "duration_s": example["duration_s"]})
    path = directory / "selfcheck.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, rows, examples


def selfcheck(check, public_dir, config):
    """The size gate, then grading's own timed pass on public clips (as many as the held-out split, 16 at a time),
    then one clip through the task's parser. The pass goes first because in grading it is the first traffic after
    /health: a warm-up request the grader never sends would flatter a server that compiles on its first request."""
    check.static("submission size", size_gate, check.snapshot, config)
    count = config.get("expected_samples")
    if type(count) is not int or count < 1:
        raise EvaluationError("The self-check needs the held-out split's size (expected_samples) in its config")
    with tempfile.TemporaryDirectory(prefix="music-transcription-selfcheck-") as directory:
        batch, rows, examples = selfcheck_batch(public_dir, count, directory)
        sample = examples[0]
        path, payload = public.build_request(sample, public_dir)
        audio = sum(float(row["duration_s"]) for row in rows)
        deadline = len(rows) / public.minimum_clips_per_second(rows)  # grading's formula: audio seconds / 50
        cap = SELFCHECK_PATIENCE * deadline
        # Diagnostic mode measures the pass past grading's deadline (up to `cap`) and never scores it; requests,
        # concurrency, retries and the failed-request limit are grading's own.
        trusted = dict(config, validation_mode=True, predictions_only=True, concurrency=public.CONCURRENCY,
                       inference_timeout_seconds=cap)
        with check.launch() as server:
            check.stage("gates")
            try:
                result = public.run(server["transport"], str(batch), trusted)
            except EvaluationError:
                raise
            except VerificationFailure as error:
                # The grader's own deadline message; the failed-request message can quote the server's error bodies.
                if str(error).startswith("Batch exceeded"):
                    check.fail("real-time factor", f"{len(rows)} clips ({audio:g} s of audio) did not finish in "
                               f"{cap:.1f} s, {SELFCHECK_PATIENCE:g} x grading's {deadline:.1f} s deadline: below "
                               f"{audio / cap:.0f}x real time", required=public.MINIMUM_REAL_TIME_FACTOR,
                               unit="x real time")
                check.fail("timed pass", f"grading's pass would fail the same way: {error}", kind="smoke")
            except (OverflowError, RecursionError) as error:
                # What a response can raise past the per-request handling (the parser on a 400-digit integer, the
                # batch transport's JSON decoder on deep nesting); grading's run fails too. Anything else is the
                # infrastructure's or the hook's, an error that costs the agent no check.
                check.fail("timed pass", "a response broke the grader's client or parser, which fails grading too: "
                           f"{type(error).__name__}: {error}", kind="smoke")
            timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
            measured = timing.get("real_time_factor") or audio / max(timing["elapsed_seconds"], 1e-12)
            detail = (f"{len(rows)} public clips ({audio:g} s of audio, the held-out pass's size) sent "
                      f"{public.CONCURRENCY} at a time in {timing['elapsed_seconds']:.1f} s; grading allows "
                      f"{deadline:.1f} s.")
            if len(rows) > len(examples):
                check.note(f"The timed pass repeated the {len(examples)} public examples under new IDs to make up "
                           f"the held-out split's {len(rows)} clips, so a server that caches by input measured "
                           "faster here than it will on held-out clips.")
            if failed["total"]:
                detail += (f" {failed['total']} of {failed['requests']} requests failed (grading tolerates up to "
                           f"{failed['limit_percent']}%, each scoring as a clip with no notes): {failed['errors'][0]}.")
            # The batch transport counts retries in its timing; the per-request path in the failure summary.
            retries = timing.get("retried_requests", failed["retried"])
            if retries:
                detail += (f" {retries} retried request(s) and their backoff "
                           f"({' or '.join(map(str, failed['backoff_seconds']))} s) count in that time.")
                if timing["elapsed_seconds"] > deadline:
                    # Past grading's deadline the 2x cap still leaves room for retries grading would not send.
                    detail += (f" Grading skips a retry whose backoff no longer fits in its {deadline:.1f} s and "
                               "scores that clip as empty instead, so its pass can finish in time where this one "
                               "did not. Fix the requests that needed a retry.")
            check.gate("real-time factor", measured, public.MINIMUM_REAL_TIME_FACTOR, higher_is_better=True,
                       unit="x real time", detail=detail)
            check.stage("smoke")
            check.request(server, path, payload, parse=public.parse_response, sample=sample,
                          timeout=SELFCHECK_REQUEST_SECONDS)
