"""Trusted provenance review plus continuous-batch loopback throughput and exact scoring.
selfcheck: the training-evidence audit, one public document and the same runner's throughput gate on resampled
public documents, without review, the held-out split or a score."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import time
import uuid

from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import load_samples
from zsc_metric import failed_prediction, validate_samples
from zsc_policy import inspect_training
from zsc_protocol import (CONCURRENCY, FAILED_REQUEST_PERCENT, MINIMUM_DOCS_PER_SECOND, WARMUP_REQUESTS,
                          build_request, failure_summary, latency_details, over_limit_message, parse_response,
                          prepare_samples, throughput_details)

public = sibling(__file__, "validate.py", require=("score",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
SELFCHECK_PUBLIC = ("validation.jsonl",)


def frozen_samples(data_path, config):
    lock = json.loads((Path(__file__).parent / "pipeline/prepared_lock.json").read_text())
    expected = lock["files"]["private/test.jsonl"]
    path = Path(data_path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
        raise VerificationFailure("Evaluation data differs from the trusted prepared-data lock")
    samples = load_samples(path)
    validate_samples(samples)
    if config.get("expected_samples") != len(samples) or len(samples) != lock["splits"]["heldout"]:
        raise VerificationFailure("Expected sample count must match the frozen evaluation data")
    if Counter(row["dataset"] for row in samples) != lock["datasets"]:
        raise VerificationFailure("Evaluation dataset coverage differs from the trusted lock")
    return samples


def prescreen(config, data_path):
    frozen_samples(data_path, config)
    config["training_audit"] = inspect_training(config["submission_path"])


def preflight(config, data_path):
    if config.get("concurrency", CONCURRENCY) != CONCURRENCY:
        raise VerificationFailure("Throughput evaluation requires concurrency 32")
    receipt = config.get("architecture_review_path")
    if not receipt or not Path(receipt).is_file():
        raise VerificationFailure("Trusted architecture_review_path required before provisioning")
    frozen_samples(data_path, config)


def checked_records(samples, report):
    """The trusted runner's report checked as grading checks it, before any score: per-document predictions (a
    failed request as the confidently wrong answer), durations, failed documents, the runner's batch timing and the
    throughput gate's numbers. The self-check reads the same checks without scoring."""
    if not isinstance(report, dict) or report.get("status") != "ok":
        error = report.get("error", "missing status") if isinstance(report, dict) else "invalid report"
        raise VerificationFailure("Trusted throughput client failed: " + str(error))
    records = report.get("records")
    if not isinstance(records, list) or len(records) != len(samples):
        raise VerificationFailure("Trusted client returned an incomplete evaluation")
    predictions, durations, failed = [], [], []
    for index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        if not isinstance(record, dict) or type(record.get("index")) is not int or record["index"] != index:
            raise VerificationFailure("Trusted timing record IDs/order differ")
        failure = record.get("request_failed")
        if failure is not None:
            # The runner has no labels; the zero for a failed document is decided here (zsc_metric.failed_prediction).
            if not isinstance(failure, dict) or not isinstance(failure.get("reason"), str) or record.get("response") is not None:
                raise VerificationFailure("Trusted failure record is inconsistent")
            predictions.append(failed_prediction(sample, failure))
            failed.append((sample["id"], failure))
        else:
            try:
                predictions.append(parse_response(sample, record.get("response")))
            except ValueError as error:
                raise VerificationFailure(f"Trusted client record failed validation: {error}") from error
        durations.append(record.get("elapsed_seconds"))
    if len(failed) * 100 > FAILED_REQUEST_PERCENT * len(samples):
        raise VerificationFailure(over_limit_message(failed, len(samples)))
    batch = report.get("timing")
    if not isinstance(batch, dict) or type(batch.get("concurrency")) is not int or batch["concurrency"] != CONCURRENCY:
        raise VerificationFailure("Missing trusted batch timing at concurrency 32")
    return predictions, durations, failed, batch, throughput_details(len(samples), batch.get("elapsed_seconds"))


def score_records(samples, report):
    predictions, durations, failed, batch, timing = checked_records(samples, report)
    if not timing["passed"]:
        raise VerificationFailure("Scored batch must sustain at least 50 documents/second")
    result = public.score(samples, predictions)
    result["details"]["gates"] = {"throughput": dict(timing, location="trusted H100 loopback client")}
    result["details"]["request_latency"] = latency_details(durations)
    reported = batch.get("failed_requests") if isinstance(batch.get("failed_requests"), dict) else {}
    result["details"]["failed_requests"] = failure_summary(failed, len(samples), retried=reported.get("retried", 0))
    result["details"]["warmup_failures"] = report.get("warmup_failures", 0)
    return result


def trusted_client_report(resource, samples, config):
    """The trusted loopback runner's report for `samples` (warm-up, then one timed batch) from inside the inference
    sandbox. Only input text/labels cross into the candidate sandbox, never targets or sources."""
    remote = "/opt/frontier-grading/zsc-" + uuid.uuid4().hex
    quoted = shlex.quote(remote)
    resource.exec("mkdir -m 700 " + quoted)
    try:
        with tempfile.TemporaryDirectory(prefix="zsc-trusted-") as directory:
            local = Path(directory)
            requests = local / "requests.jsonl"
            requests.write_text("".join(json.dumps(build_request(row)[1], ensure_ascii=True) + "\n" for row in samples))
            resource.upload(requests, remote + "/requests.jsonl")
            for name in ("zsc_runner.py", "zsc_protocol.py", "zsc_metric.py"):
                resource.upload(Path(__file__).with_name(name), remote + "/" + name)
            budget = config.get("inference_timeout_seconds", 600)
            if type(budget) not in (int, float) or not 1 <= budget <= 600:
                raise VerificationFailure("Invalid trusted inference budget")
            command = ("python3 -I " + quoted + "/zsc_runner.py " + quoted + "/requests.jsonl "
                       + quoted + "/result.json --total-seconds " + str(budget))
            # A failing runner still writes a structured error. Do not retry.
            execution = resource.exec(command, timeout=budget + 90, check=False)
            try:
                resource.download(remote + "/result.json", local / "result.json")
                report = json.loads((local / "result.json").read_text())
            except EvaluationError:
                raise
            except Exception as error:
                # A runner that exited without its report (killed for the memory the server took, a full disk) fails
                # the run under its own name; a negative code is the provider losing the exec, not the runner.
                if type(execution.return_code) is not int or execution.return_code <= 0:
                    raise
                stderr = (getattr(execution, "stderr", "") or "").strip()[-500:]
                raise VerificationFailure(f"Trusted throughput client exited with code {execution.return_code} without "
                                          f"its report ({type(error).__name__}: {error}); stderr {stderr!r}") from error
            if execution.return_code and report.get("status") == "ok":
                raise VerificationFailure("Trusted timing client exited unsuccessfully")
            return report
    finally:
        resource.exec("rm -rf -- " + quoted, timeout=30, check=False)


def evaluate_on_resource(resource, samples, config):
    return score_records(samples, trusted_client_report(resource, samples, config))


def verify(server, data_path, config):
    review = verify_review(server["snapshot"], config["architecture_review_path"],
                           "zero-shot-classification", REVIEW_CHECKS)
    audit = inspect_training(server["snapshot"])
    samples = prepare_samples(frozen_samples(data_path, config))
    if config.get("concurrency", CONCURRENCY) != CONCURRENCY or config.get("expected_samples") != len(samples):
        raise VerificationFailure("Frozen sample count and concurrency 32 are required")
    resource = getattr(server["transport"], "resource", None)
    if resource is None:
        raise VerificationFailure("Official timing requires the trusted inference resource; no remote-timing fallback")
    result = evaluate_on_resource(resource, samples, config)
    result["details"]["gates"].update(training=audit, architecture={"checks": review["checks"]})
    return result


# Measured once: as the runner sends them, the held-out batch's requests are 6.61 MB of UTF-8 and the resample below
# of the public examples 3.78 MB, mostly from label descriptions (81 bytes each against 34) and texts (361 bytes
# against 297); held-out documents carry 2 to 72 labels, 13.1 on average against 14. Encoding each text-label pair
# separately is the exception: that work is 0.80x the public resample's.
HELDOUT_NOTE = ("The throughput gate sent resampled public documents; held-out requests are about 1.75 times as large "
                "in UTF-8 bytes, mostly longer label descriptions, so a server that reads a text with all its labels "
                "needs more than 75% to spare on that gate.")


def selfcheck_batch(public_dir, count):
    """The graded batch's shape from the public rows: `count` documents cycling over every example under unique IDs,
    then label-permuted and shuffled by the grader's own preparation. Held-out texts do not repeat, so each reuse of
    a public text is rotated by a different number of whole words: the same bytes, but a new text and opening that
    no prefix or text cache has seen. Returns the documents and the number of public examples behind them."""
    try:
        rows = load_samples(Path(public_dir) / "validation.jsonl")
        validate_samples(rows)
        if type(count) is not int or count < 1:
            raise ValueError("expected_samples must be a positive integer")
        rounds, picked = -(-count // len(rows)), []
        for index in range(count):
            row = rows[index % len(rows)]
            words = row["text"].split(" ")
            shift = index // len(rows) * len(words) // rounds  # distinct per reuse while a text has enough words
            picked.append(dict(row, id=f"{row['id']}#{index}", text=" ".join(words[shift:] + words[:shift])))
        return prepare_samples(picked), len(rows)
    except (OSError, KeyError, TypeError, ValueError, VerificationFailure) as error:  # not the submission's fault
        raise EvaluationError(f"The public examples could not be prepared: {type(error).__name__}: {error}") from error


def _selfcheck_gates(check, documents, examples, report, allowance):
    """The runner's report as the gates grading applies to it: the batch deadline, the 5% failed-request limit and
    the warm-up allowance; documents that failed within the limit are reported, not scored."""
    count, scored = len(documents), len(documents) / MINIMUM_DOCS_PER_SECOND
    if report.get("status") != "ok":
        error = str(report.get("error", "missing status"))
        if error.startswith("UnreliableServer"):
            check.fail("batch responses", error, kind="smoke")
        if "Whole-batch deadline" in error:
            check.fail("throughput", f"{count} public documents at concurrency {CONCURRENCY} did not finish within the "
                       f"{scored:g} s grading allows {count} held-out documents ({error})",
                       required=MINIMUM_DOCS_PER_SECOND, unit="documents/s")
        if "deadline" in error:  # warm-up, or the whole run before the batch could start
            check.fail("warm-up", f"the public request and the {len(WARMUP_REQUESTS)} synthetic warm-up requests did "
                       f"not finish within the {allowance - scored:g} s grading leaves warm-up before the scored "
                       f"batch ({error})", required=allowance - scored, unit="s")
        check.fail("throughput", "Trusted throughput client failed: " + error)
    try:
        _, durations, failed, timing, gate = checked_records(documents, report)
    except VerificationFailure as error:
        check.fail("batch responses", str(error), kind="smoke")
    latency = latency_details(durations)
    detail = (f"{count} documents (the {examples} public examples resampled, each reuse rotated by whole words) in "
              f"{gate['elapsed_seconds']:.1f} s of the {gate['deadline_seconds']:g} s grading allows {count} held-out "
              f"documents, at concurrency {CONCURRENCY} after {len(WARMUP_REQUESTS)} warm-up requests; request "
              f"seconds mean {latency['mean_seconds']:.2f}, p95 {latency['p95_seconds']:.2f}")
    check.gate("throughput", gate["docs_per_second"], MINIMUM_DOCS_PER_SECOND, higher_is_better=True,
               unit="documents/s", detail=detail)
    reported = timing.get("failed_requests") if isinstance(timing.get("failed_requests"), dict) else {}
    retried, warmups = reported.get("retried", 0), report.get("warmup_failures", 0)
    notes = (f"; {retried} succeeded on a retry" if retried else "") + (
        f"; {warmups} warm-up requests failed, which grading tolerates" if warmups else "")
    if failed:
        summary = failure_summary(failed, count, retried=retried)
        check.record("batch responses", False, kind="smoke",
                     detail=f"{len(failed)} of {count} requests failed, timed out or were rejected "
                            f"({summary['reasons']}); grading scores each as a confidently wrong answer and fails the "
                            f"run above {FAILED_REQUEST_PERCENT}%. First errors: {summary['errors']}{notes}")
    else:
        check.record("batch responses", True, kind="smoke", detail=f"all {count} responses parsed{notes}")


def selfcheck(check, public_dir, config):
    """prescreen's training-evidence audit, one public document through POST /classify, then the throughput gate as
    verify runs it: the trusted runner inside the inference sandbox, 16 warm-up requests and one timed batch of
    expected_samples documents at concurrency 32, here resampled from the public examples. Nothing is scored; the
    held-out lock and the review receipt, the rest of prescreen and preflight, are not available here."""
    documents, examples = selfcheck_batch(public_dir, config.get("expected_samples"))
    allowance = config.get("inference_timeout_seconds", 600)
    scored = len(documents) / MINIMUM_DOCS_PER_SECOND  # the batch deadline, reserved out of the allowance
    if type(allowance) not in (int, float) or not scored < allowance <= 600:
        raise EvaluationError("The inference budget must exceed the scored batch's deadline")
    # prescreen's audit: whatever the submission's manifest makes it raise fails grading, so check.static fails on it.
    check.static("training evidence", inspect_training, config["submission_path"])
    with check.launch() as server:
        sample = documents[0]
        path, payload = build_request(sample)
        # Grading's first request is a warm-up request with the whole warm-up allowance; this one shares that
        # allowance with the runner's warm-up, so a slow first request costs what it costs in grading. check.request
        # retries a dropped connection or a non-200 answer as the runner retries a batch document; a timeout or a
        # rejected answer is not retried, as there.
        started = time.monotonic()
        check.request(server, path, payload, parse=parse_response, sample=sample, timeout=allowance - scored)
        first = time.monotonic() - started
        check.stage("gates")
        check.note(HELDOUT_NOTE)
        resource = getattr(server["transport"], "resource", None)
        if resource is None:
            raise EvaluationError("The throughput gate needs the trusted inference resource")
        # The request and the runner together hold the GPU for at most the allowance plus the runner's 90 s exec
        # margin, well inside the self-check's work budget.
        try:
            report = trusted_client_report(resource, documents,
                                           dict(config, inference_timeout_seconds=max(1, allowance - first)))
        except EvaluationError:
            raise
        except VerificationFailure as error:  # the runner exited without its report or unsuccessfully: grading fails
            check.fail("throughput", f"{error}. Grading fails a run like this; check the server's memory and disk use.",
                       required=MINIMUM_DOCS_PER_SECOND, unit="documents/s")
        _selfcheck_gates(check, documents, examples, report, allowance)
